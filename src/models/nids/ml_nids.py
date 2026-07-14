"""TANTGen's ML-NIDS (CICFlowMeter flow classifiers) as a :class:`NIDSModel`.

The ML-NIDS is the protected NIDS for the CIC-IDS-2017 / CIC-DDoS-2019 attacks
(``ddos``, ``portscan``, ``brute_force``, ``dns_ddos``, ``ldap_ddos``, ``ntp_ddos``,
``ssdp_ddos``, ``syn_ddos``); TANTGen calibrated a per-attack supervised classifier on
CICFlowMeter flow features and evades it with the size (padding) vector. We reuse
TANTGen's own trained ``<Attack>_<MODEL>.joblib`` + ``<Attack>_scaler.joblib`` verbatim.

Scoring reproduces TANTGen's ``src/NIDS/model.py`` ``exec`` path exactly: realize the
packets to a PCAP → ``cicflowmeter`` → rename columns to the CIC-IDS feature names
(``cicflowmeter_mapping``) → ``scaler.transform`` → ``model.predict``. Per-flow score =
``P(malicious)`` (or ``decision_function`` for LinearSVC); detection = the classifier's
own 0/1 verdict. ``cicflowmeter`` is a **hard, assumed-present dependency** — a missing
binary or model raises :class:`NIDSUnavailable`.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .base import BranchScore, NIDSModel, NIDSUnavailable

log = logging.getLogger(__name__)

_FE_SUBDIR = "src/Feature_Extraction"  # cicflowmeter_mapping.py lives here


def _import_mapping(tantgen_root: Path):
    """Import TANTGen's CICFlowMeter→CIC-IDS column ``mapping`` dict (single source)."""
    fe_dir = Path(tantgen_root) / _FE_SUBDIR
    if not (fe_dir / "cicflowmeter_mapping.py").exists():
        raise NIDSUnavailable(f"cicflowmeter_mapping.py not found under {fe_dir}")
    if str(fe_dir) not in sys.path:
        sys.path.insert(0, str(fe_dir))
    import cicflowmeter_mapping as m  # noqa: WPS433
    return m.mapping


def _cicflowmeter_available() -> bool:
    import shutil
    return shutil.which("cicflowmeter") is not None


def _run_cicflowmeter(pcap_path: Path, out_csv: Path) -> None:
    """``cicflowmeter -f <pcap> -c <out_csv>`` (TANTGen's extraction command)."""
    try:
        proc = subprocess.run(
            ["cicflowmeter", "-f", str(pcap_path), "-c", str(out_csv)],
            capture_output=True, text=True, timeout=3600)
    except FileNotFoundError as e:
        raise NIDSUnavailable(f"cicflowmeter binary not found on PATH: {e}") from e
    if proc.returncode != 0:
        raise NIDSUnavailable(
            f"cicflowmeter failed (rc={proc.returncode}): {proc.stderr[:300]}")


class MLNIDS(NIDSModel):
    kind = "ml_nids"
    threshold = None  # the classifier owns its own decision boundary

    def __init__(self, base: str, tg_attack: str, model_name: str,
                 model, scaler, mapping: dict, tantgen_root: Path) -> None:
        self._base = base
        self._tg_attack = tg_attack
        self._model_name = model_name
        self._model = model
        self._scaler = scaler
        self._mapping = mapping
        self._features = list(mapping.values())
        self._tantgen_root = Path(tantgen_root)
        self.name = f"ml_nids:{tg_attack}:{model_name}"
        self.model_source = "pretrained"

    def score(self, packets: list, keys: list) -> BranchScore:
        if not packets:
            return BranchScore(0, 0, None, None, 0.0)
        from scapy.all import wrpcap

        with tempfile.TemporaryDirectory(prefix="dept_mlnids_") as td:
            pcap_path = Path(td) / "branch.pcap"
            raw_csv = Path(td) / "flows.csv"
            wrpcap(str(pcap_path), packets)
            _run_cicflowmeter(pcap_path, raw_csv)
            if not raw_csv.exists():
                return BranchScore(0, 0, None, None, 0.0)
            X = self._features_from_csv(raw_csv)
        if X is None or len(X) == 0:
            return BranchScore(0, 0, None, None, 0.0)

        Xs = self._scaler.transform(X)
        pred = self._model.predict(Xs).astype(int)
        proba = self._proba(Xs)
        # All realized packets belong to attack flows → malicious is class 1.
        flow_scores = {i: float(proba[i]) for i in range(len(proba))}
        predicted = {i: int(pred[i] == 1) for i in range(len(pred))}
        return BranchScore.from_flow_scores(flow_scores, None, predicted=predicted)

    def _features_from_csv(self, raw_csv: Path):
        import pandas as pd

        df = pd.read_csv(raw_csv)
        if df.empty:
            return None
        # Reproduce TANTGen's mapping: duplicate fwd_header_len, rename, select, clean.
        if "fwd_header_len" in df.columns:
            df["fwd_header_len1"] = df["fwd_header_len"]
        df = df.rename(columns=self._mapping)
        missing = [c for c in self._features if c not in df.columns]
        if missing:
            raise NIDSUnavailable(
                f"cicflowmeter output missing {len(missing)} expected feature "
                f"column(s), e.g. {missing[:3]} — is the cicflowmeter version compatible?")
        df = df[self._features].replace([np.inf, -np.inf], np.nan).fillna(-1)
        return df.values

    def _proba(self, Xs) -> np.ndarray:
        if hasattr(self._model, "predict_proba"):
            return self._model.predict_proba(Xs)[:, 1]
        # LinearSVC etc.: use the (unbounded) decision function as the score.
        return np.asarray(self._model.decision_function(Xs), dtype=np.float64)


def build(base: str, cfg: dict, repo_root: Path) -> MLNIDS:
    """Load TANTGen's ML-NIDS for ``base`` → :class:`MLNIDS` (hard dependency)."""
    import joblib

    tantgen_root = _abs(cfg.get("tantgen_root", ""), repo_root)
    if not tantgen_root.is_dir():
        raise NIDSUnavailable(f"TANTGen root not found: {tantgen_root}")
    if not _cicflowmeter_available():
        raise NIDSUnavailable(
            "cicflowmeter not on PATH — install it in DEPT's venv "
            "(pip install cicflowmeter) to score the ML-NIDS attacks")

    entry = dict(cfg.get("attacks", {}).get(base, {}))
    tg = entry.get("tg_attack")
    if not tg:
        raise NIDSUnavailable(f"no tg_attack mapping for ML-NIDS attack '{base}'")
    model_name = str(entry.get("model", cfg.get("default_ml_model", "RF")))

    model_dir = _abs(cfg.get("ml_nids_model_dir", "src/NIDS/models"), tantgen_root)
    model_path = model_dir / f"{tg}_{model_name}.joblib"
    scaler_path = model_dir / f"{tg}_scaler.joblib"
    for p in (model_path, scaler_path):
        if not p.exists():
            raise NIDSUnavailable(f"ML-NIDS artifact not found: {p}")

    mapping = _import_mapping(tantgen_root)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return MLNIDS(base, tg, model_name, model, scaler, mapping, tantgen_root)


def _abs(p, anchor: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else anchor / p


__all__ = ["MLNIDS", "build"]

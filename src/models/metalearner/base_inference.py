"""Run RF + TCN trained models and join their per-flow streams.

The meta-learner consumes a single per-packet token stream per flow
combining:

  * the TCN's dense per-packet logit
  * the RF's sparse per-checkpoint logit (carry-forward in the dataset)

Both base learners were trained only on the seed=42 ``train_idx``, so
their predictions on the ``val`` and ``test`` splits are out-of-fold by
construction. This module loads the trained models and produces the
joined per-flow records the dataset module turns into tokens.

Flow identity is the canonical ``(source, flow_id_str)`` tuple — the
same tuple ``predict_streams()`` returns from both base evaluators. RF
``predict_streams`` skips flows that produce zero checkpoints (e.g.
single-packet flows under default cadence); those flows still appear in
the joined output with empty RF arrays so the meta-learner can fall
back to TCN-only fusion via the learnable ``rf_unknown_logit`` token.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..rf.evaluate import predict_streams as rf_predict_streams
from ..tcn.evaluate import predict_streams as tcn_predict_streams


@dataclass
class JoinedFlow:
    """RF + TCN streams for a single flow, joined on (source, flow_id_str)."""
    source: str
    flow_id: str
    label: int                     # 0 benign / 1 adversarial
    length: int                    # number of packets in the flow (= len(tcn_logit))
    tcn_packet_idx: np.ndarray     # [length]      int64, 1-based
    tcn_logit: np.ndarray          # [length]      float32, logit_1 - logit_0
    rf_packet_idx: np.ndarray      # [n_ckpt]      int64, 1-based; can be []
    rf_logit: np.ndarray           # [n_ckpt]      float64; can be []


@dataclass
class JoinedSplit:
    """All joined flows for one split (e.g. val or test)."""
    flows: list[JoinedFlow]

    def __len__(self) -> int:
        return len(self.flows)

    @property
    def labels(self) -> np.ndarray:
        return np.asarray([f.label for f in self.flows], dtype=np.int64)

    @property
    def sources(self) -> np.ndarray:
        return np.asarray([f.source for f in self.flows], dtype=object)

    @property
    def lengths(self) -> np.ndarray:
        return np.asarray([f.length for f in self.flows], dtype=np.int64)


def _to_dict(streams: dict) -> dict[tuple[str, str], int]:
    """Build a ``flow_key -> index`` lookup from a ``predict_streams`` output."""
    return {key: i for i, key in enumerate(streams["flow_keys"])}


def join_streams(
    rf_dir: Path,
    tcn_dir: Path,
    rf_sample_indices: np.ndarray | None = None,
    tcn_sample_indices: np.ndarray | None = None,
) -> JoinedSplit:
    """Run both base learners on a chosen split and join the streams.

    ``rf_sample_indices`` indexes into ``rf_dir/dataset.npz`` rows
    (per-checkpoint). ``tcn_sample_indices`` indexes into
    ``tcn_dir/dataset_index.npz`` rows (per-flow). Both default to the
    respective ``test`` split when ``None``.

    Iteration order follows the TCN side (one record per flow). RF
    misses (flow had zero checkpoints) appear as empty RF arrays.
    """
    rf = rf_predict_streams(rf_dir, sample_indices=rf_sample_indices)
    tcn = tcn_predict_streams(tcn_dir, sample_indices=tcn_sample_indices)

    rf_lookup = _to_dict(rf)
    out: list[JoinedFlow] = []
    for i, key in enumerate(tcn["flow_keys"]):
        tcn_logit = tcn["logits"][i]
        tcn_pkt = tcn["packet_idx"][i]
        L = int(len(tcn_logit))
        if key in rf_lookup:
            j = rf_lookup[key]
            rf_logit = rf["logits"][j].astype(np.float64, copy=False)
            rf_pkt = rf["packet_idx"][j].astype(np.int64, copy=False)
            # Drop RF checkpoints past the TCN flow length (e.g. when TCN
            # was max_length-truncated but RF was not). Keeping them would
            # produce tokens past the meta-learner's sequence horizon.
            if rf_pkt.size and rf_pkt[-1] > L:
                keep = rf_pkt <= L
                rf_pkt = rf_pkt[keep]
                rf_logit = rf_logit[keep]
        else:
            rf_logit = np.zeros(0, dtype=np.float64)
            rf_pkt = np.zeros(0, dtype=np.int64)
        out.append(
            JoinedFlow(
                source=key[0],
                flow_id=key[1],
                label=int(tcn["labels"][i]),
                length=L,
                tcn_packet_idx=tcn_pkt.astype(np.int64, copy=False),
                tcn_logit=tcn_logit.astype(np.float32, copy=False),
                rf_packet_idx=rf_pkt,
                rf_logit=rf_logit,
            )
        )
    return JoinedSplit(flows=out)


__all__ = ["JoinedFlow", "JoinedSplit", "join_streams"]

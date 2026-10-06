"""The representation the diffusion purifier works in: one flat vector per flow.

Phase 8 realizes any purifier's output from one compact per-packet overlay, the IPv4
total length and the z-scored IPAT (see ``realize_triplet``). The diffusion purifier
therefore works directly in that space rather than on the 1090-dim nPrint tensor. A flow
of ``L <= T`` packets becomes

    x    = [size_1 .. size_T | ipat_1 .. ipat_T]        (the data, 2T values)
    cond = [dir_1  .. dir_T  | mask_1 .. mask_T]         (fixed by realization)

with zeros past ``L``. This is the flat-vector input Merzouk et al.'s model expects.

Encoding (fitted on the training clean flows only):
  * size → total_length / 1500 (the MTU, as DEPT's ``pkt_len_norm``);
  * IPAT → log1p(raw IPAT) / log1p(cap), cap = a high quantile of the training clean
    IPATs. IPAT is heavy-tailed (mean ≈ 26, std ≈ 208 in tensor units), so a plain
    min-max, as the original applies to its tabular features, would squash almost
    every packet to ~0.
Nothing is clipped on encode, so the purifier alone decides what changes. Decoding
clips the purified size to [0, MTU] and the IPAT to >= 0 (realization would clip the
latter anyway).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_G_MAX = 4.0   # decode guard: log-IPAT above 4x the cap would overflow expm1


@dataclass
class OverlayNorm:
    size_scale: float
    ipat_mean: float
    ipat_std: float
    log_cap: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayNorm":
        return cls(**{k: float(d[k]) for k in ("size_scale", "ipat_mean", "ipat_std",
                                               "log_cap")})

    def encode_size(self, tl) -> np.ndarray:
        return np.asarray(tl, dtype=np.float32) / np.float32(self.size_scale)

    def encode_ipat(self, ipat_z) -> np.ndarray:
        raw = np.asarray(ipat_z, dtype=np.float64) * self.ipat_std + self.ipat_mean
        return (np.log1p(np.maximum(raw, 0.0)) / self.log_cap).astype(np.float32)

    def decode(self, s, g) -> tuple[np.ndarray, np.ndarray]:
        """Encoded ``(size, ipat)`` → ``(total_len int64, ipat_z float32)``."""
        tl = np.rint(np.clip(np.asarray(s, dtype=np.float64), 0.0, 1.0)
                     * self.size_scale).astype(np.int64)
        g = np.clip(np.asarray(g, dtype=np.float64), 0.0, _G_MAX)
        raw = np.expm1(g * self.log_cap)
        return tl, ((raw - self.ipat_mean) / self.ipat_std).astype(np.float32)


def fit_norm(ipat_z: np.ndarray, length: np.ndarray, *, ipat_mean: float,
             ipat_std: float, size_scale: float = 1500.0,
             quantile: float = 0.999) -> OverlayNorm:
    """Fit the IPAT scale on the valid packets of the training clean flows."""
    T = ipat_z.shape[1]
    valid = np.arange(T)[None, :] < np.asarray(length)[:, None]
    raw = np.maximum(ipat_z[valid].astype(np.float64) * ipat_std + ipat_mean, 0.0)
    cap = float(np.quantile(raw, quantile)) if raw.size else 1.0
    log_cap = float(np.log1p(cap))
    if not np.isfinite(log_cap) or log_cap <= 0:
        log_cap = 1.0
    return OverlayNorm(size_scale=float(size_scale), ipat_mean=float(ipat_mean),
                       ipat_std=float(ipat_std), log_cap=log_cap)


def encode_dense(norm: OverlayNorm, tl: np.ndarray, ipat_z: np.ndarray,
                 direction: np.ndarray, length: np.ndarray):
    """Padded ``[N, T]`` arrays → ``(x [N, 2T] f32, dir [N, T] i8, length [N])``."""
    T = tl.shape[1]
    valid = np.arange(T)[None, :] < np.asarray(length)[:, None]
    s = np.where(valid, norm.encode_size(tl), 0.0).astype(np.float32)
    g = np.where(valid, norm.encode_ipat(ipat_z), 0.0).astype(np.float32)
    d = np.where(valid, direction, 0).astype(np.int8)
    return np.concatenate([s, g], axis=1), d, np.asarray(length, dtype=np.int64)


def encode_flows(norm: OverlayNorm, overlays: list[tuple], directions: list,
                 T: int):
    """Ragged ``(total_len, ipat_z)`` flows → ``(x [n, 2T], cond [n, 2T], valid [n, 2T])``."""
    n = len(overlays)
    x = np.zeros((n, 2 * T), dtype=np.float32)
    cond = np.zeros((n, 2 * T), dtype=np.float32)
    valid = np.zeros((n, 2 * T), dtype=bool)
    for i, ((tl, ipat), d) in enumerate(zip(overlays, directions)):
        L = min(len(tl), T)
        x[i, :L] = norm.encode_size(tl[:L])
        x[i, T:T + L] = norm.encode_ipat(ipat[:L])
        cond[i, :L] = np.asarray(d[:L], dtype=np.float32)
        cond[i, T:T + L] = 1.0
        valid[i, :L] = True
        valid[i, T:T + L] = True
    return x, cond, valid


def decode_flows(norm: OverlayNorm, x: np.ndarray, lengths) -> list[tuple]:
    """``x [n, 2T]`` → per-flow ``(total_len int64 [L], ipat_z float32 [L])``."""
    T = x.shape[1] // 2
    out = []
    for i, L in enumerate(lengths):
        L = min(int(L), T)
        out.append(norm.decode(x[i, :L], x[i, T:T + L]))
    return out


__all__ = ["OverlayNorm", "fit_norm", "encode_dense", "encode_flows", "decode_flows"]

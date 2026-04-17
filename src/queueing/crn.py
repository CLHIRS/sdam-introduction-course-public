# queueing/crn.py
from __future__ import annotations

import zlib
from typing import Any

import numpy as np


def _stable_int(x: Any) -> int:
    """
    Convert x into a stable 32-bit int (platform independent).
    Strings use crc32; ints pass through (mod 2^32).
    """
    if x is None:
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x) & 0xFFFFFFFF
    if isinstance(x, float):
        # quantize floats to avoid tiny representation differences
        return int(round(float(x) * 1e6)) & 0xFFFFFFFF
    b = str(x).encode("utf-8")
    return int(zlib.crc32(b)) & 0xFFFFFFFF


def rng_for(base_seed: int, *keys: Any) -> np.random.Generator:
    """
    Create a deterministic RNG for (base_seed, keys...).
    This is the foundation for true CRN: random variates become a pure function of keys.
    """
    ent = [int(base_seed) & 0xFFFFFFFF] + [_stable_int(k) for k in keys]
    ss = np.random.SeedSequence(entropy=ent)
    return np.random.default_rng(ss)


def exp_time(base_seed: int, rate: float, *keys: Any) -> float:
    if rate <= 0:
        return float("inf")
    r = rng_for(base_seed, *keys)
    return float(r.exponential(1.0 / float(rate)))


def bernoulli(base_seed: int, p: float, *keys: Any) -> bool:
    p = float(p)
    if p <= 0:
        return False
    if p >= 1:
        return True
    r = rng_for(base_seed, *keys)
    return bool(r.random() < p)
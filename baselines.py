from __future__ import annotations

import numpy as np

from env import BITRATES_KBPS


def rate_based(obs: np.ndarray) -> int:
    est_mbps = float(np.mean(obs[2:6])) * 6.0
    safe_kbps = est_mbps * 1000.0 * 0.85
    choices = np.where(BITRATES_KBPS <= safe_kbps)[0]
    return int(choices[-1]) if len(choices) else 0


def bola_like(obs: np.ndarray) -> int:
    buffer_s = float(obs[1]) * 30.0
    if buffer_s < 4:
        return 0
    if buffer_s < 8:
        return 1
    if buffer_s < 14:
        return 2
    if buffer_s < 20:
        return 3
    return 4


def mpc_like(obs: np.ndarray) -> int:
    est_mbps = min(float(np.mean(obs[2:6])) * 6.0, float(np.percentile(obs[2:6] * 6.0, 40)))
    buffer_s = float(obs[1]) * 30.0
    margin = 1.15 if buffer_s > 10 else 1.7
    choices = np.where(BITRATES_KBPS <= est_mbps * 1000.0 / margin)[0]
    return int(choices[-1]) if len(choices) else 0

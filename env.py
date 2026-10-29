from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # lets the smoke test run before dependencies are installed
    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class _Discrete:
        def __init__(self, n):
            self.n = n

    class _Env:
        pass

    class _Spaces:
        Box = _Box
        Discrete = _Discrete

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()


BITRATES_KBPS = np.array([300, 750, 1200, 1850, 2850, 4300], dtype=np.float32)
CHUNK_DURATION = 4.0
HISTORY = 4
# Pensieve's logarithmic QoE variant (Mao et al., SIGCOMM 2017, Eq. 6/Table 1).
REBUFFER_PENALTY = 2.66
SWITCH_PENALTY = 1.0


@dataclass
class StepInfo:
    chunk_index: int
    bitrate_kbps: float
    throughput_mbps: float
    download_time: float
    rebuffer: float
    buffer: float
    switch: float


def _load_trace(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            rows.append((float(parts[0]), float(parts[1])))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    arr = np.asarray(rows, dtype=np.float32)
    # Trace files are seconds, Mbps.
    return arr[:, 1]


def _load_chunk_sizes(video_sizes_dir: Path) -> np.ndarray:
    levels = []
    for i in range(6):
        p = video_sizes_dir / f"video_size_{i}"
        values = [int(x.strip()) for x in p.read_text().splitlines() if x.strip()]
        levels.append(values)
    min_len = min(len(v) for v in levels)
    return np.asarray([v[:min_len] for v in levels], dtype=np.float32)


class ABREnv(gym.Env):
    """adaptive bitrate streaming environment for PPO experiments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        context_dir: str | Path = "context",
        split: str = "train",
        seed: int | None = None,
        trace_selector: Callable[[list[Path]], Path] | None = None,
        trace_paths: list[Path] | None = None,
    ):
        self.context_dir = Path(context_dir)
        self.split = split
        self.rng = random.Random(seed)
        self.trace_selector = trace_selector
        self._provided_trace_paths = trace_paths

        trace_dir = self.context_dir / "traces" / split
        self.trace_paths = sorted(p for p in trace_dir.iterdir() if p.is_file())
        if self._provided_trace_paths is not None:
            self.trace_paths = list(self._provided_trace_paths)
        if not self.trace_paths:
            raise FileNotFoundError(f"no traces found in {trace_dir}")
        self.chunk_sizes = _load_chunk_sizes(self.context_dir / "video_sizes")
        self.num_chunks = self.chunk_sizes.shape[1]
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(25,), dtype=np.float32)
        self.reset(seed=seed)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        path = self.trace_selector(self.trace_paths) if self.trace_selector else self.rng.choice(self.trace_paths)
        self.trace_path = path
        self.trace = _load_trace(path)
        self.trace_idx = 0
        self.chunk_idx = 0
        self.buffer = 0.0
        self.last_action = 0
        # Do not seed the observation with future trace samples.
        self.throughput_hist = [float(self.trace[0])] * HISTORY
        self.download_hist = [CHUNK_DURATION] * HISTORY
        self.total_rebuffer = 0.0
        self.total_reward = 0.0
        return self._obs(), {}

    def step(self, action: int):
        action = int(np.clip(action, 0, 5))
        throughput = max(float(self.trace[self.trace_idx % len(self.trace)]), 0.05)
        size_bytes = float(self.chunk_sizes[action, self.chunk_idx])
        download_time = size_bytes * 8.0 / (throughput * 1_000_000.0)
        rebuffer = max(download_time - self.buffer, 0.0)
        self.buffer = max(self.buffer - download_time, 0.0) + CHUNK_DURATION
        self.buffer = min(self.buffer, 30.0)

        quality = float(np.log(BITRATES_KBPS[action] / BITRATES_KBPS[0]))
        switch = abs(float(np.log(BITRATES_KBPS[action] / BITRATES_KBPS[self.last_action])))
        reward = quality - REBUFFER_PENALTY * rebuffer - SWITCH_PENALTY * switch

        self.throughput_hist.pop(0)
        self.throughput_hist.append(throughput)
        self.download_hist.pop(0)
        self.download_hist.append(download_time)
        self.last_action = action
        self.chunk_idx += 1
        self.trace_idx += max(1, int(np.ceil(download_time / 5.0)))
        terminated = self.chunk_idx >= self.num_chunks
        self.total_rebuffer += rebuffer
        self.total_reward += reward

        info = StepInfo(
            chunk_index=self.chunk_idx - 1,
            bitrate_kbps=float(BITRATES_KBPS[action]),
            throughput_mbps=throughput,
            download_time=download_time,
            rebuffer=rebuffer,
            buffer=self.buffer,
            switch=switch,
        ).__dict__
        return self._obs(), float(reward), terminated, False, info

    def _obs(self) -> np.ndarray:
        next_sizes = self.chunk_sizes[:, min(self.chunk_idx, self.num_chunks - 1)]
        obs = np.concatenate(
            [
                [self.last_action / 5.0, self.buffer / 30.0],
                np.asarray(self.throughput_hist, dtype=np.float32) / 6.0,
                np.asarray(self.download_hist, dtype=np.float32) / 10.0,
                next_sizes / max(float(np.max(self.chunk_sizes)), 1.0),
                [1.0 - self.chunk_idx / self.num_chunks],
                np.zeros(8, dtype=np.float32),
            ]
        )
        return np.clip(obs[:25], 0.0, 1.0).astype(np.float32)


def conservative_teacher(obs: np.ndarray) -> int:
    buffer_s = float(obs[1]) * 30.0
    est_mbps = float(np.mean(obs[2:6])) * 6.0
    safe_kbps = est_mbps * 1000.0 * (1.35 if buffer_s > 12 else 0.85)
    choices = np.where(BITRATES_KBPS <= safe_kbps)[0]
    return int(choices[-1]) if len(choices) else 0

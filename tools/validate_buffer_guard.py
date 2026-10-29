from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines import rate_based
from env import ABREnv
from ppo_agent import PPOAgent


def run_episode(context: Path, trace: Path, chunk_sizes: np.ndarray,
                agent: PPOAgent, policy: str, params: tuple[int, float, float] | None):
    env = ABREnv(context, split="train", seed=3, trace_paths=[trace])
    env.chunk_sizes = chunk_sizes
    env.num_chunks = chunk_sizes.shape[1]
    env.reset(seed=3)
    steps = min(env.num_chunks, len(env.trace))
    for index in range(steps):
        # A paired comparison presents the same trace sample to each policy per chunk.
        env.trace_idx = index % len(env.trace)
        obs = env._obs()
        action = rate_based(obs) if policy == "baseline" else agent.act(obs, deterministic=True)[0]
        if policy == "shield":
            history_kbps = obs[2:6].astype(np.float64) * 6000.0
            window, margin, buffer_slack = params
            predicted_kbps = max(50.0, float(np.min(history_kbps[-window:])) * margin)
            deadline_s = env.buffer + buffer_slack
            feasible = [level for level in range(action + 1)
                        if float(env.chunk_sizes[level, index]) * 8.0 /
                        (predicted_kbps * 1000.0) <= deadline_s]
            action = max(feasible) if feasible else 0
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    return {"qoe": float(env.total_reward), "rebuffer_s": float(env.total_rebuffer)}


def main():
    parser = argparse.ArgumentParser(description="Select and evaluate the PPO buffer guard on held-out FCC traces")
    parser.add_argument("--media-dir", type=Path, required=True,
                        help="encoded video directory containing q0..q5 chunk MP4 files")
    parser.add_argument("--context", type=Path, default=Path("context"))
    parser.add_argument("--checkpoint", type=Path, default=Path("results/demo/ppo_adaptive.pt"))
    parser.add_argument("--stress-trace", type=Path, default=Path("sample_stress_trace.txt"))
    parser.add_argument("--out", type=Path, default=Path("results/demo/safety_guard_evaluation.json"))
    args = parser.parse_args()

    all_traces = sorted(p for p in (args.context / "traces" / "train").iterdir() if p.is_file())
    validation_traces = all_traces[::5]
    if not validation_traces:
        raise SystemExit("no held-out validation traces found")
    rates = [300, 750, 1200, 1850, 2850, 4300]
    per_level = [sorted((args.media_dir / f"q{i}").glob("chunk*.mp4")) for i in range(6)]
    chunks = min(map(len, per_level), default=0)
    if not chunks:
        raise SystemExit("each bitrate level must contain encoded segments")
    chunk_sizes = np.asarray([[p.stat().st_size for p in files[:chunks]] for files in per_level],
                             dtype=np.float32)
    agent = PPOAgent.load(args.checkpoint)

    stress_values = [float(token) for token in args.stress_trace.read_text().replace(",", " ").split()
                     if token.replace(".", "", 1).isdigit()]
    if not stress_values:
        raise SystemExit("stress trace has no positive throughput values")
    with tempfile.TemporaryDirectory(prefix="abr_guard_") as temp:
        stress_path = Path(temp) / "stress.trace"
        stress_path.write_text("".join(f"{i} {value / 1000.0:.8f}\n"
                                        for i, value in enumerate(stress_values)))

        def aggregate(paths, policy, params=None):
            rows = [run_episode(args.context, path, chunk_sizes, agent, policy, params) for path in paths]
            return {key: float(np.mean([row[key] for row in rows])) for key in ("qoe", "rebuffer_s")}

        baseline = aggregate(validation_traces, "baseline")
        ppo = aggregate(validation_traces, "ppo")
        variants = list(itertools.product((1, 2, 3, 4), (0.6, 0.8, 1.0), (0.0, 1.0, 2.0)))
        candidates = [{"params": list(params), **aggregate(validation_traces, "shield", params)}
                      for params in variants]
        best = max(candidates, key=lambda item: item["qoe"])
        stress = {name: run_episode(args.context, stress_path, chunk_sizes, agent, policy,
                                    tuple(best["params"]) if policy == "shield" else None)
                  for name, policy in (("baseline", "baseline"), ("ppo", "ppo"), ("ppo_safety_guard", "shield"))}

    guard_window, guard_margin, guard_slack = best["params"]
    output = {"video_segments": chunks, "validation_trace_count": len(validation_traces),
              "paired_same_trace_sample_per_chunk": True,
              "guard": {"throughput_estimator":
                            f"{guard_margin:g} * minimum of the most recent {guard_window} observations",
                        "download_deadline": f"current buffer + {guard_slack:g} seconds",
                        "selected_validation_parameters": best["params"]},
              "validation_mean": {"rate_based": baseline, "ppo_raw": ppo, "ppo_safety_guard": best},
              "stress_trace": stress}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps({"selected_guard": best, "validation_baseline": baseline,
                      "validation_raw_ppo": ppo, "stress_trace": stress}, indent=2))


if __name__ == "__main__":
    main()

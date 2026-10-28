from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines import bola_like, mpc_like, rate_based
from env import ABREnv, BITRATES_KBPS, conservative_teacher
from ppo_agent import PPOAgent


def run_policy(env: ABREnv, policy, episodes: int = 30):
    totals = []
    actions = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        reward = 0.0
        rebuffer = 0.0
        switch = 0.0
        bitrate = []
        while not done:
            action = policy(obs)
            obs, r, done, _, info = env.step(action)
            reward += r
            rebuffer += info["rebuffer"]
            switch += info["switch"]
            bitrate.append(info["bitrate_kbps"])
            actions.append(action)
        totals.append(
            {
                "qoe": reward,
                "avg_bitrate": float(np.mean(bitrate)),
                "rebuffer": rebuffer,
                "switches": switch,
            }
        )
    return {k: float(np.mean([x[k] for x in totals])) for k in totals[0]}, Counter(actions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="context")
    parser.add_argument("--checkpoint", default="artifacts_adaptive_ppo/ppo_adaptive.pt")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out-dir", default="artifacts_adaptive_ppo")
    args = parser.parse_args()

    policies = {
        "Rate-based": rate_based,
        "BOLA": bola_like,
        "MPC": mpc_like,
        "Teacher": conservative_teacher,
    }
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        agent = PPOAgent.load(ckpt)
        policies["PPO"] = lambda obs: agent.act(obs, deterministic=True)[0]
    else:
        print(f"checkpoint not found: {ckpt}; evaluating heuristic policies only")

    results = {}
    action_counts = {}
    for split in ["train", "test"]:
        results[split] = {}
        action_counts[split] = {}
        for name, policy in policies.items():
            metrics, counts = run_policy(ABREnv(args.context, split=split, seed=42), policy, args.episodes)
            results[split][name] = metrics
            action_counts[split][name] = {str(int(BITRATES_KBPS[k])): v for k, v in sorted(counts.items())}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation.json").write_text(json.dumps({"metrics": results, "actions": action_counts}, indent=2))

    labels = list(policies)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, [results["train"][p]["qoe"] for p in labels], width=0.36, label="FCC train traces")
    ax.bar(x + 0.18, [results["test"][p]["qoe"] for p in labels], width=0.36, label="Norway test traces")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Average QoE")
    ax.set_title("ABR Policy Generalization: FCC vs Norway")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "qoe_generalization.png", dpi=160)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

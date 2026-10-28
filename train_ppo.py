from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from env import ABREnv, BITRATES_KBPS, conservative_teacher
from ppo_agent import PPOAgent, PPOConfig, compute_gae


def collect_teacher(env: ABREnv, episodes: int = 30):
    observations, actions = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = conservative_teacher(obs)
            observations.append(obs)
            actions.append(action)
            obs, _, done, _, _ = env.step(action)
    return np.asarray(observations), np.asarray(actions)


def evaluate_agent(agent: PPOAgent, context: str, trace_paths: list[Path], seed: int):
    env = ABREnv(context, split="train", seed=seed, trace_paths=trace_paths)
    rewards, rebuffer, actions = [], [], []
    for _ in range(len(trace_paths)):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _, _ = agent.act(obs, deterministic=True)
            actions.append(action)
            obs, _, done, _, _ = env.step(action)
        rewards.append(env.total_reward)
        rebuffer.append(env.total_rebuffer)
    counts = {int(BITRATES_KBPS[i]): actions.count(i) for i in range(6)}
    return float(np.mean(rewards)), float(np.mean(rebuffer)), counts


def main():
    parser = argparse.ArgumentParser(description="Train a seeded BC-warm-started PPO ABR policy")
    parser.add_argument("--steps", type=int, default=20000, help="positive environment steps for PPO updates")
    parser.add_argument("--context", default="context")
    parser.add_argument("--out", default="results/demo/ppo_adaptive.pt")
    parser.add_argument("--bc-epochs", type=int, default=80)
    parser.add_argument("--teacher-episodes", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive; zero performs no PPO updates")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_traces = sorted(p for p in (Path(args.context) / "traces" / "train").iterdir() if p.is_file())
    # Stable held-out validation set. Checkpoint selection never uses test traces.
    validation_paths = all_traces[::5]
    validation_set = set(validation_paths)
    training_paths = [path for path in all_traces if path not in validation_set]
    if not training_paths or not validation_paths:
        raise SystemExit("need enough train traces for a train/validation split")

    env = ABREnv(args.context, split="train", seed=args.seed, trace_paths=training_paths)
    agent = PPOAgent(config=PPOConfig(lr=1e-4, entropy_coef=0.002, epochs=3))
    obs_bc, act_bc = collect_teacher(env, episodes=args.teacher_episodes)
    bc_stats = agent.behavior_clone(obs_bc, act_bc, epochs=args.bc_epochs, weighted=True, lr=1e-3)
    print(f"seed={args.seed} train_traces={len(training_paths)} validation_traces={len(validation_paths)}")
    print(f"behavior_clone_accuracy={bc_stats['accuracy']:.3f} bc_samples={len(act_bc)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_reward, best_rebuffer, best_counts = evaluate_agent(agent, args.context, validation_paths, args.seed + 1000)
    agent.save(out)
    logs = [{"step": 0, "validation_reward": best_reward, "validation_rebuffer": best_rebuffer,
             "action_counts": best_counts, "stage": "behavior_clone", "saved": True}]
    obs, _ = env.reset()
    total_steps = 0
    while total_steps < args.steps:
        batch = {key: [] for key in ("obs", "actions", "logp", "rewards", "values", "dones")}
        target = min(args.rollout_steps, args.steps - total_steps)
        for _ in range(target):
            action, logp, value = agent.act(obs)
            next_obs, reward, done, _, _ = env.step(action)
            for key, value_item in (("obs", obs), ("actions", action), ("logp", logp),
                                    ("rewards", reward), ("values", value), ("dones", done)):
                batch[key].append(value_item)
            obs = next_obs
            total_steps += 1
            if done:
                obs, _ = env.reset()

        rewards = np.asarray(batch["rewards"], dtype=np.float32)
        values = np.asarray(batch["values"], dtype=np.float32)
        dones = np.asarray(batch["dones"], dtype=np.float32)
        # Bootstrap a rollout cut in the middle of an episode; terminal batches bootstrap to zero.
        bootstrap = 0.0 if dones[-1] else agent.act(obs, deterministic=True)[2]
        advantages, returns = compute_gae(rewards, values, dones, agent.config.gamma, agent.config.lam,
                                          bootstrap_value=bootstrap)
        update = {"obs": np.asarray(batch["obs"], dtype=np.float32),
                  "actions": np.asarray(batch["actions"], dtype=np.int64),
                  "logp": np.asarray(batch["logp"], dtype=np.float32),
                  "advantages": advantages, "returns": returns}
        loss = agent.update(update)

        # Keep the policy near the teacher on training-only labels while PPO explores.
        rng = np.random.default_rng(args.seed + total_steps)
        idx = rng.choice(len(obs_bc), size=min(512, len(obs_bc)), replace=False)
        agent.behavior_clone(obs_bc[idx], act_bc[idx], epochs=1, weighted=True, lr=2e-4)

        val_reward, val_rebuffer, val_counts = evaluate_agent(
            agent, args.context, validation_paths, args.seed + 1000
        )
        improved = val_reward > best_reward
        if improved:
            best_reward, best_rebuffer, best_counts = val_reward, val_rebuffer, val_counts
            agent.save(out)
        logs.append({"step": total_steps, "loss": loss, "validation_reward": val_reward,
                     "validation_rebuffer": val_rebuffer, "action_counts": val_counts,
                     "stage": "ppo_finetune", "saved": improved})
        print(f"steps={total_steps}/{args.steps} loss={loss:.3f} validation_reward={val_reward:.2f} "
              f"best={best_reward:.2f} validation_rebuffer={val_rebuffer:.2f} "
              f"{'saved' if improved else ''}")

    out.with_suffix(".json").write_text(json.dumps({"seed": args.seed, "ppo_steps": args.steps,
        "training_trace_count": len(training_paths), "validation_trace_count": len(validation_paths),
        "best_validation_reward": best_reward, "best_validation_rebuffer": best_rebuffer,
        "best_action_counts": best_counts, "checkpoints": logs}, indent=2))
    print(f"saved validation-selected PPO checkpoint: {out}")


if __name__ == "__main__":
    main()

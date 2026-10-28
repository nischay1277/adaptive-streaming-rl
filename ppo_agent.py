from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 25, action_dim: int = 6):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        return self.policy(x), self.value(x).squeeze(-1)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    epochs: int = 4


class PPOAgent:
    def __init__(self, obs_dim: int = 25, action_dim: int = 6, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.model = ActorCritic(obs_dim, action_dim)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = self.model(x)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def update(self, batch: dict[str, np.ndarray]):
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32)
        adv = torch.as_tensor(batch["advantages"], dtype=torch.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        last_loss = 0.0
        for _ in range(self.config.epochs):
            logits, values = self.model(obs)
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = torch.exp(logp - old_logp)
            clipped = torch.clamp(ratio, 1 - self.config.clip, 1 + self.config.clip) * adv
            policy_loss = -torch.min(ratio * adv, clipped).mean()
            value_loss = ((returns - values) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optim.step()
            last_loss = float(loss.item())
        return last_loss

    def behavior_clone(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        epochs: int = 8,
        weighted: bool = True,
        lr: float = 1e-3,
        batch_size: int = 256,
    ):
        x = torch.as_tensor(observations, dtype=torch.float32)
        y = torch.as_tensor(actions, dtype=torch.int64)
        if weighted:
            counts = torch.bincount(y, minlength=self.model.policy.out_features).float()
            weights = counts.sum() / torch.clamp(counts, min=1.0)
            weights = weights / weights.mean()
        else:
            weights = None
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        bc_optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        n = len(x)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                logits, _ = self.model(x[idx])
                loss = loss_fn(logits, y[idx])
                bc_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                bc_optim.step()

        with torch.no_grad():
            logits, _ = self.model(x)
            pred = torch.argmax(logits, dim=-1)
            accuracy = (pred == y).float().mean().item()
            pred_counts = torch.bincount(pred, minlength=self.model.policy.out_features).cpu().numpy().tolist()
        return {"accuracy": accuracy, "pred_counts": pred_counts}

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, path)

    @classmethod
    def load(cls, path: str | Path):
        data = torch.load(path, map_location="cpu")
        agent = cls(config=PPOConfig(**data.get("config", {})))
        agent.model.load_state_dict(data["model"])
        agent.model.eval()
        return agent


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95, bootstrap_value=0.0):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last_gae = 0.0
    next_value = float(bootstrap_value)
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
        next_value = values[t]
    return adv, adv + values

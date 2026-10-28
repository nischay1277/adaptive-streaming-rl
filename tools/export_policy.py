"""
Export the trained PPO actor (an MLP) from a .pt checkpoint to JSON so the
browser demo can run the REAL policy instead of a fake Q-table.

I could not see ppo_agent.py, so this is architecture-agnostic for plain MLPs
(Linear layers separated by a fixed activation). Run it in two steps:

  1) Inspect:   python export_policy.py artifacts_adaptive_ppo/ppo_adaptive.pt
  2) Export:    python export_policy.py artifacts_adaptive_ppo/ppo_adaptive.pt \
                    --prefix actor --activation relu --out simulator_web/policy.json

Step 1 prints every tensor name/shape. Pick the prefix that belongs to the
actor/policy head (the part that outputs one logit per bitrate level).
The script then verifies its numpy forward pass against torch on random
inputs; if they do not match, do NOT use the export.

If your agent normalizes observations, pass --obs-mean-key / --obs-std-key
(names from step 1) or replicate the same normalization in the browser.
"""
import argparse
import json

import numpy as np
import torch


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "policy", "agent"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def tensors(sd):
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def act_fn(name):
    return {"relu": lambda x: np.maximum(x, 0), "tanh": np.tanh}[name]


def forward(x, layers, act):
    f = act_fn(act)
    for i, (W, b) in enumerate(layers):
        x = x @ W.T + b
        if i < len(layers) - 1:
            x = f(x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--prefix", help="tensor-name prefix of the actor, e.g. actor")
    ap.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    ap.add_argument("--obs-mean-key")
    ap.add_argument("--obs-std-key")
    ap.add_argument("--out", default="policy.json")
    a = ap.parse_args()

    sd = tensors(load_state_dict(a.checkpoint))

    if not a.prefix:
        print("Tensors in checkpoint:")
        for k, v in sd.items():
            print(f"  {k:60s} {tuple(v.shape)}")
        print("\nRe-run with --prefix <actor prefix> to export.")
        return

    keys = [k for k in sd if k.startswith(a.prefix)]
    weights = [k for k in keys if k.endswith("weight") and sd[k].ndim == 2]
    if not weights:
        raise SystemExit(f"No 2-D weights found under prefix '{a.prefix}'.")

    layers = []
    for wk in weights:  # state_dict order == layer order for nn.Sequential
        bk = wk[: -len("weight")] + "bias"
        W = sd[wk].numpy().astype(np.float64)
        b = sd[bk].numpy().astype(np.float64) if bk in sd else np.zeros(W.shape[0])
        layers.append((W, b))

    # shape chain sanity check
    for (W1, _), (W2, _) in zip(layers, layers[1:]):
        assert W1.shape[0] == W2.shape[1], "Layer shapes do not chain; wrong prefix?"

    mean = sd[a.obs_mean_key].numpy().tolist() if a.obs_mean_key else None
    std = sd[a.obs_std_key].numpy().tolist() if a.obs_std_key else None

    # verify numpy forward against the original torch tensors
    rng = np.random.default_rng(0)
    in_dim = layers[0][0].shape[1]
    x = rng.normal(size=(8, in_dim))
    with torch.no_grad():
        t = torch.tensor(x, dtype=torch.float32)
        f = torch.relu if a.activation == "relu" else torch.tanh
        for i, (W, b) in enumerate(layers):
            t = t @ torch.tensor(W, dtype=torch.float32).T + torch.tensor(b, dtype=torch.float32)
            if i < len(layers) - 1:
                t = f(t)
    err = np.abs(forward(x, layers, a.activation) - t.numpy()).max()
    print(f"max |numpy - torch| = {err:.2e}")
    assert err < 1e-4, "Mismatch: check --activation / --prefix"

    out = {
        "activation": a.activation,
        "input_dim": int(in_dim),
        "n_actions": int(layers[-1][0].shape[0]),
        "obs_mean": mean,
        "obs_std": std,
        "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers],
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh)
    print(f"Wrote {a.out}: {len(layers)} layers, input {in_dim}, {out['n_actions']} actions")


if __name__ == "__main__":
    main()

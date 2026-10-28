# Audit

Date: 2026-10-27
Project: `D:\rl_project`

## Findings

### 1. Checkpoint provenance

`train_ppo.py:44-65` creates the PPO agent, collects Teacher labels, runs behavior cloning, evaluates the warm start, and saves the checkpoint. The command used recently was:

```powershell
python train_ppo.py --steps 0 --bc-epochs 120 --teacher-episodes 120
```

Because `train_ppo.py:82-114` contains the PPO update loop and `--steps 0` skips that loop, the current checkpoint is behavior-cloning-only. It is not evidence of PPO fine-tuning.

The checkpoint tensors are:

- `shared.0` 25 -> 128
- `shared.2` 128 -> 128
- `policy` 128 -> 6
- `value` 128 -> 1

The upgrade kit's exporter must be extended to combine `shared.*` and `policy.*`; using `--prefix shared` alone would export only the hidden layers and produce the wrong action dimension.

### 2. Teacher information

`env.py:112` initializes throughput history with the mean of the first three trace samples before any chunk is downloaded. This gives the initial observation limited future-trace information. `conservative_teacher()` at `env.py:168-174` uses buffer and mean observed throughput; it does not explicitly inspect the full future trace.

### 3. QoE and environment

`env.py:36-38` defines six bitrates `[300, 750, 1200, 1850, 2850, 4300]` kbps, a 4-second chunk, and history length 4. `env.py:127-129` defines:

```text
quality = log(current_bitrate / 300)
switch = abs(log(current_bitrate / previous_bitrate))
reward = quality - 4.3 * rebuffer - 1.10 * switch
```

The environment caps buffer at 30 seconds (`env.py:124-125`) and uses a 25-element observation (`env.py:99`, `env.py:153-165`).

### 4. Train/test statistics

The project currently has 59 train traces and 142 test traces.

Across all trace samples:

| Split | Count | Mean Mbps | Median Mbps | Std Mbps |
|---|---:|---:|---:|---:|
| Train | 59 | 1.3060 | 0.9989 | 1.0172 |
| Test | 142 | 1.6692 | 1.4465 | 0.9783 |

The test set has substantially higher typical throughput, which is a plausible explanation for the high test QoE, but it should be reported as distribution shift rather than presented as pure generalization.

### 5. Observation

`env.py:153-165` builds the 25-element observation in this order:

1. Last action divided by 5
2. Buffer divided by 30
3. Four throughput-history values divided by 6
4. Four download-time-history values divided by 10
5. Six next-chunk sizes divided by the global maximum chunk size
6. Remaining-chunk fraction
7. Eight zero padding values

No observation-normalization statistics are stored in the checkpoint. `ppo_agent.py:121` saves only model state and PPO config.

### 6. Policy network

`ppo_agent.py:12-26` uses a shared MLP: `25 -> 128 Tanh -> 128 Tanh`, then separate policy `128 -> 6` and value `128 -> 1` heads. Evaluation uses deterministic argmax actions (`evaluate.py:59-62`).

### 7. Trace-format mismatch

`env.py:52-63` only accepts trace rows with at least two numeric fields and interprets the second field as Mbps. The browser simulator accepts one-value-per-line kbps files. Therefore the browser comparison trace is not automatically equivalent to the Python evaluation trace. A shared loader or explicit conversion is required for an honest comparison.

### 8. Browser demo status

The current `simulator_web/index.html` has two visual players and a buffering spinner, but the RL side is a simplified browser-side policy. It does not load `artifacts_adaptive_ppo/ppo_adaptive.pt`, does not use the Python observation builder, and does not switch real video renditions. Its numbers must not be reported as PPO benchmark results.

### 9. Real-stream status

`real_stream.py:18-31` throttles HTTP delivery and `real_stream.py:38-45` selects actual `.m4s` chunks from the sample chunk context. It produces a telemetry/demo MP4, not a decoded adaptive playback comparison. The upgrade kit's HLS approach is the correct next step for a real two-player demo.

## Required changes, ranked

1. Build a Python policy server that loads the actual PPO checkpoint and accepts decisions using the shared environment observation code.
2. Convert the browser trace input to the same time/Mbps format used by `env.py`, or have the policy server own trace loading.
3. Build aligned six-level HLS renditions from the uploaded video and switch actual segments, not CSS blur.
4. Add validation and ablation experiments before making claims about PPO improvement.
5. Report the easier/higher-throughput test distribution and the fact that the recent checkpoint was BC-only.

## Honest conclusion

The project has a working ABR environment, baseline policies, PPO/BC training code, and a useful visual prototype. The current browser comparison is not yet a scientifically valid comparison of the trained PPO agent. The upgrade kit's Python policy-server plus real segmented-video architecture should replace the fake browser policy before the demo is used as evidence.

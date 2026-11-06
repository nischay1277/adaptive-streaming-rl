# Adaptive Streaming with Reinforcement Learning

A trace-driven adaptive-bitrate (ABR) streaming experiment with a PyTorch PPO policy, heuristic baselines, reproducible evaluation artifacts, and a local segmented-video comparison demo.

At each video chunk, a policy chooses a bitrate while balancing visual quality, rebuffering, and bitrate changes. The project studies how the policy behaves across fixed-broadband training traces and mobile-network test traces.

## Implementation

- `env.py`: Gymnasium-style ABR environment, trace loading, a 25-value observation, and shared QoE calculations.
- `ppo_agent.py`: actor-critic network, behavior cloning, PPO updates, and generalized advantage estimation.
- `train_ppo.py`: teacher initialization and PPO training; checkpoint selection uses a held-out validation split.
- `baselines.py`: rate-based, BOLA-like, and MPC-like heuristic policies. The latter two are simplified implementations, not full reference systems.
- `evaluate.py`: train/test comparisons, action distributions, JSON metrics, and a QoE chart.
- `demo_app.py` and `demo/index.html`: local upload, encoding, policy comparison, and playback interface.

The bitrate ladder is **300, 750, 1200, 1850, 2850, and 4300 kbps**, with 4-second chunks. The per-chunk reward is:

```text
logarithmic quality - 2.66 * rebuffer_seconds - 1.0 * logarithmic_quality_change
```

See `env.py` for exact observation construction and boundary behavior.

## Quick start

Use Python 3.10 or newer. A CPU is sufficient to run the included smoke test and checkpoint inference; training and video encoding costs depend on the workload.

```bash
git clone https://github.com/nischay1277/adaptive-streaming-rl.git
cd adaptive-streaming-rl
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux:

```bash
python -m pip install -r requirements.txt
python test_env.py
python demo_app.py --checkpoint results/demo/ppo_adaptive.pt --host 127.0.0.1 --port 8123
```

Open **http://127.0.0.1:8123/**. Upload a video and a two-column throughput trace, or select the bundled stress trace. Encoding uses the FFmpeg executable provided through `imageio-ffmpeg`; allow time for rendition generation.

The browser demo plays real encoded media, but its transfer/rebuffer comparison is **trace-based simulation**. It serves media locally and estimates timing from segment sizes and the chosen trace; do not describe this path as an end-to-end experiment on a shaped physical network. Separate throttling utilities exist in `real_stream.py` and `demo/throttle_server.py`.

## Evaluate the included checkpoint

```bash
python evaluate.py --context context --checkpoint results/demo/ppo_adaptive.pt --episodes 30 --out-dir results/local-evaluation
```

Outputs are `evaluation.json` and `qoe_generalization.png`. If the checkpoint path does not exist, the evaluator reports that fact and evaluates heuristics only. Check that PPO is present in the output before interpreting a comparison.

The `switches` field accumulates the environment's logarithmic switching penalty; it is not a count of bitrate transitions.

### Included historical results

The following values are rounded from the committed `results/demo/evaluation/evaluation.json`, not a new benchmark run:

| Test policy | Mean episode QoE | Mean bitrate (kbps) | Mean rebuffer (s) |
| --- | ---: | ---: | ---: |
| Rate-based | 51.76 | 1390.31 | 3.57 |
| BOLA-like | 49.63 | 1689.69 | 8.54 |
| MPC-like | 53.51 | 1358.47 | 2.22 |
| Teacher | 59.27 | 1870.31 | 3.57 |
| PPO | 62.37 | 1908.98 | 4.83 |

PPO has the highest QoE in this recorded test summary but does not minimize rebuffering. Results depend on the traces, random sampling, checkpoint, and objective; they do not establish universal superiority. The evaluation JSON does not embed every run parameter, so preserve your command, checkpoint, and environment when generating new results.

## Train a policy

```bash
python train_ppo.py --context context --steps 20000 --bc-epochs 80 --teacher-episodes 80 --seed 7 --out results/local-training/ppo.pt
```

Training uses teacher demonstrations followed by PPO rollouts and validation-based checkpoint selection. Keep checkpoint and hyperparameter selection separate from test traces. The committed demo training metadata records 47 training and 12 validation traces from the 59-file training pool.

## Data and artifacts

| Path | Contents |
| --- | --- |
| `context/traces/train/` | 59 FCC-derived throughput traces |
| `context/traces/test/` | 142 Norway mobile-network traces |
| `context/video_sizes/` | Six per-chunk byte-size tables, lowest bitrate at index 0 |
| `results/demo/` | Included checkpoint, training metadata, evaluation, and guard study |
| `artifacts_adaptive_ppo/` | An additional set of existing experimental artifacts |
| `sample_network_trace.txt`, `sample_stress_trace.txt` | Small local-demo inputs |
| `tools/validate_buffer_guard.py` | Validation-only buffer-guard selection |

Trace rows use `time_seconds throughput_mbps`. Run scripts from the repository root so relative paths resolve. Uploaded media and generated renditions are local artifacts and should remain untracked.

## Further documentation and provenance

- [Workflow guide](RL_ABR_WORKFLOW_GUIDE.md)
- [Data context](context/README.md)
- [Audit notes](AUDIT.md)
- [Demo design notes](demo/DEMO_GUIDE.md) include alternative/historical designs; use `demo_app.py` for the quick start above.
- Trace provenance and the logarithmic QoE formulation relate to [Pensieve](https://github.com/hongzimao/pensieve); see also [Pensieve-PPO](https://github.com/godka/Pensieve-PPO) for background. These references are not additional setup steps.

`test_env.py` checks the environment observation shape and episode progression. A successful smoke test does not validate all PPO training or browser behavior. Use only trusted checkpoint files with PyTorch loading.

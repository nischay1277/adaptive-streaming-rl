# RL ABR project guidance

This repository contains an adaptive-bitrate streaming experiment and a local browser demo.

## Key components

- `env.py`: shared trace-driven ABR environment and QoE/rebuffer dynamics.
- `ppo_agent.py`, `train_ppo.py`: PPO agent and training workflow.
- `baselines.py`, `evaluate.py`: baseline policies and benchmark runner.
- `demo_app.py`, `demo/index.html`: local segmented-video comparison demo.
- `tools/validate_buffer_guard.py`: validation-only selection of the PPO buffer safety guard.
- `context/`: network traces and video-size tables required by the experiment.
- `results/demo/`: checked-in small checkpoint and reproducible evaluation summaries. Uploaded media and encoded renditions are local-only and ignored by Git.

## Run and verify

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python test_env.py
python demo_app.py --checkpoint results\demo\ppo_adaptive.pt --host 127.0.0.1 --port 8123
```

Open `http://127.0.0.1:8123/`, upload a video and a throughput trace (or choose the built-in stress trace), then run the comparison. `demo/run_demo.ps1` is a Windows launcher.

## Experiment integrity

- Keep training, evaluation, and demo QoE calculations consistent with `env.py`.
- Choose checkpoints and hyperparameters on validation traces; never select them using test traces.
- Report negative or mixed results plainly. Performance is trace-dependent and PPO is not expected to win every metric.
- Distinguish simulated trace-based transfer timing from actual network shaping: the local demo serves encoded clips locally and estimates transfer/rebuffer timing from segment sizes and the uploaded trace.
- Do not commit uploaded videos, encoded rendition media, virtual environments, or credentials.

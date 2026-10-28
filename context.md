# RL ABR project: current handoff

## Location and run

Project root: `D:\rl_project`.

The PPO comparison backend is currently running at [http://127.0.0.1:8123/](http://127.0.0.1:8123/). To start it again from PowerShell:

```powershell
cd D:\rl_project
.\demo\run_demo.ps1
```

To reproduce the current checkpoint:

```powershell
.\.venv\Scripts\python.exe train_ppo.py --steps 10000 --out results\demo\ppo_adaptive.pt --seed 7
.\.venv\Scripts\python.exe evaluate.py --checkpoint results\demo\ppo_adaptive.pt --out-dir results\demo\evaluation --episodes 10
```

Port 8123 is used by the new comparison app; 8001 and 8002 are reserved for the user's other local sites/pages.

## Implemented

- `env.py`: ABR environment, bitrate ladder, chunk-size model, and reward. Initial throughput history now starts from the first trace sample, so the initial observation does not leak future trace values.
- `ppo_agent.py`: actor-critic PPO and behavior cloning. GAE accepts a value bootstrap at nonterminal rollout cutoffs.
- `train_ppo.py`: seeded behavior-cloning warm start plus positive PPO updates. A stable 1-in-5 holdout from FCC training traces is used for model selection; Norway test traces are not used to pick the checkpoint. Zero PPO steps are rejected.
- `demo_app.py` plus `demo/index.html`: local web app calling the deterministic PPO checkpoint and Python rate-based baseline through `ABREnv`. The PPO path is wrapped in a validation-tuned buffer-safety guard. Upload a video and throughput trace in kbps, or load the built-in stress trace. The backend encodes six H.264 bitrate levels into aligned four-second MP4 clips. During **Run comparison**, baseline and PPO have separate asynchronous playback loops and can advance or buffer independently; each policy's trace sample is keyed to its own chunk index, so matching chunk numbers use matching throughput. **Step** intentionally advances both together for a paired inspection, and Reset cancels both. Transfer estimates use generated segment byte sizes. The page plays the selected encoded clips and shows per-player reward/buffer/switch/rebuffer metrics and a spinner during rebuffer.
- `demo/run_demo.ps1`: launches the local server and opens the browser.
- `results/demo/ppo_adaptive.pt` and `.json`: seed-7 checkpoint, training log, and validation-selected policy.
- `results/demo/evaluation/evaluation.json` and `qoe_generalization.png`: fresh 10-episode benchmark from `evaluate.py`.
- `tools/validate_buffer_guard.py` and `results/demo/safety_guard_evaluation.json`: reproducible guard selection on held-out traces and stress-trace comparison.

## Current run results

Training used 10,000 PPO environment steps, 47 training traces, and 12 held-out validation traces. Behavior-cloning start scored 48.83 mean validation reward; the best PPO checkpoint scored 50.68, with 5.29s mean rebuffer on that validation slice. This is model-selection data, not test performance.

Fresh `evaluate.py` run used 10 episodes per split (not the full datasets). Mean QoE / rebuffer seconds:

| Split | Rate-based | BOLA | MPC | Teacher | PPO |
|---|---:|---:|---:|---:|---:|
| FCC train | 13.59 / 4.13 | -285.35 / 79.92 | 14.90 / 3.50 | 19.38 / 4.13 | 18.57 / 4.78 |
| Norway test | 45.48 / 3.57 | 35.00 / 8.54 | 49.42 / 2.22 | 52.35 / 3.57 | 56.79 / 4.04 |

The test sample favors PPO in QoE, while PPO rebuffers more than MPC; the train sample favors Teacher. Do not claim universal improvement. `evaluate.py`'s `switches` field is summed absolute log bitrate change, whereas the live demo's quality changes field counts discrete action changes.

## Demo limitation (state this plainly)

The demo now switches real H.264 encoded video segments at six bitrate targets; audio is omitted. Segment requests are served locally and are not rate-throttled to the uploaded trace. Download/rebuffer timing is simulated by `ABREnv` using the actual encoded segment byte sizes and the uploaded throughput trace; playback is accelerated for presentation. Describe it as a trace-driven segmented-video ABR demo, not as a real network-shaping test.

The older `simulator_web/index.html` remains a separate illustrative Q-table page and is not used by the PPO comparison app. `policy_server.py` on port 8110 is also separate.

## Verification already completed

- `python test_env.py`: PASS, 49 chunks, 1.486s total rebuffer for its smoke trace.
- Python syntax compilation passed for `env.py`, `ppo_agent.py`, `train_ppo.py`, `demo_app.py`.
- Demo JS syntax check passed.
- FFmpeg upload/encoding/serving was exercised end-to-end on a generated 8-second test clip: six H.264 ladders, two aligned segments each; an encoded MP4 segment returned HTTP 200, then both policies completed a paired environment step.
- Local `/api/info`, `/`, session creation, paired actions/rewards/rebuffer, and reset returned successfully on port 8123.
- Independent policy stepping was verified over the API: after two baseline steps and one PPO step, the respective completed chunk counts were 2 and 1; each request returned only its requested policy, and the trace indices followed each policy's own chunk count. Python and embedded JavaScript syntax checks pass.
- After the user's stress-trace screenshot, fixed the comparison so both policies use the same throughput sample for each chunk. Replayed their uploaded 33-segment video with `sample_stress_trace.txt`: 32 compared chunks, baseline rebuffer 74.13s, PPO rebuffer 75.88s; both saw the same trace sample on every chunk. The original unpaired cursor run showed 63.17s vs 103.20s because the policies had drifted to different trace samples (for example 1600 vs 1000 kbps on the displayed final chunk).
- Added a transparent PPO safety guard selected from 36 settings using the 12 held-out FCC validation traces. It uses 0.6 times the minimum of the four recent throughput observations and caps the action when estimated segment transfer exceeds buffer + 2 seconds. Validation mean QoE/rebuffer: baseline 15.83/3.11s, raw PPO 22.26/3.79s, PPO+guard 27.57/2.26s. On the separate stress trace and same uploaded video: baseline -308.41 QoE/74.13s rebuffer, raw PPO -316.91/75.88s, PPO+guard -267.83/63.93s. These are simulator results on these traces, not a universal guarantee.

## Useful files

- `env.py`, `baselines.py`, `ppo_agent.py`, `train_ppo.py`, `evaluate.py`
- `demo_app.py`, `demo/index.html`, `demo/run_demo.ps1`
- `results/demo/ppo_adaptive.json`
- `results/demo/evaluation/evaluation.json`
- `sample_network_trace.txt`, `sample_stress_trace.txt`

The project report is out of scope; do not edit or generate it unless asked again.

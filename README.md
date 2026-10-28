# RL ABR Project: PPO Adaptive Bitrate Streaming

This project trains and evaluates a PPO-based adaptive bitrate (ABR) policy for video streaming.
The experiment story is:

```text
FCC network traces -> PPO ABR training -> frozen policy -> Norway HSDPA evaluation
```

The included demo also performs trace-driven HTTP transfers of real sample `.m4s` video chunk bytes.

## What Is Included

- `env.py` - Gymnasium-style ABR simulator
- `ppo_agent.py` - PyTorch actor-critic PPO agent
- `train_ppo.py` - warm-start plus PPO fine-tuning
- `evaluate.py` - FCC vs Norway metrics and comparison plot
- `baselines.py` - rate-based, BOLA-like, and MPC-like baselines
- `real_stream.py` - local HTTP transfer demo with trace-driven byte pacing
- `test_env.py` - simulator smoke test
- `context/` - FCC traces, Norway traces, video chunk-size tables, sample chunks

## Setup On Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Smoke Test

```powershell
python test_env.py
```

Expected:

```text
PASS: ABR environment smoke tests
```

## Train PPO

```powershell
python train_ppo.py --steps 20000
```

Default checkpoint:

```text
results/demo/ppo_adaptive.pt
```

For a quicker first run, use fewer steps:

```powershell
python train_ppo.py --steps 5000
```

## Evaluate Generalization

```powershell
python evaluate.py
```

To evaluate the current demo checkpoint and save a separate result set:

```powershell
python evaluate.py --checkpoint results\demo\ppo_adaptive.pt --out-dir results\demo\evaluation --episodes 10
```

Outputs:

- `artifacts_adaptive_ppo/evaluation.json`
- `artifacts_adaptive_ppo/qoe_generalization.png`

## Run Real-Transfer Demo

```powershell
python real_stream.py
```

If no checkpoint exists yet, it uses the conservative teacher fallback so the demo still runs.
After training, it automatically uses `artifacts_adaptive_ppo/ppo_adaptive.pt`.

Output:

```text
artifacts_adaptive_ppo/real_stream_demo.mp4
```

## Presentation Explanation

During training, PPO does not process video frames. It learns from:

- recorded network throughput traces
- precomputed video chunk sizes at six bitrate levels
- buffer, rebuffering, bitrate-switching, and QoE reward dynamics

The real-transfer demo then reuses the frozen policy to select actual sample video chunks and downloads them over a local HTTP server whose byte rate is paced according to a Norway trace.

## Important Limitation

This Windows-friendly demo uses application-level HTTP byte pacing. It is a real transfer of actual chunk bytes, but it is not Linux kernel-level shaping through `tc` or Mahimahi.
# Browser simulator

Run the explainable Q-learning simulator from the project root:

```powershell
python simulator_web.py
```

Open `http://127.0.0.1:8787` in a browser. Choose an MP4/WebM for the preview and a CSV/TXT network trace. The trace may contain one throughput value per line, or rows such as `time,throughput` where throughput is in kbps. Use **Step** to inspect one Q update at a time or **Run simulation** to animate the policy.

The video is used as a local preview. The simulated player overlays the selected bitrate and reports buffer, estimated rebuffer, reward, switches, and the updated `Q(state, action)` value. The browser simulator is intentionally separate from the PPO evaluator: it is an educational Q-table view for understanding the decision loop, while `evaluate.py` remains the authoritative PPO/baseline benchmark.


## Real PPO policy server

The older educational browser prototype is not the trained agent. To serve its raw deterministic action endpoint:

```powershell
cd D:\rl_project
.venv\Scripts\activate
python policy_server.py --port 8110
```

Check the loaded checkpoint from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8110/info
```

The server exposes `POST /decide` with a JSON body containing an `observation` array of exactly 25 values. It returns the deterministic PPO action and bitrate. The observation must be built with the same ordering and scaling as `env.py`; the server deliberately rejects the wrong shape instead of silently making an invalid decision.

The launcher is also available as:

```powershell
.\demo\run_policy_server.ps1
```

The endpoint is a low-level inference API. The current comparison app described above already connects the checkpoint and baseline to the shared environment.

## Run the PPO comparison demo (current)

Train a fresh policy with actual PPO updates. Checkpoint selection uses a held-out slice of the FCC training traces; the Norway test split is not used for selection:

```powershell
cd D:\rl_project
.\.venv\Scripts\python.exe train_ppo.py --steps 10000 --out results\demo\ppo_adaptive.pt --seed 7
```

Then launch the backend and open the page on port 8123 (8001 and 8002 remain available to other local sites):

```powershell
.\demo\run_demo.ps1
```

Upload an MP4/WebM/MOV/MKV and a text/CSV throughput trace in kbps, or use **Load built-in stress trace**. The backend encodes the source into six H.264 bitrate levels and aligned four-second MP4 segments. **Run comparison** starts two independent playback loops: a slow/stalled baseline does not block PPO, and vice versa. Each policy advances its own environment and player; the trace sample is selected by that policy's chunk number, so corresponding chunks still use the same network sample. **Step** remains a paired single-chunk inspection. Reset stops both loops. The browser plays the actual segment chosen by each policy. The PPO checkpoint is wrapped in a validation-tuned buffer-safety guard; the page labels this as PPO + guard and marks when it caps the raw action. The spinner follows each environment's computed rebuffer duration. The page shows the checkpoint hash. Live QoE values use the same reward as training and evaluation.

The guard is selected reproducibly on held-out FCC validation traces with:

```powershell
python tools\validate_buffer_guard.py --media-dir results\demo\media\<video-id>
```

The chosen rule estimates throughput as 0.6 times the minimum of the four recent observations and caps the action if estimated transfer time exceeds current buffer plus two seconds. Segment requests are served from localhost, not bandwidth-throttled. The uploaded trace drives simulator transfer/rebuffer estimates using actual encoded segment byte sizes. Playback is accelerated for presentation. The old `simulator_web\index.html` is the separate educational Q-table prototype and is not the trained PPO comparison.


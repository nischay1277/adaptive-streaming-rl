# How the RL adaptive-bitrate project works

This guide explains the project from the video upload through encoding, network-trace input, bitrate selection, playback, training, and evaluation. It describes the implementation in this repository, including its simplifying assumptions.

## 1. The goal in one sentence

For every short video segment, choose one of several encoded quality levels using the recent network conditions and playback buffer, aiming for good visual quality without too much rebuffering or abrupt quality changes.

The project compares two decision methods:

1. **Rate-based baseline:** a fixed rule estimates bandwidth from recent samples and picks the highest bitrate below a safety margin.
2. **PPO + buffer safety:** a trained reinforcement-learning policy proposes a bitrate; a separate guard can lower an unsafe proposal before the segment is downloaded.

The PPO model does not inspect video pixels. It receives numerical streaming state: buffer, throughput history, download history, previous quality, and candidate segment sizes.

## 2. The end-to-end flow

```text
Source video ──FFmpeg──> 6 quality ladders × aligned 4-second MP4 segments
                                  │
Throughput trace ─────────────────┤
                                  ▼
                       ABR environment builds state
                                  │
                     baseline rule or PPO proposes action
                                  │
                      PPO safety guard may cap action
                                  │
                segment size + throughput estimate download time
                                  │
                    buffer decreases; a stall may occur
                                  │
                     segment arrives; 4 seconds are added
                                  │
              calculate reward, record metrics, choose next segment
```

The same video and trace are used for both policies. Each policy has its **own** buffer, playback, and chunk counter. The comparison does not let a slow player pause the other one.

## 3. First: encode the uploaded video

The browser sends the selected source file to the local Python backend. FFmpeg creates six H.264 versions of the video at target rates of **300, 750, 1200, 1850, 2850, and 4300 kbps**. The target picture heights are 240p, 360p, 480p, 720p, 900p, and 1080p respectively. Audio is omitted in these demo encodes.

Each rendition is divided into matching time ranges of about four seconds. For a two-minute source, that is about 30 segments per rendition:

```text
Time range      300 kbps   750 kbps   1200 kbps   ...   4300 kbps
0–4 seconds       file        file        file              file
4–8 seconds       file        file        file              file
8–12 seconds      file        file        file              file
...                ...         ...         ...               ...
```

These are not 180 unrelated clips. They are aligned alternatives for the same points in the timeline. At each segment boundary, the policy chooses which encoded alternative to request. A lower target rate usually produces a smaller file and lower picture quality; the **actual file size** also depends on the scene and encoder, so it is measured from the generated segment rather than assumed from the target bitrate.

The encoding step prepares media. It does **not** train the RL policy, and the uploaded video’s pixels are never the PPO model’s input.

## 4. Next: load the network trace

A trace is a sequence of available throughput samples, in kbps in the web demo. For example:

```text
1800
1500
900
450
700
2100
```

This represents changing network capacity. The trace is an experimental input: it controls the estimated speed at which a selected segment can be downloaded. It does not change the trained model weights.

The browser converts the uploaded samples into the format expected by the Python environment. In the live demo, a policy’s current chunk number selects its trace sample. Thus, baseline chunk 8 and PPO chunk 8 use the same sample even if the two players have progressed at different speeds. Each policy advances through the trace independently by its own chunk count.

That is a reproducible **trace-driven simulation**, not a live measurement of your Wi-Fi. The local server does not throttle the segment bytes to the trace speed. It uses the sample and the real encoded segment size to calculate a simulated download duration.

## 5. What is the ABR environment?

The environment is the streaming world in which policies make decisions. It keeps track of the trace, selected segment, buffer, recent throughput, recent download times, and previous bitrate. It answers: “Given this action, how long would the segment take, would playback stall, what is the new buffer, and what reward did the action receive?”

The available actions are six discrete choices:

| Action index | Target bitrate |
|---:|---:|
| 0 | 300 kbps |
| 1 | 750 kbps |
| 2 | 1200 kbps |
| 3 | 1850 kbps |
| 4 | 2850 kbps |
| 5 | 4300 kbps |

### The observation (what the policy sees)

The policy receives a 25-number observation. Its meaningful fields are:

1. Previous bitrate action, normalized to 0–1.
2. Current buffer, normalized against the 30-second buffer cap.
3. Four recent throughput samples, normalized against 6 Mbps.
4. Four recent segment download times, normalized against 10 seconds.
5. The next segment’s six candidate file sizes, normalized by the largest size in the table.
6. Fraction of the video still remaining.
7. Eight zero-valued padding fields retained by the model’s 25-input interface.

The same kind of observation is given to the baseline and PPO. The model does not see future trace samples when making a decision. At the beginning, the environment seeds the four-sample history with the first observed trace value because it has no past downloads yet.

## 6. What does the network sample do to playback?

For a chosen segment, the simulator measures its encoded file size and estimates:

```text
download time (seconds) = segment size (bytes) × 8 ÷ throughput (bits/second)
```

For example, a 600,000-byte segment is 4.8 million bits. At 1.2 Mbps it takes about 4 seconds to download. This is an illustration; real segment sizes vary.

Let `buffer_before` be video already downloaded and waiting to play:

```text
rebuffer time = max(download time − buffer_before, 0)
new buffer    = min(max(buffer_before − download time, 0) + 4 seconds, 30 seconds)
```

So if the existing buffer is 2 seconds and a segment takes 4 seconds to download, playback stalls for about 2 seconds. After it arrives, its four seconds of video are added to the buffer. If the download finishes while some buffer remains, playback consumes that buffer during the download and then the new segment replenishes it.

The web demo displays a buffering spinner when the simulator estimates rebuffering. The player visuals use the selected encoded MP4 segments, but the spinner duration is based on this model rather than a real network stall.

## 7. The baseline rule

The live baseline is deliberately simple and deterministic. It averages the four recent throughput samples, then leaves 15% headroom:

```text
estimated safe rate = 0.85 × mean(recent four throughputs)
baseline action     = highest available bitrate at or below that safe rate
```

If the safe estimate is below the smallest ladder rung, it chooses 300 kbps. For example, if recent mean throughput is 1,600 kbps, the safe rate is 1,360 kbps, so the rule selects 1,200 kbps.

This baseline is not “no algorithm.” It is a practical rule-based adaptive policy. The repository also contains BOLA-like, MPC-like, and Teacher policies for offline evaluation, but the live page’s non-RL comparison is the rate-based rule.

## 8. What is an RL policy?

A **policy** is a function that maps the current observation to an action. In this project:

```text
policy(observation) → one of the six bitrate actions
```

The rate-based policy is hand-written. The PPO policy is a neural network whose action preferences were learned from experience in the simulator.

**Reinforcement learning (RL)** means an agent repeatedly observes a state, takes an action, receives a reward, and adjusts its policy to improve the expected rewards of future actions. Here, one action is one segment-quality choice. One episode is a sequence of segment decisions over a video and network trace.

The project uses **PPO**, short for Proximal Policy Optimization. PPO is a policy-gradient algorithm. It updates a probability distribution over the six actions while limiting how far each update can move the policy at once. A companion value head estimates how much reward a state is expected to produce; that estimate helps calculate which actions did better or worse than expected. In the demo, action selection is deterministic: it takes the highest-scoring action from the trained policy rather than sampling randomly.

The network architecture is a shared neural network with 25 inputs, two 128-unit tanh layers, a six-action policy output, and a one-value critic output. It is a compact controller, not a large video-understanding model.

## 9. How the reward defines “good streaming”

The environment uses the logarithmic QoE objective from Pensieve’s ABR formulation:

```text
rₜ = ln(Rₜ / Rmin) − μ·rebufferₜ − |ln(Rₜ / Rₜ₋₁)|

Rmin = 300 kbps                 μ = 2.66
```

Here, `Rₜ` is the selected bitrate, `Rₜ₋₁` is the previous bitrate (initialized to 300 kbps), and `rebufferₜ` is the stall or startup-wait duration in seconds. `env.py` implements the logarithmic bitrate utility, rebuffer coefficient 2.66, and unit coefficient on absolute log bitrate change. This is the log-utility variant reported in Pensieve, rather than mixing the paper’s linear-utility rebuffer coefficient (4.3) into a log-utility objective.

The form has a research basis: logarithmic utility gives diminishing returns as bitrate increases; the rebuffer term charges per second of stall; and the log-ratio term penalizes proportional changes, so a 2× jump has the same cost at different absolute rates. The coefficients are an experimental QoE trade-off reported in the paper, not universal constants or perceptual ground truth. A different service should justify different weights with user studies or explicit objectives.

For example, a transition to 1850 kbps from 300 kbps has quality gain `ln(1850/300)=1.819` and switch cost `|ln(1850/300)|=1.819`. If it causes 3.740 seconds of startup wait, its reward is `1.819 − 2.66×3.740 − 1.819 = −9.948`.

The page’s “quality changes” counter is the number of action changes. It is not the same as the magnitude-sensitive log switch penalty.

**Research source:** H. Mao et al., “Neural Adaptive Video Streaming with Pensieve,” ACM SIGCOMM 2017, Equation 6 and Table 1. [Paper PDF](https://www.cs.princeton.edu/courses/archive/spring21/cos563/papers/Pensive.pdf). The paper presents both linear- and logarithmic-quality variants; this project uses its logarithmic variant and corresponding coefficient.

## 10. How the PPO checkpoint is trained

Training happens **before** the web demo is run. The uploaded video and trace are not used to update the checkpoint.

The current training workflow in `train_ppo.py` is:

1. **Split the available training traces.** A stable one-in-five subset is held out for validation. The separate test traces are not used to choose the checkpoint.
2. **Create teacher demonstrations.** A conservative heuristic chooses actions on training traces. These state/action pairs provide initial examples.
3. **Behavior-clone the teacher.** The network first learns to imitate those actions. This is a warm start, not the final training method.
4. **Fine-tune with PPO.** The agent collects rollouts in the ABR environment, observes rewards, calculates advantages with generalized advantage estimation (GAE), and updates its policy and value function.
5. **Keep a small teacher-imitation term during fine-tuning.** This helps prevent unstable drift away from the starting behavior.
6. **Select the saved checkpoint on validation reward.** After training updates, the checkpoint with the best validation reward is saved. Test traces are reserved for later evaluation.

The Teacher used to make training labels is different from the live Rate-based baseline. The Teacher also considers buffer: when buffer exceeds 12 seconds it uses a more aggressive throughput margin; otherwise it uses the conservative margin. The demo does not show Teacher as its baseline lane.

During the web comparison, the saved policy is **frozen**. For each chunk the server builds the observation, calls the checkpoint for an action, applies the guard described next, and executes the action in the environment. The reward is recorded and displayed; it does not cause online weight updates during that run.

## 11. What the safety guard does

The PPO network proposes a raw action. The demo then applies a validation-selected buffer safety rule:

1. Estimate conservative throughput as `max(50 kbps, 0.60 × minimum of the latest 3 throughput observations)`.
2. Set the acceptable transfer deadline to the current buffer plus 2 seconds.
3. Estimate each candidate segment’s transfer time using its actual file size and that conservative throughput.
4. If the PPO action is too risky, cap it to the highest feasible bitrate no higher than PPO’s original proposal. If none qualifies, use the lowest 300 kbps level.

The selected parameters `[3, 0.6, 2.0]` were selected from a small grid on held-out FCC validation traces and saved in `results/demo/safety_guard_evaluation.json`. The running server reads that artifact, keeping its settings aligned with the validation report. The guard can reduce PPO’s proposed quality; it never increases the raw action. It is a heuristic shield, not another RL model or a guarantee against sudden bandwidth drops.

## 12. What happens when you click Run

1. The backend creates two independent ABR environments using the uploaded trace and the uploaded video’s measured segment sizes.
2. For each policy, it obtains the current observation.
3. The baseline computes its rule-based action. PPO computes its raw action; the guard may cap it.
4. The selected segment’s actual byte size and current throughput determine simulated download time.
5. The environment updates buffer, stall time, action history, and cumulative reward.
6. The browser requests and plays the actual encoded MP4 segment selected by each policy. Playback is accelerated for presentation.
7. The page updates the trace marker, selected bitrate, chunk number, buffer, reward, switches, rebuffer total, and bitrate timeline.
8. The next segment decision repeats until each player independently finishes.

**Step one chunk** advances both policies once for a side-by-side decision. **Run full comparison** lets the two players progress independently. Reset clears their playback state and the current comparison session.

## 13. What is being compared—and what is not

The current live page compares **Rate-based baseline** against **PPO + buffer safety**. It is not comparing PPO to YouTube, and it does not show the older Q-table learning page. The Q-table prototype under `simulator_web/` is a separate educational example; the live comparison uses the trained PPO checkpoint and Python ABR environment.

The demo does use real encoded video segments and their real file sizes. However:

- Segment delivery is from the local machine; HTTP transfer is not actually throttled to the trace.
- Download and rebuffer times are simulated from segment size and trace throughput.
- The uploaded trace is replayed by chunk index in the live comparison so both policies see matching samples for matching chunks. This is a fair, simple comparison, but it is not a packet-level network emulator.
- Playback is accelerated and the encodes omit audio.

For these reasons, describe it as a **trace-driven segmented-video ABR comparison**, not as a reproduction of a commercial streaming service or a live network-shaping benchmark.

## 14. How to interpret the project’s current benchmark

`results/demo/evaluation/evaluation.json` reports 10 episodes per split using the retrained checkpoint and research-aligned reward. These are simulator trace results, not a general claim that PPO beats production streaming systems.

| Trace split | Policy | Mean QoE | Mean bitrate | Mean rebuffer |
|---|---|---:|---:|---:|
| FCC train sample | Rate-based | 20.46 | 799 kbps | 4.13 s |
| FCC train sample | PPO | 26.56 | 1093 kbps | 4.79 s |
| Norway test sample | Rate-based | 51.76 | 1390 kbps | 3.57 s |
| Norway test sample | PPO | 62.37 | 1909 kbps | 4.83 s |

PPO has higher average QoE and bitrate in this sample, but also more rebuffering than the rate-based baseline. MPC has lower mean rebuffering on the Norway test sample (2.22 s) and slightly lower QoE than PPO (53.51). State this trade-off plainly.

Separately, the buffer-guard search used 12 held-out FCC validation traces. Means were: rate-based QoE 21.01 / rebuffer 3.17 s; raw PPO 28.54 / 4.02 s; PPO + selected guard 31.23 / 2.35 s. On the built-in stress trace, guarded PPO was −160.88 QoE / 63.42 s rebuffer, versus −185.05 / 73.69 s for rate-based and −189.57 / 75.07 s for raw PPO. The stress trace remains difficult; the guard reduces harm here but does not eliminate stalls. Validation and stress results are distinct from the 10-episode benchmark above and should not be combined as one test set.

These results support claims about this simulator, reward, and selected traces. Stronger claims would require more held-out traces, uncertainty estimates, reward-weight sensitivity, and raw-PPO-versus-guard ablations.

## 15. Worked example: one first-chunk decision

For the first stress-trace sample (2400 kbps), an empty playback buffer, and the saved nature video, the baseline decision is deterministic:

```text
throughput history at start = [2400, 2400, 2400, 2400] kbps
baseline estimate           = 0.85 × mean(history) = 2040 kbps
highest supported rung      = 1850 kbps
measured segment size       = 1,121,978 bytes
transfer time               = 1,121,978 × 8 / 2,400,000 = 3.740 s
startup wait                = max(3.740 − 0, 0) = 3.740 s
new buffer                  = max(0 − 3.740, 0) + 4 = 4.000 s
```

The reward components are quality `+1.819`, rebuffer cost `−2.66×3.740 = −9.948`, and change cost `−1.819`, totaling approximately `−9.948`. The upward quality benefit and proportional-switch penalty cancel for this transition. Other bitrates and previous actions produce different values.

PPO independently reads its 25-number state and outputs six action scores. The demo selects the highest-scoring action deterministically, applies the validation-selected guard, and simulates the final rendition request. The uploaded video’s measured segment size and the current trace sample determine simulated transfer time and buffering. Afterward, the environment updates buffer, throughput and download histories, previous action, and reward; the next decision uses that updated state. Read the PPO raw and final choices from the live decision card because they depend on the loaded checkpoint and actual encoded sizes.

This single baseline example explains the arithmetic; it does not establish that PPO is better. The measured comparison and trade-offs are in the benchmark section.

## 16. A short presentation explanation

> “We encode the source video into six aligned quality levels. For every four-second segment, the player sees recent throughput, its buffer, download history, and candidate segment sizes. The rate-based baseline picks the highest bitrate below 85% of recent mean throughput. Our PPO policy learns an action preference from simulated experience using a reward that values quality and penalizes stalls and bitrate changes. A validation-selected guard can cap risky PPO actions. We replay the same network trace independently for both policies, estimate transfer time from actual encoded segment sizes, and compare QoE, rebuffering, and quality changes. The browser shows the decisions; it is trace-driven simulation, not live packet shaping.”

## 17. Useful project files

- `demo_app.py` — video upload/encoding, session setup, PPO and baseline decisions, guard, and API.
- `demo/index.html` — browser interface, trace visualization, players, and metrics.
- `env.py` — observation layout, buffer/rebuffer model, and reward equation.
- `baselines.py` — rate-based, BOLA-like, and MPC-like baseline rules.
- `ppo_agent.py` — actor-critic network, behavior cloning, PPO update, and checkpoint load.
- `train_ppo.py` — teacher warm start, PPO training, validation, and checkpoint selection.
- `evaluate.py` — offline policy benchmark on train/test trace sets.
- `results/demo/ppo_adaptive.pt` — saved policy checkpoint used by the demo.
- `results/demo/evaluation/evaluation.json` — benchmark snapshot summarized above.
- `results/demo/safety_guard_evaluation.json` — held-out guard selection and stress-trace comparison.
- [Pensieve QoE formulation](https://www.cs.princeton.edu/courses/archive/spring21/cos563/papers/Pensive.pdf).

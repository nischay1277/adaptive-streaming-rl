# Real live comparison demo: how it works and how to build it

Goal: two video players stream the SAME video under the SAME throttled network trace at the same time. The left uses a baseline policy, the right uses the real trained PPO checkpoint. Live metrics show bitrate, buffer, rebuffering, switches, and QoE, and the final numbers can be checked against `evaluate.py`.

This replaces the old browser page, which blurred a preview with CSS and used a fake Q-table. This version switches real video segments over a real (throttled) HTTP connection.

## Architecture

```
            +-----------------------+
            |  policy_server.py     |  loads artifacts_adaptive_ppo/ppo_adaptive.pt
            |  POST /decide -> level|  builds obs with env.py's own function
            +-----------+-----------+
                        ^  chunk stats (throughput, buffer, last level)
                        |
+---------------------------------------------------+
| compare.html (browser)                            |
|  left: hls.js + baseline policy (JS or server)    |
|  right: hls.js + policy_server decisions          |
+--------+------------------------------+-----------+
         | segments                     | segments
         v                              v
 throttle_server :8101           throttle_server :8102
 (same trace, same clock)        (same trace, same clock)
         \                              /
          renditions/ (six levels, aligned 4s segments)
```

Why two servers: each player gets an independent bandwidth pipe replaying the same trace, so neither player slows the other, and both see identical network conditions.

## Option A (recommended): Python policy server

The browser asks `policy_server.py` for each decision. The server imports the real agent and the environment's own observation builder, so the demo is guaranteed to use the same policy and observation as `evaluate.py`. Nothing is re-implemented in JavaScript, so there is no observation-mismatch risk.

`POST /decide` body: `{"player":"ppo","throughput_kbps":[...recent chunk throughputs...],"buffer_s":4.0,"last_level":2,"chunks_left":30,"download_time_s":[...]}` (adapt the fields to whatever the environment's observation needs). Response: `{"level":3,"policy":"ppo_adaptive.pt"}`. It also exposes `GET /info` with the checkpoint path and a hash so the page can show the "trained checkpoint" label honestly.

## Option B (fallback): Export weights and run in the browser

Use `tools/export_policy.py` and `tools/policy_infer.js`. Works offline without Python running, but the JavaScript must rebuild the observation exactly like `env.py`. Verify by feeding the same observation to Python and JS and comparing actions on 100 random observations before trusting it.

## Steps

1. Renditions: install ffmpeg, then `.\make_renditions.ps1 -Source input.mp4 -Out renditions` (edit the ladder to match `env.py`). Use a video at least as long as your trace segment (a few minutes).
2. Throttle: check how `real_stream.py` already throttles; reuse it if it can serve files. Otherwise run, in two terminals:
   ```powershell
   python demo\throttle_server.py --root renditions --trace sample_stress_trace.txt --port 8101
   python demo\throttle_server.py --root renditions --trace sample_stress_trace.txt --port 8102
   ```
   Verify the trace units (`--unit kbps|mbps|kBps`) against your real files.
3. Policy server: write `policy_server.py` (Codex prompt 04 does this).
4. Page: `compare.html` with two `<video>` elements and hls.js saved locally in `demo/vendor/hls.min.js` (download it once so the demo works offline).
5. Start: hit `/reset` on both throttle servers at the same moment, then start both players.

## hls.js control loop (the key part)

Turn off hls.js automatic switching and choose the level yourself after every segment:

```js
const hls = new Hls({ startLevel: 0, capLevelToPlayerSize: false });
hls.loadSource('http://localhost:8102/master.m3u8');   // right player (8101 for left)
hls.attachMedia(video);
hls.on(Hls.Events.MANIFEST_PARSED, () => { hls.autoLevelEnabled = false; });

hls.on(Hls.Events.FRAG_LOADED, async (_, data) => {
  const st = data.frag.stats;
  const ms = st.loading.end - st.loading.first;           // download time
  const kbps = (st.total * 8) / ms;                       // measured throughput
  const buf = video.buffered.length ? video.buffered.end(video.buffered.length - 1) - video.currentTime : 0;
  record({ kbps, buf, level: data.frag.level });
  const level = await decide({ kbps, buf, last: data.frag.level });  // policy server or local rule
  hls.nextLoadLevel = level;                              // takes effect on the NEXT segment
});

video.addEventListener('waiting', () => stallStart = performance.now());
video.addEventListener('playing', () => { if (stallStart) { rebuf += (performance.now() - stallStart) / 1000; stallStart = null; } });
```

Check these names against the hls.js version you vendor (`nextLoadLevel` is preferred over `nextLevel` because it does not flush the buffer; confirm in the hls.js API docs). Show the spinner only while the video is stalled (`waiting` to `playing`).

## Making it a fair, honest comparison

- Same video, same trace, same start instant for both players.
- Left baseline should be a real policy from `baselines.py` (Rate-based or MPC), ideally running through the same policy server so both are decided the same way.
- Compute QoE with the shared QoE code (server-side) so the live number equals the evaluation definition.
- After the run, `verify_demo.py` compares the final bitrate, rebuffer, switches, and QoE with the evaluation code on the same trace. Small differences are expected (real network jitter, segment boundaries, hls.js startup); large ones mean a bug.
- Label the right panel "PPO (trained checkpoint)" only if `/info` confirms the checkpoint was loaded.

## Pre-flight checklist (do this before presenting)

- [ ] ffmpeg renditions exist and play at every level
- [ ] Both throttle servers started, `/state` shows the same trace time on both after `/reset`
- [ ] `/info` shows the checkpoint hash you expect
- [ ] Dry run on `sample_stress_trace.txt`: a rebuffer difference is visible, or you know honestly that it is not
- [ ] Screen recording of a good run saved as a fallback
- [ ] Browser cache disabled, hls.js loaded from local file, no internet needed
- [ ] Ports free (8101, 8102, policy server port); 8001 is used by your other site

## If the demo misbehaves in front of people

- PPO shows no advantage on the live trace: say so, then show the aggregated evaluation results, which are the real evidence. A single trace is an illustration, not proof.
- A player stalls at startup: restart both servers and `/reset` again.
- Anything breaks: play the recording and explain it is the same setup.

import { pathToFileURL } from 'node:url';
import path from 'node:path';
const mod = await import(pathToFileURL('C:/Users/whats/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const { FileBlob, PresentationFile } = mod;
const input = 'D:/ABR_RL_Project_Review.pptx';
const p = await PresentationFile.importPptx(await FileBlob.load(input));
const edits = [
 ['sh/zad0nup8','Adaptive Video Streaming with Reinforcement Learning\nProject Review'],
 ['sh/y9kjep83','WORKFLOW OVERVIEW — FIVE STEPS\n\n1. Encode source video into aligned chunks at six quality levels.\n2. Replay measured network traces in a streaming environment.\n3. Observe buffer and throughput history; choose a bitrate.\n4. Train a PPO actor–critic policy and validate a safety guard.\n5. Compare QoE, quality, switching and rebuffering.\n\nKuchibhotla Nischay — 23BAI1245\nRajdeep Bose — 23BAI1289'],
 ['sh/wr6t8nel','Problem and Objective'],
 ['sh/xszahsvq','1. A video is divided into ordered, four-second chunks.\n2. Each chunk has six pre-encoded representations from 300 to 4,300 kbps.\n3. The client downloads one representation at a time over changing throughput.\n4. A bitrate choice affects download delay, playback buffer and future choices.\n5. Goal: maximize session QoE while balancing quality, stalls and quality changes.'],
 ['sh/9srid47m','Streaming as a Reinforcement Learning Problem'],
 ['sh/8rihkzqh','1. The environment supplies the current observation before each chunk request.\n2. The observation has 25 values: buffer, recent throughput and download times, next chunk sizes, previous action and progress.\n3. The agent selects one of six bitrate actions; it does not select pixels or encode video at runtime.\n4. The simulator downloads the chosen chunk using the next trace throughput sample.\n5. The environment returns reward and the next observation; this repeats until the video ends.'],
 ['sh/cz6xw36d','Training the PPO Agent'],
 ['sh/d0fy58ny','1. A conservative teacher labels bitrate choices in simulated FCC sessions.\n2. Behavior cloning gives the policy a supervised warm start.\n3. PPO then gathers on-policy rollouts and updates the policy with a clipped objective.\n4. The actor outputs probabilities for six bitrates; the critic estimates state value to form advantages.\n5. Training uses 47 FCC traces; 12 held-out FCC traces select the checkpoint and guard settings.'],
 ['sh/dgbulwnm','Data and Video Preparation'],
 ['sh/cf2tcr61','1. Training traces: 59 FCC-derived throughput files; split into 47 train and 12 validation traces.\n2. Generalization test: 142 Norway HSDPA mobile traces; Belgium LTE is not included.\n3. The simulator uses six chunk-size tables (video_size_0–5) for sequential bitrate decisions.\n4. The live demo separately encodes an uploaded video into aligned H.264 segments at six target bitrates.\n5. Each demo policy replays the same trace and requests the corresponding segment for each chunk.'],
 ['sh/3ups7mls','One Chunk: State, Action and Reward'],
 ['sh/2tgbyhk7','1. Example state: buffer = 2.0 s, previous bitrate = 750 kbps; recent throughput is weak.\n2. Candidate action: request the 300 kbps representation for the next 4 s chunk.\n3. Its encoded size determines download time: size in bits ÷ trace throughput in bits/s.\n4. If size is 180,000 bytes and throughput is 600 kbps, download time = 2.4 s; rebuffer = max(2.4 − 2.0, 0) = 0.4 s.\n5. Reward = ln(300/300) − 2.66×0.4 − |ln(300/750)| = −1.956 (approx.).'],
 ['sh/lgzqdonq','Evaluation and Live Comparison Workflow'],
 ['sh/kvq9kjm5','1. Upload a source video; FFmpeg creates six aligned H.264 segment ladders for playback comparison.\n2. Upload a throughput trace in kbps; each value drives the next segment download.\n3. Baseline uses the project’s rate-based heuristic; optimized side uses PPO plus a download-deadline safety guard.\n4. Both policies run independently on the same trace and video; a stall in one does not pause the other.\n5. Watch quality, buffer, spinner, rebuffer time, QoE and bitrate decisions update chunk by chunk.'],
 ['sh/18byd4zy','Results and What They Show'],
 ['sh/g72x4zyd','1. FCC training evaluation (10 episodes): PPO mean QoE 26.56 vs rate-based 20.46; rebuffer 4.79 s vs 4.13 s.\n2. Norway test evaluation (10 episodes): PPO QoE 62.37 vs rate-based 51.76; bitrate 1,909 vs 1,390 kbps.\n3. Norway PPO rebuffer was 4.83 s vs 3.57 s for rate-based, so higher mean QoE did not mean fewer stalls.\n4. On 12 held-out FCC traces, the guard raised PPO QoE 28.54→31.23 and lowered rebuffer 4.02→2.35 s.\n5. These are simulator results on finite trace samples; the demo is not a production streaming service or proof of universal superiority.'],
 ['sh/1cfmhgne','From Upload to Playback'],
 ['sh/0b65obm9','1. Select an MP4/WebM source video and a throughput trace.\n2. FFmpeg encodes aligned H.264 segments at six bitrate targets before playback.\n3. At each chunk, the trace drives simulated download time and the policy selects a segment.\n4. Buffer drains during playback; if it reaches zero, the player shows the buffering spinner.\n5. Compare baseline and PPO-plus-guard playback, bitrate, stalls and QoE side by side.'],
 ['sh/sjud8vm9','References and Dataset Links'],
 ['sh/t43u103u','1. Project trace inventory and preprocessing lineage: hongzimao/Pensieve traces — https://github.com/hongzimao/pensieve/tree/master/traces\n2. FCC raw Measuring Broadband America, June 2016 — http://data.fcc.gov/download/measuring-broadband-america/2016/data-raw-2016-jun.tar.gz\n3. Norway HSDPA raw mobile traces — http://home.ifi.uio.no/paalh/dataset/hsdpa-tcp-logs/\n4. Mao, Netravali & Alizadeh (2017), Pensieve, SIGCOMM — https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf\n5. Schulman et al. (2017), Proximal Policy Optimization Algorithms — https://arxiv.org/abs/1707.06347\n\nRelated work: Spiteri et al. (2016), BOLA — https://arxiv.org/abs/1601.06748. The project’s BOLA-like baseline is a simplified heuristic, not the published BOLA algorithm.']
];
for (let i=0;i<edits.length;i++) { const sh=p.resolve(edits[i][0]); sh.text.replace(sh.text.toString(), edits[i][1]); }
for (const id of ['im/xcfqdkj6','im/xgr2x4ne']) p.resolve(id).delete();
const out = 'D:/rl_project/presentation_build/candidate.pptx';
await (await PresentationFile.exportPptx(p)).save(out);
console.log(`Wrote ${out}`);




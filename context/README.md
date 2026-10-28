# ABR Project — Context Data Package

This is the raw data your simulator (step 2 of the workflow) will consume.
Nothing here is your code — it's the inputs your `env.py` reads.

## 1. `traces/train/` — 59 files, FCC-derived network traces

Two columns per line, whitespace-separated: `time_seconds  throughput_mbps`
Filenames look like `trace_<id>_http---<domain>` (this naming is the original
Pensieve convention for their cleaned FCC "Measuring Broadband America" data).

Example (`trace_10322_http---edition.cnn.com`):
```
0.0     0.890992
5.0     0.88812
10.0    0.852304
```
Meaning: at t=0s, throughput was 0.89 Mbps; at t=5s, 0.888 Mbps; etc.
This is what you train the PPO agent on.

## 2. `traces/test/` — 142 files, Norway HSDPA (3G) traces

Same two-column format, already cleaned from Norway's raw Simula HSDPA logs.
Filenames indicate the transport mode the trace was recorded on
(`norway_bus_1`, `norway_tram_14`, `norway_ferry_3`, etc.) — this diversity
is exactly what makes it a good out-of-distribution test set: different
mobility patterns than the fixed-broadband FCC traces you trained on.
This is your generalization-gap evaluation set (slide 5 / step 5 of the workflow).

Belgium LTE traces aren't bundled in this source — if you want a third
test distribution (as slide 5 mentions), we can source those separately
when we get to the evaluation step; Norway alone is already a legitimate
distribution shift from FCC.

## 3. `video_sizes/video_size_0` .. `video_size_5` — chunk byte-size tables

One integer per line = the byte size of that chunk at this bitrate level.
Line N = chunk N (1-indexed by position, 0-indexed by list position).

**Index direction, confirmed by inspection:**
`video_size_0` = smallest chunks (lowest bitrate), `video_size_5` = largest
chunks (highest bitrate) — e.g. chunk 1 is 181,801 bytes at level 0 vs.
2,354,772 bytes at level 5. Keep this direction consistent in your env —
it's easy to flip by accident and get an agent that "prefers" low quality
because your action indices are backwards.

This is the exact table your `step(action)` uses for
`download_time = chunk_size_bytes / current_throughput_mbps`.

## 4. `sample_video_chunks/video1` .. `video6` — 3 real chunks per bitrate

Actual encoded Big Buck Bunny segments (`.m4s`), 6 bitrate levels, first 3
chunks of each. `video1` = highest quality (~2.2MB/chunk), `video6` =
lowest (~150-180KB/chunk) — note this is the **opposite direction** from
the `video_size_*` files above (level 0 = lowest there, video1 = highest
here) — another naming trap to watch for when you wire these together.
These aren't needed until step 6 (the live video demo) — full chunk sets
(51 per level) can be pulled the same way once we get there.

## What's deliberately left out

Not including any of the original repos' `env.py` / training scripts here —
you should write the simulator and PPO training loop yourselves; this
package is data only. If you want to consult how an existing implementation
structures its Gym env for reference (not to copy), the repos are
`hongzimao/pensieve` (original, TensorFlow/A3C) and `godka/Pensieve-PPO`
(PyTorch/PPO) on GitHub — cite them in your report either way, since your
traces originate there.

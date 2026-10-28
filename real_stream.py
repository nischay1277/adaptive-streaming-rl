from __future__ import annotations

import argparse
import socketserver
import threading
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import requests

from env import ABREnv, BITRATES_KBPS, conservative_teacher
from ppo_agent import PPOAgent


class ThrottledHandler(SimpleHTTPRequestHandler):
    trace_mbps = [1.0]
    trace_idx = 0

    def copyfile(self, source, outputfile):
        while True:
            chunk = source.read(32 * 1024)
            if not chunk:
                break
            outputfile.write(chunk)
            outputfile.flush()
            mbps = max(float(self.trace_mbps[self.trace_idx % len(self.trace_mbps)]), 0.1)
            self.__class__.trace_idx += 1
            time.sleep((len(chunk) * 8.0) / (mbps * 1_000_000.0))


def _load_trace_for_server(env: ABREnv):
    return [max(float(x), 0.1) for x in env.trace[:200]]


def _find_chunk(root: Path, action: int, chunk_number: int):
    # The context package uses video1..video6. video6 is lowest bitrate, video1 highest.
    folder = root / f"video{6 - action}"
    preferred = folder / f"{chunk_number}.m4s"
    if preferred.exists():
        return preferred
    candidates = sorted(folder.glob("*.m4s"))
    return candidates[(chunk_number - 1) % len(candidates)] if candidates else None


def _draw_quality_preview(frame: np.ndarray, action: int, frame_no: int):
    x0, y0, w, h = 520, 70, 380, 250
    preview = np.zeros((h, w, 3), dtype=np.uint8)

    yy, xx = np.mgrid[0:h, 0:w]
    preview[..., 0] = (35 + xx // 3 + action * 16) % 255
    preview[..., 1] = (75 + yy // 2 + action * 22) % 255
    preview[..., 2] = (115 + (xx + yy) // 5 + frame_no * 4) % 255

    # Higher bitrate = more visible detail. Lower bitrate = blockier preview.
    for k in range(8 + action * 4):
        cx = int((k * 47 + frame_no * 13) % w)
        cy = int((k * 31 + frame_no * 7) % h)
        color = (230, 230 - action * 12, 80 + action * 22)
        cv2.circle(preview, (cx, cy), 4 + action, color, -1)

    block = [34, 26, 20, 14, 9, 5][action]
    small = cv2.resize(preview, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
    preview = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if action >= 3:
        preview = cv2.GaussianBlur(preview, (3, 3), 0)
        cv2.putText(preview, "cleaner high-quality chunk", (22, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    else:
        cv2.putText(preview, "blocky low-quality chunk", (22, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

    frame[y0 : y0 + h, x0 : x0 + w] = preview
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (220, 220, 220), 2)
    return x0, y0, w, h


def _draw_bitrate_ladder(frame: np.ndarray, action: int):
    x0, y0 = 48, 370
    labels = [300, 750, 1200, 1850, 2850, 4300]
    cv2.putText(frame, "Bitrate ladder", (x0, y0 - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
    for idx, bitrate in enumerate(labels):
        x = x0 + idx * 140
        active = idx == action
        color = (70, 230, 130) if active else (85, 95, 105)
        cv2.rectangle(frame, (x, y0), (x + 112, y0 + 44), color, -1)
        cv2.rectangle(frame, (x, y0), (x + 112, y0 + 44), (230, 230, 230), 2 if active else 1)
        cv2.putText(frame, f"{bitrate}", (x + 16, y0 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 20, 25), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="context")
    parser.add_argument("--checkpoint", default="artifacts_adaptive_ppo/ppo_adaptive.pt")
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--save", default="artifacts_adaptive_ppo/real_stream_demo.mp4")
    args = parser.parse_args()

    env = ABREnv(args.context, split="test", seed=9)
    obs, _ = env.reset()
    chunk_root = Path(args.context) / "sample_video_chunks"

    ckpt = Path(args.checkpoint)
    agent = PPOAgent.load(ckpt) if ckpt.exists() else None
    if agent is None:
        print("No PPO checkpoint found; using conservative teacher fallback for immediate demo.")

    handler = ThrottledHandler
    handler.trace_mbps = _load_trace_for_server(env)
    server_root = Path(args.context).resolve()

    class RootedHandler(handler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(server_root), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", args.port), RootedHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"HTTP trace-throttled server running at http://127.0.0.1:{args.port}")

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (960, 540))

    try:
        previous_action = None
        for i in range(1, args.chunks + 1):
            action = agent.act(obs, deterministic=True)[0] if agent else conservative_teacher(obs)
            local_chunk = _find_chunk(chunk_root, action, i)
            if local_chunk is None:
                print("No sample chunks found; stopping.")
                break
            rel = local_chunk.relative_to(Path(args.context)).as_posix()
            url = f"http://127.0.0.1:{args.port}/{rel}"
            t0 = time.time()
            data = requests.get(url, timeout=30).content
            elapsed = time.time() - t0
            obs, _, done, _, info = env.step(action)
            print(
                f"chunk={i:02d} bitrate={int(BITRATES_KBPS[action])}kbps "
                f"bytes={len(data)} transfer={elapsed:.2f}s buffer={info['buffer']:.1f}s"
            )
            switched = previous_action is not None and action != previous_action
            previous_bitrate = int(BITRATES_KBPS[previous_action]) if previous_action is not None else None
            current_bitrate = int(BITRATES_KBPS[action])
            for frame_no in range(16):
                frame = np.zeros((540, 960, 3), dtype=np.uint8)
                frame[:] = (25, 35, 42)
                cv2.putText(frame, "ABR PPO Real-Transfer Demo", (48, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (235, 235, 235), 2)
                cv2.putText(frame, f"Chunk {i:02d}/{args.chunks}", (48, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (235, 235, 235), 2)
                cv2.putText(frame, f"Trace throughput: {info['throughput_mbps']:.2f} Mbps", (48, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (120, 220, 255), 2)
                cv2.putText(frame, f"Selected bitrate: {current_bitrate} kbps", (48, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (120, 255, 160), 2)
                cv2.putText(frame, f"Buffer: {info['buffer']:.1f}s   Rebuffer: {info['rebuffer']:.2f}s", (48, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 210, 120), 2)
                cv2.putText(frame, f"HTTP bytes downloaded: {len(data):,} in {elapsed:.2f}s", (48, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2)

                _draw_quality_preview(frame, action, frame_no)
                _draw_bitrate_ladder(frame, action)

                if switched and frame_no < 10:
                    direction = "UP" if action > previous_action else "DOWN"
                    banner_color = (40, 180, 70) if action > previous_action else (40, 140, 255)
                    cv2.rectangle(frame, (0, 0), (960, 42), banner_color, -1)
                    cv2.putText(
                        frame,
                        f"BITRATE SWITCH {direction}: {previous_bitrate} -> {current_bitrate} kbps",
                        (150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (255, 255, 255),
                        2,
                    )
                writer.write(frame)
            previous_action = action
            if done:
                break
    finally:
        writer.release()
        httpd.shutdown()
    print(f"saved demo video: {out_path}")


if __name__ == "__main__":
    main()

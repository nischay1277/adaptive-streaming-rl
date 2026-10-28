from __future__ import annotations

import argparse
import email
import email.policy
import hashlib
import json
import math
import shutil
import subprocess
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from baselines import rate_based
from env import ABREnv, BITRATES_KBPS, CHUNK_DURATION
from ppo_agent import PPOAgent


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "demo"
SESSIONS: dict[str, dict] = {}
LOCK = threading.Lock()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print(fmt % args)

    def _send_json(self, status: int, data: dict):
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 2_000_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length))

    def translate_path(self, path):
        parts = [unquote(part) for part in urlsplit(path).path.split("/") if part]
        if parts and parts[0] == "media":
            safe = parts[1:]
            if any(part in (".", "..") or "\\" in part for part in safe):
                return str(ROOT / "__invalid_media_path__")
            return str((ROOT / "results" / "demo" / "media" / Path(*safe)).resolve())
        return super().translate_path(path)

    def _upload_video(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 500_000_000:
            raise ValueError("video upload must be between 1 byte and 500 MB")
        raw = self.rfile.read(length)
        headers = (f"Content-Type: {self.headers.get('Content-Type', '')}\r\n"
                   "MIME-Version: 1.0\r\n\r\n").encode("ascii", "replace")
        message = email.message_from_bytes(headers + raw, policy=email.policy.default)
        video_part = next((part for part in message.iter_parts()
                           if part.get_content_disposition() == "form-data"
                           and part.get_filename()), None)
        if video_part is None:
            raise ValueError("video file field missing")
        filename = Path(video_part.get_filename()).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".mp4", ".mov", ".m4v", ".webm", ".mkv"}:
            raise ValueError("upload an MP4, MOV, WebM, M4V, or MKV file")
        video_id = uuid.uuid4().hex
        media_root = ROOT / "results" / "demo" / "media" / video_id
        media_root.mkdir(parents=True, exist_ok=False)
        source = media_root / f"source{suffix}"
        source.write_bytes(video_part.get_payload(decode=True) or b"")
        (media_root / "original_filename.txt").write_text(filename)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg = get_ffmpeg_exe()
            rates = [300, 750, 1200, 1850, 2850, 4300]
            heights = [240, 360, 480, 720, 900, 1080]
            segment_counts = []
            for index, (rate, height) in enumerate(zip(rates, heights)):
                output_dir = media_root / f"q{index}"
                output_dir.mkdir()
                output_pattern = str(output_dir / "chunk%04d.mp4")
                cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                       "-map", "0:v:0", "-an", "-vf",
                       f"scale=-2:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                       "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                       "-b:v", f"{rate}k", "-maxrate", f"{rate}k", "-bufsize", f"{rate * 2}k",
                       "-g", "120", "-keyint_min", "120", "-sc_threshold", "0",
                       "-force_key_frames", "expr:gte(t,n_forced*4)",
                       "-f", "segment", "-segment_time", "4", "-segment_time_delta", "0.05",
                       "-reset_timestamps", "1", "-segment_format", "mp4",
                       "-segment_format_options", "movflags=+faststart", output_pattern]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-3000:] or f"ffmpeg failed at {rate} kbps")
                segment_counts.append(len(list(output_dir.glob("chunk*.mp4"))))
            chunks = min(segment_counts)
            if chunks < 1:
                raise ValueError("FFmpeg produced no playable video segments")
            with LOCK:
                self.server.media[video_id] = {"chunks": chunks, "rates": rates, "filename": filename}
            self._send_json(200, {"media_id": video_id, "chunks": chunks, "rates_kbps": rates,
                                  "segments": "H.264 MP4, aligned 4-second segments"})
        except Exception:
            shutil.rmtree(media_root, ignore_errors=True)
            raise

    def do_GET(self):
        if self.path == "/api/media":
            saved = sorted(self.server.media.items(),
                           key=lambda item: (ROOT / "results" / "demo" / "media" / item[0]).stat().st_mtime,
                           reverse=True)
            entries = []
            for media_id, data in saved:
                item = dict(data)
                name_path = ROOT / "results" / "demo" / "media" / media_id / "original_filename.txt"
                if name_path.exists():
                    item["filename"] = name_path.read_text()
                entries.append({"id": media_id, **item})
            self._send_json(200, {"media": entries})
            return
        if self.path == "/api/default-trace":
            values = []
            for token in (ROOT / "sample_stress_trace.txt").read_text().replace(",", " ").split():
                try:
                    value = float(token)
                    if math.isfinite(value) and value > 0:
                        values.append(value)
                except ValueError:
                    continue
            self._send_json(200, {"throughput_kbps": values, "name": "sample_stress_trace.txt"})
            return
        if self.path == "/api/info":
            checkpoint = Path(self.server.checkpoint)
            self._send_json(200, {
                "loaded": self.server.agent is not None,
                "policy": "PPO with buffer-safety guard",
                "checkpoint": str(checkpoint),
                "sha256": self.server.checkpoint_hash,
                "action_rates_kbps": [int(x) for x in BITRATES_KBPS],
                "chunk_duration_s": CHUNK_DURATION,
            })
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        try:
            if self.path == "/api/video":
                self._upload_video()
                return
            payload = self._body()
            if self.path == "/api/session":
                rates = payload.get("throughput_kbps")
                if not isinstance(rates, list) or not 1 <= len(rates) <= 2000:
                    raise ValueError("trace must contain 1 to 2000 throughput values")
                rates = [float(x) for x in rates]
                if any(not math.isfinite(x) or x <= 0 or x > 100_000 for x in rates):
                    raise ValueError("trace values must be finite throughput values in kbps")
                sid = uuid.uuid4().hex
                media_id = str(payload.get("media_id", ""))
                media = self.server.media.get(media_id)
                if media is None:
                    raise ValueError("upload a video and wait for rendition encoding first")
                session_dir = ROOT / "results" / "demo" / "sessions"
                session_dir.mkdir(parents=True, exist_ok=True)
                trace_path = session_dir / f"{sid}.trace"
                trace_path.write_text("".join(f"{i} {rate / 1000.0:.8f}\n" for i, rate in enumerate(rates)))
                try:
                    baseline = ABREnv(ROOT / "context", split="train", seed=101,
                                      trace_paths=[trace_path])
                    ppo = ABREnv(ROOT / "context", split="train", seed=101,
                                 trace_paths=[trace_path])
                    size_matrix = []
                    for quality_index in range(6):
                        files = sorted((ROOT / "results" / "demo" / "media" / media_id /
                                        f"q{quality_index}").glob("chunk*.mp4"))[:media["chunks"]]
                        if len(files) != media["chunks"]:
                            raise ValueError("encoded rendition segments are incomplete")
                        size_matrix.append([p.stat().st_size for p in files])
                    measured_sizes = np.asarray(size_matrix, dtype=np.float32)
                    for player in (baseline, ppo):
                        player.chunk_sizes = measured_sizes.copy()
                        player.num_chunks = measured_sizes.shape[1]
                    baseline.reset(seed=101)
                    ppo.reset(seed=101)
                except Exception:
                    trace_path.unlink(missing_ok=True)
                    raise
                with LOCK:
                    SESSIONS[sid] = {"baseline": baseline, "ppo": ppo, "trace": rates,
                                     "trace_path": trace_path, "switches": {"baseline": 0, "ppo": 0},
                                     "last_action": {"baseline": None, "ppo": None}, "media_id": media_id}
                self._send_json(200, {"session": sid, "trace_samples": len(rates),
                                      "video_chunks": media["chunks"],
                                      "checkpoint_sha256": self.server.checkpoint_hash})
                return
            if self.path == "/api/step":
                sid = str(payload.get("session", ""))
                with LOCK:
                    session = SESSIONS.get(sid)
                    if session is None:
                        raise ValueError("simulation session not found; reset and start again")
                    result = {}
                    requested = payload.get("policy")
                    if requested not in (None, "baseline", "ppo"):
                        raise ValueError("policy must be baseline or ppo")
                    players = (("baseline", session["baseline"]), ("ppo", session["ppo"]))
                    if requested:
                        players = tuple(item for item in players if item[0] == requested)
                    shared_trace_index = None
                    for name, env in players:
                        # Each independent player replays the same trace from sample 0.
                        # Its current chunk number selects the sample so it never waits
                        # for the other player's download or playback.
                        shared_trace_index = env.chunk_idx % len(session["trace"])
                        env.trace_idx = shared_trace_index
                        obs = env._obs()
                        if name == "baseline":
                            raw_action = action = rate_based(obs)
                            guard_applied = False
                            guard_throughput_kbps = None
                        else:
                            raw_action = self.server.agent.act(obs, deterministic=True)[0]
                            # Validation-tuned shield: cap PPO to a rendition whose
                            # estimated transfer fits in the current buffer + 2s.
                            recent_kbps = obs[2:6].astype(np.float64) * 6000.0
                            guard_throughput_kbps = max(50.0, float(np.min(recent_kbps)) * 0.60)
                            deadline_s = env.buffer + 2.0
                            feasible = [i for i in range(raw_action + 1)
                                        if float(env.chunk_sizes[i, env.chunk_idx]) * 8.0 /
                                        (guard_throughput_kbps * 1000.0) <= deadline_s]
                            action = max(feasible) if feasible else 0
                            guard_applied = action < raw_action
                        _, reward, done, _, info = env.step(action)
                        previous = session["last_action"][name]
                        if previous is not None and previous != action:
                            session["switches"][name] += 1
                        session["last_action"][name] = action
                        result[name] = {**info, "action": action, "raw_action": raw_action,
                                        "safety_guard_applied": guard_applied,
                                        "guard_throughput_kbps": guard_throughput_kbps,
                                        "reward": float(reward),
                                        "switches": session["switches"][name],
                                        "reward_total": env.total_reward,
                                        "rebuffer_total": env.total_rebuffer,
                                        "done": bool(done)}
                    chunk = (session[requested].chunk_idx if requested else
                             min(session["baseline"].chunk_idx, session["ppo"].chunk_idx))
                    self._send_json(200, {"chunk": chunk, "trace_index": shared_trace_index,
                                          "throughput_trace_kbps": session["trace"], "policies": result})
                return
            if self.path == "/api/reset":
                sid = str(payload.get("session", ""))
                with LOCK:
                    session = SESSIONS.pop(sid, None)
                if session:
                    session["trace_path"].unlink(missing_ok=True)
                self._send_json(200, {"reset": True})
                return
            self._send_json(404, {"error": "not found"})
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Run the real PPO and rate-based ABR browser demo")
    parser.add_argument("--checkpoint", default="results/demo/ppo_adaptive.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise SystemExit(f"PPO checkpoint missing: {checkpoint}. Run train_ppo.py first.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.agent = PPOAgent.load(checkpoint)
    server.checkpoint = checkpoint
    server.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    server.media = {}
    media_root = ROOT / "results" / "demo" / "media"
    if media_root.exists():
        for media_dir in media_root.iterdir():
            if not media_dir.is_dir():
                continue
            source = next(media_dir.glob("source.*"), None)
            segment_counts = [len(list((media_dir / f"q{i}").glob("chunk*.mp4")))
                              for i in range(6)]
            chunks = min(segment_counts, default=0)
            if source and chunks:
                server.media[media_dir.name] = {"chunks": chunks,
                    "rates": [300, 750, 1200, 1850, 2850, 4300],
                    "filename": ((media_dir / "original_filename.txt").read_text()
                                 if (media_dir / "original_filename.txt").exists() else source.name)}
    print(f"ABR comparison demo: http://{args.host}:{args.port}")
    print(f"Loaded PPO checkpoint: {checkpoint}")
    print(f"Checkpoint SHA-256: {server.checkpoint_hash}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        for session in SESSIONS.values():
            session["trace_path"].unlink(missing_ok=True)


if __name__ == "__main__":
    main()

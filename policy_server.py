from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ppo_agent import PPOAgent


class PolicyHandler(BaseHTTPRequestHandler):
    agent: PPOAgent | None = None
    checkpoint: Path | None = None
    checkpoint_hash: str = ""

    def log_message(self, fmt, *args):
        print(fmt % args)

    def _send(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path in ("/", "/info"):
            self._send(200, {
                "loaded": self.agent is not None,
                "policy": "PPO",
                "checkpoint": str(self.checkpoint) if self.checkpoint else None,
                "sha256": self.checkpoint_hash,
                "observation_dim": 25,
                "action_count": 6,
                "deterministic": True,
            })
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/decide":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            obs = np.asarray(payload["observation"], dtype=np.float32)
            if obs.shape != (25,):
                raise ValueError(f"observation must contain 25 values, got {obs.shape}")
            if not np.isfinite(obs).all():
                raise ValueError("observation contains non-finite values")
            action = self.agent.act(obs, deterministic=True)[0]
            self._send(200, {
                "action": int(action),
                "bitrate_kbps": [300, 750, 1200, 1850, 2850, 4300][action],
                "policy": "PPO",
                "checkpoint_sha256": self.checkpoint_hash,
            })
        except Exception as exc:
            self._send(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description="Serve deterministic PPO ABR actions")
    parser.add_argument("--checkpoint", default="artifacts_adaptive_ppo/ppo_adaptive.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    PolicyHandler.agent = PPOAgent.load(checkpoint)
    PolicyHandler.checkpoint = checkpoint
    PolicyHandler.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    server = ThreadingHTTPServer((args.host, args.port), PolicyHandler)
    print(f"PPO policy server: http://{args.host}:{args.port}")
    print(f"loaded checkpoint: {checkpoint}")
    print(f"sha256: {PolicyHandler.checkpoint_hash}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()


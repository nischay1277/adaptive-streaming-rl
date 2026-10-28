"""
Trace-throttled static file server for the live ABR demo.

Serves files under --root, limiting delivery speed to the bandwidth given by a
network trace (looped). Run one instance per player so both players replay the
SAME trace; call GET /reset on each at the same moment to align trace time.

  python throttle_server.py --root renditions --trace sample_stress_trace.txt --port 8101
  python throttle_server.py --root renditions --trace sample_stress_trace.txt --port 8102

Trace format (flexible): one value per line (step = --step seconds), or
"time value" / "time,value" per line. --unit says what the value means:
kbps (default), mbps, or kBps. Check this against your real trace files and
against how real_stream.py interprets them.

Endpoints: any file path (throttled), /reset (restart trace clock),
/state (JSON: trace time and current bandwidth).
"""
import argparse
import json
import os
import re
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

CHUNK = 4096


def load_trace(path, unit, step):
    times, vals = [], []
    t = 0.0
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = [p for p in re.split(r"[\s,;]+", line) if p]
            try:
                nums = [float(p) for p in parts]
            except ValueError:
                continue  # header line
            if len(nums) >= 2:
                times.append(nums[0])
                v = nums[1]
            else:
                times.append(t)
                v = nums[0]
                t += step
            vals.append(v)
    if not vals:
        raise SystemExit(f"No numeric data in trace {path}")
    scale = {"kbps": 1000 / 8, "mbps": 1e6 / 8, "kBps": 1000.0}[unit]
    return times, [max(v * scale, 1000.0) for v in vals]  # bytes/sec, floor 1 kB/s


class State:
    def __init__(self, times, bps):
        self.times, self.bps = times, bps
        self.span = times[-1] + (times[-1] - times[-2] if len(times) > 1 else 1.0)
        self.t0 = time.time()
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.t0 = time.time()

    def now(self):
        return (time.time() - self.t0) % self.span

    def bandwidth(self):  # bytes/sec at current trace time
        t = self.now()
        lo = 0
        for i, ti in enumerate(self.times):
            if ti <= t:
                lo = i
            else:
                break
        return self.bps[lo]


def make_handler(root, state):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=root, **k)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/reset":
                state.reset()
                return self._json({"ok": True})
            if path == "/state":
                return self._json({"trace_time": state.now(), "bytes_per_sec": state.bandwidth()})
            return super().do_GET()

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def copyfile(self, source, outputfile):  # throttled body
            try:
                while True:
                    buf = source.read(CHUNK)
                    if not buf:
                        break
                    outputfile.write(buf)
                    outputfile.flush()
                    time.sleep(len(buf) / state.bandwidth())
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--unit", default="kbps", choices=["kbps", "mbps", "kBps"])
    ap.add_argument("--step", type=float, default=1.0)
    a = ap.parse_args()
    times, bps = load_trace(a.trace, a.unit, a.step)
    state = State(times, bps)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(os.path.abspath(a.root), state))
    print(f"Throttled server on :{a.port}, {len(bps)} trace points, loops every {state.span:.0f}s")
    srv.serve_forever()


if __name__ == "__main__":
    main()

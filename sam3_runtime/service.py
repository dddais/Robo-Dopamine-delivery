"""SAM3 image grounding service; run in the existing rewardbench-sam3 environment.

Uses Python's HTTP server so no FastAPI/uvicorn installation is needed there.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


from .detector import SAM3Detector


def make_server(host, port, detector, tracker=None):
    semaphore = threading.BoundedSemaphore(1)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value, code=200):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path != "/health":
                return self.send_json({"error": "not found"}, 404)
            self.send_json({"status": "ready", "model_fingerprint": detector.fingerprint,
                "tracking_enabled": tracker is not None,
                "tracking_identity_policy": tracker.identity_policy if tracker else None,
                "tracker_fingerprint": tracker.fingerprint if tracker else None})

        def do_POST(self):
            if self.path not in {"/grounding/detect", "/tracking/update", "/tracking/close"}:
                return self.send_json({"error": "not found"}, 404)
            if not semaphore.acquire(blocking=False):
                return self.send_json({"error": "SAM3 busy; retry later"}, 503)
            try:
                self.connection.settimeout(15)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024 * 1024:
                    raise ValueError("Request must be between 1 byte and 20 MiB")
                request = json.loads(self.rfile.read(size))
                if self.path.startswith('/tracking/'):
                    if tracker is None:
                        return self.send_json({"error": "Tracking is disabled in SAM3 config"}, 400)
                    session_id = request.get('session_id')
                    if not isinstance(session_id, str) or not 1 <= len(session_id) <= 200:
                        raise ValueError('Invalid tracking session_id')
                    if self.path == '/tracking/close':
                        tracker.close(session_id)
                        return self.send_json({'closed': True, 'session_id': session_id})
                data = base64.b64decode(request["image_png_base64"], validate=True)
                sha = hashlib.sha256(data).hexdigest()
                if sha != request["image_sha256"]:
                    raise ValueError("Input image hash mismatch")
                queries = request["queries"]
                if not isinstance(queries, list) or not 1 <= len(queries) <= 8 or any(not isinstance(q,str) or not q.strip() for q in queries):
                    raise ValueError("queries must contain 1..8 nonempty strings")
                from PIL import Image
                with Image.open(io.BytesIO(data)) as im:
                    image = im.convert("RGB")
                started = time.monotonic()
                if self.path == '/tracking/update':
                    result = tracker.update(session_id, image, queries, sha,
                        initialize=request.get('initialize', False))
                    return self.send_json({**result, "request_id": request["request_id"],
                        "session_id": session_id, "image_sha256": sha, "image_size": list(image.size),
                        "coordinate_space": "input_image_xyxy", "model_fingerprint": tracker.fingerprint})
                rows = detector.detect(image, queries)
                self.send_json({"request_id": request["request_id"], "image_sha256": sha,
                    "image_size": list(image.size), "coordinate_space": "input_image_xyxy",
                    "model_fingerprint": detector.fingerprint, "status": "ok" if rows else "no_detection",
                    "candidates": rows, "latency_ms": (time.monotonic()-started)*1000,
                    "timing": getattr(detector, 'last_timing', {})})
            except (ValueError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            finally:
                semaphore.release()

    class Server(ThreadingHTTPServer):
        def service_actions(self):
            if tracker is not None and semaphore.acquire(blocking=False):
                try:
                    tracker.expire()
                finally:
                    semaphore.release()

    return Server((host, port), Handler)


def main():
    from grm_runtime.common import load_yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg, base = load_yaml(args.config, {"model_path","device","threshold","mask_threshold","host","port",
                                      "dtype", "bbox_only", "profile", "num_threads", "tracking"})
    host, port = cfg.pop("host", "127.0.0.1"), cfg.pop("port", 8878)
    from grm_runtime.common import resolve_path
    cfg["model_path"] = resolve_path(cfg["model_path"], base)
    tracking = cfg.pop('tracking', {})
    if not isinstance(tracking, dict) or not isinstance(tracking.get('enabled', False), bool):
        raise ValueError('tracking must be a mapping with boolean enabled')
    enabled = tracking.pop('enabled', False)
    detector = SAM3Detector(**cfg)
    engine = None
    if enabled:
        from .tracker import SAM3VideoTracker, TrackingEngine
        video = SAM3VideoTracker(cfg['model_path'], device=tracking.pop('device', cfg.get('device', 'cuda:0')),
            dtype=tracking.pop('dtype', 'bfloat16'), memory_frames=tracking.pop('memory_frames', 32))
        engine = TrackingEngine(detector, video, **tracking)
    elif tracking:
        raise ValueError('Remove tracking options or set tracking.enabled: true')
    server = make_server(host, port, detector, engine)
    print(f"SAM3 ready at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

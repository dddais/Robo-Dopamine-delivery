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


class SAM3Detector:
    def __init__(self, model_path, device="cuda:0", threshold=0.3, mask_threshold=0.5):
        import torch
        from transformers import Sam3Model, Sam3Processor
        self.torch, self.device = torch, device
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path).to(device).eval()
        self.threshold, self.mask_threshold = float(threshold), float(mask_threshold)
        if not 0 <= self.threshold <= 1 or not 0 <= self.mask_threshold <= 1:
            raise ValueError("SAM3 thresholds must be in [0,1]")
        metadata = {"model_path": str(Path(model_path).resolve()), "config": self.model.config.to_dict(),
                    "threshold": threshold, "mask_threshold": mask_threshold,
                    "weights": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                for p in sorted(Path(model_path).glob('*.safetensors'))]}
        self.fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def detect(self, image, queries):
        rows = []
        for query in queries:
            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(outputs,
                threshold=self.threshold, mask_threshold=self.mask_threshold,
                target_sizes=inputs["original_sizes"].tolist())[0]
            for index, score in enumerate(result["scores"]):
                if result.get("boxes") is not None:
                    box = result["boxes"][index].detach().cpu().tolist()
                else:
                    import numpy as np
                    mask = result["masks"][index].detach().cpu().numpy().squeeze()
                    ys, xs = np.nonzero(mask)
                    if not len(xs):
                        continue
                    box = [float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1)]
                from grm_runtime.grounding import validate_bbox
                try:
                    box = validate_bbox(box, image.size)
                except ValueError:
                    continue
                # Collapse duplicate instances returned by synonymous queries.
                duplicate = False
                for previous in rows:
                    b = previous["bbox"]
                    area = max(0, min(b[2],box[2])-max(b[0],box[0])) * max(0,min(b[3],box[3])-max(b[1],box[1]))
                    union = (b[2]-b[0])*(b[3]-b[1])+(box[2]-box[0])*(box[3]-box[1])-area
                    if union and area/union >= 0.8:
                        duplicate = True
                        break
                if not duplicate:
                    rows.append({"bbox": box, "score": float(score.detach().cpu()), "query": query})
        return sorted(rows, key=lambda row: -row["score"])


def make_server(host, port, detector):
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
            self.send_json({"status": "ready", "model_fingerprint": detector.fingerprint})

        def do_POST(self):
            if self.path != "/grounding/detect":
                return self.send_json({"error": "not found"}, 404)
            if not semaphore.acquire(blocking=False):
                return self.send_json({"error": "SAM3 busy; retry later"}, 503)
            try:
                self.connection.settimeout(15)
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024 * 1024:
                    raise ValueError("Request must be between 1 byte and 20 MiB")
                request = json.loads(self.rfile.read(size))
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
                rows = detector.detect(image, queries)
                self.send_json({"request_id": request["request_id"], "image_sha256": sha,
                    "image_size": list(image.size), "coordinate_space": "input_image_xyxy",
                    "model_fingerprint": detector.fingerprint, "status": "ok" if rows else "no_detection",
                    "candidates": rows, "latency_ms": (time.monotonic()-started)*1000})
            except (ValueError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)
            finally:
                semaphore.release()

    return ThreadingHTTPServer((host, port), Handler)


def main():
    from grm_runtime.common import load_yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg, base = load_yaml(args.config, {"model_path","device","threshold","mask_threshold","host","port"})
    host, port = cfg.pop("host", "127.0.0.1"), cfg.pop("port", 8878)
    from grm_runtime.common import resolve_path
    cfg["model_path"] = resolve_path(cfg["model_path"], base)
    server = make_server(host, port, SAM3Detector(**cfg))
    print(f"SAM3 ready at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

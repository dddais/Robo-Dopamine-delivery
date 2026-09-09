"""HTTP grounding on exact PNG bytes; caches are keyed by model and image content."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import threading
import urllib.request
from uuid import uuid4
from pathlib import Path

from PIL import Image
from .common import fingerprint


class GroundingError(RuntimeError):
    pass


class AlignmentError(ValueError):
    pass


def validate_bbox(box, size):
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(not math.isfinite(float(x)) for x in box):
        raise AlignmentError("bbox must contain four finite coordinates")
    w, h = size
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise AlignmentError("Empty or reversed bbox")
    clipped = [max(0., min(w, x1)), max(0., min(h, y1)), max(0., min(w, x2)), max(0., min(h, y2))]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise AlignmentError("bbox outside image")
    return clipped


def png_bytes(path):
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        # Online snapshots are already RGB PNGs. Send their exact bytes instead
        # of decoding and recompressing a full camera frame on every request.
        if im.format == "PNG" and im.mode == "RGB" and not getattr(im, "is_animated", False):
            return data, im.size
        image = im.convert("RGB")
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue(), image.size


class GroundingClient:
    def __init__(self, url="http://127.0.0.1:8878", timeout_s=3.0, cache_dir=None):
        if not url.startswith(("http://", "https://")) or timeout_s <= 0:
            raise ValueError("Grounding URL must include http(s) and timeout must be positive")
        self.url, self.timeout = url.rstrip("/"), timeout_s
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache = {}
        self._lock = threading.Lock()

    def _request(self, route, payload=None):
        request = urllib.request.Request(self.url + route,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except Exception as exc:
            raise GroundingError(f"SAM3 request failed: {exc}") from exc

    def detect(self, path, queries):
        data, size = png_bytes(path)
        sha = hashlib.sha256(data).hexdigest()
        # Recheck fingerprint before reading a persistent cache, including after a server restart.
        health = self._request("/health")
        model_id = health["model_fingerprint"]
        key = fingerprint({"image": sha, "queries": queries, "model": model_id})
        cache_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        with self._lock:
            cached = self._cache.get(key)
            if cached is None and cache_path and cache_path.exists():
                cached = json.loads(cache_path.read_text())
        if cached is None:
            cached = self._request("/grounding/detect", {
                "request_id": key, "image_sha256": sha, "queries": queries,
                "image_png_base64": base64.b64encode(data).decode()})
        if (cached.get("image_sha256") != sha or cached.get("image_size") != list(size)
                or cached.get("request_id") != key or cached.get("model_fingerprint") != model_id
                or cached.get("coordinate_space") != "input_image_xyxy"):
            raise AlignmentError("SAM3 response does not match input image/model/request")
        for row in cached.get("candidates", []):
            row["bbox"] = validate_bbox(row["bbox"], size)
            if not math.isfinite(float(row["score"])) or not 0 <= float(row["score"]) <= 1 or row.get("query") not in queries:
                raise AlignmentError("Invalid SAM3 candidate score/query")
        if cached.get("status") not in {"ok", "no_detection"}:
            raise GroundingError(f"SAM3 detection failed: {cached}")
        with self._lock:
            # Bound memory. Disk cache is optional and intended for offline runs.
            if len(self._cache) >= 512:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = cached
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix('.tmp')
                tmp.write_text(json.dumps(cached))
                tmp.replace(cache_path)
        rows = sorted(cached.get("candidates", []), key=lambda r: -r["score"])
        # Multiple same-query objects with comparable confidence are ambiguous.
        selected = rows[0] if rows else None
        ambiguous = len(rows) > 1 and rows[1]["score"] >= rows[0]["score"] - 0.05
        return {**cached, "selected": None if ambiguous else selected,
                "selection_status": "ambiguous" if ambiguous else "ok" if selected else "no_detection"}

    def track(self, path, queries, session_id):
        # Tracking is history-dependent. Never use the detector's content cache.
        data, size = png_bytes(path)
        sha, request_id = hashlib.sha256(data).hexdigest(), uuid4().hex
        result = self._request('/tracking/update', {'session_id': session_id, 'request_id': request_id,
            'image_sha256': sha, 'queries': queries, 'image_png_base64': base64.b64encode(data).decode()})
        if result.get('session_id') != session_id or result.get('request_id') != request_id:
            raise AlignmentError('Tracker response belongs to a different session/request')
        validate_grounding_result(result, sha, size, queries)
        return result

    def close_track(self, session_id):
        return self._request('/tracking/close', {'session_id': session_id})


def validate_grounding_result(result, sha, size, queries):
    """Validate an asynchronous result against the exact frozen GRM input."""
    if (result.get('image_sha256') != sha or result.get('image_size') != list(size)
            or result.get('coordinate_space') != 'input_image_xyxy'):
        raise AlignmentError('Grounding result does not match the frozen input image')
    if result.get('status') not in {'ok', 'no_detection'}:
        raise GroundingError('Grounding result failed')
    for row in result.get('candidates', []):
        validate_bbox(row.get('bbox'), size)
        score = row.get('score')
        if (not isinstance(score, (int, float)) or not math.isfinite(score)
                or not 0 <= score <= 1 or row.get('query') not in queries):
            raise AlignmentError('Invalid tracked candidate score/query')
    selected = result.get('selected')
    if selected is not None:
        if result.get('selection_status') != 'ok' or selected not in result.get('candidates', []):
            raise AlignmentError('Invalid tracked selection')
    elif result.get('selection_status') not in {'no_detection', 'ambiguous', 'tracking_error'}:
        raise AlignmentError('Missing tracked selection status')

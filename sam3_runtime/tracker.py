"""Text detection followed by bounded, per-task/per-camera SAM3 instance tracking."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from types import MethodType

from grm_runtime.common import fingerprint


def box_iou(a, b):
    area = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.


def normalize_tracker_feature_names(model):
    """Bridge the standalone tracker's upstream FPN field-name mismatch.

    Some Transformers builds return fpn_position_encoding but their tracker
    reads fpn_position_embeddings. Alias the *processed* features on this
    instance only, without modifying site-packages or tensor computation.
    """
    original = model.get_image_features
    def features(self, *args, **kwargs):
        output = original(*args, **kwargs)
        if not hasattr(output, 'fpn_position_embeddings') and hasattr(output, 'fpn_position_encoding'):
            output.fpn_position_embeddings = output.fpn_position_encoding
        return output
    model.get_image_features = MethodType(features, model)


class SAM3VideoTracker:
    def __init__(self, model_path, device="cuda:0", dtype="bfloat16", memory_frames=32):
        import torch
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Invalid tracker dtype")
        self.torch, self.device, self.dtype = torch, device, getattr(torch, dtype)
        self.model = Sam3TrackerVideoModel.from_pretrained(model_path, dtype=self.dtype).to(device).eval()
        normalize_tracker_feature_names(self.model)
        self.processor = Sam3TrackerVideoProcessor.from_pretrained(model_path)
        minimum = max(self.model.config.num_maskmem, self.model.config.max_object_pointers_in_encoder)
        if isinstance(memory_frames, bool) or not isinstance(memory_frames, int) or memory_frames < minimum:
            raise ValueError(f"tracking.memory_frames must be >= {minimum}")
        self.memory_frames = memory_frames

    def initialize(self, image, box):
        session = self.processor.init_video_session(inference_device=self.device,
            inference_state_device=self.device, video_storage_device=self.device, dtype=self.dtype)
        self.processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1,
            input_boxes=[[box]], original_size=(image.height, image.width))
        self.step(session, image, 0)
        return session

    def step(self, session, image, index):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(inference_session=session, frame_idx=index,
                                 frame=inputs.pixel_values[0].to(self.dtype))
            masks = self.processor.post_process_masks([outputs.pred_masks],
                original_sizes=[[image.height, image.width]], binarize=False)[0]
            mask = masks.reshape(-1, image.height, image.width)[0] > 0
            ys = self.torch.where(mask.any(dim=1))[0]
            xs = self.torch.where(mask.any(dim=0))[0]
            score = outputs.object_score_logits.reshape(-1)[0].sigmoid()
            row = None
            if xs.numel() and ys.numel():
                values = self.torch.stack([xs[0].float(), ys[0].float(),
                    (xs[-1]+1).float(), (ys[-1]+1).float(), score.float()]).cpu().tolist()
                row = {"bbox": values[:4], "score": values[4]}
        # Upstream streaming sessions retain all frames/outputs by default.
        # Keep the prompt frame and enough recent memory/pointers for inference.
        cutoff = index - self.memory_frames + 1
        mappings = [session.processed_frames]
        for outputs in session.output_dict_per_obj.values():
            mappings.append(outputs['non_cond_frame_outputs'])
        mappings.extend(session.frames_tracked_per_obj.values())
        for mapping in mappings:
            if mapping is not None:
                for key in list(mapping):
                    if key != 0 and key < cutoff:
                        del mapping[key]
        return row


@dataclass
class Track:
    queries: tuple
    size: tuple
    seen_at: float
    detected_at: float = 0.
    session: object = None
    index: int = 0
    last_sha: str | None = None
    last_result: dict | None = None


class TrackingEngine:
    """Called under the service GPU lock. Results are never cached across sessions."""
    def __init__(self, detector, tracker, *, redetect_interval_s=5., max_gap_s=2.,
                 min_score=.5, match_iou=.1, session_ttl_s=60., max_sessions=8):
        values = (redetect_interval_s, max_gap_s, session_ttl_s, min_score, match_iou)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Invalid tracking thresholds")
        if min(redetect_interval_s, max_gap_s, session_ttl_s) <= 0 or not 0 <= min_score <= 1 or not 0 <= match_iou <= 1:
            raise ValueError("Invalid tracking thresholds")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.detector, self.tracker = detector, tracker
        self.redetect_interval_s, self.max_gap_s = redetect_interval_s, max_gap_s
        self.min_score, self.match_iou = min_score, match_iou
        self.session_ttl_s, self.max_sessions = session_ttl_s, max_sessions
        self.sessions = {}
        self.fingerprint = fingerprint({"detector": detector.fingerprint, "kind": "sam3_tracker_video_v1",
            "dtype": str(getattr(tracker, 'dtype', None)), "redetect_interval_s": redetect_interval_s,
            "max_gap_s": max_gap_s, "min_score": min_score, "match_iou": match_iou})

    def close(self, session_id):
        self.sessions.pop(session_id, None)

    def expire(self):
        now = time.monotonic()
        for key, state in list(self.sessions.items()):
            if now-state.seen_at > self.session_ttl_s:
                del self.sessions[key]

    def update(self, session_id, image, queries, sha):
        self.expire()
        now, started = time.monotonic(), time.monotonic()
        state = self.sessions.get(session_id)
        if state is not None and (state.queries != tuple(queries) or state.size != image.size):
            raise ValueError("Tracking session target/size changed; start a new session")
        if state is None:
            if len(self.sessions) >= self.max_sessions:
                raise ValueError("Too many tracking sessions; close inactive sessions")
            state = Track(tuple(queries), image.size, now)
            self.sessions[session_id] = state
        gap = now-state.seen_at
        state.seen_at = now
        if gap <= self.max_gap_s and state.last_sha == sha and state.last_result is not None:
            return deepcopy(state.last_result)
        if gap > self.max_gap_s:
            state.session = None
        row, source, reason = None, 'sam3_tracker', None
        timing = {"tracker_ms": 0., "detect_ms": 0.}
        try:
            if state.session is not None:
                tick = time.monotonic()
                state.index += 1
                row = self.tracker.step(state.session, image, state.index)
                timing['tracker_ms'] += (time.monotonic()-tick)*1000
                if row is not None and row['score'] >= self.min_score:
                    row = {**row, 'query': state.last_result['selected']['query']}
                else:
                    row, state.session = None, None
            if state.session is None or now-state.detected_at >= self.redetect_interval_s:
                source = 'sam3_detection'
                tick = time.monotonic()
                candidates = self.detector.detect(image, queries)
                timing['detect_ms'] = (time.monotonic()-tick)*1000
                timing['detector'] = deepcopy(getattr(self.detector, 'last_timing', {}))
                if row is not None:
                    candidates = [c for c in candidates if box_iou(c['bbox'], row['bbox']) >= self.match_iou]
                    candidates.sort(key=lambda c: box_iou(c['bbox'], row['bbox']), reverse=True)
                else:
                    candidates = sorted(candidates, key=lambda c: -c['score'])
                # A tracker does not resolve ambiguous first-frame detections.
                ambiguous = row is None and len(candidates) > 1 and candidates[1]['score'] >= candidates[0]['score']-.05
                if candidates and not ambiguous:
                    row = candidates[0]
                    tick = time.monotonic()
                    state.session = self.tracker.initialize(image, row['bbox'])
                    timing['tracker_ms'] += (time.monotonic()-tick)*1000
                    state.index, state.detected_at = 0, now
                    source = 'sam3_detection'
                else:
                    row, state.session = None, None
                    reason = 'ambiguous' if ambiguous else 'no_detection'
            result = {"candidates": [row] if row else [], "selected": row,
                "status": "ok" if row else "no_detection", "selection_status": "ok" if row else reason,
                "source": source, "score_type": "object_presence" if source == 'sam3_tracker' else "detection",
                "tracker_frame_index": state.index, "timing": timing,
                "latency_ms": (time.monotonic()-started)*1000}
            state.last_sha, state.last_result = sha, deepcopy(result)
            return result
        except Exception:
            # An HTTP retry must never propagate from partially advanced GPU state.
            self.close(session_id)
            raise

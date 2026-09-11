"""Text detection followed by bounded, per-task/per-camera SAM3 instance tracking."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import time
from types import MethodType

from grm_runtime.common import fingerprint
from grm_runtime.grounding import validate_bbox


IDENTITY_POLICY = 'initial_instance'


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
    session: object = None
    index: int = 0
    query: str | None = None
    last_box: list | None = None
    loss_reason: str | None = None
    last_sha: str | None = None
    last_result: dict | None = None


class TrackingEngine:
    """Bind once per task. Loss is terminal; only an explicit new task may detect.

    Called under the service GPU lock. A continuation can never create a session,
    including after TTL expiry or a service restart.
    """
    identity_policy = IDENTITY_POLICY

    def __init__(self, detector, tracker, *, redetect_interval_s=None, max_gap_s=2.,
                 min_score=.5, match_iou=.1, session_ttl_s=60., max_sessions=8):
        # Accept old deployment YAMLs, but never re-enable text redetection.
        if redetect_interval_s is not None and (isinstance(redetect_interval_s, bool)
                or not isinstance(redetect_interval_s, (int, float))
                or not math.isfinite(redetect_interval_s) or redetect_interval_s <= 0):
            raise ValueError("Invalid legacy redetect_interval_s")
        values = (max_gap_s, session_ttl_s, min_score, match_iou)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Invalid tracking thresholds")
        if min(max_gap_s, session_ttl_s) <= 0 or not 0 <= min_score <= 1 or not 0 < match_iou <= 1:
            raise ValueError("Invalid tracking thresholds")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.detector, self.tracker = detector, tracker
        self.max_gap_s = max_gap_s
        self.min_score, self.match_iou = min_score, match_iou
        self.session_ttl_s, self.max_sessions = session_ttl_s, max_sessions
        self.sessions = {}
        self.fingerprint = fingerprint({"detector": detector.fingerprint, "kind": "sam3_tracker_video_v2",
            "dtype": str(getattr(tracker, 'dtype', None)), "identity_policy": self.identity_policy,
            "max_gap_s": max_gap_s, "min_score": min_score, "match_iou": match_iou})

    def close(self, session_id):
        self.sessions.pop(session_id, None)

    def expire(self):
        now = time.monotonic()
        for key, state in list(self.sessions.items()):
            if now-state.seen_at > self.session_ttl_s:
                del self.sessions[key]

    @staticmethod
    def _lose(state, reason):
        # Release GPU memory without forgetting that this task already tried to
        # bind an instance. Invalidate even duplicate-image results after loss.
        state.session, state.loss_reason = None, reason
        state.last_sha, state.last_result = None, None

    def _result(self, row, source, state, timing, started):
        reason = state.loss_reason
        return {"candidates": [row] if row else [], "selected": row,
            "status": "ok" if row else "no_detection",
            "selection_status": "ok" if row else reason if reason in {'ambiguous', 'no_detection'} else 'tracking_lost',
            "tracking_state": "tracking" if row else "lost", "loss_reason": reason,
            "identity_policy": self.identity_policy,
            "source": source, "score_type": "object_presence" if source == 'sam3_tracker' else "detection",
            "tracker_frame_index": state.index, "timing": timing,
            "latency_ms": (time.monotonic()-started)*1000}

    def update(self, session_id, image, queries, sha, *, initialize=False):
        if not isinstance(initialize, bool):
            raise ValueError('initialize must be boolean and true only on the first task request')
        self.expire()
        now, started = time.monotonic(), time.monotonic()
        timing = {"tracker_ms": 0., "detect_ms": 0.}
        state = self.sessions.get(session_id)
        if state is not None and (state.queries != tuple(queries) or state.size != image.size):
            raise ValueError("Tracking session target/size changed; start a new session")
        first = state is None
        if first:
            if not initialize:
                # No tombstone registry is needed: continuations fail closed
                # when GPU sessions are expired, closed, or lost on restart.
                return self._result(None, 'sam3_tracker',
                    Track(tuple(queries), image.size, now, loss_reason='session_missing'), timing, started)
            if len(self.sessions) >= self.max_sessions:
                raise ValueError("Too many tracking sessions; close inactive sessions")
            state = Track(tuple(queries), image.size, now)
            self.sessions[session_id] = state
        gap = now-state.seen_at
        state.seen_at = now
        if state.loss_reason is None and gap > self.max_gap_s:
            self._lose(state, 'update_gap')
        if state.loss_reason is None and state.last_sha == sha and state.last_result is not None:
            return deepcopy(state.last_result)
        row, source = None, 'sam3_tracker'
        try:
            if state.loss_reason is not None:
                pass  # Missing forever in this task; never return an old box.
            elif first:
                source = 'sam3_detection'
                tick = time.monotonic()
                candidates = sorted(self.detector.detect(image, queries), key=lambda c: -c['score'])
                timing['detect_ms'] = (time.monotonic()-tick)*1000
                timing['detector'] = deepcopy(getattr(self.detector, 'last_timing', {}))
                ambiguous = len(candidates) > 1 and candidates[1]['score'] >= candidates[0]['score']-.05
                if not candidates or ambiguous:
                    self._lose(state, 'ambiguous' if ambiguous else 'no_detection')
                else:
                    row = candidates[0]
                    box = validate_bbox(row['bbox'], image.size)
                    if row['query'] not in queries or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
                        raise ValueError('Invalid initial detection score/query')
                    row = {**row, 'bbox': box}
                    tick = time.monotonic()
                    state.session = self.tracker.initialize(image, box)
                    timing['tracker_ms'] += (time.monotonic()-tick)*1000
                    state.query, state.last_box = row['query'], list(box)
            else:
                tick = time.monotonic()
                state.index += 1
                row = self.tracker.step(state.session, image, state.index)
                timing['tracker_ms'] += (time.monotonic()-tick)*1000
                if row is None:
                    self._lose(state, 'empty_mask')
                elif not math.isfinite(row['score']) or not self.min_score <= row['score'] <= 1:
                    self._lose(state, 'low_score')
                else:
                    box = validate_bbox(row['bbox'], image.size)
                    if box_iou(box, state.last_box) < self.match_iou:
                        self._lose(state, 'discontinuous_bbox')
                    else:
                        row = {**row, 'bbox': box, 'query': state.query}
                        state.last_box = list(box)
                if state.loss_reason is not None:
                    row = None
            result = self._result(row, source, state, timing, started)
            state.last_sha, state.last_result = sha, deepcopy(result)
            return result
        except Exception:
            # Drop partially advanced GPU state, but keep the task terminal so
            # an HTTP retry cannot bind a different instance.
            self._lose(state, 'inference_error')
            raise

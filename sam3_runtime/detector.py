"""SAM3 box detection, with optional mask-head bypass and stage timings."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace


def bypass_mask_decoder(model, torch):
    """Boxes/presence/logits are already computed before this independent head.

    Keep the upstream forward intact, including its attention implementation.
    This is deliberately scoped to the image detector, never the video tracker.
    """
    class NoMasks(torch.nn.Module):
        def forward(self, **kwargs):
            return SimpleNamespace(pred_masks=None, semantic_seg=None, attentions=None)
    model.mask_decoder = NoMasks()


class SAM3Detector:
    def __init__(self, model_path, device="cuda:0", threshold=0.3, mask_threshold=0.5,
                 dtype="float32", bbox_only=False, profile=False, num_threads=4):
        import torch
        from transformers import Sam3Model, Sam3Processor
        if dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError("SAM3 dtype must be float32, bfloat16, or float16")
        if not isinstance(bbox_only, bool) or not isinstance(profile, bool):
            raise ValueError("bbox_only and profile must be boolean")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer")
        torch.set_num_threads(num_threads)
        self.torch, self.device, self.dtype = torch, device, getattr(torch, dtype)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model = Sam3Model.from_pretrained(model_path, dtype=self.dtype).to(device).eval()
        if bbox_only:
            bypass_mask_decoder(self.model, torch)
        self.threshold, self.mask_threshold = float(threshold), float(mask_threshold)
        if not 0 <= self.threshold <= 1 or not 0 <= self.mask_threshold <= 1:
            raise ValueError("SAM3 thresholds must be in [0,1]")
        self.last_timing = {}
        self._events, self._hooks = {}, []
        self._cuda = str(device).startswith("cuda")
        if profile and self._cuda:
            # Events do not force a synchronization at every module boundary.
            for name in ("vision_encoder", "text_encoder", "detr_encoder", "detr_decoder", "mask_decoder"):
                module = getattr(self.model, name, None)
                if module is not None:
                    def before(module, args, name=name):
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(torch.cuda.current_stream(self.device))
                        self._events.setdefault(name, []).append([event, None])
                    def after(module, args, output, name=name):
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(torch.cuda.current_stream(self.device))
                        self._events[name][-1][1] = event
                    self._hooks.extend([module.register_forward_pre_hook(before),
                                        module.register_forward_hook(after)])
        metadata = {"model_path": str(Path(model_path).resolve()), "config": self.model.config.to_dict(),
                    "postprocess": "object_detection_v1", "dtype": dtype, "bbox_only": bbox_only,
                    "threshold": threshold, "mask_threshold": mask_threshold,
                    "weights": [(p.name, p.stat().st_size, p.stat().st_mtime_ns)
                                for p in sorted(Path(model_path).glob('*.safetensors'))]}
        self.fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()

    def _sync(self):
        if getattr(self, '_cuda', False):
            self.torch.cuda.synchronize(self.device)

    def detect(self, image, queries):
        rows, timing = [], {"prepare_ms": 0., "forward_ms": 0., "postprocess_ms": 0.}
        self._events = {}
        for query in queries:
            start = time.monotonic()
            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            # Integer tokens/sizes must retain their dtype.
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=getattr(self, 'dtype', self.torch.float32))
            self._sync()
            timing["prepare_ms"] += (time.monotonic()-start)*1000
            start = time.monotonic()
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            self._sync()
            timing["forward_ms"] += (time.monotonic()-start)*1000
            start = time.monotonic()
            # Geometry/scoring postprocess should not round pixel coordinates
            # to BF16 precision a second time after the model forward.
            if self.torch.is_tensor(getattr(outputs, 'pred_boxes', None)):
                for name in ('pred_boxes', 'pred_logits', 'presence_logits'):
                    value = getattr(outputs, name, None)
                    if self.torch.is_tensor(value):
                        setattr(outputs, name, value.float())
            result = self.processor.post_process_object_detection(outputs,
                threshold=self.threshold, target_sizes=inputs["original_sizes"].tolist())[0]
            boxes = result["boxes"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            for box, score in zip(boxes, scores):
                from grm_runtime.grounding import validate_bbox
                try:
                    box = validate_bbox(box, image.size)
                except ValueError:
                    continue
                duplicate = False
                for previous in rows:
                    b = previous["bbox"]
                    area = max(0, min(b[2],box[2])-max(b[0],box[0])) * max(0,min(b[3],box[3])-max(b[1],box[1]))
                    union = (b[2]-b[0])*(b[3]-b[1])+(box[2]-box[0])*(box[3]-box[1])-area
                    if union and area/union >= 0.8:
                        duplicate = True
                        break
                if not duplicate:
                    rows.append({"bbox": box, "score": float(score), "query": query})
            timing["postprocess_ms"] += (time.monotonic()-start)*1000
        for name, pairs in self._events.items():
            timing[name + "_gpu_ms"] = sum(a.elapsed_time(b) for a, b in pairs if b is not None)
        self.last_timing = timing
        return sorted(rows, key=lambda row: -row["score"])

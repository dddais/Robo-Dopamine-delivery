"""Eight-image GRM generation with scoped, per-request attention interventions."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict

from PIL import Image

from .common import CONDITIONS, file_sha, fingerprint, parse_score, target_queries
from .config import load_heads, load_steering
from .grounding import GroundingClient, GroundingError, validate_bbox
from .masking import ImageSpan, bbox_to_token_positions, make_attention_mask_hook, matched_wrong_position_set, resolve_negative_positions
from .prompt import IMAGE_LABELS, SYSTEM_PROMPT, messages


def infer_spans(inputs, config, paths, merge):
    ids = inputs["input_ids"][0].tolist()
    token = config.image_token_id
    ranges = []
    pos = 0
    while pos < len(ids):
        if ids[pos] != token:
            pos += 1
            continue
        start = pos
        while pos < len(ids) and ids[pos] == token:
            pos += 1
        ranges.append((start, pos))
    grids = inputs["image_grid_thw"].tolist()
    if len(ranges) != 8 or len(grids) != 8 or len(paths) != 8:
        raise ValueError("Eight-image token/span alignment failed")
    spans = []
    for index, ((start, end), grid) in enumerate(zip(ranges, grids)):
        t, h, w = grid
        if h % merge or w % merge or end - start != t * (h // merge) * (w // merge):
            raise ValueError("Image token count differs from processor grid")
        spans.append(ImageSpan(IMAGE_LABELS[index], paths[index], start, end, tuple(grid)))
    # Explicit regression guard for the historical before/after adapter bug.
    assert spans[5].label == "after_cam_high" and spans[5].path == paths[5]
    return spans


class HFBackend:
    def __init__(self, model_path, *, steering_config=None, device="cuda:0", dtype="bfloat16",
                 min_pixels=12544, max_pixels=76800, max_new_tokens=64, grounding_client=None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.config = load_steering(steering_config) if not isinstance(steering_config, dict) else steering_config
        self.config = self.config or {"enabled": False}
        self.lock = threading.RLock()
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.image_processor.min_pixels = min_pixels
        self.processor.image_processor.max_pixels = max_pixels
        self.model = AutoModelForImageTextToText.from_pretrained(model_path,
            dtype=self.dtype, attn_implementation="eager", trust_remote_code=True).to(device).eval()
        text = getattr(self.model.config, "text_config", self.model.config)
        self.num_heads = int(text.num_attention_heads)
        self.num_layers = int(text.num_hidden_layers)
        self.merge = int(self.model.config.vision_config.spatial_merge_size)
        if self.merge != int(self.processor.image_processor.merge_size):
            raise ValueError("Processor/model spatial merge mismatch")
        self.heads, self.low_heads = [], []
        self.grounder = grounding_client
        if self.config["enabled"]:
            self.heads, self.low_heads = load_heads(self.config, model_path, self.num_layers, self.num_heads)
            self.grounder = self.grounder or GroundingClient(**self.config.get("grounding", {}))
        self.manifest = {"engine": "hf", "model_path": model_path, "model_config": self.model.config.to_dict(),
                         "prompt_sha256": fingerprint(SYSTEM_PROMPT), "processor": self.processor.image_processor.to_dict(),
                         "decoding": "greedy", "dtype": dtype, "max_new_tokens": self.max_new_tokens,
                         "attention_implementation": "eager", "steering_config": self.config}
        if self.config["enabled"]:
            self.manifest["ranking_sha256"] = file_sha(self.config["ranking_path"])
        self.manifest["fingerprint"] = fingerprint(self.manifest)

    @property
    def layers(self):
        for obj in (getattr(self.model.model, "language_model", None), self.model.model):
            if obj is not None and hasattr(obj, "layers"):
                return obj.layers
        raise ValueError("Cannot locate GRM language decoder layers")

    @contextmanager
    def hooks(self, heads, selected, negative, diagnostics):
        handles = []
        by_layer = {}
        for head in heads:
            by_layer.setdefault(head.layer, []).append(head.head)
        try:
            for layer, indices in by_layer.items():
                diag = diagnostics.setdefault(str(layer), {})
                hook = make_attention_mask_hook(indices, selected, negative, self.num_heads,
                    self.config["bias"], diag, query_scope=self.config["query_scope"])
                handles.append(self.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _regions(self, sample, spans):
        selected, target_spans, grounding = [], [], {}
        queries = target_queries(sample["task"], sample.get("target_queries"), self.config.get("task_queries"))
        for label in self.config["intervention_labels"]:
            span = spans[IMAGE_LABELS.index(label)]
            with Image.open(span.path) as im:
                size = im.size
            # Explicit annotations are for reproducible/offline diagnostics; they must name the exact file hash.
            if label in sample.get("grounding", {}):
                row = sample["grounding"][label]
                if row.get("file_sha256") != file_sha(span.path):
                    raise ValueError("Supplied bbox image fingerprint mismatch")
                box = validate_bbox(row["bbox"], size)
                result = {**row, "selected": {"bbox": box}, "selection_status": "ok", "source": "supplied"}
            else:
                result = self.grounder.detect(span.path, queries)
            grounding[label] = result
            if result.get("selected") is None:
                return [], [], grounding, result.get("selection_status", "no_detection")
            box = validate_bbox(result["selected"]["bbox"], size)
            positions = bbox_to_token_positions(span, box, size, self.merge)
            if not positions:
                raise ValueError("bbox mapped to no image tokens")
            selected.extend(positions)
            target_spans.append(span)
        return sorted(set(selected)), target_spans, grounding, None

    def inference_batch(self, samples):
        # Serialize whole sample generation, including baseline. No hooks leak to other sessions.
        results = []
        for sample in samples:
            with self.lock:
                results.append(self._infer(sample))
        return results

    def _infer(self, sample):
        torch = self.torch
        started = time.monotonic()
        condition = sample.get("condition", "candidate_target" if self.config["enabled"] else "baseline")
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown condition: {condition}")
        if not self.config["enabled"] and condition != "baseline":
            raise ValueError("Steering condition requires enabled steering configuration")
        paths = sample["image"]
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert("RGB"))
        prompt = self.processor.apply_chat_template(messages(sample["task"]), tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        spans = infer_spans(inputs, self.model.config, paths, self.merge)
        inputs = {k: v.to(device=self.device, dtype=self.dtype if k == "pixel_values" else v.dtype)
                  if torch.is_tensor(v) else v for k, v in inputs.items()}
        diag = {"condition": condition, "enabled": self.config["enabled"], "applied": False,
                "profile_fingerprint": self.config.get("profile_sha256"), "per_layer": {},
                "spans": [asdict(s) for s in spans], "grounding": {}, "degraded": False}
        heads, selected, negative = [], [], []
        ground_start = time.monotonic()
        if self.config["enabled"] and condition != "baseline":
            try:
                selected, target_spans, grounded, missing = self._regions(sample, spans)
                diag["grounding"] = grounded
            except GroundingError as exc:
                selected, target_spans, missing = [], [], str(exc)
            if missing:
                if self.config["on_missing_bbox"] == "error":
                    raise GroundingError(missing)
                diag.update(degraded=True, reason=missing)
            else:
                heads = self.low_heads if condition == "low_rank_target" else self.heads
                if condition == "candidate_wrong":
                    wrong = []
                    for span in target_spans:
                        target = [p for p in selected if span.start <= p < span.end]
                        part = matched_wrong_position_set(span, target, spatial_merge_size=self.merge)
                        if part is None:
                            wrong = []
                            break
                        wrong.extend(part)
                    if len(wrong) != len(selected):
                        pool = sorted({p for s in spans for p in range(s.start, s.end)} - set(selected))
                        if len(pool) < len(selected):
                            raise ValueError("No equal-sized disjoint wrong-region control")
                        wrong = pool[-len(selected):]
                        diag["control_region"] = "other_visual_tokens_fallback"
                    else:
                        diag["control_region"] = "same_plane_farthest_rectangle"
                    selected = sorted(wrong)
                negative, labels = resolve_negative_positions(spans, selected, self.config["negative_scope"])
                diag.update(selected_span_labels=labels, target_positions=selected, negative_positions=negative,
                            heads=[asdict(h) for h in heads], bias=self.config["bias"],
                            query_scope=self.config["query_scope"], negative_scope=self.config["negative_scope"])
        diag["grounding_ms"] = (time.monotonic() - ground_start) * 1000
        generate_start = time.monotonic()
        from contextlib import nullcontext
        active = bool(heads and self.config["bias"] != 0)
        with self.hooks(heads, selected, negative, diag["per_layer"]) if active else nullcontext():
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    use_cache=True, output_attentions=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id)
        if active:
            if not diag["per_layer"] or any(d["applied_calls"] == 0 for d in diag["per_layer"].values()):
                raise RuntimeError("Steering requested but hooks were not applied")
            diag["applied"] = True
        pred = self.processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            score, error = parse_score(pred), None
        except ValueError as exc:
            score, error = None, str(exc)
        diag["grm_ms"] = (time.monotonic() - generate_start) * 1000
        diag["total_ms"] = (time.monotonic() - started) * 1000
        return {**sample, "pred": pred, "parsed_score": score, "valid": score is not None,
                "error": error, "engine": "hf", "steering": diag}

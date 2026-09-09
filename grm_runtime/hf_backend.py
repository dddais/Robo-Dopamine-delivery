"""Eight-image GRM generation with scoped, per-request attention interventions."""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from uuid import uuid4

from PIL import Image

from .common import CONDITIONS, file_sha, fingerprint, parse_score, target_queries
from .config import load_heads, load_steering
from .grounding import GroundingClient, GroundingError, validate_bbox
from .masking import ImageSpan, bbox_to_token_positions, make_attention_mask_hook, make_batched_attention_mask_hook, matched_wrong_position_set, resolve_negative_positions
from .prompt import IMAGE_LABELS, SYSTEM_PROMPT, messages


class _BatchCache:
    """Bounded CPU-only reuse within one batch; never retain data across rounds."""

    def __init__(self):
        self.images = OrderedDict()
        self.grounding = OrderedDict()

    @staticmethod
    def file_key(path):
        path = Path(path).resolve()
        stat = path.stat()
        # Also invalidate if an offline caller overwrites a path within a batch.
        return (str(path), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns)

    def image(self, path):
        key = self.file_key(path)
        reused = key in self.images
        if reused:
            image = self.images.pop(key)
        else:
            with Image.open(path) as source:
                image = source.convert("RGB")
        self.images[key] = image
        if len(self.images) > 16:
            self.images.popitem(last=False)
        return image, reused

    def detect(self, grounder, path, queries):
        key = (self.file_key(path), tuple(queries))
        reused = key in self.grounding
        if reused:
            result = self.grounding.pop(key)
        else:
            # The first request still checks the server model fingerprint.
            # Successful no-detection/ambiguous results are shared too; request
            # exceptions are not cached, so the next sample can retry.
            result = grounder.detect(path, queries)
        self.grounding[key] = result
        if len(self.grounding) > 128:
            self.grounding.popitem(last=False)
        return {**deepcopy(result), "reused_in_batch": reused}


def infer_spans(inputs, config, paths, merge, row=0):
    ids = inputs["input_ids"][row].tolist()
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
    all_grids = inputs["image_grid_thw"].tolist()
    if len(all_grids) != 8 * inputs["input_ids"].shape[0]:
        raise ValueError("Expected eight image grids per batch row")
    grids = all_grids[row*8:(row+1)*8]
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
                 min_pixels=12544, max_pixels=76800, max_new_tokens=64, grounding_client=None,
                 batch_size=2):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.config = load_steering(steering_config) if not isinstance(steering_config, dict) else steering_config
        self.config = self.config or {"enabled": False}
        self.lock = threading.RLock()
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
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
                         "batch_size": self.batch_size, "padding_side": "left",
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

    @contextmanager
    def batch_hooks(self, plans, diagnostics):
        handles = []
        by_layer = {}
        for index, (heads, selected, negative) in enumerate(plans):
            if self.config.get("bias", 0) == 0:
                continue
            for head in heads:
                rows = by_layer.setdefault(head.layer, [None] * len(plans))
                if rows[index] is None:
                    rows[index] = ([], selected, negative, diagnostics[index].setdefault(str(head.layer), {}))
                rows[index][0].append(head.head)
        try:
            for layer, specs in by_layer.items():
                hook = make_batched_attention_mask_hook(specs, self.num_heads, self.config['bias'],
                                                       query_scope=self.config['query_scope'])
                handles.append(self.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _regions(self, sample, spans, batch=None):
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
                result = (batch.detect(self.grounder, span.path, queries) if batch is not None
                          else self.grounder.detect(span.path, queries))
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
        # Bound GPU memory for offline callers too. The online two-mode round
        # becomes one generate call per branch; a third mode uses a tail batch.
        results = []
        with self.lock:
            batch = _BatchCache()
            iterator = iter(samples)
            while group := list(islice(iterator, self.batch_size)):
                results.extend(self._infer_group(group, batch))
        return results

    def _infer(self, sample, batch=None):
        return self._infer_group([sample], batch if batch is not None else _BatchCache())[0]

    def _steering_plan(self, sample, spans, diag, batch):
        condition = diag['condition']
        heads, selected, negative = [], [], []
        ground_start = time.monotonic()
        if self.config["enabled"] and condition != "baseline":
            try:
                selected, target_spans, grounded, missing = self._regions(sample, spans, batch=batch)
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
        return heads, selected, negative

    def _infer_group(self, samples, batch):
        torch = self.torch
        started = time.monotonic()
        images, prompts, diagnostics = [], [], []
        batch_id = uuid4().hex
        for index, sample in enumerate(samples):
            condition = sample.get("condition", "candidate_target" if self.config["enabled"] else "baseline")
            if condition not in CONDITIONS:
                raise ValueError(f"Unknown condition: {condition}")
            if not self.config["enabled"] and condition != "baseline":
                raise ValueError("Steering condition requires enabled steering configuration")
            if len(sample['image']) != 8:
                raise ValueError('Each GRM batch row must contain exactly eight images')
            hits = 0
            for path in sample['image']:
                image, reused = batch.image(path)
                images.append(image)
                hits += int(reused)
            prompts.append(self.processor.apply_chat_template(messages(sample['task']),
                           tokenize=False, add_generation_prompt=True))
            diagnostics.append(dict(condition=condition, enabled=self.config['enabled'], applied=False,
                profile_fingerprint=self.config.get('profile_sha256'), per_layer={}, grounding={},
                degraded=False, image_cache_hits=hits, batch_id=batch_id, batch_size=len(samples), batch_index=index))
        inputs = self.processor(text=prompts, images=images, padding=True, return_tensors='pt')
        if inputs['input_ids'].shape[0] != len(samples):
            raise ValueError('Processor batch size differs from sample count')
        if not bool(inputs['attention_mask'][:, -1].all()):
            raise ValueError('GRM generation requires left padding')
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        if len(set(lengths)) > 1:
            # BF16 Qwen3-VL can change scores on padded heterogeneous batches
            # even when valid tokens, RoPE and attention masks match serial.
            # Bucket equal lengths; the online two-mode round normally stays B=2.
            regroup_ms = (time.monotonic() - started) * 1000 / len(samples)
            groups = {}
            for index, length in enumerate(lengths):
                groups.setdefault(length, []).append(index)
            results = [None] * len(samples)
            for indices in groups.values():
                rows = self._infer_group([samples[i] for i in indices], batch)
                for index, result in zip(indices, rows):
                    diag = result['steering']
                    diag['regroup_prepare_ms'] = regroup_ms
                    diag['prepare_ms'] += regroup_ms
                    diag['total_ms'] += regroup_ms
                    results[index] = result
            return results
        spans_by_row = [infer_spans(inputs, self.model.config, s['image'], self.merge, row=i)
                        for i, s in enumerate(samples)]
        for diag, spans, attention in zip(diagnostics, spans_by_row, inputs['attention_mask']):
            diag['spans'] = [asdict(s) for s in spans]
            diag['padding_left'] = int((attention == 0).sum())
        inputs = {k: v.to(device=self.device, dtype=self.dtype if k == 'pixel_values' else v.dtype)
                  if torch.is_tensor(v) else v for k, v in inputs.items()}
        prepare_ms = (time.monotonic() - started) * 1000
        plans = [self._steering_plan(s, spans, diag, batch)
                 for s, spans, diag in zip(samples, spans_by_row, diagnostics)]
        generate_start = time.monotonic()
        with self.batch_hooks(plans, [d['per_layer'] for d in diagnostics]):
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    use_cache=True, output_attentions=False, num_beams=1, num_return_sequences=1,
                    pad_token_id=self.processor.tokenizer.pad_token_id)
        if output.shape[0] != len(samples):
            raise RuntimeError('GRM generation returned a different batch size')
        texts = [self.processor.tokenizer.decode(row[inputs['input_ids'].shape[1]:],
                 skip_special_tokens=True).strip() for row in output]
        grm_ms = (time.monotonic() - generate_start) * 1000
        batch_total_ms = (time.monotonic() - started) * 1000
        results = []
        for sample, diag, (heads, _, _), pred in zip(samples, diagnostics, plans, texts):
            if heads and self.config['bias'] != 0:
                if not diag['per_layer'] or any(d['applied_calls'] == 0 for d in diag['per_layer'].values()):
                    raise RuntimeError('Steering requested but hooks were not applied')
                diag['applied'] = True
            try:
                score, error = parse_score(pred), None
            except ValueError as exc:
                score, error = None, str(exc)
            # Existing monitor aggregation sums mode timings. Attribute shared
            # work equally, and retain full batch times separately for profiling.
            diag.update(prepare_ms=prepare_ms / len(samples), grm_ms=grm_ms / len(samples),
                        batch_prepare_ms=prepare_ms, batch_grm_ms=grm_ms, batch_total_ms=batch_total_ms)
            diag['total_ms'] = diag['prepare_ms'] + diag['grounding_ms'] + diag['grm_ms']
            results.append({**sample, 'pred': pred, 'parsed_score': score, 'valid': score is not None,
                            'error': error, 'engine': 'hf', 'steering': diag})
        return results

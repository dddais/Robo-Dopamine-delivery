"""Adapted from Robo-Dopamine mydata_bench/attention_eval/masking.py (Apache-2.0)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


QUERY_SCOPES = frozenset({"all", "prefill", "last_prompt", "decode"})
NEGATIVE_SCOPES = frozenset({"all_visual", "target_span", "other_spans", "none"})


@dataclass(frozen=True)
class Head:
    layer: int
    head: int


@dataclass(frozen=True)
class ImageSpan:
    label: str
    path: str
    start: int
    end: int
    grid_thw: tuple[int, int, int]

    @property
    def token_count(self) -> int:
        return self.end - self.start


def resolve_negative_positions(
    spans: Sequence[ImageSpan],
    selected_positions: Sequence[int],
    negative_scope: str,
) -> tuple[list[int], list[str]]:
    """Resolve negative-bias keys under one cross-model contract.

    other_spans is the primary frame-isolation intervention: target bbox
    tokens receive positive bias, every other image/time span receives
    negative bias, and non-target tokens in the target span stay unchanged.
    """
    scope = str(negative_scope)
    if scope not in NEGATIVE_SCOPES:
        choices = ", ".join(sorted(NEGATIVE_SCOPES))
        raise ValueError(f"Unknown negative_scope {scope!r}; choose one of {choices}")
    selected = {int(value) for value in selected_positions}
    visual = {position for span in spans for position in range(span.start, span.end)}
    if not selected:
        raise ValueError("selected_positions must not be empty")
    if not selected <= visual:
        raise ValueError("Selected attention positions are outside visual spans")
    selected_spans = [
        span
        for span in spans
        if selected & set(range(span.start, span.end))
    ]
    labels = [span.label for span in selected_spans]
    selected_span_positions = {
        position
        for span in selected_spans
        for position in range(span.start, span.end)
    }
    if scope == "all_visual":
        negative = visual - selected
    elif scope == "target_span":
        negative = selected_span_positions - selected
    elif scope == "other_spans":
        negative = visual - selected_span_positions
    else:
        negative = set()
    if selected & negative:
        raise AssertionError("Positive and negative attention token sets overlap")
    return sorted(negative), labels


def bbox_to_token_positions(
    span: ImageSpan,
    bbox: Sequence[float],
    image_size: tuple[int, int],
    spatial_merge_size: int = 2,
) -> list[int]:
    """Map xyxy pixels to every vision cell intersecting the box."""
    width, height = image_size
    x1, y1, x2, y2 = map(float, bbox)
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return []
    t, raw_h, raw_w = span.grid_thw
    grid_h = max(1, raw_h // spatial_merge_size)
    grid_w = max(1, raw_w // spatial_merge_size)
    expected = t * grid_h * grid_w
    if expected != span.token_count:
        raise ValueError(
            f"Vision span/grid mismatch: span={span.token_count}, grid={expected}, "
            f"grid_thw={span.grid_thw}, merge={spatial_merge_size}"
        )
    xs = np.linspace(0, width, grid_w + 1)
    ys = np.linspace(0, height, grid_h + 1)
    x_keep = (xs[:-1] < x2) & (xs[1:] > x1)
    y_keep = (ys[:-1] < y2) & (ys[1:] > y1)
    keep = np.outer(y_keep, x_keep).reshape(-1)
    keep = np.tile(keep, t)
    return [span.start + index for index, selected in enumerate(keep) if selected]


def matched_wrong_position_set(
    span: ImageSpan,
    target_positions: Sequence[int],
    *,
    spatial_merge_size: int = 2,
) -> list[int] | None:
    """Find a non-overlapping grid rectangle with exactly the target token count.

    If the target covers multiple temporal planes or is not rectangular, this
    intentionally returns None instead of shrinking or fabricating a control.
    """
    t, raw_h, raw_w = span.grid_thw
    grid_h = raw_h // spatial_merge_size
    grid_w = raw_w // spatial_merge_size
    if t != 1 or grid_h < 1 or grid_w < 1:
        return None
    relative = sorted({int(value) - span.start for value in target_positions})
    if not relative:
        return None
    rows = [value // grid_w for value in relative]
    cols = [value % grid_w for value in relative]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    if height * width != len(relative):
        return None
    target = set(relative)
    candidates = []
    target_center = ((min(rows) + max(rows)) / 2, (min(cols) + max(cols)) / 2)
    for row in range(grid_h - height + 1):
        for col in range(grid_w - width + 1):
            candidate = {
                (row + dy) * grid_w + col + dx
                for dy in range(height)
                for dx in range(width)
            }
            if target & candidate:
                continue
            center = (row + (height - 1) / 2, col + (width - 1) / 2)
            distance = (center[0] - target_center[0]) ** 2 + (
                center[1] - target_center[1]
            ) ** 2
            candidates.append((distance, row, col, candidate))
    if not candidates:
        return None
    # Farthest valid location is deterministic and minimizes accidental overlap
    # with target context.
    _, _, _, selected = max(candidates)
    return [span.start + value for value in sorted(selected)]


def select_low_ranked_heads(
    ranking: list[dict], count: int, candidates: Sequence[Head]
) -> list[Head]:
    excluded = {(head.layer, head.head) for head in candidates}
    selected = []
    for row in reversed(ranking):
        pair = (int(row["layer"]), int(row["head"]))
        if pair in excluded:
            continue
        selected.append(Head(*pair))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Insufficient non-overlapping low-ranked heads")
    return selected


def make_attention_mask_hook(
    head_indices: Sequence[int],
    selected_positions: Sequence[int],
    other_visual_positions: Sequence[int],
    num_query_heads: int,
    swap_bias: float,
    diagnostics: dict | None = None,
    *,
    query_scope: str = "all",
):
    """Build an additive pre-softmax attention hook for a declared query scope.

    ``all`` preserves the historical behavior: every prefill query row and
    every cached decode call receive the same key-position bias. ``prefill``
    applies it only during the multi-query prompt forward pass,
    ``last_prompt`` only to that pass's final query row, and ``decode`` only to
    subsequent cached calls whose query length is one.
    """
    import torch

    query_scope = str(query_scope)
    if query_scope not in QUERY_SCOPES:
        choices = ", ".join(sorted(QUERY_SCOPES))
        raise ValueError(f"Unknown query_scope {query_scope!r}; choose one of {choices}")
    heads = sorted({int(value) for value in head_indices})
    selected = sorted({int(value) for value in selected_positions})
    other = sorted({int(value) for value in other_visual_positions})
    if set(selected) & set(other) or any(p < 0 for p in selected + other):
        raise ValueError("Bias positions must be nonnegative and disjoint")
    if any(value < 0 or value >= num_query_heads for value in heads):
        raise ValueError("Selected head index is outside query-head range")
    base_length = max(selected + other, default=-1) + 1
    base = torch.zeros((1, num_query_heads, 1, base_length), dtype=torch.float32)
    for head in heads:
        if selected:
            base[0, head, 0, selected] = float(swap_bias)
        if other:
            base[0, head, 0, other] = -float(swap_bias)
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(
        {
            "calls": 0,
            "prefill_calls": 0,
            "decode_calls": 0,
            "applied_calls": 0,
            "skipped_calls": 0,
            "prefill_applied_calls": 0,
            "decode_applied_calls": 0,
            "applied_query_rows": 0,
            "selected_heads": heads,
            # Per-record outputs retain the selected/visual token positions.
            # Keep only compact evidence here because this dictionary is
            # replicated once for every steered layer and every cohort sample.
            "selected_token_count": len(selected),
            "other_visual_token_count": len(other),
            "selected_other_disjoint": not bool(set(selected) & set(other)),
            "swap_bias": float(swap_bias),
            "query_scope": query_scope,
            "new_text_keys_zero_bias": True,
        }
    )

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            raise RuntimeError("Steering requires an explicit eager attention mask")
        if mask.ndim != 4 or not mask.is_floating_point():
            raise RuntimeError("Steering requires a 4D floating-point attention mask")
        diagnostics["calls"] += 1
        query_length = int(mask.shape[-2])
        is_decode = query_length == 1
        if is_decode:
            diagnostics["decode_calls"] += 1
        else:
            diagnostics["prefill_calls"] += 1
        should_apply = (
            query_scope == "all"
            or (query_scope in {"prefill", "last_prompt"} and not is_decode)
            or (query_scope == "decode" and is_decode)
        )
        if not should_apply:
            diagnostics["skipped_calls"] += 1
            return None
        bias = base.to(device=mask.device, dtype=mask.dtype)
        key_length = int(mask.shape[-1])
        if key_length > base_length:
            bias = torch.nn.functional.pad(bias, (0, key_length - base_length))
        else:
            bias = bias[..., :key_length]
        if query_scope == "last_prompt":
            scoped = torch.zeros(
                (1, num_query_heads, query_length, key_length),
                device=mask.device,
                dtype=mask.dtype,
            )
            scoped[..., -1:, :] = bias
            bias = scoped
            applied_rows = 1
        else:
            applied_rows = query_length
        diagnostics["applied_calls"] += 1
        diagnostics["applied_query_rows"] += applied_rows
        if is_decode:
            diagnostics["decode_applied_calls"] += 1
        else:
            diagnostics["prefill_applied_calls"] += 1
        new_kwargs = dict(kwargs)
        new_kwargs["attention_mask"] = mask + bias
        return args, new_kwargs

    return hook


def make_batched_attention_mask_hook(specs, num_query_heads, swap_bias, *, query_scope="all"):
    """Apply independent per-row hooks; None specs leave baseline rows untouched.

    Each non-None spec is (heads, selected_keys, negative_keys, diagnostics).
    Keys already include each row's left padding. The original causal/padding
    mask is retained for every row, both at prefill and cached decode.
    """
    import torch

    hooks = [None if spec is None else make_attention_mask_hook(
        *spec[:3], num_query_heads, swap_bias, spec[3], query_scope=query_scope)
        for spec in specs]

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 4 or not mask.is_floating_point():
            raise RuntimeError("Batched steering requires a 4D floating-point attention mask")
        if mask.shape[0] not in (1, len(hooks)):
            raise RuntimeError("Attention mask batch size differs from steering samples")
        rows, changed = [], False
        for index, row_hook in enumerate(hooks):
            row = mask[index:index+1] if mask.shape[0] != 1 else mask
            result = row_hook(module, args, {**kwargs, "attention_mask": row}) if row_hook else None
            if result is not None:
                row = result[1]["attention_mask"]
                changed = True
            rows.append(row.expand(1, num_query_heads, *row.shape[2:]))
        if not changed:
            return None
        return args, {**kwargs, "attention_mask": torch.cat(rows, dim=0)}

    return hook

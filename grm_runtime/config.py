"""Frozen head profiles and explicit, validated steering options."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .common import file_sha, fingerprint, load_yaml, resolve_path
from .masking import Head, QUERY_SCOPES, NEGATIVE_SCOPES
from .prompt import IMAGE_LABELS

KEYS = {"enabled", "profile_path", "top_k", "bias", "query_scope", "negative_scope",
        "intervention_labels", "on_missing_bbox", "grounding", "task_queries", "allow_model_transfer"}


def load_steering(path=None):
    if path is None:
        return {"enabled": False}
    data, base = load_yaml(path, KEYS)
    data.setdefault("enabled", True)
    if not isinstance(data["enabled"], bool):
        raise ValueError("enabled must be boolean")
    if not isinstance(data.get("allow_model_transfer", False), bool):
        raise ValueError("allow_model_transfer must be boolean")
    if not data["enabled"]:
        return data
    if "profile_path" not in data:
        raise ValueError("steering requires profile_path")
    profile_path = Path(resolve_path(data["profile_path"], base))
    profile = json.loads(profile_path.read_text())
    for name in ("top_k", "bias", "query_scope", "negative_scope", "intervention_labels"):
        data.setdefault(name, profile[name])
    data["profile"] = profile
    data["profile_path"] = str(profile_path)
    data["profile_sha256"] = file_sha(profile_path)
    data["ranking_path"] = resolve_path(profile["ranking_path"], profile_path.parent)
    data.setdefault("on_missing_bbox", "baseline")
    if data["query_scope"] not in QUERY_SCOPES or data["negative_scope"] not in NEGATIVE_SCOPES:
        raise ValueError("Invalid query_scope or negative_scope")
    labels = data["intervention_labels"]
    if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels) or any(l not in IMAGE_LABELS[2:] for l in labels):
        raise ValueError("intervention_labels must be unique BEFORE/AFTER camera labels")
    if not isinstance(data["top_k"], int) or data["top_k"] < 1:
        raise ValueError("top_k must be positive integer")
    if not math.isfinite(float(data["bias"])) or data["bias"] < 0:
        raise ValueError("bias must be finite and non-negative")
    if data["on_missing_bbox"] not in {"baseline", "error"}:
        raise ValueError("on_missing_bbox must be baseline or error")
    grounding = data.setdefault("grounding", {})
    if set(grounding) - {"url", "timeout_s", "cache_dir"}:
        raise ValueError("Unknown grounding config key")
    if "cache_dir" in grounding:
        grounding["cache_dir"] = resolve_path(grounding["cache_dir"], base)
    data["configuration_sha256"] = fingerprint(data)
    return data


def load_heads(config, model_path, num_layers, num_heads):
    profile = config["profile"]
    if (profile["num_layers"], profile["num_heads"]) != (num_layers, num_heads):
        raise ValueError("Head profile architecture does not match model")
    identity = str(Path(model_path).expanduser().resolve())
    expected = str(Path(profile["model_path"]).expanduser().resolve())
    if not config.get("allow_model_transfer", False):
        if identity != expected:
            raise ValueError("Checkpoint differs from head profile; export a matching profile or explicitly allow experimental transfer")
        if file_sha(Path(model_path) / "config.json") != profile["model_config_sha256"]:
            raise ValueError("Checkpoint config fingerprint mismatch")
    raw = json.loads(Path(config["ranking_path"]).read_text())
    rows = raw.get("ranking") or raw.get("top_heads") or raw.get("rankings", {}).get(raw.get("default_ranking", "mean"))
    if not rows:
        raise ValueError("Ranking has no heads")
    pairs = [(int(row["layer"]), int(row["head"])) for row in rows]
    if len(set(pairs)) != len(pairs) or any(not (0 <= l < num_layers and 0 <= h < num_heads) for l, h in pairs):
        raise ValueError("Duplicate or out-of-range ranking heads")
    ranked = [Head(l, h) for l, h in pairs if l >= profile.get("skip_early_layers", 0)]
    if len(ranked) < 2 * config["top_k"]:
        raise ValueError("Ranking too short for disjoint candidate/low-rank controls")
    return ranked[:config["top_k"]], list(reversed(ranked))[:config["top_k"]]

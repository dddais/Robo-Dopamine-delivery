"""Small dependency-free contracts for scoring, provenance and configuration."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

MODES = ("forward", "incremental", "backward")
CONDITIONS = ("baseline", "candidate_target", "candidate_wrong", "low_rank_target")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_yaml(path, allowed=None):
    import yaml
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    if allowed is not None and set(value) - set(allowed):
        raise ValueError(f"Unknown config keys: {sorted(set(value) - set(allowed))}")
    return value, path.parent


def resolve_path(value, base):
    p = Path(value).expanduser()
    return str((base / p).resolve()) if not p.is_absolute() else str(p.resolve())


def parse_score(text):
    match = re.fullmatch(r"\s*<score>\s*([+-]?\d+(?:\.\d+)?)\s*%\s*</score>\s*", text)
    if not match:
        raise ValueError(f"Invalid GRM score: {text!r}")
    score = float(match[1]) / 100
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError(f"GRM score out of range: {text!r}")
    return score


def progress_step(mode, score, previous=0.0, count=0):
    if mode == "forward":
        progress = score
    elif mode == "backward":
        progress = max(0.0, min(1.0, 1 + score))
    elif mode == "incremental":
        progress = score if count == 0 else previous + (1 - previous if score >= 0 else previous) * score
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return {"score": score, "progress": progress,
            "hop": score if mode == "incremental" else progress - previous}


def target_queries(task, explicit=None, mappings=None):
    key = " ".join(task.split())
    mappings = {" ".join(k.split()): v for k, v in (mappings or {}).items()}
    queries = explicit if explicit is not None else mappings.get(key)
    if queries is None:
        match = re.match(r"^(?:pick(?: up)?|grasp|lift|push|pull|open|close|place|put|move|touch)\s+(?:the\s+|a\s+|an\s+)?(.+)$", key, re.I)
        if not match:
            raise ValueError("Cannot resolve task target; supply target_queries or task_queries mapping")
        phrase = re.split(r"\s+(?:and|then|into|onto|on|to|in)\s+", match[1], maxsplit=1)[0].strip(" .")
        if not phrase.isascii():
            raise ValueError("Supply target_queries for non-English tasks")
        queries = [phrase]
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8 or any(not isinstance(q, str) or not q.strip() for q in queries):
        raise ValueError("target_queries must contain 1..8 nonempty strings")
    return list(dict.fromkeys(q.strip() for q in queries))

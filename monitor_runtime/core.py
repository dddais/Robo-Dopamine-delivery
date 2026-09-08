"""Shared monitor state and progress-status logic."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

MONITOR_STATUS_RUNNING = "running"
MONITOR_STATUS_SUCCESS = "success"
MONITOR_STATUS_FAIL = "failed"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class MonitorState:
    """Tracks progress history and determines success / fail / running."""

    success_threshold: float = 0.60
    success_stable_steps: int = 5
    success_max_drift: float = 0.02
    fail_stable_steps: int = 8
    fail_min_progress: float = 0.01
    status: str = MONITOR_STATUS_RUNNING
    progress_history: list[float] = field(default_factory=list)
    success_counter: int = 0
    fail_counter: int = 0

    def reset(self) -> None:
        self.status = MONITOR_STATUS_RUNNING
        self.progress_history = []
        self.success_counter = 0
        self.fail_counter = 0

    def update(self, fused_progress: float) -> str:
        if self.status != MONITOR_STATUS_RUNNING:
            return self.status

        fused_progress = clamp(float(fused_progress), 0.0, 1.0)
        self.progress_history.append(fused_progress)

        if fused_progress >= self.success_threshold:
            recent = self.progress_history[-self.success_stable_steps:]
            if len(recent) >= self.success_stable_steps:
                drift = max(recent) - min(recent)
                if drift <= self.success_max_drift:
                    self.status = MONITOR_STATUS_SUCCESS
                    self.success_counter = len(recent)
                    return self.status
            self.success_counter += 1
            self.fail_counter = 0
        else:
            self.success_counter = 0
            recent = self.progress_history[-self.fail_stable_steps:]
            if len(recent) >= self.fail_stable_steps:
                has_improvement = any(
                    recent[i] - recent[i - 1] >= self.fail_min_progress
                    for i in range(1, len(recent))
                )
                if not has_improvement:
                    self.status = MONITOR_STATUS_FAIL
                    self.fail_counter = len(recent)
                    return self.status
            self.fail_counter += 1

        return self.status

    @property
    def is_finished(self) -> bool:
        return self.status in (MONITOR_STATUS_SUCCESS, MONITOR_STATUS_FAIL)


@dataclass
class MonitorSession:
    monitor_id: str
    execution_id: str
    subtask: str
    subtask_index: int | None = None
    status: str = MONITOR_STATUS_RUNNING
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    message: str | None = None
    poll_count: int = 0
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monitor_id": self.monitor_id,
            "execution_id": self.execution_id,
            "subtask": self.subtask,
            "subtask_index": self.subtask_index,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "message": self.message,
            "poll_count": self.poll_count,
            "result": self.result,
        }


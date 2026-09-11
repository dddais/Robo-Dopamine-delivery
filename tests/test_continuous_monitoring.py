"""Continuous monitoring keeps verdicts observational until an explicit stop."""
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from monitor_runtime.core import MonitorState, MonitorConflict
from monitor_runtime.grm_backend import CAMERA_KEYS, GRMMonitorBackend

ROOT = Path(__file__).resolve().parents[1]


def eventually(check):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("monitor did not reach expected state")


class Model:
    def __init__(self):
        self.score = 80

    def inference_batch(self, samples):
        return [{**s, "pred": f"<score>{self.score}%</score>", "valid": True,
                 "steering": {"enabled": True, "applied": True}} for s in samples]


@pytest.fixture
def backend(tmp_path):
    model, baseline = Model(), Model()
    backend = GRMMonitorBackend(runtime_url="http://unused", inference_engine="hf",
        goal_image=str(ROOT / "examples/blank_goal.png"), steering_config=str(ROOT / "configs/steering.yaml"),
        output_root=str(tmp_path), model=model, baseline_model=baseline, dual_branch=True,
        active_modes=["forward"], interval=.1, success_stable_steps=1, fail_stable_steps=2)
    def capture(state, **kwargs):
        state.capture_index += 1
        folder = backend._session_dir(state) / f"capture_{state.capture_index}"
        folder.mkdir()
        frames = {}
        for camera in CAMERA_KEYS:
            path = folder / f"{camera}.png"
            Image.new("RGB", (32, 32)).save(path)
            frames[camera] = str(path)
        return frames, {"identity": str(state.capture_index), "snapshot_requested_at": time.time()}
    backend._snapshot_current = capture
    try:
        yield backend, baseline
    finally:
        backend.close()


def test_continuous_verdicts_can_change_without_resetting_progress():
    state = MonitorState(continuous_monitoring=True, success_stable_steps=1,
                         fail_stable_steps=3, progress_difference_threshold=.2)
    assert state.update(.8, progress_difference=0) == "success"
    assert state.update(.2, progress_difference=.4) == "failed"
    assert state.update(.3, progress_difference=0) == "running"
    assert not state.is_finished
    for _ in range(100):
        state.update(.8, progress_difference=0)
    assert len(state.progress_history) == 3  # bounded continuous window
    with pytest.raises(ValueError, match="boolean"):
        MonitorState(continuous_monitoring="false")


@pytest.mark.parametrize("continuous", [False, True])
def test_worker_and_journal_follow_policy_and_manual_stop(backend, continuous):
    backend, baseline = backend
    payload = {"monitor_id": "monitor", "execution_id": "exec", "subtask": "pick cup",
               "target_queries": ["cup"], "defer_inference": True, "continuous_monitoring": continuous}
    backend.start(payload)
    state = backend.sessions["monitor"]
    eventually(lambda: state.ref_start)
    assert state.step == 0
    assert backend.status(payload).result["continuous_monitoring"] is continuous
    backend.activate(payload)
    eventually(lambda: state.step >= 1)
    assert state.latest["status"] == "success"
    if continuous:
        eventually(lambda: state.step >= 3)
        assert backend.status(payload).status == "running"
        assert not state.monitor.is_finished and state.thread.is_alive()
        # A branch-difference failure is also an assessment, not a worker stop.
        baseline.score = 0
        eventually(lambda: state.latest.get("status") == "failed")
        previous = state.step
        eventually(lambda: state.step >= previous + 2)
        status = backend.status(payload)
        assert status.status == "running" and status.result["status"] == "failed"
        assert not backend.records("monitor", "exec")["complete"]
    else:
        state.thread.join(2)
        assert state.step == 1 and not state.thread.is_alive()
        assert backend.status(payload).status == "success"
        assert backend.records("monitor", "exec")["complete"]
    with pytest.raises(MonitorConflict):
        backend.start({**payload, "continuous_monitoring": not continuous})
    assert backend.stop(payload)["stopped"]
    assert not state.thread.is_alive()
    journal = backend.records("monitor", "exec")
    assert journal["complete"] and len(journal["records"]) == state.step
    assert len({r["preview"]["frame_set_id"] for r in journal["records"]}) == state.step
    assert all(r["continuous_monitoring"] is continuous for r in journal["records"])
    manifest = json.loads((backend._session_dir(state) / "manifest.json").read_text())
    assert manifest["continuous_monitoring"] is continuous


def test_start_rejects_non_boolean_policy(backend):
    with pytest.raises(ValueError, match="continuous_monitoring must be boolean"):
        backend[0].start({"monitor_id": "m", "execution_id": "e", "subtask": "pick cup",
                          "continuous_monitoring": "false"})

"""Read persisted GRM scores without model loading or advancing inference."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from monitor_runtime.grm_backend import GRMMonitorBackend
from monitor_runtime.service import create_app


@pytest.fixture
def journal(tmp_path):
    backend = GRMMonitorBackend.__new__(GRMMonitorBackend)
    backend._lock = threading.RLock()
    backend._record_journals = {"mon": ("execution", "generation", tmp_path, "抓取 carrot")}
    backend._record_image_index = {}
    backend._record_image_lock = threading.Lock()
    backend._preview_frames = {}
    backend.sessions = {"mon": SimpleNamespace(monitor=SimpleNamespace(is_finished=False))}
    path = tmp_path / "online_pred.jsonl"
    return backend, path


def test_paginated_journal_ignores_partial_tail_and_remains_readable_after_stop(journal):
    backend, path = journal
    lines = [json.dumps({"inference_step": i, "subtask": "抓取 carrot"}, ensure_ascii=False).encode() + b"\n"
             for i in range(1, 4)]
    path.write_bytes(b"".join(lines[:2]) + lines[2][:-1])
    first = backend.records("mon", "execution", limit=1)
    assert first["has_more"] and first["next_cursor"] == len(lines[0])
    second = backend.records("mon", "execution", first["next_cursor"])
    assert not second["has_more"] and len(second["records"]) == 1
    assert second["next_cursor"] == len(lines[0] + lines[1])
    assert second["read_at"] >= first["read_at"] and not second["complete"]
    with path.open("ab") as stream:
        stream.write(b"\n")
    backend.sessions.clear()  # stop() removes active state but retains journal registration.
    third = backend.records("mon", "execution", second["next_cursor"])
    assert third["complete"] and third["records"][0]["inference_step"] == 3
    assert backend.records("mon", "execution", third["next_cursor"])["records"] == []


def test_journal_requires_registered_identity_and_byte_boundary(journal):
    backend, path = journal
    path.write_text('{"inference_step": 1}\n')
    with pytest.raises(KeyError):
        backend.records("../../file", "execution")
    with pytest.raises(ValueError):
        backend.records("mon", "other")
    for cursor in (-1, 1, 1000, True):
        with pytest.raises(ValueError):
            backend.records("mon", "execution", cursor)
    for limit in (0, 501, True, 1.5):
        with pytest.raises(ValueError):
            backend.records("mon", "execution", limit=limit)


def test_records_http_route_and_empty_journal(journal):
    backend, _ = journal
    client = TestClient(create_app(backend))
    path = "/monitors/mon/records?execution_id=execution"
    response = client.get(path)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["records"] == [] and data["next_cursor"] == 0
    assert client.get(path + "&cursor=3").status_code == 409
    assert client.get("/monitors/mon/records?execution_id=other").status_code == 409
    assert client.get("/monitors/unknown/records?execution_id=execution").status_code == 404
    assert client.get(path + "&limit=501").status_code == 409


def test_durable_images_survive_preview_eviction_stop_and_partial_tail(journal):
    backend, path = journal
    cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    lines = []
    from PIL import Image
    for step in range(1, 131):
        frames = {}
        folder = path.parent / f"capture_{step}"
        folder.mkdir()
        for i, camera in enumerate(cameras):
            target = folder / f"{camera}.png"
            Image.new("RGB", (3, 4), (step, i, 0)).save(target)
            frames[camera] = str(target)
        lines.append((json.dumps({"inference_step": step, "frames": frames}) + "\n").encode())
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:-1])
    backend.sessions.clear()
    client = TestClient(create_app(backend))
    for step in (1, 129):
        for camera in cameras:
            response = client.get(f"/monitors/mon/records/{step}/{camera}.png?execution_id=execution")
            assert response.status_code == 200 and response.headers["content-type"] == "image/png"
            assert response.content == (path.parent / f"capture_{step}" / f"{camera}.png").read_bytes()
    assert backend._record_image_index["mon"][0] == sum(map(len, lines[:-1]))
    assert client.get("/monitors/mon/records/130/cam_high.png?execution_id=execution").status_code == 404
    with path.open("ab") as stream:
        stream.write(b"\n")
    assert client.get("/monitors/mon/records/130/cam_high.png?execution_id=execution").status_code == 200
    assert not backend.sessions and not backend._preview_frames  # Read-only, no inference or preview registration.


def test_image_identity_and_registered_path_are_required(journal, tmp_path):
    backend, path = journal
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(b"private")
    path.write_text(json.dumps({"inference_step": 1, "frames": {"cam_high": str(outside)}}) + "\n")
    client = TestClient(create_app(backend))
    base = "/monitors/mon/records/1/cam_high.png"
    assert client.get(base + "?execution_id=other").status_code == 409
    assert client.get(base + "?execution_id=execution").status_code == 409
    assert client.get("/monitors/unknown/records/1/cam_high.png?execution_id=execution").status_code == 404
    for step, camera in ((0, "cam_high"), (1, "unknown")):
        assert client.get(f"/monitors/mon/records/{step}/{camera}.png?execution_id=execution").status_code == 409
    assert client.get("/monitors/mon/records/2/cam_high.png?execution_id=execution").status_code == 404


def test_image_disk_read_does_not_hold_inference_lock(journal, monkeypatch):
    backend, path = journal
    image = path.parent / "cam_high.png"
    image.write_bytes(b"png")
    path.write_text(json.dumps({"inference_step": 1, "frames": {"cam_high": str(image)}}) + "\n")
    read = Path.read_bytes
    def checked_read(target):
        acquired = []
        def check_lock():
            if backend._lock.acquire(timeout=.2):
                acquired.append(True)
                backend._lock.release()
        thread = threading.Thread(target=check_lock)
        thread.start()
        thread.join(1)
        assert acquired == [True]
        return read(target)
    monkeypatch.setattr(Path, "read_bytes", checked_read)
    assert backend.record_image("mon", "execution", 1, "cam_high") == b"png"

"""Read persisted GRM scores without model loading or advancing inference."""
import json
import threading
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

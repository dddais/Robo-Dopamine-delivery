#!/usr/bin/env python3
"""HTTP monitor service implementing the Robot Runtime monitor contract.

Serves deterministic connectivity checks or background GRM inference. The GRM
backend supports the original vLLM engine and the HF attention steering engine.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

from monitor_runtime.core import MONITOR_STATUS_FAIL, MONITOR_STATUS_RUNNING, MONITOR_STATUS_SUCCESS, MonitorSession, MonitorConflict

DEFAULT_GRM_MODEL_PATH = (
    "/home/ubuntu/dais/Robo-dopamine/pretrained_models/"
    "Robo-Dopamine-GRM-2.0-4B-Preview"
)
DEFAULT_GRM_GOAL_IMAGE = str(Path(__file__).resolve().parents[1] / "examples" / "blank_goal.png")

if TYPE_CHECKING:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse


class DeterministicMonitorBackend:
    def __init__(
        self,
        *,
        auto_success_after_polls: int = 0,
        observation_client: "RobotRuntimeObservationClient | None" = None,
    ) -> None:
        self.auto_success_after_polls = max(0, auto_success_after_polls)
        self.observation_client = observation_client
        self.sessions: dict[str, MonitorSession] = {}

    def start(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        execution_id = str(payload.get("execution_id") or "")
        subtask = str(payload.get("subtask") or "")
        if not monitor_id or not execution_id or not subtask:
            raise ValueError("monitor_id, execution_id, and subtask are required")
        session = MonitorSession(
            monitor_id=monitor_id,
            execution_id=execution_id,
            subtask=subtask,
            subtask_index=payload.get("subtask_index"),
            message="deterministic monitor backend",
            result={"provider": "deterministic"},
        )
        self.sessions[monitor_id] = session
        return session

    def status(self, payload: dict[str, Any]) -> MonitorSession:
        monitor_id = str(payload.get("monitor_id") or "")
        if monitor_id not in self.sessions:
            raise KeyError(f"unknown monitor_id: {monitor_id}")
        session = self.sessions[monitor_id]
        expected_execution_id = payload.get("execution_id")
        if expected_execution_id and str(expected_execution_id) != session.execution_id:
            raise ValueError("monitor_id does not belong to execution_id")

        poll_count = session.poll_count + 1
        status = session.status
        progress = session.progress
        observation_result: dict[str, Any] = {}
        if self.observation_client is not None:
            try:
                observation_result = self.observation_client.fetch()
            except Exception as exc:
                observation_result = {"error": str(exc)}
        if self.auto_success_after_polls and poll_count >= self.auto_success_after_polls:
            status = MONITOR_STATUS_SUCCESS
            progress = 1.0
        refreshed = replace(
            session,
            status=status,
            progress=progress,
            poll_count=poll_count,
            updated_at=time.time(),
            result={
                **session.result,
                **({"observation": observation_result} if observation_result else {}),
            },
        )
        self.sessions[monitor_id] = refreshed
        return refreshed

    def stop(self, payload: dict[str, Any]) -> dict[str, Any]:
        monitor_id = str(payload.get("monitor_id") or "")
        session = self.sessions.get(monitor_id)
        if session is not None:
            self.sessions[monitor_id] = replace(
                session,
                status=MONITOR_STATUS_FAIL,
                message="monitor stopped",
                updated_at=time.time(),
            )
        return {"stopped": True, "monitor_id": monitor_id}

    def health(self) -> dict[str, Any]:
        return {
            "status": MONITOR_STATUS_RUNNING,
            "provider": "deterministic",
            "sessions": len(self.sessions),
            "auto_success_after_polls": self.auto_success_after_polls,
            "observation_client": self.observation_client.health() if self.observation_client else None,
        }


class RobotRuntimeObservationClient:
    """Pulls latest binary JPEG frames from Robot Runtime."""

    def __init__(self, *, runtime_url: str, timeout: float = 10.0) -> None:
        self.runtime_url = runtime_url.rstrip("/")
        self.cameras = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        self.timeout = timeout

    def fetch(self) -> dict[str, Any]:
        metadata = self._get_json("/observations/latest/metadata")
        endpoints = metadata.get("binary_endpoints") if isinstance(metadata, dict) else None
        if not isinstance(endpoints, dict):
            raise RuntimeError("robot runtime observation metadata did not include binary_endpoints")
        images: dict[str, dict[str, Any]] = {}
        for camera in self.cameras:
            path = endpoints.get(camera) or f"/observations/latest/{camera}.jpg"
            data, headers = self._get_bytes(str(path))
            images[camera] = {
                "bytes": len(data),
                "content_type": headers.get("Content-Type") or headers.get("content-type"),
                "frame_id": headers.get("X-Frame-Id") or headers.get("x-frame-id"),
                "timestamp": headers.get("X-Timestamp") or headers.get("x-timestamp"),
            }
        return {
            "runtime_url": self.runtime_url,
            "frame_id": metadata.get("frame_id"),
            "timestamp": metadata.get("timestamp"),
            "cameras": list(images),
            "images": images,
        }

    def health(self) -> dict[str, Any]:
        return {
            "runtime_url": self.runtime_url,
            "cameras": list(self.cameras),
            "transport": "http_binary_jpeg",
        }

    def _get_json(self, path: str) -> dict[str, Any]:
        data, _headers = self._get_bytes(path)
        payload = json.loads(data.decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        if isinstance(payload, dict):
            return payload
        raise RuntimeError("robot runtime returned non-object JSON")

    def _get_bytes(self, path: str) -> tuple[bytes, dict[str, str]]:
        request = urllib.request.Request(self.runtime_url + _safe_path(path), method="GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                headers = {key: value for key, value in response.headers.items()}
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"failed to fetch robot observation: {exc}") from exc
        return data, headers


def create_app(backend: DeterministicMonitorBackend | None = None) -> "FastAPI":
    from fastapi import FastAPI
    from starlette.concurrency import run_in_threadpool

    backend = backend or DeterministicMonitorBackend()
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        yield
        if hasattr(backend, "close"):
            await run_in_threadpool(backend.close)

    app = FastAPI(title="Robo-Dopamine Monitor Service", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return _ok(await run_in_threadpool(backend.health))

    @app.post("/monitors/start")
    async def monitors_start(body: dict[str, Any]):
        try:
            session = await run_in_threadpool(backend.start, body)
        except MonitorConflict as exc:
            return _fail(str(exc), status=409)
        except ValueError as exc:
            return _fail(str(exc), status=400)
        return _ok(session.to_dict())

    @app.post("/monitors/activate")
    async def monitors_activate(body: dict[str, Any]):
        if not hasattr(backend, "activate"):
            return _fail("backend does not support deferred inference", status=400)
        try:
            session = await run_in_threadpool(backend.activate, body)
        except KeyError as exc:
            return _fail(str(exc), status=404)
        except ValueError as exc:
            return _fail(str(exc), status=409)
        return _ok(session.to_dict())

    @app.post("/monitors/status")
    async def monitors_status(body: dict[str, Any]):
        try:
            session = await run_in_threadpool(backend.status, body)
        except KeyError as exc:
            return _fail(str(exc), status=404)
        except ValueError as exc:
            return _fail(str(exc), status=409)
        return _ok(session.to_dict())

    @app.post("/monitors/stop")
    async def monitors_stop(body: dict[str, Any]):
        return _ok(await run_in_threadpool(backend.stop, body))

    @app.get("/monitors/frames/{frame_set_id}/{camera}.png")
    async def monitor_frame(frame_set_id: str, camera: str):
        from fastapi.responses import Response

        if not hasattr(backend, "frame_image"):
            return _fail("backend does not provide inference images", status=404)
        try:
            data = await run_in_threadpool(backend.frame_image, frame_set_id, camera)
        except (KeyError, FileNotFoundError) as exc:
            return _fail(str(exc), status=404)
        return Response(data, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=60"})

    return app


def _ok(data: object = None) -> "JSONResponse":
    from fastapi.responses import JSONResponse

    return JSONResponse({"success": True, "data": data, "message": "ok"})


def _fail(message: str, *, status: int) -> "JSONResponse":
    from fastapi.responses import JSONResponse

    return JSONResponse({"success": False, "data": None, "message": message}, status_code=status)


def _safe_path(path: str) -> str:
    if not path.startswith("/") or "://" in path or ".." in path.split("/"):
        raise ValueError(f"unsafe robot runtime path: {path!r}")
    return path


def _load_config(path: str | None) -> dict[str, Any]:
    """Load YAML config. Returns {} when path is None/empty."""
    if not path:
        return {}
    import yaml

    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def _build_argparser(config: dict[str, Any]) -> argparse.ArgumentParser:
    """Build the argparser, using config values as defaults.

    CLI flags override the config: argparse sees config values as ``default``,
    so any flag the user actually passes wins, and unmentioned flags fall back
    to the config (or to the hard-coded default when the config omits the key).
    """
    def cfg(name: str, fallback: Any) -> Any:
        return config.get(name, fallback)

    parser = argparse.ArgumentParser(description="Robo-Dopamine monitor service")
    parser.add_argument("--host", default=cfg("host", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=cfg("port", 8877))
    parser.add_argument("--inference-engine", choices=("vllm", "hf"), default=cfg("inference_engine", "vllm"))
    parser.add_argument("--steering-config", default=cfg("steering_config", None))
    parser.add_argument("--dual-branch", action=argparse.BooleanOptionalAction,
                        default=cfg("dual_branch", False),
                        help="Run independent HF baseline and steering models on each observation.")
    parser.add_argument("--baseline-device", default=cfg("baseline_device", None),
                        help="Baseline model device; defaults to --device.")
    parser.add_argument("--progress-difference-threshold", type=float,
                        default=cfg("progress_difference_threshold", .20),
                        help="Fail immediately when the fused progress difference is strictly above this value [0, 1].")
    parser.add_argument("--difference-mode", choices=("absolute", "baseline_minus_steering", "steering_minus_baseline"),
                        default=cfg("difference_mode", "absolute"),
                        help="Direction of baseline/steering progress comparison.")
    parser.add_argument("--device", default=cfg("device", "cuda:0"))
    parser.add_argument("--max-new-tokens", type=int, default=cfg("max_new_tokens", 64))
    parser.add_argument("--hf-batch-size", type=int, default=cfg("hf_batch_size", 2),
                        help="Samples per HF generate call, independently in each branch; 1 enables serial comparison.")
    parser.add_argument("--output-root", default=cfg("output_root", None))
    parser.add_argument("--max-camera-skew-s", type=float, default=cfg("max_camera_skew_s", 0.25))
    parser.add_argument(
        "--backend",
        choices=("deterministic", "grm"),
        default=cfg("backend", "deterministic"),
        help="Monitor backend. 'grm' runs online Robo-Dopamine-GRM inference in a background loop.",
    )
    parser.add_argument(
        "--auto-success-after-polls",
        type=int,
        default=cfg("auto_success_after_polls", 0),
    )
    parser.add_argument(
        "--robot-runtime-url",
        default=cfg("robot_runtime_url", "<ip>:8767"),
        help="Robot Runtime URL used to fetch binary JPEG observations.",
    )
    parser.add_argument(
        "--observation-timeout",
        type=float,
        default=cfg("observation_timeout", 3.0),
    )

    # --- GRM backend options (only used when --backend grm) ---
    parser.add_argument(
        "--model-path",
        default=cfg("model_path", DEFAULT_GRM_MODEL_PATH),
        help="GRM checkpoint or HF model path.",
    )
    parser.add_argument(
        "--goal-image",
        default=cfg("goal_image", DEFAULT_GRM_GOAL_IMAGE),
        help="Goal/reference image for ref_end and backward mode. "
             "Defaults to examples/blank_goal.png (no target).",
    )
    parser.add_argument(
        "--fisheye-config",
        default=cfg("fisheye_config", None),
        help="Optional fisheye_process/config.yaml to undistort wrist cameras.",
    )
    parser.add_argument(
        "--no-backward",
        action="store_true",
        default=bool(cfg("no_backward", False)),
        help="Exclude backward mode from GRM inference and fused progress.",
    )
    parser.add_argument("--backward", dest="no_backward", action="store_false", default=argparse.SUPPRESS,
                        help="Enable backward mode, overriding no_backward in YAML.")
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=cfg("success_threshold", 0.60),
    )
    parser.add_argument(
        "--success-stable-steps",
        type=int,
        default=cfg("success_stable_steps", 5),
    )
    parser.add_argument(
        "--success-max-drift",
        type=float,
        default=cfg("success_max_drift", 0.02),
    )
    parser.add_argument(
        "--fail-stable-steps",
        type=int,
        default=cfg("fail_stable_steps", 8),
    )
    parser.add_argument(
        "--fail-min-progress",
        type=float,
        default=cfg("fail_min_progress", 0.01),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=cfg("interval", 1.0),
        help="Seconds between GRM inference steps (background cadence).",
    )
    parser.add_argument(
        "--local-rank",
        default=cfg("local_rank", None),
        help="Optional LOCAL_RANK value to set before loading GRM.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=cfg("cuda_visible_devices", None),
        help="Optional CUDA_VISIBLE_DEVICES value to set before loading GRM.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    # Two-phase parse: first peel off --config so we can use the YAML as the
    # source of argparse defaults; the second (real) parse then lets any
    # explicitly-passed CLI flag override the config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _remaining = pre.parse_known_args(argv)
    config = _load_config(known.config)
    if known.config:
        base = Path(known.config).expanduser().resolve().parent
        from grm_runtime.common import resolve_path
        for key in ("steering_config", "goal_image", "fisheye_config", "output_root"):
            if config.get(key):
                config[key] = resolve_path(config[key], base)

    parser = _build_argparser(config)
    allowed = {action.dest for action in parser._actions}
    if set(config) - allowed:
        raise ValueError(f"Unknown monitor config keys: {sorted(set(config) - allowed)}")
    parser.add_argument(
        "--config",
        default=known.config,
        help="Path to a YAML config file. CLI flags override values in the file.",
    )
    args = parser.parse_args(argv)
    if args.dual_branch and args.backend != "grm":
        parser.error("--dual-branch requires --backend grm")

    import uvicorn

    backend: DeterministicMonitorBackend | "GRMMonitorBackend"
    if args.backend == "grm":
        from monitor_runtime.grm_backend import (
            GRMMonitorBackend,
            init_fisheye_remap,
            VALID_MODES,
        )

        fisheye_remap = None
        if args.fisheye_config:
            fisheye_remap = init_fisheye_remap(args.fisheye_config)

        active_modes = [m for m in VALID_MODES if not (args.no_backward and m == "backward")]

        backend = GRMMonitorBackend(
            model_path=args.model_path,
            hf_batch_size=args.hf_batch_size,
            goal_image=args.goal_image,
            runtime_url=args.robot_runtime_url,
            observation_timeout=args.observation_timeout,
            fisheye_remap=fisheye_remap,
            active_modes=active_modes,
            interval=args.interval,
            local_rank=args.local_rank,
            cuda_visible_devices=args.cuda_visible_devices,
            success_threshold=args.success_threshold,
            success_stable_steps=args.success_stable_steps,
            success_max_drift=args.success_max_drift,
            fail_stable_steps=args.fail_stable_steps,
            fail_min_progress=args.fail_min_progress,
            inference_engine=args.inference_engine,
            steering_config=args.steering_config,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            output_root=args.output_root,
            max_camera_skew_s=args.max_camera_skew_s,
            dual_branch=args.dual_branch,
            baseline_device=args.baseline_device,
            progress_difference_threshold=args.progress_difference_threshold,
            difference_mode=args.difference_mode,
        )
    else:
        observation_client = None
        if args.robot_runtime_url:
            observation_client = RobotRuntimeObservationClient(
                runtime_url=args.robot_runtime_url,
                timeout=args.observation_timeout,
            )
        backend = DeterministicMonitorBackend(
            auto_success_after_polls=args.auto_success_after_polls,
            observation_client=observation_client,
        )

    uvicorn.run(
        create_app(backend),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

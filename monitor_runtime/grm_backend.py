"""Online GRM monitoring with immutable observations and transactional publication."""
from __future__ import annotations

import json
import math
import hashlib
import re
import shutil
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from monitor_runtime.core import (
    MONITOR_STATUS_FAIL,
    MONITOR_STATUS_RUNNING,
    MONITOR_STATUS_SUCCESS,
    MonitorState,
    MonitorSession,
    MonitorConflict,
    clamp,
    difference_exceeds_threshold,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL_PATH = (
    "/path/to/Robo-dopamine/pretrained_models/"
    "Robo-Dopamine-GRM-2.0-4B-Preview"
)
DEFAULT_GOAL_IMAGE = str(REPO_ROOT / "examples" / "blank_goal.png")

CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FISHEYE_KEYS = ("cam_left_wrist", "cam_right_wrist")
VALID_MODES = ("forward", "incremental", "backward")
DIFFERENCE_MODES = ("absolute", "baseline_minus_steering", "steering_minus_baseline")


def _safe_path(path: str) -> str:
    if not path.startswith("/") or "://" in path or ".." in path.split("/"):
        raise ValueError(f"unsafe robot runtime path: {path!r}")
    return path


# ---------------------------------------------------------------------------
# Fisheye undistortion (lazy, optional)
# ---------------------------------------------------------------------------

@dataclass
class FisheyeRemap:
    """Pre-computed fisheye -> pinhole remap table, or None when disabled."""

    map_x: np.ndarray
    map_y: np.ndarray
    interp: int
    border_mode: int
    border_value: int


def init_fisheye_remap(config_path: str) -> FisheyeRemap:
    """Build the fisheye remap table from a fisheye_process config.yaml.

    ``fisheye_process`` lives outside this repo; its ``convert`` module is
    imported on demand so the monitor service can run without it when
    undistortion is disabled.
    """
    config_file = Path(config_path).resolve()
    sys.path.insert(0, str(config_file.parent))
    from convert import (  # type: ignore[import-not-found]
        build_remap_table,
        compute_extrinsics,
        get_border_flag,
        get_interp_flag,
        load_config,
        load_pinhole_intrinsics,
    )

    cfg = load_config(str(config_file))
    config_dir = str(config_file.parent)
    load_pinhole_intrinsics(cfg, config_dir)
    compute_extrinsics(cfg, config_dir)
    cfg.setdefault("depth", {})
    cfg["depth"]["enabled"] = False

    map_x, map_y = build_remap_table(cfg)
    proc = cfg.get("processing", {})
    interp = get_interp_flag(proc.get("interpolation", "LINEAR"))
    border_mode = get_border_flag(proc.get("border_mode", "CONSTANT"))
    border_value = proc.get("border_value", 0)
    print(f"[GRM] Fisheye remap built: {cfg['pinhole']['image_width']}x"
          f"{cfg['pinhole']['image_height']}")
    return FisheyeRemap(
        map_x=map_x,
        map_y=map_y,
        interp=interp,
        border_mode=border_mode,
        border_value=border_value,
    )


def undistort_fisheye(img: np.ndarray, remap: FisheyeRemap) -> np.ndarray:
    return cv2.remap(
        img,
        remap.map_x,
        remap.map_y,
        remap.interp,
        borderMode=remap.border_mode,
        borderValue=(remap.border_value, remap.border_value, remap.border_value),
    )



from copy import deepcopy
from uuid import uuid4
from grm_runtime.common import file_sha, fingerprint, progress_step, target_queries
from grm_runtime.common import parse_score as strict_parse_score
from grm_runtime.config import load_steering


@dataclass
class ProgressTracker:
    prev_progress: dict[str, float] = field(default_factory=lambda: {m: 0.0 for m in VALID_MODES})
    counts: dict[str, int] = field(default_factory=lambda: {m: 0 for m in VALID_MODES})

    def reset(self):
        self.prev_progress = {m: 0.0 for m in VALID_MODES}
        self.counts = {m: 0 for m in VALID_MODES}

    def update(self, mode, score):
        stats = progress_step(mode, score, self.prev_progress[mode], self.counts[mode])
        self.prev_progress[mode] = stats['progress']
        self.counts[mode] += 1
        return stats


def parse_score(pred_text):
    # Keep the old public parser for vLLM clients; HF validates before publication.
    try:
        match = re.search(r"<score>(.*?)</score>", pred_text)
        if match:
            value = match[1].replace('%', '').strip()
        else:
            matches = re.findall(r"([+-]?\d+(?:\.\d+)?)\s*%", pred_text)
            value = matches[-1] if matches else '0'
        return clamp(float(value), -100., 100.) / 100.
    except Exception:
        return 0.0


def build_online_samples(
    task: str,
    step: int,
    ref_start: dict[str, str],
    ref_end_path: str,
    previous: dict[str, str],
    current: dict[str, str],
    modes: list[str],
) -> list[dict[str, Any]]:
    """Build the eight-image samples for all active modes at one step."""
    samples: list[dict[str, Any]] = []
    for mode in modes:
        if mode == "incremental":
            before = previous
            before_id = f"prev_{step - 1:06d}"
        elif mode == "forward":
            before = ref_start
            before_id = "start_000000"
        elif mode == "backward":
            before = {k: ref_end_path for k in CAMERA_KEYS}
            before_id = "goal"
        else:
            raise ValueError(f"Unknown eval mode: {mode}")

        samples.append(
            {
                "id": f"grm-{mode}-step_{step:06d}-{before_id}-af_{step:06d}",
                "task": task,
                "eval_mode": mode,
                "image": [
                    ref_start["cam_high"],
                    ref_end_path,
                    before["cam_high"],
                    before["cam_left_wrist"],
                    before["cam_right_wrist"],
                    current["cam_high"],
                    current["cam_left_wrist"],
                    current["cam_right_wrist"],
                ],
            }
        )
    return samples



@dataclass
class _SubtaskState:
    monitor_id: str
    execution_id: str
    subtask: str
    subtask_index: int | None = None
    queries: list[str] = field(default_factory=list)
    generation: str = field(default_factory=lambda: uuid4().hex)
    created_at: float = field(default_factory=time.time)
    ref_start: dict | None = None
    previous: dict | None = None
    last_observation: str | None = None
    capture_index: int = 0
    step: int = 0
    tracker: ProgressTracker = field(default_factory=ProgressTracker)
    baseline_tracker: ProgressTracker = field(default_factory=ProgressTracker)
    monitor: MonitorState = field(default_factory=MonitorState)
    latest: dict = field(default_factory=dict)
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    defer_inference: bool = False
    inference_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    tracking_stream: Any = None


class GRMMonitorBackend:
    def __init__(self, *, model_path=DEFAULT_MODEL_PATH, goal_image=DEFAULT_GOAL_IMAGE,
                 runtime_url, observation_timeout=3.0, fisheye_remap=None, active_modes=None,
                 interval=1.0, local_rank=None, cuda_visible_devices=None,
                 success_threshold=.60, success_stable_steps=5, success_max_drift=.02,
                 fail_stable_steps=8, fail_min_progress=.01, inference_engine='vllm',
                 steering_config=None, device='cuda:0', max_new_tokens=64,
                 output_root=None, model=None, max_camera_skew_s=.25,
                 dual_branch=False, baseline_device=None, baseline_model=None,
                 progress_difference_threshold=.20, difference_mode='absolute', hf_batch_size=2,
                 tracking_config=None):
        if not runtime_url.startswith(('http://','https://')):
            raise ValueError('robot_runtime_url must include http:// or https://')
        self.runtime_url = runtime_url.rstrip('/')
        self.model_path = model_path
        self.goal_image = goal_image
        self.observation_timeout = observation_timeout
        self.fisheye_remap = fisheye_remap
        self.preprocess_fingerprint = fingerprint({'fisheye': None if fisheye_remap is None else {
            'map_x':hashlib.sha256(fisheye_remap.map_x.tobytes()).hexdigest(),
            'map_y':hashlib.sha256(fisheye_remap.map_y.tobytes()).hexdigest(),
            'interpolation':fisheye_remap.interp,'border_mode':fisheye_remap.border_mode,
            'border_value':fisheye_remap.border_value}})
        self.active_modes = list(VALID_MODES if active_modes is None else active_modes)
        if not self.active_modes or len(set(self.active_modes)) != len(self.active_modes) or set(self.active_modes)-set(VALID_MODES):
            raise ValueError('Invalid active_modes')
        self.interval = max(.1,float(interval))
        self.max_camera_skew_s = max_camera_skew_s
        self.inference_engine = inference_engine
        if isinstance(hf_batch_size, bool) or not isinstance(hf_batch_size, int) or hf_batch_size < 1:
            raise ValueError('hf_batch_size must be a positive integer')
        self.hf_batch_size = hf_batch_size
        self.steering = load_steering(steering_config)
        from grm_runtime.common import load_yaml
        self.tracking_config = load_yaml(tracking_config, {'enabled', 'poll_interval_s', 'max_frame_age_s'})[0] if tracking_config else {}
        self.tracking_config.setdefault('enabled', False)
        if not isinstance(self.tracking_config['enabled'], bool):
            raise ValueError('tracking.enabled must be boolean')
        for key, default in (('poll_interval_s', .1), ('max_frame_age_s', 2.)):
            value = self.tracking_config.setdefault(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'tracking.{key} must be finite and positive')
        if self.tracking_config['enabled'] and (inference_engine != 'hf' or not self.steering['enabled']):
            raise ValueError('Tracking requires HF and enabled attention steering')
        self.tracking_cameras = [c for c in CAMERA_KEYS if 'after_' + c in self.steering.get('intervention_labels', [])]
        if self.tracking_config['enabled'] and not self.tracking_cameras:
            raise ValueError('Tracking requires at least one AFTER camera intervention label')
        if self.steering['enabled'] and inference_engine != 'hf':
            raise ValueError('Attention steering requires inference_engine=hf')
        if not isinstance(dual_branch, bool):
            raise ValueError('dual_branch must be boolean')
        if dual_branch and (inference_engine != 'hf' or not self.steering['enabled']):
            raise ValueError('dual_branch requires inference_engine=hf and enabled steering_config')
        if difference_mode not in DIFFERENCE_MODES:
            raise ValueError(f'difference_mode must be one of {DIFFERENCE_MODES}')
        if (not isinstance(progress_difference_threshold, (int, float))
                or isinstance(progress_difference_threshold, bool)
                or not math.isfinite(progress_difference_threshold)
                or not 0 <= progress_difference_threshold <= 1):
            raise ValueError('progress_difference_threshold must be finite and in [0, 1]')
        if baseline_model is not None and not dual_branch:
            raise ValueError('baseline_model requires dual_branch')
        self.dual_branch = dual_branch
        self.device = device
        self.baseline_device = baseline_device or device
        self.difference_mode = difference_mode
        self.progress_difference_threshold = float(progress_difference_threshold)
        if success_stable_steps < 1 or fail_stable_steps < 2:
            raise ValueError('Invalid monitor stability windows')
        self.monitor_options = dict(success_threshold=success_threshold, success_stable_steps=success_stable_steps,
            success_max_drift=success_max_drift, fail_stable_steps=fail_stable_steps, fail_min_progress=fail_min_progress,
            progress_difference_threshold=self.progress_difference_threshold if dual_branch else None)
        self._ref_end_path = str(Path(goal_image).expanduser().resolve())
        if not Path(self._ref_end_path).is_file():
            raise FileNotFoundError(self._ref_end_path)
        root = Path(output_root) if output_root else REPO_ROOT/'results'/'monitor_sessions'
        self._cache_root = root.resolve() / (time.strftime('%y-%m-%d-%H-%M-%S')+'_'+uuid4().hex[:8])
        self._cache_root.mkdir(parents=True,exist_ok=True)
        frozen_goal = self._cache_root / ('reference_end' + Path(self._ref_end_path).suffix)
        shutil.copyfile(self._ref_end_path, frozen_goal)
        self._ref_end_path = str(frozen_goal)
        self._lock = threading.RLock()
        self._infer_lock = threading.Lock()
        self.sessions = {}
        # Keep registered journals readable after /monitors/stop. Recording
        # clients drain by byte cursor without polling/advancing inference.
        self._record_journals = {}
        self._record_image_index = {}
        self._record_image_lock = threading.Lock()
        # Only committed GRM inputs are published. Keep URLs usable after stop
        # so the operator can inspect the final score; files remain run artifacts.
        self._preview_frames = {}
        if model is None:
            from examples.inference import GRMInference
            model = GRMInference(model_path, local_rank=local_rank, cuda_visible_devices=cuda_visible_devices,
                engine=inference_engine, steering_config=steering_config, device=device, max_new_tokens=max_new_tokens,
                hf_batch_size=hf_batch_size)
        self.model = model
        if dual_branch:
            if baseline_model is None:
                from examples.inference import GRMInference
                baseline_model = GRMInference(model_path, engine='hf', steering_config=None,
                    device=self.baseline_device, max_new_tokens=max_new_tokens, hf_batch_size=hf_batch_size)
            steering_runtime = getattr(model, 'backend', None) or model
            baseline_runtime = getattr(baseline_model, 'backend', None) or baseline_model
            if (baseline_runtime is steering_runtime
                    or getattr(baseline_runtime, 'model', baseline_runtime)
                    is getattr(steering_runtime, 'model', steering_runtime)):
                raise ValueError('Dual branches must use independent model instances')
        self.baseline_model = baseline_model

    def _dual_branch_options(self):
        return {'enabled': self.dual_branch, 'metric': 'fused_progress',
                'difference_mode': self.difference_mode,
                'threshold': self.progress_difference_threshold,
                'steering_device': self.device, 'baseline_device': self.baseline_device}

    def _difference(self, baseline, steering):
        delta = baseline - steering
        if self.difference_mode == 'absolute':
            return abs(delta)
        return delta if self.difference_mode == 'baseline_minus_steering' else -delta

    @staticmethod
    def _infer_branch(model, samples, branch):
        try:
            return model.inference_batch(samples)
        except Exception as exc:
            raise RuntimeError(f'{branch} branch inference failed: {exc}') from exc

    def _branch_results(self, samples, outputs, previous_tracker, branch):
        if len(outputs) != len(samples) or {item.get('id') for item in outputs} != {s['id'] for s in samples}:
            raise RuntimeError(f'{branch} GRM output IDs/count differ from input samples')
        expected_modes = {sample['id']: sample['eval_mode'] for sample in samples}
        if any(item.get('eval_mode') != expected_modes[item['id']] for item in outputs):
            raise RuntimeError(f'{branch} GRM output modes do not match input IDs')
        by_mode = {item.get('eval_mode'): item for item in outputs}
        if set(by_mode) != set(self.active_modes):
            raise RuntimeError(f'{branch} GRM output modes incomplete')
        tracker = deepcopy(previous_tracker)
        results = {}
        for mode in self.active_modes:
            item = by_mode[mode]
            if self.inference_engine == 'hf':
                if not item.get('valid', False):
                    raise RuntimeError(f"Invalid {branch} {mode} score: {item.get('pred')}")
                try:
                    score = strict_parse_score(item['pred'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"Invalid {branch} {mode} score: {item.get('pred')}") from exc
            else:
                score = parse_score(item.get('pred', ''))
            results[mode] = {**tracker.update(mode, score), 'pred': item['pred'],
                             'steering': item.get('steering', {})}
        fused = clamp(sum(v['progress'] for v in results.values()) / len(results), 0., 1.)
        return tracker, results, fused

    def _fetch_bytes(self, path):
        request = urllib.request.Request(self.runtime_url+_safe_path(path),method='GET')
        with urllib.request.urlopen(request,timeout=self.observation_timeout) as response:
            return response.read(), dict(response.headers)

    def _fetch_json(self,path):
        data,_ = self._fetch_bytes(path)
        value=json.loads(data)
        if isinstance(value,dict) and isinstance(value.get('data'),dict):
            value=value['data']
        if not isinstance(value,dict):
            raise ValueError('Observation metadata must be an object')
        return value

    def _session_dir(self,state):
        return self._cache_root/state.generation

    def _snapshot_current(self, state, *, reference=False):
        requested_at = time.time()
        metadata=self._fetch_json('/observations/latest/metadata')
        endpoints=metadata.get('binary_endpoints')
        if not isinstance(endpoints,dict):
            raise RuntimeError('observation metadata missing binary_endpoints')
        capture=state.capture_index
        state.capture_index+=1
        folder=self._session_dir(state)/('reference' if reference else f'capture_{capture:06d}')
        folder.mkdir(parents=True,exist_ok=True)
        images, cameras = {}, {}
        try:
            for camera in CAMERA_KEYS:
                data, headers=self._fetch_bytes(endpoints.get(camera) or f'/observations/latest/{camera}.jpg')
                img=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f'Cannot decode {camera}')
                if camera in FISHEYE_KEYS and self.fisheye_remap is not None:
                    img=undistort_fisheye(img,self.fisheye_remap)
                path=folder/f'{camera}.png'
                if not cv2.imwrite(str(path),img):
                    raise RuntimeError(f'Cannot write {path}')
                images[camera]=str(path)
                lower={k.lower():v for k,v in headers.items()}
                cameras[camera]={'frame_id':lower.get('x-frame-id'), 'timestamp':lower.get('x-timestamp'),
                                 'image_sha256':file_sha(path)}
            timestamps=[float(v['timestamp']) for v in cameras.values() if v['timestamp'] is not None]
            if any(not math.isfinite(ts) for ts in timestamps):
                raise RuntimeError('Non-finite camera timestamp')
            if timestamps and len(timestamps)==3 and max(timestamps)-min(timestamps)>self.max_camera_skew_s:
                raise RuntimeError('Camera timestamps exceed allowed skew')
            identity=fingerprint({c:(v['frame_id'] or v['image_sha256']) for c,v in cameras.items()})
            return images, {'cameras':cameras,'frame_id':metadata.get('frame_id'),
                            'timestamp':metadata.get('timestamp'), 'identity':identity,
                            'snapshot_requested_at':requested_at, 'snapshot_received_at':time.time(),
                            'synchronization_verified':len(timestamps)==3,
                            'preprocess_fingerprint':self.preprocess_fingerprint,
                            'fisheye_enabled':self.fisheye_remap is not None}
        except Exception:
            shutil.rmtree(folder,ignore_errors=True)
            raise

    def _is_active(self,state):
        return not state.stop_event.is_set() and self.sessions.get(state.monitor_id) is state

    def _run_one_step(self,state):
        started=time.monotonic()
        online_grounding = None
        if state.tracking_stream is not None:
            snapshot = state.tracking_stream.read()
            if snapshot is None:
                return None
            current, observation, online_grounding = snapshot
        else:
            current,observation=self._snapshot_current(state)
        observation_ms=(time.monotonic()-started)*1000
        if observation['identity']==state.last_observation:
            shutil.rmtree(Path(next(iter(current.values()))).parent,ignore_errors=True)
            return None
        try:
            samples=build_online_samples(state.subtask,state.step,state.ref_start,self._ref_end_path,state.previous,current,self.active_modes)
            for sample in samples:
                if state.queries:
                    sample['target_queries']=state.queries
                if online_grounding is not None:
                    sample['online_grounding'] = deepcopy(online_grounding)
            queue_start=time.monotonic()
            with self._infer_lock:
                queue_ms=(time.monotonic()-queue_start)*1000
                with self._lock:
                    if not self._is_active(state):
                        return None
                if self.dual_branch:
                    steering_samples = [{**deepcopy(s), 'condition': 'candidate_target'} for s in samples]
                    baseline_samples = [{**deepcopy(s), 'condition': 'baseline'} for s in samples]
                    # Wait for BOTH branches even on error, before releasing the lock or deleting frames.
                    with ThreadPoolExecutor(max_workers=2, thread_name_prefix='grm-branch') as pool:
                        steering_future = pool.submit(self._infer_branch, self.model, steering_samples, 'steering')
                        baseline_future = pool.submit(self._infer_branch, self.baseline_model, baseline_samples, 'baseline')
                        outputs = steering_future.result()
                        baseline_outputs = baseline_future.result()
                else:
                    outputs=self.model.inference_batch(samples)
            tracker,mode_results,fused=self._branch_results(samples,outputs,state.tracker,'steering')
            monitor=deepcopy(state.monitor)
            branch_fields = {}
            timing_modes = list(mode_results.values())
            if self.dual_branch:
                baseline_tracker,baseline_results,baseline_fused=self._branch_results(
                    samples,baseline_outputs,state.baseline_tracker,'baseline')
                difference = self._difference(baseline_fused, fused)
                status = monitor.update(fused, progress_difference=difference)
                exceeded = difference_exceeds_threshold(difference, self.progress_difference_threshold)
                branch_fields = {
                    'branches': {'steering': {'progress': fused, 'modes': mode_results},
                                 'baseline': {'progress': baseline_fused, 'modes': baseline_results}},
                    'comparison': {'metric': 'fused_progress', 'difference_mode': self.difference_mode,
                                   'difference': difference, 'threshold': self.progress_difference_threshold,
                                   'threshold_exceeded': exceeded,
                                   'modes': {mode: {
                                       'score_difference': self._difference(baseline_results[mode]['score'], mode_results[mode]['score']),
                                       'progress_difference': self._difference(baseline_results[mode]['progress'], mode_results[mode]['progress'])}
                                       for mode in self.active_modes}}}
                if exceeded:
                    branch_fields['failure_reason'] = 'branch_difference_exceeded'
                timing_modes.extend(baseline_results.values())
            else:
                status=monitor.update(fused)
            now=time.time()
            if 'snapshot_requested_at' in observation:
                observation['input_age_at_publish_s'] = now-observation['snapshot_requested_at']
            record={'step':state.step,'inference_step':state.step+1,'progress':fused,'fused':fused,
                'progress_percent':fused*100,'status':status,'modes':mode_results,'frames':current,
                'subtask':state.subtask,'subtask_idx':state.subtask_index or 0,
                'inference_updated_at':now,'observation':observation,'engine':self.inference_engine,
                'continuous_monitoring':state.monitor.continuous_monitoring,
                'latency_s':time.monotonic()-started,
                'timing':{'observation_ms':observation_ms,'queue_wait_ms':queue_ms,
                          'prepare_ms':sum(v['steering'].get('prepare_ms',0) for v in timing_modes),
                          'grounding_ms':sum(v['steering'].get('grounding_ms',0) for v in timing_modes),
                          'grm_ms':sum(v['steering'].get('grm_ms',0) for v in timing_modes),
                          'total_ms':(time.monotonic()-started)*1000}, **branch_fields}
            frame_set_id = uuid4().hex
            record['preview'] = {'frame_set_id': frame_set_id, 'cameras': list(CAMERA_KEYS),
                                 'kind': 'grm_after'}
            with self._lock:
                if not self._is_active(state):
                    return None
                # Timestamp publication under the same lock as journal reads so
                # read_at is a watermark for all previously published records.
                record['inference_updated_at'] = time.time()
                if 'snapshot_requested_at' in observation:
                    observation['input_age_at_publish_s'] = record['inference_updated_at'] - observation['snapshot_requested_at']
                # Persist before committing; a disk failure must not half-advance the trajectory.
                with (self._session_dir(state)/'online_pred.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                self._preview_frames[frame_set_id] = dict(current)
                while len(self._preview_frames) > 128:
                    self._preview_frames.pop(next(iter(self._preview_frames)))
                state.tracker,state.monitor=tracker,monitor
                if self.dual_branch:
                    state.baseline_tracker=baseline_tracker
                state.previous=current
                state.last_observation=observation['identity']
                state.step+=1
                state.latest=record
                state.error=None
            print(f"[GRM] {state.monitor_id} step={record['step']} progress={fused:.3f} [{status}] "
                  f"latency={record['latency_s']:.3f}s",flush=True)
            return record
        finally:
            # Preserve committed frames and reference for reproducible session replay only.
            if state.previous != current:
                shutil.rmtree(Path(next(iter(current.values()))).parent,ignore_errors=True)

    def _inference_loop(self,state):
        try:
            while not state.stop_event.is_set() and state.ref_start is None:
                try:
                    reference,observation=self._snapshot_current(state,reference=True)
                    stream = None
                    if self.tracking_config['enabled']:
                        from monitor_runtime.tracking import LatestTrackedFrames
                        from grm_runtime.grounding import GroundingClient
                        stream = LatestTrackedFrames(capture=lambda: self._snapshot_current(state),
                            client=GroundingClient(**self.steering.get('grounding', {})),
                            cameras=self.tracking_cameras, queries=state.queries,
                            directory=self._session_dir(state), session_id=state.generation,
                            poll_interval_s=self.tracking_config['poll_interval_s'],
                            max_frame_age_s=self.tracking_config['max_frame_age_s'])
                    with self._lock:
                        if not self._is_active(state):
                            return
                        state.ref_start=reference
                        state.previous=reference
                        state.last_observation=observation['identity']
                        state.tracking_stream=stream
                        state.error=None
                    if stream:
                        stream.start()
                except Exception as exc:
                    with self._lock:
                        state.error=str(exc)
                    state.stop_event.wait(self.interval)
            while not state.stop_event.is_set() and not state.monitor.is_finished:
                if not state.inference_event.is_set():
                    state.stop_event.wait(0.05)
                    continue
                try:
                    self._run_one_step(state)
                except Exception as exc:
                    with self._lock:
                        state.error=str(exc)
                    print(f'[GRM] {state.monitor_id}: {exc}',flush=True)
                state.stop_event.wait(self.interval)
        finally:
            # Completed observations/logs intentionally remain as run artifacts; no model hooks remain installed.
            if state.tracking_stream is not None:
                state.tracking_stream.close()

    def start(self,payload):
        mid=str(payload.get('monitor_id') or '')
        execution=str(payload.get('execution_id') or '')
        task=str(payload.get('subtask') or '')
        if not mid or not execution or not task:
            raise ValueError('monitor_id, execution_id, and subtask are required')
        deferred = payload.get('defer_inference', False)
        if not isinstance(deferred, bool):
            raise ValueError('defer_inference must be boolean')
        continuous = payload.get('continuous_monitoring', False)
        if type(continuous) is not bool:
            raise ValueError('continuous_monitoring must be boolean')
        queries=target_queries(task,payload.get('target_queries'),self.steering.get('task_queries')) if self.steering['enabled'] else []
        with self._lock:
            previous=self.sessions.get(mid)
            if previous is not None:
                if (previous.execution_id,previous.subtask,previous.queries,previous.subtask_index,previous.defer_inference,previous.monitor.continuous_monitoring)!=(execution,task,queries,payload.get('subtask_index'),deferred,continuous):
                    raise MonitorConflict('monitor_id already belongs to a different request')
                return self.status({'monitor_id':mid})
            state=_SubtaskState(mid,execution,task,subtask_index=payload.get('subtask_index'),queries=queries,
                                monitor=MonitorState(**self.monitor_options, continuous_monitoring=continuous), defer_inference=deferred)
            if not deferred:
                state.inference_event.set()
            directory=self._session_dir(state);directory.mkdir(parents=True)
            runtime=getattr(getattr(self.model,'backend',self.model),'manifest',{})
            branch_manifest = {}
            if self.dual_branch:
                baseline_runtime = getattr(self.baseline_model, 'backend', None) or self.baseline_model
                branch_manifest = {'dual_branch': self._dual_branch_options(),
                                   'baseline_runtime': getattr(baseline_runtime, 'manifest', {})}
            (directory/'manifest.json').write_text(json.dumps({'monitor_id':mid,'execution_id':execution,'subtask':task,
                'target_queries':queries,'defer_inference':deferred,'continuous_monitoring':continuous,'generation':state.generation,'runtime':runtime,'active_modes':self.active_modes,
                'monitor_options':self.monitor_options,'tracking':self.tracking_config, **branch_manifest},indent=2))
            self.sessions[mid]=state
            self._record_journals[mid] = (execution, state.generation, directory, task)
            state.thread=threading.Thread(target=self._inference_loop,args=(state,),daemon=True,name=f'grm-{state.generation}')
            state.thread.start()
            return self.status({'monitor_id':mid})

    def status(self,payload):
        with self._lock:
            state=self.sessions.get(str(payload.get('monitor_id') or ''))
            if state is None:
                raise KeyError('unknown monitor_id')
            if payload.get('execution_id') and str(payload['execution_id'])!=state.execution_id:
                raise ValueError('monitor_id does not belong to execution_id')
            latest=deepcopy(state.latest)
            updated=latest.get('inference_updated_at',state.created_at)
            tracking = state.tracking_stream.status() if state.tracking_stream is not None else None
            result={'provider':'grm','warming_up':state.ref_start is None or bool(
                        tracking and not tracking['ready'] and not state.monitor.is_finished),
                    'inference_enabled':state.inference_event.is_set(),**latest,
                    'continuous_monitoring':state.monitor.continuous_monitoring,
                    'result_age_s':time.time()-updated,'session_dir':str(self._session_dir(state))}
            if tracking:
                result['tracking'] = tracking
                if tracking['error']:
                    result['error'] = tracking['error']
            if state.error:
                result['error']=state.error
            if state.monitor.is_finished:
                result.update(final_status=state.monitor.status,progress_history=list(state.monitor.progress_history))
            return MonitorSession(state.monitor_id,state.execution_id,state.subtask,state.subtask_index,
                status='running' if state.monitor.continuous_monitoring else state.monitor.status,progress=latest.get('progress',0.),created_at=state.created_at,
                updated_at=updated,error=state.error,poll_count=state.step,result=result,
                message='grm monitor backend')

    def records(self, monitor_id, execution_id, cursor=0, limit=100):
        if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError('invalid journal cursor or limit')
        with self._lock:
            entry = self._record_journals.get(monitor_id)
            if entry is None:
                raise KeyError('unknown monitor journal')
            execution, generation, directory, task = entry
            if execution_id != execution:
                raise ValueError('monitor journal does not belong to execution_id')
            complete = monitor_id not in self.sessions or self.sessions[monitor_id].monitor.is_finished
            path = directory / 'online_pred.jsonl'
            rows, next_cursor, more = [], cursor, False
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(0, 2)
                    if cursor > stream.tell():
                        raise ValueError('journal cursor exceeds file size')
                    if cursor:
                        stream.seek(cursor - 1)
                        if stream.read(1) != b'\n':
                            raise ValueError('journal cursor must be at a record boundary')
                    stream.seek(cursor)
                    for _ in range(limit):
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            break  # A writer may still be appending this record.
                        rows.append(json.loads(line))
                        next_cursor = stream.tell()
                    more = bool(stream.readline().endswith(b'\n'))
            elif cursor:
                raise ValueError('journal is empty')
            return {'monitor_id': monitor_id, 'execution_id': execution, 'generation': generation,
                    'subtask': task, 'session_dir': str(directory), 'records': rows,
                    'next_cursor': next_cursor, 'has_more': more, 'complete': complete,
                    'read_at': time.time()}

    def activate(self, payload):
        with self._lock:
            session = self.status(payload)
            state = self.sessions[session.monitor_id]
            if state.ref_start is None:
                raise ValueError('reference is not ready')
            if state.tracking_stream is not None and not state.tracking_stream.status()['ready']:
                raise ValueError('tracking reference is not ready')
            state.inference_event.set()
            return self.status(payload)

    def record_image(self, monitor_id, execution_id, inference_step, camera):
        """Read a persisted scoring image even after preview-cache eviction."""
        if type(inference_step) is not int or inference_step < 1 or camera not in CAMERA_KEYS:
            raise ValueError('invalid inference step or camera')
        with self._lock:
            entry = self._record_journals.get(monitor_id)
            if entry is None:
                raise KeyError('unknown monitor journal')
            execution, _, directory, _ = entry
            if execution_id != execution:
                raise ValueError('monitor journal does not belong to execution_id')
        # Scanning and image reads do not hold the inference publication lock.
        # Each appended JSONL line is indexed once, across all camera requests.
        with self._record_image_lock:
            cursor, frames = self._record_image_index.setdefault(monitor_id, [0, {}])
            if inference_step not in frames:
                with (directory / 'online_pred.jsonl').open('rb') as stream:
                    stream.seek(cursor)
                    while line := stream.readline():
                        if not line.endswith(b'\n'):
                            break
                        row = json.loads(line)
                        frames[row['inference_step']] = row['frames']
                        self._record_image_index[monitor_id][0] = stream.tell()
            path = frames.get(inference_step, {}).get(camera)
            if path is None:
                raise KeyError('unknown scoring image')
        path = Path(path).resolve()
        if not path.is_relative_to(directory.resolve()) or path.suffix != '.png':
            raise ValueError('scoring image is outside its registered journal')
        return path.read_bytes()

    def frame_image(self, frame_set_id, camera):
        # Resolve opaque registered IDs, never a caller-supplied filesystem path.
        with self._lock:
            frames = self._preview_frames.get(frame_set_id)
            if frames is None or camera not in CAMERA_KEYS:
                raise KeyError('unknown or expired inference frame')
            path = frames[camera]
        return Path(path).read_bytes()

    def stop(self,payload):
        with self._lock:
            state=self.sessions.pop(str(payload.get('monitor_id') or ''),None)
            if state:
                state.stop_event.set()
        if state and state.tracking_stream is not None:
            state.tracking_stream.close()
        if state and state.thread and state.thread.is_alive():
            state.thread.join(timeout=5.)
        return {'stopped':True,'monitor_id':payload.get('monitor_id'),
                'worker_stopping':bool(state and state.thread and state.thread.is_alive())}

    def close(self):
        for mid in list(self.sessions):
            self.stop({'monitor_id':mid})

    def health(self):
        with self._lock:
            return {'status':'running','provider':'grm','model':self.model_path,'runtime_url':self.runtime_url,
                    'engine':self.inference_engine,'steering_enabled':self.steering['enabled'],
                    'profile_fingerprint':self.steering.get('profile_sha256'),'sessions':len(self.sessions),
                    'dual_branch':self._dual_branch_options(),
                    'interval':self.interval,'hf_batch_size':self.hf_batch_size,
                    'tracking':self.tracking_config,
                    'active_modes':self.active_modes,'cameras':list(CAMERA_KEYS)}

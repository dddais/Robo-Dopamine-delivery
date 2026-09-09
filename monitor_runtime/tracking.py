"""Independent capture/tracking worker with a single latest-result slot."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time

from PIL import Image

from grm_runtime.common import file_sha


class LatestTrackedFrames:
    def __init__(self, *, capture, client, cameras, queries, directory, session_id,
                 poll_interval_s=.1, max_frame_age_s=2.):
        self.capture, self.client = capture, client
        self.cameras, self.queries = tuple(cameras), list(queries)
        self.directory, self.session_id = Path(directory), session_id
        self.poll_interval_s, self.max_frame_age_s = poll_interval_s, max_frame_age_s
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._latest, self.error, self.count = None, None, 0
        self._thread = threading.Thread(target=self._run, daemon=True, name=f'tracking-{session_id}')

    def start(self):
        self._thread.start()

    def status(self):
        with self._lock:
            return {'enabled': True, 'ready': self._latest is not None, 'frames_processed': self.count,
                'latest_age_s': time.monotonic()-self._latest['captured_monotonic'] if self._latest else None,
                'error': self.error}

    def read(self):
        """Pin the newest complete set. Deleting the slot cannot delete these links."""
        with self._lock:
            if self._latest is None:
                if self.error:
                    raise RuntimeError(f'Tracking unavailable: {self.error}')
                return None
            age = time.monotonic()-self._latest['captured_monotonic']
            if age > self.max_frame_age_s:
                raise RuntimeError(f'Latest tracked frame is stale ({age:.2f}s); {self.error or "no new frame"}')
            folder = Path(tempfile.mkdtemp(prefix='score_', dir=self.directory))
            try:
                frames = {}
                for camera, path in self._latest['frames'].items():
                    target = folder / f'{camera}.png'
                    try:
                        os.link(path, target)
                    except OSError:
                        shutil.copyfile(path, target)
                    frames[camera] = str(target)
                observation = deepcopy(self._latest['observation'])
                observation['tracking']['input_age_at_read_s'] = age
                return frames, observation, deepcopy(self._latest['grounding'])
            except Exception:
                shutil.rmtree(folder, ignore_errors=True)
                raise

    @staticmethod
    def _remove(frames):
        if frames:
            shutil.rmtree(Path(next(iter(frames.values()))).parent, ignore_errors=True)

    def _run(self):
        last_identity = None
        try:
            health = self.client._request('/health')
            if not health.get('tracking_enabled'):
                raise RuntimeError('SAM3 service has tracking disabled; use configs/sam3_tracker.yaml')
            while not self._stop.is_set():
                start, frames = time.monotonic(), None
                try:
                    frames, observation = self.capture()
                    if self._stop.is_set():
                        break
                    if observation['identity'] == last_identity:
                        continue
                    grounding, errors = {}, []
                    for camera in self.cameras:
                        if self._stop.is_set():
                            break
                        path = frames[camera]
                        try:
                            result = self.client.track(path, self.queries, f'{self.session_id}:{camera}')
                        except Exception as exc:
                            # No old bbox and no synchronous detector retry inside GRM.
                            # The existing on_missing_bbox policy handles this explicit result.
                            errors.append(str(exc))
                            with Image.open(path) as image:
                                result = {'image_size': list(image.size)}
                            result.update(image_sha256=file_sha(path), coordinate_space='input_image_xyxy',
                                status='no_detection', selected=None, candidates=[],
                                selection_status='tracking_error', source='sam3_tracker', error=str(exc))
                        grounding['after_' + camera] = result
                    if self._stop.is_set():
                        break
                    now = time.time()
                    observation['tracking'] = {'ready_at': now, 'cycle_ms': (time.monotonic()-start)*1000,
                                               'grounding_sources': {k:v['source'] for k,v in grounding.items()}}
                    with self._lock:
                        previous = self._latest
                        self._latest = {'frames': frames, 'observation': observation,
                            'grounding': grounding, 'captured_monotonic': start}
                        self.count += 1
                        self.error = '; '.join(errors) if errors else None
                    last_identity = observation['identity'] if not errors else None
                    frames = None
                    if previous:
                        self._remove(previous['frames'])
                except Exception as exc:
                    with self._lock:
                        self.error = str(exc)
                finally:
                    self._remove(frames)
                    self._stop.wait(max(0., self.poll_interval_s-(time.monotonic()-start)))
        except Exception as exc:
            with self._lock:
                self.error = str(exc)
        finally:
            for camera in self.cameras:
                try:
                    self.client.close_track(f'{self.session_id}:{camera}')
                except Exception:
                    pass  # Server TTL also releases abandoned/expired sessions.
            with self._lock:
                latest, self._latest = self._latest, None
            if latest:
                self._remove(latest['frames'])

    def close(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.)

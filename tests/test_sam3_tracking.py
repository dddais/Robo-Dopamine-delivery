"""Session isolation, loss/reacquisition, exact-frame handoff and bounded buffering."""
from copy import deepcopy
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from grm_runtime.common import file_sha
from grm_runtime.grounding import AlignmentError, GroundingClient, validate_grounding_result
from monitor_runtime.tracking import LatestTrackedFrames
from sam3_runtime.service import make_server
from sam3_runtime.tracker import TrackingEngine, normalize_tracker_feature_names


def candidate(box=None, score=.9):
    return {'bbox': box or [2., 3., 10., 12.], 'score': score, 'query': 'pen'}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch('sam3_runtime.tracker.time.monotonic', return_value=100.)
        self.now = self.clock.start()
        self.addCleanup(self.clock.stop)
        self.detector = SimpleNamespace(fingerprint='detector', detect=Mock(return_value=[candidate()]))
        self.tracker = SimpleNamespace(initialize=Mock(return_value=object()), step=Mock(return_value=candidate()))
        self.engine = TrackingEngine(self.detector, self.tracker)
        self.image = Image.new('RGB', (32, 24))

    def update(self, sha='first', session='a'):
        return self.engine.update(session, self.image, ['pen'], sha)

    def test_detect_once_track_next_and_duplicate_is_idempotent(self):
        self.assertEqual(self.update()['source'], 'sam3_detection')
        self.now.return_value = 100.2
        result = self.update('next')
        self.assertEqual(result['source'], 'sam3_tracker')
        self.assertEqual(self.update('next'), result)
        self.assertEqual(self.detector.detect.call_count, 1)
        self.assertEqual(self.tracker.initialize.call_count, 1)
        self.assertEqual(self.tracker.step.call_count, 1)

    def test_ambiguous_first_frame_does_not_choose_an_instance(self):
        self.detector.detect.return_value = [candidate(), candidate([16, 3, 24, 12], .89)]
        result = self.update()
        self.assertEqual(result['selection_status'], 'ambiguous')
        self.assertIsNone(result['selected'])
        self.tracker.initialize.assert_not_called()

    def test_loss_never_reuses_old_bbox_and_reacquires(self):
        self.update()
        self.tracker.step.return_value = None
        self.detector.detect.return_value = []
        result = self.update('lost')
        self.assertIsNone(result['selected'])
        self.assertIsNone(self.engine.sessions['a'].session)
        self.detector.detect.return_value = [candidate()]
        self.assertIsNotNone(self.update('found')['selected'])
        self.assertEqual(self.tracker.initialize.call_count, 2)

    def test_periodic_detection_matches_current_instance_not_highest_score(self):
        self.update()
        self.engine.sessions['a'].detected_at = 90.
        self.detector.detect.return_value = [candidate([18, 3, 25, 12], .99), candidate()]
        result = self.update('periodic')
        self.assertEqual(result['selected']['bbox'], candidate()['bbox'])
        self.assertEqual(result['source'], 'sam3_detection')

    def test_gap_expiry_target_changes_and_close(self):
        self.update()
        self.now.return_value = 104.
        self.update('after-gap')
        self.tracker.step.assert_not_called()
        self.assertEqual(self.tracker.initialize.call_count, 2)
        with self.assertRaises(ValueError):
            self.engine.update('a', self.image, ['carrot'], 'wrong-task')
        self.update('first', 'b')
        self.engine.close('a')
        self.assertEqual(set(self.engine.sessions), {'b'})
        self.now.return_value = 200.
        self.engine.expire()
        self.assertFalse(self.engine.sessions)

    def test_partial_tracker_failure_discards_session(self):
        self.update()
        self.tracker.step.side_effect = RuntimeError('GPU failed')
        with self.assertRaises(RuntimeError):
            self.update('next')
        self.assertNotIn('a', self.engine.sessions)

    def test_feature_alias_preserves_tensors_and_existing_fields(self):
        tensor = object()
        model = SimpleNamespace(get_image_features=lambda *a, **kw: SimpleNamespace(fpn_position_encoding=tensor))
        normalize_tracker_feature_names(model)
        self.assertIs(model.get_image_features().fpn_position_embeddings, tensor)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'frame.png'
        Image.new('RGB', (32, 24)).save(self.path)
        detector = SimpleNamespace(fingerprint='test-detector', detect=Mock(return_value=[candidate()]))
        self.engine = TrackingEngine(detector, SimpleNamespace(initialize=lambda *a: object(), step=lambda *a: candidate()))
        self.server = make_server('127.0.0.1', 0, detector, self.engine)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = GroundingClient(f'http://127.0.0.1:{self.server.server_port}')

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.tmp.cleanup()

    def test_real_http_contract_and_session_cleanup(self):
        result = self.client.track(self.path, ['pen'], 'task:cam_high')
        self.assertEqual(result['image_sha256'], file_sha(self.path))
        self.assertEqual(result['source'], 'sam3_detection')
        self.assertEqual(result['selected']['bbox'], candidate()['bbox'])
        self.assertFalse(self.client._cache)
        self.client.close_track('task:cam_high')
        self.assertFalse(self.engine.sessions)

    def test_wrong_frame_or_query_is_rejected(self):
        result = self.client.track(self.path, ['pen'], 'a')
        with self.assertRaises(AlignmentError):
            validate_grounding_result(result, 'old-frame', (32,24), ['pen'])
        with self.assertRaises(AlignmentError):
            validate_grounding_result(result, file_sha(self.path), (32,24), ['carrot'])


class LatestSlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.queue = queue.Queue()
        self.track = Mock(side_effect=self.result)
        self.closed = Mock()
        self.worker = LatestTrackedFrames(capture=lambda: self.queue.get(timeout=.05),
            client=SimpleNamespace(track=self.track, close_track=self.closed,
                                   _request=lambda route: {'tracking_enabled': True}),
            cameras=['cam_high'], queries=['pen'], directory=self.root, session_id='generation',
            poll_interval_s=.001, max_frame_age_s=2.)
        self.worker.start()

    def tearDown(self):
        self.worker.close(); self.tmp.cleanup()

    def result(self, path, queries, session):
        return {'source': 'sam3_tracker', 'selected': candidate(), 'candidates': [candidate()],
            'image_sha256': file_sha(path), 'image_size': [32,24], 'coordinate_space': 'input_image_xyxy',
            'status': 'ok', 'selection_status': 'ok'}

    def push(self, index):
        folder = self.root / f'capture_{index}'
        folder.mkdir()
        frames = {}
        for camera in ('cam_high', 'cam_left_wrist', 'cam_right_wrist'):
            path = folder / f'{camera}.png'
            Image.new('RGB', (32,24), (index,0,0)).save(path)
            frames[camera] = str(path)
        self.queue.put((frames, {'identity': str(index)}))
        deadline = time.monotonic()+2
        while self.worker.status()['frames_processed'] < index:
            if time.monotonic() > deadline:
                self.fail('tracking worker did not publish')
            time.sleep(.005)

    def test_slow_consumer_gets_latest_and_pinned_frames_survive_replacement(self):
        self.push(1)
        pinned, _, grounding = self.worker.read()
        for index in range(2, 6):
            self.push(index)
        latest, observation, _ = self.worker.read()
        self.assertEqual(observation['identity'], '5')
        self.assertEqual(file_sha(pinned['cam_high']), grounding['after_cam_high']['image_sha256'])
        self.assertNotEqual(file_sha(pinned['cam_high']), file_sha(latest['cam_high']))
        self.assertEqual(len(list(self.root.glob('capture_*'))), 1)
        self.worker.close()
        self.assertTrue(Path(pinned['cam_high']).exists())
        self.closed.assert_called_with('generation:cam_high')

    def test_transport_failure_publishes_no_box_and_stale_slot_is_rejected(self):
        self.track.side_effect = RuntimeError('disconnected')
        self.push(1)
        _, _, grounding = self.worker.read()
        result = grounding['after_cam_high']
        self.assertIsNone(result['selected'])
        self.assertEqual(result['selection_status'], 'tracking_error')
        with self.worker._lock:
            self.worker._latest['captured_monotonic'] -= 3
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            self.worker.read()


class MonitorLifecycleTests(unittest.TestCase):
    def test_deferred_start_tracks_during_grm_and_releases_task_session(self):
        from monitor_runtime.grm_backend import GRMMonitorBackend, CAMERA_KEYS
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'tracking.yaml'
            config.write_text('enabled: true\npoll_interval_s: 0.01\nmax_frame_age_s: 2.0\n')
            entered, release = threading.Event(), threading.Event()
            captured_samples = []
            def inference(samples):
                captured_samples.append(deepcopy(samples))
                entered.set()
                if not release.wait(2):
                    raise RuntimeError('test generation gate timed out')
                return [{**s, 'pred': '<score>+20%</score>', 'valid': True} for s in samples]
            model = SimpleNamespace(inference_batch=inference)
            backend = GRMMonitorBackend(runtime_url='http://unused', goal_image=str(root/'examples/blank_goal.png'),
                model=model, inference_engine='hf', steering_config=str(root/'configs/steering.yaml'),
                tracking_config=str(config), output_root=temporary, active_modes=['forward','incremental'])
            def snapshot(state, **kwargs):
                state.capture_index += 1
                directory = backend._session_dir(state) / f'capture_{state.capture_index}'
                directory.mkdir()
                frames = {}
                for camera in CAMERA_KEYS:
                    path = directory / f'{camera}.png'
                    Image.new('RGB', (32,24), (state.capture_index % 255,0,0)).save(path)
                    frames[camera] = str(path)
                return frames, {'identity': str(state.capture_index), 'snapshot_requested_at': time.time()}
            backend._snapshot_current = snapshot
            def track(path, queries, session_id):
                row = {**candidate(), 'query': queries[0]}
                return dict(image_sha256=file_sha(path), image_size=[32,24],
                    coordinate_space='input_image_xyxy', status='ok', selection_status='ok',
                    selected=row, candidates=[row], source='sam3_tracker')
            client = SimpleNamespace(_request=lambda path: {'tracking_enabled': True},
                                     track=Mock(side_effect=track), close_track=Mock())
            def wait_for(predicate):
                deadline = time.monotonic()+2
                while not predicate():
                    if time.monotonic() > deadline:
                        self.fail('monitor lifecycle timed out')
                    time.sleep(.005)
            try:
                with patch('grm_runtime.grounding.GroundingClient', return_value=client):
                    backend.start({'monitor_id':'m', 'execution_id':'e', 'subtask':'pick pen',
                                   'target_queries':['pen'], 'defer_inference':True})
                    state = backend.sessions['m']
                    wait_for(lambda: not backend.status({'monitor_id':'m'}).result['warming_up'])
                    self.assertFalse(captured_samples)
                    backend.activate({'monitor_id':'m'})
                    self.assertTrue(entered.wait(2))
                    count = state.tracking_stream.status()['frames_processed']
                    wait_for(lambda: state.tracking_stream.status()['frames_processed'] >= count+3)
                    sample = captured_samples[0][0]
                    self.assertEqual(file_sha(sample['image'][5]), sample['online_grounding']['after_cam_high']['image_sha256'])
                    release.set()
                    wait_for(lambda: state.step > 0)
                    record = deepcopy(state.latest)
                    self.assertEqual(backend.frame_image(record['preview']['frame_set_id'], 'cam_high'),
                                     Path(record['frames']['cam_high']).read_bytes())
                    backend.stop({'monitor_id':'m'})
                    self.assertFalse(state.thread.is_alive())
                    self.assertFalse(state.tracking_stream._thread.is_alive())
                    client.close_track.assert_called_with(f'{state.generation}:cam_high')
                    backend.start({'monitor_id':'new', 'execution_id':'next', 'subtask':'pick carrot',
                                   'target_queries':['carrot'], 'defer_inference':True})
                    fresh = backend.sessions['new']
                    wait_for(lambda: not backend.status({'monitor_id':'new'}).result['warming_up'])
                    self.assertNotEqual(fresh.generation, state.generation)
                    self.assertEqual(client.track.call_args.args[1:], (['carrot'], f'{fresh.generation}:cam_high'))
            finally:
                release.set()
                backend.close()


if __name__ == '__main__':
    unittest.main()

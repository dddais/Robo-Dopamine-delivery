"""Locked/hybrid tracking, exact-frame handoff and bounded buffering."""
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
from grm_runtime.grounding import AlignmentError, GroundingClient, GroundingError, validate_grounding_result
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
        self.started = set()

    def update(self, sha='first', session='a'):
        initialize = session not in self.started
        self.started.add(session)
        return self.engine.update(session, self.image, ['pen'], sha, initialize=initialize)

    def assert_lost(self, result, reason):
        self.assertIsNone(result['selected'])
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['status'], 'no_detection')
        self.assertEqual(result['tracking_state'], 'lost')
        self.assertEqual(result['loss_reason'], reason)

    def test_detect_once_track_next_and_duplicate_is_idempotent(self):
        self.assertEqual(self.update()['source'], 'sam3_detection')
        self.now.return_value = 100.2
        result = self.update('next')
        self.assertEqual(result['source'], 'sam3_tracker')
        self.assertEqual(self.update('next'), result)
        self.assertEqual(self.detector.detect.call_count, 1)
        self.assertEqual(self.tracker.initialize.call_count, 1)
        self.assertEqual(self.tracker.step.call_count, 1)
        self.assertEqual(result['identity_policy'], 'initial_instance')

    def test_close_first_frame_scores_bind_highest_and_never_reselect(self):
        lower = candidate([16, 3, 24, 12], .89)
        # Deliberately unsorted so selecting the first returned row would fail.
        self.detector.detect.return_value = [lower, candidate()]
        result = self.update()
        self.assertEqual(result['selection_status'], 'ok')
        self.assertEqual(result['selected'], candidate())
        self.tracker.initialize.assert_called_once_with(self.image, candidate()['bbox'])
        self.detector.detect.return_value = [{**lower, 'score': 1.}]
        self.assertEqual(self.update('next')['selected'], candidate())
        self.tracker.step.return_value = None
        self.assert_lost(self.update('lost'), 'empty_mask')
        self.assert_lost(self.update('remaining-pen'), 'empty_mask')
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_called_once()

    def test_equal_first_frame_scores_bind_first_returned_candidate_stably(self):
        first = candidate([16, 3, 24, 12])
        self.detector.detect.return_value = [first, candidate()]
        result = self.update()
        self.assertEqual(result['selected'], first)
        self.assertEqual(result['selection_status'], 'ok')
        self.assertEqual(self.update(), result)
        self.tracker.initialize.assert_called_once_with(self.image, first['bbox'])

    def test_no_initial_detection_cannot_bind_a_later_scene(self):
        self.detector.detect.return_value = []
        self.assert_lost(self.update(), 'no_detection')
        self.detector.detect.return_value = [candidate()]
        self.assert_lost(self.update('later'), 'no_detection')
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_not_called()

    def test_left_pen_lost_never_switches_to_remaining_pen_or_reacquires(self):
        left, right = candidate(), candidate([18, 3, 25, 12], .99)
        left['query'] = right['query'] = 'left pen'
        self.detector.detect.return_value = [left]
        first = self.engine.update('left-task', self.image, ['left pen'], 'initial', initialize=True)
        self.assertEqual(first['selected'], left)
        self.tracker.step.return_value = None
        self.detector.detect.return_value = [right]
        for sha in ('occluded', 'remaining-pen', 'initial'):
            self.assert_lost(self.engine.update('left-task', self.image, ['left pen'], sha), 'empty_mask')
        # Even if the tracker could output a box again, it may be a different
        # instance: never resume a trajectory after confidence was lost.
        self.tracker.step.return_value = right
        self.assert_lost(self.engine.update('left-task', self.image, ['left pen'], 'later'), 'empty_mask')
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_called_once()
        self.tracker.step.assert_called_once()

    def test_low_confidence_loss_is_terminal(self):
        self.update()
        self.tracker.step.return_value = candidate(score=.49)
        self.assert_lost(self.update('lost'), 'low_score')
        self.assertIsNone(self.engine.sessions['a'].session)
        self.tracker.step.return_value = candidate(score=1.)
        self.assert_lost(self.update('found'), 'low_score')
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_called_once()

    def test_nonfinite_presence_is_not_a_valid_target(self):
        self.update()
        self.tracker.step.return_value = candidate(score=float('nan'))
        self.assert_lost(self.update('invalid'), 'low_score')

    def test_confident_tracker_jump_is_rejected_and_never_resumed(self):
        self.update()
        self.tracker.step.return_value = candidate([18, 3, 25, 12], 1.)
        self.assert_lost(self.update('other-pen'), 'discontinuous_bbox')
        self.tracker.step.return_value = candidate()
        self.assert_lost(self.update('original-pen'), 'discontinuous_bbox')
        self.tracker.step.assert_called_once()

    def test_loss_diagnostics_survive_later_frames_without_becoming_current_box(self):
        self.update()
        rejected = candidate([18, 3, 25, 12], .99)
        self.tracker.step.return_value = rejected
        lost = self.update('triggering-frame')
        later = self.update('later-frame')
        self.assert_lost(later, 'discontinuous_bbox')
        self.assertEqual(later['last_loss'], lost['last_loss'])
        self.assertEqual(later['last_loss']['image_sha256'], 'triggering-frame')
        self.assertEqual(later['last_loss']['bbox'], rejected['bbox'])
        self.assertEqual(later['last_loss']['previous_bbox'], candidate()['bbox'])
        self.assertEqual(later['last_loss']['score'], .99)
        self.assertEqual(later['last_loss']['iou'], 0.)

    def test_locked_mode_can_disable_overlap_guard_without_enabling_redetection(self):
        self.engine = TrackingEngine(self.detector, self.tracker, continuity_iou=0.)
        self.update()
        moved = candidate([18, 3, 25, 12], .99)
        self.tracker.step.return_value = moved
        self.assertEqual(self.update('moved')['selected'], moved)
        self.tracker.step.return_value = None
        self.assert_lost(self.update('lost'), 'empty_mask')
        self.assert_lost(self.update('retry'), 'empty_mask')
        self.detector.detect.assert_called_once()

    def test_moving_original_instance_retains_query_and_memory(self):
        self.update()
        session = self.engine.sessions['a'].session
        self.tracker.step.return_value = {**candidate([3, 2, 11, 11]), 'query': 'irrelevant'}
        result = self.update('lifted')
        self.assertEqual(result['selected']['bbox'], [3, 2, 11, 11])
        self.assertEqual(result['selected']['query'], 'pen')
        self.assertIs(self.engine.sessions['a'].session, session)

    def test_periodic_text_mismatch_cannot_reset_healthy_tracking(self):
        # Old YAML remains loadable, but locked mode ignores its interval.
        self.engine = TrackingEngine(self.detector, self.tracker, redetect_interval_s=5.)
        self.update()
        for second in range(1, 12):
            self.now.return_value = 100. + second
            self.detector.detect.return_value = [] if second % 2 else [candidate([18, 3, 25, 12], .99)]
            result = self.update(f'frame-{second}')
            self.assertEqual(result['selected']['bbox'], candidate()['bbox'])
            self.assertEqual(result['source'], 'sam3_tracker')
            self.assertEqual(result['tracker_frame_index'], second)
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_called_once()

    def test_gap_invalidates_cached_box_and_cannot_reinitialize(self):
        self.update()
        self.now.return_value = 104.
        self.assert_lost(self.update('first'), 'update_gap')
        self.assert_lost(self.update('after-gap'), 'update_gap')
        self.tracker.step.assert_not_called()
        self.tracker.initialize.assert_called_once()

    def test_expired_closed_or_unknown_continuation_never_detects(self):
        self.update()
        self.now.return_value = 200.
        self.engine.expire()
        self.assertFalse(self.engine.sessions)
        self.assert_lost(self.update('expired'), 'session_missing')
        self.update('initial', 'new-task')
        self.engine.close('new-task')
        self.assert_lost(self.update('closed', 'new-task'), 'session_missing')
        self.assert_lost(self.engine.update('unknown', self.image, ['pen'], 'frame'), 'session_missing')
        self.assertFalse(self.engine.sessions)
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_target_size_changes_and_new_task_isolation(self):
        self.update()
        with self.assertRaises(ValueError):
            self.engine.update('a', self.image, ['carrot'], 'wrong-task')
        with self.assertRaises(ValueError):
            self.engine.update('a', Image.new('RGB', (64, 48)), ['pen'], 'wrong-size')
        self.update('first', 'b')
        self.engine.close('a')
        self.assertEqual(set(self.engine.sessions), {'b'})
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_partial_tracker_failure_releases_gpu_but_keeps_terminal_identity(self):
        self.update()
        self.tracker.step.side_effect = RuntimeError('GPU failed')
        with self.assertRaises(RuntimeError):
            self.update('next')
        self.assertIsNone(self.engine.sessions['a'].session)
        self.assert_lost(self.update('first'), 'inference_error')
        self.assert_lost(self.update('retry'), 'inference_error')
        self.detector.detect.assert_called_once()
        self.tracker.initialize.assert_called_once()

    def test_failed_initialization_cannot_bind_a_second_target(self):
        self.tracker.initialize.side_effect = RuntimeError('GPU failed')
        with self.assertRaises(RuntimeError):
            self.update()
        self.tracker.initialize.side_effect = None
        self.detector.detect.return_value = [candidate([18, 3, 25, 12])]
        self.assert_lost(self.update('retry'), 'inference_error')
        self.detector.detect.assert_called_once()

    def test_repeated_initialize_flag_does_not_reset_a_lost_task(self):
        self.update()
        self.tracker.step.return_value = None
        self.assert_lost(self.update('lost'), 'empty_mask')
        self.assert_lost(self.engine.update('a', self.image, ['pen'], 'retry', initialize=True), 'empty_mask')
        self.detector.detect.assert_called_once()

    def test_feature_alias_preserves_tensors_and_existing_fields(self):
        tensor = object()
        model = SimpleNamespace(get_image_features=lambda *a, **kw: SimpleNamespace(fpn_position_encoding=tensor))
        normalize_tracker_feature_names(model)
        self.assertIs(model.get_image_features().fpn_position_embeddings, tensor)


class HybridEngineTests(unittest.TestCase):
    update = EngineTests.update
    assert_lost = EngineTests.assert_lost

    def setUp(self):
        EngineTests.setUp(self)
        self.engine = TrackingEngine(self.detector, self.tracker, mode='hybrid')

    def test_hybrid_accepts_motion_without_overlap_by_default(self):
        first = self.update()
        moved = candidate([18, 3, 25, 12], .99)
        self.tracker.step.return_value = moved
        result = self.update('lifted')
        self.assertEqual(result['selected'], moved)
        self.assertEqual(result['identity_policy'], 'text_redetection')
        self.assertEqual(result['tracking_mode'], 'hybrid')
        self.assertEqual(result['source'], 'sam3_tracker')
        self.assertEqual(result['initialization_count'], first['initialization_count'])
        self.detector.detect.assert_called_once()

    def test_periodic_detection_matches_current_frame_not_highest_score_or_old_box(self):
        self.update()
        moved = candidate([18, 3, 25, 12], .6)
        self.tracker.step.return_value = moved
        self.engine.sessions['a'].detected_at = 90.
        self.detector.detect.return_value = [candidate(score=.99), moved]
        result = self.update('periodic')
        self.assertEqual(result['selected'], moved)
        self.assertEqual(result['detection_reason'], 'periodic')
        self.assertEqual(result['detection_status'], 'matched')
        self.assertEqual(result['source'], 'sam3_detection')
        self.assertEqual(result['initialization_count'], 2)
        self.assertEqual(result['tracker_frame_index'], 0)

    def test_periodic_mismatch_preserves_healthy_track_and_does_not_retry_every_frame(self):
        self.update()
        session = self.engine.sessions['a'].session
        self.detector.detect.return_value = [candidate([18, 3, 25, 12], .99)]
        self.engine.sessions['a'].detected_at = 90.
        result = self.update('periodic')
        self.assertEqual(result['selected'], candidate())
        self.assertEqual(result['detection_status'], 'no_match_kept_tracking')
        self.assertEqual(result['source'], 'sam3_tracker')
        self.assertIs(self.engine.sessions['a'].session, session)
        self.assertEqual(self.update('next')['selected'], candidate())
        self.assertEqual(self.detector.detect.call_count, 2)
        self.tracker.initialize.assert_called_once()

    def test_periodic_empty_detection_keeps_video_result(self):
        self.update()
        self.detector.detect.return_value = []
        self.engine.sessions['a'].detected_at = 90.
        result = self.update('periodic')
        self.assertEqual(result['selected'], candidate())
        self.assertIsNone(result['loss_reason'])
        self.assertEqual(result['detection_status'], 'no_match_kept_tracking')

    def test_loss_redetects_highest_score_and_may_bind_another_instance(self):
        self.update()
        self.tracker.step.return_value = None
        other = candidate([18, 3, 25, 12], .99)
        self.detector.detect.return_value = [candidate(score=.98), other]
        result = self.update('lost')
        self.assertEqual(result['selected'], other)
        self.assertEqual(result['detection_reason'], 'empty_mask')
        self.assertEqual(result['source'], 'sam3_detection')
        self.assertIsNone(result['loss_reason'])
        self.assertEqual(result['last_loss']['reason'], 'empty_mask')
        self.assertEqual(result['last_loss']['image_sha256'], 'lost')
        self.assertEqual(result['initialization_count'], 2)
        # A retry of the same completed frame must not reinitialize yet again.
        self.assertEqual(self.update('lost'), result)
        self.assertEqual(self.detector.detect.call_count, 2)

    def test_no_detection_after_loss_retries_on_next_frame_without_old_bbox(self):
        self.update()
        self.tracker.step.return_value = None
        self.detector.detect.return_value = []
        lost = self.update('lost')
        self.assert_lost(lost, 'empty_mask')
        self.assertEqual(lost['detection_status'], 'no_detection')
        self.detector.detect.return_value = [candidate()]
        self.assertEqual(self.update('reappeared')['selected'], candidate())
        self.assertEqual(self.tracker.initialize.call_count, 2)
        self.tracker.step.assert_called_once()

    def test_no_initial_detection_can_retry_in_hybrid_mode(self):
        self.detector.detect.return_value = []
        self.assert_lost(self.update(), 'no_detection')
        self.detector.detect.return_value = [candidate()]
        self.assertEqual(self.update('found')['selected'], candidate())

    def test_low_score_triggers_same_frame_detection(self):
        self.update()
        self.tracker.step.return_value = candidate(score=.49)
        result = self.update('low')
        self.assertEqual(result['detection_reason'], 'low_score')
        self.assertEqual(result['last_loss']['score'], .49)
        self.assertEqual(result['source'], 'sam3_detection')

    def test_optional_overlap_guard_can_trigger_hybrid_recovery(self):
        self.engine = TrackingEngine(self.detector, self.tracker, mode='hybrid', continuity_iou=.1)
        self.update()
        self.tracker.step.return_value = candidate([18, 3, 25, 12], .99)
        result = self.update('jump')
        self.assertEqual(result['detection_reason'], 'discontinuous_bbox')
        self.assertEqual(result['source'], 'sam3_detection')

    def test_gap_drops_old_state_and_detects_current_frame(self):
        self.update()
        self.now.return_value = 104.
        result = self.update('first')
        self.assertEqual(result['detection_reason'], 'update_gap')
        self.assertEqual(result['last_loss']['gap_s'], 4.)
        self.assertEqual(result['initialization_count'], 2)
        self.tracker.step.assert_not_called()

    def test_partial_inference_error_can_reinitialize_only_on_next_update(self):
        self.update()
        self.tracker.step.side_effect = RuntimeError('GPU failed')
        with self.assertRaises(RuntimeError):
            self.update('failed')
        self.assertIsNone(self.engine.sessions['a'].session)
        self.tracker.step.side_effect = None
        result = self.update('retry')
        self.assertEqual(result['detection_reason'], 'inference_error')
        self.assertEqual(result['last_loss']['image_sha256'], 'failed')
        self.assertEqual(result['initialization_count'], 2)

    def test_missing_sessions_still_require_explicit_first_task_request(self):
        self.update()
        self.now.return_value = 200.
        self.assert_lost(self.update('expired'), 'session_missing')
        self.assert_lost(self.engine.update('unknown', self.image, ['pen'], 'frame'), 'session_missing')
        self.detector.detect.assert_called_once()

    def test_configuration_validation(self):
        for options in ({'mode': 'typo'}, {'mode': None}, {'mode': True},
                        {'continuity_iou': -.1}, {'continuity_iou': 1.1}, {'continuity_iou': True},
                        {'continuity_iou': float('nan')}, {'redetect_interval_s': 0},
                        {'redetect_interval_s': float('inf')}, {'redetect_interval_s': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TrackingEngine(self.detector, self.tracker, **options)


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

    def next_frame(self):
        Image.new('RGB', (32, 24), 'red').save(self.path)

    def test_http_initialization_returns_highest_close_scoring_candidate(self):
        self.engine.detector.detect.return_value = [candidate([18, 3, 25, 12], .88), candidate()]
        result = self.client.track(self.path, ['pen'], 'close-scores')
        self.assertEqual(result['selected'], candidate())
        self.assertEqual(result['selection_status'], 'ok')
        self.assertEqual(result['image_sha256'], file_sha(self.path))

    def test_http_hybrid_reacquisition_and_health_match_same_frame(self):
        self.engine.__init__(self.engine.detector, self.engine.tracker, mode='hybrid')
        self.client.track(self.path, ['pen'], 'a')
        other = candidate([18, 3, 25, 12], .99)
        self.engine.tracker.step = Mock(return_value=None)
        self.engine.detector.detect.return_value = [other]
        self.next_frame()
        result = self.client.track(self.path, ['pen'], 'a')
        self.assertEqual(result['identity_policy'], 'text_redetection')
        self.assertEqual(result['selected'], other)
        self.assertEqual(result['image_sha256'], file_sha(self.path))
        self.assertEqual(result['last_loss']['image_sha256'], file_sha(self.path))
        health = self.client._request('/health')
        self.assertEqual(health['tracking_identity_policy'], 'text_redetection')
        self.assertEqual(health['tracking_config']['mode'], 'hybrid')
        self.assertEqual(health['tracking_config']['continuity_iou'], 0.)

    def test_client_rejects_policy_change_mid_task(self):
        self.client.track(self.path, ['pen'], 'a')
        request = self.client._request
        def changed_policy(route, payload):
            result = request(route, payload)
            result['identity_policy'] = 'text_redetection'
            return result
        with patch.object(self.client, '_request', side_effect=changed_policy):
            with self.assertRaisesRegex(AlignmentError, 'identity policy changed'):
                self.client.track(self.path, ['pen'], 'a')

    def test_http_loss_publishes_empty_current_frame_and_new_task_can_bind(self):
        self.client.track(self.path, ['pen'], 'a')
        self.engine.tracker.step = Mock(return_value=None)
        self.engine.detector.detect.return_value = [candidate([18, 3, 25, 12])]
        self.next_frame()
        for _ in range(2):
            result = self.client.track(self.path, ['pen'], 'a')
            self.assertIsNone(result['selected'])
            self.assertEqual(result['selection_status'], 'tracking_lost')
            self.assertEqual(result['loss_reason'], 'empty_mask')
            self.assertEqual(result['image_sha256'], file_sha(self.path))
        self.engine.detector.detect.assert_called_once()
        result = self.client.track(self.path, ['pen'], 'new-task')
        self.assertEqual(result['selected']['bbox'], [18, 3, 25, 12])

    def test_server_state_loss_and_close_do_not_reinitialize_old_tasks(self):
        for session in ('expired-task', 'closed-task'):
            self.client.track(self.path, ['pen'], session)
            if session == 'expired-task':
                self.engine.sessions[session].seen_at = 0.
                self.engine.expire()
            else:
                self.client.close_track(session)
            result = self.client.track(self.path, ['pen'], session)
            self.assertEqual(result['loss_reason'], 'session_missing')
            self.assertIsNone(result['selected'])
        self.engine.sessions.clear()  # Same lost-state condition as a service restart.
        result = self.client.track(self.path, ['pen'], 'expired-task')
        self.assertIsNone(result['selected'])
        self.assertEqual(self.engine.detector.detect.call_count, 2)
        self.assertFalse(self.engine.sessions)

    def test_lost_first_request_is_not_retried_as_a_new_initialization(self):
        with patch.object(self.client, '_request', side_effect=GroundingError('connection failed')):
            with self.assertRaises(GroundingError):
                self.client.track(self.path, ['pen'], 'a')
        result = self.client.track(self.path, ['pen'], 'a')
        self.assertIsNone(result['selected'])
        self.assertEqual(result['loss_reason'], 'session_missing')
        self.engine.detector.detect.assert_not_called()

    def test_lost_first_response_keeps_the_existing_instance(self):
        request = self.client._request
        def drop_response(route, payload):
            request(route, payload)
            raise GroundingError('response lost after initialization')
        with patch.object(self.client, '_request', side_effect=drop_response):
            with self.assertRaises(GroundingError):
                self.client.track(self.path, ['pen'], 'a')
        self.engine.detector.detect.return_value = [candidate([18, 3, 25, 12])]
        self.next_frame()
        result = self.client.track(self.path, ['pen'], 'a')
        self.assertEqual(result['selected']['bbox'], candidate()['bbox'])
        self.assertEqual(result['source'], 'sam3_tracker')
        self.engine.detector.detect.assert_called_once()

    def test_old_server_without_identity_policy_is_rejected(self):
        request = self.client._request
        def old_server(route, payload):
            result = request(route, payload)
            result.pop('identity_policy')
            return result
        with patch.object(self.client, '_request', side_effect=old_server):
            with self.assertRaisesRegex(AlignmentError, 'supported identity policy'):
                self.client.track(self.path, ['pen'], 'a')


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

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from monitor_runtime.core import MonitorState
from monitor_runtime.grm_backend import CAMERA_KEYS, GRMMonitorBackend, _SubtaskState
from monitor_runtime.service import _build_argparser, create_app, main

ROOT = Path(__file__).resolve().parents[1]


class BranchModel:
    def __init__(self, *, steering):
        self.steering = steering
        self.scores = {'forward': 20, 'incremental': 20, 'backward': -80}
        self.calls = []
        self.manifest = {'test_branch': 'steering' if steering else 'baseline'}

    def inference_batch(self, samples):
        self.calls.append(deepcopy(samples))
        return [{**s, 'pred': f"<score>{self.scores[s['eval_mode']]:+g}%</score>", 'valid': True,
                 'steering': {'enabled': self.steering, 'applied': self.steering,
                              'degraded': False, 'grounding_ms': 2 if self.steering else 0, 'grm_ms': 10}}
                for s in samples]


class DifferenceStateTests(unittest.TestCase):
    def test_failure_veto_precedes_success_and_is_terminal(self):
        state = MonitorState(success_stable_steps=1, progress_difference_threshold=.2)
        self.assertEqual(state.update(.9, progress_difference=.3), 'failed')
        self.assertEqual(state.update(.9, progress_difference=0), 'failed')
        self.assertEqual(state.progress_history, [.9])
        state.reset()
        self.assertEqual(state.update(.9, progress_difference=0), 'success')

    def test_threshold_is_strict_and_legacy_window_still_applies(self):
        state = MonitorState(success_stable_steps=3, success_max_drift=.05,
                             progress_difference_threshold=.2)
        self.assertEqual(state.update(.58, progress_difference=.2), 'running')
        self.assertEqual(state.update(.59, progress_difference=.2), 'running')
        self.assertEqual(state.update(.60, progress_difference=.2), 'success')
        state = MonitorState(fail_stable_steps=3, progress_difference_threshold=.2)
        for progress in (.3, .2, .1):
            state.update(progress, progress_difference=0)
        self.assertEqual(state.status, 'failed')

    def test_invalid_difference_does_not_advance(self):
        state = MonitorState(progress_difference_threshold=.2)
        for value in (None, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                state.update(.5, progress_difference=value)
        self.assertEqual(state.progress_history, [])

    def test_decimal_equality_does_not_fail_due_to_roundoff(self):
        state = MonitorState(success_stable_steps=1, progress_difference_threshold=.2)
        self.assertEqual(state.update(.6, progress_difference=.8 - .6), 'success')
        state.reset()
        self.assertEqual(state.update(.6, progress_difference=.20000001), 'failed')

    def test_disabled_rule_preserves_existing_callers(self):
        state = MonitorState(success_stable_steps=1)
        self.assertEqual(state.update(.9), 'success')


class DualBranchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.steering = BranchModel(steering=True)
        self.baseline = BranchModel(steering=False)
        self.options = dict(runtime_url='http://unused', goal_image=str(ROOT / 'examples/blank_goal.png'),
                            inference_engine='hf', steering_config=str(ROOT / 'configs/steering.yaml'),
                            output_root=self.tmp.name, active_modes=['forward', 'incremental'],
                            model=self.steering, baseline_model=self.baseline, dual_branch=True,
                            success_stable_steps=3, fail_stable_steps=3)
        self.backend = GRMMonitorBackend(**self.options)
        self.state = _SubtaskState('m', 'e', 'pick carrot', queries=['carrot'],
                                   monitor=MonitorState(**self.backend.monitor_options))
        self.backend.sessions['m'] = self.state
        self.reference = self.images('reference')
        self.state.ref_start = self.state.previous = self.reference
        self.state.last_observation = 'reference'
        self.capture('first')

    def tearDown(self):
        self.backend.close()
        self.tmp.cleanup()

    def images(self, name):
        directory = self.backend._session_dir(self.state) / name
        directory.mkdir(parents=True, exist_ok=True)
        result = {}
        for camera in CAMERA_KEYS:
            path = directory / f'{camera}.png'
            Image.new('RGB', (32, 32)).save(path)
            result[camera] = str(path)
        return result

    def capture(self, name, *, identity=None):
        self.current = self.images(name)
        current = self.current
        self.backend._snapshot_current = lambda *a, **kw: (current, {'identity': identity or name})

    def run_step(self):
        return self.backend._run_one_step(self.state)

    def test_same_inputs_concurrent_branches_separate_conditions_and_trackers(self):
        barrier = threading.Barrier(2)
        for model in (self.steering, self.baseline):
            infer = model.inference_batch
            def paired(samples, infer=infer):
                barrier.wait(timeout=3)
                return infer(samples)
            model.inference_batch = paired
        self.baseline.scores.update(forward=30, incremental=30)
        first = self.run_step()
        self.assertAlmostEqual(first['progress'], .2)
        self.assertAlmostEqual(first['branches']['baseline']['progress'], .3)
        self.capture('second')
        second = self.run_step()
        self.assertAlmostEqual(second['modes']['incremental']['progress'], .36)
        self.assertAlmostEqual(second['branches']['baseline']['modes']['incremental']['progress'], .51)
        self.assertAlmostEqual(second['comparison']['difference'], .125)
        self.assertAlmostEqual(second['comparison']['modes']['incremental']['score_difference'], .1)
        self.assertAlmostEqual(second['comparison']['modes']['incremental']['progress_difference'], .15)
        for steered, baseline in zip(self.steering.calls, self.baseline.calls):
            self.assertEqual([s['condition'] for s in steered], ['candidate_target'] * 2)
            self.assertEqual([s['condition'] for s in baseline], ['baseline'] * 2)
            self.assertEqual([{k: v for k, v in s.items() if k != 'condition'} for s in steered],
                             [{k: v for k, v in s.items() if k != 'condition'} for s in baseline])
        self.assertEqual(self.steering.calls[1][1]['image'][2:5], [first['frames'][c] for c in CAMERA_KEYS])
        self.assertEqual(second['timing']['grm_ms'], 40)
        self.assertEqual(self.state.tracker.counts, self.state.baseline_tracker.counts)

    def test_difference_failure_exposed_in_http_and_log(self):
        self.steering.scores.update(forward=90, incremental=90)
        self.state.monitor.success_stable_steps = 1
        record = self.run_step()
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure_reason'], 'branch_difference_exceeded')
        self.assertAlmostEqual(record['comparison']['difference'], .7)
        with TestClient(create_app(self.backend)) as client:
            health = client.get('/health').json()['data']['dual_branch']
            self.assertTrue(health['enabled'])
            result = client.post('/monitors/status', json={'monitor_id': 'm'}).json()['data']
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['poll_count'], 1)
            self.assertEqual(result['result']['comparison'], record['comparison'])
        logged = json.loads((self.backend._session_dir(self.state) / 'online_pred.jsonl').read_text())
        self.assertEqual(logged, record)

    def test_equal_threshold_allows_success_and_low_scores_still_fail(self):
        self.steering.scores.update(forward=75, incremental=75)
        self.baseline.scores.update(forward=50, incremental=50)
        self.backend.progress_difference_threshold = .25
        self.state.monitor.progress_difference_threshold = .25
        self.state.monitor.success_stable_steps = 1
        record = self.run_step()
        self.assertEqual(record['comparison']['difference'], .25)
        self.assertFalse(record['comparison']['threshold_exceeded'])
        self.assertEqual(record['status'], 'success')

    def test_same_low_scores_preserve_stagnation_failure(self):
        for model in (self.steering, self.baseline):
            model.scores.update(forward=20, incremental=0)
        for index in range(3):
            self.capture(f'stagnant_{index}')
            record = self.run_step()
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(record['comparison']['threshold_exceeded'])
        self.assertNotIn('failure_reason', record)

    def test_signed_difference_modes(self):
        self.steering.scores.update(forward=10, incremental=10)
        self.baseline.scores.update(forward=80, incremental=80)
        self.backend.difference_mode = 'steering_minus_baseline'
        record = self.run_step()
        self.assertAlmostEqual(record['comparison']['difference'], -.7)
        self.assertEqual(record['status'], 'running')
        self.backend.difference_mode = 'baseline_minus_steering'
        self.capture('signed_second')
        self.assertEqual(self.run_step()['status'], 'failed')

    def test_all_three_modes_preserve_backward_formula(self):
        self.backend.active_modes.append('backward')
        self.baseline.scores['backward'] = -50
        result = self.run_step()
        self.assertAlmostEqual(result['branches']['baseline']['modes']['backward']['progress'], .5)
        self.assertAlmostEqual(result['branches']['baseline']['progress'], .3)
        self.assertAlmostEqual(result['progress'], .2)

    def test_invalid_output_from_either_branch_rolls_back_both_and_retry_uses_committed_frames(self):
        first = self.run_step()
        before = deepcopy((self.state.tracker, self.state.baseline_tracker, self.state.monitor))
        for model in (self.steering, self.baseline):
            original = model.inference_batch
            def invalid(samples, original=original):
                outputs = original(samples)
                outputs[-1].update(pred='<score>nan%</score>', valid=True)
                return outputs
            self.capture(f'invalid_{model.steering}')
            model.inference_batch = invalid
            with self.assertRaisesRegex(RuntimeError, 'Invalid'):
                self.run_step()
            model.inference_batch = original
            self.assertEqual(self.state.step, 1)
            self.assertEqual((self.state.tracker, self.state.baseline_tracker, self.state.monitor), before)
            self.assertEqual(self.state.previous, first['frames'])
            self.assertFalse(Path(self.current['cam_high']).exists())
        self.capture('retry')
        self.run_step()
        for model in (self.steering, self.baseline):
            self.assertEqual(model.calls[-1][1]['image'][2:5], [first['frames'][c] for c in CAMERA_KEYS])
        self.assertEqual(self.state.step, 2)

    def test_baseline_missing_duplicate_or_swapped_outputs_are_rejected(self):
        original = self.baseline.inference_batch
        def swapped(rows):
            rows[0]['eval_mode'], rows[1]['eval_mode'] = rows[1]['eval_mode'], rows[0]['eval_mode']
            return rows
        for index, transform in enumerate((lambda rows: rows[:1], lambda rows: [rows[0], rows[0]], swapped)):
            self.capture(f'bad_output_{index}')
            self.baseline.inference_batch = lambda samples: transform(original(samples))
            with self.assertRaises(RuntimeError):
                self.run_step()
            self.assertEqual(self.state.step, 0)
            self.assertEqual(self.state.tracker.counts['forward'], 0)
            self.assertEqual(self.state.baseline_tracker.counts['forward'], 0)

    def test_log_failure_does_not_commit_either_tracker(self):
        with patch.object(Path, 'open', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.run_step()
        self.assertEqual(self.state.step, 0)
        self.assertEqual(self.state.previous, self.reference)
        self.assertFalse(self.state.monitor.progress_history)
        self.assertEqual(self.state.baseline_tracker.counts['incremental'], 0)
        self.assertEqual(self.state.tracker.counts['incremental'], 0)

    def test_duplicate_observation_skips_both_models(self):
        self.run_step()
        self.capture('duplicate', identity='first')
        self.assertIsNone(self.run_step())
        self.assertEqual(len(self.steering.calls), 1)
        self.assertEqual(len(self.baseline.calls), 1)
        self.assertEqual(len(self.state.monitor.progress_history), 1)

    def test_branch_error_waits_for_other_branch_before_cleaning_frames(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        original = self.baseline.inference_batch
        def slow_baseline(samples):
            entered.set()
            if not release.wait(3):
                raise AssertionError('baseline was not released')
            return original(samples)
        def broken_steering(samples):
            if not entered.wait(3):
                raise AssertionError('baseline did not start')
            raise ValueError('generation failed')
        self.baseline.inference_batch = slow_baseline
        self.steering.inference_batch = broken_steering
        def run():
            try:
                self.run_step()
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertTrue(worker.is_alive())
            self.assertTrue(Path(self.current['cam_high']).exists())
            self.assertFalse(self.backend._infer_lock.acquire(blocking=False))
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIn('steering branch inference failed', str(errors[0]))
        self.assertFalse(Path(self.current['cam_high']).exists())
        self.assertEqual(self.state.step, 0)

    def test_stop_during_pair_never_publishes(self):
        original = self.baseline.inference_batch
        def stop(samples):
            self.state.stop_event.set()
            return original(samples)
        self.baseline.inference_batch = stop
        self.assertIsNone(self.run_step())
        self.assertEqual(self.state.step, 0)
        self.assertEqual(self.state.baseline_tracker.counts['forward'], 0)

    def test_degraded_steering_keeps_diagnostics_and_compares_actual_scores(self):
        original = self.steering.inference_batch
        def degraded(samples):
            rows = original(samples)
            for row in rows:
                row['steering'].update(applied=False, degraded=True, reason='no_detection')
            return rows
        self.steering.inference_batch = degraded
        record = self.run_step()
        self.assertEqual(record['comparison']['difference'], 0)
        self.assertTrue(record['modes']['forward']['steering']['degraded'])
        self.assertEqual(record['status'], 'running')

    def test_manifest_records_both_runtimes_and_configuration(self):
        with patch.object(self.backend, '_inference_loop'):
            self.backend.start({'monitor_id': 'manifest', 'execution_id': 'e', 'subtask': 'pick carrot'})
        state = self.backend.sessions['manifest']
        manifest = json.loads((self.backend._session_dir(state) / 'manifest.json').read_text())
        self.assertEqual(manifest['runtime'], self.steering.manifest)
        self.assertEqual(manifest['baseline_runtime'], self.baseline.manifest)
        self.assertEqual(manifest['dual_branch']['threshold'], .2)

    def test_config_errors_fail_before_loading_models(self):
        cases = [dict(inference_engine='vllm'), dict(steering_config=None), dict(dual_branch='true'),
                 dict(hf_batch_size=0), dict(hf_batch_size=True), dict(hf_batch_size=1.5),
                 dict(progress_difference_threshold=float('nan')), dict(progress_difference_threshold=float('inf')),
                 dict(progress_difference_threshold=-.1), dict(progress_difference_threshold=1.1),
                 dict(progress_difference_threshold=True), dict(difference_mode='unknown')]
        with patch('examples.inference.GRMInference') as loader:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    GRMMonitorBackend(**{**self.options, 'model': None, 'baseline_model': None, **case})
            loader.assert_not_called()

    def test_models_are_loaded_once_with_same_checkpoint_and_decoding(self):
        with patch('examples.inference.GRMInference', side_effect=[self.steering, self.baseline]) as loader:
            backend = GRMMonitorBackend(**{**self.options, 'model': None, 'baseline_model': None,
                                           'device': 'cuda:1', 'baseline_device': 'cuda:2', 'max_new_tokens': 32})
            backend.close()
        self.assertEqual(loader.call_count, 2)
        steered, baseline = loader.call_args_list
        self.assertEqual(steered.args, baseline.args)
        self.assertEqual(steered.kwargs['max_new_tokens'], baseline.kwargs['max_new_tokens'])
        self.assertEqual(steered.kwargs['hf_batch_size'], 2)
        self.assertEqual(baseline.kwargs['hf_batch_size'], 2)
        self.assertEqual(baseline.kwargs['device'], 'cuda:2')
        self.assertEqual(baseline.kwargs['engine'], 'hf')
        self.assertIsNone(baseline.kwargs['steering_config'])

    def test_shared_model_instances_rejected(self):
        for baseline in (self.steering, SimpleNamespace(backend=self.steering)):
            with self.assertRaisesRegex(ValueError, 'independent'):
                GRMMonitorBackend(**{**self.options, 'baseline_model': baseline})


class DualBranchCLITests(unittest.TestCase):
    def test_yaml_and_cli_override_and_main_wiring(self):
        parser = _build_argparser({'dual_branch': True, 'progress_difference_threshold': .3})
        self.assertTrue(parser.parse_args([]).dual_branch)
        self.assertFalse(parser.parse_args(['--no-dual-branch']).dual_branch)
        self.assertEqual(parser.parse_args(['--progress-difference-threshold', '0.4']).progress_difference_threshold, .4)
        with patch('monitor_runtime.grm_backend.GRMMonitorBackend') as loader, patch('uvicorn.run'):
            self.assertEqual(main(['--config', str(ROOT / 'configs/monitor_dual_branch.yaml'),
                                   '--baseline-device', 'cuda:2', '--progress-difference-threshold', '.4']), 0)
        options = loader.call_args.kwargs
        self.assertTrue(options['dual_branch'])
        self.assertEqual(options['baseline_device'], 'cuda:2')
        self.assertEqual(options['progress_difference_threshold'], .4)
        self.assertEqual(options['difference_mode'], 'absolute')
        self.assertEqual(options['steering_config'], str(ROOT / 'configs/steering.yaml'))


if __name__ == '__main__':
    unittest.main()

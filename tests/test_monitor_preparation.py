"""Exercise shared preparation through HF inference without loading GPU weights."""
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from grm_runtime.grounding import GroundingError, png_bytes
from grm_runtime.hf_backend import HFBackend
from grm_runtime.masking import Head


class Processor:
    tokenizer = SimpleNamespace(pad_token_id=0, padding_side='left',
                                decode=lambda ids, **k: f'<score>+{int(ids[0])}%</score>')

    def apply_chat_template(self, *args, **kwargs):
        return 'test prompt'

    def __call__(self, *, images, text, **kwargs):
        rows, grids, pixels = [], [], []
        for index in range(len(text)):
            ids = []
            for image in images[index*8:(index+1)*8]:
                grid = (1, image.height // 4, image.width // 4)
                grids.append(grid)
                ids.extend([1] + [99] * (grid[1] * grid[2] // 4))
                pixels.append([*image.size, *image.getpixel((0, 0))])
            rows.append(ids)
        width = max(map(len, rows))
        return dict(input_ids=torch.tensor([[0]*(width-len(ids)) + ids for ids in rows]),
                    attention_mask=torch.tensor([[0]*(width-len(ids)) + [1]*len(ids) for ids in rows]),
                    image_grid_thw=torch.tensor(grids),
                    pixel_values=torch.tensor(pixels, dtype=torch.float32))


class Attention(torch.nn.Module):
    def forward(self, *, attention_mask):
        return attention_mask


class Model:
    config = SimpleNamespace(image_token_id=99)

    def __init__(self):
        self.attention = Attention()
        self.model = SimpleNamespace(language_model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=self.attention)]))
        self.calls = []
        self.decode_masks = []

    def generate(self, *, input_ids, pixel_values, image_grid_thw, attention_mask, **kwargs):
        length = input_ids.shape[1]
        mask = torch.full((input_ids.shape[0], 1, length, length), -float('inf')).triu(1)
        mask.masked_fill_(attention_mask[:, None, None, :] == 0, -float('inf'))
        mask = self.attention(attention_mask=mask)
        self.calls.append((input_ids.clone(), pixel_values.clone(), image_grid_thw.clone(), mask.clone()))
        decode = torch.zeros(input_ids.shape[0], 1, 1, length+1)
        decode[:, :, :, :length].masked_fill_(attention_mask[:, None, None, :] == 0, -float('inf'))
        self.decode_masks.append(self.attention(attention_mask=decode))
        scores = (pixel_values.reshape(input_ids.shape[0], 8, -1).sum((1, 2)).long() % 50).unsqueeze(1)
        return torch.cat([input_ids, scores], dim=1)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths = []
        for i in range(9):
            path = Path(self.tmp.name) / f'{i}.png'
            Image.new('RGB', (32 if i == 8 else 16, 16), (i, 0, 0)).save(path)
            self.paths.append(str(path))
        backend = HFBackend.__new__(HFBackend)
        backend.torch, backend.device, backend.dtype = torch, 'cpu', torch.float32
        backend.lock = threading.RLock()
        backend.batch_size = 1
        backend.merge, backend.num_heads, backend.max_new_tokens = 2, 2, 64
        backend.heads, backend.low_heads = [Head(0, 0)], [Head(0, 1)]
        backend.config = dict(enabled=True, intervention_labels=['after_cam_high'],
                              on_missing_bbox='baseline', negative_scope='target_span',
                              query_scope='all', bias=6)
        backend.processor, backend.model = Processor(), Model()
        backend.grounder = SimpleNamespace(detect=Mock(side_effect=self.detection))
        self.backend = backend

    def detection(self, path, queries):
        return dict(selected=dict(bbox=[0, 0, 8, 8]), selection_status='ok',
                    candidates=[dict(bbox=[0, 0, 8, 8], query=queries[0], score=.9)])

    def samples(self):
        first = dict(id='forward', eval_mode='forward', task='pick carrot',
                     target_queries=['carrot'], image=self.paths[:8])
        second = {**first, 'id': 'incremental', 'eval_mode': 'incremental',
                  'image': [*self.paths[:2], self.paths[8], *self.paths[3:8]]}
        return [first, second]

    def test_shared_detection_keeps_each_modes_own_spans_and_model_inputs(self):
        samples = self.samples()
        separate = [self.backend.inference_batch([s])[0] for s in samples]
        expected_calls = self.backend.model.calls[:]
        self.backend.model.calls.clear()
        self.backend.grounder.detect.reset_mock()
        shared = self.backend.inference_batch(samples)
        self.assertEqual(self.backend.grounder.detect.call_count, 1)
        for old, new, expected, actual in zip(separate, shared, expected_calls, self.backend.model.calls):
            self.assertEqual(old['pred'], new['pred'])
            for key in ('spans', 'target_positions', 'negative_positions', 'per_layer', 'applied'):
                self.assertEqual(old['steering'][key], new['steering'][key])
            for a, b in zip(expected, actual):
                self.assertTrue(torch.equal(a, b))
        # The different BEFORE image grid shifts AFTER token positions. Reusing
        # the first mode's token indices instead of its bbox would break this.
        self.assertNotEqual(shared[0]['steering']['target_positions'], shared[1]['steering']['target_positions'])
        a, b = [r['steering']['grounding']['after_cam_high'] for r in shared]
        self.assertFalse(a['reused_in_batch'])
        self.assertTrue(b['reused_in_batch'])
        a['selected']['bbox'][0] = 7
        self.assertEqual(b['selected']['bbox'][0], 0)
        self.assertGreater(shared[1]['steering']['image_cache_hits'], 0)
        self.assertFalse(self.backend.model.attention._forward_pre_hooks)
        self.backend.inference_batch(samples)
        self.assertEqual(self.backend.grounder.detect.call_count, 2)  # new round revalidates

    def test_different_queries_do_not_share_detection(self):
        samples = self.samples()
        samples[1]['target_queries'] = ['cube']
        self.backend.inference_batch(samples)
        self.assertEqual(self.backend.grounder.detect.call_count, 2)

    def test_rewritten_path_invalidates_images_and_detection(self):
        sample = self.samples()[0]
        def samples():
            yield sample
            Image.new('RGB', (16, 16), 'blue').save(sample['image'][5])
            yield sample
        self.backend.inference_batch(samples())
        self.assertEqual(self.backend.grounder.detect.call_count, 2)
        self.assertFalse(torch.equal(self.backend.model.calls[0][1], self.backend.model.calls[1][1]))

    def test_missing_results_are_shared_but_transport_errors_retry(self):
        for reason in ('ambiguous', 'no_detection'):
            with self.subTest(reason=reason):
                self.backend.grounder.detect = Mock(return_value=dict(selected=None, selection_status=reason))
                rows = self.backend.inference_batch(self.samples())
                self.assertEqual(self.backend.grounder.detect.call_count, 1)
                self.assertEqual([r['steering']['reason'] for r in rows], [reason, reason])
                self.assertTrue(all(r['steering']['degraded'] for r in rows))
        self.backend.grounder.detect = Mock(side_effect=[GroundingError('timeout'), self.detection('', ['carrot'])])
        rows = self.backend.inference_batch(self.samples())
        self.assertTrue(rows[0]['steering']['degraded'])
        self.assertTrue(rows[1]['steering']['applied'])
        self.assertEqual(self.backend.grounder.detect.call_count, 2)

    def test_baseline_and_supplied_boxes_do_not_use_shared_detector(self):
        from grm_runtime.common import file_sha
        first, second = self.samples()
        rows = self.backend.inference_batch([first, {**second, 'condition': 'baseline'}, first])
        self.assertEqual(self.backend.grounder.detect.call_count, 1)
        self.assertFalse(rows[1]['steering']['applied'])
        self.assertTrue(rows[2]['steering']['applied'])
        self.assertFalse(self.backend.model.attention._forward_pre_hooks)
        supplied = {**second, 'grounding': {'after_cam_high': {
            'file_sha256': file_sha(second['image'][5]), 'bbox': [8, 8, 16, 16]}}}
        rows = self.backend.inference_batch([first, supplied])
        self.assertNotEqual(rows[0]['steering']['target_positions'], rows[1]['steering']['target_positions'])
        supplied['grounding']['after_cam_high']['file_sha256'] = 'stale'
        with self.assertRaises(ValueError):
            self.backend.inference_batch([first, supplied])
        self.assertFalse(self.backend.model.attention._forward_pre_hooks)

    def test_png_fast_path_and_conversion_preserve_rgb_pixels(self):
        rgb = Path(self.paths[0])
        data, size = png_bytes(rgb)
        self.assertEqual(data, rgb.read_bytes())
        self.assertEqual(size, (16, 16))
        for mode, extension in [('RGBA', 'png'), ('L', 'png'), ('RGB', 'jpg')]:
            with self.subTest(mode=mode, extension=extension):
                path = Path(self.tmp.name) / f'{mode}.{extension}'
                Image.new(mode, (23, 17)).save(path)
                data, size = png_bytes(path)
                with Image.open(path) as old, Image.open(io.BytesIO(data)) as new:
                    self.assertEqual(new.format, 'PNG')
                    self.assertEqual(new.mode, 'RGB')
                    self.assertEqual(new.tobytes(), old.convert('RGB').tobytes())
                    self.assertEqual(size, old.size)


class BatchedGenerationTests(unittest.TestCase):
    detection = PreparationTests.detection

    def samples(self):
        a, b = PreparationTests.samples(self)
        b['image'][2] = self.paths[7]  # different BEFORE pixels, same grid size
        return [a, b]

    def setUp(self):
        PreparationTests.setUp(self)
        self.backend.batch_size = 2

    def test_two_modes_one_generate_preserves_unpadded_inputs_and_masks(self):
        samples = self.samples()
        serial = [self.backend.inference_batch([s])[0] for s in samples]
        calls, decodes = self.backend.model.calls[:], self.backend.model.decode_masks[:]
        self.backend.model.calls.clear()
        self.backend.model.decode_masks.clear()
        self.backend.grounder.detect.reset_mock()
        rows = self.backend.inference_batch(samples)
        self.assertEqual(len(self.backend.model.calls), 1)
        self.assertEqual(self.backend.grounder.detect.call_count, 1)
        batched = self.backend.model.calls[0]
        for i, (old, new) in enumerate(zip(serial, rows)):
            diag = new['steering']
            pad = diag['padding_left']
            self.assertEqual(old['pred'], new['pred'])
            self.assertEqual(old['id'], new['id'])
            self.assertEqual(diag['batch_size'], 2)
            self.assertEqual(diag['batch_index'], i)
            self.assertTrue(torch.equal(calls[i][0], batched[0][i:i+1, pad:]))
            self.assertTrue(torch.equal(calls[i][1], batched[1][i*8:(i+1)*8]))
            self.assertTrue(torch.equal(calls[i][2], batched[2][i*8:(i+1)*8]))
            self.assertTrue(torch.equal(calls[i][3], batched[3][i:i+1, :, pad:, pad:]))
            self.assertTrue(torch.equal(decodes[i], self.backend.model.decode_masks[0][i:i+1, :, :, pad:]))
            for key in ('target_positions', 'negative_positions'):
                self.assertEqual(old['steering'][key], [p-pad for p in diag[key]])
            self.assertTrue(torch.isneginf(batched[3][i, :, :, :pad]).all())
        self.assertEqual(rows[0]['steering']['padding_left'], 0)
        self.assertEqual(rows[1]['steering']['padding_left'], 0)
        self.assertEqual(rows[0]['steering']['batch_id'], rows[1]['steering']['batch_id'])
        self.assertAlmostEqual(sum(r['steering']['grm_ms'] for r in rows), rows[0]['steering']['batch_grm_ms'])
        self.assertFalse(self.backend.model.attention._forward_pre_hooks)

    def test_unequal_lengths_are_regrouped_to_preserve_serial_scores(self):
        samples = PreparationTests.samples(self)
        serial = [self.backend.inference_batch([s])[0] for s in samples]
        self.backend.model.calls.clear()
        self.backend.grounder.detect.reset_mock()
        rows = self.backend.inference_batch(samples)
        self.assertEqual([r['pred'] for r in serial], [r['pred'] for r in rows])
        self.assertEqual([r['id'] for r in rows], ['forward', 'incremental'])
        self.assertEqual([c[0].shape[0] for c in self.backend.model.calls], [1, 1])
        self.assertEqual(self.backend.grounder.detect.call_count, 1)
        self.assertTrue(all(r['steering']['padding_left'] == 0 for r in rows))
        self.assertTrue(all(r['steering']['regroup_prepare_ms'] >= 0 for r in rows))

    def test_baseline_and_missing_row_do_not_receive_other_rows_bias(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                self.backend.model.calls.clear()
                self.backend.model.decode_masks.clear()
                a, b = self.samples()
                if missing:
                    b['target_queries'] = ['cube']
                    self.backend.grounder.detect = Mock(side_effect=[self.detection('', ['carrot']),
                        dict(selected=None, selection_status='ambiguous')])
                else:
                    b['condition'] = 'baseline'
                rows = self.backend.inference_batch([a, b])
                self.assertTrue(rows[0]['steering']['applied'])
                self.assertFalse(rows[1]['steering']['applied'])
                self.assertEqual(rows[1]['steering']['per_layer'], {})
                for mask in (self.backend.model.calls[-1][3], self.backend.model.decode_masks[-1]):
                    self.assertTrue(((mask[1] == 0) | torch.isneginf(mask[1])).all())
                    self.assertTrue((mask[0] == 6).any())

    def test_tail_batch_order_and_generation_error_cleanup(self):
        a, b = self.samples()
        rows = self.backend.inference_batch([a, b, {**a, 'id': 'tail'}])
        self.assertEqual([r['id'] for r in rows], ['forward', 'incremental', 'tail'])
        self.assertEqual([c[0].shape[0] for c in self.backend.model.calls], [2, 1])
        original = self.backend.model.generate
        self.backend.model.generate = Mock(side_effect=RuntimeError('generation failed'))
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            self.backend.inference_batch([a, b])
        self.assertFalse(self.backend.model.attention._forward_pre_hooks)
        self.backend.model.generate = original
        self.backend.inference_batch([{**a, 'condition': 'baseline'}, {**b, 'condition': 'baseline'}])
        mask = self.backend.model.calls[-1][3]
        self.assertTrue(((mask == 0) | torch.isneginf(mask)).all())

    def test_dual_monitor_batches_each_branch_once_and_publishes_together(self):
        from monitor_runtime.grm_backend import GRMMonitorBackend, _SubtaskState
        baseline = HFBackend.__new__(HFBackend)
        baseline.__dict__.update(self.backend.__dict__)
        baseline.config = {'enabled': False}
        baseline.model, baseline.processor, baseline.lock = Model(), Processor(), threading.RLock()
        baseline.grounder = None
        barrier = threading.Barrier(2)
        for backend in (self.backend, baseline):
            original = backend.model.generate
            def generate(original=original, **kw):
                barrier.wait(timeout=3)
                return original(**kw)
            backend.model.generate = generate
        root = Path(__file__).resolve().parents[1]
        monitor = GRMMonitorBackend(runtime_url='http://unused', goal_image=self.paths[0],
            output_root=self.tmp.name, inference_engine='hf', steering_config=str(root/'configs/steering.yaml'),
            model=self.backend, baseline_model=baseline, dual_branch=True, active_modes=['forward', 'incremental'])
        self.addCleanup(monitor.close)
        refs = dict(zip(('cam_high', 'cam_left_wrist', 'cam_right_wrist'), self.paths[:3]))
        current = dict(zip(refs, self.paths[3:6]))
        state = _SubtaskState('m', 'e', 'pick carrot', queries=['carrot'])
        state.ref_start = state.previous = refs
        monitor.sessions['m'] = state
        monitor._session_dir(state).mkdir(parents=True)
        monitor._snapshot_current = lambda *a, **kw: (current, {'identity': 'new'})
        record = monitor._run_one_step(state)
        self.assertEqual(state.step, 1)
        total = 0
        for name, backend in [('steering', self.backend), ('baseline', baseline)]:
            self.assertEqual(len(backend.model.calls), 1)
            self.assertEqual(backend.model.calls[0][0].shape[0], 2)
            modes = record['branches'][name]['modes']
            self.assertTrue(all(v['steering']['batch_size'] == 2 for v in modes.values()))
            self.assertTrue(all(v['steering']['applied'] == (name == 'steering') for v in modes.values()))
            total += modes['forward']['steering']['batch_grm_ms']
            self.assertFalse(backend.model.attention._forward_pre_hooks)
        self.assertAlmostEqual(record['timing']['grm_ms'], total)
        self.assertEqual(record['comparison']['difference'], 0)


if __name__ == '__main__':
    unittest.main()

"""Compare old/new CPU preparation on frozen online frames, without GPU inference.

Run with --session <directory containing manifest.json and online_pred.jsonl>.
Loads only the local GRM processor. Uses recorded SAM3 boxes; contacts no services.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grm_runtime.grounding import png_bytes
from grm_runtime.hf_backend import HFBackend, _BatchCache, infer_spans
from grm_runtime.prompt import messages
from monitor_runtime.grm_backend import build_online_samples


def old_png_bytes(path):
    with Image.open(path) as source:
        image = source.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return buffer.getvalue(), image.size


def benchmark(session, steps, repeats):
    manifest = json.loads((session / 'manifest.json').read_text())
    runtime = manifest['runtime']
    if runtime.get('engine') != 'hf':
        raise ValueError('Expected an HF monitor session')
    processor = AutoProcessor.from_pretrained(runtime['model_path'], local_files_only=True,
                                             trust_remote_code=True)
    for name in ('min_pixels', 'max_pixels'):
        setattr(processor.image_processor, name, runtime['processor'][name])
    region_model = HFBackend.__new__(HFBackend)
    region_model.config = runtime['steering_config']
    if region_model.config.get('intervention_labels') != ['after_cam_high']:
        raise ValueError('This replay benchmark requires intervention_labels: [after_cam_high]')
    region_model.merge = processor.image_processor.merge_size
    config = SimpleNamespace(image_token_id=runtime['model_config']['image_token_id'])
    rows = [json.loads(line) for line in (session / 'online_pred.jsonl').read_text().splitlines() if line.strip()][:steps]
    if not rows:
        raise ValueError('Session has no completed steps')
    start = {cam: str(session / 'reference' / f'{cam}.png') for cam in rows[0]['frames']}
    goals = list(session.parent.glob('reference_end.*'))
    if len(goals) != 1:
        raise ValueError('Expected one frozen reference_end image')
    previous = start
    old_times, new_times, png_old, png_new = [], [], [], []
    checked = 0
    for row in rows:
        samples = build_online_samples(manifest['subtask'], row['step'], start, str(goals[0]),
                                       previous, row['frames'], manifest['active_modes'])
        for sample in samples:
            sample['target_queries'] = manifest['target_queries']
        # Check transport pixel equivalence outside the timed region.
        for path in row['frames'].values():
            with Image.open(io.BytesIO(old_png_bytes(path)[0])) as old, Image.open(io.BytesIO(png_bytes(path)[0])) as new:
                assert old.size == new.size and old.tobytes() == new.tobytes()

        def prepare(shared):
            prepared = []
            batch = _BatchCache() if shared else None
            began = time.perf_counter()
            for sample in samples:
                recorded = row['modes'][sample['eval_mode']]['steering']['grounding']
                if 'after_cam_high' not in recorded:
                    raise ValueError('Step has no recorded SAM3 result (e.g. a transport timeout)')

                def detect(path, queries):
                    # Reproduce encoding overhead only, excluding HTTP and SAM3.
                    (png_bytes if shared else old_png_bytes)(path)
                    return copy.deepcopy(recorded['after_cam_high'])

                region_model.grounder = SimpleNamespace(detect=detect)
                images = []
                for path in sample['image']:
                    if batch is not None:
                        image, _ = batch.image(path)
                    else:
                        with Image.open(path) as source:
                            image = source.convert('RGB')
                    images.append(image)
                prompt = processor.apply_chat_template(messages(sample['task']), tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=images, return_tensors='pt')
                spans = infer_spans(inputs, config, sample['image'], region_model.merge)
                selected, target_spans, _, missing = region_model._regions(sample, spans, batch=batch)
                prepared.append((inputs, spans, selected, target_spans, missing))
            return time.perf_counter() - began, prepared

        prepare(False)
        prepare(True)
        for repeat in range(repeats):
            results = {}
            for shared in ((False, True) if repeat % 2 == 0 else (True, False)):
                elapsed, result = prepare(shared)
                (new_times if shared else old_times).append(elapsed)
                results[shared] = result
            for old, new in zip(results[False], results[True]):
                assert old[0].keys() == new[0].keys()
                assert all(torch.equal(old[0][key], new[0][key]) for key in old[0])
                assert old[1:] == new[1:]  # spans, bbox token indices and missing status
                checked += 1
            for operation, timings in ((old_png_bytes, png_old), (png_bytes, png_new)):
                began = time.perf_counter()
                operation(row['frames']['cam_high'])
                timings.append(time.perf_counter() - began)
        previous = row['frames']
    return dict(session=str(session), steps=len(rows), repeats=repeats,
                verified_sample_pairs=checked, pixel_and_processor_tensor_equality=True,
                bbox_token_alignment_equality=True,
                cpu_preparation_mean_s=dict(before=statistics.mean(old_times), after=statistics.mean(new_times)),
                main_camera_png_mean_s=dict(before=statistics.mean(png_old), after=statistics.mean(png_new)),
                excludes=['HTTP', 'SAM3 inference', 'GRM inference', 'robot control'],
                interval_s=dict(before=1.0, after=0.1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.repeats < 1:
        parser.error('--steps and --repeats must be positive')
    torch.set_num_threads(2)
    report = benchmark(args.session.resolve(), args.steps, args.repeats)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + '\n')
    print(rendered)

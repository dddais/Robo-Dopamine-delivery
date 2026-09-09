"""Opt-in real GRM serial/batch comparison on frozen frames and recorded SAM3 boxes.

Loads one local checkpoint on --device, never contacts Monitor/SAM3/robot services.
Reports score differences as well as timings; shared GPU load affects timings.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grm_runtime.hf_backend import HFBackend
from monitor_runtime.grm_backend import build_online_samples


def validate(session, device, steps, padding_probe=False):
    manifest = json.loads((session/'manifest.json').read_text())
    runtime = manifest['runtime']
    records = [json.loads(s) for s in (session/'online_pred.jsonl').read_text().splitlines() if s.strip()][:steps]
    detections = {}
    for record in records:
        for mode in record['modes'].values():
            paths = {s['label']: str(Path(s['path']).resolve()) for s in mode['steering']['spans']}
            for label, row in mode['steering']['grounding'].items():
                detections[paths[label]] = row
    class Grounder:
        def detect(self, path, queries):
            assert list(queries) == manifest['target_queries']
            return deepcopy(detections[str(Path(path).resolve())])
    model = HFBackend(runtime['model_path'], steering_config=runtime['steering_config'], device=device,
                      max_new_tokens=runtime['max_new_tokens'], grounding_client=Grounder())
    start = {cam: str(session/'reference'/f'{cam}.png') for cam in records[0]['frames']}
    goal, = session.parent.glob('reference_end.*')
    previous, comparisons = start, []

    def infer(samples, batch_size):
        model.batch_size = batch_size
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        rows = model.inference_batch(samples)
        torch.cuda.synchronize(device)
        return rows, time.perf_counter()-began

    for record in records:
        samples = build_online_samples(manifest['subtask'], record['step'], start, str(goal),
                                       previous, record['frames'], ['forward', 'incremental'])
        for s in samples:
            s['target_queries'] = manifest['target_queries']
        conditions = ['baseline', 'candidate_target']
        if padding_probe and record is records[0]:
            conditions.append('mixed_padding')
        for condition in conditions:
            batch = [{**s, 'condition': condition} for s in samples]
            if condition == 'mixed_padding':
                batch[0]['condition'] = 'candidate_target'
                batch[1]['condition'] = 'baseline'
                batch[1]['task'] += ' Keep the movement slow and steady.'
            if record is records[0]:
                infer(batch, 2)  # exclude first-use warmup from comparisons
            outcomes = {}
            for size in ((1, 2) if record['step'] % 2 == 0 else (2, 1)):
                outcomes[size] = infer(batch, size)
            serial, serial_s = outcomes[1]
            batched, batched_s = outcomes[2]
            if condition == 'mixed_padding':
                assert all(r['steering']['batch_size'] == 1 for r in batched)
                assert batched[0]['steering']['applied'] and not batched[1]['steering']['applied']
                assert [r['pred'] for r in serial] == [r['pred'] for r in batched]
            else:
                assert len({r['steering']['batch_id'] for r in batched}) == 1
            for a, b in zip(serial, batched):
                assert a['id'] == b['id'] and a['valid'] and b['valid'], (a['pred'], b['pred'])
                old, new = a['steering'], b['steering']
                assert old['applied'] == new['applied'] and old['degraded'] == new['degraded']
                for key in ('target_positions', 'negative_positions'):
                    assert old.get(key, []) == [v-new['padding_left'] for v in new.get(key, [])]
            assert all(not layer.self_attn._forward_pre_hooks for layer in model.layers)
            comparisons.append(dict(step=record['step'], condition=condition,
                serial_s=serial_s, batched_s=batched_s,
                serial_grm_ms=sum(r['steering']['grm_ms'] for r in serial),
                batched_grm_ms=sum(r['steering']['grm_ms'] for r in batched),
                serial_preds=[r['pred'] for r in serial], batched_preds=[r['pred'] for r in batched],
                max_score_delta=max(abs(a['parsed_score']-b['parsed_score']) for a,b in zip(serial,batched))))
            print(json.dumps(comparisons[-1]), flush=True)
        previous = record['frames']
    return dict(session=str(session), device=device, comparisons=comparisons,
                serial_mean_s=statistics.mean(r['serial_s'] for r in comparisons),
                batched_mean_s=statistics.mean(r['batched_s'] for r in comparisons),
                max_score_delta=max(r['max_score_delta'] for r in comparisons),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3,
                uses_recorded_grounding=True, real_dual_gpu_concurrency=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--padding-probe', action='store_true',
                        help='Also compare a synthetic unequal-prompt, mixed steering/baseline batch')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(2)
    report = validate(args.session.resolve(), args.device, args.steps, args.padding_probe)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'comparisons'}, indent=2))

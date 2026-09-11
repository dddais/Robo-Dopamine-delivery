"""Replay recorded baseline scores with a clean GRM model and frozen inputs.

Uses local files only; never contacts Monitor, SAM3, or robot services. Select an
idle GPU with CUDA_VISIBLE_DEVICES before running. Does not load steering heads.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grm_runtime.hf_backend import HFBackend
from monitor_runtime.grm_backend import build_online_samples


def validate(run, device, samples_per_session):
    import torch
    torch.set_num_threads(2)
    paths = sorted(run.glob('*/online_pred.jsonl'))
    if not paths:
        raise ValueError('run has no recorded sessions')
    model, model_manifest, results = None, None, []
    for path in paths:
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        expected = manifest['baseline_runtime']
        if expected['steering_config'].get('enabled'):
            raise ValueError('recorded baseline has steering enabled')
        if model is None:
            processor = expected['processor']
            model = HFBackend(expected['model_path'], steering_config=None, device=device,
                dtype=expected['dtype'], max_new_tokens=expected['max_new_tokens'],
                batch_size=expected.get('batch_size', 2),
                min_pixels=processor.get('min_pixels', 12544), max_pixels=processor.get('max_pixels', 76800))
            model_manifest = expected
            # JSON journals stringify dictionary keys and convert tuples to lists.
            actual = json.loads(json.dumps(model.manifest))
            for key in ('model_config', 'processor', 'prompt_sha256', 'dtype', 'decoding', 'attention_implementation'):
                if actual.get(key) != expected.get(key):
                    def differences(a, b, prefix):
                        if isinstance(a, dict) and isinstance(b, dict):
                            return [d for k in sorted(set(a) | set(b)) for d in differences(a.get(k), b.get(k), prefix + '.' + k)]
                        return [] if a == b else [{"key": prefix, "replay": a, "recorded": b}]
                    print(json.dumps(differences(actual.get(key), expected.get(key), key)), flush=True)
                    raise ValueError(f'replay environment differs from recorded {key}')
        elif expected != model_manifest:
            raise ValueError('run contains different baseline configurations')
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not records:
            continue
        # Keep the original batch of modes and original previous frame, even
        # when selecting nonconsecutive inference steps for replay.
        count = min(samples_per_session, len(records))
        selected = sorted({round(i * (len(records)-1) / max(count-1, 1)) for i in range(count)})
        for index in selected:
            record = records[index]
            baseline = record['branches']['baseline']['modes']
            spans = next(iter(baseline.values()))['steering']['spans']
            labeled = {span['label']: span['path'] for span in spans}
            reference = {camera: str(path.parent / 'reference' / f'{camera}.png') for camera in record['frames']}
            samples = build_online_samples(manifest['subtask'], record['step'], reference,
                labeled['reference_end'], reference if index == 0 else records[index-1]['frames'],
                record['frames'], manifest['active_modes'])
            for sample in samples:
                sample['condition'] = 'baseline'
                sample['target_queries'] = manifest['target_queries']
                recorded_paths = [span['path'] for span in baseline[sample['eval_mode']]['steering']['spans']]
                if [str(Path(p).resolve()) for p in sample['image']] != [str(Path(p).resolve()) for p in recorded_paths]:
                    raise ValueError('reconstructed baseline images differ from recorded inputs')
            replay = model.inference_batch(samples)
            if any(layer.self_attn._forward_pre_hooks for layer in model.layers):
                raise AssertionError('clean baseline has attention hooks')
            for row in replay:
                saved = baseline[row['eval_mode']]
                diag = row['steering']
                if diag['enabled'] or diag['applied'] or diag['per_layer'] or diag['grounding']:
                    raise AssertionError('clean baseline received steering')
                result = dict(session=path.parent.name, step=record['step'], mode=row['eval_mode'],
                    recorded_pred=saved['pred'], replay_pred=row['pred'],
                    exact_match=saved['pred'] == row['pred'],
                    score_delta=abs(saved['score']-row['parsed_score']) if row['valid'] else None)
                results.append(result)
                print(json.dumps(result), flush=True)
    return dict(run=str(run), device=device, local_files_only=True, real_model=True,
                steering_enabled=False, comparisons=results,
                exact_matches=sum(r['exact_match'] for r in results), total=len(results),
                max_score_delta=max((r['score_delta'] for r in results if r['score_delta'] is not None), default=None),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples-per-session', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_session < 1:
        parser.error('--samples-per-session must be positive')
    report = validate(args.run.resolve(), args.device, args.samples_per_session)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'comparisons'}), flush=True)
    if report['exact_matches'] != report['total']:
        raise SystemExit('Replay differed; inspect the saved report')

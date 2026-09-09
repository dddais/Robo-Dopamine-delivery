"""Offline SAM3 detector/tracker timing; never contacts a robot or live service."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--query', default='carrot')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tracker', action='store_true')
    parser.add_argument('--tracker-only', action='store_true')
    parser.add_argument('--tracker-frames', type=int, default=None)
    args = parser.parse_args()
    import torch
    from transformers.utils import logging
    logging.disable_progress_bar()
    from PIL import Image
    from sam3_runtime.detector import SAM3Detector, bypass_mask_decoder
    images = [Image.open(p).convert('RGB') for p in args.images]
    report = {'device': args.device, 'gpu': torch.cuda.get_device_name(args.device),
              'images': args.images, 'query': args.query, 'variants': {}}
    detector = SAM3Detector(args.model_path, device=args.device, profile=True,
        dtype='bfloat16' if args.tracker_only else 'float32', bbox_only=args.tracker_only)
    for variant in (() if args.tracker_only else ('fp32_full', 'fp32_bbox', 'bf16_bbox')):
        if variant == 'fp32_bbox':
            bypass_mask_decoder(detector.model, torch)
        if variant == 'bf16_bbox':
            detector.model.to(dtype=torch.bfloat16)
            detector.dtype = torch.bfloat16
        detector.detect(images[0], [args.query])
        records = []
        for _ in range(args.repeats):
            for image in images:
                started = time.monotonic()
                boxes = detector.detect(image, [args.query])
                records.append({'elapsed_ms': (time.monotonic()-started)*1000,
                                'timing': detector.last_timing, 'boxes': boxes})
        report['variants'][variant] = {'mean_ms': statistics.mean(r['elapsed_ms'] for r in records),
                                      'records': records}
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(variant, report['variants'][variant]['mean_ms'], flush=True)
    if args.tracker or args.tracker_only:
        from sam3_runtime.tracker import SAM3VideoTracker, TrackingEngine
        tracker = SAM3VideoTracker(args.model_path, args.device)
        engine = TrackingEngine(detector, tracker, max_gap_s=30., redetect_interval_s=60.)
        rows = []
        for i in range(args.tracker_frames or len(images) * args.repeats):
            image = images[i % len(images)]
            result = engine.update('offline-benchmark', image, [args.query], str(i))
            rows.append(result)
            session = engine.sessions['offline-benchmark'].session
            if session is not None:
                assert len(session.processed_frames) <= tracker.memory_frames + 1
                assert all(len(o['non_cond_frame_outputs']) <= tracker.memory_frames
                           for o in session.output_dict_per_obj.values())
            print('tracker', i, result['source'], result['latency_ms'], result['selection_status'], flush=True)
        report['tracker'] = rows
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated(args.device) / 1024**2
        Path(args.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

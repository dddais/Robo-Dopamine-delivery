"""Paired HF baseline/steering trajectories, using the test_data_suc three-mode protocol."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import uuid
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from grm_runtime.common import CONDITIONS, MODES, file_sha, load_yaml, progress_step, resolve_path, target_queries
from grm_runtime.hf_backend import HFBackend
from grm_runtime.grounding import GroundingClient
from examples.inference import build_samples_json, get_frame_count, make_sample_indices_by_interval, save_frames, plot_video_reward

CONFIG_KEYS = {"model_path", "data_dir", "output_root", "task", "target_queries", "goal_image",
               "frame_interval", "modes", "conditions", "steering_config", "device", "max_new_tokens",
               "visualize", "fps", "min_pixels", "max_pixels"}


def read_config(path):
    cfg, base = load_yaml(path, CONFIG_KEYS)
    for key in ("model_path", "data_dir", "output_root", "goal_image", "steering_config"):
        cfg[key] = resolve_path(cfg[key], base)
    if cfg.get("frame_interval", 20) <= 0:
        raise ValueError("frame_interval must be positive")
    for key, allowed in (("modes", MODES), ("conditions", CONDITIONS)):
        cfg.setdefault(key, list(allowed))
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or set(cfg[key]) - set(allowed):
            raise ValueError(f"Invalid {key}")
    return cfg


def run(cfg, model=None):
    root = Path(cfg["output_root"]) / (time.strftime("%y-%m-%d-%H-%M-%S") + "_" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    camera_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
    sources = [Path(cfg["data_dir"]) / f"{name}.mp4" for name in camera_names]
    sources = [p if p.exists() else p.with_suffix("") for p in sources]
    counts = [get_frame_count(p) for p in sources]
    if len({count for _, count in counts}) != 1 or counts[0][1] < 2:
        raise ValueError(f"Need at least two aligned frames from three views: {counts}")
    if not Path(cfg["goal_image"]).is_file():
        raise ValueError("goal_image must exist; use blank_goal.png explicitly if needed")
    fps_list = []
    for p, (kind, _) in zip(sources, counts):
        if kind == "video":
            cap = cv2.VideoCapture(str(p)); fps_list.append(cap.get(cv2.CAP_PROP_FPS)); cap.release()
        else:
            fps_list.append(float(cfg.get("fps", 0)))
    if min(fps_list) <= 0 or max(fps_list) - min(fps_list) > 0.01:
        raise ValueError("Three views need matching FPS; supply fps for PNG directories")
    indices = make_sample_indices_by_interval(counts[0][1], cfg.get("frame_interval", 20))
    for name, p, (kind, _) in zip(camera_names, sources, counts):
        save_frames(p, root / ".cache" / name, indices, kind)
    frozen_goal = root / '.cache' / ('reference_end' + Path(cfg['goal_image']).suffix)
    shutil.copyfile(cfg['goal_image'], frozen_goal)
    model = model or HFBackend(cfg["model_path"], steering_config=cfg["steering_config"],
        device=cfg.get("device", "cuda:0"), max_new_tokens=cfg.get("max_new_tokens",64),
        min_pixels=cfg.get("min_pixels",12544), max_pixels=cfg.get("max_pixels",76800))
    if isinstance(model.grounder, GroundingClient):
        model.grounder.cache_dir = root / "grounding_cache"
    queries = target_queries(cfg["task"], cfg.get("target_queries"), model.config.get("task_queries"))
    manifest = {"config": cfg, "runtime": model.manifest, "sample_indices": indices, "fps": fps_list,
                "inputs": [{"path": str(p), "frame_count": count} for p, (_,count) in zip(sources,counts)],
                "goal_sha256": file_sha(cfg["goal_image"]), "target_queries": queries}
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2))
    curves, rows, invalid, degraded, grounded_keys = {}, [], 0, 0, set()
    for mode in cfg["modes"]:
        samples = build_samples_json(root, cfg["task"], indices, str(frozen_goal), mode)
        for sample, after in zip(samples, indices[1:]):
            sample.update(eval_mode=mode, target_queries=queries, after_frame_id=after, time_s=after/fps_list[0])
        for condition in cfg["conditions"]:
            destination = root/condition/mode; destination.mkdir(parents=True)
            (destination/"sample.json").write_text(json.dumps(samples,indent=2))
            results, previous, count, chain_valid = [], 0., 0, True
            for sample in samples:
                request = {**sample, "condition": condition}
                try:
                    result = model.inference_batch([request])[0]
                except Exception as exc:
                    result = {**request, "pred": "", "valid": False, "error": str(exc)}
                if not result.get("valid", False):
                    invalid += 1
                    if mode == "incremental":
                        chain_valid = False
                if result.get("valid", False) and chain_valid:
                    stats = progress_step(mode, result["parsed_score"], previous, count)
                    result.update(stats); previous = stats["progress"]; count += 1
                else:
                    result.update(progress=None, hop=None)
                degraded += int(result.get("steering",{}).get("degraded",False))
                results.append(result)
                rows.append({"condition":condition,"mode":mode,"frame_id":sample["after_frame_id"],
                             "time_s":sample["time_s"],"score":result.get("parsed_score"),"progress":result["progress"],
                             "valid":result.get("valid",False),"applied":result.get("steering",{}).get("applied",False)})
                # Append every step immediately so interrupted runs retain diagnostics.
                with (destination/"predictions.jsonl").open('a') as stream:
                    stream.write(json.dumps(result)+'\n')
                for label, grounding in result.get("steering",{}).get("grounding",{}).items():
                    key = (grounding.get("image_sha256"), label)
                    if key in grounded_keys:
                        continue
                    grounded_keys.add(key)
                    with (root/"grounding.jsonl").open('a') as stream:
                        stream.write(json.dumps({"label":label, **grounding})+'\n')
                    selected = grounding.get("selected")
                    if selected:
                        from grm_runtime.prompt import IMAGE_LABELS
                        with Image.open(sample["image"][IMAGE_LABELS.index(label)]) as image:
                            image = image.convert("RGB")
                        draw = ImageDraw.Draw(image); draw.rectangle(selected["bbox"],outline="red",width=3)
                        overlay = root/"bbox_overlays";overlay.mkdir(exist_ok=True)
                        image.save(overlay/f'{sample["after_frame_id"]}_{label}.png')
            (destination/"pred_vllm.json").write_text(json.dumps(results,indent=2))
            curves[(condition,mode)] = [r["progress"] for r in results]
            if cfg.get("visualize",True) and all(r["progress"] is not None for r in results):
                plot_video_reward(destination)
    for condition in cfg["conditions"]:
        fused=[]
        for i, after in enumerate(indices[1:]):
            values=[curves[(condition,m)][i] for m in cfg["modes"]]
            value = None if any(v is None for v in values) else sum(values)/len(values)
            fused.append(None if value is None else max(0.,min(1.,value)))
            rows.append({"condition":condition,"mode":"fused","frame_id":after,"time_s":after/fps_list[0],
                         "score":None,"progress":fused[-1],"valid":value is not None,"applied":None})
        curves[(condition,"fused")]=fused
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(cfg["modes"])+1, 1, figsize=(12,4*(len(cfg["modes"])+1)), squeeze=False)
    for ax, mode in zip(axes[:,0], [*cfg["modes"],"fused"]):
        for condition in cfg["conditions"]:
            ax.plot([i/fps_list[0] for i in indices[1:]],curves[(condition,mode)],label=condition)
        ax.set(title=mode,xlabel="Time (s)",ylabel="Progress");ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(root/"progress_curve.png");plt.close(fig)
    with (root/"curve.csv").open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    predictions=[r for r in rows if r['mode']!='fused']
    interventions=[r for r in predictions if r['condition']!='baseline']
    applied=sum(bool(r['applied']) for r in interventions)
    summary={"invalid_predictions":invalid,"degraded_predictions":degraded,"prediction_count":len(predictions),
             "grounded_frames":len(grounded_keys),"complete":invalid==0,
             "steering_requested_count":len(interventions),"steering_applied_count":applied,
             "formal_scoring_ready":invalid==0 and degraded==0 and applied==len(interventions)}
    (root/"summary.json").write_text(json.dumps(summary,indent=2))
    print(f"Offline steering completed: {root} ({summary})",flush=True)
    return root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args(argv)
    root=run(read_config(args.config))
    if not json.loads((root/'summary.json').read_text())["complete"]:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

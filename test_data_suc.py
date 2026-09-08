"""
用预训练 GRM 模型测试（抓取胡萝卜放在黄色盘子上）。
运行三种模式推理，并绘制进度曲线图。
"""
import os
import sys
import json
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from examples.inference import GRMInference

# ============================
# 配置
# ============================
MODEL_PATH = "/path/to/Robo-Dopamine/train/checkpoints/my_carrot_finetune_big"
# MODEL_PATH = "/path/to/Robo-Dopamine/train/checkpoints/sub1_approach_grasp_finetune"
# MODEL_PATH = './pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview'
DATA_DIR = "/path/to/Robo-Dopamine/aligned_data/pick3suc_1"
OUTPUT_ROOT = "./results/pick_3_fail/exp_suc1_inter20_ckpt"

TASK_INSTRUCTION = "pick the carrot and put it on yellow plate "
GOAL_IMAGE = "./examples/blank_goal.png"
# GOAL_IMAGE = "./examples/exp_suc_1.png"
INTERVAL = 20
# ============================
# 加载模型
# ============================
print(f"Loading model: {MODEL_PATH} ...")
model = GRMInference(MODEL_PATH)
print("Model loaded successfully!")

# ============================
# 运行推理（无目标图 - 使用 blank_goal）
# ============================
results = {}
for mode in ["forward", "incremental", "backward"]:
    print(f"\n{'='*60}")
    print(f"Running {mode}-mode inference (no goal image)...")
    print(f"{'='*60}")

    try:
        output_dir = model.run_pipeline(
            cam_high_path  = os.path.join(DATA_DIR, "cam_high.mp4"),
            cam_left_path  = os.path.join(DATA_DIR, "cam_left_wrist.mp4"),
            cam_right_path = os.path.join(DATA_DIR, "cam_right_wrist.mp4"),
            out_root       = OUTPUT_ROOT,
            task           = TASK_INSTRUCTION,
            frame_interval = INTERVAL,
            batch_size     = 1,
            goal_image     = GOAL_IMAGE,
            eval_mode      = mode,
            visualize      = True
        )
        results[mode] = output_dir
        print(f"[{mode}] Output at: {output_dir}")
    except Exception as e:
        print(f"[{mode}] FAILED: {e}")
        import traceback
        traceback.print_exc()

# ============================
# 分析结果 & 绘图
# ============================
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")

all_progress = {}
for mode, path in results.items():
    pred_file = os.path.join(path, "pred_vllm.json")
    if not os.path.exists(pred_file):
        continue
    with open(pred_file) as f:
        data = json.load(f)

    pcts = []
    for d in data:
        m = re.search(r'([+-]?\d+\.?\d*)%', d['pred'])
        if m:
            pcts.append(float(m.group(1)))

    progress = [d['progress'] * 100 for d in data]  # 转为百分比
    all_progress[mode] = progress

    print(f"\n[{mode.upper()}]")
    print(f"  Samples: {len(data)}")
    print(f"  Score range: [{min(pcts):.1f}%, {max(pcts):.1f}%]")
    print(f"  Mean score: {sum(pcts)/len(pcts):.1f}%")
    print(f"  Final progress: {progress[-1]:.1f}%")

# 计算平均进度
if all_progress:
    modes = list(all_progress.keys())
    n = len(all_progress[modes[0]])
    avg_progress = []
    for i in range(n):
        avg_progress.append(sum(all_progress[m][i] for m in modes) / len(modes))

    all_progress["avg"] = avg_progress

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(12, 6))

    # 时间轴：采样点对应的帧号（每个点间隔 frame_interval 帧，30fps）
    frame_interval = 10
    fps = 30
    frames = [i * frame_interval for i in range(n)]
    time_sec = [f / fps for f in frames]

    colors = {
        "forward": "#2196F3",
        "incremental": "#FF9800",
        "backward": "#4CAF50",
        "avg": "#E91E63",
    }
    labels = {
        "forward": "Forward",
        "incremental": "Incremental",
        "backward": "Backward",
        "avg": "Average (Fused)",
    }
    styles = {
        "forward": "-",
        "incremental": "-",
        "backward": "-",
        "avg": "--",
    }
    widths = {
        "forward": 2.0,
        "incremental": 2.0,
        "backward": 2.0,
        "avg": 3.0,
    }

    for key in ["forward", "incremental", "backward", "avg"]:
        if key in all_progress:
            ax.plot(
                time_sec, all_progress[key],
                styles[key],
                color=colors[key],
                linewidth=widths[key],
                label=f"{labels[key]}  (final={all_progress[key][-1]:.1f}%)",
                marker="o" if key == "avg" else None,
                markersize=3,
            )

    ax.set_xlabel("Time (s)", fontsize=13)
    ax.set_ylabel("Task Progress (%)", fontsize=13)
    ax.set_title(
        f"GRM Progress Curve — \"{TASK_INSTRUCTION}\"\n"
        f"Model: {os.path.basename(MODEL_PATH)}",
        fontsize=14,
    )
    ax.legend(fontsize=11, loc="best")
    ax.set_xlim(0, time_sec[-1])
    ax.set_ylim(min(-5, ax.get_ylim()[0]), max(105, ax.get_ylim()[1]))
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(y=100, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # # 保存到第一个模式的输出目录下
    # first_path = results[modes[0]]
    # save_path = os.path.join(first_path, "progress_curve.png")
    # fig.savefig(save_path, dpi=150)
    # print(f"\nProgress curve saved to: {save_path}")

    # 同时保存一份到 results 根目录
    save_path2 = os.path.join(OUTPUT_ROOT, "progress_curve.png")
    fig.savefig(save_path2, dpi=150)
    print(f"Also saved to: {save_path2}")

print(f"\n{'='*60}")
print("Done!")

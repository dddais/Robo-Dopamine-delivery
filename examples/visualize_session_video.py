#!/usr/bin/env python3
"""
将 online_pred.jsonl 推理结果可视化为 MP4 视频。

视频布局:
  上方: 文字信息条 + 三视角图像 (cam_high | cam_left_wrist | cam_right_wrist)
  下方: 进度曲线图 (纵轴: 进度%, 横轴: 全局帧号), 曲线随帧逐步生长

用法:
  conda run -n robo-dopamine python visualize_session_video.py \
      --session_dir <session_dir> \
      [--fps 5] \
      [--output video.mp4]
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from moviepy import ImageSequenceClip
except ImportError:
    from moviepy.editor import ImageSequenceClip


# ── 配色 ──
SUBTASK_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c5644", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
FIG_BG = "#16213e"


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

DEFAULT_SESSION_DIR = "/path/to/Robo-dopamine/examples/results/online_local_img_agent/26-06-10-16-27-34_agent_session"

def main():
    parser = argparse.ArgumentParser(description="将推理 session 可视化为 MP4 视频")
    parser.add_argument("--session_dir", type=str, default= DEFAULT_SESSION_DIR,
                        help="session 目录路径 (包含 online_pred.jsonl)")
    parser.add_argument("--fps", type=int, default=2,
                        help="输出视频帧率 (默认 5)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 MP4 路径 (默认保存在 session_dir 下)")
    parser.add_argument("--chart_height", type=int, default=300,
                        help="曲线图高度像素 (默认 300)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="单视角图像缩放尺寸 (默认 224)")
    parser.add_argument("--text_bar_h", type=int, default=52,
                        help="顶部文字条高度像素 (默认 52)")
    args = parser.parse_args()

    session_dir = args.session_dir
    jsonl_path = os.path.join(session_dir, "online_pred.jsonl")
    if not os.path.isfile(jsonl_path):
        print(f"错误: 找不到 {jsonl_path}")
        sys.exit(1)

    # 加载数据
    records = load_jsonl(jsonl_path)
    n_frames = len(records)
    print(f"加载 {n_frames} 条记录")

    # 如果记录中没有 subtask_idx，则根据 subtask 名称自动分配
    if records and "subtask_idx" not in records[0]:
        subtask_name_to_idx = {}
        for r in records:
            name = r.get("subtask", "")
            if name not in subtask_name_to_idx:
                subtask_name_to_idx[name] = len(subtask_name_to_idx)
            r["subtask_idx"] = subtask_name_to_idx[name]

    subtask_indices = sorted(set(r["subtask_idx"] for r in records))
    print(f"子任务: {subtask_indices}")

    cam_names = ["cam_left_wrist", "cam_high", "cam_right_wrist"]

    # ── 布局尺寸 (与 inference.py plot_video_reward 一致) ──
    W_panel, H_panel = args.img_size, args.img_size
    W_total = 3 * W_panel
    H_text = args.text_bar_h
    H_plot = args.chart_height
    H_total = H_text + H_panel + H_plot

    print(f"视频尺寸: {W_total}x{H_total}, FPS: {args.fps}")

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(session_dir, "visualization.mp4")
    print(f"输出: {output_path}")

    # 预计算子任务数据
    subtask_data = {}
    for st_idx in subtask_indices:
        sub_recs = [r for r in records if r["subtask_idx"] == st_idx]
        subtask_data[st_idx] = {
            "global_frames": [i for i, r in enumerate(records) if r["subtask_idx"] == st_idx],
            "progress":      [r.get("progress_percent", 0) for r in sub_recs],
            "name":          sub_recs[0]["subtask"] if sub_recs else "",
        }

    max_progress = max(r.get("progress_percent", 0) for r in records)

    # ── 逐帧构建 ──
    frames_buffer = []

    for i in range(n_frames):
        rec = records[i]
        frame_paths = rec.get("frames", {})
        subtask = rec.get("subtask", "")
        step = rec.get("step", 0)
        progress = rec.get("progress_percent", 0)

        # ── 1. 文字条 ──
        text_bar = np.full((H_text, W_total, 3), 30, dtype=np.uint8)
        line1 = f"Task {rec['subtask_idx']}: {subtask}"
        line2 = f"Step: {step}  |  Progress: {progress:.1f}%  |  Frame: {i}/{n_frames - 1}"
        cv2.putText(text_bar, line1, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(text_bar, line2, (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 240), 1, cv2.LINE_AA)

        # ── 2. 三视角图像 ──
        panels = []
        for cam in cam_names:
            p = frame_paths.get(cam)
            img = cv2.imread(p) if p and os.path.isfile(p) else None
            if img is None:
                img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
            if img.shape[:2] != (H_panel, W_panel):
                img = cv2.resize(img, (W_panel, H_panel))
            panels.append(img)
        img_row = np.hstack(panels)

        # ── 3. 曲线图 (参考 inference.py plot_video_reward) ──
        dpi = 100
        fig = plt.figure(figsize=(W_total / dpi, H_plot / dpi), dpi=dpi)

        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # Plot 1: Hop (每个 mode 的 hop)
        ax1.set_title("Instant Hop per Mode", fontsize=10)
        ax1.set_xlabel("Frame", fontsize=9)
        ax1.set_ylabel("Hop", fontsize=9)
        ax1.set_xlim(0, max(n_frames - 1, 1))
        ax1.set_ylim(-1.1, 1.1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        mode_colors_hop = {"forward": "#2ca02c", "incremental": "#1f77b4", "backward": "#d62728"}
        for mode_name, color in mode_colors_hop.items():
            mode_hops = []
            for j in range(i + 1):
                h = records[j].get("modes", {}).get(mode_name, {}).get("hop", 0)
                mode_hops.append(h)
            ax1.plot(range(i + 1), mode_hops, "-", color=color, alpha=0.7,
                     linewidth=1.5, label=mode_name)
        ax1.legend(fontsize=8, loc="upper right")

        # Plot 2: Accumulated Progress
        ax2.set_title("Accumulated Progress", fontsize=10)
        ax2.set_xlabel("Frame", fontsize=9)
        ax2.set_ylabel("Progress (%)", fontsize=9)
        ax2.set_xlim(0, max(n_frames - 1, 1))
        ax2.set_ylim(-2, max(max_progress * 1.2, 15))
        ax2.grid(True, linestyle="--", alpha=0.5)

        for st_idx in subtask_indices:
            sd = subtask_data[st_idx]
            gf = sd["global_frames"]
            prog = sd["progress"]
            color = SUBTASK_COLORS[st_idx % len(SUBTASK_COLORS)]
            label = f"Task {st_idx}: {sd['name'][:25]}"

            vis_gf = [g for g in gf if g <= i]
            vis_prog = [prog[j] for j, g in enumerate(gf) if g <= i]
            if vis_gf:
                if len(vis_gf) > 1:
                    ax2.plot(vis_gf, vis_prog, "-", color=color, alpha=0.6, linewidth=1.5)
                ax2.scatter(vis_gf, vis_prog, c=[color] * len(vis_gf), s=20,
                            edgecolors="k", linewidths=0.4, label=label, zorder=5)

        ax2.legend(fontsize=7, loc="upper left")

        fig.tight_layout(pad=1.0)
        fig.canvas.draw()

        # 参考代码的方式: np.asarray + INTER_AREA 下采样
        plot_img = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plot_img = cv2.resize(plot_img, (W_total, H_plot), interpolation=cv2.INTER_AREA)

        # ── 4. 拼接 (RGB) ──
        frame_comp = np.vstack([text_bar, img_row, plot_img])
        if frame_comp.shape[:2] != (H_total, W_total):
            frame_comp = cv2.resize(frame_comp, (W_total, H_total))

        # moviepy 需要 RGB
        frame_comp = cv2.cvtColor(frame_comp, cv2.COLOR_BGR2RGB)
        frames_buffer.append(frame_comp)

        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"  [{i + 1}/{n_frames}] ({(i + 1) / n_frames * 100:.0f}%)")

    # ── 写视频 (moviepy) ──
    if frames_buffer:
        clip = ImageSequenceClip(frames_buffer, fps=float(args.fps))
        clip.write_videofile(str(output_path), logger=None)
        print(f"完成! 视频已保存: {output_path}")
    else:
        print("[WARN] 没有生成任何帧")


if __name__ == "__main__":
    main()

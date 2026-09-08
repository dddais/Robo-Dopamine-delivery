# Robo-Dopamine Monitor 说明

在线接入优先阅读 [在线任务进度监控：实现、接口与用法](docs/online_attention_steering.md)，覆盖当前 GRM/HF/SAM3 流程、HTTP 字段、客户端示例与生命周期。

Attention steering 的新增离线/在线流程、SAM3 启动方式和验证结果见 [README_steering.md](README_steering.md)。

本文档面向当前仓库的 monitor / GRM 使用，重点说明离线进度评分、HTTP monitor 服务、配置方式、代码结构和可扩展点。文档只描述当前代码中已经存在并可直接定位的入口。

原仓库完整 README 见 [README.md](README.md)。本文档只补充 monitor 相关流程。

## 整体功能

Robo-Dopamine 使用 GRM (General Reward Model) 判断机器人操作任务的进度。模型输入为任务描述和 8 张图像：

1. reference start 正面图
2. reference end / goal 图
3. before 正面图
4. before 左腕图
5. before 右腕图
6. after 正面图
7. after 左腕图
8. after 右腕图

模型输出 `<score>+NN%</score>`、`<score>-NN%</score>` 或 `<score>0%</score>`。代码会进一步转换为：

- `hop`：当前 step 的进退变化。
- `progress`：累计任务进度。
- monitor 状态：`running` / `success` / `failed`。

当前相关能力：

- 离线三视角视频评分：用 `test_data_suc.py` 一次运行 `forward`、`incremental`、`backward` 三种模式，并输出融合进度曲线。
- HTTP Monitor Service：提供 `/monitors/start`、`/monitors/status`、`/monitors/stop`，供上游 Runtime / adapter 调用。
- monitor 后端：
  - `grm`：从 Robot Runtime 拉取三视角 JPEG，后台持续跑 GRM 推理。
  - `deterministic`：轻量连通性后端，可配置 poll 若干次后自动成功。
- 数据构造与微调：保留在 `dataset/` 和 `train/`，用于后续新增任务适配。

## 环境安装

建议使用 Python 3.10 和 NVIDIA GPU。当前 `requirements.txt` 包含推理和服务依赖，例如 `torch`、`vllm`、`transformers`、`fastapi`、`uvicorn` 等。

```bash
cd Robo-Dopamine
conda create -n robo-dopamine python=3.10 -y
conda activate robo-dopamine
pip install -r requirements.txt
export PYTHONPATH=$PWD
```

模型可以使用 HuggingFace repo id，也可以使用本地 checkpoint。当前仓库中有本地模型目录：

```text
pretrained_models/Robo-Dopamine-GRM-2.0-8B-Preview
```

如果只运行 `deterministic` monitor 做接口连通性检查，不需要加载 GRM 模型，也不需要 GPU。

## 工作流及示例

### 1. 离线进度评分

可以用 [test_data_suc.py](test_data_suc.py)。这个脚本已经封装了当前常用的离线验收流程：加载模型，依次运行 `forward`、`incremental`、`backward` 三种模式，读取 `pred_vllm.json` 汇总结果，并保存 `progress_curve.png`。

先在脚本顶部修改配置：

```python
MODEL_PATH = "/path/to/checkpoint_or_hf_model"
DATA_DIR = "/path/to/aligned_episode"
OUTPUT_ROOT = "./results/my_eval"

TASK_INSTRUCTION = "pick the carrot and put it on yellow plate "
GOAL_IMAGE = "./examples/blank_goal.png"
INTERVAL = 20
```

`DATA_DIR` 需要包含三路帧数一致的视频：

```text
DATA_DIR/
├── cam_high.mp4
├── cam_left_wrist.mp4
└── cam_right_wrist.mp4
```

运行：

```bash
python test_data_suc.py
```

输出：

```text
OUTPUT_ROOT/
├── progress_curve.png
├── <timestamp>_forward_mode_<task>/
│   ├── sample.json
│   ├── pred_vllm.json
│   ├── reward_vis.mp4
│   └── .cache/
├── <timestamp>_incremental_mode_<task>/
└── <timestamp>_backward_mode_<task>/
```

三种模式含义：

- `forward`：起始帧到当前帧的绝对进度，`progress = score`。
- `incremental`：相邻帧之间的增量变化，并按递推公式累计进度。
- `backward`：目标图到当前帧的反向比较，`progress = clamp(1 + score, 0, 1)`。

没有真实目标图时用 `examples/blank_goal.png`。如果有完成状态图片，建议把 `GOAL_IMAGE` 改成真实目标图，尤其是启用 `backward` 模式时。

### 2. HTTP Monitor Service

入口是 [monitor_runtime/service.py](monitor_runtime/service.py)。运行方式建议以配置文件为准，如monitor.yaml。

#### 2.1 GRM 后端部署

修改 [configs/monitor.yaml](configs/monitor.yaml) 中的关键字段：

```yaml
host: 0.0.0.0
port: 8877
backend: grm

robot_runtime_url: <ip>:8767
observation_timeout: 3.0

model_path: /path/to/Robo-Dopamine-GRM-checkpoint
goal_image: /path/to/blank_or_real_goal.png
fisheye_config: null

no_backward: true
interval: 1.0
local_rank: null
cuda_visible_devices: null

success_threshold: 0.60
success_stable_steps: 3
success_max_drift: 0.05
fail_stable_steps: 8
fail_min_progress: 0.01
```

启动：

```bash
python -m monitor_runtime.service --config configs/monitor.yaml
```

GRM 后端会：

1. `/monitors/start` 后创建 session，并在后台线程捕获 reference start。
2. 按 `interval` 从 Robot Runtime 拉取最新三视角 JPEG。
3. 构造 `forward`、`incremental`，以及可选 `backward` 样本。
4. 调用 GRM 推理，融合各模式 progress。
5. 用 `MonitorState` 根据阈值判断 `running` / `success` / `failed`。
6. `/monitors/status` 只读取最新后台结果，不阻塞等待模型推理。

Robot Runtime 需要提供：

- `GET /observations/latest/metadata`
- metadata 中包含 `binary_endpoints`
- 可解析三路相机：`cam_high`、`cam_left_wrist`、`cam_right_wrist`
- endpoint 返回可被 OpenCV 解码的 JPEG bytes

接口调用示例：

```bash
curl -s http://127.0.0.1:8877/health

curl -s -X POST http://127.0.0.1:8877/monitors/start \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001","execution_id":"exec-1","subtask":"pick carrot","subtask_index":0}'

curl -s -X POST http://127.0.0.1:8877/monitors/status \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001","execution_id":"exec-1"}'

curl -s -X POST http://127.0.0.1:8877/monitors/stop \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001"}'
```

#### 2.2 Deterministic 后端

`deterministic` 主要用于最轻量的接口连通性检查。可在配置文件中把 `backend` 改成：

```yaml
backend: deterministic
auto_success_after_polls: 3
```

然后仍然用同一个启动命令：

```bash
python -m monitor_runtime.service --config configs/monitor.yaml
```

## 数据构造与微调

如需扩展新任务，可使用 `dataset/` 和 `train/` 下的工具链。

原始数据目录格式：

```text
dataset/my_raw_data/
├── task_instruction.json
├── episode_001/
│   ├── annotated_keyframes.json
│   ├── cam_high.mp4
│   ├── cam_left_wrist.mp4
│   └── cam_right_wrist.mp4
└── episode_002/
```

`task_instruction.json` 是任务描述列表；`annotated_keyframes.json` 标注子任务起止帧：

```json
[
  {"annotation": "pick_carrot", "start_frame_id": 0, "end_frame_id": 300},
  {"annotation": "place_carrot", "start_frame_id": 300, "end_frame_id": 605}
]
```

从 `dataset/` 目录运行三步处理：

```bash
cd dataset

python -m utils.0_preprocess_data \
  --raw_dir ./my_raw_data \
  --cvt_dir ./my_train_data \
  --sample_factor 20

python -m utils.1_generate_data \
  --base-dir ./my_train_data \
  --score-bins 25 \
  --gap-bins 4 \
  --oversample-factor 100 \
  --zero-ratio 0.05 \
  --max_sample_num 1000 \
  --workers 4

python -m utils.2_posprocess_data \
  --root-dir ./my_train_data \
  --merged-json ./my_train_data/train_jsons/finetune_data_wo_replace.json \
  --final-json ./my_train_data/train_jsons/finetune_data_final.json \
  --replace-prob 0.75
```

训练前在 [train/qwenvl/data/__init__.py](train/qwenvl/data/__init__.py) 注册数据集：

```python
MY_DATASET = {
    "annotation_path": "./dataset/my_train_data/train_jsons/finetune_data_final.json",
    "data_path": "./dataset",
}

data_dict = {
    # ...
    "my_dataset": MY_DATASET,
}
```

从 `train/` 目录运行训练脚本：

```bash
cd train
bash scripts/finetune_my_carrot.sh
```

训练脚本需要按实际机器和任务修改：

- `MODEL_PATH`
- `OUTPUT_DIR`
- `DATASETS`
- `torchrun --nproc_per_node`
- DeepSpeed 配置：`scripts/zero2.json`、`scripts/zero3.json`、`scripts/zero3_offload.json`

注意：`train/scripts/finetune_grm.sh` 是通用模板，其中 `example_grm_finetune` 当前没有在 `data_dict` 中注册，直接运行会找不到数据集。现有已注册 key 包括 `my_carrot_dataset`、`sub1_approach_grasp`、`suc_1_carrot`、`suc_3_bottle`、`suc_4_cube`。

## 代码结构

下面只列当前 monitor 交付主流程相关的代码和常用目录；本地数据、模型权重和运行结果按需随交付包另行提供。

```text
Robo-Dopamine/
├── README.md                            # 原仓库完整说明
├── README_monitor.md                    # 当前 monitor 交付说明
├── requirements.txt                     # Python 依赖
├── configs/
│   └── monitor.yaml                     # HTTP monitor 服务配置
├── monitor_runtime/
│   ├── core.py                          # MonitorSession / MonitorState 状态模型和成功失败判定
│   ├── service.py                       # FastAPI monitor 服务、CLI 和后端选择
│   └── grm_backend.py                   # 在线 GRM 后端，从 Robot Runtime 拉取三视角图像并后台推理
├── examples/
│   ├── inference.py                     # GRMInference、sample 构造、分数解析和离线推理核心
│   ├── blank_goal.png                   # 无真实目标图时使用的占位目标图
│   ├── demo_table/                      # 原仓库附带的三视角 demo 视频和 goal 图
│   ├── more_demos/                      # 原仓库附带的额外 demo 视频
│   └── visualize_session_video.py       # session JSONL 可视化工具
├── test_data_suc.py                     # 离线三模式进度评分和曲线生成脚本
├── train/
│   ├── qwenvl/
│   │   ├── data/                        # 数据注册、预处理和 RoPE 位置索引
│   │   └── train/                       # Qwen-VL 训练参数、Trainer 和入口
│   ├── scripts/                         # 微调脚本和 DeepSpeed 配置
│   └── tools/                           # 数据校验、packing 工具
├── dataset/                             # 数据构造工具和本地样例数据，通常按需提供
├── pretrained_models/                   # 本地 GRM checkpoint，按需单独提供
├── aligned_data/                        # 离线验收视频数据，按需单独提供
├── results/                             # 离线评分、可视化和训练输出目录
├── assets/                              # 原 README 使用的图片资源
└── LICENSE
```

## 可扩展模块及方式

### 新模型或新 checkpoint

- 离线验收：修改 `test_data_suc.py` 的 `MODEL_PATH`。
- monitor 部署：修改 `configs/monitor.yaml` 的 `model_path`。
- 显存紧张时调小 batch size，或在 `examples/inference.py::GRMInference` 中调整 `min_pixels` / `max_pixels`。

### 新 monitor 后端

新增后端类需要实现与现有后端一致的接口：

```python
start(payload) -> MonitorSession
status(payload) -> MonitorSession
stop(payload) -> dict
health() -> dict
```

然后在 `monitor_runtime/service.py` 的 CLI choices 和 `main()` 构造分支中注册。成功/失败判定建议复用 `monitor_runtime/core.py` 的 `MonitorState`。

### 新图像源或相机协议

当前 GRM monitor 在 `GRMMonitorBackend._snapshot_current()` 中通过 Robot Runtime HTTP 拉取 JPEG。若接入 ROS、共享内存或其他相机服务，可以替换该函数或新增 backend，但最终应生成：

```python
{
    "cam_high": "/path/to/current_high.png",
    "cam_left_wrist": "/path/to/current_left.png",
    "cam_right_wrist": "/path/to/current_right.png",
}
```

单视角任务可把同一路图复制到三个 key，以复用现有 prompt。

### 阈值和融合策略

- Prompt 在 `examples/inference.py` 的 `SYSTEM_PROMPT`。
- 离线 sample 构造在 `examples/inference.py::build_samples_json()`。
- 在线 sample 构造在 `monitor_runtime/grm_backend.py::build_online_samples()`。
- 分数解析在 `parse_score()`。
- 成功/失败判定在 `monitor_runtime/core.py::MonitorState`。
- 阈值优先通过 `configs/monitor.yaml` 调整。

## 注意事项

- 离线视频三路帧数必须一致，否则 `run_pipeline()` 会报 frame count mismatch。
- `test_data_suc.py` 当前是脚本式配置，运行前先检查 `MODEL_PATH`、`DATA_DIR`、`OUTPUT_ROOT`、`TASK_INSTRUCTION`、`GOAL_IMAGE`。
- `configs/monitor.yaml` 中的 `model_path`、`goal_image`、`robot_runtime_url` 是部署相关字段，交付到新机器时必须改成实际值。
- `goal_image` 对 backward 模式影响较大；没有真实目标图时可使用 `examples/blank_goal.png`。
- GRM monitor 启动时只加载一次模型；多个 session 的 vLLM 调用在后端中串行化，以避免并发 generate 不稳定。

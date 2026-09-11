# 在线任务进度监控：实现、接口与用法

本文对应 `Robo-Dopamine-delivery` 的 `dev_x1pro` 分支当前实现（2026-09-08），以 [monitor_steering.yaml](../configs/monitor_steering.yaml) 的 **GRM + HF + SAM3 attention steering** 配置为主。在线服务从 Robot Runtime 获取三路画面，持续计算任务进度，通过 HTTP 提供最新结果及 `running / success / failed` 状态。

Monitor 不下发机器人动作。上游负责执行任务、启动和停止监控，以及决定如何处理评分、异常和超时。这里的“在线”是后台循环采样和推理；没有视频流订阅、推送结果或固定频率的实时控制接口。

阅读顺序：[快速启动](#1-快速启动) → [上游接入示例](#2-上游接入示例) → [Monitor HTTP 接口](#3-monitor-http-接口)。实现细节见第 6 节，异常处理见第 7 节。

## 1. 快速启动

### 1.1 进程与配置

| 组件 | 职责 | 默认地址 | 配置 |
|---|---|---|---|
| Robot Runtime（外部已有服务） | 提供三路相机 JPEG 和 observation metadata | `http://127.0.0.1:8767` | 将其地址填入 monitor 配置 |
| SAM3 服务 | 对当前图片按目标词检测，返回候选 bbox | `http://127.0.0.1:8878` | [sam3.yaml](../configs/sam3.yaml) |
| Monitor 服务 | 管理会话、采图、调用 SAM3/GRM、累计进度和判断终态 | `http://127.0.0.1:8877` | [monitor_steering.yaml](../configs/monitor_steering.yaml)、[steering.yaml](../configs/steering.yaml) |

以下命令均在仓库根目录执行：

```bash
cd /home/dais/workspace/Robo-Dopamine-delivery
```

先确认 Robot Runtime 满足[相机接口约定](#4-robot-runtime-输入接口)，并修改 `monitor_steering.yaml` 中的 `robot_runtime_url`，地址必须含 `http://` 或 `https://`。检查 GRM、SAM3 权重路径和 GPU 是否适合本机。当前配置读取已有的本地 8B GRM 权重及 `/home/dais/workspace/model/sam3`。

终端 A：使用已有 SAM3 环境，启动常驻检测服务。

```bash
conda activate rewardbench-sam3
CUDA_VISIBLE_DEVICES=3 python -m sam3_runtime.service --config configs/sam3.yaml
```

终端 B：使用已有 GRM 环境，启动 Monitor。

```bash
conda activate robo-dopamine
CUDA_VISIBLE_DEVICES=0 python -m monitor_runtime.service --config configs/monitor_steering.yaml
```

GPU 0 和 3 是示例，可调整；每个进程配置的 `device: cuda:0` 指向该进程可见的第一张 GPU。SAM3 使用标准库 `ThreadingHTTPServer`，不要求在 `rewardbench-sam3` 环境安装 FastAPI。Monitor 使用 FastAPI/uvicorn。两个模型分别在各自进程启动时加载一次。

检查服务：

```bash
curl -fsS http://127.0.0.1:8878/health
curl -fsS http://127.0.0.1:8877/health
```

SAM3 返回 `status: ready`；Monitor 的 `data` 应有 `provider: grm`、`engine: hf`、`steering_enabled: true`。**Monitor health 不主动探测 Robot Runtime 或 SAM3 连通性**，还需通过实际会话检查采图和推理。模型加载完成后 Monitor 才开始监听。

### 1.2 常用配置

以下默认值指交付的在线 YAML；不传配置时，CLI/类的部分默认值不同。

| 文件 / 字段 | 当前值 | 用途 |
|---|---|---|
| monitor / `backend`、`inference_engine` | `grm`、`hf` | 在线评分与可干预的推理引擎 |
| monitor / `steering_config` | `./steering.yaml` | 加载干预配置 |
| monitor / `goal_image` | `../examples/blank_goal.png` | 全部会话共用的参考终点；有真实完成图时可替换 |
| monitor / `no_backward` | `true` | 默认仅运行 forward、incremental |
| monitor / `no_incremental` | `false` | 与 `no_backward: true` 同时设为 `true` 时只运行 forward；CLI 可用 `--no-incremental --no-backward`，`--incremental` 可覆盖 YAML 恢复 incremental |
| monitor / `interval` | `0.1` 秒 | 单、双分支 YAML 均减少轮间空等；每轮工作结束后等待，代码最低取 0.1 秒，不是总推理周期 |
| monitor / `observation_timeout` | `3.0` 秒 | 单次 Robot Runtime HTTP 请求超时 |
| monitor / `max_camera_skew_s` | `0.25` 秒（代码默认，YAML 未写） | 三路响应都有时间戳时允许的最大时间差 |
| monitor / `max_new_tokens` | `64` | GRM 生成长度上限 |
| monitor / `hf_batch_size` | `2` | 每个 HF 分支一次 generate 的样本上限；设为 1 可串行对比 |
| monitor / `output_root` | `../results/monitor_sessions` | 会话图片和日志根目录 |
| monitor / `fisheye_config` | 未设置 | 可选腕部相机去畸变配置 |
| steering / `enabled` | `true` | 启用目标区域 attention bias |
| steering / `profile_path`、`top_k` | GRM 8B profile、`8` | 选择冻结排名中的 8 个 heads |
| steering / `bias` | `6.0` | 目标视觉 keys 加 `+6`，指定负区域加 `-6` |
| steering / `intervention_labels` | `[after_cam_high]` | 默认只干预当前主视角；模型仍输入三路相机 |
| steering / `query_scope` | `all` | prefill 的全部 query 行与 cached decode |
| steering / `negative_scope` | `target_span` | 同一目标图像 span 内、bbox 外的视觉 keys 为负区域 |
| steering / `on_missing_bbox` | `baseline` | 检测缺失/歧义/服务请求失败时，该 sample 使用无 bias 的 HF 推理 |
| steering / `grounding.url`、`timeout_s` | `http://127.0.0.1:8878`、`30.0` 秒 | SAM3 地址与单次请求超时 |
| SAM3 / `threshold`、`mask_threshold` | `0.3`、`0.5` | 候选置信度阈值；mask_threshold 仅保留配置兼容，bbox 后处理不使用 |

成功/失败窗口参数及其精确定义见第 6.4 节。Monitor CLI 参数覆盖 YAML，例如临时开启 backward：

```bash
CUDA_VISIBLE_DEVICES=0 python -m monitor_runtime.service \
  --config configs/monitor_steering.yaml --backward --interval 2.0
```

Monitor YAML 中的 `steering_config / goal_image / fisheye_config / output_root` 相对该 YAML 解析；steering 中的 profile、SAM3 配置中的本地模型路径也相对各自 YAML 解析。Monitor 的 `model_path` 原样传给模型加载器，建议使用绝对本地路径或有效的模型标识。CLI 显式给出的本地相对路径按当前工作目录解释。

配置为进程级：在线 start 不接受模型、goal 图、head 列表、bias、对照条件或推理模式的动态覆盖。修改这些配置需要重启相应服务。当前 profile 针对 GRM-2.0-8B-Preview；4B/微调权重需要匹配的 ranking/profile，不能只替换模型路径。

### 1.3 双分支监控

双分支同时运行原始 GRM 和 attention steering GRM：从同一 checkpoint 各加载一个独立 HF 模型实例，每轮并发处理相同的冻结图像、任务、reference 和 previous。原始分支使用无干预的 HF greedy 推理；steering 分支按原配置定位目标并施加 attention bias。两分支的 incremental 进度分别累计。

默认 `hf_batch_size: 2`，每个分支把 forward/incremental 合成一次 batch=2 的
generate，整轮为两个独立模型各生成一次。每行有自己的图像、补齐位置、bbox
和 attention mask；两分支全部完成并验证有效后才发布完整结果。开启 backward
时默认再用一个 batch=1 处理尾项。`--hf-batch-size 1` 同时让两个分支恢复串行。

合批按 processor 输出的有效 token 长度分组，异长样本自动拆批，避免当前 BF16
Qwen3-VL 在 padded batch 下出现较大得分变化。正常在线两模式输入等长，仍为
batch=2；实际行数以结果中的 `steering.batch_size` 为准。

已有单分支配置默认行为不变。在原启动命令上增加参数即可启用：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m monitor_runtime.service \
  --config configs/monitor_steering.yaml \
  --dual-branch --baseline-device cuda:1 \
  --progress-difference-threshold 0.20 --difference-mode absolute
```

也可使用独立示例 [monitor_dual_branch.yaml](../configs/monitor_dual_branch.yaml)，先修改其中的 Robot Runtime 地址，再运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m monitor_runtime.service \
  --config configs/monitor_dual_branch.yaml
```

SAM3 仍按第 1.1 节单独启动。示例中 steering 使用可见 GPU 0、baseline 使用可见 GPU 1；若显存足够，可将 `--baseline-device cuda:0` 设为与 steering 相同的设备，仍加载两份独立模型。未配置 `baseline_device` 时默认跟随 `device`。每个模型仅在服务启动时加载一次，基线使用与 steering 相同的 checkpoint、预处理和生成长度；不使用 vLLM 采样作为对照，以免混入推理引擎和采样的差异。

| monitor 字段 / CLI | 默认值 | 说明 |
|---|---|---|
| `dual_branch` / `--dual-branch` | `false` | 启用双分支；要求 `backend=grm`、`inference_engine=hf` 且 steering 配置启用；`--no-dual-branch` 可覆盖 YAML |
| `baseline_device` / `--baseline-device` | 跟随 `device` | 原始 GRM 所在设备；双分支示例 YAML 为 `cuda:1` |
| `progress_difference_threshold` / `--progress-difference-threshold` | `0.20` | 两分支融合进度差的失败阈值，有限数且在 `[0,1]`；0.20 为 20 个百分点，初始示例值需按任务调整 |
| `difference_mode` / `--difference-mode` | `absolute` | `absolute`：绝对差；`baseline_minus_steering`：原始减 steering；`steering_minus_baseline`：steering 减原始 |

start/status/stop 的请求格式不变，双分支配置同样为进程级。`GET /health` 的 `data.dual_branch` 返回是否启用、比较指标、方向、阈值及两分支设备。具体判定与结果字段见第 6.4 节和第 3.2 节。

### 1.4 Robot Runtime 的 manual 操作台

`dualsystem-agentic` 的 `/manual` 页面现在可展示三视角、SAM3 bbox 和本服务得分。
每次成功提交评分时，`status.data.result.preview` 返回：

```json
{"frame_set_id":"<32位十六进制ID>","cameras":["cam_high","cam_left_wrist","cam_right_wrist"],"kind":"grm_after"}
```

`GET /monitors/frames/{frame_set_id}/{camera}.png` 返回该轮实际使用的当前视角 PNG，
与 `result.frames` 指向文件的字节一致，含已配置的去畸变处理；不接受文件路径参数。
图像和 `result.modes[mode].steering.grounding.after_<camera>` 的 bbox 属于同一轮
推理。bbox 只由网页绘制，不写入模型输入 PNG。

仅已提交的有效推理结果会发布图像 ID；失败或取消的推理不发布。服务保留最近
128 组评分图像索引，stop 后仍可访问最终画面，索引淘汰或进程重启后返回 404。
原始会话文件的保存方式不变。无 GRM 图像能力的 deterministic 后端返回 404。

从臂 Runtime 通过已配置的 `monitor.url` 转发这些图像，浏览器无需直连服务器或
额外开放端口。更新这项功能后重启 Monitor 即可，SAM3 接口与推理设置不变。
网页目标输入和人工开始/停止/归位仍由 Runtime 与上游 loop 管理。

## 2. 上游接入示例

### 2.1 curl 最小调用

```bash
curl -fsS -X POST http://127.0.0.1:8877/monitors/start \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001","execution_id":"exec-1","subtask":"pick the carrot and put it on yellow plate","subtask_index":0,"target_queries":["carrot"]}'

curl -fsS -X POST http://127.0.0.1:8877/monitors/status \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001","execution_id":"exec-1"}'

curl -fsS -X POST http://127.0.0.1:8877/monitors/stop \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001"}'
```

这是三个独立请求的语法示例。实际运行应在 start 后持续轮询，直到终态或应用自己的退出条件成立，再 stop。

建议在机器人动作开始前 start，等 `data.result.warming_up == false` 后由上游启动动作，使 reference 对应任务初始画面。**该标志只表示参考帧已采集**；首个评分还需 `poll_count > 0` 且 `result.inference_step` 存在。若动作已开始才 start，forward 的起点就是后来采到的画面。

### 2.2 Python 轮询客户端

以下脚本仅依赖 Python 标准库，可直接保存运行。它等待参考帧，按新的推理轮数打印进度，检查错误/结果年龄，并在终态或客户端超时后释放 Monitor 会话。将注释处接入已有机器人执行接口；脚本自身不发动作命令。示例的等待上限是客户端策略，需根据实际模型耗时调整。

```python
import json
import time
import urllib.request
from uuid import uuid4

BASE = "http://127.0.0.1:8877"
STARTUP_TIMEOUT_S = 60.0
STALE_TIMEOUT_S = 120.0
TASK_TIMEOUT_S = 600.0


def post(route, payload):
    request = urllib.request.Request(
        BASE + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15.0) as response:
        envelope = json.load(response)
    if not envelope["success"]:
        raise RuntimeError(envelope["message"])
    return envelope["data"]


identity = {"monitor_id": "m-" + uuid4().hex, "execution_id": "exec-1"}
request = {
    **identity,
    "subtask": "pick the carrot and put it on yellow plate",
    "subtask_index": 0,
    "target_queries": ["carrot"],
}

# start 在 try 外：若请求失败，保留原始异常，不盲目 stop 可能已有的会话。
# 若 start 响应因网络中断而丢失，可用完全相同的 request 幂等重试。
post("/monitors/start", request)
try:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        session = post("/monitors/status", identity)
        result = session["result"]
        if not result["warming_up"]:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"参考帧未就绪: {session['error']}")
        time.sleep(0.5)

    print("参考帧已就绪，可由上游启动机器人动作", flush=True)
    # 在此调用已有的异步任务执行接口；不要阻塞下面的状态轮询。
    deadline = time.monotonic() + TASK_TIMEOUT_S
    last_step = 0
    while True:
        session = post("/monitors/status", identity)
        result = session["result"]
        if session["error"]:
            print("最近一轮错误:", session["error"])
        if session["poll_count"] > last_step:
            last_step = session["poll_count"]
            diagnostics = {
                mode: {
                    "applied": row["steering"].get("applied"),
                    "degraded": row["steering"].get("degraded"),
                    "reason": row["steering"].get("reason"),
                }
                for mode, row in result["modes"].items()
            }
            print(last_step, session["progress"], session["status"], diagnostics)
        if session["status"] in {"success", "failed"}:
            print("Monitor 终态:", session["status"])
            break
        if result["result_age_s"] > STALE_TIMEOUT_S:
            raise TimeoutError("长时间无新评分；检查相机、重复帧和模型错误")
        if time.monotonic() >= deadline:
            raise TimeoutError("达到客户端任务时限")
        time.sleep(0.5)
finally:
    # 仅停止监控；机器人停止/取消任务仍由上游处理。
    try:
        print(post("/monitors/stop", {"monitor_id": identity["monitor_id"]}))
    except Exception as exc:
        print("Monitor 清理请求失败:", exc)
```

## 3. Monitor HTTP 接口

本节描述 `backend=grm` 的行为。成功响应 HTTP 200，统一封装：

```json
{"success": true, "data": {}, "message": "ok"}
```

已处理的业务错误封装：

```json
{"success": false, "data": null, "message": "monitor_id already belongs to a different request"}
```

**外层 `success` 表示接口调用成功；机器人任务判断在 `data.status`。** FastAPI 的请求校验错误可能返回原生 422，未捕获异常可能返回 500，不保证都使用上述错误封装。请求体当前按 `dict` 接收，不是严格的逐字段 Pydantic 模型；接入方应按下面的推荐类型传值。

### 3.1 `POST /monitors/start`

| 字段 | 推荐类型 | 必填 | 说明 |
|---|---|---|---|
| `monitor_id` | 非空 string | 是 | 上游分配的会话标识 |
| `execution_id` | 非空 string | 是 | 任务执行标识 |
| `subtask` | 非空 string | 是 | 完整任务指令，作为 GRM prompt 的任务文本 |
| `subtask_index` | integer 或 null | 否 | 子任务序号，建议从 0 开始 |
| `target_queries` | 非空 string 数组 | 建议显式提供 | 同一个操作目标的查询短语，例如 `["carrot"]`；SAM3 每次最多接受 8 个 |
| `defer_inference` | boolean | 否，默认 false | true 时只采集 reference，等待 `/monitors/activate` 后开始评分；供 Robot Runtime 的启动握手使用 |

目标词优先级：显式 `target_queries` → `steering.yaml.task_queries` 中空白归一化后的精确任务映射 → 简单英语动作指令解析。中文或复杂任务请显式提供目标词。`["carrot", "yellow plate"]` 会被当成互相替代的候选目标，不表示同时约束操作对象与目的地；完整任务关系仍写在 `subtask` 中。

start 创建会话目录和后台线程后立即返回当前 session，不等待采图或模型推理。相同 `monitor_id`、`execution_id`、原始 `subtask` 文本、解析后的目标词列表和 `subtask_index` 再次 start，返回已有会话，包括已有终态。相同 ID 的这些参数不同则返回 409。缺失必填字段或无法解析目标返回 400。

### 3.2 `POST /monitors/status`

请求：`{"monitor_id":"m-001","execution_id":"exec-1"}`。`execution_id` 可省略；提供时会检查归属，不匹配返回 409。未知或缺失 `monitor_id` 返回 404。

status 只读取最近已提交结果，不触发 GRM 推理，也不增加 `poll_count`。返回的 session 字段：

| 字段（位于 `data`） | 含义 |
|---|---|
| `monitor_id / execution_id / subtask / subtask_index` | 会话身份与任务 |
| `status` | `running`、`success` 或 `failed` |
| `progress` | 融合进度，范围 `[0,1]`；首次评分前为 0 |
| `created_at` | 会话创建的 Unix 秒时间戳，保持不变 |
| `updated_at` | 最近成功提交评分的 Unix 秒时间戳；首次提交前等于创建时间 |
| `poll_count` | **成功提交的推理轮数**，与客户端查询次数无关 |
| `error` | 最近一轮异常文本或 null；之后成功提交会清除 |
| `message` | 后端说明，当前为 `grm monitor backend` |
| `result` | 参考帧就绪状态、最近完整推理和诊断信息 |

首次评分前，`result` 包含 `provider / warming_up / inference_enabled / result_age_s / session_dir`，发生异常时另有 `error`。`inference_enabled=false` 表示评分尚未激活。首次评分后的响应片段如下，**数值仅为格式示例，省略了其他真实字段**：

```json
{
  "success": true,
  "data": {
    "monitor_id": "m-001",
    "execution_id": "exec-1",
    "status": "running",
    "progress": 0.25,
    "poll_count": 1,
    "error": null,
    "result": {
      "provider": "grm",
      "warming_up": false,
      "step": 0,
      "inference_step": 1,
      "engine": "hf",
      "result_age_s": 0.2,
      "modes": {
        "forward": {
          "pred": "<score>+25%</score>",
          "score": 0.25,
          "progress": 0.25,
          "hop": 0.25,
          "steering": {"condition": "candidate_target", "enabled": true, "applied": true, "degraded": false}
        },
        "incremental": {
          "pred": "<score>+25%</score>",
          "score": 0.25,
          "progress": 0.25,
          "hop": 0.25,
          "steering": {"condition": "candidate_target", "enabled": true, "applied": true, "degraded": false}
        }
      }
    }
  },
  "message": "ok"
}
```

完整 `result` 的主要诊断路径：

| 路径（相对于 `data.result`） | 含义 |
|---|---|
| `step / inference_step` | 从 0 / 从 1 开始的已提交轮次；后者等于 `poll_count` |
| `progress / fused / progress_percent / status` | 本轮融合结果；百分数为进度乘 100 |
| `inference_updated_at / result_age_s` | 提交时间 / 距最后提交的秒数；首次提交前 age 从创建时间算 |
| `modes.<mode>.score / progress / hop / pred` | 原始分数、模式累计值、变化量及模型输出文本 |
| `branches.steering / branches.baseline` | 仅双分支存在；各含 `progress` 和 `modes`，保存各自融合进度、各模式分数和模型输出；顶层 `progress / modes` 继续对应 steering 分支 |
| `comparison.metric / difference_mode / difference / threshold / threshold_exceeded` | 仅双分支存在；比较指标固定为 `fused_progress`，记录差值方向、当前差值、阈值及是否严格超过 |
| `comparison.modes.<mode>.score_difference / progress_difference` | 仅双分支存在；按相同方向计算的各模式原始分数差和累计进度差，供诊断；终态规则使用融合进度差 |
| `failure_reason` | 差值规则触发时为 `branch_difference_exceeded`；原有进度规则触发的终态不添加此字段 |
| `modes.<mode>.steering.applied / degraded / reason` | 是否实际施加干预、是否降级、降级原因（有时才存在） |
| `modes.<mode>.steering.grounding.<label>` | 对应图像的 SAM3 结果和客户端选择结果；网络请求失败时可能没有该条目 |
| `modes.<mode>.steering.grounding.<label>.selected.bbox` | 选中目标的像素 `xyxy` 框；歧义/无检测时 `selected` 为 null |
| `modes.<mode>.steering.grounding.<label>.reused_in_batch` | 是否复用了本轮相同冻结图片、相同目标词的定位结果；复用 bbox 后仍独立映射本模式的 token 位置 |
| `modes.<mode>.steering.image_cache_hits / prepare_ms` | 本 sample 复用已解码图片的次数 / 图片、prompt、processor 和输入准备耗时（毫秒） |
| `modes.<mode>.steering.batch_id / batch_size / batch_index / padding_left` | 实际 generate 批次、行数、行号和左侧 padding 数；spans/目标位置已包含偏移 |
| `modes.<mode>.steering.batch_prepare_ms / batch_grm_ms / batch_total_ms` | 整批准备、生成、总耗时；同批每行重复携带，需按 batch_id 去重后求和 |
| `modes.<mode>.steering.per_layer` | 各层 hook 的调用和生效次数，例如 `prefill_calls / decode_calls / applied_calls` |
| `modes.<mode>.steering.spans / target_positions / negative_positions / heads` | 图像 token spans、目标/负区域绝对 token 位置、选中 heads；部分字段仅定位成功时存在 |
| `observation.cameras.<camera>` | 相机 `frame_id / timestamp / image_sha256` |
| `observation.identity / synchronization_verified` | 用于去重的组合标识 / 三路时间戳检查是否可验证 |
| `observation.preprocess_fingerprint / fisheye_enabled` | 预处理配置指纹与去畸变开关 |
| `frames / session_dir` | 三路冻结 PNG 的服务器绝对路径 / 会话产物目录；不是图片下载 URL |
| `timing` | `observation_ms / queue_wait_ms / prepare_ms / grounding_ms / grm_ms / total_ms` |
| `latency_s` | 本轮耗时，秒；不含随后 interval 等待和客户端轮询延迟 |
| `final_status / progress_history` | 终态时附加的状态与全部已提交融合进度 |

`steering` 和 `grounding` 位于各个 `modes` 内，**没有顶层 `result.steering` 或 `result.grounding`**。`timing.prepare_ms / grounding_ms / grm_ms` 汇总各模式实际耗时；双分支下汇总两个分支。`total_ms` 是本轮墙钟耗时，还包括采图、排队等；双分支并发时耗时之和可以大于 `total_ms`。缓存中的 `grounding.<label>.latency_ms` 保留首次 SAM3 检测的耗时，不能把两个模式里的该字段相加当作本轮检测耗时。服务每条 `[GRM]` 日志也会输出本轮 `latency=…s`，不含轮间等待。

批量生成后，每个模式的 `prepare_ms / grm_ms` 是整批耗时按行数均分的记账值，
不代表该模式单独推理的延迟；顶层求和不会重复计时。`grounding_ms` 仍是该模式
实际定位准备耗时，`steering.total_ms` 为上述三项之和。

异长样本拆批时，`regroup_prepare_ms` 记录额外分组准备开销，已经计入模式的
`prepare_ms / total_ms`；`batch_*` 字段仅记录分组后对应 generate 的工作。

### 3.3 `POST /monitors/stop`

请求：`{"monitor_id":"m-001"}`。响应示例：

```json
{"success": true, "data": {"stopped": true, "monitor_id": "m-001", "worker_stopping": false}, "message": "ok"}
```

立即移除会话并通知工作线程停止，最多等待线程退出 5 秒；GPU 中已开始的生成不会被强制打断。`worker_stopping: true` 表示线程仍在结束过程中，其结果不会再发布。随后 status 返回 404，stop 不校验 `execution_id`，对未知 ID 也幂等返回成功。

stop 保留日志和图片，不停止机器人。用相同外部 ID 重新 start 会创建新的 UUID 会话目录和参考帧。服务重启不会恢复内存会话；已有磁盘日志仍保留。

### 3.4 `GET /health`

`data` 包含 `status / provider / model / runtime_url / engine / steering_enabled / profile_fingerprint / dual_branch / sessions / interval / active_modes / cameras`。`sessions` 包括尚未 stop 的终态会话，不等于正在执行模型推理的数量。

### 3.5 `POST /monitors/activate`

请求：`{"monitor_id":"m-001","execution_id":"exec-1"}`，返回统一封装的当前 session。参考帧未就绪或 execution ID 不匹配返回 409，会话不存在（包括已 stop）返回 404；重复激活幂等，不重置进度。

配合 `start` 的 `defer_inference: true`，完整时序为：启动 Monitor → 等 `warming_up=false` → 上游启动机器人动作 → activate 开始评分。Robot Runtime 已封装该流程。这样快模型也不会在动作启动前判定终态。未设置 defer 的旧调用仍自动开始评分。

## 4. Robot Runtime 输入接口

Monitor 先调用 `GET /observations/latest/metadata`，接受原始 JSON 对象或 `{"data": {...}}` 包装，例如：

```json
{
  "frame_id": 123,
  "timestamp": 1788830000.125,
  "binary_endpoints": {
    "cam_high": "/observations/latest/cam_high.jpg",
    "cam_left_wrist": "/observations/latest/cam_left_wrist.jpg",
    "cam_right_wrist": "/observations/latest/cam_right_wrist.jpg"
  }
}
```

`binary_endpoints` 必须为对象；某个相机键缺失时回退到 `/observations/latest/<camera>.jpg`。路径必须以 `/` 开头，不允许绝对 URL 或 `..` 路径段。三路相机都必需，不支持运行时任意增减相机。

图像端点返回 OpenCV 可解码的 JPEG bytes，建议携带：

```http
Content-Type: image/jpeg
X-Frame-Id: 123
X-Timestamp: 1788830000.125
```

`X-Timestamp` 为同一时间基准下的有限秒数。三路都有时间戳时，最大差值超过 `max_camera_skew_s` 会拒绝本轮；缺少任何一路时间戳时仍可运行，但 `synchronization_verified=false`。metadata 和三张 JPEG 是顺序 GET，latest 路径本身不保证原子同步。

去重按各相机的 `X-Frame-Id`，缺失时回退到保存 PNG 的 SHA-256，然后组合成 observation identity。metadata 中全局 `frame_id` 用于记录，不能代替各相机 header。真实新帧应更新相机帧号；无帧号且图片字节不变时，会按重复画面跳过评分和稳定窗口计数。

可选鱼眼去畸变只处理腕部相机。之后保存冻结 PNG，GRM 和 SAM3 都读取该画面。`observation.cameras.*.image_sha256` 是磁盘 PNG 文件字节哈希；grounding 的 `image_sha256` 是实际发送的 RGB PNG 请求载荷的哈希。当前在线 RGB PNG 直接发送原始文件字节，两者相等；非 RGB PNG、JPEG 等输入仍先转 RGB PNG，旧版日志也使用重新编码的载荷，这些情况下两种哈希可能不同。

## 5. SAM3 检测接口

SAM3 是独立检测服务；正常上游只需调用 Monitor。直接调用下面接口可排查模型或目标词问题。**SAM3 响应不使用 Monitor 的 `success/data/message` 封装。**

`GET /health` 返回：

```json
{"status": "ready", "model_fingerprint": "模型配置指纹"}
```

`POST /grounding/detect` 接收 JSON 中的 base64 PNG，当前实现不是 multipart 上传：

```json
{
  "request_id": "debug-001",
  "image_sha256": "解码后的 PNG 文件字节的 SHA-256",
  "queries": ["carrot"],
  "image_png_base64": "PNG 字节的 base64 字符串"
}
```

`queries` 为 1–8 个非空字符串；请求正文（包括 base64）上限 20 MiB，读取请求超时为 15 秒。成功响应格式示例：

```json
{
  "request_id": "debug-001",
  "image_sha256": "与请求匹配的哈希",
  "image_size": [640, 480],
  "coordinate_space": "input_image_xyxy",
  "model_fingerprint": "模型配置指纹",
  "status": "ok",
  "candidates": [{"bbox": [100.0, 120.0, 180.0, 260.0], "score": 0.92, "query": "carrot"}],
  "latency_ms": 85.0
}
```

bbox 是输入图片像素坐标 `[x1,y1,x2,y2]`，不是归一化坐标。没有检测时 HTTP 200、`status: no_detection`、`candidates: []`。服务每次只允许一个检测请求，忙时直接 503，不排队；参数错误 400、内部错误 500、未知路由 404，错误正文为 `{"error":"..."}`。

可以从仓库根目录使用已封装客户端完成编码、哈希校验和候选选择；将图片路径替换为实际在线保存的 `frames.cam_high`：

```python
from grm_runtime.grounding import GroundingClient

client = GroundingClient("http://127.0.0.1:8878", timeout_s=30.0)
result = client.detect("/path/to/session/capture_000001/cam_high.png", ["carrot"])
print(result["selection_status"], result["selected"])
```

服务会对查询词产生的重叠候选按 IoU ≥ 0.8 去重。GRM 客户端再按置信度排序：第二候选得分 ≥ 第一候选得分减 0.05 时判为 `ambiguous`，不选择实例；否则选最高分。`selected / selection_status` 是客户端新增字段，原始 SAM3 HTTP 响应没有它们。

`GroundingClient` 的跨调用缓存键包含图片内容、目标词和 SAM3 模型指纹；每次进入该客户端都会请求 health 检查指纹。在一次 HF `inference_batch` 内，forward/incremental 对相同冻结图片、相同目标词直接共享检测结果，后续模式不重复编码、请求 health 或检测。该批次缓存随调用结束丢弃，下一轮仍检查模型指纹；文件被改写或目标词变化时不复用。歧义/无检测结果可共享，请求异常不缓存，后续模式可重试。

同批次还复用已解码的 RGB 图片，最多保留 16 张；定位缓存最多 128 项。processor
一次接收最多 `hf_batch_size` 条八图样本，按每行左侧补齐后独立构造 token spans
和 masks，并在一次 generate 内并行评分。共享的是像素和 bbox，不是另一模式的
token 位置或得分。双分支各有自己的缓存和模型。配置多个干预视角时每张图独立
定位，不沿用上一帧框。性能测量方法见 [monitor_performance.md](monitor_performance.md)。

SAM3 当前使用 `post_process_object_detection`，只筛选/缩放 bbox 和分数，省去
mask 的 sigmoid、插值和二值化后处理。模型前向仍按原 SAM3 执行，未修改精度或
跳过内部 mask decoder；框和分数计算与原分割后处理一致。模型指纹包含后处理
版本，更新并重启 SAM3 后，旧检测缓存因指纹变化而失效。

## 6. 实现原理与代码入口

### 6.1 一轮在线推理

1. `start()` 创建 `_SubtaskState`、UUID 目录、manifest 和后台线程；线程采集一次三路 reference，固定为当前会话起点。
2. 每轮读取当前三路图像，保存到独立 `capture_<index>/`；重复 observation 删除本次快照并跳过。采图序号与成功推理轮数独立，因此目录编号可有间隔。
3. `build_online_samples()` 为每个启用模式构造 8 图输入。所有模式都使用同一轮冻结 AFTER 图像。
4. 获得共享推理锁，依次运行各模式的 SAM3 定位、bbox 到视觉 tokens 映射、HF attention bias 和生成。双分支时，在此锁内并发运行两个独立模型；各分支内部仍依次处理模式，原始分支不请求 SAM3。
5. 校验所有输出的 ID、对应模式和分数，使用 tracker/state 副本计算结果。双分支须全部通过校验，分别累计进度后比较差值。全部有效后，先追加 `online_pred.jsonl`，再在会话锁内一次性提交两分支进度、previous、轮次和最近结果。
6. 等待 `interval` 秒，再采下一轮。到达终态后结束循环，终态结果仍能查询，直到显式 stop。

任何模式失败都不会部分推进本轮进度；下次成功的 incremental BEFORE 仍使用最后一次成功提交的画面，因此跨过失败采样，而不累加一个不存在的中间分数。

每个会话有后台线程，但同一 Monitor 的不同会话推理轮次被共享锁串行化，一轮的多个模式和两个分支一起持锁。某一分支抛错时仍等待另一分支结束，再释放推理锁和清理未提交图片。HF 内部也锁住完整 sample，包括 grounding、hook 安装、生成和清理。status 不持有模型推理锁。多会话可共享常驻模型，但增加会话会增加排队时间；真实更新周期约为“采图 + 排队 + 定位及两分支推理 + 其他开销 + interval”。

### 6.2 8 图输入与模式

| 位置（从 0 开始） | 标签 | 来源 |
|---|---|---|
| 0 | `reference_start` | 会话起始主视角 |
| 1 | `reference_end` | 配置中的 goal 图 |
| 2、3、4 | `before_cam_high / before_cam_left_wrist / before_cam_right_wrist` | 各模式定义的 BEFORE 三视角 |
| 5、6、7 | `after_cam_high / after_cam_left_wrist / after_cam_right_wrist` | 当前冻结三视角 |

forward 的 BEFORE 为固定 reference；incremental 为上次已提交画面，第一轮为 reference；backward 为同一 goal 图重复三次。默认只开启前两个模式。只有空白 goal 时，backward 的参考信息有限；可提供真实完成图再启用。

### 6.3 attention steering 如何生效

在线 sample 不设置实验 condition；HF 在 `enabled: true` 时选择 `candidate_target`。离线的 `candidate_wrong / low_rank_target` 对照条件没有暴露为在线 start 参数。

Processor 对实际 8 图输入计算视觉 token spans；SAM3 bbox 按实际 image grid 与 merge size 映射到与框相交的视觉网格单元，再转成该图 span 的绝对 token 位置。默认在第 6 张图 `after_cam_high` 的目标区域加正 bias，同图其余视觉 keys 加负 bias。

当前冻结 top-8（层、query head，均从 0 开始）：`(19,16), (19,23), (19,10), (20,4), (19,0), (18,30), (20,13), (22,15)`。这些来自当前 GRM 8B 的排名；profile 校验模型路径、config 哈希及层/head 数等信息。

实现通过目标层 self-attention 的 forward pre-hook 修改 attention mask，在 softmax 前施加 bias；模型使用 HF eager attention、greedy generation、`use_cache=True`、`output_attentions=False`。默认 `query_scope: all` 覆盖 prefill 与后续 decode；新增文本 key 的 bias 为零，原有 causal/padding 禁止位保留。batch 每行使用自己的 heads、目标/负区域和 KV cache，baseline 或降级行保留原 mask。按输入行号解码并去掉完整补齐后的 prompt，EOS/pad 由 tokenizer 去除。hooks 通过 `finally` 清理。

其他可配置项：`query_scope` 支持 `prefill / last_prompt / decode`；`negative_scope` 支持 `all_visual / other_spans / none`；`intervention_labels` 可指定 BEFORE/AFTER 的真实相机标签，不能指定两张 reference 图。多图配置下任一目标缺失，默认对整个 sample 降级 baseline。

### 6.4 从模型分数到任务状态

HF 严格解析完整 `<score>+25%</score>` 形式，得到 `s=0.25`，合法范围 `[-1,1]`；格式错误或越界会使本轮失败，不替换成 0 分。

| 模式 | 当前模式进度 `p` | `hop` |
|---|---|---|
| forward | `p=s` | `p-p_prev` |
| incremental，第一轮 | `p=s` | `s` |
| incremental，后续且 `s>=0` | `p=p_prev+(1-p_prev)*s` | `s` |
| incremental，后续且 `s<0` | `p=p_prev+p_prev*s` | `s` |
| backward | `p=clamp(1+s,0,1)` | `p-p_prev` |

启用模式的进度取算术平均，最后裁剪至 `[0,1]`，作为融合进度。单个模式的 progress 不都保证在 `[0,1]`，例如 forward 可为负。

当前在线配置的终态规则：

- **success**：当前融合进度 ≥ `success_threshold=0.60`，已有最近 `success_stable_steps=3` 个有效结果，且这 3 个值的最大值减最小值 ≤ `success_max_drift=0.05`。并非要求三个值全部 ≥ 0.60，例如 `[0.58,0.59,0.60]` 满足成功条件。
- **failed**：当前融合进度 < 0.60，已有最近 `fail_stable_steps=8` 个有效结果，且窗口中没有一次相邻增量 ≥ `fail_min_progress=0.01`。停滞和持续回退都可能触发，窗口内也不要求每个值均低于阈值。
- 其余情况为 **running**。终态不自动反转；重复帧、错误轮次和 status 请求均不增加窗口计数。

启用双分支后，上述三条规则继续使用 **steering 分支的融合进度**。新增规则优先于 success 和原有 failed 规则：

1. 两个分支各自使用本节原公式计算模式进度并融合为 `p_baseline`、`p_steering`（均裁剪到 `[0,1]`）。
2. 默认 `difference_mode=absolute`，取 `d = abs(p_baseline - p_steering)`；两个有向选项分别取 `p_baseline - p_steering`、`p_steering - p_baseline`。
3. 本轮 `d > progress_difference_threshold` 时，立即判 **failed**，不等待稳定窗口。等于阈值时不触发新规则（以 `1e-12` 绝对容差消除浮点舍入误差）；同轮即使满足旧 success 条件，差值失败仍优先。终态仍不可反转。

例如 steering 为 0.75、baseline 为 0.40，默认绝对差为 0.35，超过 0.20 即 failed。steering 为 0.75、baseline 为 0.70 时差为 0.05，继续由原有窗口规则判断。这里比较的是两分支**各自累计、融合后的进度**；各模式本轮 `<score>` 的差另存为 `comparison.modes.*.score_difference`，不直接触发失败。融合比较会受到各模式平均及裁剪的影响，不等价于任一模式原始分数差超阈值。

缺框时 steering 分支仍按 `on_missing_bbox` 配置降级或报错。降级后的实际输出照常参与进度和差值计算，并保留 `degraded / reason`；由于两个分支此前的 incremental 累计进度可能不同，本轮原始分数一致也不保证融合差为零。任一分支生成/解析错误或日志写入失败时，两分支均不推进 tracker、previous、轮次或判定窗口，也不把缺失分数当成零。

这些是评分轨迹规则，`failed` 不等于相机/SAM3 服务故障；`success` 也不是经校准的成功概率。阈值沿用原配置，尚未针对 steering 和新部署任务重新校准。持续采图/推理错误不会自动触发 failed，服务没有内置的最大监控时长，需由上游设置超时。

### 6.5 代码定位与输出

| 文件 | 主要职责 / 入口 |
|---|---|
| [monitor_runtime/service.py](../monitor_runtime/service.py) | `create_app()`、HTTP 路由、CLI/YAML、启动后端 |
| [monitor_runtime/grm_backend.py](../monitor_runtime/grm_backend.py) | `GRMMonitorBackend`、`_snapshot_current()`、`_run_one_step()`、`build_online_samples()`、会话生命周期 |
| [monitor_runtime/core.py](../monitor_runtime/core.py) | `MonitorSession.to_dict()` 与 `MonitorState.update()` |
| [examples/inference.py](../examples/inference.py) | `GRMInference` facade，选择 HF/vLLM |
| [grm_runtime/hf_backend.py](../grm_runtime/hf_backend.py) | `HFBackend`、视觉 spans、定位、完整生成与 hook 清理 |
| [grm_runtime/masking.py](../grm_runtime/masking.py) | bbox 到 tokens、正负区域与 attention mask hook |
| [grm_runtime/grounding.py](../grm_runtime/grounding.py) | `GroundingClient`、图像/响应校验、缓存和候选选择 |
| [grm_runtime/common.py](../grm_runtime/common.py) | 分数解析、进度公式、目标词解析 |
| [grm_runtime/config.py](../grm_runtime/config.py) | 配置及 head profile 校验 |
| [grm_runtime/prompt.py](../grm_runtime/prompt.py) | 固定 8 图标签与 GRM prompt |
| [sam3_runtime/service.py](../sam3_runtime/service.py) | SAM3 常驻模型与 HTTP 检测服务 |

默认产物布局：

```text
results/monitor_sessions/<启动时间_随机后缀>/
  reference_end.png             # 启动时冻结的共享 goal；后缀保留原文件格式
  <generation_uuid>/
    manifest.json               # 会话、模型/steering 运行信息、模式、判定参数
    online_pred.jsonl           # 每次成功提交一行；未有成功评分时可能不存在
    reference/                 # 会话固定三路起点 PNG
    capture_000001/             # 成功提交的当前三路 PNG；序号可能跳跃
    capture_000002/
```

失败或重复的当前快照会清理，已提交图片和 reference 保留。终态/stop 不自动删除产物，也没有在线结果下载、会话恢复或目录保留期限接口；部署方按需要管理磁盘目录。

双分支的 `manifest.json` 额外保存 `dual_branch` 配置与 `baseline_runtime` 模型信息；每条 `online_pred.jsonl` 与 HTTP 结果一致地包含 `branches / comparison`，便于复查触发失败的两分支分数。

## 7. 异常处理与排查

| 现象 | 当前行为 | 接入方如何判断/处理 |
|---|---|---|
| 长时间 `warming_up=true` | reference 采集失败后重试 | 看 `data.error`，检查 Runtime 地址、metadata、三路 JPEG |
| `warming_up=false`，`poll_count=0` | 已有起点，还没有成功评分；可能重复帧或正在排队/推理 | 检查 age、错误与帧号；不能把初始进度 0 当成模型评分 |
| SAM3 无检测、候选歧义、网络错误、超时或忙 503 | 默认该模式的整个 sample 降级到 HF baseline | 看每个 mode 的 `degraded / reason / applied`；降级分数仍会参与终态判断 |
| 必须保证每轮都有 bbox 才评分 | 可配置 `on_missing_bbox: error` | 缺失将使整轮不提交；上游设置持续无结果的处理策略 |
| 图像/请求/模型指纹不匹配，非法 bbox/span/head 或 hook 未生效 | 本轮错误，不发布部分结果 | 看 `data.error` 与配置/模型一致性；此类错误不作为正常缺框降级 |
| 三路时间差超限或图像不能解码 | 拒绝本轮，保留上次结果 | 检查时间戳单位、时钟基准和相机服务 |
| 画面没有变化或相机帧号未更新 | 视为重复观察，跳过本轮 | `poll_count` 不变且 age 增长；检查各相机 `X-Frame-Id` |
| 模型分数格式无效、任何模式失败或日志写入失败 | 不推进 tracker、previous 或成功/失败窗口 | 修复错误并等待新提交；不要重复累计旧结果 |
| 查询正常但进度很久不变 | 可能是旧结果、重复帧或长推理 | 同时查看 `error / result_age_s / poll_count / timing`，不能只看 HTTP 200 |
| 已终态后没有新评分 | 正常，后台评分循环已结束 | 读取最终结果后 stop；新任务使用新会话 |
| stop 返回 `worker_stopping=true` | 已移除会话，线程仍在完成已有工作 | 旧结果不会发布；不会强制中断 GPU 调用 |

先确认 `engine=hf` 和 `steering_enabled=true`，再查看 **每个 mode** 的 `steering.applied`，才能确认当前结果实际施加了干预；服务启用配置本身不代表每轮都未降级。需要更细证据时检查 `per_layer.*.applied_calls` 以及 `grounding.<label>.selection_status`。

## 8. 其他在线后端与验证范围

现有在线服务仍支持 HF 无干预、原 vLLM 引擎和 deterministic 连通性后端。HF baseline 可使用一份 `enabled: false` 的 steering 配置，或不指定 steering 配置；选择 vLLM 时不能同时启用 steering，否则启动报错。上述 bbox、hook 和严格 HF 分数校验细节针对 HF 路径。

仅检查 HTTP 连通性、不加载模型或访问相机时，可以另开端口：

```bash
conda activate robo-dopamine
python -m monitor_runtime.service --backend deterministic --port 8879 \
  --robot-runtime-url '' --auto-success-after-polls 3
```

deterministic 在每次 status 时增加 `poll_count`，达到次数后返回 success；start 会覆盖同 ID，会话 stop 后仍可查询 failed。它与 GRM 的后台轮次、start 幂等和 stop 移除语义不同，只适合接口连通性演示。

现有单元/契约测试位于 [test_steering.py](../tests/test_steering.py)。双分支回归位于 [test_dual_branch.py](../tests/test_dual_branch.py)，覆盖并发同帧输入、独立进度累计、差值边界与方向、失败优先、原窗口规则、异常/日志失败回滚、停止及 HTTP/配置契约。运行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -q
```

2026-09-08 本次验证通过全部 51 项单元/契约测试，并在同一张 A100 80GB 上加载两个独立 GRM 8B，完成 2 轮 × 2 模式 × 2 分支（8 次评分）。steering 分支全部实际施加干预，baseline 无干预，baseline 单独重跑与并发时输出完全一致，两模型生成后均无残留 hooks。真实模型验证使用历史冻结图片及按 PNG 哈希匹配的 SAM3 检测框，未请求在线 SAM3 或连接物理机器人；本机报告保存于 `results/validation/dual_branch_smoke/report.json`。

此前真实 GRM/SAM3 验证为单模型的双会话在线组合：2 个会话 × 2 轮 × forward/incremental，共 8 次模式评分，验证了干预实际生效、相同在线画面重跑得到相同分数、baseline 隔离及 stop 后不发布旧结果。详见[验证报告](attention_steering_validation.md)和[真实模型验证脚本](../tests/validate_real_steering.py)。该历史报告不代表本次双模型并发的验证结果。

在线验证使用可控 HTTP JPEG 图像回放，Monitor API 通过 FastAPI TestClient 调用；没有连接物理机器人或测量实际部署网络延迟。报告中的最大 status 响应约 3.12 ms 是测试客户端读状态耗时，不是评分耗时或真实系统延迟承诺。GRM 峰值 torch allocated 17.15 GiB 也仅对应这次验证的 GRM 进程，不含 SAM3。当前结果支持实现机制与接口组合可用，不代表新任务成功率、长期遮挡跟踪或生产吞吐量已验证。

# SAM3 加速与可选连续跟踪

默认检测模式仍然可用。Tracker 是另一种可选的在线定位来源，单分支和双分支 HF Monitor 都支持，robot-bridge 不需要修改。

## 1. 为什么之前约 0.85 秒

当前图像检测使用 Transformers SAM3，输入经过 processor 缩放到模型固定尺寸。
此前模型使用 FP32；虽然后处理只返回 bbox，模型内部仍计算 mask。
2026-09-09 检查部署进程时，SAM3 的 `CUDA_VISIBLE_DEVICES=3`，物理 GPU 3 上还有另一个高负载进程，整卡利用率约 97%。baseline GRM 的物理 GPU 1 也与高负载任务共享。

在测试时较空闲、但仍驻留 GRM 权重的物理 GPU 0（A100 80 GB）上，使用三张 1280×720 真机图片、目标 `pen`，预热后重复测试，结果约为：

| 路径 | 检测/跟踪耗时 | 说明 |
| --- | ---: | --- |
| FP32 完整图像模型 | 0.40 s | 视觉编码约 0.33 s，是主要开销 |
| FP32 跳过 mask decoder | 0.39 s | 只节省约 8 ms；bbox/分数与上一行一致 |
| BF16 跳过 mask decoder | 0.09 s | bbox/置信分数有小幅数值变化 |
| BF16 实例跟踪 | 0.077 s | 最终 40 帧离线回放，除首次检测初始化外的 39 帧均值 |

这不是整套部署的周期测试：不含 SSH 采图、Monitor→SAM3 HTTP、GRM、Loop 和浏览器等待。测试限制 CPU 为 4 线程，不能将其与原部署的所有耗时差异都归因于 GPU 争用。首次冷启动明显慢于预热后：上述跟踪测试第一次检测加初始化约 0.88 s。

BF16 能利用 A100 的低精度计算能力。跳过 mask 只是较小的优化，视频 tracking 也不会在 BF16 检测基础上再带来数量级的提速。连续跟踪的额外收益包括实例连续性，以及让采图/定位与 GRM 计算重叠。高 GPU 争用下，这些离线数字不能直接当作在线保证。

## 2. 服务配置

| 配置 | 功能 |
| --- | --- |
| `configs/sam3.yaml` | FP32 检测，跳过 mask，保留原来的检测数值路径 |
| `configs/sam3_fast.yaml` | 可选 BF16 检测，跳过 mask |
| `configs/sam3_tracker.yaml` | BF16 检测 + 独立实例 tracker |
| `configs/sam3_tracker_hybrid.yaml` | BF16 视频跟踪 + 可选周期文本检测、失跟后重检测 |

`dtype` 支持 `float32` / `bfloat16` / `float16`。FP16 尚未在本次测试中验证。
`bbox_only: false` 可恢复完整 mask forward，便于对照。这里不改变 GRM 的 dtype、权重、分辨率或 batch=2 逻辑。

`num_threads: 4` 控制 SAM3 服务的 PyTorch CPU 线程数，避免小型预处理使用过多线程。
`profile: true` 在 HTTP 返回的 `timing` 中额外记录视觉编码、文本编码、DETR encoder/decoder、mask decoder 的 GPU 时间；默认仍返回准备、forward、后处理时间。

BF16 的检测数值可能在阈值或多候选分差边界附近改变“是否检测到/是否歧义”。需要严格复现旧检测结果时使用 FP32 配置；跟踪本身的 bbox 来自 mask，也不保证与逐帧文本检测完全相同。
本次三张图的检测候选数量保持一致，FP32 两条路径的框和分数完全一致；BF16 最大坐标差为 4.522 像素，最大分数差为 0.005653。这是这些测试图片的结果，不是任意输入的误差上界。
本工作区最终原始测量保存于 `results/benchmarks/sam3_20260909.json`，可用文末命令重新生成。

## 3. Tracker 数据流程

1. Monitor 准备并固定任务参考三视角，启动本任务的后台采图/跟踪线程。
2. 后台线程获取一组最新三视角快照。它与 GRM 线程独立运行，慢时不会补处理队列里的旧帧。
3. 对 steering 配置中启用的 AFTER 视角（默认 `after_cam_high`）分别维护 tracker。首次文本检测在通过检测阈值的候选中选择得分最高的 bbox 初始化单实例 tracker；多个候选分数接近也照常绑定，同分时按检测器返回顺序取第一个。
4. 后续新帧运行 tracker。默认 `locked` 模式失跟后持续空框；可选 `hybrid` 模式周期性文本检测，并在失跟后重新检测。更新期间，GRM 仍可读取上一组已完成且未过期的结果。
5. 后台只保留**最新完整三视角 + 对应 bbox**。GRM 读取时固定该组文件；之后即使后台继续更新，也不会改变本轮 GRM 输入或 UI 预览。
6. forward/incremental 共用本轮 AFTER 快照；双分支也共用此快照。steering 从该快照的定位结果构造 attention，baseline 保持无干预。BEFORE 视角如果配置了干预，仍按原来的图像检测路径处理。

这里“最新”指**最新完成跟踪的一组图像**，而不是任意更新的相机图片配上旧 bbox。结果携带文件 SHA-256、图像尺寸、目标词和会话标识；GRM 再次验证文件对应关系。
尚未完成跟踪的更新图像不能提前用于该轮评分，否则无法同时保证 bbox 对应。

## 4. 跟踪策略、丢失、内存与时效

首次绑定采用最高分候选，不因前两个候选分差小而返回空框；同分时按检测器返回顺序取第一个。此规则用于 tracker 初始化；未启用 tracking 的逐帧检测模式仍保留原有的歧义判断。

`tracking.mode` 支持两个值：

| 模式 | 正常跟踪 | 失跟或首帧无候选 | 目标身份 |
| --- | --- | --- | --- |
| `locked`（默认） | 只运行视频 tracker，不再文本检测 | 本任务持续空框，开始新任务才能再次绑定 | 不主动重选实例；模型自身仍可能漂移 |
| `hybrid`（可选） | 视频 tracker + 周期文本检测；有位置匹配候选时重新初始化，匹配失败则保留有效视频轨迹 | 每次后续更新重新检测，选最高分候选恢复；无候选则返回本帧空框 | 允许重新选择，可能选到另一个同类实例 |

`locked` 用于“宁愿没有目标，也不自动换目标”。`hybrid` 更重视恢复定位，会在当前场景上再次解释原始目标词，例如 `left pen` 可能指向桌上剩下的另一支笔；它不提供原物理实例身份保证。周期检测只有与**当前帧**跟踪框重叠的候选才能校正轨迹，检测不匹配不会主动清掉正常视频轨迹。

两种 SAM3 tracker 配置中的参数：

- `max_gap_s: 2.0`：两次更新间隔超过此值，释放旧 GPU 跟踪状态；`locked` 保持空框，`hybrid` 在当前图像上重新检测。
- `min_score`：外层存在分数阈值，类默认 `0.5`，本次部署及混合模式示例为 `0.1`。低于阈值或 mask 为空时进入对应模式的失跟处理。它不是物理身份的可信概率。当前 Transformers SAM3 在 `object_score_logits <= 0`（存在分数不超过 0.5）时已经将 mask 置空，因此降低外层阈值不能找回这些 mask。
- `continuity_iou`：相邻有效跟踪框的最小 IoU，范围 `[0, 1]`，**`0` 关闭硬拒绝**。未写时，`locked` 沿用旧 `match_iou` 值（默认 0.1），`hybrid` 默认为 0。细笔快速位移、旋转或部分遮挡可能让正确框的 IoU 变成 0，因此混合模式示例明确设置为 `0.0`。关闭此检查也会放过模型的大幅漂移。
- `match_iou: 0.1`：`hybrid` 周期检测候选与本帧 tracker 框的最小匹配 IoU，范围 `(0, 1]`。它与相邻帧的硬拒绝阈值分开配置。
- `redetect_interval_s: 5.0`：`hybrid` 正常跟踪时的周期检测间隔；失跟后不等待该间隔，下一次更新即可重试。`locked` 忽略此参数。周期检测无匹配时也更新时间，避免对同一健康轨迹每帧重复检测。
- `memory_frames: 32`：保留初始提示帧及最近历史；不得小于模型的 memory/pointer 窗口（当前至少 16）。不会随着任务长度保存所有视频帧。
- `session_ttl_s: 60.0` / `max_sessions: 8`：服务释放闲置会话并限制会话数量。

只想让原实例的视频 tracker 容忍快速移动、仍禁止文本找回时，可保留 `mode: locked` 并设置 `continuity_iou: 0.0`；空 mask、低分等失跟仍然是本任务终止状态。

2026-09-11 的部署排查确认，最近 10 个会话中 9 个失跟，均记录为 `discontinuous_bbox`，当时运行进程的 `min_score: 0.1` 已生效。对一个会话保存的 54 张评分图像做真实模型离线回放：第 49 步机械臂已提起笔，原视频模型仍框中手中的笔，存在分数 `0.97265625`，但与前框 IoU 为 `0`，旧的硬阈值 `0.1` 会把正确结果拒绝。第 50、51 步仍有框，存在分数分别约 `0.9922`、`0.9961`。同一序列保留 32 和 256 帧历史的输出框、分数全部一致，本次没有发现历史裁剪导致这次失跟。

原始报告保存在本工作区 `results/diagnostics/sam3_tracker_policy_replay_20260911.json`（运行产物，不随 Git 同步）。该回放从首个已发布框初始化，使用保存的 GRM 快照，缺少完整后台视频帧和最初提示状态，因此不是运行中 tracker 状态的精确复现，也不代表所有场景的准确率。重检测与重新初始化会增加发生当轮的耗时，不能把平时的纯视频 tracker 耗时当作混合模式每轮的固定耗时。

每个任务 generation、每个摄像头有独立 session；任务结束/stop 时释放，下一轮重新初始化。跟踪异常会释放部分更新的 GPU 状态：`locked` 保留失跟标记，`hybrid` 可在后续更新中重新检测，二者都不会继续使用损坏的 GPU 会话。

HTTP `/tracking/update` 首次请求显式携带 `initialize: true`，后续全部为 `false`；只有首次请求允许创建会话。Monitor 客户端在发送前记录初始化尝试，首次网络请求失败也不会把后来的图像作为新的首帧。会话过期、关闭或 SAM3 重启后，旧任务的后续请求返回空框，不能重新创建会话；首次请求已经在服务端成功但响应丢失时，可以继续同一个已绑定的实例。不要在重试中再次发送 `initialize: true`。

结果包含 `tracking_mode: locked/hybrid`、`identity_policy: initial_instance/text_redetection`、`tracking_state: tracking/lost` 和 `loss_reason`。失跟结果是 `status: no_detection`、`selection_status: tracking_lost`、`selected: null`、`candidates: []`；首次无检测保留 `selection_status: no_detection`，选中候选时返回 `ok`。

`initialization_count` 记录本任务已初始化多少次，`detection_reason` / `detection_status` 说明本帧检测的触发原因与结果。`last_loss` 持久记录最近失跟的原因、触发帧哈希、tracker 帧号、初始化次数与上一有效框；IoU 拒绝时还记录被拒绝框、存在分数、实际 IoU 和阈值。即使 GRM 没读到触发失跟的那一帧，后续评分仍可看到这些诊断；**`last_loss` 不是当前可用 bbox**，混合模式恢复后也可能保留它。常见原因有 `empty_mask`、`low_score`、`discontinuous_bbox`、`update_gap`、`inference_error`、`session_missing`。

两种模式都不允许后续 HTTP 请求自动创建不存在的任务会话；SAM3 重启、会话过期或显式关闭后需要新任务。客户端接受两种已知策略，但禁止同一任务中途改变策略。

Tracker 不使用图像检测的持久缓存。网络失败时该帧返回明确的缺失定位，按现有 `on_missing_bbox` 策略处理；不会套用上一帧 bbox。默认 `on_missing_bbox: baseline` 时，该轮 GRM 继续评分但不施加目标 attention；配置为 `error` 时报告缺失错误。GRM 不会因为 tracker 返回空框而再次调用文本检测。单分支、双分支以及 forward/incremental 均使用同一组当前图像和空框结果。
最新完整图像过期时，Monitor 拒绝将其作为新评分输入，并暴露错误。

`configs/tracking.yaml` 控制 Monitor 后台线程：

- `enabled: true`：启用独立跟踪。没有传入这个文件时沿用原检测模式。
- `poll_interval_s: 0.1`：采图周期的最小间隔。实际周期还受相机缓存、HTTP 和 tracker 推理限制；不是保证 10 FPS。
- `max_frame_age_s: 2.0`：GRM 读取时允许的最大本地快照年龄，包含本轮取图和跟踪时间。过期就等待后续新结果，不积压旧帧。

Runtime 默认 `camera.cache_s: 0.5`，连续跟踪仍大约受限于 2 次新快照/秒。
建议跟踪试验时改成 `0.1` 并重启 Runtime；更频繁传输三视角会增加从臂编码和 SSH 带宽负载。
当前约 0.25 秒的跨机取图开销也意味着，仅修改缓存不能保证达到 10 FPS。

## 5. 启动和切换

先用 `nvidia-smi` 检查 GPU。以下沿用物理 GPU 3 作为示例，请选择实际可用、尽量不与高负载任务共享的 GPU。
`CUDA_VISIBLE_DEVICES=3` 后配置中的 `cuda:0` 指物理 GPU 3。

仅尝试更快的逐帧检测（Monitor/从臂/Loop 启动方式不变）：

```bash
cd /home/dais/workspace/Robo-Dopamine-delivery
conda activate rewardbench-sam3
CUDA_VISIBLE_DEVICES=3 python -m sam3_runtime.service --config configs/sam3_fast.yaml
```

启用连续跟踪时，SAM3 终端改用：

```bash
CUDA_VISIBLE_DEVICES=3 python -m sam3_runtime.service --config configs/sam3_tracker.yaml
```

尝试“视频跟踪 + 文本重检测”时，改为：

```bash
CUDA_VISIBLE_DEVICES=3 python -m sam3_runtime.service --config configs/sam3_tracker_hybrid.yaml
```

也可在原 SAM3 YAML 中设置 `tracking.mode: hybrid`、`redetect_interval_s: 5.0`、`continuity_iou: 0.0`，其余设备与模型参数按实际部署保留。

Monitor 终端保留已有配置，加一个开关即可。双分支示例：

```bash
cd /home/dais/workspace/Robo-Dopamine-delivery
conda activate robo-dopamine
CUDA_VISIBLE_DEVICES=0,1 python -m monitor_runtime.service \
  --config configs/monitor_dual_branch.yaml \
  --tracking-config configs/tracking.yaml
```

单分支将 `--config` 改为 `configs/monitor_steering.yaml`。也可以在 Monitor YAML 中写 `tracking_config: ./tracking.yaml`，路径相对该 YAML。
若 SAM3 未启用 tracking，Monitor 的准备阶段会显示错误。端口继续使用 SAM3 `8878`、Monitor `8877`，现有 SSH 转发和从臂 Loop 命令不变。
切换配置前结束当前任务，在对应服务终端停止旧服务再启动，避免端口冲突。

首次升级混合模式时，**SAM3 和 Monitor 都需要同步代码并重启**，然后开始新任务；旧 Monitor 客户端只接受 `initial_instance`，会拒绝新的 `text_redetection` 结果。可通过 SAM3 `/health` 的 `tracking_identity_policy` 和 `tracking_config` 查看进程实际加载的模式、阈值和间隔，而不只看磁盘 YAML。完成升级后，仅切换 SAM3 策略配置时重启 SAM3 并开始新任务即可。

回退时去掉 Monitor 的 `--tracking-config`，SAM3 改用 `sam3.yaml` 或 `sam3_fast.yaml`。两者都恢复按 GRM 轮次检测。

## 6. 结果对应、时间信息和验证

UI 的“GRM 评分画面”仍使用同一轮三视角、bbox 和分数；默认“实时画面”则是最新图像加最近一次评分，两者含义不同。
正常持续评分时没有 FIFO 积压，但延时不是常数，会随负载、传输和轮询波动；首次检测初始化还有额外开销。服务停顿后，旧结果仍会越来越旧，不能将其视为当前机器人状态。

Monitor 状态新增 `result.tracking`，含 `ready`、`frames_processed`、`latest_age_s`、`error`。
每轮记录的 `observation.tracking` 包含跟踪周期耗时、读取时图像年龄及定位来源。
`observation.input_age_at_publish_s` 从 Monitor 开始请求快照算到评分发布，**不是硬件曝光时间**；robot-bridge 未提供相机原始时间戳，仍不声称硬件同步。
普通 `latency_s` 在 tracking 模式只计 GRM 这一轮读结果、准备和推理，后台采图/跟踪已经与其重叠，不能只看这个数判断端到端延时。

流式 tracker 使用当前 `rewardbench-sam3` 环境已有的 `Sam3TrackerVideoModel`。本地 checkpoint 包含相应权重，无需下载第二套模型。
该 Transformers 构建的 tracker 有 `fpn_position_encoding` / `fpn_position_embeddings` 字段不一致，项目在当前模型实例上添加别名适配，没有改动 site-packages 或模型张量计算。

不接触机器人或运行中服务的离线复测：

```bash
conda activate rewardbench-sam3
python tests/benchmark_sam3.py \
  --model-path /home/dais/workspace/model/sam3 --device cuda:0 \
  --query pen --images /path/to/frame1.png /path/to/frame2.png /path/to/frame3.png \
  --tracker --tracker-frames 40 --output /tmp/sam3-benchmark.json
```

请使用与图片中物体匹配的 query。该命令比较 FP32 完整 forward、FP32 bbox 路径和 BF16 bbox 路径，并检查 tracker 历史上限。
短序列重复回放用于 API/缓存/性能验证，不能替代连续视频上的跟踪准确率评估。

2026-09-11 混合模式实现验证：`python -m pytest tests -q` 在 `robo-dopamine` 环境中得到 127 项通过、1 项因缺少 SAM3 processor 跳过（另有 49 个子测试通过）。在 `rewardbench-sam3` 环境用真实检测器、视频模型和 `sam3_tracker_hybrid.yaml` 做离线检查，首次绑定、视频更新、周期检测匹配、插入空白图后的空框、恢复原场景后的重检测绑定及任务关闭均通过。该检查强制触发了一次周期检测，用于验证控制流程，不是性能或准确率测量；结果保存在本工作区 `results/diagnostics/sam3_hybrid_smoke_20260911.json`，不会随 Git 同步。

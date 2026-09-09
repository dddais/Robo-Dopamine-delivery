# Monitor 优化与测量

2026-09-09 的优化针对在线每轮重复工作，适用于单分支和双分支：

- 直接发送冻结 RGB PNG 的原始字节，省去重新解码和 PNG 压缩。JPEG、灰度或带
  alpha 等输入仍按原逻辑转 RGB PNG；SAM3 继续校验实际请求字节的哈希和图像尺寸。
- 同一次 `HFBackend.inference_batch` 内，复用相同文件、相同目标词的定位结果，
  后续模式不再重复请求 `/health`。新一轮重新检查 SAM3 模型指纹，路径被改写时
  也会失效。歧义/无检测结果可复用，网络异常不缓存，允许后续模式重试。
- 同批次复用已解码图片，缓存最多 16 张图片、128 条检测记录，调用结束后释放。
  第二阶段进一步将 forward/incremental 合成一次 processor 和 generate；每行
  token spans、bbox 映射、attention mask、KV cache 和得分仍独立。
- SAM3 使用 bbox 后处理，省去 mask 的 sigmoid、插值和二值化；boxes/scores
  一次性传回 CPU。模型前向和 FP32 精度不变，内部 mask decoder 仍执行。
- `monitor_steering.yaml`、`monitor_dual_branch.yaml` 的轮间等待均从 `1.0` 改为
  `0.1` 秒；CLI/类在不传 YAML 时的默认值仍为 1 秒。

模型权重、图片像素与分辨率、forward/incremental 模式、attention bias、greedy 解码策略、
进度融合公式和终态判定参数没有调整。稳定窗口仍按推理轮次计数；评分变快后，
相同窗口覆盖的实际时间会变短。缓存没有沿用上一轮 bbox。

## 第一阶段：CPU 准备重用验证

使用 `mon-5252c4ed8b7c4cee997ad705e5759f9e` 会话前 3 轮冻结真机画面，每轮重复
3 次，交替运行原始准备流程和优化流程。使用真实 GRM processor，PyTorch CPU
线程数设为 2；加载记录中的 SAM3 bbox，不加载模型权重，不访问机器人或在线服务。

| 测量项 | 原流程均值 | 优化后均值 |
|---|---:|---:|
| 每轮两个模式的 CPU 准备（读图、processor、定位载荷准备和 bbox 映射） | 0.960 秒 | 0.132 秒 |
| 单张主视角图的 PNG 载荷准备 | 0.378 秒 | 0.00042 秒 |
| 配置中的轮间等待 | 1.0 秒 | 0.1 秒 |

18 对样本的 processor 输出张量逐元素完全一致，图像 RGB 像素、token spans、
bbox 对应的 token 位置以及检测缺失状态也一致。回归测试另覆盖两个模式的图像
token 偏移不同、同路径文件改写、目标词变化、SAM3 模型指纹变化、降级与重试、
baseline 不受 steering hook 影响，以及单/双分支完整结果发布。

这不是完整模型推理基准：**没有重新运行 SAM3 或 GRM，也没有重新比较真实模型
生成得分**。历史会话平均评分周期为 5.29 秒；若其余耗时相近，扣除准备阶段节省
约 0.83 秒和轮间等待节省 0.9 秒，预计约 3.6 秒一轮。完整周期应以更新服务后的
`inference_updated_at` 差值为准，双分支尤其应单独测量。

## 第二阶段：真实 GRM 批处理验证

默认 `hf_batch_size: 2`，每个独立模型将两个模式合为一次生成；双分支并发时是
steering、baseline 各生成一次。开启第三个 backward 模式时，默认尾项为 batch=1。
`--hf-batch-size 1` 同时让两个分支串行，便于对比。每行都有自己的干预计划，
降级/baseline 行保留原 attention mask，异常后统一清理 hooks，整轮验证后才发布。

使用 `26-09-09-18-23-46_0ff4892b/5a62687e53564c2583c39daf955a394c` 会话的 3 轮
冻结图片，在 A100 GPU 0 上加载真实 GRM 8B BF16 权重，以记录的 SAM3 bbox 做
串行/batch=2 对照。分别覆盖 baseline 和 steering，共 12 个正常评分，输出文本完全
一致。最终复测中，正常两模式的准备加 GRM 生成均值从 **1.707 秒降到 1.249 秒，减少约 27%**；GRM 进程
峰值 torch allocated 为 **17.85 GiB**。这是单模型逐条件对比，未测两个真实 GPU
并发的完整双分支周期，也不包含 SAM3 前向、HTTP 采图与轮间等待。

补充测试发现当前 Qwen3-VL BF16 在异长提示词合成 padded batch 时可能出现明显
评分变化，即使有效 token、图像 embedding、RoPE 和 attention mask 都与串行一致；
差异在未干预的早期层已经出现。因此最终实现先按有效 token 长度分组，异长自动
拆批。额外的两项异长/混合干预评分对照在拆批后也完全一致，最终共 14 项对照通过。
同一在线任务、固定相机尺寸的 forward/incremental 仍等长合批。补齐 mask
的数学隔离另有测试；并不以此承诺任意 BF16 批处理与串行生成都逐位一致。

SAM3 在 `rewardbench-sam3` 环境中调用真实 Transformers 后处理 API 对比，
覆盖 presence 分数开关、不同尺寸、阈值及空结果，boxes/scores 逐元素完全一致。
这验证后处理等价，不是 SAM3 整体提速测量。双分支集成测试覆盖两个独立 HF 实例
并发各生成一次、baseline 不受干预、融合/差值规则和批次耗时不重复累加。

复现真实 GRM 对比（会加载模型到指定 GPU；不访问在线服务或机器人）：

```bash
python tests/validate_batched_grm.py \
  --session results/monitor_sessions/26-09-09-18-23-46_0ff4892b/5a62687e53564c2583c39daf955a394c \
  --device cuda:0 --steps 3 --padding-probe \
  --output /tmp/monitor-batch-validation.json
```

`--padding-probe` 额外使用一个不同长度的测试提示词，验证自动拆批后的串行等价；
该提示词仅用于离线测试，不下发机器人。完整周期仍以部署后的日志为准。

## 复现第一阶段准备测量

在 `robo-dopamine` 环境、仓库根目录执行；传入包含 `manifest.json` 和
`online_pred.jsonl` 的会话目录，保留其冻结图片和上一级 `reference_end.*`：

```bash
python tests/benchmark_monitor_preparation.py \
  --session results/monitor_sessions/26-09-09-16-57-05_f1faa82b/57f2548134b745e286f4d82bba81fc1f \
  --steps 3 --repeats 3 \
  --output /tmp/monitor-preparation-benchmark.json
```

脚本仅适用于记录了 `after_cam_high` 定位结果的 HF 会话；当前支持有效、歧义与
无检测记录，不支持没有 SAM3 返回值的网络失败记录。耗时随 CPU 负载变化。

## 部署后检查

任务结束后更新服务器本仓库，使用原命令重启 **SAM3 和 Monitor**。从臂 Runtime、
loop 和 SSH 转发不需要调整。SAM3 模型指纹含新的后处理版本，旧定位缓存会失效。

每条 `[GRM]` 日志现在输出 `latency=…s`，它包含本轮采图和模型处理，不含随后
0.1 秒等待。`online_pred.jsonl` 还提供：

- `timing.prepare_ms / grounding_ms / grm_ms / total_ms`：准备、定位、GRM 和整轮耗时。
- `modes.<mode>.steering.image_cache_hits`：本模式复用图片的次数。
- `modes.<mode>.steering.batch_size / batch_id / batch_index`：实际批量生成的行数、
  批次和行号；正常两模式应为同一 batch_id，batch_size=2，行号分别为 0、1。
- `batch_grm_ms` 是整批生成耗时；模式 `grm_ms` 是按行数均分的记账值。不能将
  两个模式重复携带的 `batch_grm_ms` 直接相加；顶层 `timing.grm_ms` 已正确计数。
- `modes.<mode>.steering.grounding.after_cam_high.reused_in_batch`：同轮定位结果复用；
  通常 forward 为 false，incremental 为 true。

双分支的各阶段计时是两分支加总，会与墙钟时间不同；实际评分频率以相邻记录的
`inference_updated_at` 差值衡量。复用记录中的 SAM3 `latency_ms` 是原始检测的
历史耗时，不是再次发生的检测耗时。

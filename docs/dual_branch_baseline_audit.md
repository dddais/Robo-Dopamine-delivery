# 双分支 baseline 核查（2026-09-11）

本次核查支持当前 baseline 实现正确：同一 GRM checkpoint 的独立 HF 模型实例，关闭
attention steering，使用与 steering 分支一致的任务及八图输入，独立计算分数和进度。

## 核查对象与证据

运行目录：`results/monitor_sessions/26-09-11-16-53-10_ba272a41`。
共 12 个任务、461 次评分，全部为 `forward` 模式，`continuous_monitoring: true`。
manifest 中 steering / baseline 的设备分别为进程内 `cuda:0` / `cuda:1`。

| 核查项目 | 结果 |
|---|---|
| 两分支 checkpoint、model config、processor、system prompt 哈希、解码及精度 | 全部一致 |
| manifest 差异字段 | 仅 steering_config、ranking_sha256、fingerprint |
| baseline 的干预状态 | 461/461 为 condition=baseline、enabled=false、applied=false；per_layer 和 grounding 为空 |
| 两分支的八图路径、token spans、图像网格 | 461/461 一致 |
| steering 未生效时的自然对照 | 71 次，baseline / steering 的原始输出文本全部一致 |
| 从原始百分数字符串独立重算 score / progress / hop | 2,766 项检查一致；分支融合值与差值亦一致 |
| 独立真实模型复算 | 12 个任务各抽首、中、末三帧，共 36 次，原始输出文本 36/36 完全一致，分数最大差值 0 |
| 相关回归 | test_dual_branch、test_monitor_preparation、test_steering：71 项测试、39 个子测试通过 |

复算在当时空闲的物理 GPU 2（A100 80GB）进行，`CUDA_VISIBLE_DEVICES=2`；新进程内
使用 `cuda:0`。独立加载真实 8B BF16 权重，完全不加载 steering 配置与 heads，使用保存的
八图输入，未联系在线 Monitor、SAM3 或机器人。峰值 torch allocated 为 17.07 GiB。
详细逐帧结果见 [JSON 报告](dual_branch_baseline_audit.json)。

日志经过 JSON 序列化后，model config 的数字字典键会变成字符串。复算检查先将本机
manifest 同样做 JSON 归一化，再确认模型参数、processor、prompt 和解码配置一致。
该序列化差异不代表模型参数变化。

## 代码路径

- `monitor_runtime/grm_backend.py`：baseline 使用同一 `model_path`，但传入 `steering_config=None`；
  检查两个底层模型不能是同一实例。输入由同一份 samples 深拷贝，仅条件标识不同。
- `grm_runtime/hf_backend.py`：关闭 steering 时不做定位、不生成干预计划、不挂 attention hooks；
  每个实例有自己的模型和锁。baseline 仍使用正常的模型 attention。
- `monitor_runtime/grm_backend.py`：baseline 使用独立的 `baseline_tracker`，没有混用 steering 的累计值。
- Runtime UI 读取 `branches.baseline.progress` 显示 baseline。本次 forward-only 下，融合进度为
  forward 分数裁剪到 [0, 1]，没有套用 incremental 的累计公式。

## 解释范围

这里的 baseline 是 **关闭 steering 的 HF GRM**：BF16、eager attention、greedy、
`max_new_tokens=64`。旧 vLLM 路径使用 temperature=0.1、top_p=0.9、top_k=50、
max_tokens=1024；本次核查没有证明两种推理引擎和解码配置的输出逐值等价。

本次真实复算覆盖这批 forward-only 任务；多模式和批处理隔离由回归测试覆盖。
独立复算说明日志中的 baseline 分数来自预期模型和输入。分数是否准确反映真实任务完成度
还需要人工标注或任务成功信号验证。该批任务的 reference end 实际为 274×170、RGB=(237,237,237)
的灰色占位图，缺少真实任务完成场景这一参考；两分支使用同一占位图。

本次新增核查脚本和报告，生产推理代码未修改。

## 复现

先选择空闲 GPU，使用与运行日志匹配的环境和本地 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python tests/validate_baseline_replay.py \
  --run results/monitor_sessions/26-09-11-16-53-10_ba272a41 \
  --samples-per-session 3 --output /tmp/dual-branch-baseline-replay.json
```

脚本检查模型配置及重建输入是否与记录一致，逐帧比较原始输出和分数；不一致时保存
比较结果并以非零状态退出。需保留运行目录里的 reference、逐帧 PNG 和 JSONL。

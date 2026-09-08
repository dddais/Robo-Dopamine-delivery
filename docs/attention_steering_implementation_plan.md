# GRM Attention Steering 接入方案

日期：2026-09-08。状态：待实施的设计文档；本次只新增本文档，不修改功能代码、配置或依赖，不启动模型和服务。

目标：在 `Robo-Dopamine-delivery` 中实现与 `test_data_suc.py` 对齐的离线 attention steering，以及从 Robot Runtime 获取画面、经 SAM3 提取 bbox、在 GRM 推理时施加 attention bias 的在线监控。离线和在线共用干预实现与进度公式。

## 1. 结论与边界

采用以下方案：

1. 保留现有 vLLM 推理路径；新增 Hugging Face（HF）`eager` 推理后端承载 attention steering，通过统一推理接口接入离线脚本和 `GRMMonitorBackend`。
2. 提取原仓库中的 bbox → image token 映射、按层选头、加性 attention mask hook。默认使用 `after_cam_high`、`query_scope=all`、`negative_scope=target_span`；同时设计并实现按相机独立 bbox 的多视角配置。
3. SAM3 使用独立 Python 环境和常驻进程，通过 HTTP 接收**实际送入 GRM 的冻结图像**。在线首版对每个新采样帧做图像 grounding，不依赖完整视频，不使用未来帧。
4. 离线同时输出 HF baseline 与 HF steered 三模式曲线；在线每轮只运行配置选中的策略，不默认双跑。状态接口保持现有响应外壳，增加可追溯诊断。
5. 模型权重、head ranking、bbox、token span 和生成参数均记录版本。未检测到目标时显式降级为同一 HF 模型的无 bias 推理；配置错误或 token 对齐错误不能静默宣称 steering 已生效。

首版不修改 GRM 权重、不引入训练，不移植完整 benchmark 工程。vLLM 内核级 bias、在线 SAM3 视频 tracker、跨相机三维关联属于后续优化；本方案中的逐帧 SAM3 grounding 已完整覆盖在线 bbox 提取要求。

本设计给出功能、接口、改动位置和验收方法，不声称 steering 一定改善任务判断。工程生效与模型收益须分别验收。

## 2. 已核对的现有实现

读取基线：源仓库 `/home/dais/workspace/Robo-Dopamine`，HEAD `5db2d1a351869a4a699087f9adc71df4cacb422d`；delivery HEAD `5043a42f0f523a3e013bb0b3d16b00a7dc4c0277`。源仓库有既存未提交工作，本文依据实际读取的下列文件，不修改这些工作。delivery 在编写文档前工作区干净。

### 2.1 原仓库的可复用代码

以下链接均指向当前已存在的源文件。

| 文件 / 函数 | 当前实现 | 接入决策 |
|---|---|---|
| [steer_grm_heads.py](../../Robo-Dopamine/steer_grm_heads.py)：`make_steering_hook`、`_run_with_bias`、`generate_score` | 在 decoder `self_attn` 注册 forward pre-hook；目标图像 token 加 `+delta`，同图其他 token 加 `-delta`；实际 `generate()` 自由生成分数，`finally` 移除 hook | 复用干预机制。文件顶部仍有 teacher forcing 描述，以实际生成函数为准 |
| [steer_progress_curve.py](../../Robo-Dopamine/steer_progress_curve.py)：`trace_curve`、`scores_to_progress` | 对整条轨迹逐步比较 baseline、目标干预、错误区域和控制 head，按三种模式计算进度 | 复用整轨迹实验组织方式，不直接使用其硬编码路径 |
| [scan_localization_heads_best.py](../../Robo-Dopamine/scan_localization_heads_best.py)：`infer_image_spans`、`load_model_and_processor` | HF eager 加载；识别 8 个图像 token span；提取注意力供选头分析 | 作为原始机制来源，生产推理不搬入扫描与绘图依赖 |
| [mydata_bench/attention_eval/masking.py](../../Robo-Dopamine/mydata_bench/attention_eval/masking.py)：`bbox_to_token_positions`、`resolve_negative_positions`、`make_attention_mask_hook` | bbox 网格相交映射、span 长度校验、4 种 query scope、4 种 negative scope、hook 调用诊断 | 作为共享映射与 hook 的主要移植来源，比早期脚本校验更完整 |
| [mydata_bench/attention_eval/runtime.py](../../Robo-Dopamine/mydata_bench/attention_eval/runtime.py) | HF eager、8 图对齐、hook 上下文、greedy decode、完整 incremental hops | 复用 HF 与 hook 生命周期；重新适配 delivery sample，不搬入 benchmark 的 endpoint/track 数据协议 |
| [mydata_bench/attention_eval/ranking.py](../../Robo-Dopamine/mydata_bench/attention_eval/ranking.py) | 域内 excess attention mass 排序及跨来源排序聚合 | 用于独立校准数据上的选头；运行时只加载冻结结果 |
| [rank_grm_heads_by_comics.py](../../Robo-Dopamine/rank_grm_heads_by_comics.py) | 使用漫画对应关系识别候选 heads，依赖外部 gaze-heads 工具 | 可选选头来源，不作为在线运行依赖 |
| [mydata_bench/grounding/sam3.py](../../Robo-Dopamine/mydata_bench/grounding/sam3.py)：`SAM3Grounder.candidates` | `Sam3Processor + Sam3Model` 文本条件分割，后处理为原图坐标 bbox；无 box 时可从 mask 取外接矩形 | 移植图像 grounding 到独立 SAM3 服务 |
| 同文件：`track` | 官方 SAM3 video predictor，`start_session → add_prompt → propagate_in_video → close_session`，输入完整视频路径 | 可用于离线扩展，不能当作现成实时流 API |
| [grounding/parser.py](../../Robo-Dopamine/mydata_bench/grounding/parser.py)、[grounding/base.py](../../Robo-Dopamine/mydata_bench/grounding/base.py) | 目标词解析、query 构造、bbox 合法性校验及关系目标选择 | 参考显式目标词、属性和关系处理；避免整句任务被当成单一分割目标 |

需要保留的区别：

- 早期 steering 脚本使用 GroundingDINO；SAM3 已在后续 `mydata_bench` 工程提供。此次组合两者可复用部分，不声称早期脚本本身已经接入 SAM3。
- 旧 head JSON 使用 `top_heads` / `rankings.<name>`，新 ranking 使用 `ranking`，不能用旧 `parse_heads()` 直接读取所有文件。
- 新版 masking 支持 `all_visual / target_span / other_spans / none`；runtime 默认值与不同实验配置不同。delivery 明确选择 `target_span`，不依赖原模块默认值。
- 原 runtime 默认返回生成过程 attentions 用于研究。在线只保留轻量 hook 诊断，默认 `output_attentions=False`，否则可能保存大量全层注意力矩阵。
- 原 runtime 的部分多图干预分支复用同一个 bbox，适用于重复单视角输入。真实三相机必须分别 grounding，不能把主相机坐标直接套到腕部相机。
- 原 incremental 工具在缺失 bbox 时可选择最近帧的 bbox。在线首版禁止这样做，以免使用其他时刻甚至未来帧位置。

源仓库本地 8B checkpoint 的 `config.json` 明确为 `Qwen3VLForConditionalGeneration`，36 个语言层、32 个 query heads、8 个 KV heads，视觉 `spatial_merge_size=2`。这些数值不能硬编码到通用运行时，也不能用于假定 4B 配置。

已确认可读的选头资产：`results/mydata_bench/experiments_v2/attention_12_grm_incremental_after_official/in_domain_ranking.json`，记录 34 个 discovery samples、36 层、32 heads、跳过前 8 层，fingerprint 为 `4c255027fe4ca83be7143872c14f3f7ea212f6120d32bd9408c181962ad4dc5a`。可作为该 8B incremental 协议的实验候选，不能直接视为 4B、forward/backward 或 delivery 胡萝卜微调模型的已验证配置。`steer_progress_curve.py` 中旧胡萝卜 ranking 默认路径在当前源仓库不存在。

### 2.2 delivery 的接入点

| 现有文件 | 当前行为 | 需要的接入 |
|---|---|---|
| [examples/inference.py](../examples/inference.py)：`GRMInference` | 顶层导入 vLLM；模型固定为 `LLM`；每个 sample 为任务 + 8 图；`inference_batch` 返回原字段加 `pred` | 后端工厂、HF 实现、共享提示词与图像样本工具；vLLM 改为惰性导入 |
| [test_data_suc.py](../test_data_suc.py) | 加载一次模型，运行 forward/incremental/backward，读取预测再平均画图 | 增加可配置条件循环、独立状态和 bbox 缓存，采用真实帧时间戳 |
| [monitor_runtime/grm_backend.py](../monitor_runtime/grm_backend.py) | 每会话后台线程；冻结为 PNG 后推理；共享模型用 `_infer_lock` 串行；`status` 读缓存 | 在快照与构造推理请求之间 grounding；在模型推理范围内安装/移除 hook |
| [monitor_runtime/core.py](../monitor_runtime/core.py) | 相对分数之外的状态模型；融合进度窗口决定终态 | 保留状态枚举和阈值语义；只接受完整有效的一轮评分更新 |
| [monitor_runtime/service.py](../monitor_runtime/service.py) | start/status/stop/health；扁平 YAML 加 argparse 覆盖 | 显式解析新的 steering 配置、目标词覆盖和后端能力 |

现有 prompt 图像顺序固定为：

| 索引 | label | 含义 |
|---|---|---|
| 0 | `reference_start` | 起始主视角 |
| 1 | `reference_end` | 目标图 / blank goal |
| 2–4 | `before_cam_high / before_cam_left_wrist / before_cam_right_wrist` | BEFORE 三视角 |
| 5–7 | `after_cam_high / after_cam_left_wrist / after_cam_right_wrist` | AFTER 三视角 |

### 2.3 依赖约束

delivery 的 [requirements.txt](../requirements.txt) 固定 `torch==2.8.0`、`transformers==4.57.0`、`vllm==0.11.0`。原仓库 [SAM3 环境文件](../../Robo-Dopamine/mydata_bench/environments/rewardbench-sam3.yml) 使用 Python 3.12、Torch 2.8 CUDA 12.8 wheels、Transformers 5.0.0。

不在 delivery 的 vLLM 环境中直接升级 Transformers。SAM3 独立环境部署；GRM HF 路径首先在现有 4.57.0 环境验证 Qwen3-VL mask hook。HF 支持后端不等于任意 attention kernel 都支持；首版限定 eager，禁用未经验证的 FlashAttention/SDPA steering 配置。

## 3. 共享模块与接口设计

以下名称均为**拟新增或拟修改**，本次未创建这些代码文件。

| 拟建模块 | 职责 |
|---|---|
| `grm_runtime/protocol.py` | 唯一 prompt、8 图 label、sample/输出类型、严格 score 解析和进度递推 |
| `grm_runtime/media.py` | 抽帧、不可变图像快照、frame ID / timestamp、采样索引和内容哈希 |
| `grm_runtime/factory.py` | 按 `engine=vllm|hf` 惰性构造后端；检查 steering 能力 |
| `grm_runtime/vllm_backend.py` | 封装现有 LLM 推理行为，不施加 bias |
| `grm_runtime/hf_backend.py` | processor、HF eager 模型、单样本 generate、共享模型锁 |
| `grm_runtime/steering/config.py` | schema、配置验证、ranking 格式转换、模型兼容性 |
| `grm_runtime/steering/spans.py` | 按图像出现位置建立 span，bbox → token，正负 token 集合 |
| `grm_runtime/steering/hooks.py` | hook 上下文、query scope、bias 及应用诊断 |
| `grm_runtime/grounding/client.py` | 统一 SAM3 HTTP client；离线缓存读写 |
| `grm_runtime/grounding/targets.py` | 任务目标词映射、解析与候选实例选择 |
| `sam3_runtime/service.py` | 独立环境常驻 SAM3 模型、图像解码和分割 API |
| `examples/offline_steering.py` | 三模式 × 条件的离线流程，与 `test_data_suc.py` 共用执行函数 |

继续对外暴露 `examples.inference.GRMInference`，增加可选 `engine`、`steering_config`、`grounding_client`，旧调用默认 `engine=vllm, steering.enabled=false`。`run_pipeline` 的现有参数保持可用。

内部核心契约：

```text
InferenceBackend.inference_batch(samples) -> list[Prediction]
GroundingClient.detect(frame_snapshots, target_spec) -> list[GroundingResult]
SteeringPlanner.prepare(inputs, ordered_images, grounding, head_profile) -> SteeringContext
```

`Prediction` 保留 `id/task/image/eval_mode/pred`，新增 `valid`、`parsed_score`、`engine` 和 `steering` 诊断。`eval_mode` 在离线样本中补为显式字段，不再只从目录名猜测。

HF 首版内部逐 sample 处理，实际 microbatch 固定为 1；`inference_batch` 仍保留列表输入输出、原顺序与一一对应 ID。三个模式各自 tokenize 后重新计算 token positions。禁止把首个样本的 bbox mask 广播到其他样本。真正的 padding batch 优化需要后续独立验证。

对于关闭 steering 的旧 vLLM 调用保留原解析兼容行为；新增 HF/steering 流程采用严格解析，坏输出不转换为合法的 0 分。共享进度模块统一数学公式，兼容路径的解析策略显式区分并记录。

## 4. Attention bias 的具体实现

### 4.1 数学语义

对选中的语言 decoder 层 `l`、query head `h`：

```text
logits[l,h,q,k] = QKᵀ / sqrt(d) + causal_mask[q,k] + bias[l,h,q,k]
bias = +delta   k 属于指定目标 bbox 的视觉 tokens
       -delta   k 属于所选图像 span 中其余视觉 tokens（target_span）
        0       其他 keys、其他 heads、未命中的 query scope
attention = softmax(logits)
```

默认 `query_scope=all` 表示 prefill 所有 query 行以及后续 cached decode 都施加 key bias；新生成文本 key 的 bias 恒为 0。它不是“只调整评分数字 token 的 attention”，也不是对输出分数做后处理。

`delta` 是 attention logit 的加性量，不是概率。目标与被抑制 key 的相对优势变化为 `2*delta`，实验中的 `delta=6` 已是较强干预，必须做敏感性评估。

默认只干预 AFTER 主视角；支持配置多个 AFTER 相机，以及 BEFORE+AFTER。参考起止图默认不干预。`backward` 使用目标图作为 BEFORE，但仍在当前 AFTER 图上的 bbox 施加 bias，因此 blank goal 不会成为检测或干预对象。

### 4.2 图像、bbox 和 token 一致性

1. 图像唯一键包含原始 observation 标识、相机、预处理配置哈希和图像内容哈希。使用 PNG/无损传输，SAM3 与 GRM 解码到相同像素画面。
2. 先鱼眼去畸变，再在去畸变图像上运行 SAM3，并将同一图像交给 GRM。不能直接将原始鱼眼 bbox 线性缩放到去畸变图。
3. bbox 统一为原图像素 `xyxy`，并携带 `width/height/coordinate_space`。拒绝 NaN、反向、空框；裁剪越界后仍为空则无效。
4. 从 processor 产出的 `input_ids` 中按 `image_token_id` 识别连续 span，与 8 个 `image_grid_thw`、8 个 image occurrence 一一对应。重复路径仍是不同 occurrence，不能合并 token span。
5. 从实际 processor/model vision config 读取并交叉验证 merge size。验证 `span_length == t × (grid_h/merge) × (grid_w/merge)`，不假定固定视觉 token 数、方形图或固定 patch size。
6. 在真实宽高上计算每个视觉网格单元与 bbox 的矩形相交；相交单元全部纳入，避免小目标落在网格中心间隙。按 merge 后 row-major 顺序映射到序列绝对 key 索引。
7. 当前 Qwen 图像 resize 保持归一化位置时可采用上述映射；若实际 processor 有 crop/pad/额外几何变换，必须记录并应用相应变换，不能继续直接归一化套框。
8. 不同相机分别映射 bbox。多个目标仅在同一指定 occurrence 内做 token union，并记录各目标；正负集合必须互斥且仅包含视觉 tokens。

### 4.3 hook 与并发约束

- 使用模型实际 `num_attention_heads`，不是 `num_key_value_heads`；层和 head 越界、重复、空 ranking 都必须在加载阶段报错。
- 选中层的 bias 基础张量为 `[1,Hq,1,K]`，与浮点 causal mask 相加，保留原本被禁止的 future/padding 位置。生成扩展 KV 长度时只补零，不重新解释图像位置。
- 只接受经过验证的 mask 形态和浮点语义。`mask=None`、布尔 mask 或不支持的 attention 实现不能无声 no-op；给出 `hook_not_applied` 或配置错误并停止该轮发布。
- 支持 `all/prefill/last_prompt/decode` 的显式配置；默认 all。以 prefill/cache 状态校验 query 阶段，不能将所有 `q_len=1` 场景都未经判断视为 decode。
- 锁覆盖“安装 hook → 完整 generate → finally 移除 hook”。零 bias 和 baseline 也使用同一模型锁，以免另一个会话的 hook 污染 baseline。
- 安装部分 hooks 失败、生成异常、停止会话、OOM 都经过 `finally` 清理；不复用跨样本 KV cache。当前 sample 内可用 `use_cache=True`。
- 默认不返回全量 attentions，记录各层 `prefill_applied_calls/decode_applied_calls`、实际 head/token 数。小样本 debug 模式才提取选头/选 query 的热图，确认物理作用位置。
- 首版 eager 可能分配 `[B,H,Q,K]` mask 和 attention 中间量；即使不返回 attentions，显存也不等于 vLLM。先用 delivery 的图像像素上限和真实 token 长度测量，不承诺 1 Hz。

### 4.4 选头资产

运行时支持三种输入形态：`ranking`、`top_heads`、显式指定 `rankings.<name>`，归一化为唯一 `(layer, query_head)` 有序列表。top-k 截断发生在过滤非法项、验证跳层策略之后；数量不足报错。

配套 profile 至少记录 checkpoint/revision 或权重指纹、model_type、层/head 数、processor 和 prompt 哈希、图像配置、ranking 来源、query mode、eval mode、intervention labels、negative scope、top-k 和 bias。旧 JSON 缺 metadata 时，先生成并审阅配套 profile；不将结构匹配视为效果匹配。

提供按 eval mode 选择 profile 的能力。可显式共用一个 profile 做探索，但每种模式独立报告效果。换 8B→4B、预训练→微调 checkpoint 后，重新选头或重新验证候选 heads；禁止自动沿用旧结果而标记为已验证。

没有匹配资产时，在独立成功示范上收集最后 prompt query 到 bbox 的 attention mass，使用 `excess_mass = bbox_mass - bbox_token_fraction * image_mass` 排序；incremental 必须覆盖所有 hops。此为后续实施的校准步骤，交付包需包含冻结的 profile，而不是让在线服务启动时重新选头。

## 5. SAM3 grounding 设计

### 5.1 目标词与实例选择

目标词优先级：`/monitors/start` 显式 `target_queries` > 配置中的精确任务映射 > 确定性任务解析。任务映射键先去首尾空格并合并连续空白，兼容现有脚本任务末尾空格；不通过模糊匹配改变目标属性。GRM 始终使用完整 subtask 指令，SAM3 使用操作对象短语，例如 `pick the white cube and put it on yellow plate` 对应 `white cube`。盘子等放置目标可用单独 role 配置，不能无条件与操作对象合并。

复用 parser 中的短语、颜色/属性处理思路。首版显式支持已配置任务和可解析的简单操作指令；中文、复杂组合/关系指令无法可靠解析时要求配置映射，在 start 阶段返回明确目标配置错误，不暗中省略 grounding。

相同类别多个实例时，保留全部候选、score 和 query。初始选择优先满足完整属性的 query；属性不明确时不以“泛类最高分”代替原目标。后续新帧可用同相机历史 bbox 的 IoU/中心位移辅助候选选择，但返回的 bbox 必须来自当前帧 SAM3 结果。快速运动或无法区分实例时标记 ambiguous，并执行缺失策略。

关系目标如“左侧杯子”需要显式关系选择规则或任务映射；不声称单纯文本分割保证实体身份。跨相机不假定共享 track ID，各相机只负责自己画面的目标定位。

### 5.2 独立服务契约（拟新增）

`POST /grounding/detect` 使用 multipart：一个 JSON `metadata` 部件和按 camera/occurrence 索引的 PNG 文件部件。请求包含 `request_id/session_generation/frame_id/camera/image_sha256/target_queries/preprocess_sha256`；服务从实际图片读取宽高并核验哈希。离线文件路径不作为跨服务的必需输入，部署不依赖共享文件系统。

每张图返回：

```json
{
  "request_id": "m1-g1-step10",
  "frame_id": "obs-123",
  "camera": "cam_high",
  "image_sha256": "<sha256>",
  "image_size": [640, 480],
  "coordinate_space": "input_image_xyxy",
  "status": "ok",
  "candidates": [
    {"bbox": [120.0, 80.0, 210.0, 180.0], "score": 0.92, "query": "carrot"}
  ],
  "model_fingerprint": "<sam3 weights/config hash>",
  "latency_ms": 120.0
}
```

示例坐标和耗时仅展示字段，不是实测值。状态区分 `ok/no_detection/error`；客户端目标选择再区分 `ambiguous`。`GET /health` 返回模型加载状态、版本与设备。常驻加载一次模型，内部有界请求队列和推理并发限制；繁忙时及时返回可识别错误，不无限排队。

移植 `candidates()` 的分割后处理、阈值和 bbox 裁剪；mask 不必跨网络回传，debug 时单独落盘。SAM3 threshold 0.3、mask threshold 0.5 作为与原仓库一致的初始候选，仍需按实际任务校准。

### 5.3 缺失、缓存和资源

默认 `on_missing_bbox=baseline`：若该 sample 的任一**配置要求的**干预视角无合法目标，整个 sample 运行无 bias HF 推理，记录原因，避免悄悄变成部分视角干预。未配置为干预目标的相机检测失败不影响此 sample。

SAM3 超时/不可用可按同一策略降级，并暴露 degraded；图像哈希、坐标或 span 对齐不一致是工程错误，应拒绝该轮，而不是套用错误框。默认 `max_bbox_age_steps=0`；除非完全相同 image hash，否则不复用旧 bbox。默认不使用上一帧框来假冒新检测结果。

离线缓存 key 至少包含图像哈希、query/target selector 版本、SAM3 模型/阈值和几何预处理哈希。在线复用同一冻结 current 图像的检测结果给多个评分模式；reference/previous 的 bbox 跟随其不可变图像保存。

推荐 GRM 与 SAM3 分别绑定 GPU。单 GPU 可串行调度，但需单独验证两模型驻留峰值；进程隔离不自动解决显存容量，也不建议同时驻留 vLLM 和 HF 两份 GRM。SAM3 服务只执行自己的设备推理，不改变 monitor 进程环境变量。

## 6. 离线 attention steering 流程

### 6.1 与 test_data_suc.py 对齐

新增 `examples/offline_steering.py`，并在后续实施时将 `test_data_suc.py` 改为带 `main` 保护、可读配置的薄入口，共用离线执行函数。现有默认 vLLM 单条件用法保留。

执行步骤：

1. 读取 MODEL_PATH、DATA_DIR、TASK_INSTRUCTION、GOAL_IMAGE、INTERVAL、模式和 steering profile。
2. 检查三视角有效帧范围、FPS/时间对齐；生成与现有逻辑一致的采样点：含首帧和最后帧。PNG 目录允许显式 FPS/时间戳配置。
3. 只抽取一份不可变帧缓存。为每个配置视角、每个被用到的帧请求 SAM3，缓存 bbox；无 bias 条件也共享相同图像与样本。
4. 按 forward/incremental/backward 构造相同 8 图协议。每一组 `(condition, mode)` 建立独立进度状态，禁止共享 incremental tracker。
5. 对同一 HF 模型按确定性 greedy 配置执行 `baseline` 和 `candidate_target`；baseline 使用相同 tokenizer、processor、dtype、像素限制、max_new_tokens，仅不安装 hook。三模式都完整执行，incremental 从头到尾每一 hop 都做干预。
6. 根据完整有效结果计算各模式 progress，再按真实 `after_frame_id` 对齐融合。不能只取最后一 hop 作为 incremental 全轨迹结果，也不能直接平均不同模式的原始 score。
7. 保存逐模式结果、HF 配对对照曲线、融合曲线、bbox overlay 和运行 manifest。

可选的 `candidate_wrong` 和 `random_target` 为效果校准控制组；错误区域要求 token 面积匹配且不与目标重叠，随机 head 使用冻结 seed 且与候选 head 不重叠。无法构造时记为不可用，不伪造对照。

### 6.2 进度语义与异常

| 模式 | progress | hop |
|---|---|---|
| forward | `score` | `progress - previous_progress` |
| incremental 首个有效起始 hop | `score` | `score` |
| incremental 后续，score≥0 | `p_prev + (1-p_prev)*score` | `score` |
| incremental 后续，score<0 | `p_prev + p_prev*score` | `score` |
| backward | `clamp(1+score, 0, 1)` | `progress - previous_progress` |

保留现有 per-mode 数学语义，包括 forward / incremental 原始值不额外强制 clip；在线融合在平均后 clamp 至 `[0,1]`。离线同时记录 `fused_raw` 与 `fused_clamped`，部署对齐使用后者。

HF 生成分数格式错误或截断：记录 invalid、有限重试；同一轨迹 incremental 的中间 hop 最终无效时，后续累计值标为不可用，不把缺失当 0、不跨缺口直接递推。该轨迹仍可记录后续原始 score，待补齐全部缺口后重算累计。融合要求所有配置模式在该点都有有效累计结果，不以剩余模式悄悄改变分母。SAM3 缺失导致的显式 baseline 降级仍是合法评分，但统计中区分 `steering_applied=false`。

### 6.3 输出与比较

建议输出目录为 `results/<run_id>/`，包含：

| 相对路径 | 内容 |
|---|---|
| `manifest.json` | 模型/processor/prompt/profile/SAM3 指纹，生成参数，源视频元数据，采样索引，运行版本 |
| `grounding.jsonl`、`bbox_overlays/` | 每帧候选、选择结果、来源图像哈希和可视化 |
| `baseline/<mode>/sample.json`、`pred_vllm.json` | 保持旧读取接口的预测 schema；manifest 明确实际 engine=hf，文件名仅为兼容 |
| `candidate_target/<mode>/sample.json`、`pred_vllm.json` | steered 预测，额外记录 bias/head/span/hook 诊断 |
| `<condition>/<mode>/reward_vis.mp4` | 复用现有可视化，并展示是否实际施加干预 |
| `progress_curve.png`、`curve.csv` | 三模式和融合曲线，baseline/steered 配对对照，时间轴用真实 after 帧时间戳 |
| `summary.json` | 有效率、grounding 覆盖率、降级率、各模式/融合变化、延迟和峰值显存 |

修复现有 `INTERVAL=20` 却用 `frame_interval=10, fps=30` 画图的问题；首个预测对应第二个采样点，不是 t=0，最后不整除间隔的采样点也用实际帧号。

保留 vLLM 原始输出可作为迁移参考，但不能用“vLLM 随机采样 baseline vs HF greedy steered”的差值估计 steering 收益。需要比较引擎差异时单独输出 `vllm_baseline`。

## 7. 在线 attention steering 流程

### 7.1 每会话运行链路

`backend=grm` 保留为 monitor provider，新增 `inference_engine=hf` 和独立 steering 配置。不另造一份成功失败状态机。

1. 服务启动：验证模型/profile 兼容性，构造 HF 模型和 grounding client。`steering.enabled=true + inference_engine=vllm` 在启动时明确拒绝。
2. `/monitors/start`：验证任务和 target spec，生成内部 session generation UUID，保存 subtask_index、created_at 和冻结后的配置；不在 HTTP 请求中等待模型或取帧。
3. 后台预热：获取 reference start 独立缓存，不与 step 图像共享文件名。启用 BEFORE 干预时同时获取该 reference 图像的 bbox。
4. 每轮：获取三路新 observation，完成去畸变和 PNG 冻结；记录相机 frame ID / timestamp，检查跨相机时间差。若上游只提供 latest 而无法严格同步，要检测并披露，不能宣称原子快照。
5. 对当前所需视角请求 SAM3，响应匹配 session generation、frame ID 和 image hash。多个模式复用该轮检测；BEFORE 干预读取对应 reference/previous 的独立 bbox。
6. `build_online_samples()` 构造各模式 sample，planner 逐 sample 映射 bbox 到图像 token。
7. HF 后端在共享锁内完成安装、生成和 hook 清理。stop 已触发或 session generation 已变更时，不继续启动新的模型调用。
8. 所有模式输出与解析验证成功后，计算临时 tracker/monitor 状态，持锁一次性提交 progress、previous、step 和 latest；任何模式无效，整轮不提交。
9. 发布结果后等待 interval。达到终态停止推理；`status()` 只读已提交快照。

在线推理失败后，下一次成功轮次的 incremental BEFORE 仍为上次**已提交**图像，因此会覆盖中间真实进展。采集 observation 序号独立于成功推理 step，失败重试不覆盖已有图像，也不复用错误 bbox。

重复 observation 不应反复推进成功/失败稳定窗口。优先比较上游 frame ID；缺少 ID 时用内容哈希辅助去重并在诊断中标明。阈值窗口计数针对有效的新采样结果。

### 7.2 API 兼容和诊断

现有 `{"success": true, "data": {...}, "message": "ok"}` 和 `running/success/failed` 保留。start 请求可增加：

```json
{
  "monitor_id": "m-001",
  "execution_id": "exec-1",
  "subtask": "pick the carrot and put it on yellow plate",
  "subtask_index": 0,
  "target_queries": ["carrot"]
}
```

首版不允许单个请求任意切换模型、profile 文件或 kernel；这些由服务配置决定。目标词配置错误返回 400；同 monitor_id 相同 execution/subtask/目标配置的 start 幂等返回已有会话，不创建第二个线程；同 ID 不同内容返回 409。

status 的 `result` 增加：

```text
engine, model_fingerprint, steering_profile_fingerprint
observation: frame IDs, timestamps, image hashes, preprocess hash
grounding: per-camera bbox/score/query/status, sam3 fingerprint
steering: enabled, strategy, applied, degraded, reason, labels
modes.<mode>.steering: heads, delta, target/negative token counts, hook counters
timing: observation_ms, grounding_ms, queue_wait_ms, grm_ms, total_ms
inference_updated_at, result_age_s, error
```

原 `poll_count` 在 GRM 后端实际表示推理 step 数，继续兼容并新增明确的 `inference_step`；不把 HTTP 查询次数伪装为模型刷新次数。`created_at` 在会话整个生命周期固定，`inference_updated_at` 只在成功提交推理时刷新。

SAM3 降级时仍用同模型无 bias 分数更新进度，并明确标识策略变化；这不等于已验证阈值在降级情况下同样可靠，验收需单独统计。生成异常、图像不同步或 bbox/token 错位则只记录 error/staleness，不给状态机喂入假 0 分。网络持续异常延续现有重试语义，先不新增第四种业务状态。

### 7.3 本次功能实施必须同时解决的状态问题

这些问题直接影响帧与 bbox 对齐、hook 隔离和结果可信度，作为接入前置修复，不在本次文档阶段修改：

| 当前问题 | 计划修复 |
|---|---|
| 预热与第一轮都写 `frame_000000.png`，参考图被覆盖 | reference 独立路径、不可变内容；首个 AFTER 使用新 observation 序号 |
| 相同 monitor_id 再次 start 覆盖 dict，但不停止旧线程 | 幂等/冲突检查；内部 generation UUID 隔离缓存和迟到响应 |
| tracker 在 outputs 完整性检查完成前逐项更新 | 所有模式验证后在临时状态计算，再原子提交 |
| `_run_one_step` 修改状态与 `status` 加锁读取范围不完全一致 | 后台本地计算、持锁统一发布不可变快照 |
| stop 后线程可能仍在推理；已结束线程的缓存清理条件也不完整 | 从真实 thread.is_alive 决定清理；worker finally 负责最终清理；迟到结果不发布 |
| 会话状态时间和 subtask_index 未完整保留 | 在会话 state 中持久保存，status 从同一状态读取 |

成功判定仍按现有窗口公式；它只检查当前值达到阈值及窗口极差，不要求窗口内每点都超过阈值。接入不悄悄改变此语义，测试应明确包含 `[0.58, 0.59, 0.60]` 这种边界。

## 8. 拟新增配置及运行方式

以下 YAML 和命令是**实施后的接口约定**，目前不可直接运行。

`configs/steering.yaml` 示例：

```yaml
enabled: true
profile_path: ../assets/steering/grm_profile.json
top_k: 8
bias: 6.0
query_scope: all
negative_scope: target_span
intervention_labels: [after_cam_high]
on_missing_bbox: baseline
max_bbox_age_steps: 0
debug_attention: false
grounding:
  backend: sam3_http
  url: http://127.0.0.1:8878
  timeout_s: 3.0
targets:
  task_queries:
    "pick the carrot and put it on yellow plate": [carrot]
    "pick the white cube and put it on yellow plate": [white cube]
```

top-k=8、bias=6 来自原 incremental 实验配置，仅为校准起点。建议在独立校准数据上比较 top-k `[8,32,64]` 和 bias `[0,2,4,6]`；部署使用冻结 profile 内通过验证的组合。外部 YAML 若覆盖 profile 的已校准参数，应标记 experimental，不能仍显示为原 profile 已验证。

monitor 增加扁平项：

```yaml
backend: grm
inference_engine: hf
steering_config: ./steering.yaml
model_path: /path/to/grm-checkpoint
robot_runtime_url: http://robot-runtime-host:8767
goal_image: ../examples/blank_goal.png
no_backward: true
interval: 1.0
```

HF 配置增加 `attention_implementation=eager`、`dtype=bfloat16`、`max_new_tokens=64`、`do_sample=false`、现有 `min_pixels=12544/max_pixels=76800`。64 是起始生成上限，需记录截断率并验证，与原实验的 12/16-token 上限不同。baseline 和 steered 必须使用相同值。

配置解析必须显式增加 `--inference-engine`、`--steering-config`，将 YAML 嵌套内容转换为结构化配置，未知字段报错。服务字段优先级为 CLI > monitor YAML > 内置默认；steering 字段读取 steering YAML，profile 负责兼容性和已验证参数约束。相对路径相对其所属 YAML 文件解析并保存绝对路径，避免 cwd 改变语义。布尔开关提供成对开/关参数，解决现有 store_true 无法覆盖 YAML true 的问题。

拟运行命令：

```bash
# SAM3 独立环境，服务自身加载 sam3 配置（模型路径、GPU、阈值、队列上限）。
python -m sam3_runtime.service --config configs/sam3.yaml

# GRM 环境，offline_steering.yaml 包含现有离线参数及 steering_config 引用。
python -m examples.offline_steering --config configs/offline_steering.yaml

# 同一 GRM 环境中的在线服务。
python -m monitor_runtime.service --config configs/monitor.yaml
```

新增 SAM3 独立环境说明和依赖文件，不能将 Transformers 5 直接追加到现有 requirements 中。关闭 steering 且使用 vLLM/deterministic 时不需要安装或启动 SAM3。

## 9. 实施阶段与验收

### 阶段 A：共享协议和 HF 后端

交付模块：protocol/media/factory/HF/VLLM adapter；GRMInference 兼容 facade；最小 profile schema。完成后先用无 bias 跑通 delivery 的同一份 8 图 sample，再启用 hook。

验收：旧 vLLM 接口仍可调用；关闭 steering 无 SAM3 导入；HF 自由生成合法 score；8 图/token/grid 严格对齐；真实 checkpoint 的模型层/head 获取正确。不同引擎的随机输出不要求逐字一致，但输入协议和报告必须一致。

### 阶段 B：共享 steering 与 SAM3 服务

交付模块：spans/hooks/ranking loader、SAM3 client/service、目标解析和缓存。使用小样本做真实 HF prefill/decode 验证。

验收：目标 token 对齐可视化正确；选中 head 的 mask 按预期变化；无效 bbox/不支持 kernel/无效 ranking 有明确错误；SAM3 返回原图 xyxy 和图像哈希；零 bias 与同 HF baseline 可复现一致；异常后 hooks 为零。

### 阶段 C：完整离线三模式

交付模块：offline runner、test_data_suc 薄入口、配置、CSV/JSON/曲线/视频。

验收：baseline 和 steered 都完成 forward/incremental/backward 全轨迹；同帧对齐、所有 incremental hops、不同条件独立累计；last-frame 包含且时间轴正确；bbox 缺失/模型输出无效不会被隐去；输出兼容既有结果读取。

### 阶段 D：在线监控接入

交付模块：monitor 工厂接入、快照元数据、grounding 调用、原子更新、幂等与停止清理、状态诊断和 JSONL 运行记录。运行日志保存到每 session 独立目录，可扩展现有 session 视频工具读取。

验收：真实 SAM3 + HF + Robot Runtime 跑通；状态查询不等待 grounding/GRM；不同 monitor 的图像、bbox、hooks、tracker 无串扰；stop 不再发布旧结果；网络与 SAM3 异常有明确记录和约定降级。

### 必须覆盖的测试矩阵

| 范围 | 关键用例 | 通过证据 |
|---|---|---|
| bbox→token | 非方形图、小框、边界、NaN/空框、重复路径不同 occurrence、不同相机、merge/grid 不匹配 | 精确 token 索引断言；错位明确失败 |
| mask hook | 只修改选头/选视觉 keys、query scopes、causal 禁止位不解禁、新文本 key 零 bias | 合成 logits/mask 数值断言，真实 HF hook 计数 |
| lifecycle | 多层 hook 注册中途失败、generate 异常、baseline/steered 交替 | hooks 恢复、后续 baseline 不受影响 |
| ranking | 新旧 JSON 格式、重复/越界/不足 k、4B/8B/微调权重不匹配 | 加载校验和 profile 指纹 |
| SAM3 契约 | 尺寸/哈希匹配、低置信度、无框、多实例、超时、不同 session 迟到响应 | fake service 测试 + 真实 bbox overlay |
| 离线全流程 | 三模式、首末帧、全部 hops、坏输出、融合缺失、真实 FPS | 固定短轨迹的 score/progress/CSV 一致性 |
| 在线状态 | reference 不覆盖、重复帧不计步、缺一个 mode 不部分更新、失败后跨帧 incremental | 可控 observation/model 替身的状态断言 |
| 在线并发 | 同 ID 幂等/冲突、不同任务并发、等待推理锁期间 stop、已终态 cleanup | 无旧结果发布、无缓存覆盖、无 hook 泄漏 |
| 真实闭环 | success/fail 至少各一条，模拟 HTTP observation 重放并另做实时联调 | 逐步 bbox、score、progress、状态与延迟日志 |
| 性能 | 1 和多个 monitor，单/双 GPU 部署 | grounding/排队/GRM/总耗时 p50/p95 与峰值显存，不以 interval 代替实测频率 |

不为文档阶段编写或执行上述模型测试；它们是未来实现的验收要求。

### 效果验收与部署门槛

- 校准与评估按 episode 划分，不能让同一视频的不同帧跨选头/评估集合。固定 checkpoint、目标词、profile 和生成参数后再评估。
- HF baseline 与 steered 使用完整相同 episode 集，统计 grounding 成功子集与全部样本（包括 baseline 降级）两种口径；不得只丢弃难例后宣布提升。
- 报告成功轨迹误判失败、失败轨迹误判成功、终态检测延迟、进度抖动、分数有效率、grounding 覆盖率与降级率。有逐帧进度标签时才报告进度 MAE；有终态标签时报告终态判定指标。
- attention bias 改变评分分布，原 `success_threshold=0.60` 只能作为对照起点；部署前在校准集确定阈值，在独立评估集验证，baseline/steered 同时报告原阈值下结果。
- 在可控 observation 重放中固定同一帧序列、bbox 和配置，在线与离线逐 mode score/progress 应一致（同环境、确定性生成）；重放时可关闭遇终态停止，仅用于测试完整轨迹。真实在线采样时间不同，不能要求与离线帧轨迹逐点相同。
- 不预设未经实测的效果提升比例、显存需求或在线频率。功能正确但效果未通过时保留关闭开关，默认生产配置维持原路径。

## 10. 文档完成与后续实施输入

本次完成的是源码理解和实现方案。源代码阅读、模型 config、现有 ranking JSON 与环境约束已核对；没有运行 GRM/SAM3 推理、安装环境、修改训练或监控代码。

实施开始时需要落定：实际部署 checkpoint（4B/8B/微调）、该 checkpoint 的可用冻结 profile、SAM3 权重与设备、任务目标词表、Robot Runtime 地址及其帧时间戳能力。这些不阻碍制定本方案；方案已经规定缺失资产如何校准、缺失目标如何处理及接口如何演进。

完整实施交付物应包含共享代码、独立 SAM3 服务、三份示例配置、模型/profile 版本说明、离线产物样例、在线状态样例、上述测试与真实联调报告。不能以“只跑通离线”“只算出 bbox”或“请求中出现 steering=true”替代推理时实际施加 bias 的验收。

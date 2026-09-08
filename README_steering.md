# GRM attention steering

重点使用在线功能请阅读 [在线任务进度监控：实现、接口与用法](docs/online_attention_steering.md)，包含部署、完整接口约定、轮询客户端和异常处理。

本分支新增离线轨迹 steering 和在线 monitor steering，保留原有 vLLM / deterministic 路径。运行时独立于原仓库代码，只读取配置指定的模型权重；本地已提供的 `robo-dopamine` 与 `rewardbench-sam3` 环境即可使用，无需重装。

实现以源仓库 `mydata_bench/attention_steering_exp_record.md` **2026-07-31 修正版**及其 GRM 对齐协议为依据：正确的第 6 个 image span 是 `after_cam_high`；三任务 raw-mean-12 ranking 做 normalized Borda，排除第 0、1 层，top-8，`query_scope=all`。没有使用记录中已失效的旧 Qwen adapter head 表。

## 运行

以下命令都从 `Robo-Dopamine-delivery` 根目录执行。示例绑定 GPU 0 跑 GRM、GPU 3 跑 SAM3；可以根据实际空闲 GPU 调整。进程内配置的 `cuda:0` 指向该进程可见的第一张卡。

先在一个终端启动 SAM3：

```bash
conda activate rewardbench-sam3
CUDA_VISIBLE_DEVICES=3 python -m sam3_runtime.service --config configs/sam3.yaml
```

SAM3 使用标准库 HTTP server，不依赖该环境中未安装的 FastAPI/uvicorn。模型启动时常驻加载，`GET http://127.0.0.1:8878/health` 返回 ready 与模型指纹。

在另一个终端运行离线流程：

```bash
conda activate robo-dopamine
CUDA_VISIBLE_DEVICES=0 python test_data_suc.py --config configs/offline_steering.yaml
```

等价入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.offline_steering --config configs/offline_steering.yaml
```

[离线配置](configs/offline_steering.yaml) 中修改三视角视频目录、任务、目标词、goal 图及采样间隔。示例使用仓库附带的 `demo_table`，任务为整理桌面，干预目标为图中的 `red can`。`target_queries` 是同一个目标的查询短语，不能将操作对象和放置目的地混成互相替代的目标。没有目标图时显式使用 `examples/blank_goal.png`。

不传 `--config` 的 `test_data_suc.py` 仍运行原有脚本配置；现在 import 该文件不会启动模型。HF 的旧式调用同样可用：

```python
from examples.inference import GRMInference

model = GRMInference(
    "/path/to/grm-checkpoint",
    engine="hf",
    steering_config="configs/steering.yaml",
    device="cuda:0",
)
# 原有 run_pipeline(...) 参数继续支持；inference_batch(...) 保留输入顺序和原字段。
```

在线服务：

```bash
conda activate robo-dopamine
CUDA_VISIBLE_DEVICES=0 python -m monitor_runtime.service --config configs/monitor_steering.yaml
```

启动前将 [monitor_steering.yaml](configs/monitor_steering.yaml) 中的 `robot_runtime_url` 改为真实 Robot Runtime 的完整 HTTP URL。它必须提供 `/observations/latest/metadata` 和 `binary_endpoints` 指向的三路图像。建议图像响应带 `X-Frame-Id`、`X-Timestamp`（秒）用于去重和同步检查。缺少时间戳时结果会标记 `synchronization_verified=false`，不宣称三路画面严格同步。

```bash
curl -s http://127.0.0.1:8877/health
curl -s -X POST http://127.0.0.1:8877/monitors/start \
  -H 'Content-Type: application/json' \
  -d '{"monitor_id":"m-001","execution_id":"exec-1","subtask":"pick the carrot and put it on yellow plate","target_queries":["carrot"],"subtask_index":0}'
curl -s -X POST http://127.0.0.1:8877/monitors/status \
  -H 'Content-Type: application/json' -d '{"monitor_id":"m-001","execution_id":"exec-1"}'
curl -s -X POST http://127.0.0.1:8877/monitors/stop \
  -H 'Content-Type: application/json' -d '{"monitor_id":"m-001"}'
```

相同 start 内容和 monitor_id 为幂等调用；同 ID 不同内容返回 409。stop 后 status 返回 404，已在 GPU 中执行的调用可能仍需完成，但不会发布旧结果。会话终态停止推理。

## 配置与干预语义

[steering.yaml](configs/steering.yaml) 控制 head profile、top-k、bias、query scope 和相机范围。HF 使用 eager attention；启用 steering 却选择 vLLM 会直接报错，不会静默忽略配置。HF 按单 sample 顺序推理，三个模式各自 tokenize，避免不同图像长度或 bbox 相互污染。

- `intervention_labels: [after_cam_high]` 默认只干预当前主视角。可指定 AFTER/BEFORE 的多个真实相机标签；每个图像独立运行 SAM3 和坐标映射，禁止主视角框套用到腕部相机。参考图不在可配置干预范围内。
- `negative_scope: target_span` 对目标 tokens 加 `+bias`，同一指定图像 span 的其余视觉 tokens 加 `-bias`。还支持 `all_visual/other_spans/none`。
- `query_scope: all` 覆盖 prefill 所有 query 行及每次 cached decode；也支持 `prefill/last_prompt/decode`。文本 key、未选中的 heads 和原 causal 禁止位不改变。
- SAM3 与 GRM 使用同一冻结画面；若配置鱼眼去畸变，检测在去畸变之后运行。bbox 为该图原始像素 `xyxy`，按 processor 实际 grid 和 merge size 映射到相交网格单元。
- `on_missing_bbox: baseline` 在无目标、候选实例歧义、SAM3 超时时，对整个 sample 降级为同一 HF 模型无 bias 推理；记录 `degraded/reason/applied=false`。可改为 `error`。不使用上一帧 bbox 冒充当前检测。
- 图像哈希/尺寸/请求 ID 不匹配、非法 head、错误 token span 属于错误；不作为正常 bbox 缺失吞掉。HF 输出格式无效时不转成 0 分。
- 任务词优先级为 start 显式目标词 > `task_queries` 映射 > 简单英语操作指令解析。中文或复杂任务请显式配置。多实例选择目前是逐帧置信度和歧义检查，不承诺长期身份跟踪。

HF 与 SAM3 分别独占自己的模型推理锁；两个 GRM 会话之间锁覆盖 hook 安装、完整生成和 finally 清理。baseline 也使用同一把 HF 锁。服务 `status()` 只读取已提交结果，不等待 SAM3/GRM。

YAML 中路径相对所属配置文件解析；CLI 显式路径相对当前目录。monitor CLI 覆盖 YAML；`--backward` 可覆盖 `no_backward: true`。SAM3 超时、设备、阈值等在对应配置中显式设置。

## 四个离线对照条件

| 条件 | heads | bias 的目标区域 |
|---|---|---|
| baseline | 不安装 hook | 无 |
| candidate_target | 冻结 top-k | SAM3 目标 bbox |
| candidate_wrong | 同一 top-k | 同视觉平面的等尺寸最远区域；不可用时使用等数量、不重叠的其他视觉 tokens，并记录 fallback |
| low_rank_target | 排名末端、与 top-k 不重叠 | 同一 SAM3 目标 bbox |

三种模式均完整执行；每个条件、每种模式有独立的累计状态。incremental 对每一 hop 干预，不以最后一次推理替代整条轨迹。中间 hop 无效时，后续累计值为缺失，不跨缺口继续累加。

baseline 和 steered 使用相同 HF checkpoint、prompt、processor、greedy 配置和图像，不把 vLLM 与 HF 的差异算作 steering 效果。示例默认 `max_new_tokens=64`；这是输出完整性上限，与原实验部分脚本的 12/16 不同，已在 manifest 记录。

## Head profile

[assets/steering](assets/steering) 随代码提供 3 份完整 source ranking、1 份 GRM 共识表及 profile。共识含 1,088 个 eligible heads；top-8 为：

```text
(19,16), (19,23), (19,10), (20,4), (19,0), (18,30), (20,13), (22,15)
```

这些是 **GRM checkpoint 自身**的 heads，不是实验记录表中 Qwen/RoboReward 的 heads。测试从三源文件重新计算了整张 Borda 排序并核对一致。

profile 校验 checkpoint 路径、config SHA-256、层数和 query-head 数；ranking、prompt、processor、参数也写入运行指纹。checkpoint 校验不是全部权重字节的密码学认证，部署时仍应使用固定的权重版本。

当前 profile 针对本机的 GRM-2.0-8B-Preview。4B 和微调权重需要匹配的新 ranking/profile，不能只因为层/head 数相同就沿用；如显式使用 `allow_model_transfer: true`，属于未验证的迁移实验。该 profile 的 forward 来源也不能当作 incremental/backward 效果已校准的证据。

有三条对应 checkpoint 的独立成功轨迹 ranking 时，可生成新的共识/profile：

```bash
python -m grm_runtime.ranking \
  --sources assets/steering/carrot_source_ranking.json \
            assets/steering/bottle_source_ranking.json \
            assets/steering/cube_source_ranking.json \
  --model-path /path/to/the-matching-grm-checkpoint \
  --output-dir results/new_head_profile
```

此命令只聚合已有 source ranking，不会为新 checkpoint 自动重跑 attention 扫描；不要把旧模型 source ranking 重新贴上新模型路径当作重新选头。真实新模型选头需要该模型的逐任务 12 个 progressive sample、last-prompt raw bbox attention mass 算术均值结果。

## 输出和监控状态

离线运行在 `results/offline_steering/<run_id>/` 保存：

- `manifest.json`：配置、模型/processor/prompt/ranking 指纹、真实采样帧和 FPS。
- `<condition>/<mode>/sample.json`、`predictions.jsonl`、`pred_vllm.json`、可选 `reward_vis.mp4`。`pred_vllm.json` 文件名仅用于旧工具兼容，内容中的 `engine=hf` 表示真实后端。
- `grounding.jsonl`、`grounding_cache/`、`bbox_overlays/`：检测证据与可视化。
- `curve.csv`、`progress_curve.png`、`summary.json`：模式和融合进度。曲线使用真实 AFTER 帧时间戳，包含最后帧。

`summary.complete` 表示评分输出完整；`formal_scoring_ready` 还要求无降级且所有请求的 steering 都实际施加。即使评分完整，全量降级 baseline 的运行也不会标成有效 steering 对照。

在线 `result` 增加 `observation`、`modes.<mode>.steering`、`timing`、`inference_updated_at`、`result_age_s`、`session_dir`。原 `poll_count` 继续代表已提交的推理轮数。会话目录保存 `manifest.json`、`online_pred.jsonl`、reference 和成功提交的快照，供复现；**stop 后这些运行产物保留**，不在仍有工作线程时删除。长期运行需按部署的保留策略管理结果目录。

在线同一轮所有模式校验后一次提交；失败轮次不部分推进 tracker，不更新 previous。重复 observation 不计入稳定窗口。下一成功轮次的 incremental BEFORE 使用上次已提交的画面。成功/失败窗口公式保持原语义，原阈值 0.60 尚未因 steering 重新校准。

## 验证

单元/契约测试不加载真实模型：

```bash
conda activate robo-dopamine
python -m unittest discover -s tests -v
```

真实 SAM3 服务启动后，可执行双会话 HTTP observation 重放验证：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python tests/validate_real_steering.py
```

覆盖模型真实 hook、并发会话、reference 不覆盖、在线/离线相同画面的分数一致、steering 后 baseline 不变和 stop 清理。它使用可控 JPEG HTTP 图像源与真实 SAM3/GRM，**不是物理机器人任务成功率实验**。

本次验证结果与边界见 [验证报告](docs/attention_steering_validation.md)。目前已验证本地 8B checkpoint 的机制和组合流程；4B、微调 checkpoint、复杂遮挡和真实部署阈值仍需对应任务数据验证。

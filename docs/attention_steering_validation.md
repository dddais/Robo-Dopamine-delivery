# Attention steering 实现验证

日期：2026-09-08。开发分支：`dev_x1pro`。开发和验证产物均位于 `Robo-Dopamine-delivery`；原仓库仅作为源码、模型和 ranking 的只读来源。使用既有 `robo-dopamine`、`rewardbench-sam3` 环境，没有安装依赖或改动其他分支。

实现入口和部署命令见 [README_steering.md](../README_steering.md)。实现采用实验记录 2026-07-31 修正后的目标 span 绑定和 GRM 自身排名，未使用被记录判为无效的旧 adapter head 表。

## 验证结果

| 检查 | 实际证据 | 结果 |
|---|---|---|
| 单元与契约测试 | `python -m unittest discover -s tests -v`，26 个测试 | 全部通过 |
| 三源排名复现 | 随代码提供的 carrot/bottle/cube 完整 ranking 重新计算 normalized Borda | 1,088 个 eligible heads 的完整顺序与冻结 GRM 表相同 |
| 真实离线三模式四条件 | 真实 GRM-2.0-8B-Preview、真实 SAM3、demo_table、首帧/中间帧/末帧 | 24/24 合法预测，18/18 干预实际生效，0 bbox 降级 |
| hook 生效范围 | 每条干预记录逐层读取 diagnostics | prefill 和 decode 均实际施加；baseline 未安装 hook |
| 输出完整性 | 最终离线 `summary.json` | `complete=true`、`formal_scoring_ready=true` |
| 可视化 | 相同配置的启用可视化运行 | 12 个模式/条件视频、bbox overlays、模式与融合曲线 |
| 真实在线组合 | 2 个 monitor × 2 轮 × forward/incremental，真实 SAM3/GRM，HTTP JPEG observation 回放 | 8 次在线模式评分，全部实际施加 bias，无会话污染 |
| 在线/离线同图复现 | 在线第一轮保存的完整 8 图重新推理 | 分数文本完全一致 |
| baseline 隔离 | baseline → steering → baseline 顺序生成 | 两次 baseline 分数文本相同，全部 hooks 已移除 |
| 查询响应 | GPU 正在推理时持续调用 status | 最终回放最大响应耗时约 3.12 ms |
| GRM 显存 | 最终在线脚本 `torch.cuda.max_memory_allocated()` | 17.15 GiB（仅 GRM 进程，不含另一张卡上的 SAM3） |
| 原 vLLM 接口 | 原 `GRMInference(model_path).inference_batch(...)` 真实加载和生成 | 生成合法分数 `0.415`，进程退出码 0 |
| 静态检查 | Python AST / 编译、`git diff --check`、CLI help | 通过 |

离线验证采样间隔为 1,127 帧，实际采样 `[0,1127,2253]`，三种模式均执行两个 AFTER 时间点，incremental 覆盖两个相邻 hops。它覆盖完整视频的首末范围，但不是默认 20 帧间隔的全密度性能实验。

第一次诊断选用场景中不存在的 bottle，24 次评分均有效、18 次请求 steering 均显式降级 baseline。随后根据实际画面选择 `red can`，最终运行无降级。前者是缺失目标路径的真实验证，不能作为成功干预证据。

vLLM 的单次兼容验证在正常生成并写出结果后，退出阶段出现上游 engine-core shutdown 日志；这里仅据此确认原接口的加载和生成可用，不将其解释为长期服务稳定性验证。

## 可复查产物

- [最终离线 manifest](../results/validation/offline_final/26-09-08-13-32-21_96a4b4bf/manifest.json)
- [最终离线汇总](../results/validation/offline_final/26-09-08-13-32-21_96a4b4bf/summary.json)
- [最终离线曲线](../results/validation/offline_final/26-09-08-13-32-21_96a4b4bf/progress_curve.png)
- [目标干预的 incremental 逐步结果](../results/validation/offline_final/26-09-08-13-32-21_96a4b4bf/candidate_target/incremental/pred_vllm.json)
- [含 12 份视频的运行目录](../results/validation/steered_offline/26-09-08-13-14-56_6ad32070)
- [最终在线组合报告](../results/validation/online_final/report.json)
- [单元测试日志](../results/validation/unit_tests.log)
- [原 vLLM 预测](../results/validation/vllm_result.json)

这些模型运行产物保留在当前 workspace 的 `results/`，按既有 `.gitignore` 不纳入源码版本控制。重现脚本为 [validate_real_steering.py](../tests/validate_real_steering.py)；单元/契约用例为 [test_steering.py](../tests/test_steering.py)。

## 覆盖的故障与边界

- 第 6 个 image span 的 AFTER 绑定，重复文件路径的多个独立 occurrence，真实相机各自的 bbox 和图像尺寸。
- 小目标网格相交、非方形图、越界/空/NaN bbox、grid 长度不符、SAM3 响应图像哈希不一致。
- 指定 heads 和 key 列的正负 bias、causal 禁止位、新生成文本 key 零 bias、四种 query scope、零 bias 数值一致。
- 部分 hook 安装失败和生成异常时清理、模型锁等待期间 stop、GPU 生成过程中 stop 后不发布旧结果。
- 一轮部分模式输出无效时 tracker 不部分更新；离线 incremental 中间缺失时后续累计不可用；重复 observation 不增加稳定窗口。
- 重复 start 幂等、冲突 409、未知 monitor 404、created_at 保持稳定、deterministic 后端不加载模型。
- 原成功窗口边界语义保留，成功/失败终态可由有效新观察驱动。

## 实测边界

已验证的是 8B 的干预机制、四条件输出和在线组合实现。在线图像来自可控的 Robot Runtime HTTP 协议回放，监控 API 通过 FastAPI 的测试客户端调用；未连接物理机器人，也未测真实部署网络的端到端延迟。

4B、微调 checkpoint、新任务 head profile、鱼眼配置及 success threshold 需要各自验证。逐帧 SAM3 不等于持久化实例 tracker；目标歧义和遮挡可能导致显式降级。成功判定阈值沿用旧值，本次没有据少量 smoke 数据重新校准，也不声称任务成功率获得提升。

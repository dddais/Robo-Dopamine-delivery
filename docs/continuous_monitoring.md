# 持续评分与手动停止

manual_bridge 的 `robot.auto_stop: false` 现在表示持续执行和监控，直到操作员在 Runtime UI
选择空闲 / I。Runtime 自动在 `POST /monitors/start` 中设置：

```json
{
  "monitor_id": "mon-...",
  "execution_id": "exec-...",
  "subtask": "pick cup",
  "target_queries": ["cup"],
  "defer_inference": true,
  "continuous_monitoring": true
}
```

该字段只接受布尔值，省略时为 false。模式在创建任务时固定，同一 monitor ID 换模式会返回 409。
HTTP 启动 / 状态结果的 `result.continuous_monitoring` 确认实际模式。Runtime 会拒绝未确认支持的
旧 Monitor，避免表面保持 running、后台却停止推理的情况。无需更改 Monitor YAML。

持续模式将会话状态和评分判定分开：

- 顶层 `status` 保持 `running`；`result.status` 仍显示当次 `running / success / failed` 判定。
- 成功、长期低进度失败、双分支差异超阈值均不停止 worker。每次重新判定，保留累计进度和
  滑动窗口，判定可以随新评分改变；新任务才重置 reference 和进度。
- 跟踪线程、GRM 单 / 双分支推理、评分预览和 JSONL journal 持续更新。
  `records.complete` 在任务仍运行时为 false，显式 stop 后为 true。
- `POST /monitors/stop` 结束推理及跟踪。停止后的迟到结果不能再发布。
- 模型 / 取图异常仍进入结果的 error 字段，保留重试与上游错误处理；持续模式只改变正常评分的终态语义。

`continuous_monitoring: false` 保留原行为：首个成功 / 失败终态锁存并结束推理。
GRM 的 `latency_s`、原图和 bbox 时序不变。两种模式都将每轮判定与运行模式写入
`online_pred.jsonl`，完整历史用于离线分析；持续模式的内存判定窗口保留有限长度。

部署：更新服务器本仓库并重启 Monitor，同时更新 Runtime / loop 的 dualsystem-agentic，
重启相应进程并开始新任务。无需更新 SAM3、robot-bridge 或 VLA Policy Server。

模拟模型验证：

```bash
PYTHONPATH=. python -m pytest tests/test_continuous_monitoring.py tests/test_monitor_records.py tests/test_dual_branch.py
```

测试覆盖连续成功后继续评分、差异失败后继续评分、判定变化、默认终态锁存、延迟激活、
手动停止、逐轮预览和 journal 完整性。未使用真实模型权重或机器人。

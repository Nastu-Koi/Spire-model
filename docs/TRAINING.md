# 训练

运行步骤统一维护在 [README](../README.md)：安装环境和引擎 → Steam 采集 → Bootstrap → PPO → Steam 控制。

训练输出统一使用 `current/`、`history.jsonl` 和 `status.json`。Bootstrap 按 epoch 记录，PPO 按完整采样更新轮次记录；内部各 PPO epoch 的指标保存在该轮的 `metrics` 列表中。续训读取 `current/`，并向历史文件追加记录。

PPO 的 `round-N/` 保存原始轨迹、逐步状态和实际 seed。`history.jsonl` 包含配置、策略版本、loss、entropy、KL、clip fraction、value MSE、梯度范数、累计更新次数、耗时及各角色胜率。有独立 Bootstrap 验证集时也会记录其 NLL。

训练网页通过 `python -m model monitor --root runs` 启动。它只读取这些文件，不参与训练。

模型结构与损失函数见 [ARCHITECTURE](../ARCHITECTURE.md)。

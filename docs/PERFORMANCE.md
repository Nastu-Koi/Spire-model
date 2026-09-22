# 性能与正确性验证

本版本先提供小型 Full Attention/SwiGLU + Muon/AdamW 路径。目标是 RTX PRO 6000 Blackwell；本地 CPU 测试不能代替该 GPU 的兼容性或加速验证。

本地验证记录：Python 3.11 / PyTorch 2.14 CPU，`tests` 与 `steam_recorder/tests` 合计 53 passed、5 skipped（1 个 CUDA 检查、4 个游戏程序集检查）。合成 Bootstrap→PPO→检查点重载通过；双优化器恢复后的下一次更新与连续训练一致。历史轨迹在约 503 万参数小模型上的 BF16 批量前后向与优化器更新已跑通。Flex 地图偏置反向图通过 PyTorch 图校验，但尚未在目标 GPU 编译或运行，不报告加速倍数及游戏胜率。

## 环境

使用 Python 3.11 和项目声明的 PyTorch 2.14 系列。目标机器需要安装与驱动兼容、包含该 GPU 架构的 CUDA wheel；CPU wheel 无法运行 CUDA 基准。运行报告记录 PyTorch、CUDA 构建、设备和计算能力，实际安装组合应随实验固定。

```bash
python -m pip install -r requirment.txt
python -m pip install -e .
python -m pytest tests steam_recorder/tests
python -m pytest -m cuda tests
```

CUDA 测试必须在目标机器运行；CPU 环境将跳过这些测试。新检查点格式为 2，不能继续加载旧 hybrid 权重；原始采集数据仍可导入。

## 快速反馈

```bash
python -m model --device cpu smoke --output /tmp/spire-v2-smoke
python -m model --device cpu benchmark --config configs/tiny.json --steps 3 --warmup 1
python -m model parameters --config configs/rtxpro6000.json
```

smoke 是合成协议引擎，不是游戏模拟器。重复运行 smoke 请换输出目录，以免覆盖已有检查点。

## 真实输入与 GPU 基准

```bash
python -m model data-stats --data /path/to/trajectories --max-runs 32
python -m model --device cuda benchmark --config configs/rtxpro6000.json --data /path/to/trajectories --batch-size 1 --output reports/batch1.json
python -m model --device cuda benchmark --config configs/rtxpro6000.json --data /path/to/trajectories --batch-size 4 --trace reports/batch4.trace.json --output reports/batch4.json
```

`data-stats` 输出 token、动作、效果节点、最长程序、效果边和地图节点的分位数，按角色与阶段分组。轨迹目录扫描排除 journal 和 metadata；单个 JSONL 可用于导入后的接受集。`--max-runs` 限制读取局数，结果是确定性抽样，不能当作全数据分布。

benchmark 默认不更新参数、不启动游戏，不把历史轨迹冒充新策略 on-policy 数据。它使用固定 batch 的合成 policy/value 目标测前向与反向，报告首步（包含可能的编译）、预热后 macro/s、步延迟、峰值显存和实际 attention backend。Profiler trace 另跑一次，不计入稳定耗时。默认不包含 optimizer step；加 `--optimizer-steps` 会在临时模型上执行梯度裁剪及优化器更新，不保存权重。两种模式都不包含引擎时间，端到端训练耗时仍以真实训练记录为准。

对比配置时固定数据、随机 seed、精度和测试步骤。编译实验可复制配置并切换 `compile_blocks`，检查 `TORCH_LOGS=recompiles,graph_breaks` 日志。不要同时改变模型规模和 batch 后将收益归因于某个内核。配置中的容量桶用于实际 batch padding；过大 padding 可能抵消编译收益。

## 学习与独立评估

```bash
python -m model --device cuda bootstrap --data data/steam-001/accepted.jsonl --config configs/rtxpro6000.json --epochs 1 --output runs/v2-bootstrap
python -m model --device cuda train --checkpoint runs/v2-bootstrap/current --workers 4 --rounds 1 --value-warmup --output runs/v2-ppo
python -m model --device cuda evaluate --checkpoint runs/v2-ppo/current --seeds configs/evaluation-seeds.json --workers 4 --output runs/v2-evaluation
```

真实训练与评估需要先安装游戏程序集并构建 `sts2-cli`，见 README。`workers=1` 保留单引擎调试路径；并发采样顺序会影响随机采样，不承诺与串行逐动作相同。

`configs/evaluation-seeds.json` 是候选比较队列，`configs/final-test-seeds.json` 是独立最终队列，每组每角色 20 局。两者仅作为起始规模，区间过宽时应扩充独立评估；最终队列不要用于反复选型。加载检查点时检查已记录的训练 seed 交集；仍需保证外部预处理没有引入数据泄漏。Bootstrap 验证集使用 `--validation` 指定，不自动消费训练数据做验证。

## 尚未启用的实验

Liger、Transformers 骨干、TorchRL 和 FP8 不因出现在设计候选中就强制增加依赖。当前优先复用 PyTorch 标准层、FlexAttention、Muon、fused AdamW 和 Accelerate；领域特有的效果绑定、地图 score modifier 和多选解码由本项目实现。只有目标 GPU 的对照结果显示真实收益时才替换内核。

FP8 实验需单独检查硬件/形状支持、梯度、检查点恢复及 rollout/replay 一致性。手写 Triton 只针对批量化和编译后仍占主要时间的热点。当前没有声称已完成这些 GPU 实验。

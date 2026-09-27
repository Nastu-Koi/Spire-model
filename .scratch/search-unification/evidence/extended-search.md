# 放宽 seed 预算后的通关验证 — 2026-09-27

结论：**两个 seed 都已完整通关，并通过全新原生进程独立重放。** 这证明此前 300 秒零成功不足以否定当前路线的可行性；后续优化可以建立在真实成功路径上。

## 方法

按用户建议，先放宽 seed 级时间限制。选择此前短测试没有未解决错误的 seed 01／02，从各自 300 秒测试的 tree.json 恢复，每 seed 最多追加 1800 秒，两个独立 seed 并行。其他参数完全沿用原 MCTS＋CombatSolver 配置，未改变战斗预算、策略、原生引擎或验收门槛。两局均在追加预算耗尽前成功。

[执行脚本](../experiments/extended_search.py)、[运行计划及源码／依赖哈希](extended-plan.json)、[原始结果](extended-report.json)。源码和依赖未漂移。

## 结果

| seed | 先前秒数 | 本轮追加秒数（含验收） | seed 累计秒数 | 独立验收 | 训练步骤 |
| --- | ---: | ---: | ---: | --- | ---: |
| UNIFICATION-HOLDOUT-01 | 300.547 | 682.360 | 982.906（16.38 分钟） | 通过 | 510 |
| UNIFICATION-HOLDOUT-02 | 301.087 | 690.068 | 991.155（16.52 分钟） | 通过 | 461 |

本轮两个 seed 并行的总墙钟为 690.069 秒，不能把两局耗时相加当成批次墙钟。完整计入前一轮三个 seed 的 903.765 秒，本次采集链累计 1593.834 秒取得两条成功轨迹，约 796.917 秒／条；这是事后探索性成本记录，不是冻结性能验收。单独的 profile 诊断时间不在此采集链内。

seed 01 本轮搜索有 15 次分支战败、0 未解决错误；seed 02 有 14 次分支战败、2 次 ColorfulPhilosophers 错误，最终选择的成功路线未使用失败分支。错误覆盖问题仍存在，成功不能代替修复。

## 独立验收与产物

两条 accepted.jsonl 均重新读取并经过 verified_run 和 model.data.validate_run 检查；provenance 包含 fresh_native_seed_replay、三幕 Boss [1,2,3]、A0_final_boss_victory，原始重放 SHA-256 相符。证据和文件哈希见 [验收摘要](extended-verification.json)。

- [seed 01 的训练轨迹](../../../combat_solver_cli/artifacts/unification/extended-seeds/seed-01/accepted.jsonl)
- [seed 02 的训练轨迹](../../../combat_solver_cli/artifacts/unification/extended-seeds/seed-02/accepted.jsonl)

产物保存在 ignored artifacts 目录，路径在本机有效。对应目录同时保留 winning_prefix.json、verified_trace.jsonl、tree.json 和搜索日志。学生数据仅含最终路线的每步公开观察、完整合法动作及所选标签。数据保持 recorder_bc／unverified／bc_only；顶层 partial／victory=null 是既有监督数据契约，完整胜利证据在 provenance，不能用于 PPO on-policy。

## 下一阶段

先把这两条成功路径作为回归样本，核对战斗耗时、最终 Boss 前的牌组和资源，再开展逐项优化。优先解决已发现的事件执行缺口；时间优化必须继续通过完整路径独立重放，不能用局部吞吐代替成功产出。

这是选择过 seed、延长已知树后的 2/2 探索性成功，不是总体胜率，也不证明五分钟目标达标。后续性能验收应另用未见种子；原短预算结果保留在 [performance.md](performance.md)，不被覆盖。

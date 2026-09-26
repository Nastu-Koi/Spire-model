# 早期路线多样性与每幕 Boss 预算

本次在已有固定 seed 搜索上增加两项默认行为。目标仍是尽快找到首条完整可验证胜利，未引入最优性保证。

## 调度与分片

`combat_solver_cli/diversity.py` 负责纯调度，不访问原生引擎。第一幕前 5 层的前三个非强制关键动作构成路线的三个层级：事件选项、走图、抓牌、跳过奖励或奖励替代选择。动作语义决定稳定标识，不使用进程临时候选引用，不把相似公开观察当成同一个隐藏状态。

每次开始一段新的连续试探，先比较开局层尝试次数，再比较第二、第三层次数，最后以原 A* 的 f 和 tie 排序。一次连续试探内，每层标识只计一次；未完成的标识随着后续动作扩展。失败回跳仍约束可选历史范围，保留所有未展开候选。启用 rollout 时保持同一 worker 的连续执行；rollout=0 时，每次展开就是新的调度机会，可能带来更多恢复开销。

多线启动先将不同开局分开；若开局数小于 worker 数，则按第二、第三个关键决策继续拆分；仍不足时拆分同路线下的不重叠候选。按组大小分配到较空的线。不是动态任务窃取，也没有实时跨线失败共享。

`route_visits` 保存于 frontier.search_state，合并多线时仅累加相对输入检查点的新尝试次数。旧检查点没有该字段时从空计数开始；不会丢弃旧候选。路线前缀缓存使用弱引用，不额外保留已经释放的历史。

配置：`search` / `batch` 默认启用，`--no-early-route-diversity` 用于关闭对照；Python search 同名参数为 `early_route_diversity`。单线算法标记为 `weighted_astar_early_routes_v3`。该设计促进多样性，不保证每个 seed 都更快。

## Boss 预算

搜索默认 `boss_budget_ms=5000`，命令行可用 `--boss-budget-ms` 调整。普通战斗继续使用 `--budget-ms`。直接调用 SolverEngine 时，只有显式传入 boss_budget_ms 才启用覆盖，保持其他底层调用的兼容性。

原生适配器根据活动战斗的 `RunState.CurrentRoom.RoomType == RoomType.Boss` 判断，适用于三个章节，不按楼层猜测，也不把精英算作 Boss。生效预算进入 SolverSearchProfile、软截止与硬取消设置；回合计划策略键包含 boss_budget_ms，改变预算配置时不复用旧策略的缓存。worker I/O 上限考虑普通与 Boss 预算的较大者。

**5 秒是每次原生搜索的软预算，不是整场 Boss 的总时限。** 完成搜索可以提前返回；沿用原有软截止后的最多 10 秒硬取消余量，通信另计。solver_result 返回实际 budget_ms 与 boss_combat，便于核验。

## 验证

新增检查覆盖不同开局优先、同开局不同早期地图优先、一次 rollout 不重复计数、pending 与历史标识一致、晚期选择不改变早期标识、分片候选守恒、单路线填满 worker、frontier 恢复及多线增量合并。既有回归也继续运行。

真实原生验证使用此前已验收的同 seed `7E4A91CDAE1225F0`、Ironclad A0 的 545 步胜利前缀，分别在普通战斗和每幕 Boss 插入真实只读求解，然后继续原动作序列直到原生胜利：

| 原生位置 | 类型 | 实际收到的预算 | 本次调用耗时 |
|---|---|---:|---:|
| 第一幕 2 层 | 普通战斗 | 1000 ms | 0.965 秒 |
| 第一幕 17 层 | Boss | 5000 ms | 5.031 秒 |
| 第二幕 16 层 | Boss | 5000 ms | 3.741 秒 |
| 第三幕 15 层 | Boss | 5000 ms | 5.022 秒 |

整个检查约 19.17 秒，通过三个 Boss 类型判断和预算断言，并完整重放到最终胜利。证据为 `combat_solver_cli/artifacts/boss-budget-5000-check.json`。这证明预算覆盖和状态隔离正确，不是重新搜索出完整胜利的计时结果。

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m unittest discover \
  -s combat_solver_cli/benchmarks -p 'test_*.py' -v
/home/nastukoi/miniconda3/envs/sts2/bin/python -m combat_solver_cli.benchmarks.boss_budget \
  --prefix combat_solver_cli/artifacts/perf-completion-4lanes/winning_prefix.json \
  --output combat_solver_cli/artifacts/boss-budget-repeat.json
```

此前 424.68 秒的从头通关属于旧调度、Boss 1000 ms 配置，不代表本次修改后的性能。需固定 seed、CPU 配额和时限重复对照，才能判断首次验收通关时间是否缩短。

同 seed、普通预算 1000 ms、Boss 5000 ms 的四线 120 秒原生检查展开 963 个节点、未解决分支 0，最高第三幕 12 层；实际总耗时 122.44 秒（动作边界停止）。初始队列包含 3 种开局，第四条线分担其中一种开局的其他候选。保存 9,917 个候选和 15 个分层路线计数。随后通过正式 CLI 从合并 frontier 恢复，再展开 12 个节点，未解决分支 0；预算与新调度标志均正确继承。对应目录为 `early-routes-boss5000-smoke` 与 `early-routes-boss5000-resume`。本轮短跑没有重新获得通关，不构成加速结论。

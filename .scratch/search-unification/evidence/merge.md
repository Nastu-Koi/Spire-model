# 唯一搜索主线迁移证据 — 2026-09-26

## 实现结果

生产入口统一为 MCTS＋CombatSolver，A* 与 public_* 独立规则模拟已删除；共用重放、公开状态摘要和动作语义迁至 search_support.py。保留回合计划复用、Boss 预算、选择边界、风险模拟和合法候选。已有进度输出迁入批次／定量生成，迁移前的本地备份补丁保留在 ignored `combat_solver_cli/artifacts/local-backups/`，不随源码提交。

迁入固定 dev/holdout 评测、源码及 DLL 哈希、实际墙钟和零成功 null 成本。失败独立重放保留现场。BC 导出继续使用 recorder_bc / unverified / bc_only，单元正反例验证三幕 Boss、胜利及公开重放一致性；未改变原生 C# 或游戏。

## 验证

- 官方 CombatSolver 0.44.0、RitsuLib 0.6.2 下载及哈希见 [依赖来源](dependency-provenance.md)。本机构建成功，0 错误、10 条既有警告。
- `/tmp/spire-test-env/bin/python -m pytest combat_solver_cli/tests -q`：68 passed，3 subtests passed。
- `ruff check combat_solver_cli --exclude benchmarks` 通过；全部本次修改的 Python 文件格式检查通过；git diff --check 通过。全目录格式检查发现未修改的 test_dotnet_runtime.py 有既有格式差异，未做无关重排。
- 独立审查发现并修复小预算分片饥饿、开局 warmup 中断缺少顶层 checkpoint。复验 16 passed、3 subtests passed，无残留必须修复项。target 的 stopped/interrupted 不再立即续跑，截止时间停止记为 budget_exhausted。
- 原生首战 smoke：Ironclad A0，21 solver steps，战斗 victory；只是首战通过，不是完整通关。
- 原生单树，seed BD1D31C2468C7C1F、rollout_decisions=8、reuse_turn_plan：20 秒预算实际 21.296 秒，165 节点，到第一幕第 4 层；[摘要](native-initial.json)。
- 同树恢复 15 秒预算实际 16.742 秒，节点增长至 343，无基础设施或重放错误；[摘要](native-resume.json)。
- 两 lane 原生搜索 10 秒预算实际 11.183 秒，35 次展开、4 次模拟，两 lane 都运行并保留恢复入口；[摘要](native-lanes.json)。
- [冻结开发 smoke manifest](dev-manifest.json) 在执行前写入。一次 dev case、10 秒预算，实际评测墙钟 10.580 秒、budget_exhausted、0 成功、成本 null；[完整报告](dev-evaluation-report.json)。源码哈希运行中未漂移。该报告对应其记录的源码版本；随后还完成并行恢复修复及纯格式整理，不能将它当作最终版本性能对照。holdout 未执行、未调参。

预算为协作式软停止，实际超时已计入报告。原生运行产物及真实树保存在 ignored combat_solver_cli/artifacts/unification/；上述轻量摘要进入工作区证据。

## 限制与后续

本次证明入口统一、停止恢复、原生求解连接和数据验收门槛的回归。没有本次真实整局成功或成功导出，不能宣称稳定通关或平均五分钟产出。后续按[规格](../spec.md)冻结性能实验、保留集和停止规则，至少取得 3 条完整独立原生重放成功后评估成本。测试夹具产物不进入训练。

旧 A*、旧 lanes-v1 与旧公开评测清单不再兼容；明确拒绝而非静默迁移。历史失败数据及原任务文档保留，不把路线退役记作性能达标。

后续已完成冻结性能测试及单独耗时分析，见[性能报告](performance.md)；本节先前的短预算回归不替代该正式测试。

提交版证据中的仓库内绝对路径已规范化为相对路径，测量值与哈希保持不变；原始文件保存在本机 ignored `combat_solver_cli/artifacts/local-backups/`。

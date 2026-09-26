# 公开信息搜索修复工单索引

[完整规格](spec.md)维护目标、信息边界与最终验收；下表仅提供实施入口。每张工单独立维护验收清单、`Status` 与 `Progress`，此处不复制执行状态。

## 工单与依赖

| 工单 | 前置工单 |
| --- | --- |
| [01: 固定种子对照与失败回放](issues/01-reproducible-evaluation.md) | 无 |
| [02: 修复 Second Wind 并统一效果转移边界](issues/02-second-wind-transition.md) | 01 |
| [03: 用公开历史修复 Spite 条件连击](issues/03-spite-public-history.md) | 02 |
| [04: 校正现有卡牌与常见状态估值](issues/04-effect-coverage.md) | 02、03 |
| [05: 让常见药水参与真实收益比较](issues/05-potion-evaluation.md) | 04 |
| [06: 搜索补能和费用变化后的新组合](issues/06-dynamic-legality.md) | 04 |
| [07: 从无序牌堆采样下一回合](issues/07-public-draw-lookahead.md) | 06 |
| [08: 验证成长与资源保留的跨回合价值](issues/08-future-value.md) | 07 |
| [09: 按牌组边际收益选择奖励](issues/09-deck-rewards.md) | 01 |
| [10: 让路线与补给决策兼顾成长和生存](issues/10-route-resources.md) | 01 |
| [11: 根据决策难度分配搜索预算](issues/11-adaptive-budget.md) | 05、08、09、10 |
| [12: 选择并行度并验证安全缓存收益](issues/12-throughput.md) | 11 |
| [13: 在保留种子上验收成功轨迹成本](issues/13-acceptance.md) | 12 |

## 执行约定

- 仅开始前置工单已完成的任务。01 建立评测基础后，02、09、10 可分别推进；04 完成后，05 与 06 可分别推进。
- 工单完成须有对应验收证据；`ready-for-agent` 表示规格可实施，不表示实现完成。
- 局部效果修复、策略收益和最终产出成本分别验收。13 未通过前，不宣称达到平均五分钟一条成功轨迹。
- 使用方式与当前能力见[公开信息搜索文档](../../docs/PUBLIC_SEARCH.md)；任务格式与状态约定见[工作区任务约定](../../docs/agents/issue-tracker.md)。

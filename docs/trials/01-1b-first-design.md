# 初版设计：约 1B 混合注意力、GRU 多选解码与少量录制示范

- **时间**：2026-09-19 设计（`5167cb7` 起，`3785155` 定稿）；2026-09-22 按 ADR 0001 首次改写；2026-10-04 删除 GRU（`bb0b1ee`）；2026-10-06 目标难度改为 A0；2026-10-10 目标规模由 1B 改为 200M。
- **来源**：`3785155:ARCHITECTURE.md`；`bb0b1ee^:docs/adr/0001-redesign-for-fixed-time-playing-strength.md`、`bb0b1ee^:docs/ATTENTION_REDESIGN.md`、`bb0b1ee^:docs/PERFORMANCE.md`、`bb0b1ee^:docs/MODEL_PERFORMANCE_AUDIT.md`；`bb0b1ee`、`e525af6` 对 `ARCHITECTURE.md` 的删改；`.scratch/a0-training-baseline/spec.md`。
- **状态**：被否决，分四次推翻；GRU、线性注意力、价值头、里程碑奖励的代码都已删除。

## 动机

一个模型做整局决策。初版目标是“固定 A10 难度、五角色共享约 1B 模型、严格玩家视角、采用局内里程碑奖励，最终验收目标是击败第三幕 Boss”。

## 做法

### 规模与结构

- 25 层 Transformer，Full Attention 与线性注意力交替。
- 多选由 GRU“结合当前信息和上一步隐藏状态逐项解码”，动作按指针评分。
- 价值头估计整局回报，供 PPO 用。

### 示范

“约 10 条轨迹是启动预算”：每个角色约 2 条 RunRecorder 实机录制。

### 奖励

`milestones-v1`（`bb0b1ee^` 时的版本）：“普通/事件战斗胜利 0.005、精英 0.02，每幕非 Boss 奖励最多 0.1；第一幕 Boss 0.1、第二幕 Boss 0.2、确认最终通关 1.0”，无折扣蒙特卡洛回报，优势取 `return − old_value`。之后的过幕积分见 [06](06-milestone-reward-rl.md)。

## 结果

- 示范规模（`MODEL_PERFORMANCE_AUDIT.md`，2026-09-26）：“`data/steam-001/accepted.jsonl` 的 15 局、7,969 个非强制决策”；“最近的 bootstrap 日志记录 `validation_runs=0`”。
- 填充开销（CPU，4 个 macro）：“当前桶，pad=256 0.6591 s；按实际最大长度填充 0.2982 s”，约 2.21 倍；真实 token 数 P50／P90／P95／P99／max 为 137／167／174／189／415。
- 局部效果程序：“最长效果程序始终为 1、效果边为 0……coverage 均为 opaque”。
- 初版结构的通关率：记录未留存。

## 原因分析

- **1B 与数据量不匹配**。“示范数据每轮约 550 万宏动作，行为克隆撑不起十亿参数，1B 的采样前向也会让迭代成本不可接受”（`e525af6` 之后的 `ARCHITECTURE.md`）。
- **复杂度没有证据支撑**。混合注意力与 GRU 是在没有吞吐、效果测量时加入的。ADR 0001 改为“首个新架构基线采用小型纯 Full Attention”，混合与线性注意力只留作对照候选；GRU 当时保留，理由是“GRU 的设计职责是多选解码……不以缺少长期记忆为替换理由”。`bb0b1ee` 再改为无状态决策头：“每个决策独立编码，不保留状态……进度只来自输入”。
- **示范太少**。约 10 条录制覆盖不了整局决策的分布；实际只有 15 局，日志里没有一次留出验证。
- **难度过高**。A0 上都只有个位数通关时，A10 目标没有可学的信号；2026-10-06 的规格改为“五角色 A0 平均整局通关率”。

## 结论与后续

- 结构：ADR 0001 改为 20M 级纯 Full Attention + SwiGLU + Muon；现在只有一种结构 `architecture_version` 3，20M 与 200M 是它的两个尺寸，见 [architecture 全局骨干](../architecture.md#全局骨干)、[决策头](../architecture.md#决策头)、[network](../implementation/network.md#配置)。
- 规模：目标取 200M 而不是 1B，理由写在 [architecture 设计原则](../architecture.md#设计原则)。200M 的尝试见 [14](14-200m-training.md)。
- 示范：先改为搜索生成（[02](02-astar-run-search.md)、[04](04-mcts-backtracking-search.md)），再改为 Spire Codex 摘要锚点（[05](05-replay-restoration.md)、[architecture 示范数据](../architecture.md#示范数据)）。Steam 录制只作补充导入。
- 奖励：里程碑奖励先换成过幕积分（[06](06-milestone-reward-rl.md)），再换成只给通关 ±1（[12](12-outcome-ppo.md)、[env 奖励](../implementation/env.md#奖励)）。
- GRU、线性注意力、价值头与 `milestones-v1` 的代码都不在仓库里。

## 教训

- 规模由示范量和采样成本定，不由愿景定。
- 结构先用小模型和真实吞吐测量来定，复杂部件要有证据才加。
- 初版协议里“单一合法动作直接执行、多选逐步进行”的设计保留至今。

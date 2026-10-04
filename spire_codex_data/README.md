# Spire Codex 摘要锚点数据

从 Spire Codex 公开的 `v0.111.0 / 单人 / 标准 / 胜利` 对局里生成独立监督样本。唯一的生成方法是**摘要锚点**：只用对局摘要（`.run`）逐节点重建状态，每个决策在原生引擎里从对应的节点边界单独开始。操作回放只用来审计来源和核对结果，不参与生成。

流程：`fetch` 下载 → `sources` 筛选来源 → `anchor run` 生成 → `anchor export` 导出 → `model import-independent` 导入；`anchor_audit` 和 `map_ambiguity` 用回放核对。

## 下载

```bash
# 对局摘要按五个原版角色分片翻页；回放按 run_hash 合并到同一个 raw/ 目录
python3 -m spire_codex_data fetch --source runs --output data/spire-codex
python3 -m spire_codex_data fetch --source replays --output data/spire-codex
python3 -m spire_codex_data fetch --source runs --characters IRONCLAD SILENT --resume --output data/spire-codex
```

只需 Python 标准库，不用登录。站点对每个客户端按路由分别限速：对局摘要每分钟 60 次，列表和回放各 120 次。下载器对每个路由按略低于上限的节奏发请求，429 和暂时性错误按退避重试；`--parallel` 让几个角色分片同时进行，它们共用这一份请求额度，只是让慢响应互相重叠，不会突破上限；缓存原子写入并用 SHA-256 回执校验，重复运行可续传。

- 对局列表不带角色筛选时最多返回 10000 行（实际数量约为两倍），所以对局来源按角色分片；某一片的 total 达到 10000 时拒绝继续，需要进一步缩小分片。
- 写入缓存前核对：对局文档的版本、胜负、模式、单人，以及角色、难度与列表行一致；回放的首行是 header。不一致的不进入缓存，记为该局的错误。
- `download-{source}.json` 每页写一次，记录各分片的进度、成功和失败的对局。失败记录跨多次运行保留，重新下载成功后才清除。
- 带角色筛选的列表请求单次约 20 秒。`--resume` 让各分片从已记录的页继续；它不会发现之后新提交的对局，补漏需要不带 `--resume` 的完整扫描。`--max-runs` 只计本次实际发出请求的对局，在各分片间均分。
- 全量对局约 2 万局，下载约 7 小时。列表行里的 `username`、`player_token` 留在本地缓存，不进入任何导出。

## 筛选来源

```bash
python3 -m spire_codex_data.sources --existing data/spire-codex --output data/spire-codex/sources \
  --catalog data/spire-codex/catalog.json --per-character 20 --fetch
```

`sources` 写出 `manifest.json`：各角色数量相同、每个 seed 只取一局的对局名单，带文件路径和 SHA。`--catalog` 是本机游戏版本的原生内容目录，不存在时先用已配置的引擎生成。

- **拒绝**：非原版内容 ID、非标准幕序列、难度不在 0～10；有回放时再加上改变玩法的 mod、未认可的 mod 或补丁、修改状态或效果未知的控制台命令、续局快照不完整、header 与摘要不一致。
- **来源完整性**：有回放且没有任何未决项的记为 `audited_recording`；只有摘要、缺 mod 清单、或有续局的记为 `unverified_source`。摘要本身不含 mod 清单和控制台记录，所以只有摘要的对局可以使用，但不能当作无作弊的认证；这个标记随每条样本导出。

## 生成样本

先按[求解器配置说明](../combat_solver_cli/README.md#配置)固定本机的游戏和求解器依赖。摘要账本（`ledger.py`、`state_rules.py`）逐节点推算卡组、遗物、药水、HP 和金币；每个样本在新的原生进程里从一个节点边界开始（引擎入口 `summary-anchor-v1`，见[决策协议](../sts2-cli/docs/decision-protocol.md)），互不延续。

| 类型 | 起点 | 标签 | 怎样核对 |
| --- | --- | --- | --- |
| `battle` | 节点之前的状态，进入记录的遭遇 | CombatSolver 另行求解的动作 | 另一进程逐步重放（`native_replay`）；只导出获胜的战斗 |
| `rest` | 节点之前的状态，进入营火 | 记录的营火选项，锻造再加记录的升级目标 | 执行后的 HP、最大 HP、金币、卡组必须等于摘要的节点末状态（`recorded_outcome`） |
| `reward` | 节点末状态去掉所选的牌，按记录发放候选牌 | 记录的选牌，或都没选时的跳过 | 同上 |
| `map` | 节点末状态，位于该节点 | 记录路线上的下一个节点 | 标签必须是当前合法的移动；不执行移动（`legal_label`） |
| `ancient` | 节点之前的状态，进入远古之民事件，按记录给出选项 | 记录选中的选项 | 界面上的选项必须与记录一致；不执行选择（`legal_label`） |

括号里是样本 `metadata.verified_by` 的取值。战斗的动作是搜索出来的，所以每一场都在另一个进程里逐步重放。局外样本是引擎给出的一帧加一个对照过这一帧的标签，它成立与否不取决于能否复现；默认每 20 个局外锚点抽 1 个在第二个进程里重复一遍，用来监视初始化是否确定（`--replay-every`，1 为全部，0 为不抽）。抽中的样本 `native_replay_verified=true`。

```bash
python3 -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json \
  --output data/anchors --kinds rest reward map ancient
python3 -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json \
  --output data/anchors --kinds battle
python3 -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors      # independent-training.jsonl.gz、combat-outcomes.jsonl.gz
python3 -m model import-independent data/anchors/independent-training.jsonl.gz --output data/bootstrap
python3 -m model --device cuda bootstrap --data data/bootstrap --checkpoint runs/init --holdout 0.1 --output runs/bootstrap

# 战斗结果模型的预训练（与 Bootstrap 同期，见 combat_outcome/README.md）
python3 -m combat_outcome.train data/anchors/combat-outcomes.jsonl.gz --save runs/combat-outcome
```

`run` 按对局和类型续跑：`runs/<hash>.<类型>.json` 记录每个锚点的状态、失败原因和耗时，`samples/<hash>.<类型>.jsonl.gz` 保存样本行，`summary.json` 汇总。来源和依赖哈希（战斗另含预算）不变时跳过已完成的部分，所以可以先生成局外类型、以后再补战斗。重新编译引擎或 worker 会改变依赖哈希，全部重新生成；同一份数据不要混用两次编译的结果。

战斗默认预算为普通和精英各 1000ms、Boss 5000ms（`--budget-ms`、`--elite-budget-ms`、`--boss-budget-ms`），指每次新搜索的上限，同一回合内复用已有方案。求解器按毫秒预算搜索，**生成战斗时应单独运行并固定并发数**。局外类型不搜索，不受机器负载影响。

默认并发：战斗 8 个引擎，局外类型 16 个（`--workers` 同时覆盖两者）。战斗的并发不宜再高：在 16 核机器上，8 个并发时每次搜索展开的节点数比 4 个并发少约 6%，12 个少约四分之一，16 个少约三成，相当于变相降低预算。局外类型只受引擎启动速度影响。一次命令里两类都有时先跑局外类型，再跑战斗。

样本格式为 `spire-independent-decisions-v1`：`observation` 是行动前公开状态，`options` 是完整合法动作集合，`label` 是所选动作。`metadata` 不是学生输入，包含对局、角色、难度、按 seed 的 `split_group`、契约和路由，以及 `initialization`、`anchor_kind`、`label_source`（`combat_solver`、`historical_choice`、`historical_route`）、`state_sources`、`map_position`、`source_integrity`，供过滤和消融使用。战斗标签来自能看到隐藏状态的求解器，标为 `teacher_visibility=privileged`；所有样本 `bc_only=true`，不能当作 PPO 的自主采样，也不构成连续通关证明。

`export` 把摘要里记录的本幕 Boss 遭遇写到每一行地图的 Boss 节点上（字段 `encounter`；十阶第三幕两个 Boss 按出场顺序各写一个）：游戏从每幕开始就在地图上显示 Boss，而引擎的地图节点只有类型。后面几幕的地图和 Boss 不进观测。Boss 节点数与摘要记录对不上的行不导出，计入导出摘要的 `rows_without_recorded_boss`。

`export` 同时写出 `combat-outcomes.jsonl.gz`（`combat-outcomes-v1`）：每场通过验证的战斗一行，含第一个决策时的入场实体（玩家、永久卡组、遗物、药水、玩家能力）、遭遇、入场和结束 HP、胜负、回合数和实际用药，`seed` 字段是 `split_group`。求解器打输的战斗不进独立监督样本，但保留在这份文件里。它是战斗结果模型的预训练数据（见 [combat_outcome/README.md](../combat_outcome/README.md)）。

## 用回放核对

有回放的对局可以把样本和回放逐项对照：候选牌、所选的牌、营火选项和升级目标、远古之民的选项和选择、路线坐标，以及 HP、金币。

```bash
python3 -m spire_codex_data.anchor_audit data/anchors --replays data/spire-codex/raw
python3 -m spire_codex_data.map_ambiguity data/spire-codex --output data/anchors/map-ambiguity.json
```

`anchor_audit` 只比较回放里没有续局记录的对局，结果写入 `replay-audit.json`。除了标签和候选，它还比较卡组、遗物、药水、金币、HP 和遗物上显示的计数。读它的结果时要知道回放的几处特点：旧版回放会标错"获得卡牌"记录，所以选牌以前后两次牌组快照为准；回放在奖励界面打开时记录候选，同一界面上先拿的遗物仍可能升级候选牌；HP 只在回血时有精确值；被偷的金币和事件花费没有金币记录。`map_ambiguity` 测量节点类型序列能否在同一张地图上确定路线。

## 限制

- **状态是重建的，不是历史快照**。部分遗物计数在可行范围内采样，成长牌按战斗场数估算，只影响局外的属性取默认值，都记在 `state_sources` 里。战斗随机数是新的确定性随机流。原生复验只证明这个初始化下的执行一致。
- **同名牌**：摘要只用卡牌 ID 记录升级。同名牌里有带附魔的副本时，账本把每种可能都试一遍，保留能推出最终卡组的那些；它们一致的节点照常使用，不一致的节点（两张都升级过、先后不明）隔离。属性相同的副本视为同一张牌，记录的"加入楼层"只用来缩小范围。
- **账本仍然失败的节点隔离**：没有任何一种可能能推出最终卡组时，账本在出错的节点之后不再产出状态；最终卡组里有摘要没记录的升级或附魔时，涉及的牌所在的锚点隔离。会成长的附魔（Goopy）数值不明，带这种牌的锚点隔离。
- **蜥蜴尾巴**：摘要不记录它在哪一场用掉。战斗锚点仍然隔离（它决定这场会不会死）；局外锚点保留，把"用掉"放在持有期间的某一场战斗里，同一局的所有锚点用同一场，标为 `sampled_use_time`。
- **飞行靴**：已用次数不明时路线锚点隔离（它决定哪些移动合法），其它锚点保留并标为默认值。
- **按奖励计数的遗物**（长效糖果、丝绸发带、银坩埚）按取得之后的战斗奖励数推导，规则在全部有这些遗物的对局上与最终值一致。
- **地图位置**：摘要只记录每个节点的类型。原生地图由 seed 生成，本幕恰有一条路径与记录的类型序列一致时，位置和路线标签取这条路径（`map_position=recorded_route`）。地图在中途被改变（例如持有会改地图的任务牌）或用了自由通行时找不到这样的路径，位置退回到同类型的代表节点，这一幕不产生路线标签。
- **选牌**：一个节点记录了多于四张候选时无法分清是几组奖励，整节点隔离；没有选牌而界面上有替代选项时，摘要分不清跳过和替代选项，同样隔离。候选牌按记录原样发放，不让遗物再改一遍。奖励界面只含选牌奖励：金币、遗物、药水按固定规则在选牌之前领取，同一界面上被归还的牌也已在卡组里。拿牌会触发遗物时：复制出的第二张牌按摘要一并从选牌前的卡组里去掉；拿牌给的金币从节点末金币里扣回（引擎实际给了多少就扣多少，标为 `derived_pre_pick_gold`）；五轮书每加入五张牌回血一次，计数由摘要推出（拿到五轮书的那个节点上先后顺序不明，用最终计数定出来），选牌前的计数不含这次要拿的牌；这次拿牌正好凑满五张时，节点末 HP 里已含这次回血，选牌前的 HP 是扣回之后的值（标为 `derived_pre_pick_hp`），回血后满血、或最终计数与摘要对不上时无法确定，隔离（`unresolved_pre_pick_hp`）。其他拿牌触发的效果不推断，执行结果对不上就隔离。
- **营火**：一个节点有多个选项、或选项的后续选择摘要里没有记录（如烹饪）时隔离。结果核对不比较遗物，挖掘这类给随机遗物的选项只保证标签正确。
- **远古之民**只导出选哪个选项这一步；遗物随后要求的选牌等摘要里没有。
- **商店和普通事件尚未支持**。

## 测试

```bash
python3 -m pytest spire_codex_data/tests -q
SPIRE_CODEX_DATA_NATIVE_TESTS=1 python3 -m pytest spire_codex_data/tests -q   # 需要已配置的引擎和求解器
```

覆盖下载（分片与列表上限、失败记录、重试、续传、文档与列表行的一致性）、来源审计（mod、控制台、续局、非原版内容）、账本与状态规则、锚点规划与标签的歧义拒绝、导出过滤、回放核对的取证方式、路线唯一性判定，以及原生锚点入口的契约。

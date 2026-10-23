# demos：示范数据（摘要锚点与导入）

示范只用于 BC 起步（[architecture](../architecture.md#示范数据)）。`spire_codex_data/` 生成独立监督样本，来源有两种：

- **摘要锚点**：从 Spire Codex 公开的 `v0.111.0 / 单人 / 标准 / 胜利` 对局里，只用对局摘要（`.run`）逐节点重建状态，每个决策在原生引擎里从对应的节点边界单独开始。操作回放只用来审计来源和核对结果，不参与生成。本文除最后一节外都讲这一种。
- **入场存档的战斗**：求解器重打策略在幕组采样里打过的战斗，从这场战斗的入场存档进入（见[下文](#入场存档的战斗)）。

流程：`fetch` 下载 → `sources` 筛选来源 → `anchor run` 生成 → `anchor export` 导出 → `python -m main import-independent` 导入（`main/demos/independent.py`）；`anchor_audit` 和 `map_ambiguity` 用回放核对。Steam 录制另由 `main/demos/recorder/` 导入（见 [steam](steam.md)）。

| 模块（`spire_codex_data/`） | 内容 |
| --- | --- |
| `fetch.py`、`__main__.py` | 下载：限速、退避重试、原子缓存与续传 |
| `sources.py`、`integrity.py` | 来源筛选与来源完整性：mod、控制台命令（`MUTATING_COMMANDS`）、续局快照 |
| `ledger.py`、`state_rules.py` | 摘要账本：逐节点推算卡组、遗物、药水、HP、金币；持久状态规则（`anchor-state-v1`） |
| `anchor.py` | 锚点规划、生成（`run`）与导出（`export`）；求解器战斗与必须用药的战斗 |
| `battle_refresh.py` | 复用旧目录里已验证的战斗动作（`--reuse-battles`） |
| `recorded.py` | 摘要记下的选牌目标与节点末状态，事件变量的核对 |
| `rewards.py` | 奖励能否归到奖励界面、领取了什么，以及对应的隔离码 |
| `shop_sequence.py` | 商店：商店随机流的位置、格位与购买顺序搜索 |
| `anchor_audit.py`、`map_ambiguity.py` | 用回放核对样本；路线唯一性的测量 |
| `entry_battles.py` | 入场存档的战斗：求解、复验、汇总与导出 |

| 版本 | 当前值 |
| --- | --- |
| 样本格式 | `spire-independent-decisions-v1` |
| 锚点入口 | `summary-anchor-v1` |
| 局外生成 | `summary-anchor-samples-v3` |
| 战斗、领奖、拿牌、用药生成 | `summary-battle-refresh-v1`、`summary-claim-v1`、`summary-pick-v1`、`summary-potion-v1` |
| 持久状态规则 | `anchor-state-v1` |
| 入场存档的战斗、进入契约 | `entry-battle-v1`、`fight-resume-v1` |
| 摘要锚点导入 | `independent-import-v6` |
| Steam 录制格式与导入 | `sts2-run-recorder/v2`、`run-recorder-import-v6` |

局外规则和战斗数据分别使用缓存版本，并绑定原生程序集摘要。已有战斗可用 `anchor run --kinds battle --reuse-battles <旧锚点目录>` 刷新：先要求摘要校验和、求解预算及初始化快照一致，再执行已保存动作。除显式公开历史字段外，逐步观察和完整候选必须完全一致，胜负和结束 HP 也须匹配；当前格式的完整帧再由另一进程独立重放。不兼容或缺失的战斗重新求解，恢复方式写入 `recovery`。旧目录作为只读输入，新样本写入独立输出目录。

## 下载

```bash
# 对局摘要按五个原版角色分片翻页；回放按 run_hash 合并到同一个 raw/ 目录
python3 -m spire_codex_data fetch --source runs --output data/spire-codex
python3 -m spire_codex_data fetch --source replays --output data/spire-codex
python3 -m spire_codex_data fetch --source runs --characters IRONCLAD SILENT --resume --output data/spire-codex
```

`--source` 默认 `both`（先回放、后摘要）；`--start-page` 指定起始页。

只需 Python 标准库，不用登录。站点对每个客户端按路由分别限速：对局摘要每分钟 60 次，列表和回放各 120 次。下载器对每个路由按略低于上限的节奏发请求，429 和暂时性错误按退避重试，等待中的线程醒来后重新检查共享截止时间，晚醒时不补发积压时隙；`--parallel` 让几个角色分片同时进行，它们共用这一份请求额度，只是让慢响应互相重叠，不会突破上限；缓存原子写入并用 SHA-256 回执校验，重复运行可续传。

- 对局列表不带角色筛选时最多返回 10000 行（实际数量约为两倍），所以对局来源按角色分片；某一片的 total 达到 10000 时拒绝继续，需要进一步缩小分片。
- 写入缓存前核对：对局文档的版本、胜负、模式、单人，以及角色、难度与列表行一致；回放的首行是 header。不一致的不进入缓存，记为该局的错误。
- `download-{source}.json` 每页写一次，记录各分片的进度、成功和失败的对局。失败记录跨多次运行保留，重新下载成功后才清除。
- `--resume` 让各分片从已记录的页继续；它不会发现之后新提交的对局，补漏需要不带 `--resume` 的完整扫描。`--max-runs` 只计本次实际发出请求的对局，在各分片间均分。
- 列表行里的 `username`、`player_token` 留在本地缓存，不进入任何导出。

## 筛选来源

```bash
python3 -m spire_codex_data.sources --existing data/spire-codex --output data/spire-codex/sources \
  --catalog data/spire-codex/catalog.json --per-character 20 --fetch
```

`sources` 写出 `manifest.json`：各角色数量相同、每个 seed 只取一局的对局名单，带文件路径和 SHA。`--catalog` 是本机游戏版本的原生内容目录，不存在时先用已配置的引擎生成。

- **拒绝**：非原版内容 ID、非标准幕序列、难度不在 0～10；有回放时再加上改变玩法的 mod、未认可的 mod 或补丁、修改状态或效果未知的控制台命令、续局快照不完整、header 与摘要不一致。
- **来源完整性**：有回放且没有任何未决项的记为 `audited_recording`；只有摘要、缺 mod 清单、或有续局的记为 `unverified_source`。摘要本身不含 mod 清单和控制台记录，所以只有摘要的对局可以使用，但不能当作无作弊的认证；这个标记随每条样本导出。

## 生成样本

先按[求解器配置](solver.md#配置)固定本机的游戏和求解器依赖。摘要账本（`ledger.py`、`state_rules.py`）逐节点推算卡组、遗物、药水、HP 和金币；每个样本各自从一个节点边界装入引擎（入口 `summary-anchor-v1`，见[决策协议](../../sts2-cli/docs/decision-protocol.md)），互不延续。一局的锚点共用原生进程：局外锚点一个，战斗两个（求解器在一个里打，另一个逐步重放）。装入下一个锚点时引擎先结束上一个锚点的对局，引擎拒绝或出错时该锚点改在新进程里重做。

| 类型 | 起点 | 标签 | 怎样核对 |
| --- | --- | --- | --- |
| `battle` | 节点之前的状态，进入记录的遭遇 | CombatSolver 另行求解的动作 | 另一进程逐步重放（`native_replay`）；只导出获胜的战斗 |
| `potion` | 与 `battle` 相同；每两场战斗抽一场，开场药水栏里随机若干瓶（至少一瓶）指定为必须使用 | CombatSolver 在“这几瓶必须用、其余不用”的约束下求解的动作 | 同 `battle`；导出时取代同一场战斗的 `battle` 动作 |
| `rest` | 节点之前的状态，进入营火 | 记录的营火选项，锻造再加记录的升级目标 | 执行后的 HP、最大 HP、金币、卡组必须等于摘要的节点末状态（`recorded_outcome`） |
| `reward` | 节点末状态去掉所选的牌，按记录发放候选牌 | 记录的选牌，或都没选时的跳过 | 同上 |
| `map` | 节点末状态，位于该节点 | 记录路线上的下一个节点 | 标签必须是当前合法的移动；不执行移动（`legal_label`） |
| `ancient` | 节点之前的状态，进入远古之民事件，按记录给出选项 | 记录选中的选项 | 界面上的选项必须与记录一致；不执行选择（`legal_label`） |
| `event` | 节点之前的状态，进入记录的普通事件，由游戏按这一局的 seed 再生成 | 记录的每一步选项，以及选项引出的选牌（按记录的删除、附魔、升级、变化目标） | 每一步必须是界面上的选项；事件结束时的变量必须等于摘要所记；执行后的 HP、最大 HP、金币、卡组、遗物、药水必须等于节点末状态（`recorded_outcome`） |
| `claim` | 节点末状态去掉记录为已领取的遗物、金币、药水和所选的牌；奖励界面上放着记录的全部遗物、金币、药水和选牌候选 | 按“遗物、金币、药水”的固定顺序逐个领取 | 界面上的奖励必须与记录一致；领完后的 HP、最大 HP、金币、卡组、遗物、药水必须等于选牌锚点的起始状态（`recorded_outcome`） |
| `pick` | 节点末状态去掉所选的牌和记录为已领取的药水；奖励界面上放着选牌候选、记录的药水（领了的和没领的）和记录为“未选”的遗物 | 记录的选牌，然后领取记录为已领取的药水 | 界面上的奖励必须与记录一致；执行后的 HP、最大 HP、金币、卡组、遗物、药水必须等于节点末状态（`recorded_outcome`） |
| `unclaimed` | 节点末状态，战后奖励界面上放着记录为“未选”的药水和遗物，没拿牌时再加上记录的选牌候选 | 离开奖励界面 | 界面上的奖励必须与记录一致；不执行离开（`legal_label`） |
| `treasure` | 节点之前的状态，进入宝箱房；候选取摘要记录的单件遗物 | 开箱、领取／明确记录的跳过，以及有记录的后续选牌 | 原生发放开箱金币和取得效果；核对节点末状态；另一进程逐步重放 |
| `shop` | 节点之前的状态，进入商店；库存由这一局自己的商店随机流重新抽出 | 记录的购买、删牌、用药和丢药，按一种游戏接受的顺序执行，最后是离开 | 抽出的库存必须逐格等于记录；执行后的 HP、最大 HP、金币、卡组、遗物、药水必须等于摘要的节点末状态（`recorded_outcome`） |

括号里是样本 `metadata.verified_by` 的取值。战斗、商店和宝箱还会在另一个进程逐步重放，每帧公开状态、完整候选和标签必须一致，`native_replay_verified=true`。商店和宝箱同时核对摘要的节点末状态。其它局外锚点保留表中所列的验证方式。

```bash
python3 -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json \
  --output data/anchors --kinds rest reward map ancient shop event unclaimed treasure claim pick
python3 -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json \
  --output data/anchors --kinds battle
python3 -m spire_codex_data.anchor run --manifest data/spire-codex/sources/manifest.json \
  --output data/anchors --kinds potion
python3 -m spire_codex_data.anchor export --manifest data/spire-codex/sources/manifest.json --output data/anchors      # independent-training.jsonl.gz
python -m main import-independent data/anchors/independent-training.jsonl.gz --output data/bootstrap
```

`run` 按对局和类型续跑：`runs/<hash>.<类型>.json` 记录锚点状态、失败原因和耗时，`samples/<hash>.<类型>.jsonl.gz` 保存样本行，`summary.json` 汇总，`last-run.json` 是这次命令的汇总与失败的对局（有失败时命令以 1 退出）。`--max-runs N` 只处理名单前 N 局；`--solver-config` 指定求解器配置（默认 `combat_solver_cli/lib/config.json`）。只有对局摘要校验和、该类型的生成版本、原生程序集摘要及战斗预算都一致时，才复用已完成报告；局外规则版本与战斗版本分开维护。重编后若程序集内容改变，旧报告不会被静默跳过。复用旧战斗动作的方式见本文开头的 `--reuse-battles`，无需删除原始证据。

导入后的格式、索引与复用见 [data](data.md)，BC 的命令见 [README](../../README.md)。

战斗默认预算为普通和精英各 5000ms、Boss 20000ms（`--budget-ms`、`--elite-budget-ms`、`--boss-budget-ms`），指每次新搜索的上限，同一回合内复用已有方案。预算直接决定标签质量，并进入续跑身份：对按别的预算求出的锚点目录用默认值重跑 `--kinds battle` 会把全部战斗重新求解，要沿用就显式给出原预算。`--potion-costs 基础 较高 最高` 让求解器自己的药水策略按这三档价格（HP）搜索，不给就是它自己的 9 14 18；价格同样进入续跑身份；求解进程须认识 `potion_costs`（不认识的旧 worker 会不声不响地按原价搜索，所以每次搜索都核对回报的价格）。`--final-boss-potion-costs 基础 较高 最高` 只用于整局最后一场战斗，并让那一场按完整 HP 比较（见 [solver](solver.md)），不给时那一场只为赢才用药；回报的 `final_boss_priced` 说明这次搜索按哪组价格，核对按它选，worker 不认识这个字段就拒绝。`--no-beam-portfolio` 让无药搜索每个束宽只跑一次，给用药搜索留出预算。两者同样进入续跑身份。求解器按毫秒预算搜索，**生成战斗时应单独运行并固定并发数**。局外类型不搜索，不受机器负载影响。

默认并发：战斗和其余类型都是 16 局同时进行（`--workers` 同时覆盖两者），每局在自己的 Python 进程里处理：读、比对、写一局的样本行要几秒钟的 Python 时间，放在同一个解释器的线程里会互相等待。求解器的每次搜索是单线程的，按毫秒预算截断：并发越高吞吐越高，但每次搜索更浅。想让每次搜索更深，就调低并发或调高预算；并发超过物理核数只会让搜索更浅。局外类型每局启动一次引擎。一次命令分两轮：先跑除 `battle` 外的全部类型（含同样由求解器打的 `potion`，按这一轮的并发），再跑 `battle`。

样本格式为 `spire-independent-decisions-v1`：`observation` 是行动前公开状态，`options` 是完整合法动作集合，`label` 是所选动作。`metadata` 不是学生输入，包含对局、角色、难度、按 seed 的 `split_group`、契约和路由，以及 `initialization`、`anchor_kind`、`label_source`（`combat_solver`、`historical_choice`、`historical_route`、`reconstructed_order`）、`state_sources`、`map_position`、`source_integrity`，供过滤和消融使用。战斗标签来自能看到隐藏状态的求解器，标为 `teacher_visibility=privileged`；所有样本 `bc_only=true`，不是策略自己的采样，也不构成连续通关证明。

地图的 Boss 节点带本幕 Boss 的遭遇（字段 `encounter`；十阶第三幕两个 Boss 按出场顺序各一个）：游戏从每幕开始就在地图上显示 Boss。样本里用的是摘要记录的 Boss，不是 seed 抽出的（游戏生成一幕时先按玩家已揭示的纪元裁剪事件表再洗牌，Boss 在同一个随机流后面抽；无头引擎按全解锁生成，抽出的 Boss 与进度较少的玩家实际遇到的不一定相同）：锚点把记录的 Boss 写进存档，引擎发布的帧里就是它。后面几幕的地图和 Boss 不进观测。记录的 Boss 场数与本幕的 Boss 节点数对不上时没法装入，这样的行地图上的 Boss 与记录不一致，`export` 不导出，计入导出摘要（`independent-export-summary.json`）的 `rows_without_recorded_boss`。

`export` 用 `--workers` 个进程（默认 16）读各份报告的样本，按报告文件名的顺序写出：样本文件是各份报告的 gzip 段依次相接，读起来是一条流。求解器打输的战斗不导出。

局外生成版本为 `summary-anchor-samples-v3`，战斗为 `summary-battle-refresh-v1`。只重新导出不能补上缺失的公开内容或原生复验，生成规则变了要重跑受影响的类型。商店和宝箱的 `metadata.recovery` 保存库存／金币约束、顺序依据和独立复验方式，导入后保留在 `provenance.recovery`，不进入模型观察。

控制分工（宝箱固定流程、单选事件）只改变训练过滤，现有完整候选足以判定，不要求重新生成锚点。`export` 跳过完全由控制器或单一合法动作构成的样本组，计入导出摘要的 `rows_without_decision`；组内有真正后续选择时，连同环境操作证据保留。导入时控制器步骤进 `environment_actions`；控制规则变了，对已导入的分片执行 `reindex` 更新计数（见 [env](env.md#控制分工)）。

## 必须用药的战斗

求解器按自己的 `Smart` 策略几乎不用药水，远少于玩家在同样的战斗、同样的药水栏下的用药；照它学出来的策略不会用药，带着满栏药水打到死。`potion` 锚点给出用药的示范（两者的用药统计见 [trials](../trials/route.md)）：

- 每场战斗按锚点 id 抽签，一半入选；入选且开场药水栏不空的，从栏里随机取 1 到全部瓶数，指定为必须使用（`forced`）。自动触发的瓶中精灵不指定。抽签只取决于 id，重跑得到同样的指定。
- 求解时这几瓶必须出现在整场战斗的路线里，其余药水禁用；每一步搜索都按栏里还欠的那几瓶重新下指令。何时用、对谁用由求解器决定。打完时有指定的药水没用上（离开了药水栏，或战斗先结束）的不通过。
- 与 `battle` 一样在另一进程逐步重放。导出时，打赢的 `potion` 战斗取代同一场 `battle` 的动作（两者起点相同、标签不同）；打输的不取代。

用不用药、用几瓶是随机指定的，不是玩家的记录。记录里的用药（`potion_used`）不采用：它只说这个节点用了哪几瓶，不提供时机信息，何时用仍由求解器决定，与随机指定没有区别。生成版本为 `summary-potion-v1`，求解进程须支持 `potion_directives`。

## 商店

摘要记录了一家商店卖什么（买走的，和离开时剩下的），没有价格，也没有购买顺序。

- **价格和库存来自这一局自己的商店随机流**。游戏用玩家的商店随机流给每件商品定价、在稀有度之内挑牌、抽药水；这条流由对局的 seed 决定，只有商人消耗它，每次消耗的个数是固定的（进一家商店 28 次，假商人事件 6 次，信使遗物补货时卡牌 2 次、遗物 1 次、药水 3 次）。所以把它拨到之前的商人留下的位置，游戏就会把这家店重新抽一遍。每张角色卡牌的稀有度（出自奖励随机流和保底计数）和三件遗物（出自遗物袋）游戏是从别处抽的，这些由锚点按记录给出；牌是哪一张、药水是哪三瓶、每件多少钱都由游戏自己抽。抽出的卡牌和药水必须逐格等于记录，否则这家店不用。删牌服务的价格取决于此前用过几次，按之前商店节点的删牌记录计数。
- **解锁进度按记录找出**。能抽到哪些牌和药水取决于玩家解锁了什么，摘要里没有。卡池只受少数几个纪元控制（角色卡牌 3 个、无色卡牌 5 个、药水 3 个），所以逐个试：在哪种解锁状态下游戏抽出的正是记录的库存，就用哪种。错误的状态不会碰巧抽对七张牌和三瓶药水。样本标为 `inferred_unlock_state`，报告里列出当时缺的纪元。
- **格位**。离开时剩下的商品按格位顺序记录，买走的不在其中。角色卡牌的五格依次是攻击、攻击、技能、技能、能力；同类型里买走一张时两种摆法都试，只有一种能让游戏抽出记录的牌。遗物三格是两件随机稀有度的和一件商店专属的；前两件里有被买走的，摆法由执行后的金币定出，两种摆法都对得上而价格不同时这家店隔离（`shop_relic_slots_ambiguous`）。
- **顺序是重建的**。记录的购买、删牌、喝掉和丢掉的药水是一组动作，逐个顺序试：每一步都必须是当时界面上的合法动作，最后的状态必须等于摘要的节点末状态。先试的顺序把会改变之后价格或能买什么的动作放前面（丢药和用药、遗物、药水、卡牌、删牌）；不成立再换，同一个被拒绝的开头不再重复，一家店最多完整执行 24 次。这样找到的是一种合法顺序，不是玩家当时的顺序：购买这些步标为 `reconstructed_order`，最后的离开是 `historical_choice`。
- 购买引出的选牌（删牌，或遗物要求选牌附魔、升级、变化）按摘要记录的那几张牌回答；摘要没有记录的选择隔离。
- 成功的格位、解锁状态和动作顺序冻结后，在另一个进程重放；复验进程不重新搜索顺序。最后的离开也执行，返回地图且节点末状态匹配才接受。
- 隔离的情况：有信使遗物时（每次购买都补货，补出什么取决于顺序）；牌、遗物、药水的数量不是 7／3／3，或同一个列表里混着购买引出的选牌奖励；随机效果对不上记录的（如随机升级的遗物）；进店时先弹出别的选择。

## 普通事件

事件用自己的随机流，由对局的 seed 和事件 ID 决定，与之前发生过什么无关。所以从记录的节点前状态进入同一个事件，游戏给出的页面、选项和数值就是当时的，前提是事件读到的状态与历史相同。

- 摘要的 `event_choices` 按顺序记录每一步选中的选项（标题的文本键）和事件变量。每一步必须正好对应界面上一个可选的选项；游戏不写进历史的唯一出路（如“继续”）直接走过。
- **变量是事件结束后的值**。历史条目保存的是变量对象本身，对局结束时才序列化，所以多步事件里每一步显示的都是最终值（例如滑桥事件每一步都显示最后一次的扣血量）。核对时取事件结束时的变量，与每一步的记录比较：数值和开关必须相等；文本变量是内容名，游戏按玩家的语言写下，引擎给的是文本键，用随引擎安装的本地化表（英文、简体中文）核对，对得上或对不上都算数，别的语言无法核对（报告里记为 `text`）。
- 选项引出的选牌按摘要记录的那几张牌回答（删除、附魔、升级、变化）。奖励界面只有一项奖励时，先把全部候选与摘要核对，再领取或跳过；一个选牌组算一项。多奖励／多界面无法分组、金币来源不明、同节点用药或丢药时点不明、未选卡牌与替代奖励分不清时隔离。水晶球等未记录过程仍隔离。
- 最后核对节点末状态。事件从别的来源取随机结果时对不上，隔离：例如给随机遗物的事件（遗物袋和奖励随机流）、给随机牌的事件、按牌组内部顺序挑牌的事件。
- 事件里发生战斗的节点有多个房间，没有独立的边界，隔离。没有 `event_choices` 的事件（假商人、未做选择的事件）不产生样本。

## 领取战后奖励

摘要记下了一个节点上哪些遗物和药水被领取（`was_picked`）、金币增加了多少，但没有领取的先后。`claim` 锚点把这些领取补成样本：奖励界面按记录摆出全部奖励（金币、遗物、药水、选牌候选都在），按“遗物、金币、药水”的固定顺序领取，每一步一行，标签来源是 `reconstructed_order`，不声称是历史顺序。领完之后界面上剩下的正是 `reward` 锚点的起点（选牌）或 `unclaimed` 锚点的界面（留下的东西）。

记录只说节点上得到了什么，不说从哪里得到，所以只有能归到奖励界面的才用：

- **金币**：与 `unclaimed` 相同，节点只有一个金币来源且增量为正；界面上显示的金币就是记录的增量。
- **药水**：节点至多记录一瓶药水，并且没有别的途径得到药水（卡组里有炼制药水的牌、持有每场战斗给药水的遗物、或本节点领到的遗物取得时附带药水，都隔离，`reward_potion_source_unknown`）。本节点丢弃过药水时时点不明，隔离（`reward_potion_timing_unknown`）；用掉的药水视为在战斗中用掉。
- **遗物**：只在精英战后、且只记录一件时使用（`reward_relic_source_unknown`）。
- 领完后的状态必须等于记录：取得时再次改变 HP、金币或卡组的遗物对不上，隔离；药水栏已满而记录为领取的，界面上拿不了，隔离（`recorded_reward_not_legal`）。

生成版本为 `summary-claim-v1`，引擎的 `rewards` 房间类型需要支持 `gold`。

### 界面上还有药水或遗物时的选牌

`reward` 锚点的界面上只有选牌候选，`claim` 锚点把药水放在选牌之前领。于是“选牌候选和药水同时在界面上”的状态只出现在两种样本里：接着领药水（`claim`），或什么都不拿就离开（`unclaimed`，玩家既没拿牌也没拿药水）。`pick` 锚点补上“这时拿牌”：

- **留下的药水或遗物**：游戏只在奖励被留下时把它记为“未选”，所以拿牌的那一刻它一定在界面上。这一步是历史状态上的历史选择（`historical_choice`），常见于药水栏已满、药水拿不了的时候。
- **领取了的药水**：记录没有先后。`pick` 先拿牌、再领药水，两步都标为 `reconstructed_order`；`claim` 是另一种顺序。遗物和金币仍按 `claim` 的顺序先领，不在 `pick` 的界面上。

只有拿了牌、且界面上有药水或留下的遗物时才有 `pick` 锚点；奖励能否归到奖励界面沿用 `claim` 的条件，拿牌触发遗物时的处理沿用 `reward`。生成版本为 `summary-pick-v1`。

## 留在奖励界面上的奖励

战后奖励的领取顺序摘要里没有。不依赖顺序就能确定的是最后一步：游戏只在奖励被留下时把它记为“未选”，因此离开奖励界面那一刻界面上还有什么是已知的。`unclaimed` 锚点在节点末状态上放出这些药水和遗物；记录里没有拿牌时，选牌候选也还在界面上。标签是离开。药水栏满时，合法动作里有丢弃和可用的药水，不能直接拿。界面上的东西都拿不了、只能离开时（例如带着 Sozu 留下的药水）没有可学的选择，锚点记为 `single_action`，不出样本。

- 金币留在界面上不会被记录：节点的金币增量为零时隔离（`gold_reward_unsettled`）。有其它金币来源也不能由正增量证明金币全领完：额外金币奖励、入场／加牌给钱的遗物、取得即给钱的遗物、战斗中给钱的牌、失窃金币返还等隔离（`gold_reward_multiple_sources`）。
- 候选牌多于四张（多组奖励）、没拿牌而界面上有替代选项、同一个奖励被记了两次时隔离。
- 没拿牌且还有未领药水／遗物时，只保留完整的 `unclaimed` 组；`reward` 锚点里不完整的跳过帧以 `covered_by_unclaimed_reward_group` 隔离，即使本次只请求生成 `reward`。其它已有 `reward` 锚点仍假设其它奖励已在选牌之前领取；界面上还有药水或留下的遗物时的选牌由 `pick` 给出。

## 宝箱

`treasure` 从节点入场状态向前执行，候选必须是摘要明确记录的单件遗物。摘要没写遗物时，无法分清没开箱、空箱和跳过，不能造一个空候选。混合奖励、同节点用药或丢药时点不明的宝箱隔离。

开箱金币自动发放，发生在领取遗物之前。摘要不保存这次随机流，因此枚举原生范围内的 11 个原始结果（42～52）；只约束这一次抽取，抽取仍推进原随机流，难度折算、金币修饰、遗物取得及选牌都由原生执行。HP、金币、最大 HP、卡组、遗物和药水必须等于节点末记录。不同结果若能匹配末状态、却产生不同的公开决策序列，隔离；难度舍入造成同一公开金额的多个原始结果可以保留。不会直接设置最终金币或 HP。

通过约束的开箱、领取／明确记录的跳过和后续选牌，在独立进程重放。恢复证据标为 `recorded_treasure_offer`、`derived_chest_gold`，不声称还原了历史 RNG。开箱前没有潜在遗物或即将获得的金币；领取前也没有新遗物的取得效果。

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
- **选牌**：一个节点记录了多于四张候选时无法分清是几组奖励，整节点隔离；没有选牌而界面上有替代选项时，摘要分不清跳过和替代选项，同样隔离。候选牌按记录原样发放，不让遗物再改一遍。奖励界面只含选牌奖励：`reward` 锚点假设金币、遗物、药水已在选牌之前领取，同一界面上被归还的牌也已在卡组里。拿牌会触发遗物时：复制出的第二张牌按摘要一并从选牌前的卡组里去掉；拿牌给的金币从节点末金币里扣回（引擎实际给了多少就扣多少，标为 `derived_pre_pick_gold`）；五轮书每加入五张牌回血一次，计数由摘要推出（拿到五轮书的那个节点上先后顺序不明，用最终计数定出来），选牌前的计数不含这次要拿的牌；这次拿牌正好凑满五张时，节点末 HP 里已含这次回血，选牌前的 HP 是扣回之后的值（标为 `derived_pre_pick_hp`），回血后满血、或最终计数与摘要对不上时无法确定，隔离（`unresolved_pre_pick_hp`）。其他拿牌触发的效果不推断，执行结果对不上就隔离。
- **营火**：一个节点有多个选项、或选项的后续选择摘要里没有记录（如烹饪）时隔离。结果核对不比较遗物，挖掘这类给随机遗物的选项只保证标签正确。
- **远古之民**只导出选哪个选项这一步；遗物随后要求的选牌等摘要里没有。
- **事件选项里引用的整段描述不进观察**。选项显示的数值和内容名（代价、数量、哪张牌、哪瓶药水）随帧导出；选项文本里嵌入的另一段描述（如茶艺大师选项里遗物的效果说明）和对不上唯一内容的名字没有。
- **多项战后奖励的领取顺序没有恢复**。离开奖励界面、事件内的单项奖励和原生固定顺序的宝箱不依赖顺序；`claim` 的领取步骤按固定顺序重建，`pick` 再给出“先拿牌、后领药水”的另一种顺序；`reward` 选牌锚点仍假设其它奖励已先领取。

## 测试

```bash
python -m pytest main/tests/test_independent.py -q
python3 -m pytest spire_codex_data/tests -q
SPIRE_CODEX_DATA_NATIVE_TESTS=1 python3 -m pytest spire_codex_data/tests -q   # 需要已配置的引擎和求解器
```

覆盖下载（分片与列表上限、失败记录、重试、续传、文档与列表行的一致性）、来源审计（mod、控制台、续局、非原版内容）、账本与状态规则、锚点规划与标签的歧义拒绝、商店的随机流位置、格位与顺序搜索、事件变量的核对、记录的选牌目标、留在奖励界面上的奖励、导出过滤、回放核对的取证方式、路线唯一性判定，以及原生锚点入口的契约（含商店库存的重抽、解锁状态的反推、事件的再生成和奖励界面）。

## 导入

`python -m main import-independent <independent-training.jsonl.gz> --output <目录>`（`main/demos/independent.py`）把一个样本组（一局的局外决策，或一场求解器战斗）转成一条监督“对局”，写成分片加索引（[data](data.md)）。控制器步骤移入 `environment_actions`；只含环境操作的组不参与训练。`--shard-samples` 控制分片大小（默认每片 2000 个决策步），`--workers` 是转换样本组的进程数（默认逻辑核数的一半）。导入没有任何样本时返回 2。

导入后做 BC 起步的两段（命令见 [README](../../README.md#bc-起步)）：第一段拟合全部难度；第二段 `--ascension 10 --combat-from <按更大预算另求的 A10 战斗目录>`，局外样本仍是第一段数据里 A10 的那部分。


## 入场存档的战斗

`entry_battles.py` 让求解器重打策略自己打过的战斗（[ADR 0001](../adr/0001-separate-combat-training.md) 的战斗示范）。来源是幕组采样记下的[入场存档](env.md#入场存档)（`iterate --fight-entries`，格式见 [play](play.md#幕组)）；打哪些战斗由一份 `selection.json` 给出：`fights` 每项带 `id`、入场存档的 `path`（相对于 selection 所在目录或它的某一级上级目录）、`character`、`seed`、`ascension`、`kind`、`act`、`floor`；`extra` 同样格式，只打、只汇总、不导出。幕组写的入场存档只有 `at`、`save`、`fight`，所属对局（角色、种子、难度）取 selection 这一行的。

- **进入**：两个原生进程都用 `resume_fight` 按存档进入（契约 `initialization=fight-resume-v1`），卡组、遗物与计数按存档原样恢复，不像锚点那样从字段重建。入场存档记的是进这个房间时的状态，重进时开战效果只发生一次，敌人的随机 HP 与开局手牌也与原战斗相同，求解器打的就是策略打过的那一场（见 [env](env.md#入场存档)）。引擎有 `RoomEntrySave` 之前写的入场存档记的是战斗第一个决策时的状态：重进会把开战对整局状态的效果再叠加一次（Blood Vial 多回血，Royal Poison 多扣 4 HP，Maw Bank、Seal of Gold 多改金币，Petrified Toad 多一瓶药，计数遗物多走一步），敌人的随机 HP 与开局手牌重抽；这类存档要用现在的引擎重新采集。
- **求解与复验**：同 `battle` 锚点（`anchor.play_battle`）：一个进程里求解器打，另一个逐步重放。同一局（角色、种子）的几场战斗按幕、层顺序共用这两个进程，`--workers` 局同时进行（默认 16）。
- **设置**：药价 7/9/11（`--potion-costs`），整局最后一场战斗 1/1/1 并按完整 HP 比较（`--final-boss-potion-costs`）；预算与 `anchor run` 相同（普通、精英 5000ms，Boss 20000ms）；无药搜索默认不开束宽组合（`--beam-portfolio` 打开）；腰带有药时把一半预算留给用药层（`--potion-reserve`，默认 0.5，见 [solver](solver.md#smart-用药)）。开局的选牌（赌博筹码、烤火手套、工具箱）由求解器搜索（[solver](solver.md#回合准备选牌)）。设置、生成版本与四个 DLL 的摘要进入续跑身份，一致且战斗名单相同的局直接沿用。
- **输出**：`runs/<角色>-<种子>.json` 是每场的报告（`status`、`outcome`、`entry_hp`／`exit_hp`、`potions_used`、`setup_choice`：开局选牌拿了几张、由谁选，`seconds`），`samples/<角色>-<种子>.jsonl.gz` 是每场的样本行（胜负都留），`summary.json` 按种类汇总（失败原因、胜负、每场耗时与用药、开局选牌的张数分布，`extra` 单列）。一场出错只记这一场失败；一局在开打前就出错时，这局的每场都记为失败、报告不带续跑身份，再跑同一命令时重打。
- **导出**：`export` 写 `independent-training.jsonl.gz` 与 `export-summary.json`，只导出 `fights` 里求解器打赢、且有真正决策的战斗；打输的、`extra` 的和全是单候选或控制器步骤的分别计数。样本 `anchor_kind=entry_battle`、`label_source=combat_solver`、`source_integrity=sampled_rollout`、`map_position=entry_save`，没有公开历史前缀。

```bash
python -m spire_codex_data.entry_battles run --selection <selection.json> --output <目录>
python -m spire_codex_data.entry_battles export --output <目录>     # independent-training.jsonl.gz
python -m main import-independent <目录>/independent-training.jsonl.gz --output <数据目录>
```

求解器按毫秒预算搜索，与 `anchor run` 一样要单独运行、固定并发数，不与采样或训练同时跑。

### 逐轮的战斗示范

BC 阶段每轮从这一轮幕组打过的战斗里挑一批交给求解器（[ADR 0001](../adr/0001-separate-combat-training.md)）。挑选在 `main/play/demo_picks.py`（[play](play.md#战斗示范的挑选)），求解与导入在 `main/demos/rounds.py`：

- `solve_round(<轮目录>/demos)`：读挑选写下的 `selection.json`，以子进程跑 `entry_battles run`（`--workers`，默认 16）与 `export`，再用 `import_samples` 把导出导入成 `demos/imported/`，更新从这里读这一轮的战斗示范。求解器的报告、样本与导出在 `demos/solved/`。
- 续跑：`imported/summary.json` 已在就直接沿用，不再求解；否则重跑两条命令（`entry_battles` 沿用已求完的局）。没导完的 `imported/` 删掉重导。
- 返回数据目录、求解器的 `summary.json`，以及导入的组数、决策步数、宏动作数和隔离数。

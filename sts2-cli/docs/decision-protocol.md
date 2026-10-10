# 引擎决策协议与完整对局

`engine-candidates-v1` 为模型提供同一公开快照上的完整合法候选，并由真实游戏引擎执行。正常训练不调用上游 `DetectDecisionPoint` 的自动领奖／默认选择分支。Python 训练入口与模块说明见[根目录 README](../../README.md)与 [ARCHITECTURE](../../ARCHITECTURE.md)。

## JSON Lines 入口

准备游戏程序集并构建后，启动 `Sts2Headless.dll`；读取 ready 握手，再逐行发送：

```json
{"cmd":"start_run","character":"Ironclad","seed":"example-run","ascension":10,"decision_protocol":true}
```

返回 `decision_frame`，含 `contract`、`boundary`、`routing`、`public`、`legal.candidates` 和 `events`。

当前观察版本为 `public-state-v6`，`contract.public_history_version` 为 `public-history-v1`。`public.memory` 包含系统由公开事件维护的历史事实；牌位的 owner 引用对应当前抽牌堆卡牌，只有已知位置生成 `known_draw_before` 关系。概率字段包含 value／known／applicable，读档缺前缀不是初始概率。问号房的两种商店条件分布须结合上一地图点商店事实和所选节点的后继使用。机制与重置规则见 [输入设计](../../ARCHITECTURE.md#输入与公开信息)。旧帧仍可作 BC，但不能继续用于新输入版本的 PPO；历史公开状态哈希须按原观察版本核对。

从候选中选择引用，原样带回版本：

```json
{"cmd":"execute_candidate","decision_id":"从 routing 读取","state_version":4,"candidate_ref":"c0","selection_revision":0}
```

只有选择会话需要 `selection_revision`；普通动作省略。响应是下一边界。`waiting` 时查询 `{"cmd":"advance_to_boundary"}`，不推测下一动作。游戏在动作之后仍在运行的原生续延（例如 Lord's Parasol 进店后的连续购买及其间的选牌）完成或请求输入前，只发布 `waiting`。HP 归零不等于死亡：死亡阻止（Lizard Tail）结算前发布 `waiting`，只有原生 `Died` 事件或战斗失败才发布 `terminal`。一个进程的 stdin 必须串行使用。`public_catalog` 可在不生成未来内容的情况下读取静态内容标识，用于初始化冻结词表。

契约含固定难度、观测／动作／自动推进版本、游戏和适配器程序集 SHA-256、交互类型及效果覆盖标记。只有 `decision_protocol:true` 原生创建的 A0–A10 对局允许训练；难度同时写在玩家实体的 `ascension` 上，是模型输入。上游的写命令或调试修改会使句柄失效并将 `training_ready` 设为 false；后续查询不会恢复资格。加载未验证存档也不能冒充训练对局。

`training_ready` 表示该原生运行使用候选协议且未被调试修改，不表示所有游戏内容已穷举验证、效果 AST 已完整注册或模型已具备通关能力。未支持的交互仍明确报错。

### 同一进程里的下一局

一个进程同一时刻只有一局。已有对局时再发 `start_run`、`load_save` 或 `enter_anchor`，引擎先按游戏返回主菜单的方式结束这一局（`RunManager.CleanUp`），再用一个新的模拟器开始下一局：决策序号、状态版本和全部协议状态从头开始，得到的帧与新进程的第一局相同，只有 `episode_id` 不同。

只有游戏在战斗之外等待玩家输入时才能结束一局：停在地图、房间里，或开着奖励菜单、选牌界面都可以，停在这些界面上的游戏代码不再继续。以下情况拒绝，返回 `{"type":"error","code":"run_not_ended"}`，原对局和已发布的决策不受影响：战斗进行中；仍有游戏代码在执行或占着线程（未完成的操作、原生续延、动作队列、水晶球、卡包选择）；这一局在引擎内部出过错（命令抛出异常或协议已报告失败）。被拒绝后要开始新的一局只能换一个进程。

### 摘要锚点入口（`summary-anchor-v1`）

对局摘要重建出的节点边界不是 `start_run` 产生的连续对局，用单独的入口进入，契约里多一个字段 `initialization = "summary-anchor-v1"`（`start_run` 的契约没有这个字段）。这样的帧 `training_ready=true`，但只能作为监督样本，不能当作连续原生对局或 PPO 采样。

- `enter_anchor`：参数 `save`（序列化对局）、`act_floor`、`map_point_type`，可选 `route`（该幕记录的节点类型序列）和 `second_boss`。加载存档后静默重装玩家状态，已持有的遗物不会再次触发取得效果。地图位置：本幕原生地图上恰有一条路径的节点类型与 `route` 一致时，把这条路径走到 `act_floor`；否则退回到同类型、同楼层的代表节点。返回 `anchor_installed`，其中 `map_position` 为 `recorded_route` 或 `representative_point`，前者另带整条路径的坐标。
- `anchor_state`：只读，返回游戏自己序列化的当前对局，用来核对安装结果。
- `anchor_event`：只读，返回当前事件页每个选项被选中时游戏会写进对局历史的条目（标题和变量，用游戏自己的序列化；游戏不记录的选项为 null），以及事件变量此刻的值。游戏的选项按钮在显示时才把事件变量填进标题，这里填在副本上，界面上的选项和帧不变。
- `content_info`：只读，参数 `ids`。返回商店规则读取的静态属性：卡牌的类型、稀有度、是否无色，遗物的稀有度和能否进商店，药水的稀有度。不需要对局。
- `anchor_room`：安装后只能调用一次，进入节点的房间并停在第一个决策。`room.type` 为 `combat`（`encounter`）、`rest_site`、`card_reward`（`cards`，按序列化卡牌给出）、`map`（节点之后的路线选择）、`ancient`（`event` 和 `options`：进入该远古之民事件，把它自己抽出的选项换成记录的选项，顺序不变）、`shop`（`cards`：五个角色卡牌格位各自记录的牌，只取它们的稀有度；`relics`：三个遗物格位记录的遗物）、`event`（`event`：按游戏对这一局的生成进入该事件，不改动任何选项）或 `rewards`（`gold`、`potions`、`relics`、`cards`：一个只含这些奖励的奖励界面，按游戏列出奖励的顺序，不生成任何内容；`gold` 是界面上显示的金币数）。商店的其余库存和全部价格由游戏用存档里的商店随机流照常抽出，引擎不核对也不修正，结果是否等于记录由调用方从帧里判断。`card_reward` 按给定的卡牌原样发放：记录里的候选已经包含玩家当时看到的全部改动，不再让遗物改一遍。
- 之后任何 `enter_room`、`set_player`、`load_save` 等调试命令照常使 `training_ready` 失效。

`anchor_room` 还支持 `treasure`：`relic` 是记录的单件遗物，`gold_roll` 是待核对的开箱原始金币结果（42～52）。入场钩子和宝箱抑制照常执行，未开箱不公开遗物；开箱只条件化该宝箱的一次原始金币抽取，仍推进随机流，再执行原生难度折算、金币修饰和取得效果。调用方须枚举并核对末状态与公开序列，不能把任意指定结果当成历史证据。此约束绑定当前锚点宝箱，不影响普通对局或其它房间。

## 候选与原子执行

`DecisionGate` 核对决策 ID、状态版本、选择修订、候选引用和当前原生合法性。校验通过后先消费句柄，再执行对应闭包；重复请求和过期请求不能再次扣费。非法请求不消费当前有效帧。执行异常隔离当前协议运行，不能重试可能已部分结算的效果。

| 边界 | 候选来源和执行 |
| --- | --- |
| 战斗 | 完整手牌与目标绑定，调用原生 `CanPlayTargeting`；原生出牌队列与结束回合 |
| 药水 | 实际槽位、使用时机、原生可用性和目标过滤；使用及丢弃分别保留，自动触发药水没有主动使用候选 |
| 地图 | `MapTravel.GetTravelablePointsFrom` 和原生 hook，包括实际 Boss／第二 Boss 节点 |
| 事件与远古之民 | 当前公开页所有未锁选项，恢复原生 callback；子选择完成后恢复父 continuation |
| 营火 | 原生可用选项；只有成功使用机会后才允许相应离开；锻造选牌打开即承诺 |
| 商店／假商人 | 真实库存、价格、金币、药水容量、删牌可用性；原生购买；删牌选牌打开即承诺。对商人扔臭药水得到金币，商店照常营业，可以再扔、再买 |
| 宝箱 | 显式打开后才公开遗物；原生投票／获得流程，保留拿取和跳过 |
| 战后／事件奖励 | 全部未领取奖励，完整卡牌候选及原生替代选项，领取顺序由模型决定；离开奖励也是动作 |
| 选牌／卡包 | 原生已过滤卡池和数量约束，逐项 buffered 前缀／完整卡包；取消只在战斗中或无法完成选择时提供 |
| 水晶球 | 全部隐藏中心格 × 两种占卜工具；原生次数、揭示、诅咒和最终奖励结算 |

纯地图显示、免费打开已公开卡牌奖励、装饰音效及布局不制造额外策略步骤。Steam 的桥接按同样的方式发布奖励界面：已生成的卡牌奖励连同它的牌一起给出，`take_card_reward` 由 Mod 点开面板、核对牌未变后提交（`steam-live-v2`，归类规则见 [ARCHITECTURE](../../ARCHITECTURE.md) 的“候选是有结果的动作”）。跳过奖励、结束回合、离开商店保留其原生机会语义；非战斗阶段已打开的选择不提供取消，`selection_cancel = "combat-or-dead-end-v1"`（依据见 [ARCHITECTURE](../../ARCHITECTURE.md)）。未知稳定边界、无合法候选矛盾、引擎错误和真正死亡分别处理，不以强制 proceed 或重新补能量掩盖问题。

## buffered 选择与模型缓存

`SELECT_ONE` 只编辑前缀；`FINISH_SELECTION` 一次提交。模型读取 operation、来源／去向、min/max、remaining_required、已选引用和结束／取消权限。来源未知明确标记 unknown。排序语义不能从通用选择接口证明时保守保存顺序；只有调用方证明唯一无序结果时才合并强制完成。

同一会话的 `base_public_version/action_bank_version` 与固定公开 `decoder_bank` 不变，逐步 `legal.candidates` 决定 mask。每次选择改变决策版本与选择修订。新信息揭示、新会话、模型更新或公开内容变化使缓存失效。

多选逐步进行：每一步只生成当前剩余项，不枚举所有子集；只剩一个合法候选时直接执行，不调用模型。营火锻造等原生可取消的非战斗选择不发布 `CANCEL`，选牌后以 `FINISH_SELECTION` 提交。

上游命令 `select_cards` 也整体拒绝越界、重复和数量不符；`skip_select` 不能清空强制选择；越界卡包不默认为第一个。

## 公开信息

`RunSimulator.PublicState.cs` 用白名单发布玩家资源、完整公开牌堆、当前敌人／召唤物、意图图标与显示伤害、状态层数、遗物、药水、充能球、地图 DAG 和当前阶段选项。卡牌保留内容、费用、升级、类型、关键词、公开基础动态值、附魔及负面修改类型。事件和营火选项导出当前显示模板引用的变量，作为归属该选项的 `displayed_variable`：数值带 `amount`；内容名带 `names`，值是它所指的卡牌、遗物、药水或附魔的内容 ID（名字正好是其中一个内容的标题时才导出）。发布事件帧时先把事件变量填进选项的标题和描述，与游戏的选项按钮在显示时所做的相同；模板取自随引擎加载的英文本地化数据。不输出整个事件内部变量集合。

手牌与充能球保留有公开意义的顺序。牌组、抽牌／弃牌／消耗堆按公开卡牌属性排序，模型按无序多重集合处理，保留实例与重复数量；不输出隐藏原始位置、引擎分配 ID、seed、RNG 或怪物内部移动状态名。路由引用只用于对象关联，不作为嵌入数值。

未打开宝箱不公开潜在遗物。水晶球 11×11 格子的未揭示实体只有坐标、公开形状和 unknown，不能带隐藏物品类别或覆盖范围。小工具揭示一个格，大工具揭示板内裁剪的 3×3，两者均消耗一次；每次实际揭示产生新决策和新编码，无法沿用此前静态缓存。

真实卡牌／物品／事件效果当前明确为 `OPAQUE_RULE`，辅以类型化公开数值和引用（卡牌的附魔与负面修改各带一个显示数值 `enchantment_amount`、`affliction_amount`，规则文本靠它渲染）；这不是完整规则 AST 内容库。模型对 opaque 规则使用游戏描述文本作为程序（见 [ARCHITECTURE](../../ARCHITECTURE.md) 的"规则文本"），因此描述里显示的数值必须公开：遗物和能力导出 `stats`（各自的动态变量基础值），遗物在游戏显示图标计数时另带 `counter`。玩家实体带 `free_travel`，取自地图通行使用的同一个钩子 `Hook.ShouldAllowFreeTravel`。地图的 Boss 节点带 `encounter`，即本幕 Boss 的遭遇 ID（有第二个 Boss 节点时各带各的），游戏从一幕开始就在地图上显示它。静态目录（`public_catalog`）除内容模型和选项外，还列出没有自身模型的奖励与商店条目类型名，以及全部卡牌关键词，供词表冻结时登记。模型的组合 AST 编码已经实现，完整内容注册仍需逐项核对。公开信息清洗也不是任意游戏版本的不可泄漏证明，升级游戏或适配器后需重新验收。

## 结算、里程碑与 headless 桥接

训练协议保留真实 pending continuation；父动作只执行一次，子交互完成后继续结算。奖励使用 `RewardsSet`／原生 synchronizer 的测试选择器入口，直接卡牌选择、领取、hook 与历史记录仍走原生方法。跨幕调用原生 `EnterNextAct`，事件内战斗完成后恢复父事件。

战斗开始时（原生 `CombatBegan`）发布 `encounter_started`，带这场战斗的 `encounter_id`、遭遇内容 ID `encounter`、`act`、`kind` 和是否由事件引发 `from_event`；它随战斗的第一帧到达，只说明玩家此刻面对的是哪个遭遇，不进入 `public`。战斗胜利由真实 `CombatWon` 确认，生成同一 `encounter_id` 的 `encounter_completed`；只有实际幕末 Boss 算幕里程碑，第三幕最终 Boss 生成通关事件。死亡／放弃与失败异常分开。重复查询不会重复交付上次事件；Python 奖励账本还按 encounter 去重，并落实每幕普通战斗奖励上限及 Boss／终局互斥支付。

特殊桥接复用实际规则：水晶球只替换缺失的屏幕显示，格子点击执行原生小游戏；假商人药水补足原 UI 节点触发；宝箱补足原 UI 收到奖励后的获得回调；Jungle Maze／Trial 仅拦截无头环境没有的音效和装饰布局。桥接不重写随机奖励、伤害、成本或效果顺序。原生动作执行器吞入的异常也会转为协议错误。

## 验证

```bash
dotnet build sts2-cli/src/Sts2Headless/Sts2Headless.csproj
dotnet run --project sts2-cli/tests/DecisionProtocolChecks/DecisionProtocolChecks.csproj
dotnet run --project sts2-cli/tests/PendingOperationChecks/PendingOperationChecks.csproj
python -m pytest -q sts2-cli/tests
python -m pytest -q -m "not engine"
```

游戏文件、构建产物和本地依赖不纳入源码版本管理。未来内容变更可能出现新交互；协议对此显式报错，整轮 PPO 会停止并保留诊断。

## Merchant RNG parity

`merchant_potion_prices: "native-potion-price-v1"` means the adapter executes the game's production potion-price roll even in headless TestMode. Each potion repricing consumes the same Shops RNG draw and uses the original floating-point/rounding code. This applies to normal and fake merchants and restocking through `MerchantPotionEntry.CalcCost`.

Native differential regression (same RNG state, production vs headless price and next random value):

```bash
dotnet build sts2-cli/tests/MerchantParityChecks/MerchantParityChecks.csproj -m:1
dotnet sts2-cli/tests/MerchantParityChecks/bin/Debug/net9.0/MerchantParityChecks.dll "$PWD"
```

历史状态重建可在 `start_run` 传入 `acts`（例如 `["UNDERDOCKS", "HIVE", "GLORY"]`）；省略时沿用默认幕。相同 seed 的地图形状不能替代原幕序列核验。

奖励界面沿用原生药水使用时机与目标过滤：可在任意时机使用的药水（如鲜血药水）保留 `USE_POTION`，战斗专用药水不暴露使用动作；丢弃仍经过当前奖励菜单的续接队列。喝药后继续原奖励边界，不能漏掉该合法选择或提前关闭奖励。

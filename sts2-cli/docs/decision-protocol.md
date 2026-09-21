# 引擎决策协议与完整对局

`engine-candidates-v1` 为模型提供同一公开快照上的完整合法候选，并由真实游戏引擎执行。正常训练不调用旧版 `DetectDecisionPoint` 的自动领奖／默认选择分支。Python 训练入口与模块说明见[训练文档](../../docs/TRAINING.md)。

## JSON Lines 入口

准备游戏程序集并构建后，启动 `Sts2Headless.dll`；读取 ready 握手，再逐行发送：

```json
{"cmd":"start_run","character":"Ironclad","seed":"example-run","ascension":10,"decision_protocol":true}
```

返回 `decision_frame`，含 `contract`、`boundary`、`routing`、`public`、`legal.candidates` 和 `events`。从候选中选择引用，原样带回版本：

```json
{"cmd":"execute_candidate","decision_id":"从 routing 读取","state_version":4,"candidate_ref":"c0","selection_revision":0}
```

只有选择会话需要 `selection_revision`；普通动作省略。响应是下一边界。`waiting` 时查询 `{"cmd":"advance_to_boundary"}`，不推测下一动作。一个进程的 stdin 必须串行使用。`public_catalog` 可在不生成未来内容的情况下读取静态内容标识，用于初始化冻结词表。

契约含固定难度、观测／动作／自动推进版本、游戏和适配器程序集 SHA-256、交互类型及效果覆盖标记。只有 `decision_protocol:true` 原生创建的 A10 对局允许训练。旧写命令或调试修改会失效句柄并将 `training_ready` 设为 false；后续查询不会恢复资格。加载未验证存档也不能冒充训练对局。

`training_ready` 表示该原生运行使用候选协议且未被调试修改，不表示所有游戏内容已穷举验证、效果 AST 已完整注册或模型已具备通关能力。未支持的交互仍明确报错。

## 候选与原子执行

`DecisionGate` 核对决策 ID、状态版本、选择修订、候选引用和当前原生合法性。校验通过后先消费句柄，再执行对应闭包；重复请求和过期请求不能再次扣费。非法请求不消费当前有效帧。执行异常隔离当前协议运行，不能重试可能已部分结算的效果。

| 边界 | 候选来源和执行 |
| --- | --- |
| 战斗 | 完整手牌与目标绑定，调用原生 `CanPlayTargeting`；原生出牌队列与结束回合 |
| 药水 | 实际槽位、使用时机、原生可用性和目标过滤；使用及丢弃分别保留，自动触发药水没有主动使用候选 |
| 地图 | `MapTravel.GetTravelablePointsFrom` 和原生 hook，包括实际 Boss／第二 Boss 节点 |
| 事件与远古之民 | 当前公开页所有未锁选项，恢复原生 callback；子选择完成后恢复父 continuation |
| 营火 | 原生可用选项；只有成功使用机会后才允许相应离开，取消锻造不会消耗机会 |
| 商店／假商人 | 真实库存、价格、金币、药水容量、删牌可用性；原生购买和可取消删牌 |
| 宝箱 | 显式打开后才公开遗物；原生投票／获得流程，保留拿取和跳过 |
| 战后／事件奖励 | 全部未领取奖励，完整卡牌候选及原生替代选项，领取顺序由模型决定；离开奖励也是动作 |
| 选牌／卡包 | 原生已过滤卡池和数量约束，逐项 buffered 前缀／完整卡包，原生取消权限 |
| 水晶球 | 全部隐藏中心格 × 两种占卜工具；原生次数、揭示、诅咒和最终奖励结算 |

纯地图显示、免费打开已公开卡牌奖励、装饰音效及布局不制造额外策略步骤。跳过奖励、结束回合、离开商店以及取消已进入的选择保留其原生机会语义。未知稳定边界、无合法候选矛盾、引擎错误和真正死亡分别处理，不以强制 proceed 或重新补能量掩盖问题。

## buffered 选择与模型缓存

`SELECT_ONE` 只编辑前缀；`FINISH_SELECTION` 一次提交。模型读取 operation、来源／去向、min/max、remaining_required、已选引用和结束／取消权限。来源未知明确标记 unknown。排序语义不能从通用选择接口证明时保守保存顺序；只有调用方证明唯一无序结果时才合并强制完成。

同一会话的 `base_public_version/action_bank_version` 与固定公开 `decoder_bank` 不变，逐步 `legal.candidates` 决定 mask。每次选择改变决策版本与选择修订。新信息揭示、新会话、模型更新或公开内容变化使缓存失效。

不可取消的“10 选 5”已验证一次 Transformer、五次 GRU、唯一 FINISH 零次模型调用；运行时只生成当前剩余项，不枚举所有子集。真实营火 Smith 使用原生可取消、手动确认选项：选满一张后 FINISH 与 CANCEL 都合法，因此确认仍是模型分支。不能把这个原生取消权限删掉来制造强制提交。

旧 `select_cards` 也整体拒绝越界、重复和数量不符；`skip_select` 不能清空强制选择；越界卡包不默认为第一个。

## 公开信息

`RunSimulator.PublicState.cs` 用白名单发布玩家资源、完整公开牌堆、当前敌人／召唤物、意图图标与显示伤害、状态层数、遗物、药水、充能球、地图 DAG 和当前阶段选项。卡牌保留内容、费用、升级、类型、关键词、公开基础动态值、附魔及负面修改类型。事件选项只导出当前显示模板引用的数值，不输出整个事件内部变量集合。

手牌与充能球保留有公开意义的顺序。牌组、抽牌／弃牌／消耗堆按公开卡牌属性排序，模型按无序多重集合处理，保留实例与重复数量；不输出隐藏原始位置、引擎分配 ID、seed、RNG 或怪物内部移动状态名。路由引用只用于对象关联，不作为嵌入数值。

未打开宝箱不公开潜在遗物。水晶球 11×11 格子的未揭示实体只有坐标、公开形状和 unknown，不能带隐藏物品类别或覆盖范围。小工具揭示一个格，大工具揭示板内裁剪的 3×3，两者均消耗一次；每次实际揭示产生新决策和新编码，无法沿用此前静态缓存。

真实卡牌／物品／事件效果当前明确为 `OPAQUE_RULE`，辅以类型化公开数值和引用；这不是完整规则 AST 内容库。模型的组合 AST 编码已经实现，完整内容注册仍需逐项核对。公开信息清洗也不是任意游戏版本的不可泄漏证明，升级游戏或适配器后需重新验收。

## 结算、里程碑与 headless 桥接

训练协议保留真实 pending continuation；父动作只执行一次，子交互完成后继续结算。奖励使用 `RewardsSet`／原生 synchronizer 的测试选择器入口，直接卡牌选择、领取、hook 与历史记录仍走原生方法。跨幕调用原生 `EnterNextAct`，事件内战斗完成后恢复父事件。

战斗胜利由真实 `CombatWon` 确认，生成唯一 encounter 事件；只有实际幕末 Boss 算幕里程碑，第三幕最终 Boss 生成通关事件。死亡／放弃与失败异常分开。重复查询不会重复交付上次事件；Python 奖励账本还按 encounter 去重，并落实每幕普通战斗奖励上限及 Boss／终局互斥支付。

特殊桥接复用实际规则：水晶球只替换缺失的屏幕显示，格子点击执行原生小游戏；假商人药水补足原 UI 节点触发；宝箱补足原 UI 收到奖励后的获得回调；Jungle Maze／Trial 仅拦截无头环境没有的音效和装饰布局。桥接不重写随机奖励、伤害、成本或效果顺序。原生动作执行器吞入的异常也会转为协议错误。

## 验证

```bash
dotnet build sts2-cli/src/Sts2Headless/Sts2Headless.csproj
dotnet run --project sts2-cli/tests/DecisionProtocolChecks/DecisionProtocolChecks.csproj
dotnet run --project sts2-cli/tests/PendingOperationChecks/PendingOperationChecks.csproj
python -m pytest -q sts2-cli/tests
python -m pytest -q tests -m 'not cuda'
```

2026-09-20 至 21 已运行：

- CLI 全套 118 项通过（有既有 slow 标记警告）；模型／协议测试包括 25 局原生五角色 A10 回归，全部正常结束。
- 原生取消、奖励／宝箱、假商人药水、水晶球完整交互与隐藏字段检查，事件展示桥接，过期句柄与恰好一次执行。
- 三幕高生命集成夹具覆盖三个 Boss、跨幕和最终胜利，奖励去重正确。该夹具明确禁止作为训练数据；它不构成模型通关证据。
- 小模型真实五角色采样、BC 代码验证、两遍 PPO、保存与恢复训练；旧策略重放误差 0。正式 1B 模型另有 GPU 短序列训练检查。
- 无游戏依赖的选择核心检查穷举 30,240 条有序轨迹、252 个子集，覆盖 STOP、唯一结果、重复／越界和执行失败；异步核心检查覆盖嵌套、超时和 1,600 个并发回调。

使用本机已有 headless 适配游戏 DLL，SHA-256 为 `14b57dbaee1b58cb919625a662831054fd714b3b75b13b7449eac0e95b98a3e4`，不是未修改的 Steam 原始 DLL。游戏文件、构建产物、本地依赖及实测轨迹不纳入源码版本管理。未来内容变更可能出现新交互；协议对此显式报错，整轮 PPO 会停止并保留诊断。

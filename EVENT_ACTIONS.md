# 问号事件合法动作逐项核对表

状态：**固定版本静态核对稿，所有人工与运行验收方框尚未勾选。** 返回 [动作总表](ACTION_CATALOG.md)。

本表用于定位分支、合法条件和后续流程。供模型读取的结构化后果采用 [动作／效果／目标编码方案](ACTION_ENCODING.md)，不直接将这些按钮标题或描述文本当作动作 embedding。

## 1. 覆盖与证据

本表覆盖本机 `v0.111.0 / 41cef1ea` 的 `ModelDb.AllEvents`：**57 个去重后的事件模型**。来源为四个场景事件池与 Shared 事件池；其中 `WarHistorianRepy` 由任务触发，不能当普通随机问号事件抽取。不是把中文表的 63 个前缀都当 63 个可遇事件。另有 8 位远古之民及建筑师，见 [独立清单](ANCIENT_ACTIONS.md)。DLL 与 CLI 固定值见 [动作总表第 1 节](ACTION_CATALOG.md#1-范围版本与读法)。

每项同时核对了 DLL 中 `MegaCrit.Sts2.Core.Models.Events.<Class>` 的选项生成与后续处理；下表结果栏保留中文模板中的 `{变量}`，最终数值从当前公开选项取得。文本与实现冲突时，在该事件规则或“非候选”栏明确列出。**这些表是动作规格，不是可以在 Python 中照抄的合法性过滤代码。**

普通页面的定义是：当前事件／阶段活跃、没有阻塞子交互、玩家未终局，且 `CurrentOptions` 中该对象未锁定、未被本次选择消费、真实交互允许执行。合法药水操作还须与页面选项合并。文档列的是各阶段可能出现的动作并集，不能一次性把所有页面的动作提供给模型。

统一记号：事件表中的按钮均映射为 `CHOOSE_EVENT_OPTION(event_ref,stage,option_ref)`；额外子流程的 `SELECT_ONE`、`FINISH_SELECTION` 等按 [公共选择会话](ACTION_CATALOG.md#6-公共子选择每一种操作都保留来源和去向)。没有明确可跳过／取消来源时，不追加 SKIP／CANCEL；父按钮没有显式空目标锁定时，也不由适配器臆造锁定。

**每个事件均须核对三个层次：**①事件能否生成；②当前页哪些动作合法；③执行后还有哪些玩家子选择。前两个层次不能互相代替。可能致死不等于不合法；支付金币、失去金币、受到伤害、失去最大生命各自保留原规则。

## 2. 索引

| 编号 | 事件 | 类名 | 注册池 |
| --- | --- | --- | --- |
| E01 | [深渊浴场](#e01) | `AbyssalBaths` | Underdocks |
| E02 | [熔合者](#e02) | `Amalgamator` | Hive |
| E03 | [混沌芳香](#e03) | `AromaOfChaos` | Overgrowth |
| E04 | [战痕累累的训练假人](#e04) | `BattlewornDummy` | Glory |
| E05 | [脑蛭](#e05) | `BrainLeech` | Shared |
| E06 | [害虫杀手](#e06) | `Bugslayer` | Hive |
| E07 | [多尼斯异鸟巢](#e07) | `ByrdonisNest` | Overgrowth |
| E08 | [色彩哲学家](#e08) | `ColorfulPhilosophers` | Hive |
| E09 | [巨大花卉](#e09) | `ColossalFlower` | Hive |
| E10 | [水晶球](#e10) | `CrystalSphere` | Shared |
| E11 | [茂密的植被](#e11) | `DenseVegetation` | Overgrowth |
| E12 | [玩偶室](#e12) | `DollRoom` | Shared |
| E13 | [光与暗的门扉](#e13) | `DoorsOfLightAndDark` | Underdocks |
| E14 | [淹水灯塔](#e14) | `DrowningBeacon` | Underdocks |
| E15 | [无尽传送带](#e15) | `EndlessConveyor` | Underdocks |
| E16 | [商人？？？](#e16) | `FakeMerchant` | Shared |
| E17 | [人形洞穴之地](#e17) | `FieldOfManSizedHoles` | Hive |
| E18 | [遗忘之墓](#e18) | `GraveOfTheForgotten` | Glory |
| E19 | [蘑菇饥渴](#e19) | `HungryForMushrooms` | Glory |
| E20 | [被寄生的自动机械](#e20) | `InfestedAutomaton` | Hive |
| E21 | [丛林迷宫奇遇](#e21) | `JungleMazeAdventure` | Overgrowth |
| E22 | [迷失鬼火](#e22) | `LostWisp` | Hive |
| E23 | [冷光合唱团](#e23) | `LuminousChoir` | Overgrowth |
| E24 | [变形灵林谷](#e24) | `MorphicGrove` | Overgrowth |
| E25 | [药水快递员](#e25) | `PotionCourier` | Shared |
| E26 | [重拳出击](#e26) | `PunchOff` | Underdocks |
| E27 | [长者兰伟德](#e27) | `RanwidTheElder` | Shared |
| E28 | [镜中倒影  影倒中镜](#e28) | `Reflections` | Glory |
| E29 | [遗物交换商](#e29) | `RelicTrader` | Shared |
| E30 | [满屋芝士](#e30) | `RoomFullOfCheese` | Shared |
| E31 | [圆桌茶会](#e31) | `RoundTeaParty` | Glory |
| E32 | [蓝宝石种子](#e32) | `SapphireSeed` | Overgrowth |
| E33 | [自助指南](#e33) | `SelfHelpBook` | Shared |
| E34 | [滑脚木桥](#e34) | `SlipperyBridge` | Shared |
| E35 | [螺旋漩涡](#e35) | `SpiralingWhirlpool` | Underdocks |
| E36 | [灵魂嫁接者](#e36) | `SpiritGrafter` | Hive |
| E37 | [永恒之石](#e37) | `StoneOfAllTime` | Shared |
| E38 | [沉没雕像](#e38) | `SunkenStatue` | Overgrowth / Underdocks |
| E39 | [淹水金库](#e39) | `SunkenTreasury` | Underdocks |
| E40 | [共生体](#e40) | `Symbiote` | Shared |
| E41 | [真理石板](#e41) | `TabletOfTruth` | Overgrowth |
| E42 | [茶艺大师](#e42) | `TeaMaster` | Shared |
| E43 | [药水的未来？](#e43) | `TheFutureOfPotions` | Shared |
| E44 | [灯火钥匙](#e44) | `TheLanternKey` | Hive |
| E45 | [传说是真的](#e45) | `TheLegendsWereTrue` | Shared |
| E46 | [这个还是那个？](#e46) | `ThisOrThat` | Shared |
| E47 | [打造时间](#e47) | `TinkerTime` | Glory |
| E48 | [垃圾堆](#e48) | `TrashHeap` | Underdocks |
| E49 | [审判](#e49) | `Trial` | Glory |
| E50 | [无休之处](#e50) | `UnrestSite` | Overgrowth |
| E51 | [战史学家 付袭](#e51) | `WarHistorianRepy` | Shared |
| E52 | [水漫缮写室](#e52) | `WaterloggedScriptorium` | Underdocks |
| E53 | [欢迎来到旺购百货](#e53) | `WelcomeToWongos` | Shared |
| E54 | [泉水](#e54) | `Wellspring` | Overgrowth |
| E55 | [低语空谷](#e55) | `WhisperingHollow` | Overgrowth |
| E56 | [木雕](#e56) | `WoodCarvings` | Overgrowth |
| E57 | [修禅织网者](#e57) | `ZenWeaver` | Hive |

## 3. 逐事件、逐阶段清单

共同续流程：任何获得遗物动作都可能触发遗物自身选牌／附魔／卡包子交互；任何奖励动作都必须保留其实际 alternatives、跳过和领取顺序。重复列出这些通用动作时仍使用同一协议，不另外维护事件专用模型脚本。

<a id="e01"></a>

### E01 深渊浴场（`ABYSSAL_BATHS`）

**生成范围：** Underdocks。无额外 IsAllowed 限制。

**已核对规则：** INITIAL 二选一；IMMERSE 后只有 LINGER／EXIT_BATHS。每次先增加最大生命再受伤；初始 Damage=3，每次结算后 +1，MaxHp=2。LINGER 的文案编号最多到 9，但动作仍可继续，不能按页名或次数上限强制退出。死亡警告仍保留 LINGER。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E01-01 | `ALL` | `EXIT_BATHS` · 离开浴场 | 按该事件处理器结算；见本项规则。 |
| [ ] E01-02 | `ALL` | `LINGER` · 沉溺 | 获得{MaxHp}点最大生命值。受到{Damage}点伤害。 |
| [ ] E01-03 | `INITIAL` | `ABSTAIN` · 敬而远之 | 回复{Heal}点生命。 |
| [ ] E01-04 | `INITIAL` | `IMMERSE` · 投身其中 | 获得{MaxHp}点最大生命值。受到{Damage}点伤害。 |

- [ ] E01：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e02"></a>

### E02 熔合者（`AMALGAMATOR`）

**生成范围：** Hive。至少各有 2 张可移除、Basic 稀有度且带 Strike／Defend 标签的牌。

**已核对规则：** 融合并非随机拿走两张，也不能选择非基础的同标签牌。子选择从对应合法牌中恰选 2 张，移除后加入 UltimateStrike／UltimateDefend。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E02-01 | `INITIAL` | `COMBINE_DEFENDS` · 融合防御 | 移除2张防御。并将一张{Card2}添加至你的牌组。 |
| [ ] E02-02 | `INITIAL` | `COMBINE_STRIKES` · 融合打击 | 移除2张打击。并将一张{Card1}添加至你的牌组。 |
| [ ] E02-03 | COMBINE_STRIKES 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | Basic＋Strike＋IsRemovable；无重复 |
| [ ] E02-04 | COMBINE_DEFENDS 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | Basic＋Defend＋IsRemovable；无重复 |

- [ ] E02：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e03"></a>

### E03 混沌芳香（`AROMA_OF_CHAOS`）

**生成范围：** Overgrowth。无额外 IsAllowed 限制。

**已核对规则：** 变化与升级分别使用原选择器过滤。无可升级牌不等于适配器可私自把父按钮删除；原实现可返回空结果后结束。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E03-01 | `INITIAL` | `LET_GO` · 放任自流 | 变化你牌组中的一张牌。 |
| [ ] E03-02 | `INITIAL` | `MAINTAIN_CONTROL` · 维持理智 | 升级你牌组中的一张牌。 |
| [ ] E03-03 | LET_GO 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=transform；IsTransformable |
| [ ] E03-04 | MAINTAIN_CONTROL 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=upgrade；IsUpgradable |

- [ ] E03：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e04"></a>

### E04 战痕累累的训练假人（`BATTLEWORN_DUMMY`）

**生成范围：** Glory。Glory 事件池；没有额外 IsAllowed 条件。

**已核对规则：** 三个档位分别进入三个真实 encounter。战斗结束调用 Resume：超时与胜利不同；胜利才按档位给予药水／随机升级 2 张／遗物。随机升级不是选牌。事件战斗恢复不应被 CLI 普通战后回地图逻辑吞掉。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E04-01 | `INITIAL` | `SETTING_1` · 第1档 | 与一个{Setting1Hp}点生命值的假人战斗。随机获取1瓶药水。 |
| [ ] E04-02 | `INITIAL` | `SETTING_2` · 第2档 | 与一个{Setting2Hp}点生命值的假人战斗。随机升级2张牌。 |
| [ ] E04-03 | `INITIAL` | `SETTING_3` · 第3档 | 与一个{Setting3Hp}点生命值的假人战斗。获取一件随机遗物。 |
| [ ] E04-04 | 任一 SETTING_n 后 | `公共战斗动作 C-01/C-02/P/Q` | 含三回合挑战的真实规则；结束恢复事件 |
| [ ] E04-05 | 胜利奖励 | `公共奖励动作 R/T` | 药水槽位、遗物获得子效果依原规则 |

- [ ] E04：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e05"></a>

### E05 脑蛭（`BRAIN_LEECH`）

**生成范围：** Shared。CurrentActIndex < 2（第 1、2 幕）。

**已核对规则：** SHARE_KNOWLEDGE 在本 DLL 中从 5 张生成牌中恰选 1 张，明确 Cancelable=false；不能视作可跳过卡牌奖励。RIP 先受伤，再进入无色卡牌奖励组。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E05-01 | `INITIAL` | `RIP` · 把它扯下来 | 失去{RipHpLoss}点生命值。获得一次无色卡牌奖励。 |
| [ ] E05-02 | `INITIAL` | `SHARE_KNOWLEDGE` · 分享知识 | 从{FromCardChoiceCount}张随机牌中选择{CardChoiceCount}张加入你的牌组。 |
| [ ] E05-03 | SHARE_KNOWLEDGE 网格 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=obtain；5 选 1，无 SKIP/CANCEL |
| [ ] E05-04 | RIP 奖励 | `TAKE_CARD_REWARD／真实 alternatives` | 奖励一般含跳过；以该奖励对象为准 |

- [ ] E05：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e06"></a>

### E06 害虫杀手（`BUGSLAYER`）

**生成范围：** Hive。无额外 IsAllowed 限制。

**已核对规则：** 两条动作分别直接加入 Exterminate／Squash；没有再从任意卡牌池选一张的子动作。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E06-01 | `INITIAL` | `EXTERMINATION` · 学习杀灭的技巧 | 将一张{Card1}添加至你的牌组。 |
| [ ] E06-02 | `INITIAL` | `SQUASH` · 学习压扁的技巧 | 将一张{Card2}添加至你的牌组 |

- [ ] E06：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e07"></a>

### E07 多尼斯异鸟巢（`BYRDONIS_NEST`）

**生成范围：** Overgrowth。玩家没有事件宠物 HasEventPet。

**已核对规则：** EAT 增最大生命；TAKE 加入 ByrdonisEgg。后续营火孵化是另一房间的动作，不是本页新增候选。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E07-01 | `INITIAL` | `EAT` · 吃掉这颗蛋 | 获得{MaxHp}点最大生命值。 |
| [ ] E07-02 | `INITIAL` | `TAKE` · 带走这颗蛋 | 将一张{Card}加入你的牌组。 |

- [ ] E07：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e08"></a>

### E08 色彩哲学家（`COLORFUL_PHILOSOPHERS`）

**生成范围：** Hive。已解锁角色卡池数量 >1。

**已核对规则：** 从已解锁且不是当前角色的卡池构建选项，随机删至最多 3 个；不是每次都有五种颜色。选一个卡池后给普通／罕见／稀有三个独立奖励组，每组展示 3 张并各选一或使用其 alternatives。EQUALITY 仅有遗留文本，当前实现不生成。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E08-01 | `INITIAL` | `DEFECT` · 蓝色 | 获得{Cards}张故障机器人的卡牌。 |
| [ ] E08-02 | `INITIAL` | `IRONCLAD` · 红色 | 获得{Cards}张铁甲战士的卡牌。 |
| [ ] E08-03 | `INITIAL` | `NECROBINDER` · 粉色 | 获得{Cards}张亡灵契约师的卡牌。 |
| [ ] E08-04 | `INITIAL` | `REGENT` · 橙色 | 获得{Cards}张储君的卡牌。 |
| [ ] E08-05 | `INITIAL` | `SILENT` · 绿色 | 获得{Cards}张静默猎手的卡牌。 |
| [ ] E08-06 | 选颜色后：普通奖励 | `TAKE_CARD_REWARD(group_common,card_ref)／alternatives` | 在实际打开后公开 |
| [ ] E08-07 | 选颜色后：罕见奖励 | `TAKE_CARD_REWARD(group_uncommon,card_ref)／alternatives` | 与前一奖励分别完成 |
| [ ] E08-08 | 选颜色后：稀有奖励 | `TAKE_CARD_REWARD(group_rare,card_ref)／alternatives` | 不把三组压成一次三选一 |

非候选／文本差异：

- `COLORFUL_PHILOSOPHERS.pages.INITIAL.options.EQUALITY`：该 DLL 的 GenerateInitialOptions 不生成；遗留文本。

- [ ] E08：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e09"></a>

### E09 巨大花卉（`COLOSSAL_FLOWER`）

**生成范围：** Hive。当前生命 ≥19。

**已核对规则：** 初页：拿 35 金币或承受 5 伤害深入；第二页：拿 75 或承受 6 伤害深入；第三页：拿 135 或承受 7 伤害拿 PollinousCore。三个页面的候选不能混在一帧；过去损失不可退款。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E09-01 | `INITIAL` | `EXTRACT_CURRENT_PRIZE_1` · 采集花蜜 | 获得{Prize1}金币。 |
| [ ] E09-02 | `INITIAL` | `REACH_DEEPER_1` · 深入探索 | 深入。失去5点生命值 |
| [ ] E09-03 | `REACH_DEEPER_1` | `EXTRACT_CURRENT_PRIZE_2` · 采集花蜜 | 获得{Prize2}金币。 |
| [ ] E09-04 | `REACH_DEEPER_1` | `REACH_DEEPER_2` · 深入探索 | 再深入一点。失去6点生命值。 |
| [ ] E09-05 | `REACH_DEEPER_2` | `EXTRACT_INSTEAD` · 采集花蜜 | 获得{Prize3}金币。 |
| [ ] E09-06 | `REACH_DEEPER_2` | `POLLINOUS_CORE` · 抵达核心 | 失去7点生命值。 获得花粉核心。 |

- [ ] E09：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e10"></a>

### E10 水晶球（`CRYSTAL_SPHERE`）

**生成范围：** Shared。第 2、3 幕，玩家金币 ≥100。

**已核对规则：** UNCOVER_FUTURE 支付 CalculateVars 已生成的公开费用，3 次占卜；PAYMENT_PLAN 加入 Debt，6 次占卜。初始只有这两个按钮，无免费离开。具体图板合法集及桥接见下文专节。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E10-01 | `INITIAL` | `PAYMENT_PLAN` · 分期付款 | 获得一张{CurseTitle}。占卜{PaymentPlanCount}次。 |
| [ ] E10-02 | `INITIAL` | `UNCOVER_FUTURE` · 揭幕未来 | 支付{UncoverFutureCost}金币。占卜{UncoverFutureProphesizeCount}次。 |
| [ ] E10-03 | 小游戏：小幅 | `DIVINE(small,cell_ref)` | 仍隐藏的可点击中心格；消耗 1 次，清除该格 |
| [ ] E10-04 | 小游戏：大幅 | `DIVINE(big,cell_ref)` | 仍隐藏的可点击中心格；消耗 1 次，清除边界内 3×3 |
| [ ] E10-05 | 次数归零后 | `奖励／选择子流程 → 自动完成` | 不能提供任意提前 STOP；不调用 ForceMinigameEnd 获利退出 |

- [ ] E10：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e11"></a>

### E11 茂密的植被（`DENSE_VEGETATION`）

**生成范围：** Overgrowth。单人无额外限制；联机另有生命条件，首版不应用联机条件过滤单人。

**已核对规则：** TRUDGE_ON 受伤后得金币。REST 先治疗，再出现唯一 FIGHT；已选择休息即承诺后续战斗，FIGHT 按强制动作推进，不提供逃离。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E11-01 | `INITIAL` | `REST` · 休息 | 回复{Heal}点生命。进入战斗。 |
| [ ] E11-02 | `INITIAL` | `TRUDGE_ON` · 坚持跋涉 | 获得{Gold} 金币。失去{HpLoss}点生命。 |
| [ ] E11-03 | `REST` | `FIGHT` · 战！ | 按该事件处理器结算；见本项规则。 |
| [ ] E11-04 | REST 后 FIGHT | `CHOOSE_EVENT_OPTION(FIGHT) → 公共战斗` | 自动执行唯一事件分支；仍须合并实际可用药水候选后计数 |

- [ ] E11：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e12"></a>

### E12 玩偶室（`DOLL_ROOM`）

**生成范围：** Shared。仅第 2 幕。

**已核对规则：** RANDOM 随机得到一个玩偶。TAKE_SOME_TIME 先失去 5 生命，然后从抽出的 2 个玩偶中选 1；EXAMINE 先失去 15，随后 3 选 1。第二页动态标题来自遗物，不一定有普通 option title key。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E12-01 | `INITIAL` | `EXAMINE` · 仔细检查然后挑选最好的那个 | 失去{ExamineHpLoss}点生命。从3尊玩偶遗物中选择1尊带走。 |
| [ ] E12-02 | `INITIAL` | `RANDOM` · 随机拿走一尊 | 获取一尊玩偶遗物。 |
| [ ] E12-03 | `INITIAL` | `TAKE_SOME_TIME` · 花点时间慢慢来 | 失去{TakeTimeHpLoss}点生命。从2尊玩偶遗物中选择1尊带走。 |
| [ ] E12-04 | TAKE_SOME_TIME 第二页 | `CHOOSE_EVENT_OPTION(doll_ref)` | 只列实际抽中 2 件中的每件 |
| [ ] E12-05 | EXAMINE 第二页 | `CHOOSE_EVENT_OPTION(doll_ref)` | DaughterOfTheWind／MrStruggles／BingBong 每件一个；得到后结束 |

非候选／文本差异：

- `DOLL_ROOM.pages.TAKE.options.TAKE`：动态玩偶选择的通用描述，实际候选见子流程。

- [ ] E12：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e13"></a>

### E13 光与暗的门扉（`DOORS_OF_LIGHT_AND_DARK`）

**生成范围：** Underdocks。无额外 IsAllowed 限制。

**已核对规则：** LIGHT 是随机升级，无选哪张；DARK 才是玩家选择移除 1 张。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E13-01 | `INITIAL` | `DARK` · 暗之门 | 从你的牌组中选择1张牌移除。 |
| [ ] E13-02 | `INITIAL` | `LIGHT` · 光之门 | 随机升级{Cards}张牌。 |
| [ ] E13-03 | DARK 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=remove；IsRemovable |

- [ ] E13：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e14"></a>

### E14 淹水灯塔（`DROWNING_BEACON`）

**生成范围：** Underdocks。无额外 IsAllowed 限制。

**已核对规则：** BOTTLE 进入 GlowwaterPotion 奖励，不能绕过满槽处理。CLIMB 降最大生命后获得 FresnelLens；是最大生命代价，不是普通受伤。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E14-01 | `INITIAL` | `BOTTLE` · 装瓶 | 获得一瓶{Potion}。 |
| [ ] E14-02 | `INITIAL` | `CLIMB` · 攀爬 | 获得{Relic}。失去{HpLoss}点最大生命。 |
| [ ] E14-03 | BOTTLE 奖励 | `CLAIM_REWARD(potion_ref)／合法跳过／槽位操作` | 复用奖励层，不默默覆盖已有药水 |

- [ ] E14：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e15"></a>

### E15 无尽传送带（`ENDLESS_CONVEYOR`）

**生成范围：** Underdocks。金币 ≥120。

**已核对规则：** 每帧只展示当前一种菜。初页的另一选项是 OBSERVE_CHEF；吃一次后另一选项变为 LEAVE。当前菜可拿的源码判断统一是 Gold≥40，连免付费的 GOLDEN_FYSH 也经过此门槛，适配器不能自行修正。除金色异鱼外每次花 40；金色异鱼得 75。每次吃完真实重抽后再决策，不能查询候选时重新 RollDish。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E15-01 | `ALL` | `CAVIAR` · 从传送带上拿份鱼子酱 | 支付{Gold}金币。获得{CaviarMaxHp}点最大生命。接着吃！ |
| [ ] E15-02 | `ALL` | `CLAM_ROLL` · 从传送带上拿份蚌肉卷 | 支付{Gold}金币。回复{ClamRollHeal}点生命值。接着吃！ |
| [ ] E15-03 | `ALL` | `FRIED_EEL` · 从传送带上拿份炸鳗鱼 | 支付{Gold}金币。将一张随机无色牌添加到你的牌组。接着吃！ |
| [ ] E15-04 | `ALL` | `GOLDEN_FYSH` · 从传送带上拿份金色异鱼 | 幸运头奖！获得{GoldenFyshGold}金币。 |
| [ ] E15-05 | `ALL` | `JELLY_LIVER` · 从传送带上拿份果冻肝 | 支付{Gold}金币。变化一张牌。接着吃！ |
| [ ] E15-06 | `ALL` | `SEAPUNK_SALAD` · 从传送带上拿份海洋混混沙拉 | 支付{Gold}金币。将一张疯狂进食添加到你的牌组。 接着吃！ |
| [ ] E15-07 | `ALL` | `SPICY_SNAPPY` · 从传送带上拿份香辣薄脆 | 支付{Gold}金币。 随机升级一张牌。接着吃！ |
| [ ] E15-08 | `ALL` | `SUSPICIOUS_CONDIMENT` · 从传送带上拿份可疑调味料 | 支付{Gold}金币。随机获得一瓶药水。接着吃！ |
| [ ] E15-09 | `GRAB_SOMETHING_OFF_THE_BELT` | `LEAVE` · 离开 | 按该事件处理器结算；见本项规则。 |
| [ ] E15-10 | `INITIAL` | `OBSERVE_CHEF` · 观察主厨 | 从牌组中随机升级一张牌。 |
| [ ] E15-11 | JELLY_LIVER 结算 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | 选择变化 1 张，完成后才抽下一道菜 |
| [ ] E15-12 | SUSPICIOUS_CONDIMENT 结算 | `公共药水奖励动作` | 结束奖励再恢复传送带 |

非候选／文本差异：

- `ENDLESS_CONVEYOR.pages.ALL.options.LOCKED`：锁定占位，不进入合法候选。你还能吃得下更多，但是兜里没金币了。

- [ ] E15：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e16"></a>

### E16 商人？？？（`FAKE_MERCHANT`）

**生成范围：** Shared。仅单人；第 2、3 幕；金币 ≥100 或有 FoulPotion。

**已核对规则：** Custom 布局，GenerateInitialOptions 故意返回空列表。随机 6 件假遗物库存，逐件购买；开关库存是 UI。污浊药水投商人后进入战斗，奖励含地毯及未购买的库存。当前 CLI 对空事件选项直接跳过，本事件因此必须新增桥接。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E16-01 | 库存 | `BUY_RELIC(item_ref)` | 每个有库存且能按真实价格购买的商品各一个 |
| [ ] E16-02 | 房间 | `USE_POTION(foul_potion_ref,merchant_ref)` | 合法时关闭库存、投掷，消耗该瓶药水并触发战斗 |
| [ ] E16-03 | 房间 | `DISCARD_POTION(potion_ref)` | 真实允许时，保留和投掷的差别 |
| [ ] E16-04 | 房间 | `LEAVE_EVENT` | 原商人退出流程，放弃购物／战斗机会 |
| [ ] E16-05 | 投掷后 | `公共战斗 → 奖励动作` | 不能把投掷简化成普通伤害或只有丢药 |

- [ ] E16：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e17"></a>

### E17 人形洞穴之地（`FIELD_OF_MAN_SIZED_HOLES`）

**生成范围：** Hive。至少一张牌符合 PerfectFit.CanEnchant。

**已核对规则：** RESIST 选择移除 Cards 张（本版 2），之后加入 Normality；ENTER_YOUR_HOLE 选择 1 张附魔。不是两条都只有一次按钮选择。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E17-01 | `INITIAL` | `ENTER_YOUR_HOLE` · 进入你的洞 | 为一张牌附魔：{Enchantment}。 |
| [ ] E17-02 | `INITIAL` | `RESIST` · 抵抗诱惑 | 从你的牌组中移除{Cards}张牌。将一张{ResistCurse}添加到你的牌组中。 |
| [ ] E17-03 | RESIST 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | operation=remove；提交后加入诅咒 |
| [ ] E17-04 | ENTER_YOUR_HOLE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=enchant；PerfectFit.CanEnchant |

- [ ] E17：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e18"></a>

### E18 遗忘之墓（`GRAVE_OF_THE_FORGOTTEN`）

**生成范围：** Glory。HasEnchantableCards，即存在 SoulsPower 可附魔牌。

**已核对规则：** CONFRONT 无目标时锁定；接受诅咒 Decay 在选牌之前。SoulsPower 要求牌本身带 Exhaust 且满足附魔基础限制；不能用“任意卡牌”代替。ACCEPT 直接获得 ForgottenSoul。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E18-01 | `INITIAL` | `ACCEPT` · 接受这颗{Relic} | 获得{Relic}。 |
| [ ] E18-02 | `INITIAL` | `CONFRONT` · 让它直面真相 | 将一张{Curse}到你的牌组。 为一张带有消耗的牌附魔：{Enchantment}。 |
| [ ] E18-03 | CONFRONT 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | SoulsPower.CanEnchant；添加诅咒不重做 |

非候选／文本差异：

- `GRAVE_OF_THE_FORGOTTEN.pages.INITIAL.options.CONFRONT_LOCKED`：锁定占位，不进入合法候选。你没有可被附魔的带有消耗的卡牌。

- [ ] E18：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e19"></a>

### E19 蘑菇饥渴（`HUNGRY_FOR_MUSHROOMS`）

**生成范围：** Glory。Glory 事件池，无额外 IsAllowed 条件。

**已核对规则：** 两个选项实际由 RelicOption 创建，描述来自遗物。BigMushroom 获得时加 20 最大生命，并有首回合少抽牌的持续代价；FragrantMushroom 获得时受 15 伤害、随机升级 2 张。没有任意选升级目标。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E19-01 | `INITIAL` | `BIG_MUSHROOM` · 大蘑菇 | 获得对应遗物，含其获得时效果；见本项规则。 |
| [ ] E19-02 | `INITIAL` | `FRAGRANT_MUSHROOM` · 芳香蘑菇 | 获得对应遗物，含其获得时效果；见本项规则。 |

- [ ] E19：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e20"></a>

### E20 被寄生的自动机械（`INFESTED_AUTOMATON`）

**生成范围：** Hive。无额外 IsAllowed 限制。

**已核对规则：** STUDY 随机给能力牌；TOUCH_CORE 随机给 Canonical 能量费用为 0 且不是 X 费的牌。随机生成结果不属于玩家目标。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E20-01 | `INITIAL` | `STUDY` · 学习 | 随机获得一张能力牌。 |
| [ ] E20-02 | `INITIAL` | `TOUCH_CORE` · 触碰核心 | 随机获得一张耗能为0的牌。 |

- [ ] E20：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e21"></a>

### E21 丛林迷宫奇遇（`JUNGLE_MAZE_ADVENTURE`）

**生成范围：** Overgrowth。单人无额外限制；联机有生命限制。

**已核对规则：** JOIN_FORCES 与 SOLO_QUEST 是固定语义的两条事件选择；名字含结伴不表示需要单人选择另一个玩家。金币使用已生成公开值；SOLO 受伤后给金币。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E21-01 | `INITIAL` | `JOIN_FORCES` · 结伴同行 | 获得{JoinForcesGold}金币。 |
| [ ] E21-02 | `INITIAL` | `SOLO_QUEST` · 独自挑战 | 获得{SoloGold}金币。失去{SoloHp}点生命。 |

- [ ] E21：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e22"></a>

### E22 迷失鬼火（`LOST_WISP`）

**生成范围：** Hive。无额外 IsAllowed 限制。

**已核对规则：** CLAIM 加 Decay 并得 LostWisp；SEARCH 得金币。当前实现没有最大生命锁定分支；CLAIM_LOCKED 是残留文本，不能凭该文案添加额外合法性门槛。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E22-01 | `INITIAL` | `CLAIM` · 抓住这团鬼火 | 将一张{Curse}添加到你的牌组。获得{Relic}。 |
| [ ] E22-02 | `INITIAL` | `SEARCH` · 搜索附近的区域 | 获得{Gold}金币。 |

非候选／文本差异：

- `LOST_WISP.pages.INITIAL.options.CLAIM_LOCKED`：该 DLL 不生成最大生命锁定分支；遗留文本。

- [ ] E22：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e23"></a>

### E23 冷光合唱团（`LUMINOUS_CHOIR`）

**生成范围：** Overgrowth。金币达到事件基础要求且遗物袋有可用遗物；实际展示费用经 CalculateVars 调整。

**已核对规则：** 当前 OFFER_TRIBUTE 以公开实际 Gold 费用检查，不足时为锁定占位。REACH_INTO_THE_FLESH 选择移除 2 张后加 SporeMind。不能把已支付供奉变成还能回到另一分支。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E23-01 | `INITIAL` | `OFFER_TRIBUTE` · 供奉 | 支付{Gold}金币。获取一件随机遗物。 |
| [ ] E23-02 | `INITIAL` | `REACH_INTO_THE_FLESH` · 探入菌肉 | 从你的牌组移除2张牌。将一张孢子心灵添加到你的牌组。 |
| [ ] E23-03 | REACH_INTO_THE_FLESH 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | operation=remove；IsRemovable |

非候选／文本差异：

- `LUMINOUS_CHOIR.pages.INITIAL.options.OFFER_TRIBUTE_LOCKED`：锁定占位，不进入合法候选。需要{Gold}金币。

- [ ] E23：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e24"></a>

### E24 变形灵林谷（`MORPHIC_GROVE`）

**生成范围：** Overgrowth。金币 ≥100 且至少 2 张 IsTransformable 牌。

**已核对规则：** GROUP 先失去全部金币，再选 2 张变化；选择器先收齐再执行变化。不能把看到第一张变化结果后的再选第二张替代此流程。LONER 加最大生命。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E24-01 | `INITIAL` | `GROUP` · 大群变形灵 | {IsMultiplayer:所有人／}失去所有金币。变化2张牌。 |
| [ ] E24-02 | `INITIAL` | `LONER` · 落单变形灵 | 获得{MaxHp}点最大生命。 |
| [ ] E24-03 | GROUP 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | operation=transform；buffered，最终同时承诺原牌集合 |

- [ ] E24：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e25"></a>

### E25 药水快递员（`POTION_COURIER`）

**生成范围：** Shared。第 2、3 幕。

**已核对规则：** GRAB_POTIONS 给 FoulPotions 瓶污浊药水奖励；RANSACK 给一瓶随机罕见药水。逐瓶领取／容量处理是子动作，不能因为只看到一个事件按钮就自动拿满。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E25-01 | `INITIAL` | `GRAB_POTIONS` · 拿走这批药水 | 获得{FoulPotions}瓶污浊药水。 |
| [ ] E25-02 | `INITIAL` | `RANSACK` · 洗劫 | 获得1瓶随机罕见药水。 |
| [ ] E25-03 | 任一分支奖励 | `CLAIM_REWARD(potion_ref)／合法跳过／P-01/P-02` | 多个奖励分别保留，满槽不静默覆盖 |

- [ ] E25：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e26"></a>

### E26 重拳出击（`PUNCH_OFF`）

**生成范围：** Underdocks。TotalFloor ≥6。

**已核对规则：** NAB 加 Injury 并给遗物奖励；I_CAN_TAKE_THEM 后唯一 FIGHT 开战，附加遗物及药水奖励。背景互殴动画不等于可提前出牌的战斗阶段。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E26-01 | `INITIAL` | `I_CAN_TAKE_THEM` · 我能打两个 | 和它们战斗以得到更好的奖励。 |
| [ ] E26-02 | `INITIAL` | `NAB` · 顺走 | 将一张受伤加入到你的牌组。获得一件随机遗物。 |
| [ ] E26-03 | `I_CAN_TAKE_THEM` | `FIGHT` · 战斗 | 按该事件处理器结算；见本项规则。 |
| [ ] E26-04 | I_CAN_TAKE_THEM 后 | `FIGHT → 公共战斗动作` | 规则阶段确认后才提供出牌 |
| [ ] E26-05 | NAB／胜利后 | `公共遗物／药水奖励动作` | 保留获得效果与槽位决策 |

- [ ] E26：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e27"></a>

### E27 长者兰伟德（`RANWID_THE_ELDER`）

**生成范围：** Shared。第 2、3 幕；金币 ≥100、有药水、有 IsTradable 遗物。

**已核对规则：** 入场将 CanUseOrRemovePotions=false，结束恢复。POTION 与 RELIC 各随机预选一个对象并公开展示；玩家只决定是否交出指定对象，不从全部库存任选。POTION 换 1 遗物、GOLD 付 100 换 1、RELIC 交指定遗物换 2。查询不能重新抽对象。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E27-01 | `INITIAL` | `GOLD` · 给他{Gold}金币 | 获得一件随机遗物。 |
| [ ] E27-02 | `INITIAL` | `POTION` · 给他{Potion} | 获得一件随机遗物。 |
| [ ] E27-03 | `INITIAL` | `RELIC` · 给他{Relic} | 获得2件随机遗物。 |

非候选／文本差异：

- `RANWID_THE_ELDER.pages.INITIAL.options.POTION_LOCKED`：锁定占位，不进入合法候选。你没有任何可以给他的药水。
- `RANWID_THE_ELDER.pages.INITIAL.options.RELIC_LOCKED`：锁定占位，不进入合法候选。你没有任何可以给他的遗物。

- [ ] E27：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e28"></a>

### E28 镜中倒影  影倒中镜（`REFLECTIONS`）

**生成范围：** Glory。无额外 IsAllowed 限制。

**已核对规则：** TOUCH_A_MIRROR 随机降级 2、随机升级 4；SHATTER 复制整副牌组并加 BadLuck。均不提供玩家选择被降级／升级／复制的单牌列表。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E28-01 | `INITIAL` | `SHATTER` · 打碎 | 复制你的整个牌组。获得霉运。 |
| [ ] E28-02 | `INITIAL` | `TOUCH_A_MIRROR` · 触碰镜子 | 降级2张随机卡牌。升级4张随机卡牌。 |

- [ ] E28：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e29"></a>

### E29 遗物交换商（`RELIC_TRADER`）

**生成范围：** Shared。第 2、3 幕，至少 5 件 IsTradable 遗物。

**已核对规则：** 预生成最多 3 组 owned→new 固定配对，TOP/MIDDLE/BOTTOM 分别绑定整对。不能任意交一件换任意另一件。若异常／构造状态没有可交易对象，源码有唯一 PROCEED。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E29-01 | `INITIAL` | `BOTTOM` · 拿下面那件 | 用{BottomRelicOwned}换{BottomRelicNew}。 |
| [ ] E29-02 | `INITIAL` | `MIDDLE` · 拿中间这件 | 用{MiddleRelicOwned}换{MiddleRelicNew}。 |
| [ ] E29-03 | `INITIAL` | `TOP` · 拿上面那件 | 用{TopRelicOwned}换{TopRelicNew}。 |
| [ ] E29-04 | 零配对防御分支 | `CHOOSE_EVENT_OPTION(PROCEED)` | 仅 GenerateInitialOptions 返回该项时；唯一动作自动完成 |

- [ ] E29：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e30"></a>

### E30 满屋芝士（`ROOM_FULL_OF_CHEESE`）

**生成范围：** Shared。第 1、2 幕。

**已核对规则：** GORGE 从公开 8 张普通牌中恰选 2，默认不可取消；不是两次可跳过奖励。SEARCH 受 14 伤害然后获得 ChosenCheese。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E30-01 | `INITIAL` | `GORGE` · 大快朵颐 | 从8张随机普通牌中选择2张加入到你的牌组。 |
| [ ] E30-02 | `INITIAL` | `SEARCH` · 仔细翻找 | 失去{Damage}点生命，获得天选芝士。 |
| [ ] E30-03 | GORGE 网格 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | operation=obtain；无重复，选够前无 STOP/SKIP |

- [ ] E30：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e31"></a>

### E31 圆桌茶会（`ROUND_TEA_PARTY`）

**生成范围：** Glory。当前生命 ≥12。

**已核对规则：** ENJOY_TEA 得 RoyalPoison 并回满生命。PICK_FIGHT 只进入下一页；唯一 CONTINUE_FIGHT 才结算受伤与随机遗物。这里没有 EnterCombat 调用，不能因“斗殴”字样制造战斗阶段。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E31-01 | `INITIAL` | `ENJOY_TEA` · 喝杯好茶 | 获得{Relic}。回复全部生命。 |
| [ ] E31-02 | `INITIAL` | `PICK_FIGHT` · 挑事斗殴 | 失去{Damage}点生命。获得一件随机遗物。 |
| [ ] E31-03 | `PICK_FIGHT` | `CONTINUE_FIGHT` · 继续 | 按该事件处理器结算；见本项规则。 |

- [ ] E31：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e32"></a>

### E32 蓝宝石种子（`SAPPHIRE_SEED`）

**生成范围：** Overgrowth。无额外 IsAllowed 限制。

**已核对规则：** EAT 先治疗再选 1 张升级；PLANT 从实际 Sown 可附魔牌中选 1 张。不得把已治疗的父动作重复执行。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E32-01 | `INITIAL` | `EAT` · 吃下 | 回复{Heal}生命。升级你牌组中的一张牌。 |
| [ ] E32-02 | `INITIAL` | `PLANT` · 种植培育 | 给一张牌附魔：{Enchantment}。 |
| [ ] E32-03 | EAT 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=upgrade；IsUpgradable |
| [ ] E32-04 | PLANT 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=enchant；Sown.CanEnchant |

- [ ] E32：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e33"></a>

### E33 自助指南（`SELF_HELP_BOOK`）

**生成范围：** Shared。无额外 IsAllowed 限制。

**已核对规则：** 攻击／技能／能力三类分别检查 Sharp／Nimble／Swift.CanEnchant；不可用类显示锁定，不可选择。三类全不可用时才出现唯一 NO_OPTIONS（离开）；不是任何时候都有免费离开。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E33-01 | `INITIAL` | `NO_OPTIONS` · 离开 | 你没有任何可以附魔的牌。 |
| [ ] E33-02 | `INITIAL` | `READ_ENTIRE_BOOK` · 读完整本书 | 选择一张能力牌附魔：{Enchantment3}{Enchantment3Amount}。 |
| [ ] E33-03 | `INITIAL` | `READ_PASSAGE` · 随便读个一段 | 选择一张技能牌附魔：{Enchantment2}{Enchantment2Amount}。 |
| [ ] E33-04 | `INITIAL` | `READ_THE_BACK` · 读下封底 | 选择一张攻击牌附魔：{Enchantment1}{Enchantment1Amount}。 |
| [ ] E33-05 | READ_THE_BACK 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Attack＋Sharp.CanEnchant，附魔 2 层 |
| [ ] E33-06 | READ_PASSAGE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Skill＋Nimble.CanEnchant，附魔 2 层 |
| [ ] E33-07 | READ_ENTIRE_BOOK 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Power＋Swift.CanEnchant，附魔 2 层 |

非候选／文本差异：

- `SELF_HELP_BOOK.pages.INITIAL.options.READ_ENTIRE_BOOK_LOCKED`：锁定占位，不进入合法候选。你没有任何可以附魔的能力牌。
- `SELF_HELP_BOOK.pages.INITIAL.options.READ_PASSAGE_LOCKED`：锁定占位，不进入合法候选。你没有任何可以附魔的技能牌。
- `SELF_HELP_BOOK.pages.INITIAL.options.READ_THE_BACK_LOCKED`：锁定占位，不进入合法候选。你没有任何可以附魔的攻击牌。

- [ ] E33：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e34"></a>

### E34 滑脚木桥（`SLIPPERY_BRIDGE`）

**生成范围：** Shared。TotalFloor >6 且至少有一张可移除牌。

**已核对规则：** 每页只有当前公开指定牌的 OVERCOME 与 HOLD_ON。前者移除已经抽定的那一张，绝不能另开任意删牌器。后者付生命重新抽牌并更新下一次代价；HOLD_ON_LOOP 可以重复，候选数量恒为 2 不表示卡死。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E34-01 | `HOLD_ON_0` | `HOLD_ON_1` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-02 | `HOLD_ON_1` | `HOLD_ON_2` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-03 | `HOLD_ON_2` | `HOLD_ON_3` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-04 | `HOLD_ON_3` | `HOLD_ON_4` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-05 | `HOLD_ON_4` | `HOLD_ON_5` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-06 | `HOLD_ON_5` | `HOLD_ON_6` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-07 | `HOLD_ON_6` | `HOLD_ON_LOOP` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-08 | `HOLD_ON_LOOP` | `HOLD_ON_LOOP` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-09 | `INITIAL` | `HOLD_ON_0` · 再撑一会 | 失去{HpLoss}点生命。重新随机上方选项中的卡牌。 |
| [ ] E34-10 | `INITIAL` | `OVERCOME` · 跨越 | {RandomCard}将从你的牌组中被移除。 |

- [ ] E34：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e35"></a>

### E35 螺旋漩涡（`SPIRALING_WHIRLPOOL`）

**生成范围：** Underdocks。有 Spiral.CanEnchant 的牌。

**已核对规则：** 当前只生成 OBSERVE／DRINK；REACH_IN 是未生成的遗留文本。OBSERVE 限符合附魔基础条件的 Basic Strike／Defend 标签牌。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E35-01 | `INITIAL` | `DRINK` · 饮用 | 回复{Heal}点生命。 |
| [ ] E35-02 | `INITIAL` | `OBSERVE` · 观察 | 为一张基础“打击”或“防御”附魔：涡旋。 |
| [ ] E35-03 | OBSERVE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=enchant；Spiral.CanEnchant |

非候选／文本差异：

- `SPIRALING_WHIRLPOOL.pages.INITIAL.options.REACH_IN`：该 DLL 仅生成 OBSERVE／DRINK；遗留文本。

- [ ] E35：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e36"></a>

### E36 灵魂嫁接者（`SPIRIT_GRAFTER`）

**生成范围：** Hive。无额外 IsAllowed 限制。

**已核对规则：** LET_IT_IN 治疗并加 Metamorphosis；REJECTION 先受伤再选择升级牌。升级目标不是随机决定。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E36-01 | `INITIAL` | `LET_IT_IN` · 接纳 | 回复{LetItInHealAmount}点生命。将一张羽化添加到你的牌组。 |
| [ ] E36-02 | `INITIAL` | `REJECTION` · 拒绝 | 失去{RejectionHpLoss}点生命。升级一张牌。 |
| [ ] E36-03 | REJECTION 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=upgrade；IsUpgradable |

- [ ] E36：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e37"></a>

### E37 永恒之石（`STONE_OF_ALL_TIME`）

**生成范围：** Shared。第 2 幕，至少一瓶药水。

**已核对规则：** 事件期间禁止主动使用／移除药水。LIFT 交出事件已指定药水，不是任意选瓶；没有药水时锁定。PUSH 需要 Vigorous.CanEnchant 的牌，先受伤，再选 1 张附魔。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E37-01 | `INITIAL` | `LIFT` · 喝药抬起 | 失去{DrinkRandomPotion}。获得{DrinkMaxHpGain}最大生命。 |
| [ ] E37-02 | `INITIAL` | `PUSH` · 用力去推 | 失去{PushHpLoss}点生命。附魔一张攻击牌：活力{PushVigorousAmount}。 |
| [ ] E37-03 | PUSH 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=enchant；Vigorous.CanEnchant，实际层数 PushVigorousAmount |

非候选／文本差异：

- `STONE_OF_ALL_TIME.pages.INITIAL.options.LIFT_LOCKED`：锁定占位，不进入合法候选。需要药水。
- `STONE_OF_ALL_TIME.pages.INITIAL.options.PUSH_LOCKED`：锁定占位，不进入合法候选。需要一张攻击牌。

- [ ] E37：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e38"></a>

### E38 沉没雕像（`SUNKEN_STATUE`）

**生成范围：** Overgrowth / Underdocks。Overgrowth／Underdocks 均注册；无额外 IsAllowed 限制。

**已核对规则：** GRAB_SWORD 获得 SwordOfStone；DIVE_INTO_WATER 受伤并得金币。没有再选“哪把剑／哪份金币”。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E38-01 | `INITIAL` | `DIVE_INTO_WATER` · 潜水 | 获得{Gold}金币。失去{HpLoss}点生命. |
| [ ] E38-02 | `INITIAL` | `GRAB_SWORD` · 拿起石剑 | 获得{Relic}。 |

- [ ] E38：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e39"></a>

### E39 淹水金库（`SUNKEN_TREASURY`）

**生成范围：** Underdocks。无额外 IsAllowed 限制。

**已核对规则：** FIRST_CHEST 与 SECOND_CHEST 是互斥事件分支，不是普通宝箱两个都能拿；后者伴随 Greed。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E39-01 | `INITIAL` | `FIRST_CHEST` · 第一个箱子 | 获得{SmallChestGold}金币。 |
| [ ] E39-02 | `INITIAL` | `SECOND_CHEST` · 第二个箱子 | 获得{LargeChestGold}金币。得到贪婪。 |

- [ ] E39：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e40"></a>

### E40 共生体（`SYMBIOTE`）

**生成范围：** Shared。第 2、3 幕。

**已核对规则：** APPROACH 只有存在 Corrupted.CanEnchant 攻击牌时可用；KILL_WITH_FIRE 选择变化。锁定项只用于展示原因。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E40-01 | `INITIAL` | `APPROACH` · 靠近 | 将一张攻击牌附魔：{Enchantment}。 |
| [ ] E40-02 | `INITIAL` | `KILL_WITH_FIRE` · 用火烧杀 | 选择一张牌变化。 |
| [ ] E40-03 | APPROACH 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=enchant；Corrupted.CanEnchant |
| [ ] E40-04 | KILL_WITH_FIRE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=transform；数量使用 Cards 当前值 |

非候选／文本差异：

- `SYMBIOTE.pages.INITIAL.options.APPROACH_LOCKED`：锁定占位，不进入合法候选。你没有可以附魔的攻击牌。

- [ ] E40：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e41"></a>

### E41 真理石板（`TABLET_OF_TRUTH`）

**生成范围：** Overgrowth。无额外 IsAllowed 限制。

**已核对规则：** 初始是 DECIPHER_1／SMASH；继续阅读后是 DECIPHER／GIVE_UP，不能再砸碎。五次阅读代价依次为 3、6、12、24、当前最大生命−1；前四次随机升级 1 张，最后升级全部并结束。最大生命不够时的致死提示不是锁定。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E41-01 | `DECIPHER` | `GIVE_UP` · 放弃 | 停止阅读文字，就这样离开。 |
| [ ] E41-02 | `DECIPHER_1` | `DECIPHER` · 继续解读 | 失去{DecipherMaxHpLoss}点最大生命。随机升级一张牌。 |
| [ ] E41-03 | `DECIPHER_2` | `DECIPHER` · 继续解读 | 失去{DecipherMaxHpLoss}点最大生命。随机升级一张牌。 |
| [ ] E41-04 | `DECIPHER_3` | `DECIPHER` · 继 续 解 读 | 失去{DecipherMaxHpLoss}点最大生命。随机升级一张牌。 |
| [ ] E41-05 | `DECIPHER_4` | `DECIPHER` · 抛下一切 | 失去{DecipherMaxHpLoss}点最大生命。升级你的所有卡牌。 |
| [ ] E41-06 | `INITIAL` | `DECIPHER_1` · 解读 | 失去{DecipherMaxHpLoss}点最大生命。随机升级一张牌。 |
| [ ] E41-07 | `INITIAL` | `SMASH` · 砸碎 | 回复{SmashHPGain}点生命。 |

- [ ] E41：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e42"></a>

### E42 茶艺大师（`TEA_MASTER`）

**生成范围：** Shared。第 1、2 幕，金币 ≥150。

**已核对规则：** 骨茶费用 50、余烬茶 150，各自不足则锁定；无礼之茶不收费。效果来自对应 BoneTea／EmberTea／TeaOfDiscourtesy 遗物动态描述，不能把它们当空效果按钮或无代价离开。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E42-01 | `INITIAL` | `BONE_TEA` · 骨茶 | 支付{BoneTeaCost}金币。{BoneTeaDescription} |
| [ ] E42-02 | `INITIAL` | `EMBER_TEA` · 余烬茶 | 支付{EmberTeaCost}金币。{EmberTeaDescription} |
| [ ] E42-03 | `INITIAL` | `TEA_OF_DISCOURTESY` · 无礼之茶 | {TeaOfDiscourtesyDescription} |

非候选／文本差异：

- `TEA_MASTER.pages.INITIAL.options.BONE_TEA_LOCKED`：锁定占位，不进入合法候选。需要{BoneTeaCost}金币。
- `TEA_MASTER.pages.INITIAL.options.EMBER_TEA_LOCKED`：锁定占位，不进入合法候选。需要{EmberTeaCost}金币。

- [ ] E42：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e43"></a>

### E43 药水的未来？（`THE_FUTURE_OF_POTIONS`）

**生成范围：** Shared。至少 2 瓶药水。

**已核对规则：** 事件期间禁止主动使用／移除药水。枚举当前药水列表前 min(3,count) 瓶，每瓶一个同 TextKey 的不同候选，绑定瓶实例与已公开结果稀有度／类型。交出后进入展示 3 张已升级牌的奖励。不是任选全部药水，也不是任选奖励卡类型。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E43-01 | `INITIAL` | `POTION` · 放入{Rarity}药水 | 失去{Potion}。获得一张升级过的{Rarity}{Type}牌。 |
| [ ] E43-02 | 交易后奖励 | `TAKE_CARD_REWARD(card_ref)／真实 alternatives` | 药水先消耗；奖励跳过不会退回药水 |

- [ ] E43：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e44"></a>

### E44 灯火钥匙（`THE_LANTERN_KEY`）

**生成范围：** Hive。Hive 事件池，无额外 IsAllowed 限制。

**已核对规则：** RETURN_THE_KEY 得金币后结束；KEEP_THE_KEY 后唯一 FIGHT 进入 MysteriousKnightEventEncounter，胜利奖励里有 LanternKey。DONE.options.RETURN_THE_KEY.description 实际是结束页叙述，不是第二次交钥匙按钮。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E44-01 | `INITIAL` | `KEEP_THE_KEY` · 留下钥匙 | 战斗来取得钥匙。 |
| [ ] E44-02 | `INITIAL` | `RETURN_THE_KEY` · 交还钥匙 | 获得{Gold}金币。 |
| [ ] E44-03 | `KEEP_THE_KEY` | `FIGHT` · 战斗 | 按该事件处理器结算；见本项规则。 |
| [ ] E44-04 | KEEP_THE_KEY 后 | `FIGHT → 公共战斗 → 奖励` | 获得钥匙可能触发后续 WarHistorianRepy，不能提前暴露未来节点结果 |

非候选／文本差异：

- `THE_LANTERN_KEY.pages.DONE.options.RETURN_THE_KEY`：用作 SetEventFinished 的叙述 key，并非动作。

- [ ] E44：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e45"></a>

### E45 传说是真的（`THE_LEGENDS_WERE_TRUE`）

**生成范围：** Shared。第 1 幕，牌组非空且当前生命 ≥10。

**已核对规则：** NAB_THE_MAP 加 SpoilsMap；SLOWLY_FIND_AN_EXIT 受伤后给药水奖励，必须处理槽位。任务牌对后续地图的影响交原引擎。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E45-01 | `INITIAL` | `NAB_THE_MAP` · 顺走地图 | 获得藏宝图。 |
| [ ] E45-02 | `INITIAL` | `SLOWLY_FIND_AN_EXIT` · 耐心寻找出口 | 失去{Damage}点生命。获得1瓶随机药水。 |
| [ ] E45-03 | SLOWLY_FIND_AN_EXIT 后 | `公共药水奖励动作` | 不可提前读出随机药水 |

- [ ] E45：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e46"></a>

### E46 这个还是那个？（`THIS_OR_THAT`）

**生成范围：** Shared。无额外 IsAllowed 限制。

**已核对规则：** PLAIN 受伤后得金币；ORNATE 的实际调用顺序是先获得随机遗物，再加 Clumsy，与下方文本顺序相反。遗物拾取触发的子流程也在加诅咒之前。随机遗物身份未公开前不放入候选。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E46-01 | `INITIAL` | `ORNATE` · 那个 | 将一张{Curse}加入你的牌组。获得一件随机遗物。 |
| [ ] E46-02 | `INITIAL` | `PLAIN` · 这个 | 失去{HpLoss}点生命。获得{Gold}金币。 |

- [ ] E46：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e47"></a>

### E47 打造时间（`TINKER_TIME`）

**生成范围：** Glory。Glory 事件池，无额外 IsAllowed 限制。

**已核对规则：** 初始 CHOOSE_CARD_TYPE 是唯一接受动作。随后从 Attack／Skill／Power 抽 2 个展示；选类型后从该类型的 3 个 rider 抽 2 个展示。攻击池=SAPPING/VIOLENCE/CHOKING，技能池=ENERGIZED/WISDOM/CHAOS，能力池=EXPERTISE/CURIOUS/IMPROVEMENT。最后得到含已选类型及 rider 的 MadScience；不能在初页枚举全部九种成品。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E47-01 | `CHOOSE_CARD_TYPE` | `ATTACK` · 武器 | 制作一张攻击牌。 |
| [ ] E47-02 | `CHOOSE_CARD_TYPE` | `POWER` · 装置 | 制作一张能力牌。 |
| [ ] E47-03 | `CHOOSE_CARD_TYPE` | `SKILL` · 护具 | 制作一张技能牌。 |
| [ ] E47-04 | `CHOOSE_RIDER` | `CHAOS` · 混沌 | 将一张随机牌放入你的手牌。这张牌在本回合可以免费打出。 |
| [ ] E47-05 | `CHOOSE_RIDER` | `CHOKING` · 扼杀 | 本回合，你每打出一张牌，该敌人失去{ChokingDamage}点生命。 |
| [ ] E47-06 | `CHOOSE_RIDER` | `CURIOUS` · 好奇 | 能力牌的耗能减少{CuriousReduction}{energyPrefix:energyIcons(1)}。 |
| [ ] E47-07 | `CHOOSE_RIDER` | `ENERGIZED` · 充能 | 获得{EnergizedEnergy:energyIcons()}。 |
| [ ] E47-08 | `CHOOSE_RIDER` | `EXPERTISE` · 专长 | 获得{ExpertiseStrength}点力量。获得{ExpertiseDexterity}点敏捷。 |
| [ ] E47-09 | `CHOOSE_RIDER` | `IMPROVEMENT` · 精进 | 在战斗结束时，随机升级一张牌。 |
| [ ] E47-10 | `CHOOSE_RIDER` | `SAPPING` · 削弱 | 给予{SappingWeak}层虚弱。给予{SappingVulnerable}层易伤。 |
| [ ] E47-11 | `CHOOSE_RIDER` | `VIOLENCE` · 暴力 | 额外命中2次。 |
| [ ] E47-12 | `CHOOSE_RIDER` | `WISDOM` · 智慧 | 抽{WisdomCards}张牌。 |
| [ ] E47-13 | `INITIAL` | `CHOOSE_CARD_TYPE` · 接受 | 制作一张定制卡牌加入你的牌组。 |

- [ ] E47：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e48"></a>

### E48 垃圾堆（`TRASH_HEAP`）

**生成范围：** Underdocks。当前生命 >5。

**已核对规则：** DIVE_IN 受当前 HpLoss（本版 8）并随机给旧日遗物；GRAB 得金币和随机旧日卡。入场生命门槛不是保证任何分支存活，生命 6～8 时也不能自行过滤扎入垃圾堆。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E48-01 | `INITIAL` | `DIVE_IN` · 扎进垃圾堆 | 失去{HpLoss}点生命。获得一件被遗忘的旧日遗物。 |
| [ ] E48-02 | `INITIAL` | `GRAB` · 随便拿点垃圾 | 获得{Gold}金币。获得一张被遗忘的旧日卡牌。 |

- [ ] E48：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e49"></a>

### E49 审判（`TRIAL`）

**生成范围：** Glory。Glory 事件池，无额外 IsAllowed 限制。

**已核对规则：** INITIAL 接受后才随机揭示 MERCHANT/NOBLE/NONDESCRIPT 三类案件之一，当前只保留该案的 GUILTY/INNOCENT。REJECT 进入 ACCEPT／DOUBLE_DOWN；DOUBLE_DOWN 打开真正的放弃对局确认弹窗，不能当普通退出或假定直接已死亡。详见专节。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E49-01 | `INITIAL` | `ACCEPT` · 接受 | 担当今天的判决者 |
| [ ] E49-02 | `INITIAL` | `REJECT` · 拒绝 | 拒绝是不被允许的。 |
| [ ] E49-03 | `MERCHANT` | `GUILTY` · 判决：有罪 | 将一张悔恨加入你的牌组。获得2件随机遗物。 |
| [ ] E49-04 | `MERCHANT` | `INNOCENT` · 判决：无罪 | 将一张羞耻加入你的牌组。升级2张牌。 |
| [ ] E49-05 | `NOBLE` | `GUILTY` · 判决：有罪 | 回复10点生命。 |
| [ ] E49-06 | `NOBLE` | `INNOCENT` · 判决：无罪 | 将一张悔恨加入你的牌组。获得300金币。 |
| [ ] E49-07 | `NONDESCRIPT` | `GUILTY` · 判决：有罪 | 将一张疑虑加入你的牌组。获得2次卡牌奖励。 |
| [ ] E49-08 | `NONDESCRIPT` | `INNOCENT` · 判决：无罪 | 将一张疑虑加入你的牌组。变化2张牌。 |
| [ ] E49-09 | `REJECT` | `ACCEPT` · 接受 | 放弃。担当今天的判决者。 |
| [ ] E49-10 | `REJECT` | `DOUBLE_DOWN` · 坚持到底 | 直面致命的后果。 |
| [ ] E49-11 | MERCHANT.INNOCENT 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | 先加 Shame，再选择升级 2 张 |
| [ ] E49-12 | NONDESCRIPT.INNOCENT 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | 先加 Doubt，再选择变化 2 张 |
| [ ] E49-13 | NONDESCRIPT.GUILTY 奖励 | `两组 TAKE_CARD_REWARD／alternatives` | 先加 Doubt；两组不能遗漏第二组 |
| [ ] E49-14 | REJECT.DOUBLE_DOWN | `最终放弃承诺／确认弹窗桥接` | 保留确认与回到拒绝页的原语义，不自动退出游戏进程 |

- [ ] E49：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e50"></a>

### E50 无休之处（`UNREST_SITE`）

**生成范围：** Overgrowth。当前生命 ≤70% 最大生命。

**已核对规则：** REST 回满并加 PoorSleep；KILL 减最大生命并得随机遗物。不是普通营火，不能提供 SMITH／DIG。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E50-01 | `INITIAL` | `KILL` · 杀死树木 | 失去{MaxHpLoss}点最大生命。获得一件随机遗物。 |
| [ ] E50-02 | `INITIAL` | `REST` · 就这样休息 | 回复全部生命值。获得睡眠不佳。 |

- [ ] E50：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e51"></a>

### E51 战史学家 付袭（`WAR_HISTORIAN_REPY`）

**生成范围：** Shared。IsAllowed 恒为 false，不能由普通随机事件池抽到；LanternKey 任务流程另行触发。

**已核对规则：** 初选 UNLOCK_CAGE 或 UNLOCK_CHEST。单人首次只移除第一张钥匙；如果仍有钥匙，得到唯一另一分支，可领取另一奖励并移除剩余钥匙。不可把 IsAllowed=false 误写成整个游戏不可能遇到。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E51-01 | `INITIAL` | `UNLOCK_CAGE` · 打开笼子 | 失去灯火钥匙。获得历史课。 |
| [ ] E51-02 | `INITIAL` | `UNLOCK_CHEST` · 打开宝箱 | 失去灯火钥匙。获得2瓶随机药水和2件随机遗物。 |
| [ ] E51-03 | 首次开笼后仍有钥匙 | `CHOOSE_EVENT_OPTION(UNLOCK_CHEST)` | 来源=后续页，唯一相反奖励 |
| [ ] E51-04 | 首次开箱后仍有钥匙 | `CHOOSE_EVENT_OPTION(UNLOCK_CAGE)` | 来源=后续页，唯一相反奖励 |
| [ ] E51-05 | 开箱奖励 | `2 瓶药水＋2 件遗物的公共奖励流程` | 逐项领取、容量、遗物子效果 |

- [ ] E51：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e52"></a>

### E52 水漫缮写室（`WATERLOGGED_SCRIPTORIUM`）

**生成范围：** Underdocks。金币 ≥55。

**已核对规则：** BLOODY_INK 加 6 最大生命；TENTACLE_QUILL 需 55 金币、选 1 张 Steady 可附魔牌；PRICKLY_SPONGE 需 99 金币、选 Cards 张。两付费分支先扣金币再选牌；不能重复扣费或默认取消退款。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E52-01 | `INITIAL` | `BLOODY_INK` · 血液墨水 | 获得6点最大生命。 |
| [ ] E52-02 | `INITIAL` | `PRICKLY_SPONGE` · 扎手海绵 | 支付{PricklySpongeGold}金币。给{Cards}张牌附魔：稳定。 |
| [ ] E52-03 | `INITIAL` | `TENTACLE_QUILL` · 触手羽毛笔 | 支付{Gold}金币。给一张牌附魔：稳定。 |
| [ ] E52-04 | TENTACLE_QUILL 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Steady.CanEnchant，附魔 1 层 |
| [ ] E52-05 | PRICKLY_SPONGE 子选择 | `SELECT_ONE(card_ref) × Cards → FINISH_SELECTION` | Steady.CanEnchant，数量取实际会话 |

非候选／文本差异：

- `WATERLOGGED_SCRIPTORIUM.pages.INITIAL.options.PRICKLY_SPONGE_LOCKED`：锁定占位，不进入合法候选。需要{PricklySpongeGold}金币。
- `WATERLOGGED_SCRIPTORIUM.pages.INITIAL.options.TENTACLE_QUILL_LOCKED`：锁定占位，不进入合法候选。需要{Gold}金币。

- [ ] E52：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e53"></a>

### E53 欢迎来到旺购百货（`WELCOME_TO_WONGOS`）

**生成范围：** Shared。第 2 幕，金币 ≥100。

**已核对规则：** 三档分别 100／200／300 金币，各自不足锁定；任一次购买后事件结束，不是可连续买三件的普通商店。FEATURED_ITEM 是已公开指定遗物；MYSTERY_BOX 是未来战斗后兑现的票据，不能提前显示随机遗物。LEAVE 会随机降级 1 张，始终是有后果的语义分支。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E53-01 | `INITIAL` | `BARGAIN_BIN` · 旺购的打折货物 | 支付{BargainBinCost}金币。获得1件随机普通遗物。 |
| [ ] E53-02 | `INITIAL` | `FEATURED_ITEM` · 旺购的特选商品 | 支付{FeaturedItemCost}金币。获得{RandomRelic}。 |
| [ ] E53-03 | `INITIAL` | `LEAVE` · 离开 | 随机降级一张牌。 |
| [ ] E53-04 | `INITIAL` | `MYSTERY_BOX` · 旺购的神秘盲盒 | 支付{MysteryBoxCost}金币。在{MysteryBoxCombatCount}场战斗后获得{MysteryBoxRelicCount}个随机遗物。 |

非候选／文本差异：

- `WELCOME_TO_WONGOS.pages.INITIAL.options.BARGAIN_BIN_LOCKED`：锁定占位，不进入合法候选。需要{BargainBinCost}金币。
- `WELCOME_TO_WONGOS.pages.INITIAL.options.FEATURED_ITEM_LOCKED`：锁定占位，不进入合法候选。需要{FeaturedItemCost}金币。
- `WELCOME_TO_WONGOS.pages.INITIAL.options.MYSTERY_BOX_LOCKED`：锁定占位，不进入合法候选。需要{MysteryBoxCost}金币。

- [ ] E53：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e54"></a>

### E54 泉水（`WELLSPRING`）

**生成范围：** Overgrowth。无额外 IsAllowed 限制。

**已核对规则：** BOTTLE 进入随机药水奖励；BATHE 先选牌移除，再加入 BatheCurses 张 Guilty。永久诅咒不属于可跳过卡牌奖励。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E54-01 | `INITIAL` | `BATHE` · 沐浴 | 从你的牌组中移除1张牌。将{BatheCurses}张愧疚添加到你的牌组。 |
| [ ] E54-02 | `INITIAL` | `BOTTLE` · 装瓶 | 获得1瓶随机药水。 |
| [ ] E54-03 | BATHE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=remove；IsRemovable |
| [ ] E54-04 | BOTTLE 奖励 | `公共药水奖励动作` | 含满槽处理 |

- [ ] E54：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e55"></a>

### E55 低语空谷（`WHISPERING_HOLLOW`）

**生成范围：** Overgrowth。金币 ≥44。

**已核对规则：** GOLD 支付当前 Gold 得 2 瓶药水奖励；HUG 的代码先选牌变化，再扣生命，与把文案从左到右当执行顺序不同。保持真实效果顺序。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E55-01 | `INITIAL` | `GOLD` · 交换金币 | 失去{Gold}金币。获得2瓶随机药水。 |
| [ ] E55-02 | `INITIAL` | `HUG` · 拥抱树木 | 失去{HpLoss}点生命。选择一张牌变化。 |
| [ ] E55-03 | HUG 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=transform；变化后才受伤 |
| [ ] E55-04 | GOLD 奖励 | `两瓶独立药水奖励动作` | 领取／使用／丢弃的先后按原规则 |

- [ ] E55：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e56"></a>

### E56 木雕（`WOOD_CARVINGS`）

**生成范围：** Overgrowth。有 Basic 且 IsRemovable 的牌；这是入场谓词。

**已核对规则：** BIRD／TORUS 的真正选择过滤是 Basic＋IsTransformable，分别变化为 Peck／ToricToughness；不能错用入场 IsRemovable 当子选择过滤。SNAKE 使用 Slither.CanEnchant，无目标锁定。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E56-01 | `INITIAL` | `BIRD` · 鸟 | 选择1张初始牌变化为{BirdCard}。 |
| [ ] E56-02 | `INITIAL` | `SNAKE` · 蛇 | 给1张牌附魔：{SnakeEnchantment}。 |
| [ ] E56-03 | `INITIAL` | `TORUS` · 圆环 | 选择1张初始牌变化为{ToricCard}。 |
| [ ] E56-04 | BIRD 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Basic＋IsTransformable；transform_to=Peck |
| [ ] E56-05 | TORUS 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Basic＋IsTransformable；transform_to=ToricToughness |
| [ ] E56-06 | SNAKE 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | Slither.CanEnchant |

非候选／文本差异：

- `WOOD_CARVINGS.pages.INITIAL.options.SNAKE_LOCKED`：锁定占位，不进入合法候选。你没有任何牌可以添加附魔：蛇行。

- [ ] E56：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

<a id="e57"></a>

### E57 修禅织网者（`ZEN_WEAVER`）

**生成范围：** Hive。金币 ≥EmotionalAwarenessCost（本版 125）。

**已核对规则：** BREATHING_TECHNIQUES 付 50 加两张 Enlightenment；EMOTIONAL_AWARENESS 需 125；ARACHNID_ACUPUNCTURE 需 250，不足对应锁定。两移除分支代码先选择并移除，再扣金币；不等同水漫缮写室的先扣费。

| 核对号 | 页面／子阶段 | 可选动作 | 公开后果或约束 |
| --- | --- | --- | --- |
| [ ] E57-01 | `INITIAL` | `ARACHNID_ACUPUNCTURE` · 蜘蛛针灸 | 支付{ArachnidAcupunctureCost}金币。从你的牌组中移除2张牌。 |
| [ ] E57-02 | `INITIAL` | `BREATHING_TECHNIQUES` · 呼吸技法 | 支付{BreathingTechniquesCost}金币。将2张开悟加入到你的牌组。 |
| [ ] E57-03 | `INITIAL` | `EMOTIONAL_AWARENESS` · 情绪觉察 | 支付{EmotionalAwarenessCost}金币。从你的牌组中移除1张牌。 |
| [ ] E57-04 | EMOTIONAL_AWARENESS 子选择 | `SELECT_ONE(card_ref) → FINISH_SELECTION` | operation=remove，1 张；成功处理后扣费 |
| [ ] E57-05 | ARACHNID_ACUPUNCTURE 子选择 | `SELECT_ONE(card_ref) × 2 → FINISH_SELECTION` | operation=remove，2 张；不足处理沿用引擎 |

非候选／文本差异：

- `ZEN_WEAVER.pages.INITIAL.options.LOCKED`：锁定占位，不进入合法候选。金币不足。

- [ ] E57：逐行确认当前页合法集、子选择数量与过滤、后果顺序、结束方式；补充真实场景／执行记录。

## 4. 水晶球：必须逐项实现的完整交互

静态证据：`CrystalSphere.GenerateInitialOptions/UncoverFuture/PaymentPlan`、`CrystalSphereMinigame`、`NCrystalSphereCell._Ready/EntityClicked`、`NCrystalSphereScreen.OnCellClicked/OnMinigameFinished`、`CrystalSphereCurse.RevealItem`、`OneOffSynchronizer.OfferCrystalSphereRewards`。**不仅看事件模型，还核对了原 UI 对点击目标的限制。**

| 核对号 | 阶段 | 完整合法动作／行为 | 明确不合法或不能自动代选的行为 |
| --- | --- | --- | --- |
| [ ] CS-01 | 入场 | `UNCOVER_FUTURE`：支付当前已公开费用并获得 3 次；`PAYMENT_PLAN`：加 Debt 并获得 6 次 | 不存在额外免费离开；不能重新 CalculateVars 挑价格 |
| [ ] CS-02 | 建板 | 原引擎创建 11×11 图板、初始雾层与隐藏摆放，随后生成公开帧 | 建板随机结果、`Items` 完整对象和内部位置不能直接序列化给模型 |
| [ ] CS-03 | 点击中心合法性 | 次数 >0、未完成、无阻塞子交互；中心坐标属于图板且该格仍 `IsHidden` | 已清除格在 GUI 中 MouseFilter=Ignore、不可聚焦，不得仅因底层 CellClicked 可调用就当合法中心 |
| [ ] CS-04 | 小幅 | `DIVINE(small,(x,y))`，清除中心一格，消耗 **1** 次 | 不能只提供大幅占卜，也不能由模型选择隐藏奖品 ID |
| [ ] CS-05 | 大幅 | `DIVINE(big,(x,y))`，清除含中心的八邻域，即裁剪到板内的 3×3，消耗 **1** 次 | 不存在大幅花多次、小幅花少次的本版规则；不能限制为必须整块 3×3 都未揭示 |
| [ ] CS-06 | 模式按钮 | 把 `SetTool` 和点击目标绑定成一个完整候选；模式切换自身不扣次数、不揭示 | 不为反复切模式制造策略循环；`None` 不是玩家可选第三模式 |
| [ ] CS-07 | 局部清除 | 真实执行 `ClearCell`；已清除邻格为无操作，中心必须合法 | 不能把“该范围含诅咒”用于过滤候选，不能偷偷选择收益最大位置 |
| [ ] CS-08 | 完整揭示 | 物品占据的所有格清除后触发该物品 `RevealItem`；这与仅露出部分图形不同 | 不把局部清除误当整件物品已经获得 |
| [ ] CS-09 | 诅咒揭示 | **Doubt 在完整揭示时立即加入牌组**；形成真实公开状态变化 | 不把诅咒全部延后到次数用尽，不在最终奖励再次发同一诅咒 |
| [ ] CS-10 | 普通物品揭示 | 记录已揭示金币／药水类别／卡奖励类别／遗物类别，按原流程在结束时生成奖励 | 已揭示“某稀有度药水／遗物图标”不等于已知最终随机内容 |
| [ ] CS-11 | 后续点击 | 前一次揭示和即时效果结算后重新枚举剩余 `mode×cell` | 不把 3／6 次位置一次性当 buffered 多选，后续必须能利用新公开信息 |
| [ ] CS-12 | 次数归零 | 完成 completion source，按揭示物品顺序构建实际奖励组，处理选牌／容量／遗物子效果 | 不重复 CompleteMinigame；不因出现奖励页丢失父事件 |
| [ ] CS-13 | 结束 | 原完成流程后出现 Proceed；若只剩已承诺的页面退出则自动推进 | 没有剩余次数时继续点格、或提前按 STOP 领取部分奖励，均不提供 |
| [ ] CS-14 | 强制关闭 | `ForceMinigameEnd` 是关闭应用／存档退出时的取消路径，会清 `_revealed` 并取消任务 | 不能把它包装成普通可获利退出／SKIP |

合法集定义（去重前；P 代表当前真的允许的公共药水动作）：

```text
H = 当前 11×11 图板中 IsHidden=true 且 GUI 规则允许点击的中心格
若 remaining_uses > 0 且事件可交互：
    legal = { DIVINE(small,c), DIVINE(big,c) | c ∈ H } ∪ P
若 remaining_uses = 0：
    进入结算／奖励／完成边界，不再生成 DIVINE
```

模型可看图板坐标、公开雾层、已经看到的图形片段、已经完整揭示的物品类别、剩余次数、公开奖励与动作范围。**部分露出的图形应保留玩家实际能看见的信息，但不能借其 `Item` 指针暴露仍在雾下的完整占据矩形或其他物品位置。** 若采用结构化片段编码，须与实际渲染可见性对照，不能只用“完整揭示／未知”两档而丢失玩家已见的线索。

本版四角各自动清除距角曼哈顿距离 ≤2 的 6 格，共 24 格；初始隐藏中心为 97 格。无额外合法动作、且去除严格等价别名前，首轮小游戏应有 **194 个 mode×cell 绑定**，不是 2 个模式按钮，也不是 121 个格子动作。候选实际数量仍由稳定快照枚举；不能在部署端写死 194。

- [ ] CS-T1：付费／诅咒两种入场分别只结算一次；次数分别 3／6。
- [ ] CS-T2：中间、边缘、已清除邻格的范围正确；已清除中心不可选；每次都只减 1。
- [ ] CS-T3：同一公开雾层下置换隐藏物品位置，当前候选及公开语义不变；不公开随机未来奖励。
- [ ] CS-T4：只露出部分物品时保留可见片段；整件揭示时触发一次；Doubt 当场加入。
- [ ] CS-T5：最后一次揭示后完整处理多组奖励、满药水槽及遗物子选择，随后恢复事件。
- [ ] CS-T6：不提供提前 STOP／ForceMinigameEnd；未知自定义阶段显式报错，不能 `leave_room` 跳过。

## 5. 特殊事件的协议处理要点

### 5.1 空 `CurrentOptions` 与假商人

`FakeMerchant.GenerateInitialOptions` 合法返回空列表，动作在库存及药水目标中。候选提供器必须按 `LayoutType.Custom` 路由，保留其 6 件实际商品、当前价格、可购买性、真实离开和污浊药水攻击。普通 `DoBuyRelic` 只识别 MerchantRoom，不能直接把 EventRoom 的假商人冒充普通商店。

关闭库存才能指向商人是 UI 条件。候选枚举可包括“关闭库存→投掷”的完整执行绑定；不能因 headless 没有 MerchantButton，让 `PassesCustomUsabilityCheck` 永远为 false。这是需要等价桥接的条件，不是修改污浊药水规则的理由。药水投掷后必须真实消耗、进入专用 encounter、发放仍可得奖励。

假商人的 Proceed 实际打开地图，选择下一个节点才放弃房间机会。为符合架构的 `LEAVE_EVENT` 边界，可将最终离开承诺与下个合法地图选择关联；免费查看地图／返回当前库存不能反复生成训练动作，也不能把只打开地图误记为已经不可返回。

### 5.2 审判的坚持到底与确认弹窗

`Trial.DoubleDown` 创建 `NAbandonRunConfirmPopup`；其 Yes 调用 `RunManager.Abandon()`，No 关闭弹窗。TestMode 下创建弹窗可返回 null，不能把这个 headless 差异当成事件已经兼容。

按架构去掉纯 UI 往返后，拒绝页可表示为两条最终语义路径：

| 核对号 | 候选 | 执行绑定 |
| --- | --- | --- |
| TR-01 | 接受判决者职责 | `REJECT.ACCEPT`，之后才揭示案件 |
| TR-02 | 确认坚持到底／放弃本局 | `REJECT.DOUBLE_DOWN`＋真正 Yes 承诺，按引擎记录终局原因 |

如果工程必须显式暴露确认页，则该页提供“确认”和“取消返回拒绝页”的真实能力，但纯打开／取消不能产生可无限循环的策略样本；最终训练表示仍须去掉无新信息、无成本的往返。无论选择哪种编码，不能把 UI 确认自动成一个此前未明确承诺的致死结果。不要把 `IsProceed=true` 误解为无后果页面前进。

### 5.3 药水被冻结的三个事件

以下事件在 `BeforeEventStarted` 设置 `Owner.CanUseOrRemovePotions=false`，结束时恢复：

| 事件 | 当前页合法的交药动作 | 不允许追加的公共动作 |
| --- | --- | --- |
| 长者兰伟德 | 交出事件已指定药水 | 任意使用／丢弃药水来改变该指定对象 |
| 永恒之石 | `LIFT` 交出事件已指定药水 | 用药／丢药后仍执行旧 LIFT 句柄 |
| 药水的未来？ | 已展示的前至多 3 瓶交易候选 | 把全部库存、当前未展示的瓶子都加为交易对象 |

事件内部的 `PotionCmd.Discard` 是已选分支的成本，不等于玩家获得了一个公共 `DISCARD_POTION` 权限。

### 5.4 重复页面、随机生成与真实效果顺序

- 深渊浴场、滑脚木桥、无尽传送带：可以持续返回两选项页面；版本／资源／公开结果在变化，不能以选项数不变判为卡死。
- 玩偶室、色彩哲学家、打造时间：随机决定“本次展示哪些选项”，随后模型只选已展示的部分；不允许查询时再次抽取。
- 水漫缮写室：先付费，再选附魔牌；修禅织网者的移除分支：先选牌并移除，再扣费；低语空谷的拥抱：先变化，再受伤。
- 圆桌茶会的“斗殴”是事件结算；战痕假人、茂密植被、重拳出击、灯火钥匙、假商人的战斗才切换公共战斗阶段。
- 灯火钥匙的 `ModifyUnknownMapPointRoomTypes/ModifyNextEvent` 在第 3 幕把后续问号路由到战史学家。生成条件恒 false 不影响这个任务替换路径。

## 6. 静态覆盖对账与非事件内容

| 对账项 | 数量／处理 |
| --- | --- |
| 中文 `events.json` 顶层前缀 | 63 |
| 对应固定 DLL 已注册事件 | 57，本文 E01～E57，每个恰好一项 |
| 这 57 项包含的本地化 option 前缀 | 201 |
| 放入各页候选模板的 option 前缀 | 176；只是跨阶段模板数，不是任何一帧的合法候选数 |
| 单列为非候选／通用动态描述的 option 前缀 | 25，含锁定占位、未生成旧选项和完成页叙述 |
| 额外明确列出的子交互／动态候选行 | 67，补足纯文本抽取遗漏的流程 |
| 不在 57 项内的 option 前缀 | `MOCK_EVENT_MODEL.pages.INITIAL.options.TEST`，测试内容，不进入正常对局 |

其余 6 个顶层前缀分别为：`DEPRECATED_EVENT`（已删除）、`ERROR`（错误显示）、`GENERIC`（通用死亡说明）、`MOCK_EVENT_MODEL`（测试）、`PROCEED`（通用继续文字）、`SHARED_EVENT_INFO`（联机提示）。不要把它们生成成 6 个正常问号事件。

上表 176 个模板中同一 key 可能绑定多个对象（药水交易），同一 key 也可在重复页面多次出现（LINGER）。反之，文案里列出的所有颜色、所有菜或所有 rider 不会同时合法。逐项验收必须使用 `事件实例＋阶段＋选项实例＋状态版本`，不能仅靠文案 key。

## 7. 人工核对和运行验收记录模板

每个 E 编号及其子行可附以下记录，先确认动作规格，再补实现测试：

```text
事件／核对号：
源码核对：GenerateInitialOptions、后续 SetEventState、选择器／奖励／特殊 UI
场景：角色、A10、幕、公开金币／生命／牌组／药水／遗物
当前阶段与公开候选：
应该存在的动作：
必须排除的锁定／旧页／纯 UI 动作：
每条动作之后：成本时机、子选择、下一阶段、是否可能死亡
退出／跳过／取消：是否真实存在、如何承诺、是否退款
引擎版本／DLL 哈希／协议版本：
实际执行记录与结果：
人工核对结论：通过／需改；具体修改意见：
```

验收前仍需核对的工程能力包括：custom UI 等价桥接、完整公开片段编码、原子执行、异步恢复、全部奖励 alternatives、原始界面与候选集合一致性。**静态动作清单完整不等于 headless 全事件已经能运行。**

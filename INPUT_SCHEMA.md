# 新版输入契约：固定 A10，面向约 0.5B 模型

本文件定义待实现的输入字段。状态实体和引擎合法候选使用同一套字段／数值／效果编码器，再共同进入同一个约 0.5B Transformer。抽牌堆、弃牌堆、消耗牌堆均按无序多重集合输入。旧代码作为字段覆盖和边界案例的参考，不作为新版网络或训练流程的模板。下文“旧有”仅表示在相应源码中找到定义，不证明每种局面和每份 RunRecorder 记录都完整提供该字段。

## 1. 参考范围与发现

| 本地来源 | 已有能力与新版处理 |
| --- | --- |
| [observations.py](/home/nastukoi/main/sts2-model/sts_ai/observations.py) | 卡牌、角色资源、地图、商店和嵌套选择白名单；新版改用类型专属字段，取消难度输入 |
| [combat_memory.py](/home/nastukoi/main/sts2-model/sts_ai/combat_memory.py) | 入场生命、敌人稳定引用、意图与状态历史、资源统计；新版区分精确计数器与可截断历史 |
| [selection_memory.py](/home/nastukoi/main/sts2-model/sts_ai/selection_memory.py) | 选择来源、区域迁移、效果范围、选后免费；新版由统一选择上下文表示 |
| [action_contract.py](/home/nastukoi/main/sts2-model/sts_ai/action_contract.py) | 动作 verb 与 gain/remove/upgrade 等 operation；新版保留两层语义 |
| [relations.py](/home/nastukoi/main/sts2-model/sts_ai/relations.py) | 来伤、名义伤害／格挡余量、能量／金币余量；新版保留已知标志和适用条件 |
| [tokens.py](/home/nastukoi/main/sts2-model/sts_ai/tokens.py) | 未知值、词汇回退、实体区域与候选引用；新版不把每个 JSON 字段路径当长文本处理 |
| [environment.py](/home/nastukoi/main/sts2-model/sts_ai/environment.py) | 按 owner 取最近 16 条历史；新版统一记录公开动作历史，不按训练 owner 分裂观测 |
| [RunSimulator.Decisions.cs](/home/nastukoi/main/sts2-model/engine/headless/Simulation/RunSimulator.Decisions.cs) | 动态卡面、目标相关预览、升级预览、意图、角色机制、商店／事件数据 |
| [RunSimulator.CardEffects.cs](/home/nastukoi/main/sts2-model/engine/headless/Simulation/RunSimulator.CardEffects.cs) | 旧 effects 主要是关键词、费用、保留／消耗等修改；不能把它当作完整有序效果图 |

参考旧信息时需要主动修正的歧义：`upgraded` 不能替代多次升级等级；`cost` 不能同时表示能量与商店价格；`index` 不能跨区域充当实体身份；`vars` 不能替代效果语义；空集合、未知与不适用必须区分。旧版局部历史有条数／回合截断，不能据此恢复任意历史依赖机制。

## 2. 三层数据，不混作网络输入

```text
DecisionFrame
  metadata                  # 不进入网络
    fixed_ascension=10, game_build, schemas, content_hash, reward_version
    run_id, segment_id, decision_id, policy_version, teacher_source
  public                    # Actor 与 critic 的共同输入
    context, player, cards, relics, potions, creatures, powers
    character_mechanics, map, decision_surface, public_memory
    progress_ledger, public_relations
  legal                     # 候选语义进入网络；句柄仅由适配器持有
    candidate_features, entity_refs, selection_constraints
  execution                 # 不进入网络
    state_version, opaque_handle_to_command
```

每个可选字段都有 `known`，必要时另有 `applicable`；实体组标记 `complete / partial / unavailable`。缺字段不能生成零值、空集合或默认可用动作。特征的公开来源在 schema 中登记为 `visible_now / remembered / derived_public`；运行时缺失则明确置 unknown。来源标记不能区分 Steam／headless 或教师身份。

public 与 legal 是协议职责的分组，不是两条独立模型通路。张量化后为 `[VALUE / GLOBAL, public entity tokens, legal action tokens]`，所有有效 token 使用同一宽度、同一层的 attention／FFN 参数和联合关系矩阵；只有最终读出位置不同。

## 3. 全局、玩家与战斗背景

| 组 | 字段 |
| --- | --- |
| 对局 | character、act、act_floor、total_floor、当前公开房间类型／内容 ID、当前决策阶段、公开 Boss 身份 |
| 玩家 | hp、max_hp、block、gold、药水容量及空槽、永久牌组数量、遗物数量 |
| 战斗 | in_combat、round、公开回合阶段、energy、max_energy、当前公开待选择操作、各区域牌数 |
| 资源概要 | 各费用／卡牌类别／升级等级的计数，状态牌与诅咒计数；从完整公开牌组确定性计算 |
| 里程碑状态 | 已完成的幕 Boss、当前幕非 Boss 战斗奖励已用额度、公开胜利计数及记忆完整性 |

战斗中弹出选牌、事件或商店类界面时，仍保留底层战斗的手牌、资源、敌人、意图和状态；只替换 `decision_surface`。不能因为当前动词是“选择”就抹掉战斗信息。

运行时间、线程、执行器锁、UI 动画进度、训练步、未来课程起点、模型输出和教师结果不进入网络。public phase 表示规则阶段，不以瞬时 `actions_disabled` 代替。

## 4. 卡牌与牌堆

| 类别 | 具体字段与语义 |
| --- | --- |
| 身份 | content_id、card_type、rarity、upgrade_level；局部引用只用于实体连接 |
| 所属 | permanent_deck / hand / draw / discard / exhaust / resolving / selection / offered；复制体与永久牌关系只在公开可确定时保留 |
| 费用 | base_energy_cost、current_energy_cost、energy_cost_kind=fixed/x/unplayable；star_cost 采用独立数值与类型；免费标记与作用期限 |
| 卡面数值 | 命名数值的基础／当前公开值、是否是 UI 预览值、单位与作用域；不依赖字段排列顺序 |
| 修改 | enchantment / affliction 的 ID、数量与公开变量；动态增删关键词、retain_this_turn、exhaust_on_next_play、sly_this_turn、replay_count |
| 持久内容 | 经白名单核对的卡牌成长计数、自定义效果与数值；包含记录器支持的 mad_science 等公开修改 |
| 升级预览 | 当前实际实例升级后的费用、数值、关键词差异和适用性；不能用基础卡克隆覆盖该实例已有修改 |
| 效果 | 有序效果节点，绑定操作、目标、数值表达式、条件、时机、重复次数；解析覆盖状态显式可查 |
| 可用性 | 引擎公开可出牌判定、公开原因、合法目标引用、是否接受空目标；完整动作合法性由候选集决定 |
| 目标预览 | 对每个公开目标的单段／次数／总量和可信范围；只允许 UI 等价预览或公开规则计算，不运行未来随机结果 |

手牌保留公开槽位及有规则意义的相对顺序。三个牌堆分别如下：

| 牌堆 | 表示 | 需要保留 | 不输入 |
| --- | --- | --- | --- |
| draw 抽牌堆 | 无序多重集合，zone=draw | 每张已知牌的公开内容、修改、数量；未知部分的公开计数 | 未揭示的下一张、底层数组顺序、引擎索引 |
| discard 弃牌堆 | 无序多重集合，zone=discard | 每张牌的公开内容与实例差异、数量，供检索／洗回等效果读取 | 用容器插入顺序当作牌堆位置 |
| exhaust 消耗牌堆 | 无序多重集合，zone=exhaust | 每张牌的公开内容与实例差异、数量，供依赖消耗／恢复的机制读取 | 因“不容易再抽到”而省略整个牌堆 |

第一版逐牌生成 card token，使用相同 Card 内容／数值／效果编码，再加对应 zone 与 card→pile 关系。每堆另有摘要 token，包含总数、已知数、内容完整性和按公开属性统计的计数。空堆也有 count=0 的摘要；未知堆不伪装成空堆。手牌、永久牌组和三堆统一使用实体表示，但各自语义区域不同。

三堆的 card token 全部与其他状态及动作 token 共同参与注意力，不先做平均池化而丢掉牌级信息。多张完全相同的牌仍保留重复数量；例如 `Strike × 3` 与 `Strike × 1` 必须可区分。Softmax 注意力并非天然可靠的计数器，因此显式保留每堆和公开同类牌的计数。

这些 token 不使用打包行号、全局绝对位置或 RoPE；重排同一牌堆的卡牌并同步重排引用／关系后，状态语义、对应动作概率及 V 应保持不变，逐牌隐藏表示只随置换重排。对无序集合使用注意力的依据可参考 [Set Transformer](https://arxiv.org/abs/1810.00825)；这里的牌堆设计是本任务的具体选择。

明确揭示的顶部／底部或若干位置，单独表达为带有效期的公开关系；洗牌、插入或其他影响顺序的操作使相应知识更新或失效。“刚才弃掉／消耗的卡”可以从公开事件历史记住，不因此给整个弃牌／消耗堆施加隐藏的全序。

同内容牌不能随意合并：费用、附魔、升级、临时效果或可引用身份有差异时必须分开。只有所有公开语义等价且不需要分别选取时，才可压成“实体＋数量”；多选中即使表示压缩，也要能表达选取数量和映射合法实例。

永久牌组与当前战斗中的复制体是不同事实。既保留完整永久牌组用于构筑判断，也保留战斗区域；汇总统计分别计数，不能把两者相加当牌组大小。

## 5. 遗物、药水、能力与角色机制

| 类型 | 字段 |
| --- | --- |
| 遗物 | content_id、公开 counter、是否耗尽／启用、公开触发周期、效果及附加公开参数 |
| 药水 | 内容、槽位、容量、可用阶段、目标类型、合法目标、公开效果；药水消耗与丢弃在历史里分开 |
| 状态／能力 | 内容、所属实体、层数、持续时机、公开参数、已知剩余触发次数 |
| 故障充能球 | 类型、槽位顺序、当前 passive／evoke、槽位容量；暗黑球使用实际累积激发值 |
| 储君 | 当前星星、卡牌星星费用、星星 X 费语义，以及公开的相关卡牌／锻造修改 |
| 亡灵 | 奥斯提存活、hp/max_hp/block、公开能力及可见目标关系 |
| 全角色 | 不为机制硬编码固定策略；新增机制能通过卡牌、状态、计数器和关系扩展 |

公开可推导的本回合出牌数、零费攻击使用数、丢弃／消耗数、触发次数，优先从公开事件流精确维护；如果调用引擎字段作交叉核验，需要证明它只包含同样的公开历史。中途接入缺失历史时标未知，不能悄悄借用隐藏计数。

## 6. 敌人、意图与公共数值关系

每个敌人包含 content_id、公开稳定局部引用、hp/max_hp/block、alive、powers、公开目标关系与当前意图。多段意图分别保存 `damage_per_hit / hit_count / total_damage / known`，总量存在时不再重复乘次数。隐藏意图保持 unknown，不能读取 `NextMove` 中 UI 没有展示的具体动作。

敌人从列表中移除后，其他敌人的历史身份不能跟着重编号；局部引用仅用于关联，不学习整数大小。召唤、变形、复活按公开事件决定是否延续身份。

候选附加的数值关系采用命名特征：

- `nominal_damage_margin = public_damage_total - target_hp - target_block`；
- `nominal_block_margin = current_block + public_block_gain - public_incoming_damage`；
- `energy_after_cost`、`stars_after_cost`、`gold_after_price`，各自独立；
- 生命比例、药水空槽、可见牌堆数量、X 费当前可支付量；
- 升级前后和其他已知确定性修改的数值差。

每项附带已知与适用标记。上述名义余量不等于精确击杀、精确防住或真实下一状态；伤害上限、护盾、多段触发、目标重定向等可能使简单减法不适用。随机效果只编码公开分布描述或未知，不提供实际随机抽样结果。

## 7. 决策界面、选项和动作

| 界面 | 必须覆盖的信息 |
| --- | --- |
| 事件 | event_id、当前公开页／阶段、实际选项次序、稳定 option_id/text_key、公开数值、锁定条件、已揭示卡牌／遗物预览 |
| 营火 | 当前实际可用操作、公开治疗量／费用、升级及其他选择所需上下文 |
| 商店 | 商品引用、price_gold、discount、stock、卡牌 play_cost、删牌服务价格与可用状态、药水空间 |
| 奖励组 | reward_kind、可领取性、已处理／未处理、可跳过／可取消；不提前暴露未打开牌面 |
| 卡牌奖励 | 已展示的所有选项、升级状态、替代奖励、跳过、返回；保留多组奖励的公开关系 |
| 卡包／组合 | 每包成员及对应关系，选择数量范围和已公开的组合限制 |
| 嵌套选择 | operation_kind、source_kind/source_ref、source_zone、destination_zone、scope、公开提示、min/max、cancel/skip、选后费用与持续期 |
| 特殊交互 | 例如已揭示格子、工具、坐标、剩余公开次数；新交互先扩展 schema，再允许采样 |

`operation_kind` 至少区分 gain/remove/upgrade/transform/enchant/discard/exhaust/retain/retrieve/topdeck/duplicate/trade/assemble/reveal。同一个 select_item 动词必须结合该语义。每个动作的来源、目标与选项都引用状态实体，不在候选里重复堆叠整个牌组或怪物对象。

每个当前合法动作生成 action token：共享字段编码器读取 verb、operation、公开费用、确定性效果与候选预览，再以角色化关系关联 source／target／selected-option／pile。状态与动作的数值和效果语法相同，相同公开效果使用同一参数；动作类型和关系角色提供必要区分。

例如“使用药水，从弃牌堆取回某牌”中，药水、弃牌堆卡牌、牌堆摘要和 retrieve 动作均在同一主干中交互；source-potion、selected-card、source-zone=discard、destination-zone=hand、选后费用分别编码。不能把 source 和 target 先合并成无角色平均向量。

选择标签不回填到选择前输入；只有组合解码过程已经选择的前缀可以进入当前子步。公开界面位置只在选择顺序具有规则意义或用于确定规范多选序列时保留其语义，不把它当作所有 action token 的全局位置编码；不把命令序列号、对象地址或随机生成次序作为特征。候选的张量打包顺序可改变，规范多选序列不能随之改变。

联合主干中候选↔状态、候选↔候选及状态↔状态均使用双向注意力。无序候选重排后，logit 随之重排，映射到同一执行语义后的概率不变，value 不变。若多选前缀或候选集发生变化，则重新运行联合主干；输入契约不允许复用此前失效的上下文化表示。

## 8. 地图与公开历史

地图保留完整已揭示 DAG，节点类型、所在幕／层、相对当前层、当前／已访问、普通图可达、引擎当前合法可选、入出度，以及六类成对关系。未知房间保持未知类型；特殊移动能力及剩余次数来自玩家公开状态。未来节点统计沿可行图计算，不能把互斥分支的资源相加。

公开记忆分成三层：

1. **规则相关精确记忆**：本回合／本战斗计数、遗物公开周期、已知牌序、已查看奖励组与牌面、选择来源链、奖励里程碑账本。按规则的失效条件维护，不按固定条数裁掉。
2. **近期事件摘要**：最近若干公开动作、敌人已观察意图／行动、状态变化，初始预算最近 4 个已完成回合和最近 32 个语义动作。超预算截断较旧摘要并标记 truncated，不删除第 1 层的充分统计。
3. **覆盖标记**：observed_from_run_start、observed_from_combat_start、history_complete、history_truncated；中途加载不冒充完整记忆。

动作历史是模型之前实际执行且结果已公开的行为，不含对当前未执行动作的教师建议。奖励查看记忆用于比较已见选项，不因为查看本身再给奖励。

## 9. 编码、容量与验收

先以同一个小型字段／数值／效果编码器，将每个状态实体或 action 实体压成 256 维局部表示，再通过共享投影进入 1280 维空间；所有 token 的交互交给 25 层、约 0.5B 联合主干。共享编码器由 type／field／role 条件区分不同语义，不另设一个动作专用 Transformer。静态内容可缓存，动态费用、计数、升级修改和目标预览必须每个决策重建。

数值至少保留线性归一化幅值和压缩幅值，重要比例另列；字段 mask、字段类型和单位共同编码。枚举表按固定游戏内容生成，不从约 10 条示范里推断完整词表。新增未知内容先标记并扩充注册表，不能全部压成同一个 unknown ID 后继续正式训练。

主干容量初始覆盖总长 512／1024／2048 三个分桶，总长包含状态实体、三堆的逐牌实体、历史、摘要、全局／value 和全部当前候选。分别记录 S 与 A 的分布，attention 的二次项按 `(S+A)²` 估算。它们是计算配置，不是允许静默丢弃信息的上限。高于最大桶时显式扩容或隔离；记录覆盖率以免评测只保留简单局面。长效果列表先在局部编码器处理，不能按 token 上限切掉效果后半段。

需要验证的行为：

- 同一公开实体重编号／候选重排，语义概率与 V 不变；抽牌、弃牌、消耗三堆分别及联合置换，不改变对应动作概率；抽牌堆隐藏顺序改变不影响公开输入。
- 一张与多张同牌可区分；费用／升级不同的同名牌可区分；空牌堆与未知牌堆可区分；跨牌堆移动改变 zone 和相应关系。
- 双次升级、X 费为零、星星不足、费用变化、暗黑球增长、同名不同修改卡牌均能区分。
- 奖励重开保留已见内容，未见内容保持未知；弃牌与删牌有不同 operation。
- 战斗内选择仍包含底层战斗；嵌套选择不会重复统计父动作，也不泄漏最终选择。
- 多段意图只求和一次，名义余量带适用标志；未知意图不变成零来伤。
- Steam 示范与 headless 输入使用同一字段语义；逐字段统计可用率，不能假定旧引擎有字段就代表现有轨迹也有。
- 同局不同副本、不同区域和不同选择界面的引用不串位；奖励账本和精确记忆可随对局恢复。

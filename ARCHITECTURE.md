# 模型架构

本项目把 Slay the Spire 2 的一个决策表示为“公开状态 + 引擎给出的合法动作”。模型同时编码场上实体和候选动作，再用指针策略选择动作。Bootstrap、PPO 和 Steam 推理使用同一个 `PolicyValue`，没有另外一套部署模型。

核心实现分别在 `model/representation.py`、`model/model.py`、`model/policy.py`、`model/trainer.py`。本文描述当前代码实际实现的结构。

## 整条链路

```text
Steam 游戏 + steam_recorder
        │ 行动前状态、合法动作、实际选择
        ▼
原始 JSONL → recorder.PublicSnapshot → Bootstrap 轨迹
                                      │
                                      ▼
                               共享策略 / 价值模型
                                      │
                         sts2-cli 完整对局 → PPO
                                      │
                                      ▼
                              current 检查点
                                      │
                 Steam 公开状态 → 策略 argmax → 原生游戏动作
```

Bootstrap 数据来自 Steam 录制。PPO 数据来自 headless `sts2-cli` 当前策略的完整对局。Steam 实玩通过 Mod 读取原生状态，并复用 Bootstrap 导入器的公开特征转换。headless 和原生 UI 的决策边界并不完全一致，比如有些奖励查看在 headless 中自动完成；两者共享特征格式，但不能假定每一步轨迹完全相同。

## 1. 输入表示

### 决策帧

一个 `decision_frame` 分为四部分：

| 部分 | 内容 | 用途 |
| --- | --- | --- |
| `contract` | 适配器版本、状态和动作格式、A10 标记 | 确定协议 |
| `routing` | 对局、决策、状态版本、选牌 session 和 revision | 路由、缓存、拒绝过期动作 |
| `public` | 实体、关系、记忆、动作库、选择上下文 | 模型输入 |
| `legal.candidates` | 当前可执行候选及其引用 | 动作 mask |

模型不会把 seed、运行 ID、对象地址、候选编号直接当特征。引用只用于找到实体、建立关系和把模型输出映射回游戏操作。

公开实体包括玩家、卡牌、怪物、意图、能力、遗物、药水、充能球、牌堆摘要、地图节点、事件选项、商店商品、奖励和选择对象。各角色共享编码器，通过角色和实体字段区分。

白名单在 `representation.py` 定义。抽牌堆不包含内部顺序；不可见奖励、怪物内部 move、游戏 RNG、原始 UI 反射及行动后的状态不进入模型。没有公开顺序的牌堆保留完整多重集合，同名牌不会合并。手牌位置、充能球槽位和公开地图坐标则可以作为特征。

疯狂科学的类型、rider 效果和动态参数分别保留；黑球的被动值与当前累计激发伤害也分别编码。

### token 序列

序列按以下方式构造：

1. 一个 global token，携带当前决策阶段。
2. 公开实体和公开记忆 token。
3. 可选的事件上下文，以及 bundle 内每张卡牌的独立 token。
4. 当前 decoder bank 中的所有动作 token。

这里不是自然语言 tokenizer，也没有按实体排列下标添加绝对位置 embedding。实体之间的身份和拓扑由关系表示。动态选牌前缀不混入 Transformer 的静态输入，便于多选时复用编码。

### 字段编码

每个字段由符号、字段路径和五维数值向量组成：

```text
e_field = E_symbol(value) + E_field(path) + numeric_MLP(number)
entity_fields = LayerNorm(sum(e_field) / sqrt(字段数))
```

两个 embedding 查表分别编码值与字段：符号表表示值，如 `verb=PLAY_CARD`；字段表表示字段路径，如 `stats.damage`。字段路径使用稳定哈希映射到 512 个桶；符号词表容量为 16,384，保留 padding 和 unknown 两个条目。词表随检查点冻结，续训不会重新编号。

数值向量是：

```text
[x / scale, sign(x) * log(1 + abs(x)), known, applicable, is_numeric]
```

HP、金币、伤害和格挡类字段的 scale 为 100，其余为 10。未知或不适用字段有独立 mask，不等同于数值 0。非数值字段使用符号 embedding，数值部分保留 known/applicable 标记。数值 MLP 为 `5 → 256 → GELU → 256`。

字段求和保留重复项的信息，以平方根归一化控制量级。`source`、`target`、`owner` 和 `option` 引用的局部实体表示通过线性投影加入当前实体。

## 2. 局部效果编码器

每个实体可以携带 `semantic_program`。效果被拆成一棵小图，包含操作、条件、重复、选择、变量和实体绑定。程序中的父子、变量绑定、同一变量使用显式关系连接。程序顺序可以编码，实体打包顺序不能成为特征。

默认局部编码器为：

| 参数 | 数值 |
| --- | --- |
| 宽度 | 256 |
| 层数 | 2 |
| 注意力头数 | 8 |
| 每头维度 | 32 |
| FFN | 256 → 1024 → GELU → 256 |
| 关系桶数 | 128 |

程序中的实体引用绑定到对应实体的局部字段表示。经过两层带关系偏置的全注意力后，将程序首节点、按 `sqrt(节点数)` 归一化的求和池化、实体字段及引用融合结果相加，再做 LayerNorm 和 `256 → 1792` 投影。

目前许多游戏效果仍表示为 `OPAQUE_RULE + content_id`。这让模型可以根据内容 ID 和示范学习，但并不表示所有卡牌、遗物和事件的规则都已经转换为完整可执行效果程序。缺少程序的实体使用显式 unknown 节点。

## 3. 全局 Hybrid Transformer

正式配置在 `configs/rtxpro6000.json`：

| 参数 | 数值 |
| --- | --- |
| hidden size | 1792 |
| 层数 | 25 |
| 注意力头数 | 28 |
| 每头维度 | 64 |
| FFN 宽度 | 7168 |
| 调度 | `[Full, Linear] × 12 + Full` |
| Full / Linear 层数 | 13 / 12 |
| dropout | 0 |
| 总参数量 | 1,000,371,709 |
| 全局骨干参数量，含最终 LayerNorm | 964,023,788 |
| 主解码 GRU 参数量 | 19,278,336 |

每层使用 pre-norm 残差结构：

```text
x = x + Attention(LayerNorm(x))
x = x + Linear(GELU(Linear(LayerNorm(x))))
```

输出再经过最终 LayerNorm。padding 在注意力和残差输出中屏蔽。

### Full Attention

双向全注意力分数为：

```text
score(i,j,h) = Q(i,h) · K(j,h) / sqrt(head_dim)
               + relation_bias(i,j,h) + floor_bias(i,j,h)
```

同一对实体可以叠加多种关系，不会只保留其中一种。关系包括动作来源、目标、所属者、选项及其反向关系；地图还包括正反向直接连接、间接可达、不可达和 self。地图层差裁剪至 `[-16,16]`，使用 33 项 embedding。

`backend=auto` 在 CUDA 上使用编译后的 FlexAttention，CPU 使用 reference attention。显式 `sdpa` 也保留关系偏置。当前 Full 层仍构造稠密关系偏置，长序列的内存开销包含 `O(heads × N²)`，使用 FlexAttention 不会消除这个张量。

### Linear Attention

Linear 层使用 `φ(x)=ELU(x)+1`：

```text
S = Σj φ(Kj)ᵀ Vj
z = Σj φ(Kj)
output_i = φ(Qi) S / max(φ(Qi) z, 1e-6)
```

它同样是非因果的。聚合和分母在 FP32 中计算，然后转换回原来的 dtype。Linear 层没有单独的关系偏置，关系由相邻 Full 层和输入实体融合传播。

可选的 activation checkpointing 按全局 block 重算；`compile_blocks` 可以编译各 block。默认配置中两项均关闭。

## 4. GRU 指针策略与价值头

全局编码结果中，每个动作 token 对应一个动作向量。动作 key 通过 `Linear(1792,1792)` 得到。

每次解码首先生成当前选择上下文：

- 普通字段使用共享字段编码器。
- 已选对象无顺序要求时，对局部向量求和。
- 顺序有意义且已知时，使用 `GRUCell(256,256)` 依次编码已选对象。
- 前缀通过 `256 → 1792` 投影。

构造决策状态：

```text
start = global_hidden + mean(当前合法动作 hidden) + prefix
input = LayerNorm(start + previous_projection(上一个动作或 BOS))
hidden = GRUCell(input, previous_hidden)
query = Linear(hidden)
logit(action) = query · action_key / sqrt(1792)
```

非法动作的 logit 为负无穷，softmax 和指针对数概率用 FP32 计算。PPO 从完整合法分布采样；Steam 实玩和 evaluate 默认 argmax。`top-k` 只用于显示。

价值头是 `Linear(1792,1)`，输入为 `start`。一个宏动作只在第一个非强制分支计算一次价值。

如果只有一个合法动作，直接执行，不调用编码器、GRU 或价值头。强制动作仍保留在轨迹中，必要时提供下一个分支的 previous-action 语义。

### 多选 session 与缓存

一段无需揭示新信息的多选属于一个 buffered session。Transformer 编码静态实体及完整动作库，GRU 随着已选前缀逐步更新。每一步只改变前缀和合法 mask，不重新编码整张牌组。

缓存键包括 session、公开状态版本、动作库版本、输入 digest、策略版本、词表和数值精度。出现新信息或新决策边界时，重新编码并重置 session。GRU 记忆只跨当前选择 session，不跨整局游戏。

选择结束使用 `FINISH_SELECTION`；允许取消时有 `CANCEL`。Steam 控制端在冻结 offer 上完成全部前缀推理，然后把最终卡牌集合提交给游戏，游戏再执行后续效果。

## 5. Bootstrap 训练

一条训练样本是普通决策或一个完整多选宏动作。宏动作对数概率是其中所有非强制分支的和：

```text
log π(macro) = Σt log π(at | state, prefix_t)
L_bootstrap = -log π(macro)
```

样本按“角色 × 阶段”平衡。设出现过的角色阶段组数为 `G`，组内宏动作数为 `Ng`，每条样本权重为 `1/(G×Ng)`。这避免战斗中大量出牌记录完全压过地图、商店等较少出现的决策。

默认使用全部导入数据。有显式验证集时，只计算其 NLL，不参与更新。PPO 轨迹与 Bootstrap 示范有不同来源标记，录制数据不会直接充当 on-policy 数据。

每个 epoch 完成后立即追加历史、更新 `current`。记录训练配置、loss、entropy、梯度范数、更新次数、运行时间和策略版本；验证指标按角色阶段分组。Bootstrap 记录中的 KL、value MSE、clip fraction 为 0，因为没有优化这些目标。

## 6. PPO 训练

一轮由“完整采样 → 更新 → 保存”组成。每轮五个角色使用相同局数，默认各 4 局。采样器使用该轮固定的模型参数，直到整轮数据收集完才更新模型。当前实现逐局串行采样，单设备训练。

### 回报和优势

奖励由引擎确认的里程碑产生：

| 事件 | 奖励 |
| --- | --- |
| 普通或事件战斗胜利 | 0.005 |
| 精英胜利 | 0.02 |
| 每幕非 Boss 战斗累计上限 | 0.1 |
| 第一幕最终 Boss | 0.1 |
| 第二幕最终 Boss | 0.2 |
| 第三幕最终 Boss 确认通关 | 1.0 |

同一战斗和 Boss 不重复发奖。死亡没有额外负奖励，后续得不到的奖励自然体现在回报中。奖励归入最近一个非强制宏动作；首个宏动作前获得的奖励单独记为 `initial_reward`。

当前实现采用完整局 Monte Carlo return，**没有 GAE、折扣或优势标准化**：

```text
R_t = Σk>=t reward_k
A_t = R_t - V_old(s_t)
```

PPO 样本权重为 `1/(5 × 每角色局数 × horizon_scale)`，默认 `horizon_scale=100`。它控制完整局累计梯度规模，不按每局长度另行归一化。

### 优化目标

```text
ratio = exp(log π_new(macro) - log π_old(macro))
L_actor = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
L_value = (V_new - R)²
L = L_actor + 0.5 × L_value - 0.001 × entropy
```

默认 `ε=0.2`，每轮内部最多更新 2 个 epoch。熵为宏动作中各非强制分支熵的和。

更新前用当前模型重放旧策略概率和值，确保数据确实属于当前策略。近似 KL 使用 `(ratio-1)-log(ratio)`；一个逻辑 batch 的加权 KL 超过 0.015 时，停止本轮后续更新。记录总体及角色阶段分组 KL、clip fraction、value RMSE、explained variance 等指标。

可选 `bootstrap_coef` 在 PPO 逻辑 batch 中加入一个按 Bootstrap 权重采样的示范损失，默认关闭。`--value-warmup` 第一轮只训练价值头，之后必须重新采样，不能复用旧 value。

### 优化器和计算方式

使用 AdamW，backbone 学习率 `1e-5`，其余参数 `3e-5`，betas `(0.9,0.999)`，eps `1e-8`。weight decay 为 `0.01`，归一化、embedding 和一维参数不衰减。最大梯度范数为 1。

默认逻辑 batch 32，microbatch 上限 4。token/action 桶及容量预算控制分组；当前优化器逐个 session 前向和反向，累积到逻辑 batch 才 step，尚未实现高吞吐的批量 session 解码。超过桶上限的样本保持完整，不截断牌堆或合法动作。

Accelerate 管理单进程优化器和反向传播。默认 BF16 autocast，概率、Linear Attention 聚合等敏感计算保持 FP32。TF32 关闭。checkpoint 保存优化器、scheduler 和 RNG；当前 scheduler 是常数倍率，没有学习率衰减。

## 7. 检查点、训练记录和网页

Bootstrap 和 PPO 输出目录具有相同的基本结构：

```text
current/manifest.json       模型配置、冻结词表、训练配置、累计进度
current/weights-*.pt        一份模型，按大小分片
current/optimizer-*.pt      当前 AdamW 状态
current/runtime.pt         Python / Torch / CUDA RNG 和 scheduler
history.jsonl              逐轮追加的完整指标
status.json                当前阶段、进程、采样局数或错误
```

`weights-*.pt` 是同一份权重的分片，不是多个 epoch 的模型。保存先写临时目录，再通过 Linux `renameat2(RENAME_EXCHANGE)` 原子交换 `current`，随后移除旧权重。旧版任意目录名的检查点仍能加载，新的训练输出统一写 `current`。

历史记录在该轮优化结束后写入并 fsync，再保存检查点。检查点中也保存该轮的 `last_training_record`。如果保存时出错，历史中可能有已经计算完但尚未成为 `current` 的一轮；恢复进度以 `current/manifest.json` 为准。每次启动有独立 session ID，方便区分这种重试。

同一个输出目录用文件锁限制为一个训练进程。续训保留累计 epoch/PPO 轮次、优化器更新数和随机 seed 计划。采样目录编号还会跳过中断留下的目录，避免覆盖原始轨迹。

`model/monitor.py` 使用 Python 标准库 HTTP 服务。前端位于 `model/web/index.html`，无需 Node 或外部 CDN，每 3 秒读取 `/api/runs`。它扫描指定根目录下的历史，展示曲线和最近 100 条表格记录，完整日志仍保留在磁盘。PPO 图表取每轮最后一个内部 epoch；完整信息里可展开所有内部 epoch。

## 8. Steam 推理与控制

`model/steam.py` 加载 `current`，`steam_recorder/RunRecorder/LiveBridge.cs` 在 Godot 主线程轮询请求。通信目录是记录目录下的 `bridge/`，Linux 和 Proton/WSL 都只需要双方能访问这个目录。

```text
observe → 原生状态与完整候选
        → PublicSnapshot（与 Bootstrap 共用）
        → SessionPolicy / argmax
        → execute(token, action)
        → 重新核对状态及候选
        → 原生动作队列或选择界面
```

原始快照中的隐藏字段不会传给网络。模型选择的是完整动作语义，C# 侧使用临时对象引用解析成原游戏对象。出牌、地图等走游戏动作队列；营火、事件、奖励、购买调用原生 API；选牌提交到当前 UI 的异步完成对象。水晶球包含当前公开格子、工具及剩余次数。

每个请求有独立 ID，每个决策 token 只执行一次。执行前比对状态，过期动作返回 `stale` 并重新推理。游戏动画、正在执行的动作及尚未准备好的界面返回 waiting。异步操作异常会传回 Python；长时间没有可操作状态会退出。普通非决策 UI 的推进单独通过 `advance` 处理。

启动和选角色由人完成，模型接管已经进入的单人 A10 对局。退出 Python 就停止继续发出动作。录制器将桥接来源标记为 `model`，与人工示范区分。

## 9. 当前能力边界

这套实现提供数据、训练、推理控制的程序链路，不预设训练后的胜率。当前仍有几个实际限制：

- 原生 Steam 适配绑定 0.111.0，游戏更新后必须检查 API 和状态字段。
- 许多效果仍是 opaque 表示，模型需要从数据中学习其含义。
- 全局 Full Attention 的偏置仍是二次复杂度，大牌组需要更多显存。
- PPO 逐局采样和逐 session 更新，吞吐还有提升空间。
- Steam 公开状态来自录制适配器，PPO 来自 headless 适配器，存在输入和决策分布差异。
- 恢复训练从已保存轮次开始，不恢复一局打到一半的 headless 进程。

这些约束对应代码现在的行为，后续更改网络或训练算法时应同步更新这里。

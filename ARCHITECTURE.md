# 模型架构 v2

v2 使用公开状态的纯 Full Attention 骨干、SwiGLU FFN 和多选 GRU。CUDA 上采用 PyTorch FlexAttention；极小 CPU 配置使用参考计算。原 Full/Linear 交替的约十亿参数模型不再是默认配置，旧检查点不兼容。

设计取舍见 [ADR](docs/adr/0001-redesign-for-fixed-time-playing-strength.md)，目标 GPU 的验证方法见 [性能验证](docs/PERFORMANCE.md)。

## 输入与公开信息

`decision_frame` 包含 `contract`、`routing`、`public`、`legal.candidates`。模型只使用行动前公开信息；seed、内部 RNG、隐藏牌序、候选编号和对象地址不作为特征。引用用于建立实体关系与把动作输出映射回游戏操作。

公开实体和完整动作库分别成为 token。没有公开顺序的牌堆保留完整多重集合；同名牌不会合并。实体打包下标不添加位置编码。效果执行顺序、公开槽位等有意义的顺序仍需表达。

字段编码组合符号 embedding、字段路径 embedding 和五维数值向量 `[scaled_value, signed_log_value, known, applicable, is_numeric]`。字段求和并按字段数平方根归一化，未知值不等同于零。静态字段与效果结构的 CPU 解析可以缓存；学习得到的向量不可跨优化器更新缓存。

动作来源、目标、所属等引用仍进入输入编码。移除的是全局非地图 attention bias，而不是这些公开输入信息。

## 效果程序

`semantic_program` 被表示为局部节点与关系，保留程序结构、公开顺序及实体绑定。字段、引用融合和效果程序按批次计算，再汇入实体 token。局部效果编码与全局地图偏置是不同职责。

内容 ID embedding 只是输入的一部分；同一内容在升级、费用、伤害或目标变化时仍可得到不同表示。未覆盖规则继续显式表示为 opaque/unknown，不暗示程序可以完整模拟游戏。

## 全局骨干与地图偏置

全局层全部使用双向 Full Attention。地图偏置由地图节点对的类别（自身、正反向直接连接、正反向间接可达、不可达）及裁剪至 `[-16,16]` 的楼层差构成。非地图实体对不添加全局关系偏置。不可达不是 attention mask，不同路线仍能互相比较。

FlexAttention 读取紧凑地图元数据计算分数修改，不提前构造全实体、全 head 的浮点偏置。CPU/reference 和显式 SDPA 路径可构造稠密偏置，作为输出和梯度对照，不应将其显存表现当作 Flex 路径表现。

FFN 使用 `down(silu(gate(x)) * up(x))`。`ffn_size` 是 SwiGLU 的中间宽度；默认宽度约按原 GELU FFN 等参数量换算，再对齐矩阵维度。字段数值 MLP 与 GRU 的内部非线性不机械替换为 SwiGLU。

`configs/tiny.json` 用于 CPU 正确性及快速反馈；`configs/rtxpro6000.json` 是小型学习起点，而非最终规模或已测最优配置。可选 block 编译与 activation checkpointing 保留，是否提高速度需实测。

## 多选解码与批处理

GRU 的职责是一次多选过程的逐步解码，不跨整局保存游戏历史。静态实体和动作库编码可以在同一 session 中复用；已选前缀、合法动作 mask、GRU 状态各 session 独立。

单一合法动作直接执行，绕过模型头；强制动作仍进入轨迹，并保留后续选择所需的 previous-action 语义。选中动作的语义不可因动作库变化而静默丢失。

`choose_batch` 同时处理多个独立 session；`replay_batch` 按步骤推进变长 macro，仅处理尚未结束的 session。编码器和多选 GRU 实际批量执行。`replay` 和单局推理复用同一路径。

## 优化器与精度

Transformer Attention/FFN 的 Linear 权重使用 PyTorch Muon；embedding（包括地图关系表）、Norm、bias、GRU、策略头与价值头使用 AdamW。按参数职责区分，不按二维形状粗略判断。AdamW 在 CUDA 上采用 fused 实现。

Muon 与 AdamW 有独立学习率，在同一 logical batch 完成梯度累积、有限性检查及全模型梯度裁剪后更新。`optimizer=adamw` 可用于对照。检查点同时保存两种优化器的状态、scheduler 和 RNG。

默认 BF16 autocast，策略 log-prob、PPO ratio/KL 等敏感计算保持 FP32。FP8 未作为默认功能启用；Muon 内部计算精度遵循固定 PyTorch 版本的实现。采样与 replay 必须使用匹配的策略版本、词表和数值后端。

## Bootstrap 与 PPO

训练样本是一个普通决策或完整多选 macro，macro log-prob 是非强制步骤 log-prob 之和。Bootstrap 按角色与阶段平衡，PPO 按五角色等完整局数与 horizon_scale 加权。真正的 microbatch 只改变计算组织，不改变样本权重；一个 logical batch 更新一次。

奖励与 v1 保持一致：普通/事件战斗胜利 0.005、精英 0.02，每幕非 Boss 奖励最多 0.1；第一幕 Boss 0.1、第二幕 Boss 0.2、确认最终通关 1.0。事件去重记账；失败不获得未来奖励。

完整局回报使用无折扣 Monte Carlo 累积，优势为 `return - old_value`。PPO 使用 clipped ratio、价值 MSE 与 entropy；不新增 GAE 或优势归一化。宏动作首个非强制分支计算一次价值。更新前重放旧策略，检查概率与价值一致性；KL 超阈值停止后续更新。

价值预热只训练 value 头。其后重新采样；不能使用预热前的 old value 继续更新。

## 采样、评估与恢复

`--workers` 控制 CPU 引擎并发，每个引擎独占一条串行连接，中央推理队列将就绪决策提交给单个 GPU 模型。整轮模型参数固定，训练轮任一对局失败则阻止 PPO 更新，逐局 journal 和 seed 分配保留。

独立 `evaluate` 采用 argmax，使用与已知训练 seed 隔离的固定队列。评估报告五角色平均胜率、最弱角色、完成数、失败数及区间。训练采样胜率不能称作独立评估结果。

检查点格式为 2，记录模型配置、词表、优化器、scheduler、RNG 和训练进度。加载旧格式会明确要求重训。`current` 通过同文件系统原子目录交换替换；不要从未完成的临时目录加载。

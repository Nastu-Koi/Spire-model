# 模型代码与性能分析（2026-09-26）

结论：还有明确的工程加速空间，优先级是减少 padding、拆分 CPU 数据准备与模型计算、批量化解码输出，再验证编译和 GPU 内核。当前证据不支持立即替换 Full Attention 或 GRU。数据表示和性能验证体系也有不足，需要与算子优化同时处理。

本次只新增诊断脚本和报告，没有修改模型、配置、训练权重或用户已有改动。遵循 diagnosing-bugs 的先测量原则；本任务是架构审计，没有指定性能回归阈值，故没有虚构“原始故障已修复”的结论，也没有进行提交二分或生产修复。

## 1. 测量范围

- 当前配置及 bootstrap checkpoint 的架构均为 24 层、hidden=256、FFN=704、4 heads；随机实例有 20,252,609 个参数，其中全局骨干约 19.37M。不是旧的十亿参数架构。
- Python 3.11.16、PyTorch 2.14.0+cu130、CPU 4 线程；CUDA 不可用。CPU 的全局 `auto` 路径采用 reference attention，局部采用 SDPA。因此 CPU 倍数不能直接外推到 GPU FlexAttention。
- 全量容量统计覆盖 `data/steam-001/accepted.jsonl` 的 15 局、7,969 个非强制决策。
- 对照计时使用第一局固定随机抽样中的 4 个 macro；随机初始化、固定 seed、BF16、前向及反向，不执行 optimizer，不启动引擎。每种方式预热一次、测量五次。不是完整训练吞吐或胜率实验。
- 抽样容量比较使用第一局的 128 个 macro，不能代替全训练集 batch 分布。pair 数是形状代理指标，不是实测 FLOPs 或端到端加速倍数；多步 macro 的成本不能仅由首步长度概括。

复现命令（从仓库根目录运行）：

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m model --device cpu benchmark --config configs/tiny.json --steps 3 --warmup 1
/home/nastukoi/miniconda3/envs/sts2/bin/python -m model data-stats --data data/steam-001/accepted.jsonl --max-runs 32
/home/nastukoi/miniconda3/envs/sts2/bin/python docs/benchmarks/model_architecture_audit.py
```

已经执行的关键输出：tiny 合成基线平均 0.01156 s/batch；真实配置的固定 batch 结果如下。计时不包含梯度快照拷贝；单独进行输出和梯度比较。

| 方式 | 前后向中位耗时 | 说明 |
| --- | ---: | --- |
| 当前桶，pad=256 | 0.6591 s | 4 个 macro，首步 token 为 152/145/136/117 |
| 按当前编码 batch 实际最大长度填充 | 0.2982 s | 同权重、同输入、同目标 |

本地 CPU 对照约 2.21 倍、耗时减少约 55%。此前包含梯度快照的两次探索测量也观察到约减半；它们不并入此表。该小样本没有随机交错计时或置信区间，结果用于确定优化方向。

FP32 padding 对照含输出与所有非空参数梯度的 `allclose` 断言，能对不等价实现报错。BF16 记录差异，不以 FP32 阈值要求低精度位级一致。脚本没有硬编码速度通过阈值，避免把系统负载抖动误判为回归。

## 2. 优先修改：容量桶与 batch 组织

位置：`model/data.py:170–207`、`model/model.py:229–245`、`configs/rtxpro6000.json`。

真实 token 的 P50/P90/P95/P99/max 为 137/167/174/189/415；动作数为 7/13/16/23/196。当前最小 token bucket=256，使大量中短输入在全部 24 层 attention、Linear、FFN 上执行填充位置的计算。`valid` 在输出清零不等于跳过这部分矩阵计算；Flex 路径的 score modifier 也未提供用于裁剪 padding 块的 block mask。

128 个抽样 macro 的单样本最大实际长度平方之和为 2,118,501；当前按 batch 填充后的 pair 代理量为 8,388,608。不要将两者比值解释为可实现的加速倍数：真实 batch 仍需共享形状，且模型有线性复杂度部分。

对照候选桶 `[64,96,128,160,192,256,384,512,1024,2048,4096]`：

| 组织方式 | pair 代理量 |
| --- | ---: |
| 当前桶，原顺序 | 8,388,608 |
| 当前桶，logical batch 内按桶排序 | 8,388,608 |
| 细桶，原顺序 | 3,825,664 |
| 细桶，logical batch 内按桶排序 | 3,125,248 |

说明仅排序现有桶在本样本上无收益；需要同时调整粒度。建议优先对照 128/192/256 等少量桶与更细桶，在目标 GPU 统计编译次数、首次耗时、稳定吞吐后选择。不要直接把 CPU 最快的任意动态长度当作 GPU 最佳方案。

只在同一个 logical batch 内重新组织 microbatch，保持样本权重和 optimizer 更新边界。不能为了相似长度跨 logical batch 任意搬运样本后声称训练轨迹完全等价。

BF16 本次输出（log_prob/value/entropy 合计）的最大绝对差为 0.013671875，参数梯度最大绝对差为 0.03125；改变形状会改变低精度计算结果。FP32 对照通过。落地前仍需跑 rollout/replay 检查、多选缓存与梯度一致性，并在目标 GPU 验证；不能简单放宽 PPO 一致性门限掩盖问题。

## 3. CPU 预处理仍与 forward 紧密耦合

位置：`model/representation.py:251–320,354–370`、`model/model.py:73–144,210–227`、`model/policy.py:87–110`。

`choose_batch` 在判断 encoder 缓存命中前构造完整 observation；每次 replay 又清理字典、提取 fields、计算地图可达关系及 fingerprint。`field_tensors` 在模型 forward 内重新遍历字段，执行 vocabulary lookup、字段名哈希、Python padding 和张量创建。效果树已有有界解析缓存，词表 digest 也已缓存，因此不应笼统声称“完全没有缓存”。

一次带 cProfile 的本地 replay/backward：总约 0.753 s，其中 observation 约 0.035 s、field_tensors 约 0.018 s、紧凑 map 元数据构造约 0.016 s。它们是可观察到的成本，但没有超过本次 CPU backward 的约 0.516 s。cProfile 有额外开销，不能拿此表当 GPU 时间分解。

建议建立显式的 `PreparedObservation → CollatedBatch → Model` 边界：

1. 离线示范数据预计算不可训练的字段 ID、数值、局部边、binding、地图类别及长度；按词表 digest 和表示版本失效。
2. collator 负责长度分桶与张量打包；GPU 环境再测 pinned memory、异步传输与预取是否有收益。
3. 在线 session 分离静态公开状态/动作库和动态 selection context、legal mask；先验证版本后复用静态 CPU 结果。
4. 缓存有界，不能把所有原始 JSON、解析树、CPU/GPU 张量无限叠加保存。

学习得到的向量不能跨 optimizer 更新复用；session 隔离、previous-action 语义和合法动作完整性必须保留。

## 4. 解码批量化尚未贯穿整个数据流

位置：`model/model.py:262–320`、`model/policy.py:135–159`、`model/rollout.py:94–100,153–180`。

GRU 和 key/query 乘法已经批量执行，但合法动作均值仍逐 session 使用布尔索引，分布、采样、log_prob、entropy 又逐 session 计算。GPU 张量上的 `bool(legal_counts...)`、`int(sample/argmax)` 和 rollout 的 `float(value/log_prob)` 会在相应位置要求 CPU 得到结果；这是同步风险位置的代码证据，尚未测量其 GPU 占比。Categorical 默认验证也应纳入 profiler 检查，不能未经验证就删除合法性保护。

建议 decoder 返回 padded logits、mask 和批量 state；用 mask 加权求和/计数替代变长布尔抽取；批量采样或 teacher gather、批量计算 log_prob/entropy，最后统一传回所需的索引与轨迹数值。ordered prefix 目前仍逐 session/逐已选项执行 GRU，可在确认其占比后按选取位置批量推进。

InferenceQueue 已经将多个引擎汇聚到中央推理，但采用固定 2 ms 合批窗口、按 workers 限数量，没有 token budget 或长度队列，也没有传递固定 `pad_to`。建议先记录实际 batch 大小分布、排队时间、引擎等待、GPU 服务时间，再调整窗口与 worker 数；只增加 workers 不保证 GPU 更忙。不要为异步采样破坏同一 PPO 轮次固定策略版本。

## 5. 数据表示限制了局部效果编码器的作用

位置：`model/recorder.py:39–40,50–73`、`model/representation.py:233–236`、`model/model.py:109–145`。

全量容量统计中最长效果程序始终为 1、效果边为 0。第一局原始输入检查有 34,559 个实体/动作出现 program，且这些条目的 coverage 均为 opaque；其余没有 program 的条目会获得 unknown 单节点。recorder 明确用 `OPAQUE_RULE + content_id` 构造程序。这是导入表示的现状，不是本次证实的效果树解析丢失 bug。

因此，当前样本没有利用局部 attention 建模多节点程序关系的能力。字段、数值和内容 ID 仍有信息，不能据此说模型“没有任何效果信息”。在采集层补充可验证的公开效果结构，比盲目增加 local_layers 更值得验证。

速度上可实验：同一次 forward 内，按完整语义及 bindings 对相同无绑定单节点程序去重编码后 gather；必须保留梯度累加，不能跨参数更新缓存 learned embedding。有实体绑定或动态数值差异的程序不可仅按 content_id 合并。

局部程序全 batch pad 到最长程序的代码确实存在，但当前数据 max=1，不能将它列为已证实瓶颈。未来接入丰富程序后，再按程序长度分桶，并把局部 `程序数 × 最大长度²` 及字段宽度纳入容量预算。现有 token/pair budget 仅限制全局长度代理量，而且单个超预算样本仍会整体放行，不能保证不会 OOM。

## 6. 编译、骨干与测试框架

当前 `compile_blocks=false`，只有 CUDA attention 路径内部进行编译，全局层外的 Norm、FFN、residual 等尚未由该开关编译融合。先稳定输入准备和有限形状，再对照开启 block 编译；记录 graph breaks、recompile、编译时间与稳定时间。不要只测预热后的吞吐而忽略训练总时间。

`flash` 在启用 autograd 时显式回退 Triton FlexAttention。当前 `benchmark` 含 backward，即使设置 flash 也不能证明 FA4 推理加速；需要独立 no-grad 推理基准。CPU `auto` 为 reference，可另测显式 SDPA，但本次未做该对照，不宣称收益。

24 层、256 宽的串行深度值得与较浅较宽模型对照；地图节点中位数 59，占输入相当一部分，也可以实验独立地图编码与较少的跨模块交互。但这些都改变模型函数和学习能力，需要重训并比较固定总时间下五角色胜率、最弱角色和不确定性。完整合法动作、无序牌堆多重性、公开顺序必须保留。全局上下文化后的地图 hidden 依赖动态实体，不能仅因地图拓扑没变就跨决策复用。

目前无需优先改成线性 attention、替换多选 GRU、开启 activation checkpointing 或 FP8。当前真实序列多数不足 200 token；先消除填充和调度开销更有直接证据。checkpointing 是内存/重算取舍；FP8、深宽比和 attention 公式属于后续独立实验。

性能验证框架还需补足：

- `benchmark.py:92` 固定取轨迹最前面的 batch，反复计时，不能覆盖角色、阶段和长度尾部；需分层固定样本并报告分组吞吐。
- 分开报告数据准备、H2D、encoder、decoder、backward、Muon/AdamW、环境、journal/checkpoint 时间；当前日志只有训练总时间，不足以确定端到端热点。
- 补 no-grad/真实 session 推理基准、缓存命中率、batch 分布、recompile 和显存测量；训练和采样使用同样的数值策略。
- 当前工作区中模型测试文件被删除，`tests/` 不存在。文档的历史 “53 passed、5 skipped” 不能视为当前工作树验证。应按用户删除测试的意图建立新的最小验证集，本次没有擅自恢复这些文件。
- 最近的 bootstrap 日志记录 `validation_runs=0`；训练 NLL 不能代替泛化或独立游戏胜率。增加架构规模前应补齐固定验证。

## 7. 建议执行顺序

1. 细化 token 桶，在 logical batch 内按长度组织；先通过 FP32 输出/梯度与 BF16 rollout/replay 验证，再测 GPU。
2. 分离 PreparedObservation/collator；缓存纯 CPU 表示，批量化 decoder 与结果传输。
3. 在稳定形状上测 block compile、microbatch 大小、推理队列窗口与环境并发。
4. 补全效果语义数据及当前测试/benchmark 覆盖。
5. 最后做较浅较宽骨干、分层地图编码和低精度实验，以固定总时间内的独立胜率选择架构。

本次没有运行 GPU profiler、完整训练或游戏评估。因此除 CPU padding 对照外，其余项目的加速幅度均待测量，所有架构修改的胜率收益均未知。

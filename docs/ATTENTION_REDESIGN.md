# 首阶段注意力与训练执行重设计

状态：用户已确认设计并授权实现。首阶段实现和验证结果以代码、测试及 `docs/PERFORMANCE.md` 为准；列为后续候选的库和精度不代表已启用。

## 目标与已确定约束

在 RTX PRO 6000 上以固定端到端训练时间内的游戏胜率比较候选。允许全面重设计和重新训练，保留已有采集数据，不要求旧 checkpoint 兼容。GPU 总实验预算不限，决策延迟不设硬性上限；优先用小模型得到快速调试反馈。

只使用行动前公开信息，保留完整合法动作、无序牌堆的重复数量及排列不变性，以及公开且有意义的顺序。第一阶段保持奖励和训练目标不变；正确性缺陷应单独说明并修复。

## 首个基线

- 全局骨干使用小型纯 Full Attention，取消固定的 Full/Linear 交替调度。混合和双向线性注意力是后续对照候选。
- 用户已选择 SwiGLU 和 Muon。全局及局部 Transformer FFN 使用 SwiGLU；字段数值 MLP 和 GRU 不机械替换激活。为避免仅靠增加参数造成比较偏差，SwiGLU 中间宽度按约等参数量设置，再按所选内核的维度要求对齐。
- CUDA 主路径采用 PyTorch FlexAttention，小规模参考实现用于输出和梯度校验。优先复用 PyTorch 的标准层、编译和优化器；首轮不编写自定义 Triton kernel。
- 地图节点对保存紧凑拓扑类别，节点保存楼层；计算注意力时查询可学习偏置。避免为每层提前展开全实体、全 head 的浮点关系偏置。不可达关系不代表注意力屏蔽。
- 删除全局注意力中的非地图关系偏置，但保留动作来源、目标、所属等输入引用和效果程序的实体绑定。局部效果程序仍需表达连接与顺序；它不使用全局非地图关系偏置来承担这些职责。
- 保留效果程序结构和动态字段。缓存节点、连接、字段等解析结果，批量执行字段编码、实体引用融合及局部效果编码。不得跨权重更新缓存依赖权重的效果向量。
- 保留 GRU 用于多选解码，各 session 独立维护状态、合法动作和完成条件；不增加跨整局历史记忆。

## Module 职责

- 输入编码 module 接受公开决策输入，集中处理解析、缓存、张量化、效果结构和引用绑定。缓存失效规则留在该 module 的 implementation 内。
- 地图注意力 module 的 interface 接受张量与地图元数据，隐藏偏置存储和计算细节。优化实现与参考实现使用相同的信息语义。
- 多选解码 module 集中管理 GRU、已选前缀、合法动作及完成条件。调用方不负责拼接内部隐藏状态。
- 在现有 replay seam 增加批量执行能力：跨 session 并行，同一 session 内保持步骤依赖；维持样本权重、macro log-prob 求和与 logical batch 更新规则。

先实现实际批量 replay 与输入编码，再让多个 CPU 引擎进程向单个 GPU 模型提交就绪决策。本轮采样使用固定策略，完成整轮采样后再更新。减少逐样本指标回传，不取消必要的非有限值检查。

## 验收

1. 极小模型验证数据流、输出与梯度、padding 隔离、地图关系、无序实体排列不变性、效果顺序与实体绑定。
2. 对照逐 session 与批量 replay 的 log-prob、value、损失和梯度，覆盖不同长度、强制动作与多选结束；检查旧策略 replay 一致性。
3. 用真实轨迹统计 token、动作及效果程序长度，建立代表性基准，记录编译冷启动与稳定运行时间、重编译情况、吞吐及峰值显存。不可用合成 smoke 代替目标 GPU 性能证据。
4. 小型学习模型验证可拟合示范并开展游戏学习；后续扩模由学习曲线与运行数据决定。具体宽度、层数及桶大小属于可逆实验参数，不把旧模型的约十亿参数规模作为约束。
5. 示范按整局和 seed 隔离，候选使用相同的独立评估 seed，覆盖五角色；报告平均胜率、最弱角色、失败率及统计不确定性，最终测试 seed 不参与反复选型。

## 当前执行条件

本地没有可用 CUDA，当前 checkout 缺少真实引擎所需游戏程序集和构建输出。因此 CPU 正确性与合成流程可以先验证，RTX PRO 6000 性能和完整游戏胜率需要目标环境补测。已发现的历史 PPO 轨迹不能直接冒充新策略的 on-policy 数据或玩家示范；可以用于输入形状统计和离线 replay 基准，使用前仍需检查协议与数据有效性。

## 库调查依据

首阶段直接复用 PyTorch 和 Accelerate；其他框架仍为对照候选。需要比较成熟库实际替代的代码与适配成本，而不能把采用 Transformers 等同于获得批处理或算子加速。候选现成骨干必须支持双向实体编码、不引入无序实体的位置依赖，并允许接入地图偏置及验证其梯度。`PreTrainedModel` 的配置和存取能力、现成骨干复用、`Trainer` 的训练循环是三个独立选择，不要求同时采用。

倾向继续复用 PyTorch 标准层、FlexAttention、编译与 Accelerate；评估 Transformers 对配置/权重存取和骨干的实际收益，以及 TorchRL 对环境采样与轨迹组织的适配程度。仅当复用能减少维护内容且保留既定训练语义时引入，不为使用框架而复制或大幅改写其内部实现。未引入未经验证的额外 GPU 内核依赖。

内核级优化同样列入候选：优先测试 torch.compile/Inductor 和适配现有计算的 Liger 融合内核，再考虑自定义 Triton。以实际 profiler 热点和端到端收益决定采用，不假设手写内核胜过编译器。地图偏置优先通过 FlexAttention 分数修改实现；字段归约、引用融合和多选解码仅在批量化与编译后仍为显著热点时考虑专用内核。自定义内核须验证前向、反向、padding 和变长形状。

用户已决定使用 Muon，替代此前保留全量 AdamW 的建议。已确认的分组是：全局与局部 Transformer 的 attention/FFN 隐藏层矩阵使用 Muon，embedding、归一化、bias、多选 GRU 与策略/价值头保留 AdamW。按参数的职责分组而非单凭二维形状判断；Muon 和 AdamW 的学习率分别配置，两者在同一 logical batch 完成并通过有限性检查后更新。更换优化器会改变学习动态，不能视作等价内核替换。

混合精度先采用 BF16 计算基线，参数与优化器状态精度遵循选定实现，策略概率、PPO 比率/KL 和敏感归约使用 FP32。Muon 实现内部的矩阵迭代精度也须记录，不能假定全 FP32。FP8 仅作为后续大 Linear/FFN 实验，验证 rollout/replay 的数值一致性。8-bit optimizer 仅在优化器状态显存确有压力时另行评估。

Transformer Engine/torchao 的低精度训练需要核实具体 RTX PRO 6000、软件版本、形状与算子支持。TE 的 FP8 attention 不能直接当作带自定义 score_mod 的 FlexAttention 替代品；对 Linear/GEMM 的低精度替换应独立评估。不能从 Blackwell 系列名称直接推断全部内核兼容。

- [PyTorch FlexAttention](https://docs.pytorch.org/docs/2.14/nn.attention.flex_attention.html)：提供分数修改与 mask 定制；可学习偏置的实际梯度路径须在选定版本验证。
- [Flash Bidirectional Linear Attention](https://github.com/fla-org/flash-bidirectional-linear-attention)：非因果线性注意力候选；当前公开基准不能替代 RTX PRO 6000 实测，首个基线不依赖它。
- [Transformers 自定义模型](https://huggingface.co/docs/transformers/main/custom_models)：自定义配置和模型可接入存取及 AutoClass 生态；不会自动提供项目的效果编码和多选解码。
- [Transformers 注意力后端](https://huggingface.co/docs/transformers/main/attention_interface)：支持的模型可选择或注册注意力实现；具体骨干的 mask、位置处理和参数转发仍需核查。
- [TorchRL](https://docs.pytorch.org/rl/stable/index.html)：环境、采样和轨迹组织的候选设施，需验证与完整局、macro 和固定轮次策略语义的兼容性。
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)：可独立使用融合算子；大词表线性层与交叉熵优化不能直接套用到动态动作指针与 PPO 损失。
- [PyTorch Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html)：参数更新规则的候选实验。
- [自定义 Triton 与 torch.compile](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)：内核级优化的后备实现路径。

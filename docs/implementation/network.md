# network：网络、优化器、检查点

`main/network/`。设计见 [architecture](../architecture.md#全局骨干)。

| 模块 | 内容 |
| --- | --- |
| `model.py`、`attention.py` | `PolicyNetwork`：实体编码器、字段间注意力、局部程序编码器、带关系偏置的全局层、决策头 |
| `config.py` | profile 的 `model` 与 `training` 节（`ModelConfig`、`TrainConfig`），采样温度 |
| `optim.py` | Muon／AdamW 分组、BC 学习率日程（预热、查找表衰减、收尾衰减） |
| `checkpoint.py` | 检查点保存、载入、版本检查、归档 |
| `policy.py` | `choose_batch`（多会话一批选择）、`replay_batch`／`prepare_replay`（把宏动作的决策步合成一批算对数概率） |

## 配置

| | 20M（`configs/rtxpro6000.json`） | 200M（`configs/rtxpro6000-200m.json`） |
| --- | --- | --- |
| 层数、宽度、头数 | 24、256、4 | 24、768、12 |
| FFN | 704 | 2304 |
| 局部编码器 | 宽 128、4 头、2 层 | 宽 384、6 头、2 层 |
| 参数 | 约 21.3M | 约 195M |
| 层重算 | 关 | 必须开（`checkpoint_layers`） |

200M 是 20M 的三倍宽度版本：深度不变，沿关系做多跳推理的层数和学习率随深度的行为与小配置一致；局部编码器与骨干保持 1:2 宽度比，头维度 64。关系偏置是 `[批, 头, token, token]` 的稠密张量（各全局层的表并排一次查表、一次散布、一次填充建出全部层的偏置，每次散布的元素数不超过 `attention.BIAS_ELEMENTS`：采样一批几千对一次建完，训练微批到 pair 预算时退回逐层），头数增至 12 后 `pair_budget` 降到 16777216；一个 512 宏动作的逻辑批不开层重算放不进 80 GB 的进程上限，`token_budget` 取 49152，给同机运行的评估留显存。`ModelConfig.tiny()` 只用于 CPU 正确性检查。

全局层以 `torch.compile(dynamic=True)` 编译（`compile_blocks`）：宽度 256 的矩阵很小，一步要发几千个小算子，融合 norm、残差与门控比算力更要紧。编译在每个进程首次前向时进行（约一分钟），形状变化不重编。

## 优化器与学习率

| 参数组 | 优化器 | 学习率 |
| --- | --- | --- |
| Transformer 的 Attention／FFN 矩阵 | Muon | `1e-3` |
| 查找表（符号、字段名嵌入，局部与各全局层的关系偏置表） | AdamW，无权重衰减 | 起始 `1e-2`，第一轮内（`embedding_decay_updates`）几何衰减到 `1e-3`（`embedding_lr_final`）后恒定 |
| 其余骨干参数 | AdamW | 20M `1e-5` |
| 头部 | AdamW | 20M `3e-5`，200M `1e-5`（fan-in 增至三倍） |

- 日程：前 200 次更新线性预热（200M 为 1000），之后恒定；梯度裁剪 1.0；逻辑批 512 个宏动作，超出显存预算时拆成微批、梯度累加后更新一次。
- 收尾衰减：`--decay-updates N` 从 `current` 严格续训，所有参数组在 N 次更新内线性降到 0 后停止（N 取已完成更新数的 10%–20%）。衰减长度记入调度状态，中断后只能以同一长度续接；衰减完的检查点不能再续训。
- 一次运行的学习率可以单独给（`--learning-rates <json>`，`bootstrap` 与 `iterate` 都收；`network.config.LearningRates`）：四组 `muon`、`backbone`、`embedding`、`head` 的峰值，`embedding` 是查找表衰减结束后的值，有衰减时起点按原比例一起改，衰减的长度与比例不变。严格续训时优化器状态、更新计数与日程进度照旧恢复，只换学习率；保存的检查点 `training` 里写实际用的值，下次不给文件就沿用。带 `--config` 严格续训时，给了文件则 profile 里的学习率不参与一致性比较（用文件的），不给文件则 profile 的学习率必须与检查点一致；查找表衰减长度不算学习率，照旧要一致。`configs/learning-rates-grpo.json` 是幕组小步更新用的一组（BC 的十分之一）。不给文件时续训沿用衰减后的查找表 `1e-3` 和其余各组的 BC 学习率。
- 日志记各组实际学习率、裁剪前梯度范数分布与裁剪比例，每 100 次更新抽样各参数组的绝对与相对 L2 变化（embedding 的相对变化按整组范数，含本批未触及的行）。
- Muon 在 PyTorch 实现里按矩阵形状调整更新量，最优值仍可能随宽度漂移：200M 正式训练前用 `1e-3` 与 `5e-4` 两条各 2,500 次更新的探针确认。

## 精度

默认 BF16 autocast；实体编码器与决策头（query、key）在所有路径上用 FP32，对数概率与加权损失 FP32。原因：同一个决策的概率在采样时几个会话一批、不填充，训练时填充到桶；全局层在 BF16 下对相同输入逐位一致，但实体编码器的批小且大小不定，FP32 矩阵核在小批上末位不同、散射求和不保证累加顺序，这些差异进入 BF16 后被逐层放大。`encoder_precision=fp32` 可让整个编码不用 autocast。裁剪目标的比值用采样时记下的 `old_log_prob`，第一批的比值均值与 1 的偏差（约 1e-3）就是这个差异的量级。

## 检查点

每轮保存 `epoch-NNNNNN/` 并更新 `current/`：`weights.pt`、`optimizer.pt`、`runtime.pt`（随机状态）、`manifest.json`（结构与训练配置、词表、规则文本摘要、编码／历史版本、`progress`：轮次、更新数、训练过的种子 `training_seeds`、留出计数、BC 的难度与 `--combat-from` 记录、迭代记录）。只存 `state_dict`，包结构变化不影响载入。

| 启动方式 | 行为 |
| --- | --- |
| `--checkpoint <检查点> --weights-only` | 只取兼容权重，用当前配置与全新优化器、随机状态和预热计数；新输出目录 |
| `--checkpoint <目录>/current` | 严格续训：恢复优化器、随机状态和调度进度，配置与训练版本必须一致 |
| `--checkpoint <任意轮次检查点> --output <新目录>` | 从该轮严格续训到新目录（已有输出目录才要求从它自己的 `current` 续） |
| `--checkpoint … --config <profile>` | 严格续训，只把采样温度换成 profile 里的；同时给 `--learning-rates` 时 profile 的学习率不比较 |
| `--checkpoint …/current --decay-updates N --output <新目录>` | 收尾衰减 |
| 任一方式加 `--max-updates N` | 做满 N 次更新就保存并停止（可停在一轮中途），用于按固定更新数比较设置的探针 |

训练版本 `bc-decay-v5` 覆盖损失尺度、优化器分组、控制步骤过滤和衰减。检查点记结构版本，其它版本按版本报错；新训练从 `init` 开始。Ctrl+C 保留最后一个完整 epoch；只编辑 manifest 不能改已保存优化器的学习率（manifest 与 `runtime.pt` 不一致即拒绝），换学习率用 `--learning-rates`。

## 已知的坑

- 查找表的学习率只能在从初始化出发时取高值；对一个已围绕固定随机表训练过的网络，`1e-2` 会破坏已学内容。改变表的学习率要从初始化重新训练。
- 其余 AdamW 组的学习率提高十倍或百倍没有带来留出样本上的改善，这些值未作进一步筛选。

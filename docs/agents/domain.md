# Domain docs

本仓库采用 single-context：根目录 `CONTEXT.md` 维护领域术语，`docs/architecture.md` 保存系统设计与决策依据，`docs/implementation/` 保存各模块的实现方法，`docs/trials/` 保存路线总览与试错记录。

## 阅读规则

- 探索代码前读取 `CONTEXT.md`，并阅读 `docs/architecture.md` 与 `docs/implementation/` 中与当前工作相关的部分。
- 文件缺失时继续工作；术语和决策在实际确定后按需补充。
- 规格、问题描述、假设和测试使用词汇表中的领域术语。
- 与现有设计冲突时明确指出冲突及重新讨论的理由。

## 维护位置

- 功能目标、验收条件和任务进度：`.scratch/<feature>/`。
- 安装与命令：根目录 `README.md`；实现方法与已知的坑：`docs/implementation/<模块>.md`。
- 设计决策及其依据：`docs/architecture.md`；不写日期、实测记录与历史叙述。
- 试过而放弃的做法（动机、做法、实测结果、原因、教训）与路线演变：`docs/trials/`（格式见 `AGENTS.md` 的“试错记录”）；进行中的实验与原始数字在 `.scratch/`；实验的原始产物不入库。
- 代码改动在同一次改动里更新对应文档，文档描述工作区代码的现状。
- 同一内容只保留一个维护位置，其他文档使用链接引用。

# Issue tracker: Local Markdown

本仓库的任务与规格保存在工作区 `.scratch/`；“发布到 issue tracker”指写入本地文件。

## 目录与状态

- 每项功能一个目录：`.scratch/<feature-slug>/`。
- 规格的唯一维护位置为 `.scratch/<feature-slug>/spec.md`。
- 拆分后的实施工单各占一个文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，编号从 01 开始。
- 规格和工单顶部使用 `Status:` 记录 triage 标签，取值见 `triage-labels.md`。
- 若需要执行进度，另用 `Progress: pending | claimed | resolved`，与 triage 状态分开。
- 评论追加到对应文件的 `## Comments`，保留先前记录。

## 读写约定

获取任务时读取其路径；只有编号时，在当前功能的 issues 目录解析，存在歧义则确认。
发布前检查同功能的现有规格或工单，优先更新已有文件。发布后返回本地文件链接。
任务正文保留目标、验收和未完成项；使用文档链接到该任务，不复制整份规格。

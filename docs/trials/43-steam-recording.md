# Steam 实机录制做示范

- **时间**：2026-09-21 随初版代码加入；录制文件集中在 09-21 ~ 10-05；2026-10-28 删除。
- **来源**：`.scratch/steam-takeover-rename/spec.md`；删除前的代码在提交 `661af5b` 的 `steam_recorder/RunRecorder/Recorder.cs`、`Journal.cs` 与 `main/demos/recorder/journal.py`。
- **状态**：删除。Mod 只留实机接管，改名为 `SpireTakeover`（`steam_takeover/`）；`import-recorder` 命令与录制导入一并删除。

## 动机

初版设计（[01](01-1b-first-design.md)）要人类示范做 Bootstrap：在 Steam 版里正常打一局，Mod 记下每个决策前后的公开状态、合法候选与所选动作，导入成和无头引擎同一套帧契约的样本。

## 做法

- Mod（游戏内 ID `RunRecorder`）挂在出牌入队、用药、事件、营火、领奖、选择、购买、离店、宝箱、水晶球等原生入口上，按局写 JSON Lines（`sts2-run-recorder/v2`）到 `user://run_recorder`，记录动作来源（人手、控制器、规则）与证据。
- `python -m main import-recorder <目录> [--bootstrap-only]` 校验完整对局与可见性后导入（`run-recorder-import-v6`）：完整的人类对局记为 `source=demonstration`，`--bootstrap-only` 按决策导入、记为 `recorder_bc`。
- 同一个 Mod 后来加上控制桥接（实机接管），录制与接管共用快照、选择捕获和部分钩子。

## 结果

- 录下 21 个文件，约 677 MB（游戏数据目录 `run_recorder/`，2026-10-28 删除）；只够初版的约 10 条示范。
- 10-04 起示范改由 Spire Codex 公开胜局的摘要锚点重建（[05](05-replay-restoration.md)），录制导入此后不再进训练；仓库里已没有录制导入出的数据目录。

## 原因分析

人手录制一局要几十分钟，产量比摘要锚点低几个数量级；后续路线（BC 起步、之后只从自己的对局学，战斗示范由求解器从入场存档求）都不需要人类实机示范。

## 结论与后续

- 删除录制代码：Mod 侧 `Recorder`、`Journal`、`InputSnapshot`、`Pending`、`Origin`、`SelectionTrace` 与只为写记录的快照方法；Python 侧 `main/demos/recorder/journal.py`、`macros.py` 和 `import-recorder`。接管仍用的公开状态转换搬到 `main/steam/snapshot.py`，阶段表与取消规则并入 `main/steam/__init__.py`。
- 样本来源 `recorder_bc`、`demonstration` 保留：求解器 `verify` 仍写 `recorder_bc`，旧数据也可能带它们。
- 桥接协议升到 `steam-live-v5`，数据目录改为 `user://spire_takeover`。

## 教训

- 同一个 Mod 承担两种用途时，钩子和快照会长在一起；一种用途停用后要尽早拆掉，免得维护没人用的路径。

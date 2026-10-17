# Spire-model

术语见 [CONTEXT.md](CONTEXT.md)，设计见 [docs/architecture.md](docs/architecture.md)，各模块实现见 [docs/implementation/](docs/implementation/)，用法见 [README.md](README.md)。任务与工单在 `.scratch/<feature>/`（见 [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md)），它取代全局规范里的 `.claude/issue/`。

## 验证命令

Python 用 conda 环境 `sts2`（`/home/nastukoi/miniconda3/envs/sts2/bin/python`，已 editable 安装）；shell 默认的 `python` 不是项目环境。

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m pytest -q main
```

```bash
/home/nastukoi/miniconda3/envs/sts2/bin/python -m pytest -q -m 'not engine and not cuda'
```

引擎测试要独占机器、约两分钟，改引擎或求解器时才跑：

```bash
COMBAT_SOLVER_CONFIG=combat_solver_cli/lib/config.json /home/nastukoi/miniconda3/envs/sts2/bin/python -m pytest -q -m engine
```

没有 lint 与类型检查配置。

## 约定

- 仓库只放功能代码与设计依据；试跑记录、实测数字写在工单里，不进 `docs/`。
- 有训练或采样任务在跑时不在主仓库编译引擎；重编要在隔离副本里做。

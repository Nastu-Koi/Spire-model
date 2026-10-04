# 上游与本地维护

本目录是 Spire-model 内维护的 sts2-cli 源码，后续开发以此目录为准，由 Spire-model 的 Git 仓库管理。

- 上游：[wuhao21/sts2-cli](https://github.com/wuhao21/sts2-cli)
- 引入基线：`084d1aa3d8e118ca7ce8d8774ad16d6be9c92367`
- 引入日期：2026-09-20
- 许可证：[MIT](LICENSE)
- 本地维护内容：引擎决策协议（[decision-protocol.md](docs/decision-protocol.md)）、固定等待与异步选择调度及其测试。

上游 Git 元数据未嵌套复制到本目录，也不包含游戏 DLL 和构建产物；运行完整引擎仍按 [README](README.md) 准备本地依赖。

从 Spire-model 根目录运行不依赖游戏 DLL 的异步调度检查：

```bash
dotnet run --project sts2-cli/tests/PendingOperationChecks/PendingOperationChecks.csproj -c Release
```

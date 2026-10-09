# 公开历史维护

`History.cs` 只接受公开事件，不依赖游戏程序集；`NativeHistory.cs` 把 0.111.0 的公开事件接到状态机。无头引擎和 Steam Mod 直接编译同一份代码，不能各自复制修改。输入语义与版本策略见 [ARCHITECTURE](../ARCHITECTURE.md#输入与公开信息)。

`AsyncLocal` 保存原生 Add／Shuffle 调用的公开位置语义，并随原有异步续延流动。不会把随机插入的实际 index 传给状态机，也不阻塞选牌 Task。随机插入、洗牌和无法证明的内部重排保守清除牌位知识；数量和全部重复卡牌仍由原公开快照提供。

`Capture` 只输出公开历史及当前对象的引用。原始录制使用实例 ID 路由，Python 导入器将其映射成当前帧的实体引用；引用本身不成为特征。未知值带 known/applicable 掩码。加载旧摘要和旧录制不会读取存档 odds 或填上新局默认值。

规则测试包含公开前缀回放、累计概率边界、强制掉落、跨幕、缺失 tutorial 前缀恢复，以及一万次隐藏牌堆变化后的知识校验：

```bash
dotnet run --project public_history/tests/HistoryTests.csproj
python -m pytest -q model/tests/test_public_history.py
```

原生测试将两端独立编译的状态机装入同一个诊断进程，驱动原生动作并比较每帧 memory；私有 odds 和实际牌序只作为测试断言的参照，从不进入生产 `Capture`。需要提前构建 worker 和 Steam Mod，配置文件的 `worker_dll` 指向待测版本：

```bash
dotnet run --project public_history/tests/native/NativeHistoryTests.csproj -- \
  /absolute/repo /absolute/worker-config.json /absolute/RunRecorder.dll

python public_history/tests/full_runs.py /absolute/Sts2Headless.dll /tmp/history-full-runs.json
```

原生测试覆盖完整通关及事件战斗前缀、暂停选牌后的公开置顶、抽出／洗牌、上一回合意图、中途接管未知值和读档。五角色各五局脚本仅作流程回归，不报告模型通关效果。更多验证结果见 [实施报告](../.scratch/a0-training-baseline/implementation-report.md)。

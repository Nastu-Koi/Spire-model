# Spire-model

- [本地游戏引擎](sts2-cli/README.md)：源码位于 `sts2-cli/`，包含已验证的固定等待优化；[上游版本记录](sts2-cli/UPSTREAM.md)、[异步操作说明](sts2-cli/docs/background-operations.md)。
- [架构设计](ARCHITECTURE.md)：RTX PRO 6000 单卡 v1 定稿，约 1B 混合注意力＋GRU、引擎边界、动作协议与训练配置；实现和性能验收待完成。
- [硬件定稿配置](specs/architecture_rtx_pro_6000_v1.json)：模型结构、精度、批量和资源预算的机器可读设计规格。
- [动作、效果与目标编码](ACTION_ENCODING.md)：原语、数值、对象绑定、随机、顺序与后续选择；附 [21 个 JSON 示例](specs/action_semantics_v1.examples.json)。
- [全流程动作与接口映射](ACTION_CATALOG.md)：地图、战斗、药水、商店、奖励、营火、公共选择，以及现有 17 个命令的兼容缺口。
- [问号事件逐项核对](EVENT_ACTIONS.md)：57 个事件的阶段、合法动作、锁定条件和子交互；水晶球、假商人等特殊流程单列。
- [开局与远古之民](ANCIENT_ACTIONS.md)：8 位远古之民的 102 个静态选项族、子选择和结局处理。

表示层设计先看动作编码方案；逐项规则审核看问号事件表和接口映射。事件核对表基于固定版本静态检查；方框留给人工审核与后续运行验收，尚不表示引擎改造完成，也不表示完整内容已注册为效果程序。

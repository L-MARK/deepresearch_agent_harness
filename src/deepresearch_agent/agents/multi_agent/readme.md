# Multi-Agent Plan-Execute-Report

多智能体执行栈由 Harness 驱动，唯一的生产编排顺序是：

```text
RunService
  -> HarnessRuntime
    -> PlanExecuteReportDriver
      -> MultiAgentOrchestrator.plan
      -> WorkerCoordinator
        -> RetrievalExecutor / ReflectionExecutor
      -> MultiAgentOrchestrator.report
  -> Harness 验证、修复、检查点与恢复
```

`MultiAgentOrchestrator` 只负责暴露可检查点化的阶段入口。Run 生命周期、预算、暂停/取消、重规划、报告一致性修复和完成验证均由 Harness 负责，不再通过一个绕过 Harness 的 `run()` 方法串联。

## Planner

Planner 按“澄清 → 任务分解 → 计划审校”生成 `PlanSpec` 和 `PlanExecutionSignal`。`refined_query` 是澄清后的查询，属于正式输入链路，不能删除。

任务类型由 `TaskNode.task_type` 明确定义：

- `local_search`、`global_search`、`hybrid_search`、`naive_search`
- `deep_research`、`deeper_research`、`chain_exploration`
- `web_search`、`reflection`

未知任务类型会在 Planner 边界直接拒绝，不再降级成没有执行器的 `custom` 节点。执行模式只有 `sequential` 和 `parallel`；不存在实际调度器的 `adaptive` 已移除。

## Executor

`WorkerCoordinator` 根据计划信号检查依赖、选择执行器，并以串行或并行模式执行：

- `RetrievalExecutor` 统一处理图谱搜索、Web 搜索、深度研究和更深度研究任务，并接入 source policy、evidence ledger 与 retrieval provider。
- `ReflectionExecutor` 使用 `AnswerValidationTool` 校验答案；反思重试由 `WorkerCoordinator` 的正式配置控制。
- 反思相关的 `MULTI_AGENT_REFLECTION_ALLOW_RETRY` 和 `MULTI_AGENT_REFLECTION_MAX_RETRIES` 保留。

旧的 `ResearchExecutor` 与 `RetrievalExecutor` 对同一深度研究任务的重复分派已删除，避免默认列表中出现不可达的第二条链路。

## Reporter

Reporter 负责大纲、章节、证据卡、引用、Map-Reduce 和一致性检查。Harness 在报告后还会执行最终验证，必要时做定向修复或重规划。

## 当前接口

```python
from deepresearch_agent.agents.multi_agent.integration import MultiAgentFactory

bundle = MultiAgentFactory.create_default_bundle()
planner_result = bundle.orchestrator.plan(state)
records = bundle.orchestrator.execute(state, planner_result)
report = bundle.orchestrator.report(state, report_type="long_document")
```

通常不应直接调用上述阶段；后端入口应使用 `RunService`，本地脚本应使用 `harness.bootstrap.run_persistent_query`。

## 配置边界

有效的多智能体配置包括 Planner 任务上限、澄清策略、Worker 执行模式与并发数、反思重试策略，以及 Reporter 的报告和证据预算。已经删除没有生产消费者的旧配置：

- `MA_AUTO_GENERATE_REPORT`
- `MA_STOP_ON_CLARIFICATION`
- `MA_STRICT_PLAN_SIGNAL`
- `ExecutorConfig.max_retries`、`retry_delay_seconds`、`enable_reflection`

任务重试只保留 Harness 的 Run 级恢复和正式的反思重试，避免出现第二套隐式重试计数器。

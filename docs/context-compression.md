# 双阶段上下文压缩

## 运行链路

`ContextBuilder.build()` 在 Planner 之前生成 `PlanningContextPack`，并把其文本加入 `RunContext.model_input`。`PlanExecuteReportDriver` 初始化 `PlanContext` 时同时读取结构化 Pack，Planner 的 Clarifier、TaskDecomposer 和 PlanReviewer 都会收到该只读上下文。

Execute 完成后，Runtime 调用 Driver 的 `prepare_overall_reflection()`。该方法只生成 `EvidenceContextPack`，不会把 Worker 内部的局部 reflection 当成整体 reflection。当前整体 reflection 还没有语义执行器；未来实现只需注入 `OverallReflectionHook.reflect(context, evidence_context_pack)`，调用位置已经固定在 Execute 之后、Report 之前。

DeepResearch 分支也实现同一钩子。它会对当前 Run 的检索结果生成证据包；Plan-Execute-Report 分支还会合并任务记录、检索缓存和 Evidence Registry。

## 数据与幂等

迁移 `20261027_0006_context_compression.py` 创建 `session_topic_summaries`。同一个 `(session_id, run_id)` 只有一条逻辑记录，`summary_json` 保存结构化 Run 主题摘要，`source_refs_json` 保存实际覆盖过的来源，更新由程序计算字符数并使用规范化 JSON 比较，重复压缩不会插入重复记录。

原始消息、检索结果和 Evidence 不会被删除。`ContextSegment`/`ContextBlock` 保存来源 ID、内容哈希、阶段和压缩状态。内容哈希用于检测内容变化，不能替代 `source_id`。

## 记忆边界

- Session 短期记忆使用现有 `memories` 表的 `target=session`，必须绑定 `session_id`，每次 ContextBuilder 构建时动态读取，因此撤销或过期后不会继续注入。
- 用户长期记忆和项目长期记忆仍使用 `target=user/project`，继续使用原有冻结 Session 快照。
- Run 主题摘要只记录本 Run 的主题与轨迹，不替代任何记忆。
- 临时要求由 Planning 压缩器提取为候选，再由程序调用已有 Memory Policy 和容量检查后激活；失败只记录 warning，不阻塞主流程。

## 配置与迁移

可通过环境变量配置：

`CONTEXT_WINDOW_TRIGGER_RATIO`（默认 `0.80`）、`CONTEXT_WINDOW_TARGET_RATIO`（默认 `0.70`）、`CONTEXT_COMPRESSION_MAX_ROUNDS`、`CONTEXT_COMPRESSION_MODEL`、`CONTEXT_COMPRESSION_RESERVED_OUTPUT_TOKENS`、`CONTEXT_COMPRESSION_SAFETY_MARGIN_TOKENS`、`PLANNER_CONTEXT_COMPRESSION_ENABLED`、`REFLECTION_CONTEXT_COMPRESSION_ENABLED` 和 `MEMORY_CANDIDATE_EXTRACTION_ENABLED`。

在已有数据库上执行：

```powershell
.\.venv\Scripts\alembic.exe upgrade head
```

不需要删除或重新初始化数据库。

## 排查

查看 `session_topic_summaries` 的 `summary_json.metadata.phase`、`source_refs_json` 和 `summary_char_count`，确认摘要阶段、覆盖来源和字符数。Run checkpoint 中的 `planning_context_pack` 和 `evidence_context_pack` 分别对应 Planner 前与整体 Reflection 前上下文；`worker_reflection_context_pack` 只表示现有局部 Worker reflection。

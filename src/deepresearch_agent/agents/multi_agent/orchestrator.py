"""Plan-Execute-Report 阶段编排器。

HarnessRuntime 负责 Run 生命周期、检查点、恢复和验证；本类只保留三个
可检查点化的阶段入口，以及报告一致性复核入口。
"""

from typing import Any, List, Optional, Sequence

from deepresearch_agent.agents.multi_agent.core.execution_record import ExecutionRecord
from deepresearch_agent.agents.multi_agent.core.state import PlanExecuteState
from deepresearch_agent.agents.multi_agent.executor.worker_coordinator import WorkerCoordinator
from deepresearch_agent.agents.multi_agent.planner.base_planner import BasePlanner, PlannerResult
from deepresearch_agent.agents.multi_agent.reporter.base_reporter import BaseReporter, ReportResult


class MultiAgentOrchestrator:
    """Expose the stage boundaries consumed by :class:`HarnessRuntime`."""

    def __init__(
        self,
        *,
        planner: BasePlanner,
        worker_coordinator: WorkerCoordinator,
        reporter: BaseReporter,
    ) -> None:
        self._planner = planner
        self._worker = worker_coordinator
        self._reporter = reporter

    def plan(
        self,
        state: PlanExecuteState,
        *,
        assumptions: Optional[Sequence[str]] = None,
    ) -> PlannerResult:
        """Run the planner stage for Harness checkpointing."""
        return self._planner.generate_plan(
            state,
            assumptions=list(assumptions) if assumptions else None,
        )

    def execute(
        self,
        state: PlanExecuteState,
        planner_result: PlannerResult,
        *,
        stop_predicate=None,
        progress_callback=None,
    ) -> List[ExecutionRecord]:
        """Run the worker stage for the planner's execution signal."""
        signal = planner_result.executor_signal
        if signal is None:
            raise ValueError("Planner未提供执行信号，无法继续执行")
        return self._worker.execute_plan(
            state,
            signal,
            stop_predicate=stop_predicate,
            progress_callback=progress_callback,
        )

    def report(
        self,
        state: PlanExecuteState,
        *,
        report_type: Optional[str] = None,
    ) -> ReportResult:
        """Run the reporter and its consistency checker."""
        return self._reporter.generate_report(state, report_type=report_type)

    def recheck_report_consistency(
        self,
        report_content: str,
        evidence: Sequence[Any],
    ):
        """Re-run consistency checking after a targeted Harness repair."""
        return self._reporter.recheck_consistency(report_content, evidence)

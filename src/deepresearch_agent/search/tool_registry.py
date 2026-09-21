"""Registry for the source-neutral tools kept by the Web runtime."""

from typing import Any, Dict, Type

from deepresearch_agent.search.tool.base import BaseSearchTool
from deepresearch_agent.search.tool.deep_research_tool import DeepResearchTool
from deepresearch_agent.search.tool.hypothesis_tool import HypothesisGeneratorTool
from deepresearch_agent.search.tool.validation_tool import AnswerValidationTool

TOOL_REGISTRY: Dict[str, Type[BaseSearchTool]] = {
    "deep_research": DeepResearchTool,
    # Kept as a compatibility alias for persisted plans; it uses the same
    # Web-only implementation and has no separate graph execution path.
    "deeper_research": DeepResearchTool,
}

EXTRA_TOOL_FACTORIES: Dict[str, Any] = {
    "hypothesis_generator": HypothesisGeneratorTool,
    "answer_validator": AnswerValidationTool,
}

__all__ = ["TOOL_REGISTRY", "EXTRA_TOOL_FACTORIES"]

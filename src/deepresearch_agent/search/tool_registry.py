"""
搜索工具注册表

集中列出LangChain可调用的搜索类，便于多Agent层统一引用。
"""

from typing import Any, Dict, Type

from deepresearch_agent.search.tool.base import BaseSearchTool
from deepresearch_agent.search.tool.local_search_tool import LocalSearchTool
from deepresearch_agent.search.tool.global_search_tool import GlobalSearchTool
from deepresearch_agent.search.tool.hybrid_tool import HybridSearchTool
from deepresearch_agent.search.tool.naive_search_tool import NaiveSearchTool
from deepresearch_agent.search.tool.deep_research_tool import DeepResearchTool
from deepresearch_agent.search.tool.deeper_research_tool import DeeperResearchTool
from deepresearch_agent.search.tool.chain_exploration_tool import ChainOfExplorationTool
from deepresearch_agent.search.tool.hypothesis_tool import HypothesisGeneratorTool
from deepresearch_agent.search.tool.validation_tool import AnswerValidationTool

TOOL_REGISTRY: Dict[str, Type[BaseSearchTool]] = {
    "local_search": LocalSearchTool,
    "global_search": GlobalSearchTool,
    "hybrid_search": HybridSearchTool,
    "naive_search": NaiveSearchTool,
    "deep_research": DeepResearchTool,
    "deeper_research": DeeperResearchTool,
}

# 额外暴露的专用工具（不继承BaseSearchTool）
EXTRA_TOOL_FACTORIES: Dict[str, Any] = {
    "chain_exploration": ChainOfExplorationTool,
    "hypothesis_generator": HypothesisGeneratorTool,
    "answer_validator": AnswerValidationTool,
}


__all__ = [
    "TOOL_REGISTRY",
    "EXTRA_TOOL_FACTORIES",
]

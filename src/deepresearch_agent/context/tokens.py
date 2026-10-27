"""供上下文和精选 Memory 预算使用的轻量 Token 计数封装。"""

from __future__ import annotations

import re


def count_tokens(value: str, *, model: str | None = None) -> int:
    """返回确定性且偏保守的 Token 估算值。"""
    if not value:
        return 0
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(value))
    except Exception:
        cjk = len(re.findall(r"[\u3400-\u9fff]", value))
        remainder = re.sub(r"[\u3400-\u9fff]", "", value)
        latin = len(re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", remainder))
        return max(1, cjk + latin)


class ContextWindowGuard:
    """Budget calculation shared by planning and evidence context builders."""

    def __init__(
        self,
        *,
        context_window: int = 16_000,
        provider_input_limit: int | None = None,
        reserved_output_tokens: int = 2_048,
        safety_margin: int = 256,
        trigger_ratio: float = 0.80,
        target_ratio: float = 0.70,
        max_rounds: int = 2,
        model: str | None = None,
    ) -> None:
        self.context_window = max(1, context_window)
        self.provider_input_limit = provider_input_limit or self.context_window
        self.reserved_output_tokens = max(0, reserved_output_tokens)
        self.safety_margin = max(0, safety_margin)
        self.trigger_ratio = max(0.0, min(1.0, trigger_ratio))
        self.target_ratio = max(0.1, min(self.trigger_ratio, target_ratio))
        self.max_rounds = max(1, max_rounds)
        self.model = model

    @property
    def input_budget(self) -> int:
        return max(
            1,
            min(
                self.provider_input_limit,
                self.context_window - self.reserved_output_tokens - self.safety_margin,
            ),
        )

    def estimate(self, sections: list[str] | tuple[str, ...]) -> dict[str, int | float | bool]:
        full_tokens = sum(count_tokens(section, model=self.model) for section in sections if section)
        budget = self.input_budget
        ratio = full_tokens / max(1, budget)
        return {
            "input_budget": budget,
            "full_request_input_tokens": full_tokens,
            "usage_ratio": ratio,
            "triggered": ratio >= self.trigger_ratio,
            "target_tokens": max(1, int(budget * self.target_ratio)),
        }

    def fit_text(self, text: str, *, target_tokens: int | None = None) -> str:
        """Deterministic final guard; never silently removes the first/last boundary."""
        if not text:
            return ""
        target = target_tokens or max(1, int(self.input_budget * self.target_ratio))
        if count_tokens(text, model=self.model) <= target:
            return text
        # Character slicing is only a last-resort boundary guard. The semantic
        # compressors perform the meaningful selection before this method.
        ratio = target / max(1, count_tokens(text, model=self.model))
        chars = max(160, int(len(text) * ratio))
        if chars >= len(text):
            return text
        head = max(80, chars * 2 // 3)
        tail = max(40, chars - head)
        return text[:head].rstrip() + "\n…[budget-trimmed]…\n" + text[-tail:].lstrip()

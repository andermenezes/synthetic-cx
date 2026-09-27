"""Model providers implementing SyntheticCX-owned generation contracts."""

from syntheticcx.adapters.llm.openai import OpenAIProvider
from syntheticcx.adapters.llm.openai_compatible import OpenAICompatibleProvider

__all__ = ["OpenAIProvider", "OpenAICompatibleProvider"]

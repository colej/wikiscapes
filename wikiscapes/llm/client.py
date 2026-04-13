"""Anthropic API client wrapper with model-tier routing and prompt caching.

Model routing:
  tier="fast"  → claude-haiku-4-5   (cluster labels, lint checks, classification)
  tier="full"  → claude-sonnet-4-6  (ingest generation, query synthesis, bridge articles)

Prompt caching:
  cache_system=True sends the system prompt with cache_control: {"type": "ephemeral"}.
  Applied automatically to INGEST and SYNTHESIS system prompts (high repetition).
  Haiku calls do not cache (already cheap; caching has a minimum token threshold).
"""

from __future__ import annotations

from typing import Literal

import anthropic


class WikiClient:
    def __init__(
        self,
        api_key: str,
        generation_model: str = "claude-sonnet-4-6",
        fast_model: str = "claude-haiku-4-5",
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._generation_model = generation_model
        self._fast_model = fast_model

    def generate(
        self,
        prompt: str,
        *,
        tier: Literal["fast", "full"] = "full",
        max_tokens: int = 4096,
        system: str | None = None,
        cache_system: bool = False,
    ) -> str:
        """Call Claude and return the text response.

        Args:
            prompt: The user-turn message.
            tier: "fast" uses Haiku, "full" uses Sonnet.
            max_tokens: Maximum tokens to generate.
            system: Optional system prompt.
            cache_system: If True, mark the system prompt for prompt caching.
                          Only effective for "full" tier (Sonnet); ignored for Haiku.
        """
        model = self._fast_model if tier == "fast" else self._generation_model

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            if cache_system and tier == "full":
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            else:
                kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text  # type: ignore[union-attr]

    @classmethod
    def from_config(cls, config: object) -> WikiClient:
        """Construct from a Config object."""
        return cls(
            api_key=config.anthropic_api_key,  # type: ignore[attr-defined]
            generation_model=config.llm.generation_model,  # type: ignore[attr-defined]
            fast_model=config.llm.fast_model,  # type: ignore[attr-defined]
        )

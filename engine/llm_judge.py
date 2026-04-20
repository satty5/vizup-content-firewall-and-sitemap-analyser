"""
LLM judge — the contextual, nuanced reviewer.
Supports OpenAI (GPT-5.2, o3, o4-mini) and Anthropic (Opus 4.6, Sonnet 4.6).
Both standard and thinking/extended-thinking variants.

All 6 reviewers and the repair engine call through this single gateway.
Switch models via set_model() before running a review cycle.
"""

from __future__ import annotations

import json
import asyncio
from typing import Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    REVIEW_TEMPERATURE,
    MODEL_REGISTRY,
)


_openai_client: AsyncOpenAI | None = None
_anthropic_client: AsyncAnthropic | None = None

_active_provider: str = DEFAULT_PROVIDER
_active_model: str = DEFAULT_MODEL

_MODEL_TO_PROVIDER: dict[str, str] = {
    m["id"]: m["provider"] for m in MODEL_REGISTRY
}

_MODEL_META: dict[str, dict] = {m["id"]: m for m in MODEL_REGISTRY}


def set_provider(provider: str, model: str | None = None) -> None:
    """Configure which LLM provider and model all reviewers will use."""
    global _active_provider, _active_model
    provider = provider.lower().strip()
    if provider not in ("openai", "anthropic"):
        raise ValueError(f"Unknown provider '{provider}'. Use 'openai' or 'anthropic'.")
    _active_provider = provider
    if model:
        _active_model = model
    else:
        _active_model = OPENAI_MODEL if provider == "openai" else ANTHROPIC_MODEL


def set_model(model_id: str) -> None:
    """Set the active model by its ID. Provider is inferred automatically."""
    global _active_provider, _active_model
    provider = _MODEL_TO_PROVIDER.get(model_id)
    if provider:
        _active_provider = provider
        _active_model = model_id
    else:
        _active_model = model_id
        if model_id.startswith("gpt") or model_id.startswith("o"):
            _active_provider = "openai"
        elif model_id.startswith("claude"):
            _active_provider = "anthropic"


def get_active_provider() -> str:
    return _active_provider


def get_active_model() -> str:
    return _active_model


def _get_openai() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _get_anthropic() -> AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 16384,
    thinking: bool = False,
    reasoning_effort: str | None = None,
) -> str:
    client = _get_openai()

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if thinking and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    else:
        kwargs["temperature"] = temperature

    if max_tokens:
        kwargs["max_completion_tokens"] = max_tokens

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_tokens: int = 16384,
    thinking: bool = False,
    base_model: str | None = None,
) -> str:
    client = _get_anthropic()
    actual_model = base_model or model

    if thinking:
        thinking_budget = min(max_tokens // 2, 16000)
        async with client.messages.stream(
            model=actual_model,
            max_tokens=max_tokens,
            temperature=1,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget,
            },
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        ) as stream:
            response = await stream.get_final_message()
        text_parts = [
            block.text for block in response.content if block.type == "text"
        ]
        return "\n".join(text_parts)
    else:
        async with client.messages.stream(
            model=actual_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        ) as stream:
            response = await stream.get_final_message()
        text_parts = [
            block.text for block in response.content if block.type == "text"
        ]
        return "\n".join(text_parts)


async def llm_review(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
    provider: str | None = None,
    max_tokens: int = 16384,
) -> dict[str, Any]:
    """
    Run a single LLM review call through the active provider.
    Returns parsed JSON from the model.
    max_tokens defaults to 16384; callers can raise for large outputs.
    Automatically enables thinking mode for thinking model variants.
    """
    model = model or _active_model
    meta = _MODEL_META.get(model, {})
    is_thinking = meta.get("thinking", False)
    base_model = meta.get("base_model")

    provider = provider or _MODEL_TO_PROVIDER.get(model, _active_provider)
    temperature = temperature if temperature is not None else REVIEW_TEMPERATURE

    # Thinking models need extra token budget for reasoning tokens
    if is_thinking:
        max_tokens = max(max_tokens, 32768)

    if provider == "anthropic":
        raw = await _call_anthropic(
            system_prompt, user_prompt,
            model, temperature, max_tokens,
            thinking=is_thinking,
            base_model=base_model,
        )
    else:
        actual_model = base_model or model
        reasoning_effort = meta.get("reasoning_effort")
        raw = await _call_openai(
            system_prompt, user_prompt,
            actual_model, temperature, max_tokens,
            thinking=is_thinking,
            reasoning_effort=reasoning_effort,
        )

    content = _strip_code_fences(raw)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw_response": content, "parse_error": True}


async def llm_review_batch(
    calls: list[tuple[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_concurrent: int = 6,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    """Run multiple LLM review calls concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded_call(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        async with semaphore:
            return await llm_review(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                temperature=temperature,
                provider=provider,
            )

    tasks = [_bounded_call(sp, up) for sp, up in calls]
    return await asyncio.gather(*tasks)

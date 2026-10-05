"""LLM chat-completion clients for the RAG answer-generation pipeline (Scaleway, Ollama).

Every backend goes through litellm's native provider support — Scaleway via
``scaleway/<name>`` (it already knows Scaleway's base URL and auth header;
only ``api_key`` needs passing), Ollama via ``ollama_chat/<name>`` — instead of
a hand-rolled HTTP client per vendor. The OLAF tool-calling agent
(:mod:`cogmaps.ontology.agent`) uses the same ``scaleway/`` provider. Swapping
or adding a cloud provider is then a one-line change (model prefix, maybe
``api_key``) rather than a new client class.
"""
from __future__ import annotations

import logging
import re

from openai import APIConnectionError, APIError

from cogmaps.config import chat_max_tokens
from cogmaps.core.llm_impacts import tracked_completion

logger = logging.getLogger(__name__)

# Some reasoning models (DeepSeek-R1, Qwen3… on Ollama or Scaleway) inline their
# chain of thought in the answer text instead of a separate field.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove inline ``<think>…</think>`` blocks (and a dangling unclosed one) from a model answer."""
    text = _THINK_BLOCK.sub("", text)
    if re.match(r"\s*<think>", text, re.IGNORECASE):
        return ""  # reasoning cut off before the answer even started
    return text.strip()


def _chat_completion(
    *,
    vendor: str,
    model: str,
    api_base: str | None,
    api_key: str | None,
    system_prompt: str,
    user_content: str,
    timeout: int,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Shared litellm.completion call + response handling for every provider.

    ``model`` is litellm's provider-prefixed form (e.g. ``"scaleway/<name>"``,
    ``"ollama_chat/<name>"``). Raises on transport/HTTP errors or empty content.
    """
    logger.info("%s chat: model=%s prompt=%r", vendor, model, user_content[:80])
    kwargs = {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "timeout": timeout,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        response = tracked_completion(**kwargs)
    except APIConnectionError as e:
        logger.error("Cannot reach %s: %s", vendor, e)
        raise RuntimeError(f"Cannot reach {vendor}: {e}") from e
    except APIError as e:
        status = getattr(e, "status_code", "?")
        logger.error("%s API error %s: %s", vendor, status, e)
        raise RuntimeError(f"{vendor} API {status}: {e}") from e

    choice = response.choices[0]
    msg = choice.message
    # Reasoning models return their chain of thought separately
    # (reasoning_content / reasoning) — only the answer is shown to the user.
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        logger.debug("%s reasoning (not shown): %s", vendor, reasoning[:500])
    answer = strip_reasoning(msg.get("content") or "")

    if not answer:
        if reasoning and choice.finish_reason == "length":
            raise RuntimeError(
                f"{model} used its whole token budget reasoning and produced no answer — "
                "raise CHAT_MAX_TOKENS, or pick a non-reasoning model."
            )
        logger.error("%s empty content in response: %s", vendor, response)
        raise RuntimeError(f"Empty content in response: {response}")
    logger.debug("%s chat done: answer_len=%d", vendor, len(answer))
    return answer


class ScalewayClient:
    """Client for Scaleway Generative APIs, via litellm's native ``scaleway/`` provider.

    litellm already knows Scaleway's base URL and auth header for this
    provider, so only the model id and API key are passed.
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.litellm_model = f"scaleway/{model}"
        self.vendor = "Scaleway"
        logger.debug("ScalewayClient: model=%s", self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on non-200 responses or empty content."""
        return _chat_completion(
            vendor=self.vendor,
            model=self.litellm_model,
            api_base=None,
            api_key=self.api_key,
            system_prompt=system_prompt,
            user_content=user_content,
            timeout=120,
            max_tokens=chat_max_tokens(),
            temperature=0.7,
        )


class OllamaClient:
    """Client for a local Ollama server, via litellm's ``ollama_chat`` provider.

    Exposes the same ``chat(system_prompt, user_content)`` interface as
    :class:`ScalewayClient`, so the Chat page stays provider-agnostic. No API key
    is required — generation happens entirely on the local machine. Unlike
    :class:`ScalewayClient`, no ``max_tokens``/``temperature`` are forced, so
    Ollama keeps using each model's own defaults.
    """

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")
        self.litellm_model = f"ollama_chat/{model}"
        self.vendor = "Ollama"
        logger.debug("OllamaClient: host=%s model=%s", self.host, self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on transport/HTTP errors or empty content."""
        return _chat_completion(
            vendor=self.vendor,
            model=self.litellm_model,
            api_base=self.host,
            api_key=None,
            system_prompt=system_prompt,
            user_content=user_content,
            timeout=600,
        )


def build_llm_client(state):
    """Construct the chat client for the provider selected in the sidebar.

    ``state`` is a :class:`cogmaps.ui.components.SidebarState` — kept duck-typed here
    (only ``llm_provider``, ``llm_model``, ``ollama_host``, ``scaleway_api_key`` are read)
    so this module doesn't need to import the UI layer.
    """
    logger.info("Building LLM client: provider=%s model=%s", state.llm_provider, state.llm_model)
    if state.llm_provider == "ollama":
        return OllamaClient(model=state.llm_model, host=state.ollama_host)
    return ScalewayClient(model=state.llm_model, api_key=state.scaleway_api_key)

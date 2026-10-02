"""LLM chat-completion clients for the RAG answer-generation pipeline (Nebius / AI Hub, Ollama)."""
from __future__ import annotations

import logging
import re

import requests

from cogmaps.config import chat_max_tokens

logger = logging.getLogger(__name__)

# Vendor orgs (the part of "org/model-name" before the slash) routed through
# the TokenFactory endpoint; every other org uses Studio. Matching the org
# exactly (rather than a substring search over the whole model name) avoids
# accidental matches if a future model name happens to contain "openai"
# elsewhere in it.
_TOKENFACTORY_ORGS = frozenset({"moonshotai", "openai", "zai-org"})

# Some reasoning models (DeepSeek-R1, Qwen3… on Ollama or Nebius) inline their
# chain of thought in the answer text instead of a separate field.
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove inline ``<think>…</think>`` blocks (and a dangling unclosed one) from a model answer."""
    text = _THINK_BLOCK.sub("", text)
    if re.match(r"\s*<think>", text, re.IGNORECASE):
        return ""  # reasoning cut off before the answer even started
    return text.strip()


def resolve_nebius_endpoint(model: str) -> tuple[str, str]:
    """Pick the Nebius Chat Completions URL for ``model`` (TokenFactory for Kimi/GPT-OSS, Studio for the rest).

    Returns ``(api_url, vendor)``. Shared by :class:`NebiusClient` and the
    ontology-building agent (:mod:`cogmaps.ontology.agent`), which calls
    Nebius through ``litellm`` instead of this module's ``chat()``.
    """
    org = model.split("/", 1)[0]
    if org in _TOKENFACTORY_ORGS:
        return "https://api.tokenfactory.nebius.com/v1/chat/completions", "AI Hub"
    return "https://api.studio.nebius.ai/v1/chat/completions", "AI Hub Studio"


class NebiusClient:
    """Client for the Nebius / AI Hub Chat Completions endpoints.

    The endpoint is selected automatically based on the model name (TokenFactory for
    Kimi/GPT-OSS, Studio for the rest).
    """

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
        self.api_url, self.vendor = resolve_nebius_endpoint(model)
        self._user_content_is_blocks = self.vendor == "AI Hub"
        logger.debug("NebiusClient: vendor=%s model=%s", self.vendor, self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on non-200 responses or empty content."""
        logger.info("Nebius chat: model=%s prompt=%r", self.model, user_content[:80])
        user_content_payload = (
            [{"type": "text", "text": user_content}]
            if self._user_content_is_blocks
            else user_content
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content_payload},
            ],
            "max_tokens": chat_max_tokens(),
            "temperature": 0.7,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            logger.error("Cannot reach %s: %s", self.vendor, e)
            raise RuntimeError(f"Cannot reach {self.vendor}: {e}") from e
        if response.status_code != 200:
            logger.error("Nebius API error %s: %s", response.status_code, response.text[:200])
            raise RuntimeError(f"{self.vendor} API {response.status_code}: {response.text}")

        result = response.json()
        if not (result.get("choices") and result["choices"][0].get("message")):
            logger.error("Nebius unexpected response format: %s", result)
            raise RuntimeError(f"Unexpected response format: {result}")

        choice = result["choices"][0]
        msg = choice["message"]
        # Reasoning models return their chain of thought separately
        # (reasoning_content / reasoning) — only the answer is shown to the user.
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning:
            logger.debug("Nebius reasoning (not shown): %s", reasoning[:500])
        answer = strip_reasoning(msg.get("content") or "")

        if not answer:
            if reasoning and choice.get("finish_reason") == "length":
                raise RuntimeError(
                    f"{self.model} used its whole token budget reasoning and produced no answer — "
                    "raise CHAT_MAX_TOKENS, or pick a non-reasoning model."
                )
            logger.error("Nebius empty content in response: %s", result)
            raise RuntimeError(f"Empty content in response: {result}")
        logger.debug("Nebius chat done: answer_len=%d", len(answer))
        return answer


class OllamaClient:
    """Client for a local Ollama server's native chat endpoint (``/api/chat``).

    Exposes the same ``chat(system_prompt, user_content)`` interface as
    :class:`NebiusClient`, so the Chat page stays provider-agnostic. No API key
    is required — generation happens entirely on the local machine.
    """

    def __init__(self, model: str, host: str):
        self.model = model
        self.host = host.rstrip("/")
        self.api_url = f"{self.host}/api/chat"
        self.vendor = "Ollama"
        logger.debug("OllamaClient: host=%s model=%s", self.host, self.model)

    def chat(self, system_prompt: str, user_content: str) -> str:
        """Single-turn chat completion. Raises on transport/HTTP errors or empty content."""
        logger.info("Ollama chat: model=%s prompt=%r", self.model, user_content[:80])
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=600)
        except requests.RequestException as e:
            logger.error("Cannot reach Ollama at %s: %s", self.host, e)
            raise RuntimeError(f"Cannot reach Ollama at {self.host}: {e}") from e
        if response.status_code != 200:
            logger.error("Ollama API error %s: %s", response.status_code, response.text[:200])
            raise RuntimeError(f"{self.vendor} API {response.status_code}: {response.text}")

        result = response.json()
        content = strip_reasoning((result.get("message") or {}).get("content", "") or "")
        if not content:
            logger.error("Ollama empty content in response: %s", result)
            raise RuntimeError(f"Empty content in response: {result}")
        logger.debug("Ollama chat done: model=%s answer_len=%d", self.model, len(content))
        return content


def build_llm_client(state):
    """Construct the chat client for the provider selected in the sidebar.

    ``state`` is a :class:`cogmaps.ui.components.SidebarState` — kept duck-typed here
    (only ``llm_provider``, ``llm_model``, ``ollama_host``, ``nebius_api_key`` are read)
    so this module doesn't need to import the UI layer.
    """
    logger.info("Building LLM client: provider=%s model=%s", state.llm_provider, state.llm_model)
    if state.llm_provider == "ollama":
        return OllamaClient(model=state.llm_model, host=state.ollama_host)
    return NebiusClient(model=state.llm_model, api_key=state.nebius_api_key)

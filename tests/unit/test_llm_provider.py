"""Unit tests for the pluggable Chat-page LLM backend (Scaleway cloud / local Ollama)."""
from __future__ import annotations

import pytest

from litellm.types.utils import Choices, Message, ModelResponse

from cogmaps import config
from cogmaps.rag.llm_clients import OllamaClient, ScalewayClient, build_llm_client
from cogmaps.ui.components import SidebarState


def _state(**overrides) -> SidebarState:
    base = {
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "is_connected": True,
        "selected_device": "cpu",
        "llm_provider": "scaleway",
        "llm_model": "llama-3.3-70b-instruct",
        "scaleway_api_key": "key",
        "ollama_host": "http://localhost:11434",
        "llm_ready": True,
    }
    base.update(overrides)
    return SidebarState(**base)


# ── config.llm_provider ──

def test_llm_provider_defaults_to_scaleway(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.llm_provider() == "scaleway"


def test_llm_provider_normalises_case_and_whitespace(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  Ollama  ")
    assert config.llm_provider() == "ollama"


def test_llm_provider_empty_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "")
    assert config.llm_provider() == config.DEFAULT_LLM_PROVIDER


# ── config.ollama_host / ollama_models ──

def test_ollama_host_default(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert config.ollama_host() == config.DEFAULT_OLLAMA_HOST


def test_ollama_host_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    assert config.ollama_host() == "http://gpu-box:11434"


def test_ollama_models_parsing(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", " qwen2.5:7b , llama3.1 ,, ")
    assert config.ollama_models() == ["qwen2.5:7b", "llama3.1"]


def test_ollama_models_empty(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    assert config.ollama_models() == []


# ── config._int_env / qdrant_port / max_analysis_documents ──
#
# `.env.example` ships several int-typed vars as bare `NAME=` placeholders
# (e.g. MAX_ANALYSIS_DOCUMENTS=) for the user to fill in — copied as-is into
# `.env`, the var is *set* to an empty string, not absent.
# `int(os.getenv(name, default))` only falls back to `default` when the var
# is missing entirely, so these tests verify an empty-but-present var still
# falls back cleanly instead of raising `ValueError: invalid literal for
# int() with base 10: ''`.

def test_qdrant_port_falls_back_to_default_when_unset(monkeypatch):
    monkeypatch.delenv("QDRANT_PORT", raising=False)
    assert config.qdrant_port() == 6333


def test_qdrant_port_falls_back_to_default_when_set_but_empty(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "")
    assert config.qdrant_port() == 6333


def test_qdrant_port_parses_a_real_override(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "7000")
    assert config.qdrant_port() == 7000


def test_qdrant_port_falls_back_on_garbage_value(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "not-a-port")
    assert config.qdrant_port() == 6333


def test_qdrant_host_falls_back_to_default_when_set_but_empty(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "")
    assert config.qdrant_host() == "localhost"


def test_max_analysis_documents_falls_back_when_set_but_empty(monkeypatch):
    monkeypatch.setenv("MAX_ANALYSIS_DOCUMENTS", "")
    assert config.max_analysis_documents() == 3000


def test_max_analysis_documents_parses_a_real_override(monkeypatch):
    monkeypatch.setenv("MAX_ANALYSIS_DOCUMENTS", "500")
    assert config.max_analysis_documents() == 500


# ── OllamaClient ──

def test_ollama_client_strips_trailing_slash_from_host():
    client = OllamaClient(model="qwen2.5:7b", host="http://localhost:11434/")
    assert client.host == "http://localhost:11434"
    assert client.litellm_model == "ollama_chat/qwen2.5:7b"
    assert client.model == "qwen2.5:7b"


# ── build_llm_client factory ──

def test_build_llm_client_selects_ollama():
    client = build_llm_client(_state(llm_provider="ollama", llm_model="qwen2.5:7b"))
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"


def test_build_llm_client_defaults_to_scaleway():
    client = build_llm_client(_state(llm_provider="scaleway", llm_model="llama-3.3-70b-instruct",
                                      scaleway_api_key="secret-key"))
    assert isinstance(client, ScalewayClient)
    assert client.model == "llama-3.3-70b-instruct"
    assert client.api_key == "secret-key"


# ── ScalewayClient routes through litellm's native "scaleway/" provider ──

def test_scaleway_client_uses_the_native_litellm_provider(monkeypatch):
    client = ScalewayClient(model="glm-5.2", api_key="k")
    assert client.litellm_model == "scaleway/glm-5.2"

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return ModelResponse(choices=[Choices(finish_reason="stop", index=0, message=Message(content="hi"))])

    monkeypatch.setattr("litellm.completion", fake_completion)
    client.chat("sys", "q")
    assert captured["model"] == "scaleway/glm-5.2"
    assert captured["api_key"] == "k"
    # No api_base override — litellm's "scaleway/" provider already knows it.
    assert captured["api_base"] is None


# ── Reasoning is never shown in the answer ──

def _fake_completion_response(message: dict, finish_reason: str = "stop") -> ModelResponse:
    return ModelResponse(choices=[Choices(finish_reason=finish_reason, index=0, message=Message(**message))])


def _scaleway_reply(monkeypatch, message, finish_reason="stop"):
    response = _fake_completion_response(message, finish_reason)
    monkeypatch.setattr("litellm.completion", lambda **kw: response)
    return ScalewayClient(model="glm-5.2", api_key="key")


def test_strip_reasoning_removes_think_blocks():
    from cogmaps.rag.llm_clients import strip_reasoning

    assert strip_reasoning("<think>let me see</think>\n\nThe answer [Chunk 1].") == "The answer [Chunk 1]."
    assert strip_reasoning("<think>cut off mid-thought") == ""
    assert strip_reasoning("Plain answer.") == "Plain answer."


def test_scaleway_client_hides_reasoning_content(monkeypatch):
    client = _scaleway_reply(monkeypatch, {"content": "The answer [Chunk 1].", "reasoning_content": "Step 1..."})
    assert client.chat("sys", "q") == "The answer [Chunk 1]."


def test_scaleway_client_explains_reasoning_that_ate_the_budget(monkeypatch):
    client = _scaleway_reply(monkeypatch, {"content": "", "reasoning_content": "Step 1..."}, finish_reason="length")
    with pytest.raises(RuntimeError, match="token budget"):
        client.chat("sys", "q")


def test_ollama_client_strips_inline_think_block(monkeypatch):
    response = _fake_completion_response({"content": "<think>hmm</think>The answer."})
    monkeypatch.setattr("litellm.completion", lambda **kw: response)
    assert OllamaClient(model="qwen3:8b", host="http://localhost:11434").chat("sys", "q") == "The answer."

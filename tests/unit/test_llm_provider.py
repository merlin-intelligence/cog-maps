"""Unit tests for the pluggable Chat-page LLM backend (Nebius cloud / local Ollama)."""
from __future__ import annotations

from cogmaps import config
from cogmaps.rag.llm_clients import NebiusClient, OllamaClient, build_llm_client
from cogmaps.ui.components import SidebarState


def _state(**overrides) -> SidebarState:
    base = {
        "qdrant_host": "localhost",
        "qdrant_port": 6333,
        "is_connected": True,
        "selected_device": "cpu",
        "llm_provider": "nebius",
        "llm_model": "meta-llama/Llama-3.3-70B-Instruct",
        "nebius_api_key": "key",
        "ollama_host": "http://localhost:11434",
        "llm_ready": True,
    }
    base.update(overrides)
    return SidebarState(**base)


# ── config.llm_provider ──

def test_llm_provider_defaults_to_nebius(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert config.llm_provider() == "nebius"


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

def test_ollama_client_builds_native_chat_url():
    client = OllamaClient(model="qwen2.5:7b", host="http://localhost:11434/")
    assert client.api_url == "http://localhost:11434/api/chat"
    assert client.model == "qwen2.5:7b"


# ── build_llm_client factory ──

def test_build_llm_client_selects_ollama():
    client = build_llm_client(_state(llm_provider="ollama", llm_model="qwen2.5:7b"))
    assert isinstance(client, OllamaClient)
    assert client.model == "qwen2.5:7b"


def test_build_llm_client_defaults_to_nebius():
    client = build_llm_client(_state(llm_provider="nebius", llm_model="meta-llama/Llama-3.3-70B-Instruct",
                                      nebius_api_key="secret-key"))
    assert isinstance(client, NebiusClient)
    assert client.model == "meta-llama/Llama-3.3-70B-Instruct"
    assert client.api_key == "secret-key"


# ── NebiusClient endpoint routing ──

def test_nebius_client_routes_moonshotai_to_tokenfactory():
    client = NebiusClient(model="moonshotai/Kimi-K2.5-fast", api_key="k")
    assert client.api_url == "https://api.tokenfactory.nebius.com/v1/chat/completions"


def test_nebius_client_routes_other_orgs_to_studio():
    client = NebiusClient(model="meta-llama/Llama-3.3-70B-Instruct", api_key="k")
    assert client.api_url == "https://api.studio.nebius.ai/v1/chat/completions"


def test_nebius_client_org_match_is_exact_not_substring():
    # An org name that merely *contains* "openai" must not be misrouted —
    # only an exact "openai" org (before the first "/") should match.
    client = NebiusClient(model="notopenai/some-model", api_key="k")
    assert client.api_url == "https://api.studio.nebius.ai/v1/chat/completions"

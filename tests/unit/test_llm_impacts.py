"""Tests for cogmaps.core.llm_impacts — EcoLogits estimates and per-context recording."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import litellm

from cogmaps.core.llm_impacts import (
    LLMCallImpact,
    estimate_impacts,
    record_llm_impacts,
    tracked_completion,
)
from cogmaps.ontology.store import OntologyJobStore


def _response(prompt_tokens: int = 1000, completion_tokens: int = 200):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def test_known_scaleway_model_gets_an_estimate():
    call = estimate_impacts("scaleway/qwen3.8-27b", _response(), latency_s=3.0)
    assert call.estimated
    assert call.input_tokens == 1000 and call.output_tokens == 200
    assert call.energy_kwh > 0 and call.gwp_kgco2eq > 0 and call.wcf_l > 0


def test_bigger_model_has_bigger_impact():
    small = estimate_impacts("scaleway/qwen3.8-27b", _response(), latency_s=3.0)
    big = estimate_impacts("scaleway/glm-5.2", _response(), latency_s=3.0)
    assert big.energy_kwh > small.energy_kwh


def test_ollama_and_unknown_models_are_recorded_without_estimate():
    for model in ("ollama_chat/qwen2.5:7b", "scaleway/some-new-model"):
        call = estimate_impacts(model, _response(), latency_s=1.0)
        assert not call.estimated
        assert call.output_tokens == 200


def test_tracked_completion_reports_to_the_active_sink_only(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kw: _response())
    recorded: list[LLMCallImpact] = []

    tracked_completion(model="scaleway/glm-5.2", messages=[])  # no sink: nothing recorded
    with record_llm_impacts(recorded.append):
        tracked_completion(model="scaleway/glm-5.2", messages=[])
    tracked_completion(model="scaleway/glm-5.2", messages=[])

    assert [c.model for c in recorded] == ["scaleway/glm-5.2"]


def test_sink_is_reached_from_asyncio_to_thread(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kw: _response())
    recorded: list[LLMCallImpact] = []

    async def agent():
        await asyncio.to_thread(tracked_completion, model="scaleway/glm-5.2", messages=[])

    with record_llm_impacts(recorded.append):
        asyncio.run(agent())
    assert len(recorded) == 1


def test_a_failing_sink_does_not_break_the_request(monkeypatch):
    response = _response()
    monkeypatch.setattr(litellm, "completion", lambda **kw: response)

    def broken_sink(_call):
        raise RuntimeError("boom")

    with record_llm_impacts(broken_sink):
        assert tracked_completion(model="scaleway/glm-5.2", messages=[]) is response


def test_job_store_round_trips_llm_calls(tmp_path):
    store = OntologyJobStore(str(tmp_path / "jobs.db"))
    call = estimate_impacts("scaleway/glm-5.2", _response(), latency_s=2.0)
    store.append_llm_call("job-1", call.to_dict())
    store.append_llm_call("job-2", call.to_dict())

    assert [LLMCallImpact.from_dict(d) for d in store.get_llm_calls("job-1")] == [call]

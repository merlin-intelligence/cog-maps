"""Tests for cogmaps.core.llm_impacts — EcoLogits estimates and per-context recording."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import litellm
import pytest

import cogmaps.core.llm_impacts as llm_impacts
from cogmaps.core.llm_impacts import (
    BACKGROUND_RETRY,
    INTERACTIVE_RETRY,
    LLMCallImpact,
    acompletion_with_retry,
    completion_with_retry,
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


# ── Retries ──────────────────────────────────────────────────────────

def _rate_limit(retry_after: str | None = None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.scaleway.ai"))
    return litellm.RateLimitError("429 quota", "scaleway", "m", response=response)


def _scripted(monkeypatch, outcomes: list) -> list[dict]:
    """Patch litellm.completion to raise or return ``outcomes`` in order; returns the kwargs sent."""
    sent: list[dict] = []
    it = iter(outcomes)

    def fake_completion(**kwargs):
        sent.append(dict(kwargs))
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(litellm, "completion", fake_completion)
    return sent


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    waits: list[float] = []

    async def fake_async_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(llm_impacts, "_sleep", waits.append)
    monkeypatch.setattr(llm_impacts, "_async_sleep", fake_async_sleep)
    return waits


def test_retry_turns_off_the_sdk_retries(monkeypatch, sleeps):
    sent = _scripted(monkeypatch, [_response()])
    completion_with_retry(model="scaleway/glm-5.2", messages=[])
    assert sent[0]["max_retries"] == 0


def test_rate_limit_waits_what_the_provider_asks(monkeypatch, sleeps):
    _scripted(monkeypatch, [_rate_limit("7"), _response()])
    completion_with_retry(retry=BACKGROUND_RETRY, model="scaleway/glm-5.2", messages=[])
    assert sleeps == [7]


def test_rate_limit_gives_up_when_the_provider_asks_for_too_long(monkeypatch, sleeps):
    sent = _scripted(monkeypatch, [_rate_limit("120")])
    with pytest.raises(litellm.RateLimitError):
        completion_with_retry(retry=INTERACTIVE_RETRY, model="scaleway/glm-5.2", messages=[])
    assert len(sent) == 1 and sleeps == []


def test_rate_limit_backs_off_then_raises_once_the_budget_is_spent(monkeypatch, sleeps):
    sent = _scripted(monkeypatch, [_rate_limit()] * 7)
    logs: list[str] = []
    with pytest.raises(litellm.RateLimitError):
        completion_with_retry(retry=BACKGROUND_RETRY, log=logs.append, model="scaleway/glm-5.2", messages=[])
    assert len(sent) == 7
    assert sleeps == [15, 30, 60, 60, 60, 60]
    assert len(logs) == 6


def test_timeout_is_retried_with_a_longer_timeout(monkeypatch, sleeps):
    sent = _scripted(monkeypatch, [litellm.Timeout("timeout", "m", "scaleway")] * 2)
    with pytest.raises(litellm.Timeout):
        completion_with_retry(retry=BACKGROUND_RETRY, model="scaleway/glm-5.2", messages=[], timeout=90)
    assert [kw["timeout"] for kw in sent] == [90, 180]
    assert sleeps == []


def test_connection_error_is_retried_after_a_pause(monkeypatch, sleeps):
    _scripted(monkeypatch, [litellm.APIConnectionError("reset", "scaleway", "m"), _response()])
    completion_with_retry(retry=INTERACTIVE_RETRY, model="scaleway/glm-5.2", messages=[])
    assert sleeps == [10]


def test_other_errors_are_not_retried(monkeypatch, sleeps):
    sent = _scripted(monkeypatch, [litellm.BadRequestError("bad", "m", "scaleway")])
    with pytest.raises(litellm.BadRequestError):
        completion_with_retry(model="scaleway/glm-5.2", messages=[])
    assert len(sent) == 1


def test_async_retry_logs_and_waits_on_the_event_loop_thread(monkeypatch):
    import threading

    _scripted(monkeypatch, [_rate_limit("2"), _response()])
    log_threads: list[threading.Thread] = []
    sleep_threads: list[threading.Thread] = []

    async def fake_async_sleep(_seconds):
        sleep_threads.append(threading.current_thread())

    monkeypatch.setattr(llm_impacts, "_async_sleep", fake_async_sleep)
    asyncio.run(acompletion_with_retry(
        log=lambda _m: log_threads.append(threading.current_thread()),
        model="scaleway/glm-5.2", messages=[],
    ))
    assert log_threads == sleep_threads == [threading.current_thread()]


def test_cancelling_the_task_cancels_a_pending_wait(monkeypatch):
    _scripted(monkeypatch, [_rate_limit("60")])

    async def main():
        task = asyncio.create_task(acompletion_with_retry(model="scaleway/glm-5.2", messages=[]))
        await asyncio.sleep(0.2)  # the request failed, the 60s wait has started
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(asyncio.wait_for(main(), timeout=5))

"""Unit tests for cogmaps.core.llm_retry: waiting out rate limits and transient errors."""
from __future__ import annotations

import httpx
import litellm
import pytest

from cogmaps.core import llm_retry


def _rate_limit(retry_after: str | None = None) -> litellm.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.scaleway.ai"))
    return litellm.RateLimitError(
        "Quota exceeded: tokens per minute", llm_provider="scaleway", model="m", response=response,
    )


@pytest.fixture
def waits(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(llm_retry.time, "sleep", slept.append)
    return slept


def _script(monkeypatch, outcomes: list):
    """tracked_completion raises or returns each of ``outcomes`` in turn."""
    it = iter(outcomes)

    def fake(**kwargs):
        outcome = next(it)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm_retry, "tracked_completion", fake)


def test_rate_limits_are_waited_out_with_a_growing_wait(monkeypatch, waits):
    _script(monkeypatch, [_rate_limit(), _rate_limit(), _rate_limit(), "ok"])
    logs: list[str] = []

    assert llm_retry.retrying_completion(logs.append, model="m") == "ok"
    assert waits == [15, 30, 60]
    assert "tokens per minute" in logs[0]  # the provider's message says which quota


def test_retry_after_header_is_honoured(monkeypatch, waits):
    _script(monkeypatch, [_rate_limit(retry_after="45"), "ok"])
    llm_retry.retrying_completion(lambda _m: None, model="m")
    assert waits == [45]


def test_gives_up_after_the_last_rate_limit_retry(monkeypatch, waits):
    _script(monkeypatch, [_rate_limit()] * (llm_retry.RATE_LIMIT_RETRIES + 1))
    with pytest.raises(litellm.RateLimitError):
        llm_retry.retrying_completion(lambda _m: None, model="m")
    assert len(waits) == llm_retry.RATE_LIMIT_RETRIES


def test_transient_errors_are_retried_a_few_times(monkeypatch, waits):
    timeout = litellm.Timeout("slow", model="m", llm_provider="scaleway")
    _script(monkeypatch, [timeout, "ok"])
    assert llm_retry.retrying_completion(lambda _m: None, model="m") == "ok"
    assert waits == [llm_retry.TRANSIENT_WAIT]


def test_other_errors_are_not_retried(monkeypatch, waits):
    _script(monkeypatch, [ValueError("bad request")])
    with pytest.raises(ValueError):
        llm_retry.retrying_completion(lambda _m: None, model="m")
    assert waits == []

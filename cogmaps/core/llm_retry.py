"""LLM requests retried on rate limits and transient provider errors.

Scaleway's quotas are per minute (requests and tokens): a long agent loop
(ontology build, reasoning) hits them regularly, and a single HTTP 429 must
not kill a job that has been running for an hour. :func:`retrying_completion`
wraps :func:`cogmaps.core.llm_impacts.tracked_completion` — so every attempt
that gets a response is still reported — and waits the quota out.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import litellm

from cogmaps.core.llm_impacts import tracked_completion

# Waits of 15, 30, then 60 s: up to ~5 minutes in all before giving up.
RATE_LIMIT_RETRIES = 6
MAX_RATE_LIMIT_WAIT = 60
TRANSIENT_RETRIES = 2
TRANSIENT_WAIT = 10

# Errors worth retrying as-is: the provider was slow or briefly unavailable.
TRANSIENT_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def _retry_after(exc: Exception) -> float | None:
    """The provider's ``Retry-After`` header, in seconds, if it sent one."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None


def _short(exc: Exception) -> str:
    """The provider's message — it says which quota was hit (requests or tokens per minute…)."""
    return " ".join(str(exc).split())[:300]


def retrying_completion(log: Callable[[str], None], **kwargs):
    """``tracked_completion(**kwargs)``, retried on rate limits (HTTP 429) with a growing wait —
    quotas are usually per minute, so the wait goes up to a full minute, or what the provider's
    ``Retry-After`` asks — and on transient errors (timeout, connection, 5xx), retried a few
    times after a short pause. ``log`` reports each retry."""
    rate_limited = transient = 0
    while True:
        try:
            return tracked_completion(**kwargs)
        except litellm.RateLimitError as exc:
            if rate_limited == RATE_LIMIT_RETRIES:
                raise
            wait = min(15 * 2 ** rate_limited, MAX_RATE_LIMIT_WAIT)
            if (asked := _retry_after(exc)) is not None:
                wait = min(max(wait, asked), 2 * MAX_RATE_LIMIT_WAIT)
            rate_limited += 1
            log(f"Rate limited by the LLM provider — retrying in {wait:.0f}s "
                f"({rate_limited}/{RATE_LIMIT_RETRIES}): {_short(exc)}")
            time.sleep(wait)
        except TRANSIENT_ERRORS as exc:
            if transient == TRANSIENT_RETRIES:
                raise
            transient += 1
            log(f"LLM call failed ({type(exc).__name__}) — retrying in {TRANSIENT_WAIT}s "
                f"({transient}/{TRANSIENT_RETRIES}): {_short(exc)}")
            time.sleep(TRANSIENT_WAIT)

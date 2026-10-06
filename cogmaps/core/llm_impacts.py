"""Environmental impact of each LLM request, estimated with EcoLogits.

Every ``litellm.completion`` call in the app goes through :func:`tracked_completion`,
which times the request and estimates its impacts (energy, GWP, ADPe, PE, water)
with EcoLogits' methodology. Estimates are handed to whichever sink is active in
the current context — see :func:`record_llm_impacts` — so a Streamlit page (or a
background job) collects the requests it triggered without the LLM code knowing
about the UI. ``asyncio.run``/``asyncio.to_thread`` copy the context, so calls made
from the async agents are collected too.

EcoLogits' own litellm instrumentor is not used: it fuzzy-matches the response's
model name against its repository and silently picks unrelated models for the
Scaleway ids (e.g. ``qwen3-235b-a22b-instruct-2507`` → ``gpt-35-turbo-instruct``).
The models' parameter counts and Scaleway's datacenter figures are given
explicitly instead, and fed to :func:`ecologits.impacts.llm.compute_llm_impacts`.

When Langfuse keys are configured (see :func:`cogmaps.config.langfuse_enabled`),
every request is also traced to Langfuse through litellm's ``langfuse_otel`` callback.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime

import litellm
from ecologits.electricity_mix_repository import electricity_mixes
from ecologits.impacts.llm import compute_llm_impacts
from ecologits.utils.range_value import RangeValue

from cogmaps.config import langfuse_enabled

logger = logging.getLogger(__name__)

if langfuse_enabled() and "langfuse_otel" not in litellm.callbacks:
    litellm.callbacks.append("langfuse_otel")

# (total, active) parameters in billions, per Scaleway model id. Dense models have
# total == active. A model missing here is still recorded, without an estimate.
SCALEWAY_MODEL_PARAMETERS: dict[str, tuple[float, float]] = {
    "glm-5.2": (744, 40),  # EcoLogits repository (mistralai/glm-5-2)
    "qwen3.8-27b": (27.8, 27.8),  # dense
    "qwen3-235b-a22b-instruct-2507": (235, 22),
    "deepseek-v4-flash-0731": (284, 13),
    "gpt-oss-120b": (117, 5.1),  # EcoLogits repository (openai/gpt-oss-120b)
}

# Scaleway hosts every Generative APIs model in Paris (DC5, fr-par-2), which
# publishes a PUE of 1.25 and a WUE of 0.25 L/kWh.
SCALEWAY_ELECTRICITY_MIX_ZONE = "FRA"
SCALEWAY_DATACENTER_PUE = 1.25
SCALEWAY_DATACENTER_WUE = 0.25


@dataclass
class LLMCallImpact:
    """One LLM request and its estimated impacts (``None`` when not estimated)."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    energy_kwh: float | None = None
    gwp_kgco2eq: float | None = None
    adpe_kgsbeq: float | None = None
    pe_mj: float | None = None
    wcf_l: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def estimated(self) -> bool:
        return self.energy_kwh is not None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LLMCallImpact:
        return cls(**data)


_sink: ContextVar[Callable[[LLMCallImpact], None] | None] = ContextVar("llm_impact_sink", default=None)


@contextmanager
def record_llm_impacts(sink: Callable[[LLMCallImpact], None]) -> Iterator[None]:
    """Send every :func:`tracked_completion` made inside this block to ``sink``.

    ``sink`` may be called from a worker thread (``asyncio.to_thread``), so it must
    not touch Streamlit APIs — appending to a list is fine.
    """
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)


def tracked_completion(**kwargs):
    """``litellm.completion(**kwargs)``, reporting the request's impacts to the active sink."""
    start = time.perf_counter()
    response = litellm.completion(**kwargs)
    latency = time.perf_counter() - start

    sink = _sink.get()
    if sink is not None:
        try:
            sink(estimate_impacts(kwargs["model"], response, latency))
        except Exception:  # noqa: BLE001 — impact reporting must never break a request
            logger.exception("Could not estimate the impacts of an LLM request")
    return response


def estimate_impacts(litellm_model: str, response, latency_s: float) -> LLMCallImpact:
    """Estimate one request's impacts from its litellm model id, response usage and latency."""
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    call = LLMCallImpact(
        model=litellm_model, input_tokens=input_tokens,
        output_tokens=output_tokens, latency_s=latency_s,
    )

    provider, _, model = litellm_model.partition("/")
    params = SCALEWAY_MODEL_PARAMETERS.get(model) if provider == "scaleway" else None
    if params is None or output_tokens == 0:
        # Local Ollama models run on unknown hardware; unknown cloud models have no
        # parameter count to estimate from.
        return call

    mix = electricity_mixes.find_electricity_mix(zone=SCALEWAY_ELECTRICITY_MIX_ZONE)
    total, active = params
    impacts = compute_llm_impacts(
        model_active_parameter_count=active,
        model_total_parameter_count=total,
        output_token_count=output_tokens,
        request_latency=latency_s,
        if_electricity_mix_adpe=mix.adpe,
        if_electricity_mix_pe=mix.pe,
        if_electricity_mix_gwp=mix.gwp,
        if_electricity_mix_wue=mix.wue,
        datacenter_pue=SCALEWAY_DATACENTER_PUE,
        datacenter_wue=SCALEWAY_DATACENTER_WUE,
    )
    call.energy_kwh = _mean(impacts.energy.value)
    call.gwp_kgco2eq = _mean(impacts.gwp.value)
    call.adpe_kgsbeq = _mean(impacts.adpe.value)
    call.pe_mj = _mean(impacts.pe.value)
    call.wcf_l = _mean(impacts.wcf.value)
    return call


def _mean(value: float | RangeValue) -> float:
    """EcoLogits returns a min/max range when an input is uncertain — keep its midpoint."""
    if isinstance(value, RangeValue):
        return (value.min + value.max) / 2
    return float(value)

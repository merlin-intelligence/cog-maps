"""Domain discovery engine for CogMaps & OLAF.

Discovers the latent domain, epistemological nature, taxonomical pillars,
key entity types, and cross-cutting relationships of a corpus using fast,
per-document chunk sampling and an LLM synthesis call
(``DOMAIN_DISCOVERY_MODEL``, qwen3.8-27b by default).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from datetime import datetime
from typing import Any

from openai import APIConnectionError, APIError
from pydantic import BaseModel, Field
from qdrant_client import models

from cogmaps.config import (
    domain_discovery_model,
    domain_profiles_dir,
    scaleway_api_key,
)
from cogmaps.core.llm_impacts import tracked_completion
from cogmaps.ontology.olaf_config import sanitize_ontology_id
from cogmaps.qdrant.store import QdrantStore

logger = logging.getLogger(__name__)


# ─── Data Models ─────────────────────────────────────────────────────────────

class DomainPillar(BaseModel):
    """A distinct taxonomical dimension of the domain."""
    name: str = Field(description="Name of the taxonomical domain dimension")
    target_parent_class: str = Field(
        default="",
        description="Universal OWL parent class to anchor concepts under (e.g. Asset, LegalNorm, CognitiveProcess)",
    )
    sample_classes: list[str] = Field(
        default_factory=list,
        description="8-12 candidate concepts belonging to this dimension",
    )
    typical_relations: list[str] = Field(
        default_factory=list,
        description="Expected relationship predicates connecting or characterizing these concepts",
    )


class DomainBlueprint(BaseModel):
    """Complete domain architecture blueprint synthesized by domain discovery."""
    inferred_domain: str = Field(description="High-level identity of the corpus")
    epistemological_nature: str = Field(
        default="Interdisciplinary",
        description="Epistemological nature of knowledge (e.g. Empirical, Deontic, Interdisciplinary)",
    )
    pillars: list[DomainPillar] = Field(
        default_factory=list,
        description="6-7 structured domain dimensions",
    )
    individual_types: list[str] = Field(
        default_factory=list,
        description="Entity categories for owl:NamedIndividual (e.g. Theorist, Agency, Specific Index)",
    )
    suggested_object_properties: list[str] = Field(
        default_factory=list,
        description="Key OWL object properties connecting classes across the domain",
    )
    cross_cutting_themes: list[str] = Field(
        default_factory=list,
        description="Cross-pillar relationships to prevent siloed, disconnected sub-graphs",
    )
    generated_at: str | None = Field(
        default=None,
        description="ISO timestamp of the discovery run that produced this blueprint (None if hand-made)",
    )
    source_document_count: int | None = Field(
        default=None,
        description="Number of documents in the collection when discovery ran (None if hand-made)",
    )


# ─── Stage 1: Stratified Corpus Profiling ───────────────────────────────────

_SAMPLE_PAYLOAD = ["filename", "chunk_number", "text"]


def _random_points(store: QdrantStore, collection: str, limit: int, flt: models.Filter | None = None) -> list:
    """Draw ``limit`` random points (optionally filtered) from a collection.

    Uses Qdrant's native random sampling (server >= 1.11). Falls back to a
    plain scroll — i.e. the first points in storage order — on older servers.
    """
    try:
        return store.client.query_points(
            collection_name=collection,
            query=models.SampleQuery(sample=models.Sample.RANDOM),
            query_filter=flt,
            limit=limit,
            with_payload=_SAMPLE_PAYLOAD,
            with_vectors=False,
        ).points
    except Exception:  # noqa: BLE001
        logger.warning("Random sampling unavailable on Qdrant, falling back to scroll", exc_info=True)
        points, _ = store.client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=_SAMPLE_PAYLOAD,
            with_vectors=False,
        )
        return points


def _to_sample(point, default_doc: str, char_limit: int) -> dict[str, Any]:
    payload = point.payload or {}
    return {
        "doc": payload.get("filename", default_doc),
        "chunk_number": payload.get("chunk_number", 0),
        "text": str(payload.get("text", ""))[:char_limit],
    }


def sample_corpus_for_discovery(
    store: QdrantStore,
    collection: str,
    max_chunks: int = 15,
    char_limit_per_chunk: int = 500,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Sample document filenames and random text chunks across a collection.

    Pulls 1 randomly chosen chunk per document (up to max_chunks docs) rather
    than each document's first chunk, which is often a title page or table of
    contents. Chunks are drawn at random on every call, so two discoveries on
    the same collection may see different excerpts.
    """
    all_filenames = sorted(store.existing_filenames(collection))
    if not all_filenames:
        return [], []

    sampled_filenames: list[str]
    if len(all_filenames) <= max_chunks:
        sampled_filenames = all_filenames
    else:
        sampled_filenames = random.Random(seed).sample(all_filenames, max_chunks)

    # If only 1 document is present, sample multiple random chunks across it
    if len(all_filenames) == 1:
        points = _random_points(store, collection, limit=max_chunks)
        samples = [_to_sample(p, all_filenames[0], char_limit_per_chunk) for p in points]
        samples.sort(key=lambda s: s["chunk_number"])
        return all_filenames, samples

    samples: list[dict[str, Any]] = []
    for doc in sampled_filenames:
        flt = models.Filter(must=[models.FieldCondition(key="filename", match=models.MatchValue(value=doc))])
        points = _random_points(store, collection, limit=1, flt=flt)
        if points:
            samples.append(_to_sample(points[0], doc, char_limit_per_chunk))

    return all_filenames, samples


# ─── Stage 2 & 3: Meta-Discovery Synthesizer Prompt ──────────────────────────

DISCOVERY_SYSTEM_PROMPT = """You are an expert ontology engineer and taxonomist specializing in domain discovery and knowledge modeling.
Your mission is domain discovery. Given a corpus profile (collection name, sampled document filenames, and representative text excerpts), you analyze the latent themes, identify the true domain (looking past misleading, metaphorical, or colloquial collection names), and construct a rigorous taxonomical domain blueprint.

## Methodological Grounding & Epistemological Types
Domains generally fall into one of three epistemological categories:
1. Empirical / Dynamic (e.g., Quantitative Finance, Economics, Climate Science):
   - Defined by observable state variables, quantitative metrics, mechanisms, instruments, causal feedback loops.
   - Example Archetype (a macroeconomic research corpus):
     Pillars: Financial Instruments (parent: Asset), Yield Curve & Rates (parent: Metric/Rate), Central Bank Mechanisms (parent: MonetaryOperation), Fiscal Operations (parent: SovereignDebtOperation), Macro Indicators (parent: MacroeconomicIndicator), Market Regimes (parent: MarketState).
     Object properties: drives, absorbsSupplyOf, hedgesAgainst, setsPolicyRate, impactsInflation.
2. Deontic / Hierarchical (e.g., Administrative Law, Regulatory Compliance, Public Governance):
   - Defined by statutory authorities, normative hierarchies, legal procedures, jurisdictions, compliance rules.
   - Example Archetype (an administrative law corpus):
     Pillars: Normative Acts (parent: LegalNorm), Administrative Authorities (parent: PublicAuthority), Legal Procedures (parent: ProceduralRemedy), Public Contracts (parent: AdministrativeAgreement), Civil Service (parent: GovernanceFramework), Jurisprudence (parent: LegalPrecedent).
     Object properties: promulgates, overrules, implements, appealsAgainst, bindsAuthority.
3. Latent / Interdisciplinary Synthesis (e.g., Behavioral Dynamics, Cognitive Science, Socio-Psychology):
   - Defined by biological/affective drivers, psychological mechanisms, relational dynamics, cultural constructs, decision architecture.
   - Example Archetype (a behavioral science corpus):
     Pillars: Affective & Biological Systems (parent: BiologicalDrive), Cognitive Biases & Heuristics (parent: CognitiveProcess), Relational Dynamics & Mimesis (parent: SocialInteraction), Cultural & Narrative Constructs (parent: SymbolicModel), Behavioral Interventions & Nudges (parent: DecisionArchitecture).
     Object properties: regulatesAffectiveState, sublimatesDrive, mitigatesBias, embodiesArchetype, triggersResponse.

## Instructions
1. Infer the TRUE domain and epistemological nature of the collection from the titles and excerpts.
2. Formulate 6 to 7 distinct, cohesive taxonomical pillars.
   For each pillar:
   - "name": Concise title of the pillar
   - "target_parent_class": The root OWL class to anchor subclasses under (e.g. Asset, LegalNorm, BiologicalDrive)
   - "sample_classes": 8-12 specific candidate domain concepts belonging to this dimension
   - "typical_relations": Expected relationship predicates relevant to this dimension
3. Identify individual types for `owl:NamedIndividual` (e.g. agencies, key theorists, indices, specific programs).
4. Suggest key `owl:ObjectProperty` names connecting classes across the domain.
5. Identify cross-cutting themes / bridge relations that connect different pillars to prevent disconnected subgraphs.

## Output Format & Rules
1. Think briefly: keep your internal reasoning concise (under 200 words), then immediately produce the JSON.
2. You MUST output ONLY a valid JSON object strictly conforming to the schema below.
3. Do NOT write conversational explanations before or after the JSON.
4. Do NOT use unescaped double quotes inside string values; use single quotes instead.
5. Ensure no trailing commas exist before closing braces or brackets.

{
  "inferred_domain": "...",
  "epistemological_nature": "Empirical | Deontic | Interdisciplinary",
  "pillars": [
    {
      "name": "...",
      "target_parent_class": "...",
      "sample_classes": ["...", "..."],
      "typical_relations": ["...", "..."]
    }
  ],
  "individual_types": ["...", "..."],
  "suggested_object_properties": ["...", "..."],
  "cross_cutting_themes": ["...", "..."]
}
"""


def _extract_json(raw_text: str) -> dict:
    """Extract clean JSON dict from model output, handling fences, trailing commas, and formatting quirks."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object '{{...}}' found in text (length={len(raw_text)})")

    text = text[start : end + 1]

    # Attempt 1: Direct JSON load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt 2: Strip trailing commas
    t_commas = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        return json.loads(t_commas)
    except Exception:
        pass

    # Attempt 3: Tolerate raw control characters (e.g. newlines) inside strings
    try:
        return json.loads(t_commas, strict=False)
    except Exception as e:
        raise ValueError(f"Could not parse valid JSON from text: {e}") from e


def synthesize_domain_blueprint(
    collection: str,
    doc_filenames: list[str],
    chunk_samples: list[dict[str, Any]],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> DomainBlueprint:
    """Synthesize a DomainBlueprint via Scaleway Chat Completions (``DOMAIN_DISCOVERY_MODEL`` by default)."""
    target_model = model or domain_discovery_model()
    target_key = api_key or scaleway_api_key()
    if not target_key:
        raise ValueError("Scaleway API key is required for automated domain discovery.")

    user_lines = [
        f"Collection name: {collection}",
        f"Total unique documents: {len(doc_filenames)}",
        "Sampled Document Titles:",
    ]
    for fn in doc_filenames[:25]:
        user_lines.append(f"- {fn}")
    if len(doc_filenames) > 25:
        user_lines.append(f"- ... and {len(doc_filenames) - 25} more documents")

    user_lines.append("\nRepresentative Text Excerpts from Sampled Documents:")
    for s in chunk_samples:
        user_lines.append(f"--- Document: {s['doc']} (chunk {s['chunk_number']}) ---\n{s['text']}\n")

    user_content = "\n".join(user_lines)

    logger.info("Running domain discovery with model=%s on collection=%s", target_model, collection)
    try:
        response = tracked_completion(
            model=f"scaleway/{target_model}",
            api_key=target_key,
            messages=[
                {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=8192,
            temperature=0.2,
            timeout=90,
        )
    except APIConnectionError as e:
        logger.error("Failed to connect to Scaleway for domain discovery: %s", e)
        raise RuntimeError(f"Domain discovery failed to reach Scaleway: {e}") from e
    except APIError as e:
        status = getattr(e, "status_code", "?")
        logger.error("Scaleway domain discovery returned error %s: %s", status, e)
        raise RuntimeError(f"Scaleway API error {status}: {e}") from e

    if not response.choices or not response.choices[0].message:
        raise RuntimeError(f"Unexpected Scaleway response structure: {response}")

    msg = response.choices[0].message
    raw_content = msg.get("content") or ""
    if "{" not in raw_content:
        reasoning = msg.get("reasoning_content") or ""
        if "{" in reasoning:
            raw_content = reasoning

    try:
        data = _extract_json(raw_content)
        return DomainBlueprint.model_validate(data)
    except Exception as e:
        logger.error("Failed to parse DomainBlueprint JSON: %s\nRaw output:\n%s", e, raw_content[:400])
        raise RuntimeError(f"Could not parse valid DomainBlueprint from model output: {e}") from e


# ─── Stage 4: Profile Caching & Persistence ──────────────────────────────────

def profile_path_for(collection: str, base_dir: str | None = None) -> str:
    """Path to the cached domain profile JSON file for a collection."""
    directory = base_dir or domain_profiles_dir()
    safe_id = sanitize_ontology_id(collection)
    return os.path.join(directory, f"{safe_id}.json")


def save_blueprint(collection: str, blueprint: DomainBlueprint, base_dir: str | None = None) -> str:
    """Save a synthesized DomainBlueprint to disk cache."""
    path = profile_path_for(collection, base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(blueprint.model_dump_json(indent=2))
    logger.info("Saved domain blueprint for %s to %s", collection, path)
    return path


def delete_blueprint(collection: str, base_dir: str | None = None) -> bool:
    """Remove a collection's cached DomainBlueprint. Returns False if there was none."""
    path = profile_path_for(collection, base_dir)
    if not os.path.exists(path):
        return False
    os.remove(path)
    logger.info("Deleted domain blueprint for %s (%s)", collection, path)
    return True


def load_blueprint(collection: str, base_dir: str | None = None) -> DomainBlueprint | None:
    """Load a cached DomainBlueprint from disk if it exists."""
    path = profile_path_for(collection, base_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return DomainBlueprint.model_validate_json(f.read())
    except Exception:
        logger.warning("Failed to load cached domain blueprint from %s", path, exc_info=True)
        return None


def discover_domain(
    store: QdrantStore,
    collection: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    force: bool = False,
    base_dir: str | None = None,
) -> DomainBlueprint:
    """End-to-end domain discovery.

    Returns cached blueprint if available (unless force=True), otherwise samples
    the corpus and calls the discovery model, caches the blueprint, and returns it.
    """
    if not force:
        cached = load_blueprint(collection, base_dir=base_dir)
        if cached is not None:
            logger.info("Reusing cached domain blueprint for collection %r", collection)
            return cached

    all_docs, samples = sample_corpus_for_discovery(store, collection)
    if not samples:
        raise ValueError(f"Collection {collection!r} has no document chunks in Qdrant to discover from.")

    blueprint = synthesize_domain_blueprint(
        collection=collection,
        doc_filenames=all_docs,
        chunk_samples=samples,
        model=model,
        api_key=api_key,
    )
    blueprint.generated_at = datetime.now().isoformat(timespec="seconds")
    blueprint.source_document_count = len(all_docs)
    save_blueprint(collection, blueprint, base_dir=base_dir)
    return blueprint

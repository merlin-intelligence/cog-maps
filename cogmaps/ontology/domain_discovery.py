"""Automated Phase 0 Domain Discovery Engine for CogMaps & OLAF.

Discovers the latent domain, epistemological nature, taxonomical pillars,
key entity types, and cross-cutting relationships of a corpus using fast,
stratified pre-retrieval sampling and GLM-5.3-Flash synthesis.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import models
import requests

from cogmaps.config import (
    domain_discovery_model,
    domain_profiles_dir,
    nebius_api_key,
)
from cogmaps.ontology.olaf_config import sanitize_ontology_id
from cogmaps.qdrant.store import QdrantStore
from cogmaps.rag.llm_clients import resolve_nebius_endpoint

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
    """Complete domain architecture blueprint synthesized in Phase 0."""
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


# ─── Stage 1: Stratified Corpus Profiling ───────────────────────────────────

def sample_corpus_for_discovery(
    store: QdrantStore,
    collection: str,
    max_chunks: int = 15,
    char_limit_per_chunk: int = 500,
    seed: int = 42,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Sample document filenames and representative text chunks across a collection.

    Pulls 1 representative chunk per document (up to max_chunks docs) to avoid
    early-document bias and explore the full breadth of the corpus in <1 second.
    """
    all_filenames = sorted(store.existing_filenames(collection))
    if not all_filenames:
        return [], []

    sampled_filenames: list[str]
    if len(all_filenames) <= max_chunks:
        sampled_filenames = all_filenames
    else:
        sampled_filenames = random.Random(seed).sample(all_filenames, max_chunks)

    samples: list[dict[str, Any]] = []

    # If only 1 document is present, sample multiple chunks across it
    if len(all_filenames) == 1:
        points, _ = store.client.scroll(
            collection_name=collection,
            limit=max_chunks,
            with_payload=["filename", "chunk_number", "text"],
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            samples.append({
                "doc": payload.get("filename", all_filenames[0]),
                "chunk_number": payload.get("chunk_number", 0),
                "text": str(payload.get("text", ""))[:char_limit_per_chunk],
            })
        return all_filenames, samples

    for doc in sampled_filenames:
        flt = models.Filter(must=[models.FieldCondition(key="filename", match=models.MatchValue(value=doc))])
        points, _ = store.client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=1,
            with_payload=["filename", "chunk_number", "text"],
            with_vectors=False,
        )
        if points:
            payload = points[0].payload or {}
            samples.append({
                "doc": payload.get("filename", doc),
                "chunk_number": payload.get("chunk_number", 0),
                "text": str(payload.get("text", ""))[:char_limit_per_chunk],
            })

    return all_filenames, samples


# ─── Stage 2 & 3: Meta-Discovery Synthesizer Prompt ──────────────────────────

DISCOVERY_SYSTEM_PROMPT = """You are an expert ontology engineer and taxonomist specializing in domain discovery and knowledge modeling.
Your mission is Phase 0: Meta-Discovery. Given a corpus profile (collection name, sampled document filenames, and representative text excerpts), you analyze the latent themes, identify the true domain (looking past misleading, metaphorical, or colloquial collection names), and construct a rigorous taxonomical domain blueprint.

## Methodological Grounding & Epistemological Types
Domains generally fall into one of three epistemological categories:
1. Empirical / Dynamic (e.g., Quantitative Finance, Economics, Climate Science):
   - Defined by observable state variables, quantitative metrics, mechanisms, instruments, causal feedback loops.
   - Example Archetype (quant-macro-research):
     Pillars: Financial Instruments (parent: Asset), Yield Curve & Rates (parent: Metric/Rate), Central Bank Mechanisms (parent: MonetaryOperation), Fiscal Operations (parent: SovereignDebtOperation), Macro Indicators (parent: MacroeconomicIndicator), Market Regimes (parent: MarketState).
     Object properties: drives, absorbsSupplyOf, hedgesAgainst, setsPolicyRate, impactsInflation.
2. Deontic / Hierarchical (e.g., Administrative Law, Regulatory Compliance, Public Governance):
   - Defined by statutory authorities, normative hierarchies, legal procedures, jurisdictions, compliance rules.
   - Example Archetype (agiradm):
     Pillars: Normative Acts (parent: LegalNorm), Administrative Authorities (parent: PublicAuthority), Legal Procedures (parent: ProceduralRemedy), Public Contracts (parent: AdministrativeAgreement), Civil Service (parent: GovernanceFramework), Jurisprudence (parent: LegalPrecedent).
     Object properties: promulgates, overrules, implements, appealsAgainst, bindsAuthority.
3. Latent / Interdisciplinary Synthesis (e.g., Behavioral Dynamics, Cognitive Science, Socio-Psychology):
   - Defined by biological/affective drivers, psychological mechanisms, relational dynamics, cultural constructs, decision architecture.
   - Example Archetype (animal trail):
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

    # Attempt 3: Escape unescaped newlines inside strings
    def replace_newlines(match: re.Match) -> str:
        return match.group(0).replace("\n", "\\n")

    t_newlines = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', replace_newlines, t_commas, flags=re.DOTALL)
    try:
        return json.loads(t_newlines)
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
    """Synthesize a DomainBlueprint using GLM-5.3-Flash via Nebius Chat Completions."""
    target_model = model or domain_discovery_model()
    target_key = api_key or nebius_api_key()
    if not target_key:
        raise ValueError("Nebius API key is required for automated domain discovery.")

    api_url, _vendor = resolve_nebius_endpoint(target_model)

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

    headers = {
        "Authorization": f"Bearer {target_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": target_model,
        "messages": [
            {"role": "system", "content": DISCOVERY_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_content}]},
        ],
        "max_tokens": 8192,
        "temperature": 0.2,
    }

    logger.info("Running domain discovery with model=%s on collection=%s", target_model, collection)
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as e:
        logger.error("Failed to connect to Nebius for domain discovery: %s", e)
        raise RuntimeError(f"Domain discovery failed to reach Nebius: {e}") from e

    if response.status_code != 200:
        logger.error("Nebius domain discovery returned error %s: %s", response.status_code, response.text[:200])
        raise RuntimeError(f"Nebius API error {response.status_code}: {response.text}")

    resp_json = response.json()
    choices = resp_json.get("choices", [])
    if not choices or not choices[0].get("message"):
        raise RuntimeError(f"Unexpected Nebius response structure: {resp_json}")

    msg = choices[0]["message"]
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
    """End-to-end Phase 0 Domain Discovery.

    Returns cached blueprint if available (unless force=True), otherwise samples
    the corpus and calls GLM-5.3-Flash, caches the blueprint, and returns it.
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
    save_blueprint(collection, blueprint, base_dir=base_dir)
    return blueprint

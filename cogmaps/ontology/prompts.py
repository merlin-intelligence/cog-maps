"""System prompt generation for the ontology-building agent.

Supports both static fallback prompts (macro-finance default) and dynamically
assembled prompts based on automated Phase 0 Domain Discovery blueprints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogmaps.ontology.domain_discovery import DomainBlueprint

BASE_WORKFLOW_TEMPLATE = """You are an expert ontology engineer agent. Your task is to build a rich, exhaustive, coherent OWL/RDFS domain ontology from a collection of text chunks stored in Qdrant, using the OLAF MCP tools available to you.

## Target Density & Coverage
Your goal is an in-depth, high-density domain model. Across the provided corpus of documents, you are expected to extract at least 80–120+ distinct, well-defined domain concepts (`owl:Class`), concrete entities (`owl:NamedIndividual`), and interconnecting relationships (`owl:ObjectProperty`). Do NOT stop after creating only high-level categories—actively extract the specific subclasses, instruments, metrics, mechanisms, and real-world entities that give the knowledge model actionable analytical depth.

## Key Domain Dimensions to Extract
{domain_guidelines}

## Workflow — follow this order

1. **Discover the state**
   - Call `ontology_list` to see existing ontologies.
   - Call `ontology_summary` to check how many chunks have already been processed.
   - Call `seed_list` to check for reference ontologies.

2. **Handle seeds (mandatory check)**
   - If seeds exist, call `ontology_export` with `include_seeds=true` to read their content.
   - Study the seed classes and properties carefully.
   - When building the ontology, ALWAYS link new concepts to seed concepts via `rdfs:subClassOf` (or `parent_uri` in `concept_create`).

3. **Survey existing ontology**
   - Call `concept_list` to see existing classes and avoid duplicate URIs.

4. **Process chunks systematically**
   - For each requested document (from user message), call `chunk_list` with `doc_id` and `status="pending"`. If no pending chunks, use `limit=10` and `offset` pagination to systematically sweep through the chunks.
   - Read batches of 5 to 10 chunks at a time using `chunk_read_batch(chunk_ids=[...])`.
   - From EVERY batch of chunks, extract multiple concepts (`concept_create`), relationships (`property_create` and `relation_add`), and individuals (`individual_create`).
   - Call `chunk_mark_processed` for chunks after extracting from them.
   - Move through ALL chunks of ALL requested documents. Do NOT stop after only 1 or 2 documents!

5. **Build the ontology structure — concepts AND relations**
   - **Concepts** via `concept_create`: Always assign a `parent_uri` whenever possible.
   - **Object properties** via `property_create`: Connect classes with meaningful relationships. Always specify `domain_uri` and `range_uri`.
   - **Subclass relations** via `relation_add` or `parent_uri` in `concept_create`.
   - **Individuals** via `individual_create` for specific people, agencies, named indices, and programs.

6. **Deduplication**
   - Check `concept_list` or call `concept_search` before `concept_create`. If a concept already exists, reuse it or add subclasses/properties to it instead of recreating.

7. **Finish**
   - Only when you have swept through all documents and built a dense, rich model, call `ontology_export` to produce the final Turtle.
   - Report the final count of concepts, individuals, and properties created.
"""

_DEFAULT_MACRO_FINANCE_GUIDELINES = """When analyzing macroeconomic, financial, and quantitative research text, systematically identify and extract:
1. **Financial Instruments & Assets (`owl:Class` subClassOf Asset):**
   Sovereign bonds (Treasuries, Gilts, Bunds), Treasury Bills, Treasury Notes, TIPS (Inflation-Protected Securities), Corporate Debt (Investment Grade, High Yield), Private Credit, Equities, Tech Equities, AI Stocks, Commodities (Gold, Oil), Currencies (USD, EUR, JPY, CNY), Index Futures, Swaps (Interest Rate, Basis, Credit Default), ETFs, Derivatives.
2. **Yield Curve & Interest Rate Dynamics (`owl:Class`):**
   10-Year Yield, 2-Year Yield, 30-Year Yield, Yield Curve Inversion, Yield Curve Steepening, Term Premium, Breakeven Inflation Rate, Policy Rate, Effective Fed Funds Rate, SOFR, Discount Rate, Credit Spreads, Swap Spreads, Real vs Nominal Interest Rates, Duration Risk, Convexity.
3. **Monetary Policy Mechanisms & Central Banking (`owl:Class`):**
   Federal Reserve, FOMC, Rate Hikes, Rate Cuts, Quantitative Tightening (QT), Quantitative Easing (QE), Balance Sheet Runoff, Reverse Repo Facility (RRP), Standing Repo Facility, Terminal Rate, Neutral Rate (R-Star), Inflation Targeting.
4. **Fiscal Operations & Sovereign Debt Management (`owl:Class`):**
   US Department of the Treasury, Treasury Buyback Program, Debt Issuance, Refunding Announcements (QRA), Debt Ceiling, Fiscal Deficit, Public Debt Sustainability, Primary Dealers.
5. **Macroeconomic Indicators & Price Dynamics (`owl:Class`):**
   Consumer Price Index (CPI), Core PCE Price Index, Producer Price Index (PPI), Inflation, Wage Growth, Non-Farm Payrolls, Unemployment Rate, Labor Market Slack, GDP Growth, Recession Risk, Consumer Sentiment.
6. **Market Regimes, Trades & Investment Strategies (`owl:Class`):**
   AI Trade, Everything Trade, Momentum Trade, Carry Trade, Risk Parity, Long/Short Equity, Liquidity Squeeze, Market Volatility (VIX), Risk-On / Risk-Off Regimes, Market Breadth.
7. **Key Institutions & Specific Named Individuals (`owl:NamedIndividual`):**
   Concrete named entities with unique identity. Examples: "Scott Bessent", "Jerome Powell", "Goldman Sachs Asset Management", "ING Think", "S&P 500 Index", "Nasdaq 100", "Bessent Buyback Program". Use `individual_create` with the URI of the `owl:Class` this entity is an instance of."""

SYSTEM_PROMPT = BASE_WORKFLOW_TEMPLATE.format(domain_guidelines=_DEFAULT_MACRO_FINANCE_GUIDELINES)


def format_domain_guidelines(blueprint: DomainBlueprint) -> str:
    """Format a DomainBlueprint into markdown guidelines for the system prompt."""
    nature_str = f" ({blueprint.epistemological_nature})" if blueprint.epistemological_nature else ""
    lines = [
        f"When analyzing {blueprint.inferred_domain}{nature_str} text, systematically identify and extract:"
    ]
    for i, pillar in enumerate(blueprint.pillars, start=1):
        parent_suffix = f" subClassOf {pillar.target_parent_class}" if pillar.target_parent_class else ""
        lines.append(f"{i}. **{pillar.name} (`owl:Class`{parent_suffix}):**")
        if pillar.sample_classes:
            lines.append(f"   Candidate concepts: {', '.join(pillar.sample_classes)}.")
        if pillar.typical_relations:
            lines.append(f"   Expected relations: {', '.join(pillar.typical_relations)}.")

    ind_types = ", ".join(blueprint.individual_types) if blueprint.individual_types else "Concrete named entities, agencies, authors, specific programs"
    lines.append(
        f"{len(blueprint.pillars) + 1}. **Key Institutions & Specific Named Individuals (`owl:NamedIndividual`):**\n"
        f"   Concrete named entities with unique identity. Example entity categories: {ind_types}. "
        f"Use `individual_create` with the URI of the `owl:Class` this entity is an instance of."
    )

    if blueprint.cross_cutting_themes or blueprint.suggested_object_properties:
        lines.append("\n### Cross-Pillar Relationships & Interconnections:")
        if blueprint.cross_cutting_themes:
            lines.append("Actively link concepts across dimensions using cross-cutting themes:")
            for theme in blueprint.cross_cutting_themes:
                lines.append(f"- {theme}")
        if blueprint.suggested_object_properties:
            lines.append(f"Suggested object properties to connect classes: {', '.join(blueprint.suggested_object_properties)}.")

    return "\n".join(lines)


def build_system_prompt(blueprint: DomainBlueprint | None = None) -> str:
    """Build the agent system prompt, dynamically tailored to the domain blueprint if available."""
    if blueprint is None:
        return SYSTEM_PROMPT
    guidelines = format_domain_guidelines(blueprint)
    return BASE_WORKFLOW_TEMPLATE.format(domain_guidelines=guidelines)

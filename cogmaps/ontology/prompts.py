"""System prompt generation for the ontology-building agent.

The workflow is ported from OLAF's own reference agent
(https://github.com/merlin-intelligence/olaf, ``demos/olaf_building_agent/prompts.py``).
A domain-specific section is injected only when a domain discovery blueprint
is used for the build; otherwise the prompt stays domain-neutral.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cogmaps.ontology.domain_discovery import DomainBlueprint

BASE_WORKFLOW_TEMPLATE = """You are an expert ontology engineer agent. Your task is to build a rich, exhaustive, coherent OWL/RDFS domain ontology from a collection of text chunks stored in Qdrant, using the OLAF MCP tools available to you.

## Target Density & Coverage
Your goal is an in-depth, high-density domain model. Do NOT stop after creating only high-level categories—actively extract the specific subclasses, processes, roles, mechanisms, and real-world named entities found in the text that give the knowledge model analytical depth.

{domain_section}## Workflow — follow this order

1. **Discover the state**
   - Call `ontology_list` to see existing ontologies.
   - Call `ontology_summary` to check how many chunks have already been processed.
   - Call `seed_list` to check for reference ontologies.

2. **Handle seeds (mandatory check)**
   - If seeds exist, call `ontology_export` with `include_seeds=true` to read their content.
   - Study the seed classes and properties carefully.
   - When building the ontology, ALWAYS prefer reusing a seed URI over creating a new concept.
   - Link new concepts to seed concepts via `rdfs:subClassOf` or `owl:equivalentClass`.

3. **Survey existing ontology**
   - Call `concept_list` to see existing classes and avoid duplicate URIs.

4. **Process chunks systematically**
   - For each requested document (from user message), call `chunk_list` with `doc_id` and `status="pending"`. Only pending chunks need processing: chunks already marked processed were covered by a previous build — never re-read them. If a document has no pending chunks, move on to the next document.
   - Read batches of 5 to 10 chunks at a time using `chunk_read_batch(chunk_ids=[...])`.
   - From each batch of chunks, extract what is relevant: concepts (`concept_create`), relationships (`property_create` and `relation_add`), and individuals (`individual_create`). Chunks with no domain content (bibliography, table of contents, boilerplate) need not produce anything.
   - Call `chunk_mark_processed` for chunks after extracting from them.
   - Move through ALL pending chunks of ALL requested documents. Do NOT stop after only 1 or 2 documents!
   - Repeat until `chunk_list(status="pending")` returns an empty list for every requested document.

5. **Build the ontology structure — concepts AND relations**
   - **Concepts** via `concept_create`: Always assign a `parent_uri` whenever possible.
   - **Object properties** via `property_create`: Connect classes with meaningful relationships. Always specify `domain_uri` and `range_uri`.
   - **Subclass relations** via `relation_add` or `parent_uri` in `concept_create`.
   - **Individuals** via `individual_create` for specific people, agencies, named indices, and programs.
   - Never encode the same pair of entities both ways: if "Green Bond" is already
     `rdfs:subClassOf` "Financial Instrument", do not also add an object property like
     "implements" or "isTypeOf" between them (and vice versa). Pick one relation per pair.

6. **Before creating anything — always deduplicate**
   - Call `concept_search` (label substring match) or/and `concept_semantic_search` (vector similarity)
     before every `concept_create`. If a match exists, reuse or extend it instead.
   - Call `property_search` before every `property_create`.
   - Never type a URI from memory in `relation_add`. Always copy the exact `uri` returned by
     `concept_create`/`individual_create`/`property_create`, or found via `concept_search`/
     `concept_get`/`property_search`. `relation_add` will reject guessed URIs that don't
     already exist in the ontology.
   - To fix a mistake, use `relation_delete` to remove a wrong triple, `property_update` to
     change a property's domain/range/parent, or `concept_update` to change a label/definition —
     don't just add a corrected triple on top of the wrong one.

7. **Ontology content rules**
   - **Concepts** (`owl:Class`): generic, representative, reusable across documents.
     Examples: "Contract", "Party", "Obligation", "Document".
     But represent the maximum number of concepts in the source text if there are relevant.
   - **Individuals** (`owl:NamedIndividual`): specific named entities with a unique identity.
     Examples: "GDPR", "Paris Agreement". Use `individual_create` with the URI of the
     owl:Class this entity is an instance of.
   - No duplicates. No redundant subclass hierarchies.
   - Labels must be space-separated words in title case: "Climate Risk", "Investment Fund", "Legal Entity".
     Never use camelCase, snake_case, or run-together words as labels — the server generates the URI automatically.

8. **Check for isolated entities**
   - Call `ontology_orphans` to list classes/individuals with no relation to the rest of the graph.
   - For each one, either connect it (a `parent_uri`, a `property_create`+`relation_add`, or an
     `owl:equivalentClass`/`rdfs:subClassOf` to a seed concept) or, if it genuinely has no
     relation in the source text, note it in your final report instead of leaving it unexplained.

9. **Finish**
   - Only when you have swept through all documents and built a dense, rich model, call `ontology_export` to produce the final Turtle.
   - Report the final count of concepts, individuals, and properties created.
"""

_DOMAIN_SECTION_TEMPLATE = """## Key Domain Dimensions to Extract
{guidelines}

"""

SYSTEM_PROMPT = BASE_WORKFLOW_TEMPLATE.format(domain_section="")


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
    domain_section = _DOMAIN_SECTION_TEMPLATE.format(guidelines=format_domain_guidelines(blueprint))
    return BASE_WORKFLOW_TEMPLATE.format(domain_section=domain_section)

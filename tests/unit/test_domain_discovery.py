"""Unit tests for the domain discovery engine."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models

from cogmaps.ontology.domain_discovery import (
    DomainBlueprint,
    DomainPillar,
    _extract_json,
    delete_blueprint,
    load_blueprint,
    profile_path_for,
    sample_corpus_for_discovery,
    save_blueprint,
    synthesize_domain_blueprint,
)
from cogmaps.ontology.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    format_domain_guidelines,
)


@pytest.fixture
def sample_blueprint() -> DomainBlueprint:
    return DomainBlueprint(
        inferred_domain="Quantitative Macroeconomics & Sovereign Debt",
        epistemological_nature="Empirical",
        pillars=[
            DomainPillar(
                name="Financial Instruments",
                target_parent_class="Asset",
                sample_classes=["TreasuryBond", "TIPS", "CorporateDebt"],
                typical_relations=["issues", "hedgesAgainst"],
            ),
            DomainPillar(
                name="Yield Curve Dynamics",
                target_parent_class="Metric",
                sample_classes=["TenYearYield", "YieldCurveInversion"],
                typical_relations=["hasYield", "measuresSpread"],
            ),
        ],
        individual_types=["CentralBank", "PrimaryDealer", "SovereignIssuer"],
        suggested_object_properties=["issues", "hedgesAgainst", "influencesInflation"],
        cross_cutting_themes=[
            "Yield curve steepening impact on corporate credit spreads",
            "Fiscal deficit funding interaction with central bank balance sheet runoff",
        ],
    )


def test_domain_blueprint_model_validation(sample_blueprint):
    assert sample_blueprint.inferred_domain == "Quantitative Macroeconomics & Sovereign Debt"
    assert sample_blueprint.epistemological_nature == "Empirical"
    assert len(sample_blueprint.pillars) == 2
    assert sample_blueprint.pillars[0].target_parent_class == "Asset"
    assert "TreasuryBond" in sample_blueprint.pillars[0].sample_classes


def test_save_and_load_blueprint(tmp_path, sample_blueprint):
    path = save_blueprint("test_col", sample_blueprint, base_dir=str(tmp_path))
    assert (tmp_path / "test_col.json").exists()

    loaded = load_blueprint("test_col", base_dir=str(tmp_path))
    assert loaded is not None
    assert loaded.inferred_domain == sample_blueprint.inferred_domain
    assert len(loaded.pillars) == 2
    assert loaded.pillars[0].name == "Financial Instruments"
    assert loaded.individual_types == ["CentralBank", "PrimaryDealer", "SovereignIssuer"]


def test_load_blueprint_nonexistent(tmp_path):
    assert load_blueprint("nonexistent_collection", base_dir=str(tmp_path)) is None


def test_delete_blueprint(tmp_path, sample_blueprint):
    save_blueprint("test_col", sample_blueprint, base_dir=str(tmp_path))
    assert delete_blueprint("test_col", base_dir=str(tmp_path)) is True
    assert load_blueprint("test_col", base_dir=str(tmp_path)) is None
    assert delete_blueprint("test_col", base_dir=str(tmp_path)) is False


def test_format_domain_guidelines(sample_blueprint):
    guidelines = format_domain_guidelines(sample_blueprint)
    assert "Quantitative Macroeconomics & Sovereign Debt" in guidelines
    assert "Financial Instruments (`owl:Class` subClassOf Asset):" in guidelines
    assert "TreasuryBond, TIPS, CorporateDebt" in guidelines
    assert "CentralBank, PrimaryDealer, SovereignIssuer" in guidelines
    assert "Yield curve steepening impact on corporate credit spreads" in guidelines
    assert "issues, hedgesAgainst, influencesInflation" in guidelines


def test_build_system_prompt_dynamic(sample_blueprint):
    prompt = build_system_prompt(sample_blueprint)
    assert "Quantitative Macroeconomics & Sovereign Debt" in prompt
    assert "Target Density & Coverage" in prompt
    assert "Workflow — follow this order" in prompt
    assert "TreasuryBond" in prompt


def test_build_system_prompt_default_fallback():
    prompt = build_system_prompt(None)
    assert prompt == SYSTEM_PROMPT


def test_extract_json_clean_and_fenced():
    expected = {"inferred_domain": "Finance", "pillars": []}
    raw_json = '{"inferred_domain": "Finance", "pillars": []}'
    assert _extract_json(raw_json) == expected

    fenced_json = '```json\n{"inferred_domain": "Finance", "pillars": []}\n```'
    assert _extract_json(fenced_json) == expected

    conversational_json = 'Here is the blueprint:\n```json\n{"inferred_domain": "Finance", "pillars": []}\n```\nHope this helps!'
    assert _extract_json(conversational_json) == expected

    trailing_comma_json = '{"inferred_domain": "Finance", "pillars": [],}'
    assert _extract_json(trailing_comma_json) == expected


def test_sample_corpus_for_discovery_empty():
    mock_store = MagicMock()
    mock_store.existing_filenames.return_value = set()
    docs, samples = sample_corpus_for_discovery(mock_store, "empty_col")
    assert docs == []
    assert samples == []


def test_sample_corpus_for_discovery_multi_doc():
    mock_store = MagicMock()
    mock_store.existing_filenames.return_value = {"doc1.pdf", "doc2.pdf", "doc3.pdf"}

    mock_point = MagicMock()
    mock_point.payload = {"filename": "doc1.pdf", "chunk_number": 0, "text": "Sample text excerpt"}
    mock_store.client.query_points.return_value = MagicMock(points=[mock_point])

    docs, samples = sample_corpus_for_discovery(mock_store, "test_col", max_chunks=2)
    assert len(docs) == 3
    assert len(samples) == 2
    assert samples[0]["text"] == "Sample text excerpt"

    # One random draw per sampled document, filtered on that document
    calls = mock_store.client.query_points.call_args_list
    assert len(calls) == 2
    for call in calls:
        assert call.kwargs["limit"] == 1
        assert call.kwargs["query"].sample == models.Sample.RANDOM
        assert call.kwargs["query_filter"] is not None
    mock_store.client.scroll.assert_not_called()


def test_sample_corpus_for_discovery_single_doc_draws_random_chunks_sorted():
    mock_store = MagicMock()
    mock_store.existing_filenames.return_value = {"only.pdf"}

    def point(n):
        p = MagicMock()
        p.payload = {"filename": "only.pdf", "chunk_number": n, "text": f"chunk {n}"}
        return p

    mock_store.client.query_points.return_value = MagicMock(points=[point(7), point(2), point(41)])

    docs, samples = sample_corpus_for_discovery(mock_store, "test_col", max_chunks=3)
    assert docs == ["only.pdf"]
    assert [s["chunk_number"] for s in samples] == [2, 7, 41]
    assert mock_store.client.query_points.call_args.kwargs["limit"] == 3


def test_sample_corpus_for_discovery_falls_back_to_scroll_without_random_sampling():
    mock_store = MagicMock()
    mock_store.existing_filenames.return_value = {"doc1.pdf", "doc2.pdf"}
    mock_store.client.query_points.side_effect = RuntimeError("unknown query type")

    mock_point = MagicMock()
    mock_point.payload = {"filename": "doc1.pdf", "chunk_number": 0, "text": "first chunk"}
    mock_store.client.scroll.return_value = ([mock_point], None)

    _, samples = sample_corpus_for_discovery(mock_store, "test_col")
    assert len(samples) == 2
    assert mock_store.client.scroll.call_count == 2


def test_synthesize_domain_blueprint_success(sample_blueprint):
    from litellm.types.utils import Choices, Message, ModelResponse

    mock_response = ModelResponse(
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content=sample_blueprint.model_dump_json(),
                    reasoning_content="Reasoning about finance corpus...",
                ),
            )
        ]
    )

    with patch("litellm.completion", return_value=mock_response):
        blueprint = synthesize_domain_blueprint(
            collection="test_finance",
            doc_filenames=["report1.pdf", "report2.pdf"],
            chunk_samples=[{"doc": "report1.pdf", "chunk_number": 0, "text": "Fed rate cut"}],
            model="qwen3.8-27b",
            api_key="test_key",
        )
        assert blueprint.inferred_domain == sample_blueprint.inferred_domain
        assert len(blueprint.pillars) == 2


def test_extract_json_tolerates_raw_newlines_in_strings():
    raw = '{"inferred_domain": "Line one\nline two", "pillars": [],}'
    assert _extract_json(raw) == {"inferred_domain": "Line one\nline two", "pillars": []}


def test_default_prompt_has_no_domain_section():
    prompt = build_system_prompt(None)
    assert "Key Domain Dimensions to Extract" not in prompt
    assert "{domain_section}" not in prompt
    assert "Treasur" not in prompt
    assert "Powell" not in prompt
    assert "80–120" not in prompt


def test_domain_section_only_present_with_blueprint(sample_blueprint):
    prompt = build_system_prompt(sample_blueprint)
    assert prompt.count("## Key Domain Dimensions to Extract") == 1
    assert prompt.index("## Key Domain Dimensions to Extract") < prompt.index("## Workflow")


def test_prompt_keeps_original_olaf_rules():
    prompt = build_system_prompt(None)
    for rule in (
        "ALWAYS prefer reusing a seed URI over creating a new concept",
        "Pick one relation per pair",
        "Call `property_search` before every `property_create`",
        "Never type a URI from memory in `relation_add`",
        "Labels must be space-separated words in title case",
        "Call `ontology_orphans`",
        "extract what is relevant",
    ):
        assert rule in prompt, rule
    assert "Avoid overly specific classes" not in prompt
    assert "instruments, metrics" not in prompt


def test_discovery_prompt_does_not_name_internal_collections():
    from cogmaps.ontology.domain_discovery import DISCOVERY_SYSTEM_PROMPT

    for name in ("quant-macro-research", "agiradm", "animal trail"):
        assert name not in DISCOVERY_SYSTEM_PROMPT


def test_discover_domain_records_provenance(tmp_path, sample_blueprint):
    from cogmaps.ontology import domain_discovery

    mock_store = MagicMock()
    with patch.object(domain_discovery, "sample_corpus_for_discovery",
                      return_value=(["a.pdf", "b.pdf", "c.pdf"], [{"doc": "a.pdf", "chunk_number": 0, "text": "x"}])), \
         patch.object(domain_discovery, "synthesize_domain_blueprint", return_value=sample_blueprint):
        bp = domain_discovery.discover_domain(mock_store, "col", force=True, base_dir=str(tmp_path))

    assert bp.source_document_count == 3
    assert bp.generated_at is not None
    reloaded = load_blueprint("col", base_dir=str(tmp_path))
    assert reloaded.source_document_count == 3


def test_old_blueprint_json_without_provenance_still_loads(tmp_path):
    (tmp_path / "legacy.json").write_text(json.dumps({"inferred_domain": "Law", "pillars": []}), encoding="utf-8")
    bp = load_blueprint("legacy", base_dir=str(tmp_path))
    assert bp is not None
    assert bp.generated_at is None
    assert bp.source_document_count is None

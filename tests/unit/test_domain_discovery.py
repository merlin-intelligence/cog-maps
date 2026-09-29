"""Unit tests for Phase 0 Domain Discovery Engine (GLM-5.3-Flash)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from cogmaps.ontology.domain_discovery import (
    DomainBlueprint,
    DomainPillar,
    _extract_json,
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
    mock_store.client.scroll.return_value = ([mock_point], None)

    docs, samples = sample_corpus_for_discovery(mock_store, "test_col", max_chunks=2)
    assert len(docs) == 3
    assert len(samples) == 2
    assert samples[0]["text"] == "Sample text excerpt"


def test_synthesize_domain_blueprint_success(sample_blueprint):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": sample_blueprint.model_dump_json(),
                    "reasoning_content": "Reasoning about finance corpus...",
                }
            }
        ]
    }

    with patch("requests.post", return_value=mock_response):
        blueprint = synthesize_domain_blueprint(
            collection="test_finance",
            doc_filenames=["report1.pdf", "report2.pdf"],
            chunk_samples=[{"doc": "report1.pdf", "chunk_number": 0, "text": "Fed rate cut"}],
            model="zai-org/GLM-5.3-Flash",
            api_key="test_key",
        )
        assert blueprint.inferred_domain == sample_blueprint.inferred_domain
        assert len(blueprint.pillars) == 2

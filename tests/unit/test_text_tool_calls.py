"""Unit tests for cogmaps.core.text_tool_calls — tool calls written as text."""
from __future__ import annotations

import json
from types import SimpleNamespace

from cogmaps.core.text_tool_calls import parse_text_tool_calls, recover_text_tool_calls


def test_parses_glm_malformed_closing_tag():
    # Seen in a GLM 5.3 build job: the block is closed by a second <tool_call>.
    content = '<tool_call>\n{"name": "chunk_mark_processed", "arguments": {"chunk_id": "d342"}}\n<tool_call>'
    assert parse_text_tool_calls(content) == [("chunk_mark_processed", {"chunk_id": "d342"})]


def test_parses_well_formed_and_several_blocks():
    content = (
        'Let me search.\n<tool_call>{"name": "concept_search", "arguments": {"query": "Data"}}</tool_call>\n'
        '<tool_call>{"name": "property_search", "arguments": "{\\"query\\": \\"owns\\"}"}</tool_call>'
    )
    assert parse_text_tool_calls(content) == [
        ("concept_search", {"query": "Data"}),
        ("property_search", {"query": "owns"}),
    ]


def test_parses_unclosed_block():
    assert parse_text_tool_calls('<tool_call>{"name": "ontology_summary", "arguments": {}}') == [
        ("ontology_summary", {})
    ]


def test_parses_glm_native_format():
    content = (
        "<tool_call>chunk_list\n<arg_key>doc_id</arg_key>\n<arg_value>doc1.pdf</arg_value>\n"
        "<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n</tool_call>"
    )
    assert parse_text_tool_calls(content) == [("chunk_list", {"doc_id": "doc1.pdf", "limit": 5})]


def test_ignores_plain_text_and_unparseable_blocks():
    assert parse_text_tool_calls("I'm done.") == []
    assert parse_text_tool_calls(None) == []
    # One broken block: nothing is recovered rather than running part of the calls.
    content = '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{not json</tool_call>'
    assert parse_text_tool_calls(content) == []


def test_recover_moves_calls_to_tool_calls():
    msg = SimpleNamespace(
        content='Searching now.\n<tool_call>\n{"name": "concept_search", "arguments": {"query": "é"}}\n<tool_call>',
        tool_calls=None,
    )
    assert recover_text_tool_calls(msg) == 1
    assert msg.content == "Searching now."
    (tc,) = msg.tool_calls
    assert tc.id.startswith("call_")
    assert tc.function.name == "concept_search"
    assert json.loads(tc.function.arguments) == {"query": "é"}


def test_recover_leaves_structured_calls_and_plain_text_alone():
    existing = [SimpleNamespace(id="x")]
    msg = SimpleNamespace(content='<tool_call>{"name": "a", "arguments": {}}</tool_call>', tool_calls=existing)
    assert recover_text_tool_calls(msg) == 0
    assert msg.tool_calls is existing
    msg = SimpleNamespace(content="All chunks processed.", tool_calls=None)
    assert recover_text_tool_calls(msg) == 0
    assert msg.content == "All chunks processed."

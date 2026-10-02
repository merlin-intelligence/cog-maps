"""Unit tests for cogmaps.ontology.olaf_launcher and how OLAF gets spawned."""
from __future__ import annotations

import sys

from fastembed import TextEmbedding

from cogmaps.config import EMBEDDING_MODEL_NAME, PROJECT_ROOT
from cogmaps.ontology.mcp_client import launcher_params
from cogmaps.ontology.olaf_config import build_config_toml
from cogmaps.ontology.olaf_launcher import register_embedding_model


def _supported_models() -> list[str]:
    return [m["model"] for m in TextEmbedding.list_supported_models()]


def test_register_embedding_model_makes_it_loadable_and_is_idempotent():
    register_embedding_model()
    register_embedding_model()  # a second call must not raise "already registered"
    assert _supported_models().count(EMBEDDING_MODEL_NAME) == 1


def test_olaf_config_uses_the_app_embedding_model():
    toml = build_config_toml(
        qdrant_url="http://localhost:6333", qdrant_collection="c", qdrant_api_key="",
        oxigraph_url="http://localhost:7878", ontology_id="c", ontology_name="c",
    )
    assert f'model   = "{EMBEDDING_MODEL_NAME}"' in toml


def test_launcher_params_runs_launcher_with_current_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/some/other/path")
    params = launcher_params(str(tmp_path))

    assert params.command == sys.executable
    assert params.args == ["-m", "cogmaps.ontology.olaf_launcher"]
    assert params.cwd == str(tmp_path)
    assert params.env["PYTHONPATH"].split(":") == [str(PROJECT_ROOT), "/some/other/path"]

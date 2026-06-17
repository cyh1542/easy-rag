from __future__ import annotations

from pathlib import Path

import pytest

from easy_rag.knowledge_bases import (
    KnowledgeBaseProfile,
    KnowledgeBaseRegistry,
    default_profiles_from_settings,
    load_knowledge_base_registry,
    merge_retrieval_results,
    parse_registry_from_form,
    save_knowledge_base_registry,
)
from easy_rag.rag_engine import EasyRAG
from easy_rag.config import settings_from_env_values
from tests.conftest import make_env_values


def test_default_profiles_from_settings(settings) -> None:
    profiles = default_profiles_from_settings(settings)
    assert len(profiles) == 4
    assert {profile.chunk_strategy for profile in profiles} == {"fixed", "recursive", "semantic", "structure"}


def test_parse_registry_from_form(settings) -> None:
    form = {
        "MULTI_KB_ENABLED": "true",
        "KB1_ID": "kb-fixed",
        "KB1_NAME": "固定库",
        "KB1_COLLECTION_NAME": "demo-fixed",
        "KB1_CHUNK_STRATEGY": "fixed",
        "KB1_CHUNK_SIZE": "800",
        "KB1_CHUNK_OVERLAP": "120",
        "KB1_SEMANTIC_CHUNK_THRESHOLD": "0.75",
        "KB1_ENABLED": "true",
        "KB2_CHUNK_STRATEGY": "structure",
        "KB2_ENABLED": "true",
    }
    registry = parse_registry_from_form(form, settings)
    assert registry.enabled is True
    assert len(registry.bases) == 2
    assert registry.active_profiles()[0].collection_name == "demo-fixed"


def test_parse_registry_from_form_reads_checkbox_with_hidden_false(settings) -> None:
    class _Form:
        def get(self, key: str, default: str = "") -> str:
            return self._values.get(key, default)

        def getlist(self, key: str) -> list[str]:
            return self._values.get(key, [])

        def __init__(self) -> None:
            self._values = {
                "MULTI_KB_ENABLED": ["false", "true"],
                "KB1_ID": "kb-fixed",
                "KB1_NAME": "固定库",
                "KB1_COLLECTION_NAME": "demo-fixed",
                "KB1_CHUNK_STRATEGY": "fixed",
                "KB1_CHUNK_SIZE": "800",
                "KB1_CHUNK_OVERLAP": "120",
                "KB1_SEMANTIC_CHUNK_THRESHOLD": "0.75",
                "KB1_ENABLED": ["false", "true"],
            }

    registry = parse_registry_from_form(_Form(), settings)
    assert registry.enabled is True
    assert registry.active_profiles()[0].enabled is True


def test_save_and_load_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings) -> None:
    registry_path = tmp_path / "knowledge_bases.json"
    monkeypatch.setattr("easy_rag.knowledge_bases.KNOWLEDGE_BASES_FILE", registry_path)
    registry = KnowledgeBaseRegistry(
        enabled=True,
        bases=default_profiles_from_settings(settings)[:2],
    )
    registry.bases[0].enabled = True
    registry.bases[1].enabled = True
    save_knowledge_base_registry(registry)
    loaded = load_knowledge_base_registry(settings)
    assert loaded.enabled is True
    assert len(loaded.bases) == 2


def test_merge_retrieval_results_deduplicates_and_ranks() -> None:
    groups = [
        [
            {"source": "a.md", "content": "same", "distance": 0.4},
            {"source": "b.md", "content": "other", "distance": 0.2},
        ],
        [
            {"source": "a.md", "content": "same", "distance": 0.1},
            {"source": "c.md", "content": "third", "distance": 0.3},
        ],
    ]
    merged = merge_retrieval_results(groups, top_k=3)
    assert len(merged) == 3
    assert merged[0]["source"] == "b.md"


def test_retrieve_multi_merges_profiles(
    env_file: Path,
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
    sample_markdown: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = make_env_values(tmp_dirs, base_env_values)
    settings = settings_from_env_values(values)
    rag = EasyRAG(settings)

    profile_a = KnowledgeBaseProfile(
        id="kb-a",
        name="A",
        collection_name="coll-a",
        chunk_strategy="fixed",
        chunk_size=800,
        chunk_overlap=120,
        semantic_chunk_threshold=0.75,
        enabled=True,
    )
    profile_b = KnowledgeBaseProfile(
        id="kb-b",
        name="B",
        collection_name="coll-b",
        chunk_strategy="structure",
        chunk_size=800,
        chunk_overlap=120,
        semantic_chunk_threshold=0.75,
        enabled=True,
    )

    def _fake_retrieve(question: str, collection_name: str) -> list[dict]:
        return [
            {
                "content": f"{collection_name} content",
                "source": f"{collection_name}.md",
                "path": str(sample_markdown),
                "distance": 0.2 if collection_name == "coll-b" else 0.5,
                "collection_name": collection_name,
            }
        ]

    monkeypatch.setattr(rag, "_retrieve_from_collection", _fake_retrieve)
    items = rag.retrieve_multi("测试", [profile_a, profile_b])
    assert len(items) == 2
    assert items[0]["collection_name"] == "coll-b"


def test_build_multi_knowledge_bases(
    rag: EasyRAG,
    sample_markdown: Path,
    mock_embedding: None,
) -> None:
    profiles = default_profiles_from_settings(rag.settings)[:2]
    profiles[0].enabled = True
    profiles[1].enabled = True

    summary = rag.build_multi_knowledge_bases(profiles, reset=True)
    assert "kb-fixed" in summary
    assert "kb-recursive" in summary
    assert rag._get_collection(profiles[0].collection_name).count() > 0
    assert rag._get_collection(profiles[1].collection_name).count() > 0

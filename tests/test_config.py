from __future__ import annotations

from pathlib import Path

import pytest

from easy_rag.config import (
    DEFAULT_ENV_VALUES,
    read_env_values,
    save_env_values,
    settings_from_env_values,
)


def test_read_env_values_uses_defaults_when_no_env_file(env_file: Path) -> None:
    values = read_env_values()
    assert values["CHAT_MODEL"] == DEFAULT_ENV_VALUES["CHAT_MODEL"]
    assert values["RETRIEVAL_METHOD"] == "vector"


def test_save_and_read_env_values_roundtrip(env_file: Path) -> None:
    payload = {
        "OPENAI_API_KEY": "saved-key",
        "CHAT_MODEL": "custom-model",
        "TOP_K": "8",
        "MYSQL_ENABLED": "true",
    }
    save_env_values(payload)
    values = read_env_values()
    assert values["OPENAI_API_KEY"] == "saved-key"
    assert values["CHAT_MODEL"] == "custom-model"
    assert values["TOP_K"] == "8"
    assert values["MYSQL_ENABLED"] == "true"


def test_settings_from_env_values_paths(tmp_path: Path) -> None:
    knowledge = tmp_path / "kb"
    chroma = tmp_path / "vec"
    values = dict(DEFAULT_ENV_VALUES)
    values.update(
        {
            "OPENAI_API_KEY": "key",
            "KNOWLEDGE_DIR": str(knowledge),
            "CHROMA_DIR": str(chroma),
        }
    )
    settings = settings_from_env_values(values)
    assert settings.knowledge_dir == knowledge.resolve()
    assert settings.chroma_dir == chroma.resolve()


def test_settings_mysql_tables_parsing() -> None:
    values = dict(DEFAULT_ENV_VALUES)
    values["MYSQL_TABLES"] = "users, orders ,products"
    settings = settings_from_env_values(values)
    assert settings.mysql_tables == ["users", "orders", "products"]


def test_settings_mysql_enabled_bool_parsing() -> None:
    values = dict(DEFAULT_ENV_VALUES)
    values["MYSQL_ENABLED"] = "yes"
    settings = settings_from_env_values(values)
    assert settings.mysql_enabled is True

    values["MYSQL_ENABLED"] = "off"
    settings = settings_from_env_values(values)
    assert settings.mysql_enabled is False


def test_settings_invalid_integer_raises() -> None:
    values = dict(DEFAULT_ENV_VALUES)
    values["CHUNK_SIZE"] = "not-a-number"
    with pytest.raises(ValueError, match="CHUNK_SIZE"):
        settings_from_env_values(values)


def test_settings_invalid_boolean_raises() -> None:
    values = dict(DEFAULT_ENV_VALUES)
    values["MYSQL_ENABLED"] = "maybe"
    with pytest.raises(ValueError, match="MYSQL_ENABLED"):
        settings_from_env_values(values)


def test_migrate_env_file_appends_missing_keys(env_file: Path) -> None:
    from easy_rag.config import migrate_env_file, read_env_values

    env_file.write_text("OPENAI_API_KEY=test-key\nCHAT_MODEL=custom\n", encoding="utf-8")
    assert migrate_env_file() is True

    text = env_file.read_text(encoding="utf-8")
    assert "HF_ENDPOINT=" in text
    assert "CHUNK_STRATEGY=fixed" in text
    assert "API_BIND_PORT=8000" in text

    values = read_env_values()
    assert values["OPENAI_API_KEY"] == "test-key"
    assert values["CHAT_MODEL"] == "custom"
    assert values["HF_ENDPOINT"] == ""


def test_normalize_hf_endpoint() -> None:
    from easy_rag.config import normalize_hf_endpoint

    assert normalize_hf_endpoint("") == ""
    assert normalize_hf_endpoint("  ") == ""
    assert normalize_hf_endpoint("https://hf-mirror.com/") == "https://hf-mirror.com"
    assert normalize_hf_endpoint("hf-mirror.com") == "https://hf-mirror.com"


def test_apply_hf_hub_settings_sets_and_clears_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from easy_rag.config import apply_hf_hub_settings

    values = dict(DEFAULT_ENV_VALUES)
    values["HF_ENDPOINT"] = "https://hf-mirror.com"
    settings = settings_from_env_values(values)

    apply_hf_hub_settings(settings)
    import os

    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"

    values["HF_ENDPOINT"] = ""
    settings = settings_from_env_values(values)
    apply_hf_hub_settings(settings)
    assert "HF_ENDPOINT" not in os.environ


def test_easy_rag_applies_hf_hub_settings(
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    from easy_rag.rag_engine import EasyRAG
    from tests.conftest import make_env_values

    values = make_env_values(
        tmp_dirs,
        base_env_values,
        EMBEDDING_PROVIDER="local",
        HF_ENDPOINT="https://hf-mirror.com",
    )
    settings = settings_from_env_values(values)
    EasyRAG(settings)
    assert os.environ.get("HF_ENDPOINT") == "https://hf-mirror.com"

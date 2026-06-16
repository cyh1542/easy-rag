from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from easy_rag.config import DEFAULT_ENV_VALUES, ENV_FILE, settings_from_env_values
from easy_rag.rag_engine import EasyRAG


@pytest.fixture
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """将 .env 指向临时文件，避免污染项目配置。"""
    path = tmp_path / ".env"
    monkeypatch.setattr("easy_rag.config.ENV_FILE", path)
    return path


@pytest.fixture
def base_env_values() -> dict[str, str]:
    return {
        **DEFAULT_ENV_VALUES,
        "OPENAI_API_KEY": "test-api-key",
        "EMBEDDING_API_KEY": "test-embedding-key",
        "CHAT_MODEL": "gpt-4o-mini",
        "EMBEDDING_PROVIDER": "remote",
        "EMBEDDING_MODEL": "text-embedding-3-small",
    }


@pytest.fixture
def tmp_dirs(tmp_path: Path) -> dict[str, Path]:
    knowledge_dir = tmp_path / "knowledge"
    chroma_dir = tmp_path / "chroma"
    knowledge_dir.mkdir()
    chroma_dir.mkdir()
    return {
        "root": tmp_path,
        "knowledge_dir": knowledge_dir,
        "chroma_dir": chroma_dir,
    }


@pytest.fixture
def sample_markdown(tmp_dirs: dict[str, Path]) -> Path:
    content = (
        "# 测试文档\n\n"
        "RAG 是检索增强生成。\n\n"
        "第二段包含关键词 embedding 和向量数据库。\n\n"
        "第三段用于验证切分与检索。"
    )
    file_path = tmp_dirs["knowledge_dir"] / "sample.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def make_env_values(
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
    **overrides: str,
) -> dict[str, str]:
    values = dict(base_env_values)
    values["KNOWLEDGE_DIR"] = str(tmp_dirs["knowledge_dir"])
    values["CHROMA_DIR"] = str(tmp_dirs["chroma_dir"])
    values.update(overrides)
    return values


@pytest.fixture
def settings(tmp_dirs: dict[str, Path], base_env_values: dict[str, str]) -> Any:
    values = make_env_values(tmp_dirs, base_env_values)
    return settings_from_env_values(values)


@pytest.fixture
def rag(settings: Any) -> EasyRAG:
    return EasyRAG(settings)


def mock_embed_texts(dimension: int = 3) -> Any:
    def _embed(texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for index, text in enumerate(texts):
            vector = [0.0] * dimension
            vector[index % dimension] = 1.0
            if "RAG" in text or "rag" in text.lower():
                vector[0] = 1.0
            results.append(vector)
        return results

    return _embed


@pytest.fixture
def mock_embedding(rag: EasyRAG, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rag, "_embed_texts", mock_embed_texts())

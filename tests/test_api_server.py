from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from easy_rag.api.server import create_api
from tests.conftest import make_env_values


@pytest.fixture
def api_client(env_file: Path, tmp_dirs: dict[str, Path], base_env_values: dict[str, str]) -> TestClient:
    values = make_env_values(tmp_dirs, base_env_values)
    from easy_rag.config import save_env_values

    save_env_values(values)
    return TestClient(create_api())


@pytest.fixture
def masked_api_client(
    env_file: Path,
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
) -> TestClient:
    values = make_env_values(
        tmp_dirs,
        base_env_values,
        API_PUBLIC_BASE_URL="http://www.easy-rag.com",
        API_PATH_PREFIX="www.easy-rag.com",
    )
    from easy_rag.config import save_env_values

    save_env_values(values)
    return TestClient(create_api())


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["retrieval_method"] == "vector"
    assert "knowledge_dir" in payload
    assert "endpoints" in payload


def test_health_endpoint_with_path_prefix(masked_api_client: TestClient) -> None:
    response = masked_api_client.get("/www.easy-rag.com/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_path_prefix"] == "/www.easy-rag.com"
    assert payload["endpoints"]["health"] == "http://www.easy-rag.com/www.easy-rag.com/health"


def test_rag_chat_success(api_client: TestClient, sample_markdown: Path) -> None:
    mock_result = {
        "answer": "测试回答",
        "references": ["sample.md"],
        "contexts": [
            {
                "content": "RAG 内容",
                "source": "sample.md",
                "path": str(sample_markdown),
                "distance": 0.1,
            }
        ],
    }

    with patch("easy_rag.api.server._get_rag") as mock_build:
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_result
        mock_build.return_value = mock_rag

        response = api_client.post(
            "/api/v1/rag/chat",
            json={"question": "RAG 是什么？", "include_contexts": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "测试回答"
    assert payload["references"] == ["sample.md"]
    assert payload["contexts"] is not None
    assert payload["timing"] is not None
    mock_rag.answer.assert_called_once()


def test_rag_chat_with_path_prefix(masked_api_client: TestClient) -> None:
    with patch("easy_rag.api.server._get_rag") as mock_build:
        mock_rag = MagicMock()
        mock_rag.answer.return_value = {
            "answer": "掩饰路径回答",
            "references": [],
            "contexts": [],
        }
        mock_build.return_value = mock_rag

        response = masked_api_client.post(
            "/www.easy-rag.com/api/v1/rag/chat",
            json={"question": "测试", "include_contexts": False},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "掩饰路径回答"


def test_rag_chat_without_contexts(api_client: TestClient) -> None:
    mock_result = {
        "answer": "仅返回答案",
        "references": ["sample.md"],
        "contexts": [],
    }

    with patch("easy_rag.api.server._get_rag") as mock_build:
        mock_rag = MagicMock()
        mock_rag.answer.return_value = mock_result
        mock_build.return_value = mock_rag

        response = api_client.post(
            "/api/v1/rag/chat",
            json={"question": "你好", "include_contexts": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["contexts"] is None


def test_rag_chat_returns_500_on_failure(api_client: TestClient) -> None:
    with patch("easy_rag.api.server._get_rag") as mock_build:
        mock_rag = MagicMock()
        mock_rag.answer.side_effect = RuntimeError("向量库为空")
        mock_build.return_value = mock_rag

        response = api_client.post(
            "/api/v1/rag/chat",
            json={"question": "测试"},
        )

    assert response.status_code == 500
    assert "向量库为空" in response.json()["detail"]


def test_get_rag_reuses_singleton_instance(
    env_file: Path,
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values
    from easy_rag.api.server import _get_rag, _reset_rag_cache

    values = make_env_values(tmp_dirs, base_env_values)
    save_env_values(values)
    _reset_rag_cache()

    with patch("easy_rag.api.server.EasyRAG") as mock_cls:
        first = MagicMock(name="rag-1")
        mock_cls.return_value = first

        assert _get_rag() is first
        assert _get_rag() is first
        mock_cls.assert_called_once()
        first.close.assert_not_called()

    _reset_rag_cache()


def test_get_rag_rebuilds_when_env_changes(
    env_file: Path,
    tmp_dirs: dict[str, Path],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values
    from easy_rag.api.server import _get_rag, _reset_rag_cache

    values = make_env_values(tmp_dirs, base_env_values)
    save_env_values(values)
    _reset_rag_cache()

    with patch("easy_rag.api.server.EasyRAG") as mock_cls:
        first = MagicMock(name="rag-1")
        second = MagicMock(name="rag-2")
        mock_cls.side_effect = [first, second]

        assert _get_rag() is first
        save_env_values({**values, "TOP_K": "8"})
        assert _get_rag() is second
        first.close.assert_called_once()
        assert mock_cls.call_count == 2

    _reset_rag_cache()

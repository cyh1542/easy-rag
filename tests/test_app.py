from __future__ import annotations

from typing import Any

import pytest

from werkzeug.datastructures import MultiDict

from easy_rag.web.app import _collect_form_values, _format_preview_error, app
from easy_rag.timing_utils import StageTimer
from pathlib import Path

from unittest.mock import MagicMock

from openai import AuthenticationError
from easy_rag.config import DEFAULT_ENV_VALUES
from tests.conftest import make_env_values


@pytest.fixture
def flask_client(env_file: Any, tmp_dirs: dict[str, Any], base_env_values: dict[str, str]) -> Any:
    values = make_env_values(tmp_dirs, base_env_values)
    from easy_rag.config import save_env_values

    save_env_values(values)
    return app.test_client()


def test_home_page(flask_client: Any) -> None:
    response = flask_client.get("/")
    assert response.status_code == 200


def test_model_page(flask_client: Any) -> None:
    response = flask_client.get("/model")
    assert response.status_code == 200


def test_mysql_page(flask_client: Any) -> None:
    response = flask_client.get("/mysql")
    assert response.status_code == 200


def test_rag_page(flask_client: Any) -> None:
    response = flask_client.get("/rag")
    assert response.status_code == 200


def test_preview_page(flask_client: Any) -> None:
    response = flask_client.get("/preview")
    assert response.status_code == 200


def test_api_docs_page(flask_client: Any) -> None:
    response = flask_client.get("/api-docs-page")
    assert response.status_code == 200


def test_save_config_persists_values(
    flask_client: Any,
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    response = flask_client.post(
        "/save",
        data={
            "CURRENT_PAGE": "model",
            "OPENAI_API_KEY": "persisted-key",
            "OPENAI_BASE_URL": base_env_values["OPENAI_BASE_URL"],
            "EMBEDDING_API_KEY": base_env_values["EMBEDDING_API_KEY"],
            "EMBEDDING_BASE_URL": base_env_values["EMBEDDING_BASE_URL"],
            "CHAT_MODEL": "persisted-model",
            "EMBEDDING_PROVIDER": "remote",
            "EMBEDDING_MODEL": base_env_values["EMBEDDING_MODEL"],
            "LOCAL_EMBEDDING_MODEL": base_env_values["LOCAL_EMBEDDING_MODEL"],
            "LOCAL_EMBEDDING_DEVICE": base_env_values["LOCAL_EMBEDDING_DEVICE"],
            "HF_EMBEDDING_REPO_ID": base_env_values["HF_EMBEDDING_REPO_ID"],
            "HF_EMBEDDING_CACHE_DIR": base_env_values["HF_EMBEDDING_CACHE_DIR"],
            "COLLECTION_NAME": base_env_values["COLLECTION_NAME"],
            "KNOWLEDGE_DIR": str(tmp_dirs["knowledge_dir"]),
            "CHROMA_DIR": str(tmp_dirs["chroma_dir"]),
            "CHUNK_SIZE": base_env_values["CHUNK_SIZE"],
            "CHUNK_OVERLAP": base_env_values["CHUNK_OVERLAP"],
            "TOP_K": "6",
            "RETRIEVAL_METHOD": "vector",
            "RERANK_CANDIDATE_K": base_env_values["RERANK_CANDIDATE_K"],
            "RRF_K": base_env_values["RRF_K"],
            "MYSQL_ENABLED": "false",
            "MYSQL_HOST": base_env_values["MYSQL_HOST"],
            "MYSQL_PORT": base_env_values["MYSQL_PORT"],
            "MYSQL_USER": base_env_values["MYSQL_USER"],
            "MYSQL_PASSWORD": base_env_values["MYSQL_PASSWORD"],
            "MYSQL_DATABASE": base_env_values["MYSQL_DATABASE"],
            "MYSQL_CHARSET": base_env_values["MYSQL_CHARSET"],
            "MYSQL_TABLES": "",
            "MYSQL_QUERY": "",
            "MYSQL_LIMIT_PER_TABLE": base_env_values["MYSQL_LIMIT_PER_TABLE"],
        },
        follow_redirects=True,
    )
    assert response.status_code == 200

    from easy_rag.config import read_env_values

    values = read_env_values()
    assert values["OPENAI_API_KEY"] == "persisted-key"
    assert values["CHAT_MODEL"] == "persisted-model"
    assert values["TOP_K"] == "6"


def test_preview_files_lists_markdown(
    flask_client: Any,
    sample_markdown: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    response = flask_client.post(
        "/preview-files",
        data={
            "NEXT_PAGE": "rag",
            "KNOWLEDGE_DIR": str(tmp_dirs["knowledge_dir"]),
            "CHROMA_DIR": str(tmp_dirs["chroma_dir"]),
            "COLLECTION_NAME": base_env_values["COLLECTION_NAME"],
            "CHUNK_SIZE": base_env_values["CHUNK_SIZE"],
            "CHUNK_OVERLAP": base_env_values["CHUNK_OVERLAP"],
            "TOP_K": base_env_values["TOP_K"],
            "RETRIEVAL_METHOD": "vector",
            "RERANK_CANDIDATE_K": base_env_values["RERANK_CANDIDATE_K"],
            "RRF_K": base_env_values["RRF_K"],
            "OPENAI_API_KEY": base_env_values["OPENAI_API_KEY"],
            "OPENAI_BASE_URL": base_env_values["OPENAI_BASE_URL"],
            "EMBEDDING_API_KEY": base_env_values["EMBEDDING_API_KEY"],
            "EMBEDDING_BASE_URL": base_env_values["EMBEDDING_BASE_URL"],
            "CHAT_MODEL": base_env_values["CHAT_MODEL"],
            "EMBEDDING_PROVIDER": "remote",
            "EMBEDDING_MODEL": base_env_values["EMBEDDING_MODEL"],
            "LOCAL_EMBEDDING_MODEL": base_env_values["LOCAL_EMBEDDING_MODEL"],
            "LOCAL_EMBEDDING_DEVICE": base_env_values["LOCAL_EMBEDDING_DEVICE"],
            "HF_EMBEDDING_REPO_ID": base_env_values["HF_EMBEDDING_REPO_ID"],
            "HF_EMBEDDING_CACHE_DIR": base_env_values["HF_EMBEDDING_CACHE_DIR"],
            "MYSQL_ENABLED": "false",
            "MYSQL_HOST": base_env_values["MYSQL_HOST"],
            "MYSQL_PORT": base_env_values["MYSQL_PORT"],
            "MYSQL_USER": base_env_values["MYSQL_USER"],
            "MYSQL_PASSWORD": base_env_values["MYSQL_PASSWORD"],
            "MYSQL_DATABASE": base_env_values["MYSQL_DATABASE"],
            "MYSQL_CHARSET": base_env_values["MYSQL_CHARSET"],
            "MYSQL_TABLES": "",
            "MYSQL_QUERY": "",
            "MYSQL_LIMIT_PER_TABLE": base_env_values["MYSQL_LIMIT_PER_TABLE"],
        },
    )
    assert response.status_code == 200
    assert "sample.md" in response.get_data(as_text=True)


def test_collect_form_values_preserves_missing_fields(
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values

    values = make_env_values(
        tmp_dirs,
        base_env_values,
        KNOWLEDGE_DIR=str(tmp_dirs["knowledge_dir"]),
        OPENAI_API_KEY="stored-openai-key",
        EMBEDDING_API_KEY="stored-embedding-key",
    )
    save_env_values(values)

    partial_form = MultiDict(
        [
            ("OPENAI_API_KEY", "stored-openai-key"),
            ("OPENAI_BASE_URL", base_env_values["OPENAI_BASE_URL"]),
            ("CHAT_MODEL", "updated-model"),
            ("EMBEDDING_PROVIDER", "remote"),
            ("EMBEDDING_MODEL", base_env_values["EMBEDDING_MODEL"]),
        ]
    )
    collected = _collect_form_values(partial_form)

    assert collected["CHAT_MODEL"] == "updated-model"
    assert collected["KNOWLEDGE_DIR"] == str(tmp_dirs["knowledge_dir"])
    assert collected["OPENAI_API_KEY"] == "stored-openai-key"
    assert collected["EMBEDDING_API_KEY"] == "stored-embedding-key"


def test_collect_form_values_preserves_empty_submission(
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values

    values = make_env_values(
        tmp_dirs,
        base_env_values,
        OPENAI_API_KEY="wrong-but-stored-key",
        EMBEDDING_API_KEY="stored-embedding-key",
        MYSQL_PASSWORD="stored-mysql-password",
    )
    save_env_values(values)

    partial_form = MultiDict(
        [
            ("OPENAI_API_KEY", ""),
            ("OPENAI_BASE_URL", base_env_values["OPENAI_BASE_URL"]),
            ("CHAT_MODEL", base_env_values["CHAT_MODEL"]),
            ("EMBEDDING_PROVIDER", "remote"),
            ("EMBEDDING_API_KEY", ""),
            ("EMBEDDING_BASE_URL", base_env_values["EMBEDDING_BASE_URL"]),
            ("EMBEDDING_MODEL", base_env_values["EMBEDDING_MODEL"]),
            ("MYSQL_PASSWORD", ""),
        ]
    )
    collected = _collect_form_values(partial_form)

    assert collected["OPENAI_API_KEY"] == "wrong-but-stored-key"
    assert collected["EMBEDDING_API_KEY"] == "stored-embedding-key"
    assert collected["MYSQL_PASSWORD"] == "stored-mysql-password"
    assert collected["OPENAI_API_KEY"] != DEFAULT_ENV_VALUES["OPENAI_API_KEY"]


def test_navigate_after_save_uses_nav_button_target(
    flask_client: Any,
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="saved-model")
    values["CURRENT_PAGE"] = "model"
    save_response = flask_client.post("/save", data=values)
    assert save_response.status_code == 200
    assert "saved-model" in save_response.get_data(as_text=True)

    nav_values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="saved-model")
    nav_values["CURRENT_PAGE"] = "model"
    nav_values["NEXT_PAGE"] = "rag"
    nav_response = flask_client.post("/draft-navigate", data=nav_values, follow_redirects=False)
    assert nav_response.status_code == 302
    assert nav_response.headers["Location"].endswith("/rag")

    rag_page = flask_client.get("/rag")
    assert "saved-model" in rag_page.get_data(as_text=True)


def test_draft_navigate_preserves_session_values(
    flask_client: Any,
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values

    values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="disk-model")
    save_env_values(values)

    draft_values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="draft-model")
    draft_values["NEXT_PAGE"] = "rag"
    response = flask_client.post("/draft-navigate", data=draft_values, follow_redirects=False)
    assert response.status_code == 302

    model_page = flask_client.get("/model")
    assert "draft-model" in model_page.get_data(as_text=True)
    assert "disk-model" not in model_page.get_data(as_text=True)


def test_reload_config_clears_session_draft(
    flask_client: Any,
    env_file: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    from easy_rag.config import save_env_values

    values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="disk-model")
    save_env_values(values)

    draft_values = make_env_values(tmp_dirs, base_env_values, CHAT_MODEL="draft-model")
    draft_values["NEXT_PAGE"] = "model"
    flask_client.post("/draft-navigate", data=draft_values)

    reload_values = make_env_values(tmp_dirs, base_env_values)
    reload_values["CURRENT_PAGE"] = "model"
    flask_client.post("/reload", data=reload_values, follow_redirects=True)

    model_page = flask_client.get("/model")
    assert "disk-model" in model_page.get_data(as_text=True)
    assert "draft-model" not in model_page.get_data(as_text=True)


def test_preview_page_shows_batch_eval_section(flask_client: Any) -> None:
    response = flask_client.get("/preview")
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "EVAL_CASES_FILE" in text
    assert "运行批量评测" in text
    assert "homework_sop_test_cases" in text


def test_run_batch_eval_renders_metrics(
    flask_client: Any,
    tmp_dirs: dict[str, Any],
    base_env_values: dict[str, str],
) -> None:
    from unittest.mock import patch

    values = make_env_values(tmp_dirs, base_env_values)
    values.update(
        {
            "CURRENT_PAGE": "preview",
            "EVAL_CASES_FILE": "tests/eval/homework_sop_test_cases.json",
            "EVAL_CASE_LIMIT": "2",
            "EVAL_API_BASE_URL": "http://127.0.0.1:8000",
        }
    )
    mock_report = {
        "base_url": "http://127.0.0.1:8000",
        "cases_path_display": "tests/eval/homework_sop_test_cases.json",
        "summary": {
            "total_cases": 2,
            "successful_cases": 2,
            "failed_cases": 0,
            "retrieval_recall": 0.5,
            "answer_accuracy": 0.75,
            "avg_retrieval_keyword_recall": 0.5,
            "avg_answer_keyword_accuracy": 0.75,
            "avg_latency_ms": 123.0,
            "by_category": {
                "适用场景": {
                    "total": 2,
                    "retrieval_hit": 1,
                    "answer_hit": 2,
                    "retrieval_recall": 0.5,
                    "answer_accuracy": 1.0,
                }
            },
        },
        "results": [],
    }

    with patch("easy_rag.web.app.run_evaluation", return_value=mock_report):
        with patch(
            "easy_rag.web.app.resolve_eval_cases_path",
            return_value=Path("tests/eval/homework_sop_test_cases.json"),
        ):
            response = flask_client.post("/run-batch-eval", data=values)

    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "召回率" in text
    assert "准确率" in text
    assert "123" in text
    assert "批量评测完成" in text


def _auth_error() -> AuthenticationError:
    return AuthenticationError(
        "bad key",
        response=MagicMock(status_code=401),
        body={"error": "invalid_api_key"},
    )


def test_format_preview_error_openai_auth() -> None:
    timer = StageTimer()
    timer.start("vector_retrieve")
    timer.end("vector_retrieve")
    timer.start("llm_completion")
    message = _format_preview_error(_auth_error(), timer)
    assert "OPENAI_API_KEY" in message
    assert "EMBEDDING_API_KEY" not in message


def test_format_preview_error_embedding_auth() -> None:
    timer = StageTimer()
    timer.start("vector_retrieve")
    message = _format_preview_error(_auth_error(), timer)
    assert "EMBEDDING_API_KEY" in message
    assert "OPENAI_API_KEY" not in message

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import AuthenticationError

import os

from flask import Flask, has_request_context, redirect, render_template, request, session, url_for

from easy_rag.config import (
    DEFAULT_ENV_VALUES,
    PROJECT_ROOT,
    build_api_public_url,
    decode_prompt_text,
    read_env_values,
    save_env_values,
    settings_from_env_values,
)
from easy_rag.eval_runner import (
    default_api_base_url,
    list_eval_datasets,
    normalize_eval_api_base_url,
    resolve_eval_cases_path,
    run_evaluation,
)
from easy_rag.knowledge_bases import (
    MAX_KB_PROFILES,
    load_knowledge_base_registry,
    parse_registry_from_form,
    registry_from_settings,
    save_knowledge_base_registry,
)
from easy_rag.logger_config import LOG_FILE, setup_logging
from easy_rag.rag_engine import EasyRAG
from easy_rag.timing_utils import StageTimer


TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


FORM_FIELDS = [
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "CHAT_MODEL",
    "CHAT_THINKING_PROMPT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_DEVICE",
    "HF_EMBEDDING_REPO_ID",
    "HF_EMBEDDING_CACHE_DIR",
    "HF_ENDPOINT",
    "COLLECTION_NAME",
    "KNOWLEDGE_DIR",
    "CHROMA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_STRATEGY",
    "SEMANTIC_CHUNK_THRESHOLD",
    "TOP_K",
    "RETRIEVAL_METHOD",
    "RERANK_CANDIDATE_K",
    "RRF_K",
    "MYSQL_ENABLED",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "MYSQL_CHARSET",
    "MYSQL_TABLES",
    "MYSQL_QUERY",
    "MYSQL_LIMIT_PER_TABLE",
    "API_PUBLIC_BASE_URL",
    "API_PATH_PREFIX",
    "API_BIND_HOST",
    "API_BIND_PORT",
    "RAG_API_KEY",
]

NUMBER_FIELDS = {
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "TOP_K",
    "MYSQL_PORT",
    "MYSQL_LIMIT_PER_TABLE",
    "API_BIND_PORT",
}

BOOLEAN_FIELDS = {"MYSQL_ENABLED"}
EMBEDDING_MODE_MAP = {
    "remote": "远程接口",
    "local": "本地模型",
    "huggingface": "Hugging Face 下载",
}
CHUNK_STRATEGY_LABELS = {
    "fixed": "固定大小分块",
    "recursive": "递归分块",
    "semantic": "语义分块",
    "structure": "基于文档结构的分块",
}

SESSION_FORM_KEY = "draft_form_values"

MODEL_PAGE_FIELDS = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "CHAT_MODEL",
    "CHAT_THINKING_PROMPT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_MODEL",
    "LOCAL_EMBEDDING_DEVICE",
    "HF_EMBEDDING_REPO_ID",
    "HF_EMBEDDING_CACHE_DIR",
    "HF_ENDPOINT",
}

RAG_PAGE_FIELDS = {
    "COLLECTION_NAME",
    "KNOWLEDGE_DIR",
    "CHROMA_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "CHUNK_STRATEGY",
    "SEMANTIC_CHUNK_THRESHOLD",
    "TOP_K",
    "RETRIEVAL_METHOD",
    "RERANK_CANDIDATE_K",
    "RRF_K",
}

MYSQL_PAGE_FIELDS = {
    "MYSQL_ENABLED",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "MYSQL_CHARSET",
    "MYSQL_TABLES",
    "MYSQL_QUERY",
    "MYSQL_LIMIT_PER_TABLE",
}

API_PAGE_FIELDS = {
    "API_PUBLIC_BASE_URL",
    "API_PATH_PREFIX",
    "API_BIND_HOST",
    "API_BIND_PORT",
    "RAG_API_KEY",
}


app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "easy-rag-local-session-key")
logger = setup_logging()
app.logger.handlers = logger.handlers
app.logger.setLevel(logger.level)


def _get_disk_form_values() -> dict[str, str]:
    values = read_env_values()
    result = {key: values.get(key, DEFAULT_ENV_VALUES.get(key, "")) for key in FORM_FIELDS}
    if result.get("CHAT_THINKING_PROMPT"):
        result["CHAT_THINKING_PROMPT"] = decode_prompt_text(result["CHAT_THINKING_PROMPT"])
    return result


def _load_form_values() -> dict[str, str]:
    disk_values = _get_disk_form_values()
    if not has_request_context():
        return disk_values

    draft = session.get(SESSION_FORM_KEY)
    if not isinstance(draft, dict):
        return disk_values

    merged = disk_values.copy()
    for key in FORM_FIELDS:
        if key in draft:
            merged[key] = str(draft[key])
    return merged


def _save_session_draft(values: dict[str, str]) -> None:
    if not has_request_context():
        return
    session[SESSION_FORM_KEY] = {key: values[key] for key in FORM_FIELDS}
    session.modified = True


def _clear_session_draft() -> None:
    if not has_request_context():
        return
    session.pop(SESSION_FORM_KEY, None)
    session.modified = True


def _persist_form_draft(form_data: Any) -> dict[str, str]:
    values = _collect_form_values(form_data)
    _save_session_draft(values)
    return values


def _collect_form_values(form_data: Any) -> dict[str, str]:
    current_values = _load_form_values()
    values: dict[str, str] = {}
    for key in FORM_FIELDS:
        if key in BOOLEAN_FIELDS:
            if key in form_data:
                submitted_values = form_data.getlist(key)
                values[key] = "true" if any(item in {"true", "on", "1"} for item in submitted_values) else "false"
            else:
                values[key] = current_values.get(key, DEFAULT_ENV_VALUES.get(key, "false"))
            continue

        if key not in form_data:
            values[key] = current_values.get(key, DEFAULT_ENV_VALUES.get(key, ""))
            continue

        submitted = str(form_data.get(key, "")).strip()
        if submitted:
            values[key] = submitted
        else:
            values[key] = current_values.get(key, DEFAULT_ENV_VALUES.get(key, ""))
    return values


def _resolve_preview_dir(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _build_rag_from_values(values: dict[str, str]) -> tuple[EasyRAG | None, str | None]:
    try:
        settings = settings_from_env_values(values)
        return EasyRAG(settings), None
    except Exception as exc:
        logger.exception("构建 EasyRAG 实例失败")
        return None, str(exc)


def _kb_profiles_for_template(values: dict[str, str]) -> list[Any]:
    try:
        settings = settings_from_env_values(values)
    except Exception:
        settings = settings_from_env_values(read_env_values())
    registry = load_knowledge_base_registry(settings)
    profiles = list(registry.bases)
    if not profiles:
        profiles = registry_from_settings(settings).bases
    while len(profiles) < MAX_KB_PROFILES:
        defaults = registry_from_settings(settings).bases
        profiles.append(defaults[len(profiles) % len(defaults)])
    return profiles[:MAX_KB_PROFILES]


def _maybe_save_kb_registry(form_data: Any, values: dict[str, str]) -> None:
    if str(form_data.get("CURRENT_PAGE", "")).strip() != "rag":
        return
    settings = settings_from_env_values(values)
    registry = parse_registry_from_form(form_data, settings)
    save_knowledge_base_registry(registry)


def _format_preview_error(exc: Exception, timer: StageTimer) -> str:
    if isinstance(exc, ValueError):
        message = str(exc)
        if "OPENAI_API_KEY" in message or "EMBEDDING_API_KEY" in message:
            return message
        return f"模型效果预览失败：{message}"

    if isinstance(exc, AuthenticationError):
        stage_names = [item["name"] for item in timer.summary().get("stages", [])]
        if timer.has_active("llm_completion") or "llm_completion" in stage_names:
            return "OPENAI_API_KEY 认证失败，请检查模型设置页中的 OPENAI_API_KEY 与 OPENAI_BASE_URL。"
        return "EMBEDDING_API_KEY 认证失败，请检查模型设置页中的 EMBEDDING_API_KEY 与 EMBEDDING_BASE_URL。"

    return f"模型效果预览失败：{exc}"


def _build_api_endpoint_urls(values: dict[str, str]) -> dict[str, str]:
    try:
        settings = settings_from_env_values(values)
    except Exception:
        return {
            "health": "/health",
            "chat": "/api/v1/rag/chat",
            "docs": "/docs",
            "redoc": "/redoc",
        }

    return {
        "health": build_api_public_url(settings, "/health"),
        "chat": build_api_public_url(settings, "/api/v1/rag/chat"),
        "docs": build_api_public_url(settings, "/docs"),
        "redoc": build_api_public_url(settings, "/redoc"),
    }


def _build_context(
    form_values: dict[str, str] | None = None,
    *,
    message: str = "",
    error: str = "",
    file_rows: list[dict[str, str]] | None = None,
    mysql_info: dict[str, Any] | None = None,
    mysql_table_preview: dict[str, Any] | None = None,
    mysql_preview_table_name: str = "",
    build_summary: dict[str, Any] | None = None,
    preview_question: str = "",
    preview_result: dict[str, Any] | None = None,
    eval_cases_file: str = "",
    eval_case_limit: str = "10",
    eval_api_base_url: str = "",
    eval_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = form_values or _load_form_values()
    knowledge_dir = values.get("KNOWLEDGE_DIR", DEFAULT_ENV_VALUES["KNOWLEDGE_DIR"])
    settings_error = ""

    try:
        settings_from_env_values(values)
    except Exception as exc:
        logger.exception("页面上下文配置校验失败")
        settings_error = str(exc)

    eval_options = list_eval_datasets()
    default_eval_file = eval_options[0]["path"] if eval_options else "tests/eval/homework_sop_test_cases.json"

    try:
        kb_settings = settings_from_env_values(values)
    except Exception:
        kb_settings = settings_from_env_values(read_env_values())
    kb_registry = load_knowledge_base_registry(kb_settings)

    return {
        "form_values": values,
        "message": message,
        "error": error,
        "settings_error": settings_error or "",
        "knowledge_preview_dir": str(_resolve_preview_dir(knowledge_dir)),
        "embedding_mode_text": EMBEDDING_MODE_MAP.get(values.get("EMBEDDING_PROVIDER", ""), "未设置"),
        "chunk_strategy_text": CHUNK_STRATEGY_LABELS.get(
            values.get("CHUNK_STRATEGY", DEFAULT_ENV_VALUES["CHUNK_STRATEGY"]),
            "未设置",
        ),
        "chunk_strategy_options": list(CHUNK_STRATEGY_LABELS.keys()),
        "mysql_status_text": "已启用" if values.get("MYSQL_ENABLED", "false").lower() == "true" else "未启用",
        "file_rows": file_rows or [],
        "mysql_info": mysql_info,
        "mysql_table_preview": mysql_table_preview,
        "mysql_preview_table_name": mysql_preview_table_name,
        "build_summary": build_summary,
        "preview_question": preview_question,
        "preview_result": preview_result,
        "provider_options": ["remote", "local", "huggingface"],
        "rag_ready": settings_error == "",
        "log_file_path": str(LOG_FILE),
        "api_endpoint_urls": _build_api_endpoint_urls(values),
        "model_page_fields": sorted(MODEL_PAGE_FIELDS),
        "rag_page_fields": sorted(RAG_PAGE_FIELDS),
        "mysql_page_fields": sorted(MYSQL_PAGE_FIELDS),
        "api_page_fields": sorted(API_PAGE_FIELDS),
        "eval_dataset_options": eval_options,
        "eval_cases_file": eval_cases_file or default_eval_file,
        "eval_case_limit": eval_case_limit,
        "eval_api_base_url": eval_api_base_url or default_api_base_url(),
        "eval_report": eval_report,
        "eval_summary": eval_report.get("summary") if eval_report else None,
        "kb_registry": kb_registry,
        "kb_profiles": _kb_profiles_for_template(values),
        "kb_profile_slots": list(range(1, MAX_KB_PROFILES + 1)),
    }


def _render_page(template_name: str, **kwargs: Any) -> str:
    return render_template(template_name, **_build_context(**kwargs))


@app.get("/")
def home() -> str:
    return _render_page("home.html")


@app.get("/model")
def model_page() -> str:
    return _render_page("model.html")


@app.get("/mysql")
def mysql_page() -> str:
    return _render_page("mysql.html")


@app.get("/rag")
def rag_page() -> str:
    return _render_page("rag.html")


@app.get("/preview")
def preview_page() -> str:
    return _render_page("preview.html")


@app.get("/api-docs-page")
def api_docs_page() -> str:
    return _render_page("api_docs.html")


@app.post("/draft-navigate")
def draft_navigate() -> str:
    _persist_form_draft(request.form)
    next_page = request.form.get("NEXT_PAGE", "home")
    return redirect(_safe_next_url(next_page))


@app.post("/save")
def save_config() -> str:
    values = _persist_form_draft(request.form)
    try:
        save_env_values(values)
        _maybe_save_kb_registry(request.form, values)
        logger.info("配置已保存到 .env")
        context = _build_context(values, message="配置已保存到 .env。")
    except Exception as exc:
        logger.exception("保存配置失败")
        context = _build_context(values, error=f"保存失败：{exc}")
    return render_template(_resolve_template_name(request), **context)


@app.post("/reload")
def reload_config() -> str:
    _clear_session_draft()
    next_page = _current_page_from_form(request.form)
    return redirect(_safe_next_url(next_page))


@app.post("/preview-files")
def preview_files() -> str:
    values = _persist_form_draft(request.form)
    rag_instance, error = _build_rag_from_values(values)
    if error or rag_instance is None:
        return render_template("rag.html", **_build_context(values, error=error or "配置无法解析。"))

    try:
        files = rag_instance.list_supported_files(rag_instance.settings.knowledge_dir)
        rows = [
            {
                "name": file_path.name,
                "suffix": file_path.suffix.lower(),
                "path": str(file_path),
            }
            for file_path in files[:100]
        ]
        logger.info("知识库文件预览完成: count=%s dir=%s", len(files), rag_instance.settings.knowledge_dir)
        message = f"共发现 {len(files)} 个可索引文件。" if files else "当前目录下没有找到支持的文件类型。"
        return render_template("rag.html", **_build_context(values, message=message, file_rows=rows))
    except Exception as exc:
        logger.exception("知识库文件预览失败")
        return render_template("rag.html", **_build_context(values, error=f"知识库文件预览失败：{exc}"))


@app.post("/test-mysql")
def test_mysql() -> str:
    values = _persist_form_draft(request.form)
    rag_instance, error = _build_rag_from_values(values)
    if error or rag_instance is None:
        return render_template("mysql.html", **_build_context(values, error=error or "配置无法解析。"))

    try:
        info = rag_instance.test_mysql_connection()
        save_env_values(values)
        logger.info(
            "MySQL 测试成功: database=%s table_count=%s",
            info.get("database"),
            info.get("table_count"),
        )
        return render_template(
            "mysql.html",
            **_build_context(values, message="MySQL 连接成功，当前配置已自动保存，可直接用于建索引和模型预览。", mysql_info=info),
        )
    except Exception as exc:
        logger.exception("MySQL 测试失败")
        return render_template("mysql.html", **_build_context(values, error=f"MySQL 连接失败：{exc}"))


@app.post("/preview-mysql-table")
def preview_mysql_table() -> str:
    values = _persist_form_draft(request.form)
    preview_table_name = str(request.form.get("MYSQL_PREVIEW_TABLE", "")).strip()
    rag_instance, error = _build_rag_from_values(values)

    if not preview_table_name:
        return render_template(
            "mysql.html",
            **_build_context(
                values,
                error="请输入要预览的表名。",
                mysql_preview_table_name=preview_table_name,
            ),
        )

    if error or rag_instance is None:
        return render_template(
            "mysql.html",
            **_build_context(
                values,
                error=error or "配置无法解析。",
                mysql_preview_table_name=preview_table_name,
            ),
        )

    try:
        preview_data = rag_instance.preview_mysql_table(preview_table_name, limit=10)
        save_env_values(values)
        logger.info(
            "MySQL 表数据预览成功: table=%s row_count=%s",
            preview_data.get("table_name"),
            preview_data.get("row_count"),
        )
        return render_template(
            "mysql.html",
            **_build_context(
                values,
                message=f"MySQL 表 {preview_data.get('table_name')} 的前 10 条数据预览完成。",
                mysql_table_preview=preview_data,
                mysql_preview_table_name=preview_table_name,
            ),
        )
    except Exception as exc:
        logger.exception("MySQL 表数据预览失败")
        return render_template(
            "mysql.html",
            **_build_context(
                values,
                error=f"MySQL 表数据预览失败：{exc}",
                mysql_preview_table_name=preview_table_name,
            ),
        )


@app.post("/build-index")
def build_index() -> str:
    values = _persist_form_draft(request.form)
    save_env_values(values)
    try:
        _maybe_save_kb_registry(request.form, values)
    except Exception as exc:
        return render_template("rag.html", **_build_context(values, error=f"多知识库配置无效：{exc}"))

    rag_instance, error = _build_rag_from_values(values)
    if error or rag_instance is None:
        return render_template("rag.html", **_build_context(values, error=error or "配置无法解析。"))

    reset_index = request.form.get("RESET_INDEX") in {"true", "on", "1"}
    settings = settings_from_env_values(values)
    registry = load_knowledge_base_registry(settings)

    try:
        if registry.enabled and registry.active_profiles():
            logger.info(
                "开始构建多知识库索引: profiles=%s knowledge_dir=%s",
                [profile.id for profile in registry.active_profiles()],
                rag_instance.settings.knowledge_dir,
            )
            summary = rag_instance.build_multi_knowledge_bases(
                registry.active_profiles(),
                reset=reset_index,
            )
        else:
            logger.info("开始构建索引: reset=%s knowledge_dir=%s", reset_index, rag_instance.settings.knowledge_dir)
            summary = rag_instance.build_index(reset=reset_index)
        logger.info("索引构建完成: summary=%s", summary)
        return render_template(
            "rag.html",
            **_build_context(values, message="索引构建完成。", build_summary=summary),
        )
    except Exception as exc:
        logger.exception("索引构建失败")
        return render_template("rag.html", **_build_context(values, error=f"索引构建失败：{exc}"))


@app.post("/preview-answer")
def preview_answer() -> str:
    timer = StageTimer()
    started_at = timer.start("preview_request")
    logger.info("STAGE_START | preview_request | started_at=%s", started_at)
    values = _persist_form_draft(request.form)
    preview_question = str(request.form.get("PREVIEW_QUESTION", "")).strip()

    started_at = timer.start("load_preview_config")
    logger.info("STAGE_START | load_preview_config | started_at=%s", started_at)
    record = timer.end("load_preview_config")
    logger.info(
        "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
        record.name,
        record.started_at,
        record.ended_at,
        record.duration_ms,
    )

    started_at = timer.start("rag_init")
    logger.info("STAGE_START | rag_init | started_at=%s", started_at)
    rag_instance, error = _build_rag_from_values(values)
    record = timer.end("rag_init")
    logger.info(
        "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
        record.name,
        record.started_at,
        record.ended_at,
        record.duration_ms,
    )

    if not preview_question:
        record = timer.end("preview_request")
        logger.info(
            "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
            record.name,
            record.started_at,
            record.ended_at,
            record.duration_ms,
        )
        return render_template(
            "preview.html",
            **_build_context(
                values,
                error="请输入要测试的预览问题。",
                preview_question=preview_question,
            ),
        )

    if error or rag_instance is None:
        record = timer.end("preview_request")
        logger.info(
            "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
            record.name,
            record.started_at,
            record.ended_at,
            record.duration_ms,
        )
        return render_template(
            "preview.html",
            **_build_context(
                values,
                error=error or "已保存配置无法解析，请先到模型设置页或 RAG 设置页修正配置。",
                preview_question=preview_question,
            ),
        )

    try:
        logger.info("BEFORE_MODEL_TEST | question=%s", preview_question)
        print(f"BEFORE_MODEL_TEST | question={preview_question}", flush=True)
        started_at = timer.start("model_preview")
        logger.info("STAGE_START | model_preview | started_at=%s", started_at)
        result = rag_instance.answer(preview_question, timer=timer)
        record = timer.end("model_preview")
        logger.info(
            "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
            record.name,
            record.started_at,
            record.ended_at,
            record.duration_ms,
        )

        record = timer.end("preview_request")
        logger.info(
            "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
            record.name,
            record.started_at,
            record.ended_at,
            record.duration_ms,
        )

        timing_summary = timer.summary()
        logger.info("MODEL_PREVIEW_TIMING_SUMMARY | %s", timing_summary)
        logger.info("MODEL_PREVIEW_TIMING_CHAIN | %s", timing_summary.get("chain_text", ""))
        print(f"MODEL_PREVIEW_TIMING_SUMMARY | {timing_summary}", flush=True)
        print(f"MODEL_PREVIEW_TIMING_CHAIN | {timing_summary.get('chain_text', '')}", flush=True)
        logger.info(
            "AFTER_MODEL_TEST | question=%s | references=%s | context_count=%s",
            preview_question,
            result.get("references"),
            len(result.get("contexts", [])),
        )
        print(
            f"AFTER_MODEL_TEST | question={preview_question} | references={result.get('references')} | context_count={len(result.get('contexts', []))}",
            flush=True,
        )
        return render_template(
            "preview.html",
            **_build_context(
                values,
                message="模型效果预览完成。",
                preview_question=preview_question,
                preview_result=result,
            ),
        )
    except Exception as exc:
        logger.exception("模型效果预览失败")
        record = timer.end_if_active("preview_request")
        if record:
            logger.info(
                "STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
                record.name,
                record.started_at,
                record.ended_at,
                record.duration_ms,
            )
        failure_summary = timer.summary()
        logger.info("MODEL_PREVIEW_TIMING_SUMMARY_FAILED | %s", failure_summary)
        logger.info("MODEL_PREVIEW_TIMING_CHAIN_FAILED | %s", failure_summary.get("chain_text", ""))
        return render_template(
            "preview.html",
            **_build_context(
                values,
                error=_format_preview_error(exc, timer),
                preview_question=preview_question,
            ),
        )


@app.post("/run-batch-eval")
def run_batch_eval() -> str:
    values = _persist_form_draft(request.form)
    eval_cases_file = str(request.form.get("EVAL_CASES_FILE", "")).strip()
    eval_case_limit_raw = str(request.form.get("EVAL_CASE_LIMIT", "10")).strip()
    eval_api_base_url = str(request.form.get("EVAL_API_BASE_URL", "")).strip()
    preview_question = str(request.form.get("PREVIEW_QUESTION", "")).strip()

    try:
        limit = int(eval_case_limit_raw or "10")
        if limit < 0:
            raise ValueError("评测条数不能为负数。")
    except ValueError as exc:
        return render_template(
            "preview.html",
            **_build_context(
                values,
                error=f"评测条数无效：{exc}",
                eval_cases_file=eval_cases_file,
                eval_case_limit=eval_case_limit_raw or "10",
                eval_api_base_url=eval_api_base_url,
                preview_question=preview_question,
            ),
        )

    try:
        cases_path = resolve_eval_cases_path(eval_cases_file)
        logger.info(
            "开始批量评测: cases=%s limit=%s api=%s",
            cases_path,
            limit,
            eval_api_base_url or "default",
        )
        report = run_evaluation(
            cases_path,
            base_url=normalize_eval_api_base_url(eval_api_base_url),
            limit=limit,
            timeout=120.0,
        )
        summary = report["summary"]
        message = (
            f"批量评测完成：运行 {summary['total_cases']} 条，"
            f"召回率 {summary['retrieval_recall']:.1%}，"
            f"准确率 {summary['answer_accuracy']:.1%}，"
            f"平均响应 {summary['avg_latency_ms']:.0f} ms"
        )
        return render_template(
            "preview.html",
            **_build_context(
                values,
                message=message,
                eval_cases_file=report["cases_path_display"],
                eval_case_limit=eval_case_limit_raw or "10",
                eval_api_base_url=report["base_url"],
                eval_report=report,
                preview_question=preview_question,
            ),
        )
    except Exception as exc:
        logger.exception("批量评测失败")
        return render_template(
            "preview.html",
            **_build_context(
                values,
                error=f"批量评测失败：{exc}",
                eval_cases_file=eval_cases_file,
                eval_case_limit=eval_case_limit_raw or "10",
                eval_api_base_url=eval_api_base_url,
                preview_question=preview_question,
            ),
        )


def _current_page_from_form(form_data: Any) -> str:
    return str(form_data.get("CURRENT_PAGE") or form_data.get("NEXT_PAGE") or "home")


def _resolve_template_name(req: Any) -> str:
    next_page = _current_page_from_form(req.form)
    page_map = {
        "home": "home.html",
        "model": "model.html",
        "mysql": "mysql.html",
        "rag": "rag.html",
        "preview": "preview.html",
        "api_docs": "api_docs.html",
    }
    return page_map.get(next_page, "home.html")


def _safe_next_url(next_page: str) -> str:
    page_map = {
        "home": url_for("home"),
        "model": url_for("model_page"),
        "mysql": url_for("mysql_page"),
        "rag": url_for("rag_page"),
        "preview": url_for("preview_page"),
        "api_docs": url_for("api_docs_page"),
    }
    return page_map.get(next_page, url_for("home"))


@app.errorhandler(Exception)
def handle_unexpected_error(exc: Exception) -> tuple[str, int]:
    logger.exception("Flask 未捕获异常")
    context = _build_context(error=f"系统发生未捕获异常：{exc}")
    return render_template("home.html", **context), 500


def main() -> None:
    import os

    host = os.getenv("EASY_RAG_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("EASY_RAG_WEB_PORT", "5000"))
    logger.info("Flask 服务启动: http://%s:%s", host, port)
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

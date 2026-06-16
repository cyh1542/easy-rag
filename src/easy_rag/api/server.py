from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from easy_rag.config import ENV_FILE, build_api_public_url, get_settings, read_env_values
from easy_rag.logger_config import setup_logging
from easy_rag.rag_engine import EasyRAG
from easy_rag.timing_utils import StageTimer


logger = setup_logging()
router = APIRouter()

_rag_lock = threading.Lock()
_rag_instance: EasyRAG | None = None
_rag_cache_key: tuple[Any, ...] | None = None

_RAG_CACHE_KEYS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "CHAT_MODEL",
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
    "CHUNK_STRATEGY",
    "TOP_K",
    "RETRIEVAL_METHOD",
    "RERANK_CANDIDATE_K",
    "RRF_K",
    "MYSQL_ENABLED",
)


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户问题")
    include_contexts: bool = Field(default=True, description="是否返回命中上下文")


class ChatResponse(BaseModel):
    question: str
    answer: str
    references: list[str]
    contexts: list[dict[str, Any]] | None = None
    timing: dict[str, Any] | None = None


def _build_rag_cache_key() -> tuple[Any, ...]:
    env_mtime = ENV_FILE.stat().st_mtime if ENV_FILE.exists() else 0.0
    values = read_env_values()
    return (env_mtime, *(values.get(key, "") for key in _RAG_CACHE_KEYS))


def _reset_rag_cache() -> None:
    global _rag_instance, _rag_cache_key
    with _rag_lock:
        if _rag_instance is not None:
            _rag_instance.close()
            _rag_instance = None
        _rag_cache_key = None


def _get_rag() -> EasyRAG:
    global _rag_instance, _rag_cache_key

    cache_key = _build_rag_cache_key()
    with _rag_lock:
        if _rag_instance is None or _rag_cache_key != cache_key:
            if _rag_instance is not None:
                logger.info("检测到配置变更，刷新 EasyRAG 实例")
                _rag_instance.close()
            settings = get_settings()
            _rag_instance = EasyRAG(settings)
            _rag_cache_key = cache_key
            logger.info(
                "EasyRAG 实例已就绪: embedding=%s retrieval=%s collection=%s",
                settings.embedding_provider,
                settings.retrieval_method,
                settings.collection_name,
            )
        return _rag_instance


def _build_rag() -> EasyRAG:
    """向后兼容：复用单例实例，避免每次请求重复加载模型与 Chroma。"""
    return _get_rag()


@router.get("/health")
def health() -> dict[str, Any]:
    try:
        settings = get_settings()
        return {
            "status": "ok",
            "chat_model": settings.chat_model,
            "embedding_provider": settings.embedding_provider,
            "retrieval_method": settings.retrieval_method,
            "knowledge_dir": str(settings.knowledge_dir),
            "api_public_base_url": settings.api_public_base_url,
            "api_path_prefix": settings.api_path_prefix,
            "rag_instance_ready": _rag_instance is not None,
            "endpoints": {
                "health": build_api_public_url(settings, "/health"),
                "chat": build_api_public_url(settings, "/api/v1/rag/chat"),
                "docs": build_api_public_url(settings, "/docs"),
                "redoc": build_api_public_url(settings, "/redoc"),
            },
        }
    except Exception as exc:
        logger.exception("健康检查失败")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/api/v1/rag/chat", response_model=ChatResponse)
def rag_chat(payload: ChatRequest) -> ChatResponse:
    timer = StageTimer()
    started_at = timer.start("api_rag_chat")
    logger.info("API_STAGE_START | api_rag_chat | started_at=%s", started_at)

    try:
        rag = _get_rag()
        result = rag.answer(payload.question, timer=timer)
        record = timer.end("api_rag_chat")
        logger.info(
            "API_STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
            record.name,
            record.started_at,
            record.ended_at,
            record.duration_ms,
        )
        summary = timer.summary()
        logger.info("API_RAG_TIMING_SUMMARY | %s", summary)

        return ChatResponse(
            question=payload.question,
            answer=result["answer"],
            references=result["references"],
            contexts=result["contexts"] if payload.include_contexts else None,
            timing=summary,
        )
    except Exception as exc:
        logger.exception("RAG API 调用失败")
        if timer.has_active("api_rag_chat"):
            record = timer.end("api_rag_chat")
            logger.info(
                "API_STAGE_END | %s | started_at=%s | ended_at=%s | duration_ms=%s",
                record.name,
                record.started_at,
                record.ended_at,
                record.duration_ms,
            )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@asynccontextmanager
async def _api_lifespan(_: FastAPI):
    logger.info("FastAPI 启动，预加载 EasyRAG 实例")
    try:
        _get_rag()
        logger.info("EasyRAG 预加载完成")
    except Exception as exc:
        logger.warning("EasyRAG 预加载失败，将在首次请求时重试: %s", exc)
    yield
    logger.info("FastAPI 关闭，释放 EasyRAG 实例")
    _reset_rag_cache()


def create_api() -> FastAPI:
    settings = get_settings()
    prefix = settings.api_path_prefix
    description = "暴露带有 RAG 能力的问答 API，供外部系统调用。"
    if prefix:
        description += f" 当前路径前缀：{prefix}"

    app = FastAPI(
        title="Easy RAG API",
        version="1.0.0",
        description=description,
        lifespan=_api_lifespan,
    )
    if prefix:
        app.include_router(router, prefix=prefix)
    else:
        app.include_router(router)
    return app


api = create_api()


def main() -> None:
    import uvicorn

    settings = get_settings()
    service_url = build_api_public_url(settings, "/health").replace("/health", "")
    logger.info("FastAPI 服务启动: %s", service_url)
    uvicorn.run(
        "easy_rag.api.server:api",
        host=settings.api_bind_host,
        port=settings.api_bind_port,
        reload=False,
    )


if __name__ == "__main__":
    main()

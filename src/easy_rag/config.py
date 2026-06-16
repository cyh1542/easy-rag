from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_ENV_VALUES: dict[str, str] = {
    "OPENAI_API_KEY": "your_api_key_here",
    "OPENAI_BASE_URL": "https://api.openai.com/v1",
    "EMBEDDING_API_KEY": "",
    "EMBEDDING_BASE_URL": "https://api.openai.com/v1",
    "CHAT_MODEL": "gpt-4o-mini",
    "EMBEDDING_PROVIDER": "remote",
    "EMBEDDING_MODEL": "text-embedding-3-small",
    "LOCAL_EMBEDDING_MODEL": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "LOCAL_EMBEDDING_DEVICE": "cpu",
    "HF_EMBEDDING_REPO_ID": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "HF_EMBEDDING_CACHE_DIR": "models/huggingface",
    "HF_ENDPOINT": "",
    "COLLECTION_NAME": "easy-rag",
    "KNOWLEDGE_DIR": "data/knowledge",
    "CHROMA_DIR": "storage/chroma",
    "CHUNK_SIZE": "800",
    "CHUNK_OVERLAP": "120",
    "CHUNK_STRATEGY": "fixed",
    "SEMANTIC_CHUNK_THRESHOLD": "0.75",
    "TOP_K": "4",
    "RETRIEVAL_METHOD": "vector",
    "RERANK_CANDIDATE_K": "12",
    "RRF_K": "60",
    "MYSQL_ENABLED": "false",
    "MYSQL_HOST": "127.0.0.1",
    "MYSQL_PORT": "3306",
    "MYSQL_USER": "root",
    "MYSQL_PASSWORD": "your_mysql_password",
    "MYSQL_DATABASE": "your_database",
    "MYSQL_CHARSET": "utf8mb4",
    "MYSQL_TABLES": "",
    "MYSQL_QUERY": "",
    "MYSQL_LIMIT_PER_TABLE": "500",
    "API_PUBLIC_BASE_URL": "",
    "API_PATH_PREFIX": "",
    "API_BIND_HOST": "0.0.0.0",
    "API_BIND_PORT": "8000",
}

ENV_ORDER = [
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
]


def _stringify(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _int_value(values: Mapping[str, str], name: str, default: int) -> int:
    value = values.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数，当前值为: {value}") from exc


def _bool_value(values: Mapping[str, str], name: str, default: bool) -> bool:
    value = values.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是布尔值，当前值为: {value}")


def _list_value(values: Mapping[str, str], name: str) -> list[str]:
    value = values.get(name, "").strip()
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_value(values: Mapping[str, str], name: str, default: float) -> float:
    value = values.get(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字，当前值为: {value}") from exc


def _path_value(values: Mapping[str, str], name: str, default: str) -> Path:
    raw_value = values.get(name, default).strip() or default
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_api_path_prefix(raw: str) -> str:
    text = raw.strip()
    if not text:
        return ""
    if not text.startswith("/"):
        text = f"/{text}"
    return text.rstrip("/")


def normalize_api_public_base_url(raw: str) -> str:
    return raw.strip().rstrip("/")


def normalize_hf_endpoint(raw: str) -> str:
    text = raw.strip().rstrip("/")
    if not text:
        return ""
    if not text.startswith(("http://", "https://")):
        text = f"https://{text}"
    return text.rstrip("/")


def apply_hf_hub_settings(settings: Settings) -> None:
    endpoint = settings.hf_endpoint
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    else:
        os.environ.pop("HF_ENDPOINT", None)


def build_api_endpoint_path(path_prefix: str, route_path: str) -> str:
    route = route_path if route_path.startswith("/") else f"/{route_path}"
    if path_prefix:
        return f"{path_prefix}{route}"
    return route


def build_api_service_base_url(settings: Settings) -> str:
    if settings.api_public_base_url:
        return settings.api_public_base_url
    host = settings.api_bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{settings.api_bind_port}"


def build_api_public_url(settings: Settings, route_path: str) -> str:
    return f"{build_api_service_base_url(settings)}{build_api_endpoint_path(settings.api_path_prefix, route_path)}"


@dataclass(slots=True)
class Settings:
    api_key: str
    base_url: str
    embedding_api_key: str
    embedding_base_url: str
    chat_model: str
    embedding_provider: str
    embedding_model: str
    local_embedding_model: str
    local_embedding_device: str
    hf_embedding_repo_id: str
    hf_embedding_cache_dir: Path
    hf_endpoint: str
    collection_name: str
    knowledge_dir: Path
    chroma_dir: Path
    chunk_size: int
    chunk_overlap: int
    chunk_strategy: str
    semantic_chunk_threshold: float
    top_k: int
    retrieval_method: str
    rerank_candidate_k: int
    rrf_k: int
    mysql_enabled: bool
    mysql_host: str
    mysql_port: int
    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_charset: str
    mysql_tables: list[str]
    mysql_query: str
    mysql_limit_per_table: int
    api_public_base_url: str
    api_path_prefix: str
    api_bind_host: str
    api_bind_port: int


def save_env_values(values: Mapping[str, object]) -> None:
    serialized = DEFAULT_ENV_VALUES.copy()
    for key in ENV_ORDER:
        if key in values:
            serialized[key] = _stringify(values[key])

    lines = [f"{key}={serialized[key]}" for key in ENV_ORDER]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def migrate_env_file() -> bool:
    """将 .env 中缺失的配置项补全为默认值，保持已有配置不变。"""
    if not ENV_FILE.exists():
        return False

    disk_values = dotenv_values(ENV_FILE)
    missing_keys = [key for key in ENV_ORDER if key not in disk_values]
    if not missing_keys:
        return False

    merged = DEFAULT_ENV_VALUES.copy()
    for key, value in disk_values.items():
        if value is not None:
            merged[key] = value

    lines = [f"{key}={merged[key]}" for key in ENV_ORDER]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def read_env_values() -> dict[str, str]:
    migrate_env_file()
    values = DEFAULT_ENV_VALUES.copy()

    if ENV_FILE.exists():
        for key, value in dotenv_values(ENV_FILE).items():
            if value is not None:
                values[key] = value

    for key in ENV_ORDER:
        env_value = os.getenv(key)
        if env_value is not None:
            values[key] = env_value

    return values


def save_env_values(values: Mapping[str, object]) -> None:
    serialized = DEFAULT_ENV_VALUES.copy()
    for key in ENV_ORDER:
        if key in values:
            serialized[key] = _stringify(values[key])

    lines = [f"{key}={serialized[key]}" for key in ENV_ORDER]
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def settings_from_env_values(values: Mapping[str, str]) -> Settings:
    return Settings(
        api_key=values.get("OPENAI_API_KEY", "").strip(),
        base_url=values.get("OPENAI_BASE_URL", DEFAULT_ENV_VALUES["OPENAI_BASE_URL"]).strip().rstrip("/"),
        embedding_api_key=values.get("EMBEDDING_API_KEY", DEFAULT_ENV_VALUES["EMBEDDING_API_KEY"]).strip(),
        embedding_base_url=values.get(
            "EMBEDDING_BASE_URL",
            DEFAULT_ENV_VALUES["EMBEDDING_BASE_URL"],
        ).strip().rstrip("/"),
        chat_model=values.get("CHAT_MODEL", DEFAULT_ENV_VALUES["CHAT_MODEL"]).strip(),
        embedding_provider=values.get("EMBEDDING_PROVIDER", DEFAULT_ENV_VALUES["EMBEDDING_PROVIDER"]).strip().lower(),
        embedding_model=values.get("EMBEDDING_MODEL", DEFAULT_ENV_VALUES["EMBEDDING_MODEL"]).strip(),
        local_embedding_model=values.get(
            "LOCAL_EMBEDDING_MODEL",
            DEFAULT_ENV_VALUES["LOCAL_EMBEDDING_MODEL"],
        ).strip(),
        local_embedding_device=values.get(
            "LOCAL_EMBEDDING_DEVICE",
            DEFAULT_ENV_VALUES["LOCAL_EMBEDDING_DEVICE"],
        ).strip(),
        hf_embedding_repo_id=values.get(
            "HF_EMBEDDING_REPO_ID",
            DEFAULT_ENV_VALUES["HF_EMBEDDING_REPO_ID"],
        ).strip(),
        hf_embedding_cache_dir=_path_value(
            values,
            "HF_EMBEDDING_CACHE_DIR",
            DEFAULT_ENV_VALUES["HF_EMBEDDING_CACHE_DIR"],
        ),
        hf_endpoint=normalize_hf_endpoint(
            values.get("HF_ENDPOINT", DEFAULT_ENV_VALUES["HF_ENDPOINT"])
        ),
        collection_name=values.get("COLLECTION_NAME", DEFAULT_ENV_VALUES["COLLECTION_NAME"]).strip(),
        knowledge_dir=_path_value(values, "KNOWLEDGE_DIR", DEFAULT_ENV_VALUES["KNOWLEDGE_DIR"]),
        chroma_dir=_path_value(values, "CHROMA_DIR", DEFAULT_ENV_VALUES["CHROMA_DIR"]),
        chunk_size=_int_value(values, "CHUNK_SIZE", int(DEFAULT_ENV_VALUES["CHUNK_SIZE"])),
        chunk_overlap=_int_value(values, "CHUNK_OVERLAP", int(DEFAULT_ENV_VALUES["CHUNK_OVERLAP"])),
        chunk_strategy=values.get("CHUNK_STRATEGY", DEFAULT_ENV_VALUES["CHUNK_STRATEGY"]).strip().lower(),
        semantic_chunk_threshold=_float_value(
            values,
            "SEMANTIC_CHUNK_THRESHOLD",
            float(DEFAULT_ENV_VALUES["SEMANTIC_CHUNK_THRESHOLD"]),
        ),
        top_k=_int_value(values, "TOP_K", int(DEFAULT_ENV_VALUES["TOP_K"])),
        retrieval_method=values.get("RETRIEVAL_METHOD", DEFAULT_ENV_VALUES["RETRIEVAL_METHOD"]).strip().lower(),
        rerank_candidate_k=_int_value(
            values,
            "RERANK_CANDIDATE_K",
            int(DEFAULT_ENV_VALUES["RERANK_CANDIDATE_K"]),
        ),
        rrf_k=_int_value(
            values,
            "RRF_K",
            int(DEFAULT_ENV_VALUES["RRF_K"]),
        ),
        mysql_enabled=_bool_value(values, "MYSQL_ENABLED", False),
        mysql_host=values.get("MYSQL_HOST", DEFAULT_ENV_VALUES["MYSQL_HOST"]).strip(),
        mysql_port=_int_value(values, "MYSQL_PORT", int(DEFAULT_ENV_VALUES["MYSQL_PORT"])),
        mysql_user=values.get("MYSQL_USER", DEFAULT_ENV_VALUES["MYSQL_USER"]).strip(),
        mysql_password=values.get("MYSQL_PASSWORD", DEFAULT_ENV_VALUES["MYSQL_PASSWORD"]).strip(),
        mysql_database=values.get("MYSQL_DATABASE", DEFAULT_ENV_VALUES["MYSQL_DATABASE"]).strip(),
        mysql_charset=values.get("MYSQL_CHARSET", DEFAULT_ENV_VALUES["MYSQL_CHARSET"]).strip(),
        mysql_tables=_list_value(values, "MYSQL_TABLES"),
        mysql_query=values.get("MYSQL_QUERY", DEFAULT_ENV_VALUES["MYSQL_QUERY"]).strip(),
        mysql_limit_per_table=_int_value(
            values,
            "MYSQL_LIMIT_PER_TABLE",
            int(DEFAULT_ENV_VALUES["MYSQL_LIMIT_PER_TABLE"]),
        ),
        api_public_base_url=normalize_api_public_base_url(
            values.get("API_PUBLIC_BASE_URL", DEFAULT_ENV_VALUES["API_PUBLIC_BASE_URL"])
        ),
        api_path_prefix=normalize_api_path_prefix(
            values.get("API_PATH_PREFIX", DEFAULT_ENV_VALUES["API_PATH_PREFIX"])
        ),
        api_bind_host=values.get("API_BIND_HOST", DEFAULT_ENV_VALUES["API_BIND_HOST"]).strip(),
        api_bind_port=_int_value(values, "API_BIND_PORT", int(DEFAULT_ENV_VALUES["API_BIND_PORT"])),
    )


def get_settings() -> Settings:
    return settings_from_env_values(read_env_values())

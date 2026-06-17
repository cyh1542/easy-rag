from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from easy_rag.config import get_settings


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def extract_client_api_key(
    x_api_key: str | None,
    authorization: str | None,
) -> str:
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return ""


def require_rag_api_key(
    x_api_key: Annotated[str | None, Security(_api_key_header)] = None,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    expected = get_settings().rag_api_key
    if not expected:
        return

    provided = extract_client_api_key(x_api_key, authorization)
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="无效或缺失 RAG API Key。")


def build_rag_api_auth_headers(api_key: str | None = None) -> dict[str, str]:
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"X-API-Key": key}

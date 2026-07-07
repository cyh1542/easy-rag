"""向后兼容：uvicorn api_server:api"""
from easy_rag.api.server import api

__all__ = ["api"]

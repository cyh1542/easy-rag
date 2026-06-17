from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from easy_rag.config import PROJECT_ROOT, Settings


KNOWLEDGE_BASES_FILE = PROJECT_ROOT / "storage" / "knowledge_bases.json"
MAX_KB_PROFILES = 4
CHUNK_STRATEGIES = ("fixed", "recursive", "semantic", "structure")
_COLLECTION_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")


@dataclass(slots=True)
class ChunkConfig:
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    semantic_chunk_threshold: float

    def validate(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE。")
        if self.chunk_strategy not in CHUNK_STRATEGIES:
            raise ValueError("CHUNK_STRATEGY 只支持 fixed、recursive、semantic 或 structure。")
        if not 0 < self.semantic_chunk_threshold <= 1:
            raise ValueError("SEMANTIC_CHUNK_THRESHOLD 必须在 0 到 1 之间。")


@dataclass(slots=True)
class KnowledgeBaseProfile:
    id: str
    name: str
    collection_name: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    semantic_chunk_threshold: float
    enabled: bool = True

    def to_chunk_config(self) -> ChunkConfig:
        config = ChunkConfig(
            chunk_strategy=self.chunk_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            semantic_chunk_threshold=self.semantic_chunk_threshold,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.id.strip():
            raise ValueError("知识库 id 不能为空。")
        if not self.name.strip():
            raise ValueError(f"知识库 {self.id} 的名称不能为空。")
        if not _COLLECTION_NAME_PATTERN.match(self.collection_name):
            raise ValueError(f"知识库 {self.id} 的 collection_name 格式无效。")
        self.to_chunk_config()


@dataclass(slots=True)
class KnowledgeBaseRegistry:
    enabled: bool = False
    bases: list[KnowledgeBaseProfile] = field(default_factory=list)

    def active_profiles(self) -> list[KnowledgeBaseProfile]:
        return [profile for profile in self.bases if profile.enabled]

    def validate(self) -> None:
        if not self.enabled:
            return
        profiles = self.active_profiles()
        if not profiles:
            raise ValueError("启用多知识库后，至少需要启用一个知识库配置。")
        seen_ids: set[str] = set()
        seen_collections: set[str] = set()
        for profile in self.bases:
            profile.validate()
            if profile.id in seen_ids:
                raise ValueError(f"知识库 id 重复: {profile.id}")
            seen_ids.add(profile.id)
            if profile.collection_name in seen_collections:
                raise ValueError(f"collection_name 重复: {profile.collection_name}")
            seen_collections.add(profile.collection_name)


def default_collection_name(base_name: str, strategy: str) -> str:
    suffix = strategy.replace("_", "-")
    candidate = f"{base_name}-{suffix}"
    return candidate[:63]


def chunk_config_from_settings(settings: Settings) -> ChunkConfig:
    return ChunkConfig(
        chunk_strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        semantic_chunk_threshold=settings.semantic_chunk_threshold,
    )


def default_profiles_from_settings(settings: Settings) -> list[KnowledgeBaseProfile]:
    base_name = settings.collection_name or "easy-rag"
    profiles: list[KnowledgeBaseProfile] = []
    for index, strategy in enumerate(CHUNK_STRATEGIES, start=1):
        profiles.append(
            KnowledgeBaseProfile(
                id=f"kb-{strategy}",
                name=f"{strategy} 切分库",
                collection_name=default_collection_name(base_name, strategy),
                chunk_strategy=strategy,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                semantic_chunk_threshold=settings.semantic_chunk_threshold,
                enabled=index == 1,
            )
        )
    return profiles


def registry_from_settings(settings: Settings) -> KnowledgeBaseRegistry:
    return KnowledgeBaseRegistry(
        enabled=False,
        bases=default_profiles_from_settings(settings),
    )


def _profile_from_mapping(raw: Mapping[str, Any]) -> KnowledgeBaseProfile:
    return KnowledgeBaseProfile(
        id=str(raw.get("id", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        collection_name=str(raw.get("collection_name", "")).strip(),
        chunk_strategy=str(raw.get("chunk_strategy", "fixed")).strip().lower(),
        chunk_size=int(raw.get("chunk_size", 800)),
        chunk_overlap=int(raw.get("chunk_overlap", 120)),
        semantic_chunk_threshold=float(raw.get("semantic_chunk_threshold", 0.75)),
        enabled=bool(raw.get("enabled", True)),
    )


def load_knowledge_base_registry(settings: Settings | None = None) -> KnowledgeBaseRegistry:
    if not KNOWLEDGE_BASES_FILE.exists():
        if settings is None:
            return KnowledgeBaseRegistry()
        return registry_from_settings(settings)

    payload = json.loads(KNOWLEDGE_BASES_FILE.read_text(encoding="utf-8"))
    bases = [_profile_from_mapping(item) for item in payload.get("bases", []) if isinstance(item, dict)]
    registry = KnowledgeBaseRegistry(enabled=bool(payload.get("enabled", False)), bases=bases)
    if settings is not None and not bases:
        registry.bases = default_profiles_from_settings(settings)
    return registry


def save_knowledge_base_registry(registry: KnowledgeBaseRegistry) -> None:
    registry.validate()
    KNOWLEDGE_BASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": registry.enabled,
        "bases": [asdict(profile) for profile in registry.bases],
    }
    KNOWLEDGE_BASES_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _form_bool(form: Any, key: str, *, default: bool = False) -> bool:
    getlist = getattr(form, "getlist", None)
    if callable(getlist):
        values = getlist(key)
    else:
        raw = form.get(key)
        values = [raw] if raw is not None else []
    if not values:
        return default
    return any(str(item).strip().lower() in {"true", "on", "1"} for item in values)


def parse_registry_from_form(form: Any, settings: Settings) -> KnowledgeBaseRegistry:
    enabled = _form_bool(form, "MULTI_KB_ENABLED")
    profiles: list[KnowledgeBaseProfile] = []
    for index in range(1, MAX_KB_PROFILES + 1):
        prefix = f"KB{index}_"
        strategy = str(form.get(f"{prefix}CHUNK_STRATEGY", "")).strip().lower()
        if not strategy:
            continue
        profile_id = str(form.get(f"{prefix}ID", "")).strip() or f"kb-{index}"
        base_name = settings.collection_name or "easy-rag"
        default_collection = default_collection_name(base_name, strategy)
        collection_name = str(form.get(f"{prefix}COLLECTION_NAME", "")).strip() or default_collection
        profiles.append(
            KnowledgeBaseProfile(
                id=profile_id,
                name=str(form.get(f"{prefix}NAME", "")).strip() or f"{strategy} 切分库",
                collection_name=collection_name,
                chunk_strategy=strategy,
                chunk_size=int(str(form.get(f"{prefix}CHUNK_SIZE", settings.chunk_size)).strip() or settings.chunk_size),
                chunk_overlap=int(
                    str(form.get(f"{prefix}CHUNK_OVERLAP", settings.chunk_overlap)).strip() or settings.chunk_overlap
                ),
                semantic_chunk_threshold=float(
                    str(form.get(f"{prefix}SEMANTIC_CHUNK_THRESHOLD", settings.semantic_chunk_threshold)).strip()
                    or settings.semantic_chunk_threshold
                ),
                enabled=_form_bool(form, f"{prefix}ENABLED"),
            )
        )

    if not profiles:
        profiles = default_profiles_from_settings(settings)
    return KnowledgeBaseRegistry(enabled=enabled, bases=profiles)


def resolve_retrieval_profiles(settings: Settings) -> list[KnowledgeBaseProfile]:
    registry = load_knowledge_base_registry(settings)
    if not registry.enabled:
        return []
    return registry.active_profiles()


def merge_retrieval_results(groups: list[list[dict[str, Any]]], top_k: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for group in groups:
        for item in group:
            key = (str(item.get("source", "")), str(item.get("content", "")))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

    def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
        distance = item.get("distance")
        if distance is not None:
            try:
                return (float(distance), -float(item.get("score", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
        score = item.get("score")
        if score is not None:
            try:
                return (1.0 - float(score), -float(score))
            except (TypeError, ValueError):
                pass
        return (999.0, 0.0)

    merged.sort(key=_sort_key)
    if top_k <= 0:
        return merged
    return merged[:top_k]

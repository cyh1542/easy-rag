from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from typing import Any


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class StageRecord:
    name: str
    started_at: str
    ended_at: str
    duration_ms: float


class StageTimer:
    def __init__(self) -> None:
        self._timeline: list[StageRecord] = []
        self._active: dict[str, tuple[str, float]] = {}

    def start(self, stage_name: str) -> str:
        started_at = now_text()
        self._active[stage_name] = (started_at, perf_counter())
        return started_at

    def end(self, stage_name: str) -> StageRecord:
        started_at, started_perf = self._active.pop(stage_name)
        ended_at = now_text()
        duration_ms = round((perf_counter() - started_perf) * 1000, 2)
        record = StageRecord(
            name=stage_name,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
        )
        self._timeline.append(record)
        return record

    def has_active(self, stage_name: str) -> bool:
        return stage_name in self._active

    def end_if_active(self, stage_name: str) -> StageRecord | None:
        if not self.has_active(stage_name):
            return None
        return self.end(stage_name)

    def summary(self) -> dict[str, Any]:
        total_ms = round(sum(item.duration_ms for item in self._timeline), 2)
        chain_text = " -> ".join(
            f"{item.name}({item.duration_ms}ms)"
            for item in self._timeline
        )
        return {
            "total_ms": total_ms,
            "chain_text": chain_text,
            "stages": [
                {
                    "name": item.name,
                    "started_at": item.started_at,
                    "ended_at": item.ended_at,
                    "duration_ms": item.duration_ms,
                }
                for item in self._timeline
            ],
        }

from __future__ import annotations

from easy_rag.timing_utils import StageTimer


def test_stage_timer_records_duration() -> None:
    timer = StageTimer()
    timer.start("step_a")
    timer.end("step_a")

    summary = timer.summary()
    assert summary["total_ms"] >= 0
    assert len(summary["stages"]) == 1
    assert summary["stages"][0]["name"] == "step_a"
    assert "step_a" in summary["chain_text"]


def test_stage_timer_multiple_stages_chain() -> None:
    timer = StageTimer()
    timer.start("retrieve")
    timer.end("retrieve")
    timer.start("llm")
    timer.end("llm")

    summary = timer.summary()
    assert len(summary["stages"]) == 2
    assert summary["chain_text"] == "retrieve({}ms) -> llm({}ms)".format(
        summary["stages"][0]["duration_ms"],
        summary["stages"][1]["duration_ms"],
    )


def test_stage_timer_end_if_active() -> None:
    timer = StageTimer()
    assert timer.end_if_active("missing") is None

    timer.start("active")
    record = timer.end_if_active("active")
    assert record is not None
    assert record.name == "active"


def test_stage_timer_has_active() -> None:
    timer = StageTimer()
    timer.start("running")
    assert timer.has_active("running") is True
    timer.end("running")
    assert timer.has_active("running") is False

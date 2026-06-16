#!/usr/bin/env python3
"""命令行入口：调用 FastAPI 评测 tests/eval 下的 JSON / JSONL 测试集。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from easy_rag.eval_runner import (
    DEFAULT_CASES_FILE,
    aggregate_results,
    check_health,
    evaluate_case,
    load_dataset,
    resolve_base_url,
    resolve_eval_cases_path,
    run_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _print_summary(summary: dict, base_url: str, cases_path: Path) -> None:
    print("=" * 72)
    print("Easy RAG API 评测结果")
    print("=" * 72)
    print(f"API 地址      : {base_url}")
    print(f"测试集        : {cases_path}")
    print(f"总用例数      : {summary['total_cases']}")
    print(f"成功调用      : {summary['successful_cases']}")
    print(f"调用失败      : {summary['failed_cases']}")
    print("-" * 72)
    print(f"召回率 (Retrieval Recall) : {summary['retrieval_recall']:.2%}")
    print(f"准确率 (Answer Accuracy)  : {summary['answer_accuracy']:.2%}")
    print(f"平均响应耗时              : {summary['avg_latency_ms']:.0f} ms")
    print("-" * 72)
    print("按分类统计:")
    for category, stats in sorted(summary["by_category"].items()):
        print(
            f"  [{category}] "
            f"召回 {stats['retrieval_recall']:.2%} ({stats['retrieval_hit']}/{stats['total']}) | "
            f"准确 {stats['answer_accuracy']:.2%} ({stats['answer_hit']}/{stats['total']})"
        )
    print("=" * 72)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 Easy RAG FastAPI 问答效果")
    parser.add_argument("--base-url", default=None, help="API 根地址，默认读取 .env")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_FILE, help="测试集路径")
    parser.add_argument("--output", type=Path, default=None, help="详细结果输出 JSON 路径")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒数")
    parser.add_argument("--min-keyword-ratio", type=float, default=0.5, help="关键词命中比例阈值")
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 条，0 表示全部")
    parser.add_argument("--skip-health-check", action="store_true", help="跳过 /health 检查")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases_path = args.cases.resolve()
    if not cases_path.exists():
        print(f"测试集不存在: {cases_path}", file=sys.stderr)
        return 1

    dataset = load_dataset(cases_path)
    cases = list(dataset.get("cases", []))
    if args.limit > 0:
        cases = cases[: args.limit]

    base_url = resolve_base_url(args.base_url)
    print(f"使用 API: {base_url}")
    print(f"加载测试用例: {len(cases)} 条")

    if not args.skip_health_check:
        try:
            health = check_health(base_url, timeout=min(args.timeout, 10.0))
            print(f"健康检查通过: chat_model={health.get('chat_model')}")
        except Exception as exc:
            print(f"健康检查失败: {exc}", file=sys.stderr)
            print("请先启动 API: easy-rag-api", file=sys.stderr)
            return 1

    results = []
    for index, case in enumerate(cases, start=1):
        case_id = case.get("id", f"#{index}")
        print(f"[{index}/{len(cases)}] 评测 {case_id} ...", flush=True)
        result = evaluate_case(
            base_url,
            case,
            timeout=args.timeout,
            min_keyword_ratio=args.min_keyword_ratio,
        )
        results.append(result)
        if result.error:
            print(f"  失败: {result.error}")
        else:
            print(
                f"  召回={'Y' if result.retrieval_hit else 'N'} "
                f"准确={'Y' if result.answer_hit else 'N'} "
                f"耗时={result.latency_ms}ms"
            )

    summary = aggregate_results(results)
    _print_summary(summary, base_url, cases_path)

    if args.output:
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "cases_path": str(cases_path),
            "meta": dataset.get("meta", {}),
            "summary": summary,
            "results": [result.__dict__ for result in results],
        }
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"详细报告已保存: {output_path}")

    return 0 if summary["failed_cases"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from easy_rag.api.auth import build_rag_api_auth_headers
from easy_rag.config import PROJECT_ROOT, get_settings, read_env_values


DEFAULT_EVAL_DIR = PROJECT_ROOT / "tests" / "eval"
DEFAULT_CASES_FILE = DEFAULT_EVAL_DIR / "homework_sop_test_cases.json"


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    expected_keywords: list[str]
    retrieval_hit: bool
    retrieval_matched_keywords: list[str]
    retrieval_keyword_recall: float
    answer_hit: bool
    answer_matched_keywords: list[str]
    answer_keyword_accuracy: float
    references: list[str] = field(default_factory=list)
    answer: str = ""
    context_count: int = 0
    error: str = ""
    latency_ms: int = 0


def list_eval_datasets(directory: Path | None = None) -> list[dict[str, str]]:
    root = directory or DEFAULT_EVAL_DIR
    if not root.exists():
        return []

    options: list[dict[str, str]] = []
    for pattern in ("*.json", "*.jsonl"):
        for file_path in sorted(root.glob(pattern)):
            try:
                rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_path = str(file_path)
            options.append(
                {
                    "path": rel_path,
                    "name": file_path.name,
                    "abs_path": str(file_path.resolve()),
                }
            )
    return options


def resolve_eval_cases_path(raw_path: str) -> Path:
    text = raw_path.strip()
    if not text:
        text = DEFAULT_CASES_FILE.relative_to(PROJECT_ROOT).as_posix()

    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    else:
        path = path.resolve()

    project_root = PROJECT_ROOT.resolve()
    if project_root not in path.parents and path != project_root:
        raise ValueError("测试集路径必须在项目目录内。")

    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError("测试集仅支持 .json 或 .jsonl 文件。")
    if not path.exists():
        raise FileNotFoundError(f"测试集不存在: {path}")
    return path


def load_dataset(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
        return {"meta": {"format": "jsonl", "case_count": len(cases)}, "cases": cases}
    return json.loads(text)


def resolve_base_url(explicit: str | None = None) -> str:
    if explicit and explicit.strip():
        return explicit.strip().rstrip("/")
    settings = get_settings()
    host = settings.api_bind_host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    prefix = settings.api_path_prefix.rstrip("/")
    return f"http://{host}:{settings.api_bind_port}{prefix}".rstrip("/")


def default_api_base_url() -> str:
    """评测应连接本机实际监听的 API 地址，而非 API_PUBLIC_BASE_URL 对外展示地址。"""
    return resolve_base_url(None)


def _build_http_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _format_http_error(exc: HTTPError, url: str) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()[:500]
    hints: list[str] = []
    if exc.code in {502, 503, 504}:
        hints.extend(
            [
                "请确认 easy-rag-api 已启动（默认 http://127.0.0.1:8000）",
                "FastAPI 地址应填本机监听地址，不要填 API_PUBLIC_BASE_URL 对外展示域名",
                "若开启系统 HTTP 代理，请为本机地址配置 NO_PROXY 或使用 127.0.0.1",
                "批量评测时可将条数调小，避免 API 或模型接口超时",
            ]
        )
    message = f"HTTP {exc.code} 请求失败: {url}"
    if body:
        message += f" | 响应: {body}"
    if hints:
        message += " | 建议: " + "；".join(hints)
    return message


def _http_json_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    retry_delay: float = 1.0,
    api_key: str | None = None,
) -> dict[str, Any]:
    opener = _build_http_opener()
    data = None
    headers = {"Accept": "application/json"}
    headers.update(build_rag_api_auth_headers(api_key or read_env_values().get("RAG_API_KEY", "")))
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    last_error = ""
    for attempt in range(retries):
        try:
            request = Request(url, data=data, headers=headers, method=method)
            with opener.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            last_error = _format_http_error(exc, url)
            if exc.code in {502, 503, 504} and attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(last_error) from exc
        except URLError as exc:
            last_error = f"无法连接 {url}：{exc.reason}。请先启动 easy-rag-api。"
            if attempt < retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            raise RuntimeError(last_error) from exc

    raise RuntimeError(last_error or f"请求失败: {url}")


def check_health(base_url: str, timeout: float = 10.0) -> dict[str, Any]:
    return _http_json_request(f"{base_url}/health", timeout=timeout, retries=2)


def post_chat(base_url: str, question: str, timeout: float) -> dict[str, Any]:
    return _http_json_request(
        f"{base_url}/api/v1/rag/chat",
        method="POST",
        payload={"question": question, "include_contexts": True},
        timeout=timeout,
        retries=3,
    )


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword.lower() in lowered]


def collect_context_text(contexts: list[dict[str, Any]] | None) -> str:
    if not contexts:
        return ""
    parts: list[str] = []
    for item in contexts:
        parts.extend(
            [
                str(item.get("content", "")),
                str(item.get("source", "")),
                str(item.get("path", "")),
            ]
        )
    return "\n".join(parts)


def score_keywords(matched: list[str], total: int, min_ratio: float) -> tuple[bool, float]:
    if total <= 0:
        return True, 1.0
    ratio = len(matched) / total
    required = max(1, int(total * min_ratio + 0.999999))
    return len(matched) >= required, ratio


def evaluate_case(
    base_url: str,
    case: dict[str, Any],
    *,
    timeout: float,
    min_keyword_ratio: float,
) -> CaseResult:
    question = str(case["question"])
    keywords = [str(item) for item in case.get("expected_keywords", [])]
    started = time.perf_counter()
    try:
        response = post_chat(base_url, question, timeout)
        latency_ms = int((time.perf_counter() - started) * 1000)
        contexts = response.get("contexts") or []
        context_text = collect_context_text(contexts)
        answer = str(response.get("answer", ""))

        retrieval_matched = keyword_hits(context_text, keywords)
        answer_matched = keyword_hits(answer, keywords)
        retrieval_hit, retrieval_ratio = score_keywords(retrieval_matched, len(keywords), min_keyword_ratio)
        answer_hit, answer_ratio = score_keywords(answer_matched, len(keywords), min_keyword_ratio)

        return CaseResult(
            id=str(case.get("id", "")),
            category=str(case.get("category", "")),
            question=question,
            expected_keywords=keywords,
            retrieval_hit=retrieval_hit,
            retrieval_matched_keywords=retrieval_matched,
            retrieval_keyword_recall=retrieval_ratio,
            answer_hit=answer_hit,
            answer_matched_keywords=answer_matched,
            answer_keyword_accuracy=answer_ratio,
            references=[str(item) for item in response.get("references", [])],
            answer=answer,
            context_count=len(contexts),
            latency_ms=latency_ms,
        )
    except HTTPError as exc:
        return CaseResult(
            id=str(case.get("id", "")),
            category=str(case.get("category", "")),
            question=question,
            expected_keywords=keywords,
            retrieval_hit=False,
            retrieval_matched_keywords=[],
            retrieval_keyword_recall=0.0,
            answer_hit=False,
            answer_matched_keywords=[],
            answer_keyword_accuracy=0.0,
            error=_format_http_error(exc, f"{base_url}/api/v1/rag/chat"),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except RuntimeError as exc:
        return CaseResult(
            id=str(case.get("id", "")),
            category=str(case.get("category", "")),
            question=question,
            expected_keywords=keywords,
            retrieval_hit=False,
            retrieval_matched_keywords=[],
            retrieval_keyword_recall=0.0,
            answer_hit=False,
            answer_matched_keywords=[],
            answer_keyword_accuracy=0.0,
            error=str(exc),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except URLError as exc:
        return CaseResult(
            id=str(case.get("id", "")),
            category=str(case.get("category", "")),
            question=question,
            expected_keywords=keywords,
            retrieval_hit=False,
            retrieval_matched_keywords=[],
            retrieval_keyword_recall=0.0,
            answer_hit=False,
            answer_matched_keywords=[],
            answer_keyword_accuracy=0.0,
            error=f"连接失败: {exc.reason}。请先启动 easy-rag-api，地址建议 http://127.0.0.1:8000",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        return CaseResult(
            id=str(case.get("id", "")),
            category=str(case.get("category", "")),
            question=question,
            expected_keywords=keywords,
            retrieval_hit=False,
            retrieval_matched_keywords=[],
            retrieval_keyword_recall=0.0,
            answer_hit=False,
            answer_matched_keywords=[],
            answer_keyword_accuracy=0.0,
            error=str(exc),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


def aggregate_results(results: list[CaseResult]) -> dict[str, Any]:
    valid = [item for item in results if not item.error]
    failed = [item for item in results if item.error]
    retrieval_hits = sum(1 for item in valid if item.retrieval_hit)
    answer_hits = sum(1 for item in valid if item.answer_hit)
    avg_retrieval = sum(item.retrieval_keyword_recall for item in valid) / len(valid) if valid else 0.0
    avg_accuracy = sum(item.answer_keyword_accuracy for item in valid) / len(valid) if valid else 0.0
    avg_latency = sum(item.latency_ms for item in valid) / len(valid) if valid else 0.0

    by_category: dict[str, dict[str, Any]] = {}
    for item in valid:
        bucket = by_category.setdefault(
            item.category,
            {"total": 0, "retrieval_hit": 0, "answer_hit": 0},
        )
        bucket["total"] += 1
        bucket["retrieval_hit"] += int(item.retrieval_hit)
        bucket["answer_hit"] += int(item.answer_hit)

    for bucket in by_category.values():
        bucket["retrieval_recall"] = bucket["retrieval_hit"] / bucket["total"] if bucket["total"] else 0.0
        bucket["answer_accuracy"] = bucket["answer_hit"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "total_cases": len(results),
        "successful_cases": len(valid),
        "failed_cases": len(failed),
        "retrieval_recall": retrieval_hits / len(valid) if valid else 0.0,
        "answer_accuracy": answer_hits / len(valid) if valid else 0.0,
        "avg_retrieval_keyword_recall": avg_retrieval,
        "avg_answer_keyword_accuracy": avg_accuracy,
        "avg_latency_ms": avg_latency,
        "by_category": by_category,
    }


def run_evaluation(
    cases_path: Path | str,
    *,
    base_url: str | None = None,
    limit: int = 0,
    timeout: float = 120.0,
    min_keyword_ratio: float = 0.5,
    skip_health_check: bool = False,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    path = cases_path if isinstance(cases_path, Path) else resolve_eval_cases_path(str(cases_path))
    dataset = load_dataset(path)
    cases = list(dataset.get("cases", []))
    if limit > 0:
        cases = cases[:limit]

    api_base = resolve_base_url(base_url)
    if not skip_health_check:
        check_health(api_base, timeout=min(timeout, 10.0))

    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        if progress_callback is not None:
            progress_callback(index, len(cases), case)
        results.append(
            evaluate_case(
                api_base,
                case,
                timeout=timeout,
                min_keyword_ratio=min_keyword_ratio,
            )
        )
        if index < len(cases):
            time.sleep(0.1)

    summary = aggregate_results(results)
    return {
        "base_url": api_base,
        "cases_path": str(path),
        "cases_path_display": path.relative_to(PROJECT_ROOT).as_posix()
        if PROJECT_ROOT.resolve() in path.parents
        else str(path),
        "meta": dataset.get("meta", {}),
        "summary": summary,
        "results": [asdict(item) for item in results],
    }


def normalize_eval_api_base_url(raw: str) -> str:
    text = raw.strip()
    if not text:
        return default_api_base_url()
    return text.rstrip("/")

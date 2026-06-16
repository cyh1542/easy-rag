from __future__ import annotations

import argparse

from easy_rag.config import get_settings
from easy_rag.rag_engine import EasyRAG


def ask_once(rag: EasyRAG, question: str) -> None:
    result = rag.answer(question)
    print("\n回答:")
    print(result["answer"])
    if result["references"]:
        print("\n参考来源:")
        for source in result["references"]:
            print(f"- {source}")


def interactive_chat(rag: EasyRAG) -> None:
    print("RAG 对话已启动，输入 exit / quit 结束。")
    while True:
        question = input("\n问题> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("已退出。")
            break

        try:
            ask_once(rag, question)
        except Exception as exc:
            print(f"\n发生错误: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="和本地 RAG 知识库进行问答。")
    parser.add_argument(
        "-q",
        "--question",
        help="直接提一个问题并返回答案，不进入交互模式。",
    )
    args = parser.parse_args()

    settings = get_settings()
    rag = EasyRAG(settings)

    if args.question:
        ask_once(rag, args.question)
        return

    interactive_chat(rag)


if __name__ == "__main__":
    main()

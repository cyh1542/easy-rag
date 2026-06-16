from __future__ import annotations

import argparse

from easy_rag.config import get_settings
from easy_rag.rag_engine import EasyRAG


def main() -> None:
    parser = argparse.ArgumentParser(description="构建本地 RAG 知识库索引。")
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加到现有索引，而不是先清空再重建。",
    )
    args = parser.parse_args()

    settings = get_settings()
    rag = EasyRAG(settings)
    summary = rag.build_index(reset=not args.append)

    print("索引构建完成。")
    print(f"本地文件文档数: {summary['file_documents']}")
    print(f"MySQL 文档数: {summary['mysql_documents']}")
    print(f"总文档数: {summary['documents']}")
    print(f"切分后的片段数量: {summary['chunks']}")
    print(f"向量库目录: {settings.chroma_dir}")


if __name__ == "__main__":
    main()

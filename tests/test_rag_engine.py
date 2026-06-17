from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from easy_rag.config import settings_from_env_values
from easy_rag.rag_engine import EasyRAG, SUPPORTED_EXTENSIONS
from tests.conftest import make_env_values, mock_embed_texts


class TestEasyRAGValidation:
    def test_chunk_overlap_must_be_less_than_chunk_size(self, settings: Any) -> None:
        settings.chunk_overlap = settings.chunk_size
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            EasyRAG(settings)

    def test_invalid_embedding_provider(self, settings: Any) -> None:
        settings.embedding_provider = "invalid"
        with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
            EasyRAG(settings)

    def test_remote_embedding_requires_model(self, settings: Any) -> None:
        settings.embedding_model = ""
        with pytest.raises(ValueError, match="EMBEDDING_MODEL"):
            EasyRAG(settings)

    def test_invalid_retrieval_method(self, settings: Any) -> None:
        settings.retrieval_method = "hybrid"
        with pytest.raises(ValueError, match="RETRIEVAL_METHOD"):
            EasyRAG(settings)

    def test_mysql_enabled_requires_host_and_tables(
        self,
        tmp_dirs: dict[str, Path],
        base_env_values: dict[str, str],
    ) -> None:
        values = make_env_values(
            tmp_dirs,
            base_env_values,
            MYSQL_ENABLED="true",
            MYSQL_HOST="",
            MYSQL_TABLES="",
            MYSQL_QUERY="",
        )
        settings = settings_from_env_values(values)
        with pytest.raises(ValueError, match="MYSQL_HOST"):
            EasyRAG(settings)


class TestApiKeyValidation:
    def test_openai_key_error_is_separate(self, rag: EasyRAG) -> None:
        rag.settings.api_key = "your_api_key_here"
        with pytest.raises(ValueError, match="OPENAI_API_KEY") as exc_info:
            rag._validate_openai_api_key()
        assert "EMBEDDING_API_KEY" not in str(exc_info.value)

    def test_embedding_key_error_when_explicit_key_invalid(self, rag: EasyRAG) -> None:
        rag.settings.embedding_api_key = "your_api_key_here"
        with pytest.raises(ValueError, match="EMBEDDING_API_KEY") as exc_info:
            rag._validate_embedding_api_key()
        assert "OPENAI_API_KEY" not in str(exc_info.value)

    def test_embedding_key_error_when_fallback_openai_missing(self, rag: EasyRAG) -> None:
        rag.settings.embedding_api_key = ""
        rag.settings.api_key = "your_api_key_here"
        with pytest.raises(ValueError, match="EMBEDDING_API_KEY") as exc_info:
            rag._validate_embedding_api_key()
        assert "OPENAI_API_KEY" not in str(exc_info.value)

    def test_get_embedding_client_raises_embedding_key_only(self, rag: EasyRAG) -> None:
        rag.settings.embedding_api_key = "your_api_key_here"
        with pytest.raises(ValueError, match="EMBEDDING_API_KEY") as exc_info:
            rag._get_embedding_client()
        assert "OPENAI_API_KEY" not in str(exc_info.value)


class TestTextSplitting:
    def test_custom_chat_system_prompt(self, rag: EasyRAG) -> None:
        rag.settings.chat_thinking_prompt = "请先列出思考步骤，再给出结论。"
        assert rag._resolve_chat_system_prompt() == "请先列出思考步骤，再给出结论。"

    def test_default_chat_system_prompt_when_empty(self, rag: EasyRAG) -> None:
        rag.settings.chat_thinking_prompt = ""
        assert "知识库问答助手" in rag._resolve_chat_system_prompt()
    def test_split_text_merges_short_paragraphs(self, rag: EasyRAG) -> None:
        rag.settings.chunk_strategy = "fixed"
        rag.settings.chunk_size = 200
        rag.settings.chunk_overlap = 20
        text = "第一段。\n\n第二段。\n\n第三段。"
        chunks = rag.split_text(text)
        assert chunks
        assert all(len(chunk) <= rag.settings.chunk_size for chunk in chunks)

    def test_split_text_splits_long_paragraph(self, rag: EasyRAG) -> None:
        rag.settings.chunk_strategy = "fixed"
        rag.settings.chunk_size = 50
        rag.settings.chunk_overlap = 10
        text = "A" * 120
        chunks = rag.split_text(text)
        assert len(chunks) >= 2

    def test_split_text_empty_input(self, rag: EasyRAG) -> None:
        assert rag.split_text("") == []
        assert rag.split_text("   \n\n   ") == []

    def test_recursive_split_uses_sentence_boundary(self, rag: EasyRAG) -> None:
        rag.settings.chunk_strategy = "recursive"
        rag.settings.chunk_size = 40
        rag.settings.chunk_overlap = 0
        text = "第一句内容比较长。第二句内容也比较长。第三句继续补充。"
        chunks = rag.split_text(text)
        assert chunks
        assert all(len(chunk) <= rag.settings.chunk_size for chunk in chunks)

    def test_structure_split_by_markdown_heading(self, rag: EasyRAG) -> None:
        rag.settings.chunk_strategy = "structure"
        rag.settings.chunk_size = 60
        rag.settings.chunk_overlap = 0
        text = "# 标题一\n\n段落 A 内容。\n\n## 标题二\n\n段落 B 内容。"
        chunks = rag.split_text(text)
        assert len(chunks) >= 2
        assert any("标题一" in chunk for chunk in chunks)
        assert any("标题二" in chunk for chunk in chunks)

    def test_semantic_split_with_mock_embedding(self, rag: EasyRAG, monkeypatch: pytest.MonkeyPatch) -> None:
        rag.settings.chunk_strategy = "semantic"
        rag.settings.chunk_size = 200
        rag.settings.semantic_chunk_threshold = 0.5

        def mock_embed(texts: list[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for index, text in enumerate(texts):
                vector = [0.0, 0.0, 1.0]
                if "RAG" in text:
                    vector = [1.0, 0.0, 0.0]
                elif "数据库" in text:
                    vector = [0.0, 1.0, 0.0]
                vectors.append(vector)
            return vectors

        monkeypatch.setattr(rag, "_embed_texts", mock_embed)
        text = "RAG 是检索增强生成。向量数据库用于存储 embedding。RAG 可以结合外部知识。"
        chunks = rag.split_text(text)
        assert chunks
        assert all(len(chunk) <= rag.settings.chunk_size for chunk in chunks)

    def test_invalid_chunk_strategy_raises(self, settings: Any) -> None:
        settings.chunk_strategy = "invalid"
        with pytest.raises(ValueError, match="CHUNK_STRATEGY"):
            EasyRAG(settings)


class TestKeywordAndSimilarity:
    def test_tokenize_text_handles_cjk(self, rag: EasyRAG) -> None:
        tokens = rag._tokenize_text("RAG检索测试")
        assert "rag" in tokens
        assert any("\u4e00" <= char <= "\u9fff" for char in tokens)

    def test_keyword_score_matches_terms(self, rag: EasyRAG) -> None:
        score = rag._keyword_score("RAG 向量", "RAG 是检索增强生成，使用向量数据库。")
        assert score > 0

    def test_keyword_score_zero_when_no_match(self, rag: EasyRAG) -> None:
        score = rag._keyword_score("不存在的关键词", "普通文本内容")
        assert score == 0.0

    def test_cosine_similarity_identical_vectors(self, rag: EasyRAG) -> None:
        assert rag._cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_cosine_similarity_orthogonal_vectors(self, rag: EasyRAG) -> None:
        assert rag._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


class TestFileLoading:
    def test_list_supported_files(self, rag: EasyRAG, sample_markdown: Path) -> None:
        files = rag.list_supported_files()
        assert sample_markdown in files

    def test_list_supported_files_filters_unsupported(self, rag: EasyRAG, tmp_dirs: dict[str, Path]) -> None:
        unsupported = tmp_dirs["knowledge_dir"] / "image.png"
        unsupported.write_bytes(b"fake")
        files = rag.list_supported_files()
        assert unsupported not in files

    def test_load_file_documents(self, rag: EasyRAG, sample_markdown: Path) -> None:
        documents = rag.load_file_documents()
        assert len(documents) == 1
        assert documents[0]["source"] == "sample.md"
        assert documents[0]["file_name"] == "sample.md"
        assert documents[0]["relative_path"] == "sample.md"
        assert documents[0]["document_type"] == "file"
        assert "RAG" in documents[0]["content"]

    def test_load_file_documents_includes_subdirectory_path(
        self,
        rag: EasyRAG,
        tmp_dirs: dict[str, Path],
    ) -> None:
        nested_dir = tmp_dirs["knowledge_dir"] / "sop"
        nested_dir.mkdir()
        nested_file = nested_dir / "guide.md"
        nested_file.write_text("子目录文档", encoding="utf-8")

        documents = rag.load_file_documents()
        matched = [item for item in documents if item["file_name"] == "guide.md"]
        assert len(matched) == 1
        assert matched[0]["relative_path"] == "sop/guide.md"
        assert matched[0]["parent_dir"] == "sop"

    def test_read_text_file_multiple_encodings(self, rag: EasyRAG, tmp_dirs: dict[str, Path]) -> None:
        file_path = tmp_dirs["knowledge_dir"] / "gbk.txt"
        file_path.write_bytes("中文内容".encode("gbk"))
        content = rag._read_file(file_path)
        assert "中文" in content

    def test_supported_extensions_cover_readers(self) -> None:
        expected = {".txt", ".md", ".pdf", ".csv", ".json", ".html", ".xml", ".docx"}
        assert expected.issubset(SUPPORTED_EXTENSIONS)


class TestMySQLHelpers:
    def test_safe_mysql_identifier_rejects_invalid(self, rag: EasyRAG) -> None:
        with pytest.raises(ValueError, match="不安全"):
            rag._safe_mysql_identifier("users;drop")

    def test_safe_mysql_identifier_accepts_valid(self, rag: EasyRAG) -> None:
        assert rag._safe_mysql_identifier("user_orders") == "user_orders"

    def test_load_mysql_documents_disabled(self, rag: EasyRAG) -> None:
        assert rag.load_mysql_documents() == []


class TestIndexAndRetrieve:
    def test_build_index_from_files(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        summary = rag.build_index(reset=True)
        assert summary["file_documents"] == 1
        assert summary["chunks"] > 0

    def test_build_index_does_not_spam_embedding_mode_logs(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        caplog.set_level(logging.INFO)
        rag.build_index(reset=True)
        embedding_mode_logs = [
            record.message
            for record in caplog.records
            if "当前 embedding 调用方式" in record.message
        ]
        assert embedding_mode_logs == []

    def test_build_index_raises_when_no_documents(self, rag: EasyRAG) -> None:
        with pytest.raises(FileNotFoundError, match="没有读取到可用数据"):
            rag.build_index(reset=True)

    def test_vector_retrieve_returns_results(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        rag.build_index(reset=True)
        rag.settings.retrieval_method = "vector"
        rag.settings.top_k = 2
        items = rag.retrieve("RAG 是什么")
        assert items
        assert items[0]["source"] == "sample.md"
        assert items[0]["file_name"] == "sample.md"
        assert items[0]["relative_path"] == "sample.md"
        assert "content" in items[0]

    def test_keyword_retrieve_returns_matches(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        rag.build_index(reset=True)
        rag.settings.retrieval_method = "keyword"
        items = rag.retrieve("向量数据库")
        assert items
        assert all(item["source"] == "sample.md" for item in items)

    def test_rerank_retrieve(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        rag.build_index(reset=True)
        rag.settings.retrieval_method = "rerank"
        rag.settings.rerank_candidate_k = 4
        rag.settings.top_k = 2
        items = rag.retrieve("embedding")
        assert len(items) <= 2

    def test_rrf_retrieve(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        rag.build_index(reset=True)
        rag.settings.retrieval_method = "rrf"
        rag.settings.rerank_candidate_k = 4
        rag.settings.top_k = 2
        items = rag.retrieve("RAG 检索")
        assert len(items) <= 2

    def test_answer_requires_chat_model(self, rag: EasyRAG) -> None:
        rag.settings.chat_model = ""
        with pytest.raises(ValueError, match="CHAT_MODEL"):
            rag.answer("测试问题")

    def test_answer_raises_when_index_empty(self, rag: EasyRAG) -> None:
        with pytest.raises(RuntimeError, match="向量库为空"):
            rag.answer("测试问题")

    def test_answer_with_mocked_llm(
        self,
        rag: EasyRAG,
        sample_markdown: Path,
        mock_embedding: None,
    ) -> None:
        rag.build_index(reset=True)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="这是 RAG 的简要说明。"))]

        with patch.object(rag, "_get_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_response
            result = rag.answer("RAG 是什么")

        assert result["answer"] == "这是 RAG 的简要说明。"
        assert "sample.md" in result["references"]
        assert result["contexts"]


class TestDataframeHelpers:
    def test_dataframe_to_text_with_title(self, rag: EasyRAG) -> None:
        import pandas as pd

        dataframe = pd.DataFrame({"name": ["Alice"], "age": [30]})
        text = rag._dataframe_to_text(dataframe, title="用户表")
        assert "用户表" in text
        assert "Alice" in text

    def test_rows_to_text_formats_records(self, rag: EasyRAG) -> None:
        rows = [{"id": 1, "name": "测试"}]
        text = rag._rows_to_text(rows, title="MySQL 表: users")
        assert "MySQL 表: users" in text
        assert "name: 测试" in text

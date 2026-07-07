# Changelog

本文件记录项目的 notable 变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-06-13

### Added

- `src/easy_rag` 标准 Python 包布局
- Flask Web 配置界面与 RAG 管理
- FastAPI 问答 API（`/health`、`/api/v1/rag/chat`）
- 命令行工具：`easy-rag-ingest`、`easy-rag-chat`
- 多格式文档与 MySQL 数据源支持
- 四种检索策略：keyword、vector、rerank、rrf
- 接口文档 `docs/API.md` 与 Postman 集合
- pytest 测试套件与 GitHub Actions CI
- Docker / docker-compose 部署支持
- MIT 许可证

[1.0.0]: https://github.com/your-org/easy-rag/releases/tag/v1.0.0

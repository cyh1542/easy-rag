# Easy RAG

[![CI](https://github.com/your-org/easy-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/easy-rag/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

可独立部署的 **RAG（检索增强生成）服务**：支持多种文档与 MySQL 数据源、可配置切分与检索策略，提供 **Flask 管理界面**、**FastAPI 开放 API** 与 **命令行工具**。

## 特性

- OpenAI 兼容接口调用聊天模型与远程 Embedding
- 本地 / Hugging Face Embedding 模式
- 多格式文档：txt、md、pdf、csv、xlsx、docx、pptx、html、json、xml 等
- MySQL 按表或自定义 SQL 导入
- 四种检索策略：关键词、向量、重排序、RRF 融合
- Chroma 本地向量库
- Web 配置页：保存 `.env`、建索引、预览问答
- REST API + Postman 示例

## 快速开始

### 环境要求

- Python 3.10+
- 推荐 Python 3.12 / 3.13（请使用 `chromadb>=1.5.9`，旧版在 3.13 下可能安装失败）

### 安装

```bash
git clone https://github.com/your-org/easy-rag.git
cd easy-rag

python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
```

### 配置

```bash
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY、CHAT_MODEL 等
```

### 建索引

```bash
easy-rag-ingest
# 或：python ingest.py
```

### 启动服务

**Web 管理界面**（默认 `http://127.0.0.1:5000`）

```bash
easy-rag-web
# 或：python app.py
# 或：scripts/start_web.ps1
```

**FastAPI API**（默认 `http://127.0.0.1:8000`）

```bash
easy-rag-api
# 或：uvicorn api_server:api --host 0.0.0.0 --port 8000
# 或：scripts/start_api.ps1
```

**命令行问答**

```bash
easy-rag-chat
easy-rag-chat --question "RAG 的核心流程是什么？"
```

## 项目结构

```text
easy-rag/
├── .github/workflows/ci.yml   # GitHub Actions 测试
├── docs/
│   ├── API.md                 # 接口文档
│   └── postman/               # Postman 集合与环境
├── src/easy_rag/              # 主程序包
│   ├── config.py              # 配置读写
│   ├── rag_engine.py          # RAG 核心
│   ├── web/app.py             # Flask 界面
│   ├── api/server.py          # FastAPI 服务
│   ├── cli/                   # ingest / chat
│   └── templates/             # Web 模板
├── data/knowledge/            # 默认知识库
├── storage/chroma/            # 向量库（运行时生成）
├── tests/                     # pytest 测试
├── scripts/                   # 启动脚本
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

## API 文档

- Markdown：[docs/API.md](docs/API.md)
- Swagger UI：启动 API 后访问 `http://127.0.0.1:8000/docs`
- Postman：导入 `docs/postman/` 下两个 JSON 文件

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/rag/chat` | RAG 问答 |

## Docker

```bash
cp .env.example .env
docker compose up -d --build
```

- Web：`http://localhost:5000`
- API：`http://localhost:8000`

## 开发与测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### RAG 效果评测（召回率 / 准确率）

针对真实 API 与模型的批量问答评测，见 **[docs/EVAL.md](docs/EVAL.md)**。

```bash
# 启动 API 并完成建索引后
python scripts/eval_rag_api.py
```

测试用例默认位于 `tests/eval/`，支持 `.json` 与 `.jsonl` 格式。

## 配置说明

完整配置项见 [.env.example](.env.example)。核心字段：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | 聊天模型 API Key |
| `CHAT_MODEL` | 聊天模型名称 |
| `EMBEDDING_PROVIDER` | `remote` / `local` / `huggingface` |
| `KNOWLEDGE_DIR` | 知识库目录 |
| `RETRIEVAL_METHOD` | `keyword` / `vector` / `rerank` / `rrf` |
| `MYSQL_ENABLED` | 是否启用 MySQL 数据源 |

## 许可证

[MIT License](LICENSE)

## 参与贡献

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。

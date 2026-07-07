# Easy RAG API 接口文档

版本：`1.0.0`  
协议：HTTP / JSON  
默认 Base URL：`http://127.0.0.1:8000`

---

## 1. 概述

Easy RAG API 基于 FastAPI 对外暴露 RAG（检索增强生成）问答能力。外部系统通过 HTTP 调用接口提问，服务会：

1. 从本地 Chroma 向量库检索相关片段；
2. 将检索结果与用户问题一并发送给大语言模型；
3. 返回生成的答案、参考来源及可选的命中上下文。

**配置来源**：API 服务读取项目根目录 `.env` 中的配置（与 Flask 管理页面共用），包括模型、Embedding、知识库路径、检索策略等。调用 API 前需先完成配置保存与索引构建。

**交互式文档**（服务启动后访问）：

| 地址 | 说明 |
|------|------|
| `http://127.0.0.1:8000/docs` | Swagger UI |
| `http://127.0.0.1:8000/redoc` | ReDoc |
| `http://127.0.0.1:8000/openapi.json` | OpenAPI JSON Schema |

---

## 2. 启动服务

```bash
uvicorn api_server:api --host 0.0.0.0 --port 8000
```

| 参数 | 说明 |
|------|------|
| `--host 0.0.0.0` | 允许局域网访问 |
| `--port 8000` | 监听端口，可按需修改 |

---

## 3. 通用约定

### 3.1 请求头

| Header | 值 | 必填 | 说明 |
|--------|-----|------|------|
| `Content-Type` | `application/json` | POST 接口必填 | 请求体为 JSON |
| `Accept` | `application/json` | 可选 | 建议统一使用 |
| `X-API-Key` | 配置的 `RAG_API_KEY` | 启用鉴权时必填 | 推荐方式 |
| `Authorization` | `Bearer <RAG_API_KEY>` | 启用鉴权时可选 | 与 `X-API-Key` 二选一 |

在 `.env` 中设置 `RAG_API_KEY` 后，问答接口 `/api/v1/rag/chat` 将要求携带有效密钥；留空则不校验（适合本地开发）。`/health` 始终无需鉴权。

### 3.2 响应格式

**成功**：HTTP 状态码 `200`，响应体为 JSON。

**失败**：HTTP 状态码 `4xx` / `5xx`，响应体格式：

```json
{
  "detail": "错误描述信息"
}
```

| 状态码 | 场景 |
|--------|------|
| `401` | 已启用 `RAG_API_KEY` 但请求未携带或密钥错误 |
| `422` | 请求体 JSON 格式错误或字段校验失败 |
| `500` | 服务内部错误（配置无效、向量库为空、模型调用失败等） |

### 3.3 前置条件

调用问答接口前请确认：

- [ ] `.env` 中已配置有效的 `OPENAI_API_KEY` 与 `CHAT_MODEL`
- [ ] 若设置了 `RAG_API_KEY`，调用方需在请求头携带对应密钥
- [ ] 已通过 Flask 页面或 `python ingest.py` 完成建索引
- [ ] `storage/chroma` 下存在对应集合的向量数据

---

## 4. 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查与配置摘要 |
| `POST` | `/api/v1/rag/chat` | RAG 问答 |

---

## 5. 健康检查

### `GET /health`

检查服务是否可用，并返回当前生效的核心配置（只读，不触发问答）。

#### 请求

无请求体，无必填参数。

#### 请求示例

```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```

#### 响应 `200 OK`

```json
{
  "status": "ok",
  "auth_enabled": false,
  "chat_model": "gpt-4o-mini",
  "embedding_provider": "remote",
  "retrieval_method": "vector",
  "knowledge_dir": "D:\\git_program\\easy rag\\data\\knowledge"
}
```

#### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 服务状态，正常时为 `ok` |
| `auth_enabled` | boolean | 是否已启用 `RAG_API_KEY` 鉴权 |
| `chat_model` | string | 当前聊天模型名称 |
| `embedding_provider` | string | Embedding 模式：`remote` / `local` / `huggingface` |
| `retrieval_method` | string | 检索策略：`keyword` / `vector` / `rerank` / `rrf` |
| `knowledge_dir` | string | 知识库目录绝对路径 |

#### 错误响应 `500`

```json
{
  "detail": "环境变量 CHUNK_SIZE 必须是整数，当前值为: abc"
}
```

---

## 6. RAG 问答

### `POST /api/v1/rag/chat`

根据用户问题检索知识库并生成回答。

#### 请求体

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `question` | string | 是 | — | 用户问题，不能为空 |
| `include_contexts` | boolean | 否 | `true` | 是否在响应中返回命中的上下文片段 |

#### 请求示例（含上下文）

```http
POST /api/v1/rag/chat HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
Accept: application/json
X-API-Key: your-rag-api-key

{
  "question": "RAG 的核心流程是什么？",
  "include_contexts": true
}
```

#### 请求示例（仅返回答案）

```json
{
  "question": "这份知识库主要介绍了什么？",
  "include_contexts": false
}
```

#### 响应 `200 OK`

```json
{
  "question": "RAG 的核心流程是什么？",
  "answer": "RAG 的核心流程包括：将文档切分、向量化并存入向量库，用户提问时检索相关片段，再将检索结果与问题一并发送给大语言模型生成答案。",
  "references": [
    "sample_rag_intro.md"
  ],
  "contexts": [
    {
      "content": "一个典型的 RAG 系统通常包含下面几个步骤：\n1. 先把本地知识库文档切分成多个片段。\n2. 再使用 embedding 模型把这些片段转换成向量。",
      "source": "sample_rag_intro.md",
      "path": "D:\\git_program\\easy rag\\data\\knowledge\\sample_rag_intro.md",
      "distance": 0.234521
    }
  ],
  "timing": {
    "total_ms": 1523.45,
    "chain_text": "api_rag_chat(1520.1ms) -> vector_retrieve(45.2ms) -> build_prompt_context(1.5ms) -> llm_completion(1473.8ms)",
    "stages": [
      {
        "name": "vector_retrieve",
        "started_at": "2026-06-13T17:30:01",
        "ended_at": "2026-06-13T17:30:01",
        "duration_ms": 45.2
      },
      {
        "name": "build_prompt_context",
        "started_at": "2026-06-13T17:30:01",
        "ended_at": "2026-06-13T17:30:01",
        "duration_ms": 1.5
      },
      {
        "name": "llm_completion",
        "started_at": "2026-06-13T17:30:01",
        "ended_at": "2026-06-13T17:30:03",
        "duration_ms": 1473.8
      },
      {
        "name": "api_rag_chat",
        "started_at": "2026-06-13T17:30:01",
        "ended_at": "2026-06-13T17:30:03",
        "duration_ms": 1520.1
      }
    ]
  }
}
```

#### 响应字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 回显的用户问题 |
| `answer` | string | 模型生成的回答 |
| `references` | array[string] | 参考来源列表（去重后的文档名或数据源标识） |
| `contexts` | array[object] \| null | 命中上下文；`include_contexts=false` 时为 `null` |
| `contexts[].content` | string | 检索到的文本片段 |
| `contexts[].source` | string | 来源标识（文件名或 `mysql_table:表名` 等） |
| `contexts[].path` | string | 文件路径或 MySQL 数据源 URI |
| `contexts[].distance` | number | 相似度距离（越小通常表示越相关） |
| `timing` | object | 各阶段耗时统计 |
| `timing.total_ms` | number | 总耗时（毫秒） |
| `timing.chain_text` | string | 阶段耗时链式描述 |
| `timing.stages` | array[object] | 各阶段明细 |

#### 错误响应示例

**请求校验失败 `422`**

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "question"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

**向量库为空 `500`**

```json
{
  "detail": "当前向量库为空，请先执行 ingest.py 建立索引。"
}
```

**配置无效 `500`**

```json
{
  "detail": "请先在配置中填写有效的 OPENAI_API_KEY。"
}
```

---

## 7. 调用流程建议

```text
1. GET  /health          → 确认服务与配置正常
2. POST /api/v1/rag/chat → 发送问题并获取回答
3. 解析 references / contexts → 追溯答案依据
```

---

## 8. Postman 使用说明

项目提供可直接导入的 Postman 资源：

| 文件 | 说明 |
|------|------|
| `docs/postman/Easy_RAG_API.postman_collection.json` | 接口集合（含示例请求） |
| `docs/postman/Easy_RAG_API.postman_environment.json` | 环境变量（Base URL） |

### 导入步骤

1. 打开 Postman → **Import**
2. 选择上述两个 JSON 文件导入
3. 右上角环境切换为 **Easy RAG API - Local**
4. 确认变量 `baseUrl` 为 `http://127.0.0.1:8000`
5. 先执行 **Health Check**，再执行 **RAG Chat** 相关请求

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `baseUrl` | `http://127.0.0.1:8000` | API 服务地址 |

---

## 9. 附录：检索策略说明

检索行为由 `.env` 中的 `RETRIEVAL_METHOD` 控制，API 请求本身不携带检索参数。

| 值 | 说明 |
|----|------|
| `keyword` | 关键词匹配 |
| `vector` | 向量相似度检索（默认） |
| `rerank` | 向量候选 + 重排序 |
| `rrf` | 关键词与向量 RRF 融合 |

相关参数：`TOP_K`、`RERANK_CANDIDATE_K`、`RRF_K`，均在 `.env` / Flask 页面配置。

---

## 10. 变更记录

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0.0 | 2026-06-13 | 初始版本：`/health`、`/api/v1/rag/chat` |

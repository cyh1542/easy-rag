# RAG 效果评测指南

本文说明如何准备测试用例、放置路径，以及如何通过 FastAPI 接口评估召回率与准确率。

---

## 1. 两类「文件」不要混淆

| 用途 | 文件形式 | 放置位置 | 作用 |
|------|----------|----------|------|
| **知识库文档**（被 RAG 检索） | `.md` `.txt` `.pdf` `.docx` `.pptx` 等 | `.env` 中 `KNOWLEDGE_DIR` 指向的目录 | 建索引后供检索 |
| **评测测试用例**（问题 + 期望关键词） | `.json` 或 `.jsonl` | 推荐 `tests/eval/` | 调用 API 批量提问并打分 |

评测前必须：**先把知识库文档建索引**，再运行评测脚本。

---

## 2. 测试用例支持的文件格式

### 2.1 JSON（推荐）

单个文件包含元信息与用例数组，适合正式评测集。

**默认示例**：

| 文件 | 条数 | 说明 |
|------|------|------|
| [`homework_sop_test_cases.json`](../tests/eval/homework_sop_test_cases.json) | 80 | 精简版 |
| [`homework_sop_test_cases_300.json`](../tests/eval/homework_sop_test_cases_300.json) | 300 | 完整版（推荐批量评测） |

重新生成 300 条用例：

```bash
python scripts/generate_homework_sop_cases.py
```

```json
{
  "meta": {
    "source_document": "D:/path/to/knowledge/作业批改sop.docx",
    "title": "作业批改 SOP 评测集",
    "version": "1.0"
  },
  "cases": [
    {
      "id": "Q001",
      "category": "适用场景",
      "question": "作业批改 SOP 适用于什么场景？",
      "expected_keywords": ["单元作业", "批改", "登记", "链接返还"],
      "reference": "一、适用场景"
    }
  ]
}
```

### 2.2 JSONL（可选）

每行一条用例，便于逐行追加或脚本生成。

**示例**：[`tests/eval/example_test_cases.jsonl`](../tests/eval/example_test_cases.jsonl)

```jsonl
{"id":"E001","category":"示例","question":"谁负责发布小作业链接？","expected_keywords":["学管"]}
{"id":"E002","category":"示例","question":"谁负责发布大作业链接？","expected_keywords":["助教"]}
```

使用方式与 JSON 相同，通过 `--cases` 指定路径即可。

### 2.3 暂不支持直接导入的格式

`.csv`、`.xlsx`、`.md` 等需先转换为上述 JSON / JSONL。可用 Excel 导出 CSV 后写小脚本转成 JSON。

---

## 3. 测试用例字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 建议 | 用例编号，如 `Q001` |
| `question` | **是** | 发给 `/api/v1/rag/chat` 的问题 |
| `expected_keywords` | **是** | 期望出现在检索结果或回答中的关键词列表 |
| `category` | 否 | 分类标签，用于分组统计 |
| `reference` | 否 | 对应知识库文档章节，便于人工核对 |

`meta` 块仅用于描述，不参与打分。

---

## 4. 文件应放在哪里

```
easy-rag/
├── tests/eval/                    ← 推荐：正式/示例评测集
│   ├── homework_sop_test_cases.json
│   └── example_test_cases.jsonl
├── reports/                       ← 评测报告输出（自动创建）
│   └── sop_eval_report.json
└── scripts/
    └── eval_rag_api.py            ← 评测脚本
```

自定义评测集可放在任意路径，运行时用 `--cases` 指定绝对或相对路径即可。

**知识库文档**（被测内容）放在 `KNOWLEDGE_DIR`，例如：

```env
KNOWLEDGE_DIR=D:/git_program/test/data/knowledge
```

该目录下的 `作业批改sop.docx` 等文件需先完成建索引。

---

## 5. 如何运行评测

### 5.1 Web 界面（模型效果预览页）

1. 启动 Web 与 API：`easy-rag-web`、`easy-rag-api`
2. 打开 **模型效果预览** 页（`/preview`）
3. 在 **FastAPI 测试集评测** 区域选择 `tests/eval` 下的测试集文件
4. 设置评测条数（`0` = 全部），点击 **运行批量评测**
5. 页面展示：**召回率**、**准确率**、**平均响应时间** 及分类统计

### 5.2 命令行

**调用文件**：`scripts/eval_rag_api.py`（底层逻辑在 `src/easy_rag/eval_runner.py`）

```bash
# 使用默认测试集（80 条 SOP 用例）
python scripts/eval_rag_api.py

# 300 条完整测试集
python scripts/eval_rag_api.py --cases tests/eval/homework_sop_test_cases_300.json

# 指定 API 地址与报告输出
python scripts/eval_rag_api.py \
  --cases tests/eval/homework_sop_test_cases.json \
  --output reports/sop_eval_report.json
```

### 5.3 基本命令（命令行补充）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--cases` | `tests/eval/homework_sop_test_cases.json` | 测试集路径（`.json` / `.jsonl`）；300 条完整集见 `homework_sop_test_cases_300.json` |
| `--base-url` | 读取 `.env` 中 API 配置 | FastAPI 根地址 |
| `--output` | 无 | 详细 JSON 报告保存路径 |
| `--limit` | `0`（全部） | 仅运行前 N 条 |
| `--min-keyword-ratio` | `0.5` | 至少命中多少比例的关键词算通过 |
| `--timeout` | `120` | 单次请求超时（秒） |

---

## 6. 指标含义

| 指标 | 含义 |
|------|------|
| **召回率 (Retrieval Recall)** | 检索返回的 `contexts` 中，是否包含 `expected_keywords`（默认至少命中 50% 关键词） |
| **准确率 (Answer Accuracy)** | 模型 `answer` 中，是否包含 `expected_keywords`（默认至少命中 50% 关键词） |

终端会输出总体召回率、准确率，以及按 `category` 的分组统计。若指定 `--output`，会额外保存每条用例的详细命中情况、回答原文与耗时。

---

## 7. 如何编写自己的测试集

1. 阅读知识库文档，列出 50–100 个可验证的问题
2. 为每个问题提取 2–5 个**必须出现**的关键词（来自文档原文）
3. 复制 [`homework_sop_test_cases.json`](../tests/eval/homework_sop_test_cases.json) 或 [`example_test_cases.jsonl`](../tests/eval/example_test_cases.jsonl) 修改
4. 确保 `KNOWLEDGE_DIR` 包含被测文档并完成建索引
5. 运行 `python scripts/eval_rag_api.py --cases 你的文件.json`

---

## 8. 与单元测试的区别

| 类型 | 命令 | 说明 |
|------|------|------|
| **单元测试** | `pytest tests/ -v` | 不调用真实 LLM，验证代码逻辑 |
| **RAG 效果评测** | `python scripts/eval_rag_api.py` | 调用真实 API + 模型，评估检索与回答质量 |

评测指标相关的纯逻辑测试见 [`tests/eval/test_eval_metrics.py`](../tests/eval/test_eval_metrics.py)。

---

## 9. 常见问题

**Q：召回率低怎么办？**  
检查是否已建索引、`RETRIEVAL_METHOD` / `TOP_K` 是否合理，或知识库路径是否正确。

**Q：准确率低但召回率高？**  
说明检索到了相关内容，但 LLM 回答未覆盖关键词；可优化 prompt 或换更强聊天模型。

**Q：连接失败？**  
确认 `easy-rag-api` 已启动，且 `--base-url` 与 `.env` 中 `API_BIND_PORT`、`API_PATH_PREFIX` 一致。

**Q：测试用例能否用 Markdown 写？**  
目前评测脚本仅支持 JSON / JSONL；Markdown 适合作为知识库文档，不适合直接作为评测用例格式。

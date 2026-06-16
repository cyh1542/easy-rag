# 参与贡献

感谢关注 Easy RAG！

## 开发环境

```bash
git clone https://github.com/your-org/easy-rag.git
cd easy-rag
pip install -e ".[dev]"
cp .env.example .env
```

## 提交前检查

```bash
pytest tests/ -v
```

## Pull Request

1. Fork 仓库并创建特性分支
2. 保持变更聚焦，附带必要测试
3. 更新相关文档（README / docs/API.md）
4. 确保 CI 通过

## 报告问题

请在 Issues 中提供：操作系统、Python 版本、复现步骤与相关日志（`logs/app.log`）。

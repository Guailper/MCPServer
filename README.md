# Standalone RAG MCP Server

这是一个与 CookingAgent 解耦的独立 RAG MCP 服务。它提供完整的本地 RAG 流程：文档加载、文本切分、embedding、向量持久化、检索和知识库管理。

## MCP 工具

- `rag_index_path`: 索引一个文件或目录。
- `rag_search`: 在指定知识库中检索相关片段。
- `rag_list_knowledge_bases`: 查看已索引的知识库。
- `rag_delete_knowledge_base`: 删除指定知识库。
- `rag_health_check`: 查看配置和存储状态。

## 默认行为

默认使用内置的哈希 embedding 和本地 JSONL 向量库，不需要 Milvus、Redis 或 CookingAgent 代码。适合快速启动和验证完整链路。

支持的文件类型：

- `.txt`
- `.md`
- `.markdown`

## 启动

```powershell
cd D:\AppData\Code\MCPServer
python -m pip install -e .
python server.py
```

MCP 客户端配置示例：

```json
{
  "mcpServers": {
    "rag": {
      "command": "python",
      "args": ["D:\\AppData\\Code\\MCPServer\\server.py"]
    }
  }
}
```

## 常用环境变量

```powershell
$env:RAG_STORE_PATH="D:\AppData\Code\MCPServer\.rag_store\chunks.jsonl"
$env:RAG_DEFAULT_KNOWLEDGE_BASE_ID="default"
$env:RAG_EMBEDDING_PROVIDER="hash"
```

如果要使用 OpenAI-compatible embedding：

```powershell
$env:RAG_EMBEDDING_PROVIDER="openai"
$env:RAG_EMBEDDING_BASE_URL="https://api.openai.com/v1"
$env:RAG_EMBEDDING_API_KEY="your-api-key"
$env:RAG_EMBEDDING_MODEL="text-embedding-3-small"
```

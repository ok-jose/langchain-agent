# LangChain + Ollama 本地 Agent

基于 LangChain 和 Ollama 的本地 AI Agent 项目，支持工具调用、RAG 知识库检索和 Web 界面。

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.ai) 已安装并运行

## 快速开始

### 1. 安装 Ollama 和模型

```bash
# 安装 Ollama 后，拉取所需模型
ollama pull llama3.1
ollama pull nomic-embed-text  # RAG 嵌入模型
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，修改 Ollama 地址和模型名称
```

### 4. 运行

```bash
# CLI 交互式对话
python agent.py

# Streamlit Web UI
streamlit run app.py
```

## 项目结构

```
langchain-demo/
├── agent.py              # 主入口 - Agent 组装 + CLI 对话
├── app.py                # Streamlit Web UI
├── tools/                # 工具模块
│   ├── __init__.py       # 包初始化，统一导出工具
│   ├── weather.py        # 天气查询工具
│   ├── time_tool.py      # 时间查询工具
│   ├── calculator.py     # 计算器工具
│   └── search.py         # 网络搜索工具 (DuckDuckGo)
├── rag/                  # RAG 模块
│   ├── __init__.py
│   ├── loader.py         # 文档加载和分割
│   └── retriever.py      # 向量存储和检索 (Chroma)
├── knowledge/            # 知识库目录（放 .txt/.md 文件）
│   └── .gitkeep
├── chroma_db/            # 向量数据库持久化目录 (自动生成)
├── requirements.txt
├── .env
├── .env.example
└── README.md
```

## 可用工具

| 工具 | 说明 |
|------|------|
| `get_weather` | 查询城市天气（模拟数据） |
| `get_current_time` | 获取当前日期和时间 |
| `calculate` | 数学表达式计算 |
| `search_web` | DuckDuckGo 网络搜索 |
| `query_knowledge_base` | 本地知识库检索 (RAG) |

## 知识库 (RAG)

将 `.txt` 或 `.md` 文件放入 `knowledge/` 目录，Agent 启动时会自动索引。
也可以在 Web UI 的侧边栏上传文件到知识库。

## 自定义工具

在 `tools/` 目录下创建新模块，使用 `@tool` 装饰器：

```python
# tools/my_tool.py
from langchain.tools import tool

@tool
def my_tool(param: str) -> str:
    """工具描述（LLM 依赖它来决定是否调用）"""
    return result
```

然后在 `tools/__init__.py` 中导出，并在 `agent.py` 的 `tools` 列表中注册。

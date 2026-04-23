"""
基于 LangChain + Ollama 本地 Agent 主入口

功能：
- 使用 Ollama 本地模型 (llama3.1)
- 自定义工具 (天气查询、时间查询、计算器、网络搜索)
- RAG 知识库检索
- 对话记忆 (多轮对话)
- 流式输出

使用前请确保：
1. 已安装并运行 Ollama (https://ollama.ai)
2. 已拉取模型: ollama pull llama3.1
3. 安装依赖: pip install -r requirements.txt
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from rag import KnowledgeRetriever, load_documents, split_documents

# 加载环境变量（必须在导入会读取环境变量的本地模块之前）
load_dotenv()

from tools import calculate, get_current_time, get_weather, read_emails, search_web

# ---------------------------------------------------------------------------
# 1. 配置模型
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. 定义工具
# ---------------------------------------------------------------------------

# 基础工具
tools = [get_weather, get_current_time, calculate, search_web, read_emails]

# RAG 工具
retriever = KnowledgeRetriever()

# 启动时加载并索引已有知识文档
_existing_docs = load_documents()
if _existing_docs:
    _splits = split_documents(_existing_docs)
    if _splits:
        retriever.add_documents(_splits)


@tool
def query_knowledge_base(query: str) -> str:
    """从本地知识库中检索相关信息。当用户问到私有文档、笔记或上传文件内容时使用。"""
    docs = retriever.search(query)
    if not docs:
        return "知识库中没有找到相关内容。"
    results = []
    for i, doc in enumerate(docs, 1):
        snippet = doc.page_content[:300]
        source = doc.metadata.get("source", "未知来源")
        results.append(f"[{i}] {snippet}\n   来源: {source}")
    return "\n\n".join(results)


tools.append(query_knowledge_base)

# ---------------------------------------------------------------------------
# 3. 创建 Agent
# ---------------------------------------------------------------------------

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个有用的 AI 助手，可以使用各种工具来帮助用户。"
        "请用中文回答用户的问题。"
        "你可以查询天气、当前时间、进行数学计算、搜索互联网、读取邮件，以及检索本地知识库。"
    ),
    checkpointer=checkpointer,
)

# ---------------------------------------------------------------------------
# 4. 交互式对话
# ---------------------------------------------------------------------------


def chat_loop():
    """启动交互式对话循环。"""
    thread_config = {"configurable": {"thread_id": "default"}}

    print("=" * 60)
    print("  LangChain + Ollama 本地 Agent")
    print("=" * 60)
    print(f"  模型: {OLLAMA_MODEL}")
    print(f"  可用工具: {', '.join(t.name for t in tools)}")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        user_input = input("\n你: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break

        print("\nAgent: ", end="", flush=True)

        # 流式输出
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            thread_config,
            stream_mode="updates",
        ):
            # 获取最终消息
            for node_output in chunk.values():
                if "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    if hasattr(last_msg, "content"):
                        print(last_msg.content, end="", flush=True)

        print()  # 换行


if __name__ == "__main__":
    chat_loop()

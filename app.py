"""
Streamlit Web UI for the LangChain Agent.

功能：
- 聊天界面，支持流式输出
- 侧边栏：RAG 文档上传 / 索引
- 显示 Agent 的工具调用过程（可展开）
- 对话历史记录

运行: streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()

from rag import KnowledgeRetriever, load_documents, split_documents
from tools import calculate, get_current_time, get_weather, search_web, read_emails

# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


@st.cache_resource
def build_agent():
    """构建 Agent 实例并缓存，避免每次请求重建。"""
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    tools = [get_weather, get_current_time, calculate, search_web, read_emails]

    retriever = KnowledgeRetriever()

    # 启动时加载已有文档
    existing_docs = load_documents()
    if existing_docs:
        splits = split_documents(existing_docs)
        if splits:
            retriever.add_documents(splits)

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

    return agent, retriever


def get_retriever():
    """获取 retriever 实例用于索引新文档。"""
    return KnowledgeRetriever()


# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI Agent", page_icon=":robot_face:")
st.title("LangChain + Ollama Agent")

with st.sidebar:
    st.header("知识库管理")
    uploaded_files = st.file_uploader(
        "上传 .txt 或 .md 文件到知识库",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        retriever = get_retriever()
        all_docs = []
        for uf in uploaded_files:
            # 保存到 knowledge/ 目录
            knowledge_dir = Path("knowledge")
            knowledge_dir.mkdir(exist_ok=True)
            save_path = knowledge_dir / uf.name
            save_path.write_bytes(uf.getvalue())

            # 加载并切分
            loader = TextLoader(str(save_path), encoding="utf-8")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)

        if all_docs:
            retriever.add_documents(all_docs)
            st.success(f"已索引 {len(all_docs)} 个文本块")

    st.divider()
    st.caption(f"模型: {OLLAMA_MODEL}")


# ---------------------------------------------------------------------------
# 会话状态初始化
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tool_calls_expanded" not in st.session_state:
    st.session_state.tool_calls_expanded = False

# ---------------------------------------------------------------------------
# 显示历史消息
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

    # 如果消息包含工具调用信息，显示为可展开区域
    if "tool_calls" in msg and msg["tool_calls"]:
        with st.chat_message("assistant"):
            with st.expander("工具调用详情"):
                for tc in msg["tool_calls"]:
                    st.markdown(f"**工具**: `{tc['name']}`")
                    st.markdown(f"**输入**: `{tc['args']}`")
                    st.markdown(f"**输出**: {tc['output']}")
                    st.divider()

# ---------------------------------------------------------------------------
# 用户输入
# ---------------------------------------------------------------------------

if prompt := st.chat_input("输入消息..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    agent, _ = build_agent()
    thread_config = {"configurable": {"thread_id": "default"}}

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        tool_call_chunks = []
        full_response = ""

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            thread_config,
            stream_mode="updates",
        ):
            for node_name, node_output in chunk.items():
                # 捕获工具调用
                if node_name == "agent" and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_call_chunks.append({
                                    "name": tc["name"],
                                    "args": tc["args"],
                                    "output": "...",
                                })

                # 捕获工具执行结果
                if node_name == "tools" and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if hasattr(msg, "tool_call_id") and tool_call_chunks:
                            # 匹配最后一个未设置 output 的工具调用
                            for tc in reversed(tool_call_chunks):
                                if tc["output"] == "...":
                                    tc["output"] = str(msg.content)
                                    break

                # 最终回复
                if "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        full_response = last_msg.content
                        message_placeholder.markdown(full_response)

        message_placeholder.markdown(full_response)

        # 保存助手消息（含工具调用信息）
        assistant_msg = {"role": "assistant", "content": full_response}
        if tool_call_chunks:
            assistant_msg["tool_calls"] = tool_call_chunks

    st.session_state.messages.append(assistant_msg)

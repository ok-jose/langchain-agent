"""
分步教程示例 - 逐步学习 LangChain + Ollama

这个文件包含多个独立的小示例，可以单独运行来理解每个概念。
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


# ====== 示例 1: 基础对话 (无工具) ======
def example_1_basic_chat():
    """最基础的对话示例。"""
    print("\n=== 示例 1: 基础对话 ===")

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    response = llm.invoke("你好，请用一句话介绍你自己。")
    print(f"回答: {response.content}")


# ====== 示例 2: 定义和使用工具 ======
def example_2_tools():
    """演示如何定义和使用工具。"""
    print("\n=== 示例 2: 工具定义和使用 ===")

    @tool
    def get_current_time() -> str:
        """获取当前时间。"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 直接调用工具（不经过 LLM）
    print(f"直接调用工具: {get_current_time.invoke({})}")

    # 让 LLM 绑定工具
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    llm_with_tools = llm.bind_tools([get_current_time])

    # 询问时间
    response = llm_with_tools.invoke("现在几点了？")
    print(f"LLM 回复: {response.content}")
    if response.tool_calls:
        print(f"工具调用: {response.tool_calls}")


# ====== 示例 3: 创建 Agent ======
def example_3_agent():
    """创建并使用一个简单的 Agent。"""
    print("\n=== 示例 3: 创建 Agent ===")

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气。"""
        return f"{city} 的天气是晴天，25°C。"

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="你是一个有用的助手。",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}
    )
    print(f"回答: {result['messages'][-1].content}")


# ====== 示例 4: 多轮对话 (带记忆) ======
def example_4_memory():
    """演示 Agent 的多轮对话记忆。"""
    print("\n=== 示例 4: 多轮对话记忆 ===")

    checkpointer = InMemorySaver()

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt="你是一个有用的助手。",
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "conv1"}}

    # 第一轮
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫小明，记住我的名字。"}]},
        config,
    )
    print(f"第一轮回复: {result['messages'][-1].content}")

    # 第二轮 - 测试是否记住了
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config,
    )
    print(f"第二轮回复: {result['messages'][-1].content}")


# ====== 示例 5: 流式输出 ======
def example_5_streaming():
    """演示流式输出。"""
    print("\n=== 示例 5: 流式输出 ===")

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

    for chunk in llm.stream("请写一首关于春天的短诗，4句话即可。"):
        print(chunk.content, end="", flush=True)
    print()


# ====== 运行所有示例 ======
if __name__ == "__main__":
    examples = {
        "1": ("基础对话", example_1_basic_chat),
        "2": ("工具使用", example_2_tools),
        "3": ("创建 Agent", example_3_agent),
        "4": ("多轮对话", example_4_memory),
        "5": ("流式输出", example_5_streaming),
    }

    print("LangChain + Ollama 分步教程")
    print("选择要运行的示例 (1-5)，或输入 'all' 运行全部：")

    choice = input("选择: ").strip().lower()

    if choice == "all":
        for name, func in examples.values():
            func()
    elif choice in examples:
        name, func = examples[choice]
        func()
    else:
        print("无效选择，默认运行示例 1")
        examples["1"][1]()

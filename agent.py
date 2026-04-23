"""
基于 LangChain + Ollama 的本地 Agent 示例

功能：
- 使用 Ollama 本地模型 (llama3.1)
- 自定义工具 (天气查询、时间查询、计算器)
- 对话记忆 (多轮对话)
- 流式输出

使用前请确保：
1. 已安装并运行 Ollama (https://ollama.ai)
2. 已拉取模型: ollama pull llama3.1
3. 安装依赖: pip install -r requirements.txt
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# 加载环境变量
load_dotenv()

# ---------------------------------------------------------------------------
# 1. 配置模型
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. 定义工具
# ---------------------------------------------------------------------------


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气情况。返回模拟数据。"""
    weather_data = {
        "beijing": "晴，温度 22°C，微风",
        "shanghai": "多云，温度 25°C，东南风 3 级",
        "shenzhen": "小雨，温度 28°C，南风 2 级",
        "guangzhou": "雷阵雨，温度 30°C，西南风 4 级",
        "hangzhou": "晴，温度 24°C，东风 2 级",
    }
    city_lower = city.lower()
    return weather_data.get(city_lower, f"抱歉，暂无 {city}的天气数据。")


@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式。支持加减乘除和括号。

    例如: "2 + 3 * 4", "(10 - 5) / 2"
    """
    try:
        # 仅允许安全的数学字符
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含非法字符"
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def search_web(query: str) -> str:
    """搜索互联网。"""
    return f"搜索结果: {query}"

# 注册工具列表
tools = [get_weather, get_current_time, calculate, search_web]

# ---------------------------------------------------------------------------
# 3. 创建 Agent
# ---------------------------------------------------------------------------

# 内存检查点 - 支持多轮对话记忆
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一个有用的 AI 助手，可以使用各种工具来帮助用户。"
        "请用中文回答用户的问题。"
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

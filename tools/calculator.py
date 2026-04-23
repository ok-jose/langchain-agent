"""Calculator tool."""

from langchain.tools import tool


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

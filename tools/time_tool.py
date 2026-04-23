"""Current time query tool."""

from datetime import datetime

from langchain.tools import tool


@tool
def get_current_time() -> str:
    """获取当前的日期和时间。"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"

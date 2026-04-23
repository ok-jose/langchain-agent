"""Web search tool using DDGS (DuckDuckGo)."""

from langchain.tools import tool


@tool
def search_web(query: str) -> str:
    """搜索互联网，获取与查询相关的信息。"""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)

        if not results:
            return f"未找到与「{query}」相关的搜索结果。"

        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            snippets.append(f"**{title}**\n{body}")

        return "\n\n".join(snippets)
    except Exception as e:
        return f"搜索失败: {e}"

"""Agent 工具集合。"""

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from src.config.settings import settings


def _build_tavily() -> TavilySearch:
    """构造 Tavily 搜索底层实例。"""
    return TavilySearch(
        tavily_api_key=settings.tavily_api_key,
        max_results=5,
        topic="general",
        include_answer="basic",
        include_raw_content=False,
    )


@tool
def web_search(query: str) -> str:
    """联网搜索工具。用于获取当前日期、最新新闻、实时数据（股价/天气/汇率）、
    近期事件、2024 年之后发生的事情等实时性信息。

    当用户问到"今天""现在""最新""近期""目前""今年"等时效性内容时，必须调用本工具。
    输入应该是一个简洁的搜索关键词（中英文皆可）。
    """
    tavily = _build_tavily()
    result = tavily.invoke({"query": query})
    if isinstance(result, list):
        # Tavily 返回 List[dict]，拼成文本
        parts = []
        for item in result:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            parts.append(f"标题：{title}\n内容：{content}\n来源：{url}")
        return "\n\n---\n\n".join(parts) if parts else "(无搜索结果)"
    return str(result)


def build_search_tool():
    """对外暴露的工厂函数，返回带中文描述的搜索工具。"""
    return web_search

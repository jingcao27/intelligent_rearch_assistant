"""Agent 工具集合。"""

import httpx
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
        parts = []
        for item in result:
            title = item.get("title", "")
            content = item.get("content", "")
            url = item.get("url", "")
            parts.append(f"标题：{title}\n内容：{content}\n来源：{url}")
        return "\n\n---\n\n".join(parts) if parts else "(无搜索结果)"
    return str(result)


# WMO 天气代码 → 中文描述
WMO_WEATHER_CODE = {
    0: "晴",
    1: "晴间多云",
    2: "多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨（小）",
    53: "毛毛雨（中）",
    55: "毛毛雨（大）",
    56: "冻毛毛雨（小）",
    57: "冻毛毛雨（大）",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨（小）",
    67: "冻雨（大）",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "冰粒",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    85: "小阵雪",
    86: "大阵雪",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。输入城市名（中文或英文皆可，如"北京""shanghai"）。

    当用户问"X天气""X今天天气""X的天气如何""X现在多少度"等天气相关问题时调用本工具。
    返回温度、湿度、风速、天气状况等信息。
    """
    # 第 1 步：Geocoding API 城市名 → 经纬度
    geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
    geocode_params = {
        "name": city,
        "count": 1,
        "language": "zh",
        "format": "json",
    }
    try:
        geo_resp = httpx.get(geocode_url, params=geocode_params, timeout=10.0)
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
    except Exception as e:
        return f"查询城市「{city}」失败：{e}"

    results = geo_data.get("results", [])
    if not results:
        return f"未找到城市「{city}」，请确认城市名"

    location = results[0]
    lat = location["latitude"]
    lon = location["longitude"]
    city_name_zh = location.get("name", city)
    country = location.get("country", "")
    admin1 = location.get("admin1", "")  # 省/州

    # 第 2 步：Weather API 经纬度 → 当前天气
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,wind_direction_10m",
        "timezone": "Asia/Shanghai",
    }
    try:
        weather_resp = httpx.get(weather_url, params=weather_params, timeout=10.0)
        weather_resp.raise_for_status()
        weather_data = weather_resp.json()
    except Exception as e:
        return f"查询「{city_name_zh}」天气失败：{e}"

    current = weather_data.get("current", {})
    temp = current.get("temperature_2m", "?")
    humidity = current.get("relative_humidity_2m", "?")
    wind_speed = current.get("wind_speed_10m", "?")
    wind_dir = current.get("wind_direction_10m", "?")
    code = current.get("weather_code", -1)
    weather_desc = WMO_WEATHER_CODE.get(code, f"未知代码 {code}")

    # 风向角度 → 中文方位
    wind_dir_desc = _wind_direction_to_chinese(wind_dir)

    location_str = city_name_zh
    if admin1 and admin1 != city_name_zh:
        location_str = f"{admin1}·{city_name_zh}"
    if country:
        location_str = f"{location_str}（{country}）"

    return (
        f"地点：{location_str}\n"
        f"天气：{weather_desc}\n"
        f"温度：{temp}°C\n"
        f"湿度：{humidity}%\n"
        f"风速：{wind_speed} km/h {wind_dir_desc}\n"
    )


def _wind_direction_to_chinese(degrees) -> str:
    """把风向角度（0-360）转中文方位。"""
    try:
        d = float(degrees)
    except (TypeError, ValueError):
        return ""
    directions = ["北", "东北", "东", "东南", "南", "西南", "西", "西北"]
    index = int((d + 22.5) // 45) % 8
    return f"（{directions[index]}风）"


def build_search_tool():
    """对外暴露的工厂函数，返回带中文描述的搜索工具。"""
    return web_search


def build_weather_tool():
    """对外暴露的工厂函数，返回天气查询工具。"""
    return get_weather

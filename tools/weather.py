"""Weather query tool."""

from langchain.tools import tool


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
    return weather_data.get(city_lower, f"抱歉，暂无 {city} 的天气数据。")

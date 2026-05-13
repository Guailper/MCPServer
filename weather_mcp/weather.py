from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()  # 加载环境变量

# 初始化FastMCP服务器
mcp = FastMCP("weather")

# 常量
QWEATHER_API_BASE = os.getenv("QWEATHER_API_BASE_URL")
QWEATHER_GEO_BASE = os.getenv("QWEATHER_GEO_BASE_URL")
QWEATHER_API_KEY = os.getenv("QWEATHER_API_KEY")
USER_AGENT = "weather-app/1.0"

def _get_location_id(location: str) -> str:
    """
    获取地理位置ID
    """
    geo_params = {
        "location": location,
        "key": QWEATHER_API_KEY
    }
    geo_response = httpx.get(QWEATHER_GEO_BASE, params=geo_params, headers={"User-Agent": USER_AGENT})
    geo_data = geo_response.json()
    
    if not geo_data.get("location"):
        raise ValueError("无法找到该位置")
    
    return geo_data["location"][0]["id"]

@mcp.tool()
def get_now_weather(location: str) -> Any:
    """
    获取当前天气
    location: 城市名称
    """
    # 获取地理位置ID
    location_id = _get_location_id(location)
    
    # 获取当前天气
    weather_params = {
        "location": location_id,
        "key": QWEATHER_API_KEY
    }
    weather_response = httpx.get(QWEATHER_API_BASE + "now", params=weather_params, headers={"User-Agent": USER_AGENT})
    weather_data = weather_response.json()
    
    if weather_data.get("code") != "200":
        return {"error": "无法获取天气数据"}
    
    return weather_data["now"]
from typing import Any, Optional
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
        - location: 城市名称
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

@mcp.tool()
def get_3d_forecast_weather(location: str, offset: Optional[int] = None) -> Any:
    """
    获取未来三天的天气预报
        - location: 城市名称
        - offset: 偏移量, 用于获取未来第N天的天气预报, 0表示今天, 1表示明天, 2表示后天
    """
    # 获取地理位置ID
    location_id = _get_location_id(location)
    
    # 获取天气预报
    weather_params = {
        "location": location_id,
        "key": QWEATHER_API_KEY
    }
    weather_response = httpx.get(QWEATHER_API_BASE + "3d", params=weather_params, headers={"User-Agent": USER_AGENT})
    weather_data = weather_response.json()
    
    if weather_data.get("code") != "200":
        return {"error": "无法获取天气数据"}
    
    if offset is not None:
        if offset < 0 or offset > 2:
            return {"error": "偏移量必须在0到2之间"}
        return weather_data["daily"][offset]
    
    return weather_data["daily"]

@mcp.tool()
def get_7d_forecast_weather(location: str, offset: Optional[int] = None) -> Any:
    """
    获取未来七天的天气预报
        - location: 城市名称
        - offset: 偏移量, 用于获取未来第N天的天气预报, 0表示今天, 1表示明天, 以此类推，直到6表示未来第七天
    """
    # 获取地理位置ID
    location_id = _get_location_id(location)
    
    # 获取天气预报
    weather_params = {
        "location": location_id,
        "key": QWEATHER_API_KEY
    }
    weather_response = httpx.get(QWEATHER_API_BASE + "7d", params=weather_params, headers={"User-Agent": USER_AGENT})
    weather_data = weather_response.json()
    
    if weather_data.get("code") != "200":
        return {"error": "无法获取天气数据"}
    
    if offset is not None:
        if offset < 0 or offset > 6:
            return {"error": "偏移量必须在0到6之间"}
        return weather_data["daily"][offset]
    
    return weather_data["daily"]
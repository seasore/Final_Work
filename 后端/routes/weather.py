# -*- coding: utf-8 -*-
"""天气路由：预报元数据供前端展示"""
from fastapi import APIRouter, Depends

from auth import get_current_user
from services.weather_service import get_weather_report

router = APIRouter(prefix="/api/weather", tags=["weather"])


@router.get("/forecast")
def weather_forecast(days: int = 2, user: dict = Depends(get_current_user)):
    """返回与预测一致来源的天气预报（彭水县 / 气候学回退）。"""
    return get_weather_report(days=min(max(days, 1), 5))

# -*- coding: utf-8 -*-
"""预测路由"""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from auth import get_current_user
from services.prediction_service import get_prediction_service
from services.weather_service import get_weather_report

router = APIRouter(prefix="/api/prediction", tags=["prediction"])


@router.get("/two-days")
def get_two_days_prediction(
    start: Optional[str] = None,
    steps: int = Query(
        96,
        ge=1,
        le=192,
        description="滚动预测步数：96≈一天（默认，较快），192=两天（较慢）",
    ),
    user: dict = Depends(get_current_user),
):
    """
    获取未来若干 15min 步的负荷预测（默认 96 步以缩短 CPU 耗时）。
    """
    try:
        svc = get_prediction_service()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    start_dt = None
    if start:
        try:
            start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except Exception:
            pass
    weather_report = get_weather_report(days=2)
    results = svc.predict_two_days(
        start_dt=start_dt,
        weather_forecast=weather_report["forecast"],
        n_steps=steps,
    )
    return {
        "data": results,
        "interval_minutes": 15,
        "n_steps": steps,
        "weather": weather_report,
        "compare_models_loaded": svc.get_loaded_compare_models(),
    }

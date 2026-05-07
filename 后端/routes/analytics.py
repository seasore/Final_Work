# -*- coding: utf-8 -*-
"""训练集与预测回测分析 API"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from auth import get_current_user
from db import get_db
from services import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/location")
def get_location(_: dict = Depends(get_current_user)):
    return analytics_service.location_meta()


@router.get("/correlation")
def get_correlation(
    day_type: str = Query("all", description="all | normal | weekend | holiday"),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.correlation_matrix(day_type=day_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/model-scatter")
def get_model_scatter(
    start: Optional[str] = None,
    steps: int = Query(
        96,
        ge=1,
        le=192,
        description="预测步数：越小越快。96≈1天，192=2天",
    ),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.model_scatter_points(start_iso=start, n_steps=steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/backtest")
def get_backtest(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: Optional[str] = None,
    max_days: int = Query(3, ge=1, le=7, description="最多回测天数，避免单次请求过久"),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.backtest_eval(
            start_date=start, end_date=end, max_days=max_days
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/heatmap-96")
def get_heatmap_96(
    start: str = Query(..., description="YYYY-MM-DD"),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.heatmap_96(start_date=start)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/holiday-load")
def holiday_load(
    year: int = Query(...),
    db: Database = Depends(get_db),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.holiday_period_load_stats(year=year, db=db)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/holiday-accuracy")
def holiday_accuracy(
    year: int = Query(...),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.holiday_model_accuracy_bars(year=year)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/assessment")
def assessment(
    year: int = Query(...),
    month: int = Query(..., ge=1, le=12),
    _: dict = Depends(get_current_user),
):
    try:
        return analytics_service.assessment_for_month(year=year, month=month)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

# -*- coding: utf-8 -*-
"""
基于训练集与预测服务的真实统计分析（无随机假数据）。
"""
from __future__ import annotations

import os
import pickle
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from services.scaler_compat import scaler_inverse_transform
from db import get_db
from holiday_seed import seed_statutory_holidays_if_empty
from services.prediction_service import get_prediction_service
from services.weather_service import get_weather_report


def _train_path() -> str:
    return settings.TRAIN_CSV


def _parse_holiday_cfg_date(val: Any) -> Optional[date]:
    """解析 holiday_config 的日期字段（str / datetime / date / Timestamp）。"""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, date):
        return val
    if hasattr(val, "to_pydatetime"):
        try:
            return val.to_pydatetime().date()
        except Exception:
            pass
    s = str(val).strip().replace("T", " ")[:10]
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def load_train_df() -> pd.DataFrame:
    path = _train_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"训练集不存在: {path}")
    df = pd.read_csv(path, parse_dates=["DATETIME"])
    return df.sort_values("DATETIME").reset_index(drop=True)


def _load_scalers_dict() -> dict:
    """仅读标准化器（节假日负荷柱图无需加载 PyTorch，避免与准确率接口争用模型冷启动）。"""
    path = settings.SCALERS_PKL
    if not os.path.exists(path):
        raise FileNotFoundError(f"标准化器不存在: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _denorm_one(scalers: dict, col: str, z: float) -> float:
    if col not in scalers:
        return float(z)
    return float(scaler_inverse_transform(scalers[col], np.array([[z]], dtype=np.float64))[0, 0])


def _denorm_load_series(scalers: dict, z: np.ndarray) -> np.ndarray:
    sc = scalers.get("M019Value")
    if sc is None:
        return z.astype(np.float64)
    return scaler_inverse_transform(sc, z.reshape(-1, 1)).ravel()


def correlation_matrix(day_type: str = "all") -> Dict[str, Any]:
    """负荷与气象列 Pearson 相关矩阵（训练集标准化值，相关性与物理量一致）。"""
    df = load_train_df()
    cols = ["M019Value", "max_temp", "min_temp", "humidity", "wind_speed"]
    if day_type == "weekend":
        df = df[df["is_weekend"] == 1]
    elif day_type == "holiday":
        df = df[df["is_holiday"] == 1]
    elif day_type == "normal":
        df = df[(df["is_weekend"] == 0) & (df["is_holiday"] == 0)]
    if len(df) < 10:
        raise ValueError("该筛选下样本过少，请换日类型或改用「全部」")
    mat = df[cols].corr().values.tolist()
    labels = ["负荷", "最高温", "最低温", "湿度", "风速"]
    return {"labels": labels, "matrix": mat, "sample_size": int(len(df))}


def model_scatter_points(
    start_iso: Optional[str] = None, n_steps: int = 96
) -> Dict[str, Any]:
    """主模型 vs 持久化模型散点。默认只算 96 步（约一天），约为全量 192 步的一半耗时。"""
    svc = get_prediction_service()
    start_dt = None
    if start_iso:
        try:
            start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            start_dt = None
    wr = get_weather_report(days=2)
    n_steps = max(1, min(int(n_steps), 192))
    preds = svc.predict_two_days(
        start_dt=start_dt, weather_forecast=wr["forecast"], n_steps=n_steps
    )
    pairs = [[p["main_mw"], p["baseline_mw"]] for p in preds]
    return {
        "points": pairs,
        "n_steps": n_steps,
        "weather_source": wr.get("source"),
        "location_label": settings.WEATHER_LOCATION_LABEL,
    }


def _build_weather_forecast_from_rows(
    df_slice: pd.DataFrame, scalers: dict
) -> List[Dict[str, Any]]:
    """将连续 15min 行按日聚合为 predict_two_days 所需 forecast。"""
    if df_slice.empty:
        return []
    tmp = df_slice.copy()
    tmp["_d"] = tmp["DATETIME"].dt.date
    out: List[Dict[str, Any]] = []
    for d, g in tmp.groupby("_d"):
        mx = float(g["max_temp"].max())
        mn = float(g["min_temp"].min())
        hm = float(g["humidity"].mean())
        wd = float(g["wind_speed"].max())
        tmax, tmin, hum, wind = (
            _denorm_one(scalers, "max_temp", mx),
            _denorm_one(scalers, "min_temp", mn),
            _denorm_one(scalers, "humidity", hm),
            _denorm_one(scalers, "wind_speed", wd),
        )
        out.append(
            {
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "max_temp": round(tmax, 3),
                "min_temp": round(tmin, 3),
                "humidity": round(hum, 3),
                "wind_speed": round(wind, 3),
            }
        )
    return sorted(out, key=lambda x: x["date"])


def backtest_eval(
    start_date: str,
    end_date: Optional[str] = None,
    max_days: int = 14,
) -> Dict[str, Any]:
    """
    历史回测：训练集中真实负荷 vs 主模型/持久化预测。
    start_date/end_date: 'YYYY-MM-DD'，含端点；默认 end= start。
    """
    df = load_train_df()
    svc = get_prediction_service()
    scalers = svc._scalers  # noqa: SLF001

    d0 = datetime.strptime(start_date[:10], "%Y-%m-%d")
    d1 = datetime.strptime((end_date or start_date)[:10], "%Y-%m-%d")
    if d1 < d0:
        d0, d1 = d1, d0
    days = (d1 - d0).days + 1
    days = min(days, max_days)

    rows: List[Dict[str, Any]] = []
    for k in range(days):
        day = d0 + timedelta(days=k)
        start_dt = day.replace(hour=0, minute=0, second=0, microsecond=0)
        mask = df["DATETIME"] == start_dt
        if not mask.any():
            continue
        idx = int(df.loc[mask].index[0])
        if idx < 95 or idx + 192 >= len(df):
            continue
        hist = df.iloc[idx - 95 : idx + 1]["M019Value"].values.reshape(96, 1).astype(np.float32)
        future = df.iloc[idx + 1 : idx + 193]
        weather_forecast = _build_weather_forecast_from_rows(future, scalers)
        preds = svc.predict_two_days(
            start_dt=start_dt,
            historical_load=hist,
            weather_forecast=weather_forecast,
        )
        actual_z = future["M019Value"].values[:192]
        actual = _denorm_load_series(scalers, actual_z)
        for i, p in enumerate(preds[: len(actual)]):
            a = float(actual[i])
            m = float(p["main_mw"])
            b = float(p["baseline_mw"])
            if a <= 0:
                continue
            mape_m = abs(m - a) / a * 100
            mape_b = abs(b - a) / a * 100
            rows.append(
                {
                    "datetime": p["datetime"],
                    "actual": round(a, 2),
                    "main_mw": round(m, 2),
                    "baseline_mw": round(b, 2),
                    "mape_main_pct": round(mape_m, 2),
                    "mape_baseline_pct": round(mape_b, 2),
                    "accuracy_main_pct": round(max(0.0, 100.0 - mape_m), 2),
                    "accuracy_baseline_pct": round(max(0.0, 100.0 - mape_b), 2),
                }
            )
    return {
        "rows": rows,
        "location_label": settings.WEATHER_LOCATION_LABEL,
        "note": "实际负荷来自训练集反标准化真值；预测为当时刻滚动预测。",
    }


def heatmap_96(start_date: str) -> Dict[str, Any]:
    """单日 96 点：主模型相对真实值的准确率（热力图用）。"""
    res = backtest_eval(start_date, start_date, max_days=1)
    grid = []  # [quarter, hour, value]  ECharts: x 0-3 刻钟, y 0-23 时
    for r in res["rows"]:
        dt = datetime.strptime(r["datetime"], "%Y-%m-%d %H:%M")
        h, qdiv = dt.hour, dt.minute // 15
        acc = r["accuracy_main_pct"]
        grid.append([qdiv, h, round(acc, 2)])
    mean_main = float(np.mean([r["accuracy_main_pct"] for r in res["rows"]])) if res["rows"] else None
    mean_base = float(np.mean([r["accuracy_baseline_pct"] for r in res["rows"]])) if res["rows"] else None
    return {
        "cells": grid,
        "mean_accuracy_main": round(mean_main, 2) if mean_main is not None else None,
        "mean_accuracy_baseline": round(mean_base, 2) if mean_base is not None else None,
        "location_label": settings.WEATHER_LOCATION_LABEL,
    }


def holiday_period_load_stats(year: int, db) -> Dict[str, Any]:
    """节假日期间训练集真实最大/最小负荷（按 Mongo 节假日配置日期范围）。"""
    seed_statutory_holidays_if_empty(db)
    df = load_train_df()
    train_start = df["DATETIME"].min()
    train_end = df["DATETIME"].max()
    train_start_d = train_start.date() if hasattr(train_start, "date") else train_start
    train_end_d = train_end.date() if hasattr(train_end, "date") else train_end
    scalers = _load_scalers_dict()
    items = list(db.holiday_config.find({}))
    bars: List[Dict[str, Any]] = []
    y0 = datetime(year, 1, 1).date()
    y1 = datetime(year, 12, 31).date()
    skipped_no_overlap = 0
    for h in items:
        s = _parse_holiday_cfg_date(h.get("start_date"))
        e = _parse_holiday_cfg_date(h.get("end_date"))
        if s is None or e is None:
            continue
        s2 = max(s, y0)
        e2 = min(e, y1)
        if s2 > e2:
            continue
        sub = df[(df["DATETIME"].dt.date >= s2) & (df["DATETIME"].dt.date <= e2)]
        if sub.empty:
            skipped_no_overlap += 1
            continue
        loads = _denorm_load_series(scalers, sub["M019Value"].values)
        bars.append(
            {
                "name": h.get("name", "节假日"),
                "start_date": s2.isoformat(),
                "end_date": e2.isoformat(),
                "max_load": round(float(np.max(loads)), 2),
                "min_load": round(float(np.min(loads)), 2),
            }
        )
    hint: Optional[str] = None
    if not bars:
        if not items:
            hint = (
                "节假日库为空且无法自动写入（请确认已安装 chinese_calendar 并重启后端，"
                "或由管理员在「节假日配置」中添加假期）。"
            )
        elif skipped_no_overlap >= len(items):
            hint = (
                f"当前训练集仅覆盖 {train_start_d}～{train_end_d}，"
                f"所选 {year} 年的法定假日区间与该范围没有重叠。请改用训练集覆盖的年份（例如有全年数据的年份），"
                "或扩充 数据/训练集/train_data.csv。"
            )
        else:
            hint = (
                f"训练集范围为 {train_start_d}～{train_end_d}；"
                f"{year} 年的节假日大多落在此范围之外，图中暂无重叠片段。"
            )
    return {
        "year": year,
        "bars": bars,
        "location_label": settings.WEATHER_LOCATION_LABEL,
        "train_range": {
            "start": str(train_start_d),
            "end": str(train_end_d),
        },
        "holiday_periods_in_db": len(items),
        "hint": hint,
    }


def holiday_model_accuracy_bars(year: int) -> Dict[str, Any]:
    """每个节假日首日 00:00 起预测首日 96 点，相对训练集真实的平均准确率。"""
    _MAX_PER_REQUEST = 10  # 单次请求内回测档位数上限，避免 CPU 上总耗时超过代理/浏览器容忍
    db = get_db()
    seed_statutory_holidays_if_empty(db)
    df = load_train_df()
    train_start = df["DATETIME"].min()
    train_end = df["DATETIME"].max()
    train_start_d = train_start.date() if hasattr(train_start, "date") else train_start
    train_end_d = train_end.date() if hasattr(train_end, "date") else train_end
    items = list(db.holiday_config.find({}))
    eligible: List[Tuple[date, Any, int]] = []
    for h in items:
        s_d = _parse_holiday_cfg_date(h.get("start_date"))
        if s_d is None or s_d.year != year:
            continue
        s_date = s_d
        day_rows = df[df["DATETIME"].dt.date == s_date].sort_values("DATETIME")
        if day_rows.empty:
            continue
        idx = int(day_rows.index[0])
        if idx < 95 or idx + 97 >= len(df):
            continue
        eligible.append((s_d, h, idx))
    eligible.sort(key=lambda t: t[0])
    total_eligible = len(eligible)
    to_run = eligible[:_MAX_PER_REQUEST]
    svc = get_prediction_service()
    scalers = svc._scalers  # noqa: SLF001
    labels: List[str] = []
    acc_main: List[Optional[float]] = []
    acc_base: List[Optional[float]] = []
    for _s_d, h, idx in to_run:
        start_dt = df.iloc[idx]["DATETIME"]
        if hasattr(start_dt, "to_pydatetime"):
            start_dt = start_dt.to_pydatetime()
        start_dt = start_dt.replace(tzinfo=None)
        hist = df.iloc[idx - 95 : idx + 1]["M019Value"].values.reshape(96, 1).astype(np.float32)
        future = df.iloc[idx + 1 : idx + 97]
        weather_forecast = _build_weather_forecast_from_rows(
            df.iloc[idx + 1 : idx + 193], scalers
        )
        preds = svc.predict_two_days(
            start_dt=start_dt,
            historical_load=hist,
            weather_forecast=weather_forecast,
            n_steps=96,
        )
        actual_z = future["M019Value"].values[:96]
        actual = _denorm_load_series(scalers, actual_z)
        am, ab = [], []
        for i in range(min(96, len(preds), len(actual))):
            a = float(actual[i])
            if a <= 0:
                continue
            m = float(preds[i]["main_mw"])
            b = float(preds[i]["baseline_mw"])
            am.append(max(0.0, 100.0 - abs(m - a) / a * 100))
            ab.append(max(0.0, 100.0 - abs(b - a) / a * 100))
        if not am:
            continue
        labels.append(h.get("name", "节假日"))
        acc_main.append(round(float(np.mean(am)), 2))
        acc_base.append(round(float(np.mean(ab)), 2))
    hint: Optional[str] = None
    if not labels and not items:
        hint = "节假日库为空，已尝试自动写入法定假日；若仍无数据请检查 chinese_calendar 与 MongoDB。"
    elif not labels and items:
        hint = (
            f"训练集为 {train_start_d}～{train_end_d}；"
            f"{year} 年节假日首日无完整 96 点上下文或不在训练集内，故准确率图为空。可换选训练集覆盖更完整的年份。"
        )
    elif total_eligible > _MAX_PER_REQUEST:
        hint = (
            f"本年仅展示按日期排序的前 {_MAX_PER_REQUEST} 个可回测节假日（共 {total_eligible} 个），以控制单次请求耗时。"
        )
    return {
        "year": year,
        "labels": labels,
        "accuracy_main": acc_main,
        "accuracy_baseline": acc_base,
        "location_label": settings.WEATHER_LOCATION_LABEL,
        "train_range": {"start": str(train_start_d), "end": str(train_end_d)},
        "holiday_periods_in_db": len(items),
        "hint": hint,
        "accuracy_eligible_count": total_eligible,
        "accuracy_computed_count": len(labels),
    }


def assessment_for_month(year: int, month: int) -> Dict[str, Any]:
    """指定月份内抽样数日回测，给出区域考核指标。"""
    df = load_train_df()
    t0 = datetime(year, month, 1)
    if month == 12:
        t1 = datetime(year + 1, 1, 1) - timedelta(minutes=15)
    else:
        t1 = datetime(year, month + 1, 1) - timedelta(minutes=15)
    sub = df[(df["DATETIME"] >= t0) & (df["DATETIME"] <= t1)]
    if sub.empty:
        return {
            "rows": [],
            "location_label": settings.WEATHER_LOCATION_LABEL,
            "month": f"{year}-{month:02d}",
        }
    # 取月中连续 3 天做回测（若存在）
    mid = t0 + timedelta(days=14)
    dates = [mid, t0 + timedelta(days=7), t0 + timedelta(days=1)]
    acc_vals = []
    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        try:
            r = backtest_eval(ds, ds, max_days=1)
            for row in r["rows"]:
                acc_vals.append(row["accuracy_main_pct"])
        except Exception:
            continue
    avg = round(float(np.mean(acc_vals)), 2) if acc_vals else None
    return {
        "rows": [
            {
                "unit": settings.WEATHER_LOCATION_LABEL,
                "month": f"{year}-{month:02d}",
                "accuracy": f"{avg}%" if avg is not None else "—",
                "rank": 1,
                "sample_points": len(acc_vals),
            }
        ],
        "location_label": settings.WEATHER_LOCATION_LABEL,
    }


def location_meta() -> Dict[str, str]:
    return {
        "label": settings.WEATHER_LOCATION_LABEL,
        "weather_city_query": settings.WEATHER_CITY,
    }

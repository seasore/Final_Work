# -*- coding: utf-8 -*-
"""
天气服务：优先从 OpenWeatherMap 按配置经纬度（默认彭水县城附近）获取预报；
无 API Key 或请求失败时，用训练集历史同日气候统计作为回退（非固定常数）。
"""
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import settings
from services.scaler_compat import scaler_inverse_transform


def _minimal_forecast_from_means(ndays: int) -> List[Dict[str, Any]]:
    """训练集气象全局均值反标准化，生成 ndays 天兜底预报（保证前端总有数据可展示）。"""
    mx, mn, h, w = get_training_denorm_weather_means()
    base = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    out: List[Dict[str, Any]] = []
    for i in range(max(1, min(ndays, 5))):
        d = base + timedelta(days=i)
        out.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "datetime": d.isoformat(),
                "max_temp": round(mx, 2),
                "min_temp": round(mn, 2),
                "humidity": round(h, 2),
                "wind_speed": round(w, 2),
                "pressure": None,
                "description": "训练集气象均值（兜底）",
                "weather_main": "N/A",
            }
        )
    return out


def get_weather_report(days: int = 2) -> Dict[str, Any]:
    """
    返回天气预报列表及元数据，供预测与前端展示。
    forecast 每项含: date, max_temp, min_temp, humidity, wind_speed,
    pressure(可选), description(可选), weather_main(可选)
    """
    city = settings.WEATHER_CITY
    label = settings.WEATHER_LOCATION_LABEL
    ndays = min(max(days, 1), 5)

    forecast, source, err = _try_openweather(ndays)
    if forecast:
        out = {
            "forecast": forecast,
            "source": source,
            "city_query": city,
            "location_label": label,
            "error": None,
        }
        return _ensure_forecast_not_empty(out, ndays, err)

    try:
        forecast_fb = _fallback_climatology_from_training(ndays)
        out = {
            "forecast": forecast_fb,
            "source": "training_climatology",
            "city_query": city,
            "location_label": label,
            "error": err
            or "未配置 WEATHER_API_KEY 或 API 不可用，已使用训练集历史同日统计（反标准化）作为气象输入。",
        }
        return _ensure_forecast_not_empty(out, ndays, err)
    except Exception as ex:
        out = {
            "forecast": [],
            "source": "error",
            "city_query": city,
            "location_label": label,
            "error": f"{err or ''}; 气候学回退失败: {ex}",
        }
        return _ensure_forecast_not_empty(out, ndays, err)


def _ensure_forecast_not_empty(
    meta: Dict[str, Any], ndays: int, openweather_err: Optional[str]
) -> Dict[str, Any]:
    """若 forecast 仍为空，用均值生成，避免前端「暂无气象」。"""
    fc = meta.get("forecast") or []
    if fc:
        return meta
    try:
        meta["forecast"] = _minimal_forecast_from_means(ndays)
        if meta.get("source") == "error":
            meta["source"] = "mean_fallback"
        prev = meta.get("error") or ""
        meta["error"] = (prev + "；已使用训练集气象均值兜底展示。").strip("；")
    except Exception:
        pass
    return meta


def fetch_weather_forecast(days: int = 2) -> List[Dict]:
    """兼容旧接口：仅返回预报列表。"""
    return get_weather_report(days)["forecast"]


def get_training_denorm_weather_means() -> Tuple[float, float, float, float]:
    """训练集气象列（标准化后）全局均值再反标准化，作预测兜底。"""
    path = settings.TRAIN_CSV
    if not os.path.exists(path) or not os.path.exists(settings.SCALERS_PKL):
        return 25.0, 15.0, 60.0, 3.0
    with open(settings.SCALERS_PKL, "rb") as f:
        scalers = pickle.load(f)
    df = pd.read_csv(path)
    need = ["max_temp", "min_temp", "humidity", "wind_speed"]
    if not all(c in df.columns for c in need):
        return 25.0, 15.0, 60.0, 3.0
    return _denormalize_weather_row(
        scalers,
        float(df["max_temp"].mean()),
        float(df["min_temp"].mean()),
        float(df["humidity"].mean()),
        float(df["wind_speed"].mean()),
    )


def _try_openweather(days: int) -> Tuple[List[Dict], Optional[str], Optional[str]]:
    """调用 OpenWeatherMap 5 日/3 小时预报，按日聚合。成功返回 (list, 'openweather', None)。"""
    if not settings.WEATHER_API_KEY:
        return [], None, None

    try:
        import httpx

        url = settings.WEATHER_API_URL
        # cnt 最大 40（约 5 天×8 次/天）
        cnt = min(40, max(days * 8, 16))
        params = {
            "lat": settings.WEATHER_LAT,
            "lon": settings.WEATHER_LON,
            "appid": settings.WEATHER_API_KEY,
            "units": "metric",
            "cnt": cnt,
        }
        with httpx.Client() as client:
            r = client.get(url, params=params, timeout=15)
        if r.status_code != 200:
            detail = r.text[:200]
            if r.status_code == 401:
                detail += (
                    "（新申请的 Key 常需数小时才生效，见注册邮件说明；"
                    "生效前界面会用训练集气候学数据兜底。）"
                )
            if r.status_code == 404:
                detail += (
                    "（若曾用城市名查询：小县城可能不在库中，已在配置中改为 lat/lon；"
                    "仍 404 请检查 WEATHER_LAT/WEATHER_LON 或改用附近大城市坐标。）"
                )
            return [], None, f"OpenWeather HTTP {r.status_code}: {detail}"

        data = r.json()
        items = data.get("list") or []
        if not items:
            return [], None, "OpenWeather 返回空列表"

        daily = _aggregate_openweather_by_date(items)
        out = []
        for i in range(min(days, len(daily))):
            out.append(daily[i])
        return out, "openweather", None
    except Exception as e:
        return [], None, str(e)


def _aggregate_openweather_by_date(items: list) -> List[Dict]:
    """将 3h 列表按本地日期聚合为日最高/最低温、平均湿度、最大风速、平均气压、代表天气描述。"""
    from collections import defaultdict

    by_date: Dict[str, list] = defaultdict(list)
    for item in items:
        dt = datetime.fromtimestamp(item["dt"])
        date_str = dt.strftime("%Y-%m-%d")
        by_date[date_str].append(item)

    result = []
    for date_str in sorted(by_date.keys()):
        day_items = by_date[date_str]
        temps_hi = []
        temps_lo = []
        hums = []
        winds = []
        pressures = []
        desc = "—"
        wmain = "Clear"
        for it in day_items:
            m = it.get("main") or {}
            t = m.get("temp")
            if t is not None:
                temps_hi.append(float(t))
                temps_lo.append(float(t))
            if m.get("temp_max") is not None:
                temps_hi.append(float(m["temp_max"]))
            if m.get("temp_min") is not None:
                temps_lo.append(float(m["temp_min"]))
            if m.get("humidity") is not None:
                hums.append(float(m["humidity"]))
            w = it.get("wind") or {}
            if w.get("speed") is not None:
                winds.append(float(w["speed"]))
            if m.get("pressure") is not None:
                pressures.append(float(m["pressure"]))
            wx = it.get("weather") or []
            if wx and wx[0].get("description"):
                desc = wx[0].get("description", desc)
                wmain = wx[0].get("main", wmain)

        max_t = max(temps_hi) if temps_hi else 20.0
        min_t = min(temps_lo) if temps_lo else 10.0
        result.append(
            {
                "date": date_str,
                "datetime": f"{date_str}T12:00:00",
                "max_temp": round(max_t, 2),
                "min_temp": round(min_t, 2),
                "humidity": round(sum(hums) / len(hums), 1) if hums else 60.0,
                "wind_speed": round(max(winds), 2) if winds else 0.0,
                "pressure": round(sum(pressures) / len(pressures), 1) if pressures else None,
                "description": desc,
                "weather_main": wmain,
            }
        )
    return result


def _denormalize_weather_row(
    scalers: dict, max_n: float, min_n: float, hum_n: float, wind_n: float
) -> Tuple[float, float, float, float]:
    """训练集中气象列为 Z-score 标准化值，反变换为原始物理量。"""
    arr = np.array([[max_n, min_n, hum_n, wind_n]], dtype=np.float64)
    cols = ["max_temp", "min_temp", "humidity", "wind_speed"]
    for i, col in enumerate(cols):
        if col in scalers:
            arr[:, i] = scaler_inverse_transform(scalers[col], arr[:, i : i + 1]).ravel()
    return float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2]), float(arr[0, 3])


def _fallback_climatology_from_training(days: int) -> List[Dict]:
    """
    用训练集中「历年同一月日」的气象标准化值取均值，再经 scalers 反变换，
    得到未来 days 天的原始温湿风（随日期变化，非固定常数）。
    """
    path = settings.TRAIN_CSV
    if not os.path.exists(path):
        raise RuntimeError(
            f"无天气 API 且训练集不存在: {path}，请配置 WEATHER_API_KEY 或放置 train_data.csv"
        )
    if not os.path.exists(settings.SCALERS_PKL):
        raise RuntimeError(f"缺少标准化器 {settings.SCALERS_PKL}，无法将训练集气象反变换为原始值")

    with open(settings.SCALERS_PKL, "rb") as f:
        scalers = pickle.load(f)

    df = pd.read_csv(path)
    time_col = "DATETIME" if "DATETIME" in df.columns else ("DATE" if "DATE" in df.columns else "datetime")
    if time_col not in df.columns:
        raise RuntimeError("训练集缺少 DATETIME / DATE / datetime 列")

    df["_dt"] = pd.to_datetime(df[time_col])
    df["_md"] = df["_dt"].dt.month * 100 + df["_dt"].dt.day

    need = ["max_temp", "min_temp", "humidity", "wind_speed"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"训练集缺少列 {c}")

    clim = df.groupby("_md", as_index=False)[need].mean()
    global_mean = df[need].mean()

    out = []
    base = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    for i in range(days):
        d = base + timedelta(days=i)
        md = d.month * 100 + d.day
        row = clim[clim["_md"] == md]
        if row.empty:
            r = global_mean
        else:
            r = row.iloc[0]
        max_t, min_t, hum, wind = _denormalize_weather_row(
            scalers,
            float(r["max_temp"]),
            float(r["min_temp"]),
            float(r["humidity"]),
            float(r["wind_speed"]),
        )
        out.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "datetime": d.isoformat(),
                "max_temp": round(max_t, 2),
                "min_temp": round(min_t, 2),
                "humidity": round(hum, 2),
                "wind_speed": round(wind, 2),
                "pressure": None,
                "description": "Climatology (training, denormalized)",
                "weather_main": "N/A",
            }
        )
    return out

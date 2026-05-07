# -*- coding: utf-8 -*-
"""首次启动时在 MongoDB 中预设中国法定节假日区间（依据 chinese_calendar）。"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List

# chinese_calendar.get_holiday_detail 返回 (bool, 英文节日名 str) 等，见库文档
_PHRASE_ZH = {
    "Spring Festival": "春节",
    "New Year's Day": "元旦",
    "Tomb-sweeping Day": "清明",
    "Labour Day": "劳动节",
    "Dragon Boat Festival": "端午",
    "National Day": "国庆",
    "Mid-autumn Festival": "中秋",
    "Mid-Autumn Festival": "中秋",
    "Chinese New Year's Eve": "除夕",
    "Children's Day": "儿童节",
}


def _holiday_label(detail) -> str:
    """从 get_holiday_detail 的返回值解析中文节日名（避免误用 detail[0] 的 bool 导致「年True」）。"""
    if not detail:
        return "法定节假日"
    raw: str | None = None
    if len(detail) >= 2 and detail[1] is not None and not isinstance(detail[1], bool):
        second = detail[1]
        if isinstance(second, str):
            raw = second.strip() or None
        else:
            # chinese_calendar.Holiday 枚举等有 .chinese
            zh = getattr(second, "chinese", None)
            if isinstance(zh, str) and zh.strip():
                raw = zh.strip()
            else:
                en = getattr(second, "value", second)
                if isinstance(en, str) and en.strip():
                    raw = en.strip()
                else:
                    raw = str(second).strip() or None
    if not raw and len(detail) >= 1:
        first = detail[0]
        if first is not None and not isinstance(first, bool):
            if hasattr(first, "chinese") and isinstance(getattr(first, "chinese", None), str):
                raw = first.chinese.strip()
            else:
                raw = str(getattr(first, "name", first)).strip() or None
    if not raw or raw in ("True", "False"):
        return "法定节假日"
    return _PHRASE_ZH.get(raw, raw)


def _safe_period_label(block_label: str | None) -> str:
    """flush 时禁止使用 `block_label or 默认`：在 Python 中 True or \"x\" 仍为 True，会生成「年True」。"""
    if isinstance(block_label, str) and block_label.strip():
        return block_label.strip()
    return "法定节假日"


def statutory_holiday_periods_for_year(year: int) -> List[Dict[str, Any]]:
    import chinese_calendar as cal

    periods: List[Dict[str, Any]] = []
    d = date(year, 1, 1)
    end = date(year, 12, 31)
    block_start: date | None = None
    block_end: date | None = None
    block_label: str | None = None

    def flush():
        nonlocal block_start, block_end, block_label
        if block_start is not None and block_end is not None:
            label = _safe_period_label(block_label)
            periods.append(
                {
                    "name": f"{year}年{label}",
                    "start_date": block_start.isoformat(),
                    "end_date": block_end.isoformat(),
                    "type": "法定",
                }
            )
        block_start = block_end = block_label = None

    while d <= end:
        if cal.is_holiday(d):
            detail = cal.get_holiday_detail(d)
            # 普通周末 is_holiday 为 True 但 detail[1] 为 None，不计入「法定节假日」配置
            if not detail or detail[1] is None:
                flush()
                d += timedelta(days=1)
                continue
            label = _holiday_label(detail)
            if block_start is None:
                block_start = block_end = d
                block_label = label
            elif d == block_end + timedelta(days=1):
                block_end = d
            else:
                flush()
                block_start = block_end = d
                block_label = label
        else:
            flush()
        d += timedelta(days=1)
    flush()
    return periods


def resync_statutory_holidays(db) -> int:
    """
    删除库中 type=法定 的记录并按日历重新写入（用于修复历史 bug 导致的「2023年True」等名称）。
    不删除「调休」等非法定记录。
    """
    try:
        import chinese_calendar  # noqa: F401
    except ImportError:
        return 0
    db.holiday_config.delete_many({"type": "法定"})
    to_insert: List[Dict[str, Any]] = []
    for y in range(2023, 2027):
        to_insert.extend(statutory_holiday_periods_for_year(y))
    if not to_insert:
        return 0
    db.holiday_config.insert_many(to_insert)
    return len(to_insert)


def seed_statutory_holidays_if_empty(db) -> int:
    """若 holiday_config 为空，则写入多年法定节假日期段。返回插入条数。"""
    try:
        n_existing = db.holiday_config.count_documents({})
    except Exception:
        return 0
    # 修复旧版误用 bool 或 的命名：一旦出现「年True」等则重写法定假日
    try:
        if n_existing > 0 and db.holiday_config.count_documents({"name": {"$regex": r"年True"}}) > 0:
            print("[holiday] 检测到错误节假日名称（如 年True），正在重写法定假日配置…")
            return resync_statutory_holidays(db)
    except Exception:
        pass
    try:
        if n_existing > 0:
            return 0
    except Exception:
        return 0
    try:
        import chinese_calendar  # noqa: F401
    except ImportError:
        return 0
    to_insert: List[Dict[str, Any]] = []
    for y in range(2023, 2027):
        to_insert.extend(statutory_holiday_periods_for_year(y))
    if not to_insert:
        return 0
    db.holiday_config.insert_many(to_insert)
    return len(to_insert)

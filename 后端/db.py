# -*- coding: utf-8 -*-
"""MongoDB 数据库连接"""
from pymongo import MongoClient
from pymongo.database import Database
from config import settings
from holiday_seed import seed_statutory_holidays_if_empty

_client: MongoClient = None


def get_db() -> Database:
    """获取数据库实例"""
    global _client
    if _client is None:
        # 缩短等待，避免启动时卡住 30s；连接失败见 app 启动提示
        _client = MongoClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )
    return _client[settings.MONGODB_DB]


def init_db():
    """初始化数据库索引"""
    db = get_db()
    # 用户索引
    db.users.create_index("username", unique=True)
    db.users.create_index("group_id")
    db.users.create_index([("role", 1), ("group_id", 1)])
    # 单位索引
    db.units.create_index("code", unique=True)
    db.units.create_index("region")
    # 负荷数据索引
    db.load_data.create_index([("unit_id", 1), ("datetime", 1)])
    db.load_data.create_index("datetime")
    # 气象数据索引
    db.weather_data.create_index([("unit_id", 1), ("datetime", 1)])
    db.weather_data.create_index("datetime")
    # 预测数据索引
    db.predictions.create_index([("unit_id", 1), ("datetime", 1)])
    db.predictions.create_index([("model_type", 1), ("datetime", 1)])
    db.predictions.create_index("created_at")
    # 节假日配置索引
    db.holiday_config.create_index("date")
    db.holiday_config.create_index("name")
    # 消息索引
    db.messages.create_index([("receiver_id", 1), ("created_at", -1)])
    db.messages.create_index("sender_id")
    # 汇报索引
    db.reports.create_index([("urgency", -1), ("created_at", -1)])
    db.reports.create_index([("replied", 1), ("urgency", -1)])
    db.reports.create_index("reporter_id")
    # 日志索引
    db.operation_logs.create_index("created_at")
    db.operation_logs.create_index("user_id")

    # 法定节假日：库中无配置时按日历自动预设（chinese_calendar）
    n_h = seed_statutory_holidays_if_empty(db)
    if n_h:
        print(f"[holiday] 已预设法定节假日区间 {n_h} 条（holiday_config 原为空）")

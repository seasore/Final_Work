# -*- coding: utf-8 -*-
"""后端配置"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
PREDICT_MODEL_DIR = os.path.join(PROJECT_ROOT, '预测模型')
DATA_PREPROCESS_DIR = os.path.join(PROJECT_ROOT, '数据预处理')
DATA_DIR = os.path.join(PROJECT_ROOT, '数据')


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
    )
    # MongoDB
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "power_forecast"
    
    # JWT
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24小时
    
    # 预测模型路径
    BEST_MODEL_PT: str = os.path.join(PREDICT_MODEL_DIR, 'best_model.pt')
    # 对比模型检查点目录（由 预测模型/main.py 训练基线后写入；缺失时接口仅返回主模型+持久化基线）
    BASELINE_CHECKPOINT_DIR: str = os.path.join(PREDICT_MODEL_DIR, 'baseline_checkpoints')
    # 是否加载对比模型参与滚动预测（关可缩短冷启动与显存占用）
    LOAD_BASELINE_MODELS: bool = True
    SCALERS_PKL: str = os.path.join(DATA_PREPROCESS_DIR, 'scalers.pkl')
    TRAIN_CSV: str = os.path.join(DATA_DIR, '训练集', 'train_data.csv')
    
    # 天气 API（OpenWeatherMap 2.5 forecast；未配置 KEY 时用训练集气候学回退）
    WEATHER_API_KEY: str = ""
    WEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5/forecast"
    # 县城英文名在 OpenWeather 的 `q` 里常 404，默认用县城中心经纬度（与 WEATHER_LOCATION_LABEL 对应）
    WEATHER_LAT: float = 29.2933
    WEATHER_LON: float = 108.1664
    # 按城市名查询时的 `q`（备用，例如 Chongqing,CN）；当前请求优先使用 WEATHER_LAT/LON
    WEATHER_CITY: str = "Pengshui,CN"
    WEATHER_LOCATION_LABEL: str = "重庆市彭水县"


settings = Settings()

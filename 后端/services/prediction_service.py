# -*- coding: utf-8 -*-
"""
预测服务：主模型 TCN-BiLSTM-Attention + 可选对比模型（LSTM / BiLSTM / TCN / BiLSTM-Attention）
- GPU 加速推理；多模型各自维护滚动窗口，公平自回归外推
- 预计算气象/时间张量；对比权重来自 预测模型/baseline_checkpoints/*.pt（训练主流程生成）
"""
import os
import sys
import pickle
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PREDICT_DIR = os.path.join(_PROJECT_ROOT, '预测模型')
if _PREDICT_DIR not in sys.path:
    sys.path.insert(0, _PREDICT_DIR)

from config import settings
from services.scaler_compat import scaler_inverse_transform, scaler_transform
from services.weather_service import get_training_denorm_weather_means

# 与 main.py 写入的文件名一致：BiLSTM-Attention -> BiLSTM_Attention.pt
_BASELINE_ORDER = [
    ('LSTM', 'LSTM.pt', 'lstm_mw'),
    ('BiLSTM', 'BiLSTM.pt', 'bilstm_mw'),
    ('TCN', 'TCN.pt', 'tcn_mw'),
    ('BiLSTM-Attention', 'BiLSTM_Attention.pt', 'bilstm_attention_mw'),
]


def _compute_time_features(dt: datetime) -> np.ndarray:
    hour = dt.hour + dt.minute / 60
    weekday = dt.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    return np.array([hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend], dtype=np.float32)


class PredictionService:
    _instance = None
    _model = None
    _baseline_models: Dict[str, torch.nn.Module] = {}
    _scalers = None
    _best_params = None
    _device = None
    _loaded_compare_names: List[str] = []

    @classmethod
    def get_instance(cls) -> "PredictionService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._baseline_models = {}
        self._loaded_compare_names = []
        self._load_model()

    def _load_model(self):
        if not os.path.exists(settings.BEST_MODEL_PT):
            raise FileNotFoundError(f"模型文件不存在: {settings.BEST_MODEL_PT}")
        if not os.path.exists(settings.SCALERS_PKL):
            raise FileNotFoundError(f"标准化器不存在: {settings.SCALERS_PKL}")

        try:
            sys.path.insert(0, _PREDICT_DIR)
            from device_utils import get_best_device
            _dev_str = get_best_device(verbose=True)
        except Exception:
            _dev_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(_dev_str)

        from model import TCNBiLstmZhuYiLiYuCeMoXing

        ckpt = torch.load(settings.BEST_MODEL_PT, map_location="cpu")
        self._best_params = ckpt.get("best_params", {})
        self._model = TCNBiLstmZhuYiLiYuCeMoXing(
            fuHeTeZhengShu=1,
            qiXiangTeZhengShu=4,
            shiJianTeZhengShu=5,
            tcnYinCangWeiDu=self._best_params.get("tcnYinCangWeiDu", 64),
            tcnCengShu=self._best_params.get("tcnCengShu", 4),
            bilstmYinCangWeiDu=self._best_params.get("bilstmYinCangWeiDu", 64),
            bilstmCengShu=self._best_params.get("bilstmCengShu", 2),
            zhuYiLiTouShu=self._best_params.get("zhuYiLiTouShu", 4),
            yuCeBuChang=1,
            tuoQiLv=self._best_params.get("tuoQiLv", 0.2),
        )
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.to(self._device)
        self._model.eval()

        if self._device.type == "cuda":
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
            except Exception:
                pass

        with open(settings.SCALERS_PKL, "rb") as f:
            self._scalers = pickle.load(f)

        self._load_baselines_if_enabled()

    def _load_baselines_if_enabled(self):
        self._baseline_models = {}
        self._loaded_compare_names = []
        if not getattr(settings, "LOAD_BASELINE_MODELS", True):
            return
        bdir = getattr(settings, "BASELINE_CHECKPOINT_DIR", None)
        if not bdir or not os.path.isdir(bdir):
            return
        try:
            from baselines import chuangJianDuiBiMoXing
        except Exception:
            return

        for display_name, fname, _ in _BASELINE_ORDER:
            path = os.path.join(bdir, fname)
            if not os.path.isfile(path):
                continue
            try:
                bck = torch.load(path, map_location="cpu")
                params = bck.get("params") or {}
                m = chuangJianDuiBiMoXing(display_name, params, 1)
                m.load_state_dict(bck["state_dict"])
                m.to(self._device)
                m.eval()
                self._baseline_models[display_name] = m
                self._loaded_compare_names.append(display_name)
            except Exception:
                continue

    def get_loaded_compare_models(self) -> List[str]:
        return list(self._loaded_compare_names)

    def _get_historical_load(self) -> np.ndarray:
        if not os.path.exists(settings.TRAIN_CSV):
            return np.zeros((96, 1), dtype=np.float32)
        df = pd.read_csv(settings.TRAIN_CSV)
        load = df["M019Value"].values[-96:].reshape(-1, 1).astype(np.float32)
        return load

    def _get_weather_for_datetime(
        self, dt: datetime, weather_forecast: Optional[List[Dict]] = None
    ) -> Tuple[float, float, float, float]:
        d = get_training_denorm_weather_means()
        if weather_forecast:
            date_str = dt.strftime("%Y-%m-%d")
            for w in weather_forecast:
                if w.get("date") == date_str or (w.get("datetime") and date_str in str(w["datetime"])):
                    return (
                        float(w.get("max_temp", d[0])),
                        float(w.get("min_temp", d[1])),
                        float(w.get("humidity", d[2])),
                        float(w.get("wind_speed", d[3])),
                    )
        return d

    def _transform_weather(
        self, max_t: float, min_t: float, humidity: float, wind: float
    ) -> np.ndarray:
        arr = np.array([[max_t, min_t, humidity, wind]], dtype=np.float32)
        for i, col in enumerate(["max_temp", "min_temp", "humidity", "wind_speed"]):
            if col in self._scalers:
                arr[:, i] = scaler_transform(self._scalers[col], arr[:, i : i + 1]).ravel()
        return arr.flatten()

    def _precompute_step_tensors(
        self,
        start_dt: datetime,
        n_steps: int,
        weather_forecast: Optional[List[Dict]],
        seq_len: int = 96,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qi_xiang_list = []
        shi_jian_list = []
        for step in range(n_steps):
            pred_dt = start_dt + timedelta(minutes=15 * (step + 1))
            max_t, min_t, hum, wind = self._get_weather_for_datetime(pred_dt, weather_forecast)
            qi_feat = self._transform_weather(max_t, min_t, hum, wind)
            t_feat = _compute_time_features(pred_dt)
            qi_xiang_list.append(np.tile(qi_feat, (seq_len, 1)))
            shi_jian_list.append(np.tile(t_feat, (seq_len, 1)))
        qi_xiang_arr = np.stack(qi_xiang_list, axis=0)
        shi_jian_arr = np.stack(shi_jian_list, axis=0)
        qi_xiang_t = torch.from_numpy(qi_xiang_arr).to(self._device)
        shi_jian_t = torch.from_numpy(shi_jian_arr).to(self._device)
        return qi_xiang_t, shi_jian_t

    def _norm_to_mw(self, load_scaler, pred_norm_val: float) -> float:
        pred_mw = float(scaler_inverse_transform(load_scaler, np.array([[pred_norm_val]], dtype=np.float64))[0, 0])
        if not math.isfinite(pred_mw):
            pred_mw = 0.0
        return max(0.0, pred_mw)

    def predict_two_days(
        self,
        start_dt: Optional[datetime] = None,
        historical_load: Optional[np.ndarray] = None,
        weather_forecast: Optional[List[Dict]] = None,
        n_steps: int = 192,
    ) -> List[Dict]:
        if start_dt is None:
            start_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
        if getattr(start_dt, "tzinfo", None) is not None:
            start_dt = start_dt.replace(tzinfo=None)
        start_dt = start_dt.replace(minute=(start_dt.minute // 15) * 15, second=0, microsecond=0)

        if historical_load is None:
            historical_load = self._get_historical_load()
        if historical_load.shape[0] < 96:
            pad = np.zeros((96 - historical_load.shape[0], historical_load.shape[1]), dtype=np.float32)
            historical_load = np.vstack([pad, historical_load])

        n_steps = max(1, min(int(n_steps), 192))
        seq_len = 96

        load_scaler = self._scalers.get("M019Value")
        if load_scaler is None:
            load_scaler = type("Dummy", (), {"inverse_transform": lambda self, x: x})()

        qi_xiang_all, shi_jian_all = self._precompute_step_tensors(
            start_dt, n_steps, weather_forecast, seq_len
        )

        init_load = historical_load[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)
        init_t = torch.from_numpy(init_load).to(self._device)
        win_main = init_t.clone()
        win_bl = {k: init_t.clone() for k in self._baseline_models}

        results = []

        with torch.no_grad():
            for step in range(n_steps):
                pred_dt = start_dt + timedelta(minutes=15 * (step + 1))
                q = qi_xiang_all[step : step + 1]
                s = shi_jian_all[step : step + 1]

                last_main = win_main[0, -1, 0].item()

                pred_norm_t = self._model(win_main, q, s)
                pred_norm_val = pred_norm_t[0, 0].item()
                new_val = pred_norm_t.view(1, 1, 1)
                win_main = torch.cat([win_main[:, 1:, :], new_val], dim=1)

                row: Dict = {
                    "datetime": pred_dt.strftime("%Y-%m-%d %H:%M"),
                    "main_mw": round(self._norm_to_mw(load_scaler, pred_norm_val), 2),
                    "baseline_mw": round(self._norm_to_mw(load_scaler, last_main), 2),
                }

                for bname, bmodel in self._baseline_models.items():
                    w = win_bl[bname]
                    pb = bmodel(w, q, s)
                    pv = pb[0, 0].item()
                    key = next((k for dn, _fn, k in _BASELINE_ORDER if dn == bname), None)
                    if key:
                        row[key] = round(self._norm_to_mw(load_scaler, pv), 2)
                    nb = pb.view(1, 1, 1)
                    win_bl[bname] = torch.cat([w[:, 1:, :], nb], dim=1)

                results.append(row)

        return results


def get_prediction_service() -> PredictionService:
    return PredictionService.get_instance()

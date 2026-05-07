# -*- coding: utf-8 -*-
"""
时间序列 K 折交叉验证模块
Time Series K-Fold Cross-Validation Module

开题报告要求：采用交叉验证策略确保模型评估结果的可靠性
使用时序扩展窗口划分，保持时间顺序
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader

from data import DianLiFuHeShuJuJi, _TRAIN_CSV, _VAL_CSV, _SCALERS_PKL, _TIME_FEATURE_COLS
from model import TCNBiLstmZhuYiLiYuCeMoXing
from train import xunLianMoXing, pingGuMoXing
from device_utils import get_best_device


def shiXuKZheJiaoChaYanZheng(
    n_splits=5,
    xuLieChangDu=96,
    yuCeBuChang=1,
    piCiDaXiao=64,
    xueXiLv=0.001,
    xunLianDaiShu=30,
    sheBei=None,
    shiJianTeZhengShu=5,
    best_params=None
):
    """
    时序 K 折交叉验证（扩展窗口）
    将全量数据按时间顺序划分为 K 段，第 i 折用前 i 段训练、第 i+1 段验证
    
    参数:
        n_splits: 折数
        ...: 与 main 相同的超参数
    
    返回: (各折 MAPE 列表, 平均 MAPE, 标准差)
    """
    if sheBei is None:
        try:
            sheBei = get_best_device(verbose=False)
        except RuntimeError:
            sheBei = 'cpu'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, '数据', '训练集', 'train_data.csv')
    val_path = os.path.join(base_dir, '数据', '验证集', 'val_data.csv')
    scaler_path = os.path.join(base_dir, '数据预处理', 'scalers.pkl')
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("请先运行数据预处理: python data_preprocessing.py")
    
    # 合并训练集和验证集用于交叉验证
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    
    scalers = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    
    def _extract_arrays(df):
        fu_he = df['M019Value'].values.reshape(-1, 1).astype(np.float32)
        qi_xiang = df[['max_temp', 'min_temp', 'humidity', 'wind_speed']].values.astype(np.float32)
        shi_jian = df[_TIME_FEATURE_COLS].values.astype(np.float32)
        mu_biao = df['M019Value'].values.astype(np.float32)
        return fu_he, qi_xiang, shi_jian, mu_biao
    
    full_arrays = _extract_arrays(full_df)
    n_total = len(full_df)
    
    # 按时间顺序划分 K 段（扩展窗口：第 i 折用前 i+1 段训练，第 i+2 段验证）
    fold_size = n_total // n_splits
    mape_list = []
    
    for fold in range(n_splits - 1):
        train_end = (fold + 1) * fold_size
        val_start = train_end
        val_end = (fold + 2) * fold_size if fold < n_splits - 2 else n_total
        
        if train_end < xuLieChangDu + 100 or val_end - val_start < xuLieChangDu + 10:
            continue
        
        # 训练数据：0 到 train_end
        train_arrays = [arr[:train_end] for arr in full_arrays]
        val_arrays = [arr[val_start:val_end] for arr in full_arrays]
        
        train_ds = DianLiFuHeShuJuJi(*train_arrays, xuLieChangDu, yuCeBuChang)
        val_ds = DianLiFuHeShuJuJi(*val_arrays, xuLieChangDu, yuCeBuChang)
        
        p = best_params or {}
        batch = p.get('piCiDaXiao', piCiDaXiao)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
        moXing = TCNBiLstmZhuYiLiYuCeMoXing(
            fuHeTeZhengShu=1,
            qiXiangTeZhengShu=4,
            shiJianTeZhengShu=shiJianTeZhengShu,
            tcnYinCangWeiDu=p.get('tcnYinCangWeiDu', 64),
            tcnCengShu=p.get('tcnCengShu', 4),
            bilstmYinCangWeiDu=p.get('bilstmYinCangWeiDu', 64),
            bilstmCengShu=p.get('bilstmCengShu', 2),
            zhuYiLiTouShu=p.get('zhuYiLiTouShu', 4),
            yuCeBuChang=yuCeBuChang,
            tuoQiLv=p.get('tuoQiLv', 0.2)
        )
        
        print(f"\n--- 第 {fold+1}/{n_splits-1} 折: 训练 {train_end} 样本, 验证 {val_end-val_start} 样本 ---")
        xunLianLiShi, zuiJiaZhuangTai = xunLianMoXing(
            moXing, train_loader, val_loader,
            sheBei=sheBei, xueXiLv=p.get('xueXiLv', xueXiLv), xunLianDaiShu=xunLianDaiShu,
            scalers=scalers, lingChenQuanZhong=1.0
        )
        moXing.load_state_dict(zuiJiaZhuangTai)
        res = pingGuMoXing(moXing, val_loader, sheBei, scalers)
        mape = res[2]  # MAPE 为第 3 个返回值
        mape_list.append(mape)
        print(f"  第 {fold+1} 折 MAPE: {mape:.4f}%")
    
    mape_arr = np.array(mape_list)
    mean_mape = float(np.mean(mape_arr))
    std_mape = float(np.std(mape_arr))
    print(f"\n交叉验证完成: 平均 MAPE = {mean_mape:.4f}% ± {std_mape:.4f}%")
    return mape_list, mean_mape, std_mape

# -*- coding: utf-8 -*-
"""
特殊场景验证模块
Special Scenario Evaluation Module

开题报告要求：评估模型在极端天气、节假日等特殊场景下的适应性和稳定性
"""

import os
import numpy as np
import pandas as pd
import torch


def pingGuTeShuChangJing(moXing, jiaZaiQi, val_df, xuLieChangDu, sheBei, scalers):
    """
    按特殊场景子集评估 MAPE
    - 法定节假日
    - 极端高温（最高温>=35°C）
    - 极端高湿（湿度>=90%）
    
    参数:
        moXing: 已训练的模型
        jiaZaiQi: 验证集 DataLoader
        val_df: 验证集 DataFrame（含 is_holiday, is_extreme_heat, is_extreme_humidity）
        xuLieChangDu: 序列长度（用于对齐样本与行索引）
        sheBei: 设备
        scalers: 标准化器
    
    返回: dict {场景名: (样本数, MAPE)}
    """
    moXing.eval()
    yuCeLieBiao = []
    
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe = fuHe.to(sheBei)
            qiXiang = qiXiang.to(sheBei)
            shiJian = shiJian.to(sheBei)
            yuCe = moXing(fuHe, qiXiang, shiJian)
            yuCeLieBiao.append(yuCe.cpu().numpy())
    
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0).flatten()
    
    # 反标准化
    if scalers and 'M019Value' in scalers:
        yuCeZhi = scalers['M019Value'].inverse_transform(yuCeZhi.reshape(-1, 1)).flatten()
    
    # 每个样本对应 val_df 中的行索引
    n_samples = len(yuCeZhi)
    # 样本 j 预测的是 val_df 第 xuLieChangDu + j 行的目标值
    row_indices = np.arange(xuLieChangDu, xuLieChangDu + n_samples)
    
    # 获取真实值（从 val_df 的 M019Value，需反标准化）
    if scalers and 'M019Value' in scalers:
        zhenShiZhi = scalers['M019Value'].inverse_transform(
            val_df['M019Value'].values[row_indices].reshape(-1, 1)
        ).flatten()
    else:
        zhenShiZhi = val_df['M019Value'].values[row_indices]
    
    # 获取场景标记
    if 'is_holiday' not in val_df.columns:
        return {}
    
    is_holiday = val_df['is_holiday'].values[row_indices]
    is_extreme_heat = val_df['is_extreme_heat'].values[row_indices] if 'is_extreme_heat' in val_df.columns else np.zeros_like(is_holiday)
    is_extreme_humidity = val_df['is_extreme_humidity'].values[row_indices] if 'is_extreme_humidity' in val_df.columns else np.zeros_like(is_holiday)
    
    def _mape(mask, eps=1e-8):
        if mask.sum() < 10:
            return None
        return np.mean(np.abs((zhenShiZhi[mask] - yuCeZhi[mask]) / (zhenShiZhi[mask] + eps))) * 100
    
    results = {}
    
    # 法定节假日
    mask_hol = is_holiday == 1
    if mask_hol.sum() >= 10:
        results['法定节假日'] = (int(mask_hol.sum()), _mape(mask_hol))
    
    # 非节假日
    mask_non = is_holiday == 0
    if mask_non.sum() >= 10:
        results['非节假日'] = (int(mask_non.sum()), _mape(mask_non))
    
    # 极端高温
    mask_heat = is_extreme_heat == 1
    if mask_heat.sum() >= 10:
        results['极端高温(≥35°C)'] = (int(mask_heat.sum()), _mape(mask_heat))
    
    # 极端高湿
    mask_hum = is_extreme_humidity == 1
    if mask_hum.sum() >= 10:
        results['极端高湿(≥90%)'] = (int(mask_hum.sum()), _mape(mask_hum))
    
    return results

# -*- coding: utf-8 -*-
"""
训练流程模块
Training Pipeline Module

根据开题报告：
- 实现数据加载、模型定义、训练循环及验证评估等核心功能模块
- 采用MSE、MAE、MAPE作为评估指标
- 通过优化模型结构和超参数，解决过拟合和收敛缓慢等问题
- 采用学习率调度和正则化等技术
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time


def jiSuanRMSE(yuCeZhi, zhenShiZhi):
    """计算均方根误差 RMSE"""
    return float(np.sqrt(np.mean((np.array(yuCeZhi).flatten() - np.array(zhenShiZhi).flatten()) ** 2)))


def jiSuanNRMSE(yuCeZhi, zhenShiZhi, epsilon=1e-8):
    """
    计算归一化均方根误差 NRMSE（百分比形式）
    NRMSE(%) = RMSE / mean(|y_true|) * 100
    与 MAPE 量级可比，对应论文中 RMSE(%) 指标
    """
    rmse = jiSuanRMSE(yuCeZhi, zhenShiZhi)
    mean_true = np.mean(np.abs(np.array(zhenShiZhi).flatten()))
    return rmse / (mean_true + epsilon) * 100


def jiSuanMAPE(yuCeZhi, zhenShiZhi, epsilon=1e-8):
    """
    计算平均绝对百分比误差 MAPE
    JiSuan = 计算
    
    参数:
        yuCeZhi: 预测值
        zhenShiZhi: 真实值
        epsilon: 防止除零的小常数
        
    返回:
        MAPE值（百分比形式）
    """
    yuCeZhi = np.array(yuCeZhi).flatten()
    zhenShiZhi = np.array(zhenShiZhi).flatten()
    return np.mean(np.abs((zhenShiZhi - yuCeZhi) / (zhenShiZhi + epsilon))) * 100


def jiSuanMAE(yuCeZhi, zhenShiZhi):
    """计算平均绝对误差 MAE"""
    return np.mean(np.abs(np.array(yuCeZhi).flatten() - np.array(zhenShiZhi).flatten()))


def jiSuanMSE(yuCeZhi, zhenShiZhi):
    """计算均方误差 MSE"""
    return np.mean((np.array(yuCeZhi).flatten() - np.array(zhenShiZhi).flatten()) ** 2)


def jiSuanR2(yuCeZhi, zhenShiZhi):
    """计算决定系数 R² (R-squared)"""
    yuCeZhi = np.array(yuCeZhi).flatten()
    zhenShiZhi = np.array(zhenShiZhi).flatten()
    ss_res = np.sum((zhenShiZhi - yuCeZhi) ** 2)
    ss_tot = np.sum((zhenShiZhi - np.mean(zhenShiZhi)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-10))


def jiSuanWAPE(yuCeZhi, zhenShiZhi, epsilon=1e-8):
    """
    计算加权绝对百分比误差 WAPE (Weighted Absolute Percentage Error)
    WAPE = sum(|y_true - y_pred|) / sum(|y_true|) * 100
    对小值不敏感，适合负荷预测
    """
    yuCeZhi = np.array(yuCeZhi).flatten()
    zhenShiZhi = np.array(zhenShiZhi).flatten()
    return np.sum(np.abs(zhenShiZhi - yuCeZhi)) / (np.sum(np.abs(zhenShiZhi)) + epsilon) * 100


def jiSuanSMAPE(yuCeZhi, zhenShiZhi, epsilon=1e-8):
    """
    计算对称平均绝对百分比误差 SMAPE (Symmetric MAPE)
    SMAPE = mean(2 * |y_pred - y_true| / (|y_pred| + |y_true|)) * 100
    有界 [0, 100]，对高估和低估更对称
    """
    yuCeZhi = np.array(yuCeZhi).flatten()
    zhenShiZhi = np.array(zhenShiZhi).flatten()
    return np.mean(2 * np.abs(yuCeZhi - zhenShiZhi) / (np.abs(yuCeZhi) + np.abs(zhenShiZhi) + epsilon)) * 100


def pingGuMoXing(moXing, jiaZaiQi, sheBei, scalers=None):
    """
    评估模型性能
    PingGu = 评估, MoXing = 模型
    
    参数:
        scalers: 标准化器字典，若提供则对负荷预测结果反标准化后计算 MAPE（原始尺度更符合工程意义）
    
    返回: (MSE, MAE, MAPE, R², SMAPE, WAPE)
    """
    moXing.eval()
    yuCeLieBiao = []
    zhenShiLieBiao = []
    
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe = fuHe.to(sheBei)
            qiXiang = qiXiang.to(sheBei)
            shiJian = shiJian.to(sheBei)
            muBiao = muBiao.to(sheBei)
            
            yuCe = moXing(fuHe, qiXiang, shiJian)
            yuCeLieBiao.append(yuCe.cpu().numpy())
            zhenShiLieBiao.append(muBiao.cpu().numpy())
    
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0)
    zhenShiZhi = np.concatenate(zhenShiLieBiao, axis=0)
    
    # 若提供 scalers，反标准化后计算（原始尺度更符合工程意义）
    if scalers is not None and 'M019Value' in scalers:
        scaler = scalers['M019Value']
        yuCeZhi_calc = scaler.inverse_transform(yuCeZhi)
        zhenShiZhi_calc = scaler.inverse_transform(zhenShiZhi)
    else:
        yuCeZhi_calc = yuCeZhi
        zhenShiZhi_calc = zhenShiZhi

    mse = jiSuanMSE(yuCeZhi_calc, zhenShiZhi_calc)
    mae = jiSuanMAE(yuCeZhi_calc, zhenShiZhi_calc)
    mape = jiSuanMAPE(yuCeZhi_calc, zhenShiZhi_calc)
    r2 = jiSuanR2(yuCeZhi_calc, zhenShiZhi_calc)
    smape = jiSuanSMAPE(yuCeZhi_calc, zhenShiZhi_calc)
    wape = jiSuanWAPE(yuCeZhi_calc, zhenShiZhi_calc)
    nrmse = jiSuanNRMSE(yuCeZhi_calc, zhenShiZhi_calc)

    return mse, mae, mape, r2, smape, wape, nrmse


def _jiSuanShiDuanQuanZhong(shiJian, lingChenQuanZhong=2.0):
    """
    根据时间特征计算样本权重，对凌晨(0-6点)样本加大权重以改善凌晨MAPE
    shiJian: (batch, seq, 5) 含 hour_sin, hour_cos 于 0,1 位
    返回: (batch,) 权重，凌晨样本权重为 lingChenQuanZhong，其余为 1.0
    """
    h_sin = shiJian[:, -1, 0].cpu().numpy()
    h_cos = shiJian[:, -1, 1].cpu().numpy()
    hour_rad = np.arctan2(h_sin, h_cos)
    hour = (hour_rad * 12 / np.pi) % 24
    hour = np.clip(hour, 0, 23.99).astype(int)
    # 0-6点为凌晨（含0点不含6点）
    is_ling_chen = (hour >= 0) & (hour < 6)
    quan_zhong = np.where(is_ling_chen, lingChenQuanZhong, 1.0).astype(np.float32)
    return torch.from_numpy(quan_zhong).to(shiJian.device)


def xunLianMoXing(
    moXing,
    xunLianJiaZaiQi,
    yanZhengJiaZaiQi,
    sheBei='cuda',
    xueXiLv=0.001,
    xunLianDaiShu=50,
    scalers=None,
    lingChenQuanZhong=1.0,
    zaoZhiTingZhiPatience=10,
    mape_min_delta=0.005,
    loss_min_delta_rel=0.005,
    per_epoch_callback=None,
    quiet=False
):
    """
    训练模型主流程
    XunLian = 训练, MoXing = 模型
    
    参数:
        moXing: TCN-BiLSTM-Attention模型
        xunLianJiaZaiQi: 训练数据加载器
        yanZhengJiaZaiQi: 验证数据加载器
        sheBei: 计算设备
        xueXiLv: 初始学习率
        xunLianDaiShu: 训练轮数
        lingChenQuanZhong: 凌晨(0-6点)样本损失权重，>1 可改善凌晨MAPE，默认1.0不启用
        zaoZhiTingZhiPatience: 早停耐心值：验证 MAPE 与 Loss **同时** 连续 N 轮无「有意义」改善才结束；0 表示禁用
        mape_min_delta: 有意义改善的 MAPE 阈值（百分点，如 0.005 即 0.005%）
        loss_min_delta_rel: 有意义改善的 Loss 相对阈值（如 0.005 即 0.5% 相对下降）
        per_epoch_callback: 每轮结束回调 (epoch, val_loss, val_mape)，可抛 TrialPruned 实现 Optuna 剪枝
        quiet: 为 True 时不打印每轮信息（用于 Optuna 等批量训练）
        
    返回:
        (训练历史, 最佳模型状态)
    """
    moXing = moXing.to(sheBei)
    sunShiHanShu = nn.MSELoss(reduction='none') if lingChenQuanZhong > 1.0 else nn.MSELoss()
    youHuaQi = optim.Adam(moXing.parameters(), lr=xueXiLv)
    # 学习率调度：验证损失不下降时降低学习率
    xueXiLvDiaoDu = ReduceLROnPlateau(
        youHuaQi, mode='min', factor=0.5, patience=5
    )
    
    zuiJiaYanZhengLoss = float('inf')
    zuiJiaMAPE = float('inf')          # 历史最低 MAPE（用于保存权重）
    # MAPE 早停计数
    mape_can_kao = float('inf')        # 早停参考：仅当 MAPE 相对其下降 ≥ mape_min_delta 才重置计数
    weiGaiShanJiShu = 0                # MAPE 连续无改善轮数
    # Loss 早停计数（双重依据：loss 与 MAPE 同时停滞才真正早停）
    loss_can_kao = float('inf')        # Loss 早停参考
    loss_weiGaiShan = 0                # Loss 连续无改善轮数
    zuiJiaMoXingZhuangTai = None
    xunLianLiShi = {'xunLianLoss': [], 'yanZhengLoss': [], 'yanZhengMAPE': []}

    for dai in range(xunLianDaiShu):
        if not quiet:
            print(f"[TCN-BiLSTM-Attention] 第{dai+1}轮训练中...", end=" ", flush=True)
        moXing.train()
        daiXunLianLoss = 0.0

        for fuHe, qiXiang, shiJian, muBiao in xunLianJiaZaiQi:
            fuHe = fuHe.to(sheBei)
            qiXiang = qiXiang.to(sheBei)
            shiJian = shiJian.to(sheBei)
            muBiao = muBiao.to(sheBei)

            youHuaQi.zero_grad()
            yuCe = moXing(fuHe, qiXiang, shiJian)
            if lingChenQuanZhong > 1.0:
                loss_per = sunShiHanShu(yuCe, muBiao)
                quan_zhong = _jiSuanShiDuanQuanZhong(shiJian, lingChenQuanZhong)
                loss = (loss_per.squeeze(-1) * quan_zhong).mean()
            else:
                loss = sunShiHanShu(yuCe, muBiao)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moXing.parameters(), max_norm=1.0)
            youHuaQi.step()

            daiXunLianLoss += loss.item()

        pingJunXunLianLoss = daiXunLianLoss / len(xunLianJiaZaiQi)
        xunLianLiShi['xunLianLoss'].append(pingJunXunLianLoss)

        # 验证：Loss 用于学习率调度，MAPE 用于早停和保存最优模型
        moXing.eval()
        yanZhengSunShi = nn.MSELoss()
        yanZhengLossZong = 0.0
        with torch.no_grad():
            for fuHe, qiXiang, shiJian, muBiao in yanZhengJiaZaiQi:
                fuHe = fuHe.to(sheBei)
                qiXiang = qiXiang.to(sheBei)
                shiJian = shiJian.to(sheBei)
                muBiao = muBiao.to(sheBei)
                yuCe = moXing(fuHe, qiXiang, shiJian)
                yanZhengLossZong += yanZhengSunShi(yuCe, muBiao).item()

        pingJunYanZhengLoss = yanZhengLossZong / len(yanZhengJiaZaiQi)
        res = pingGuMoXing(moXing, yanZhengJiaZaiQi, sheBei, scalers)
        yanZhengMAPE = res[2]
        yanZhengSMAPE = res[4] if len(res) >= 5 else yanZhengMAPE
        yanZhengWAPE  = res[5] if len(res) >= 6 else yanZhengMAPE

        xunLianLiShi['yanZhengLoss'].append(pingJunYanZhengLoss)
        xunLianLiShi['yanZhengMAPE'].append(yanZhengMAPE)

        # 学习率调度仍用 Loss（更平滑，避免 MAPE 噪声影响 LR）
        xueXiLvDiaoDu.step(pingJunYanZhengLoss)

        if per_epoch_callback is not None:
            per_epoch_callback(dai + 1, pingJunYanZhengLoss,
                               {'mape': yanZhengMAPE, 'smape': yanZhengSMAPE, 'wape': yanZhengWAPE})

        if not quiet:
            best_mark = ' ★' if yanZhengMAPE < zuiJiaMAPE else ''
            print(f"第{dai+1}轮 - 训练Loss: {pingJunXunLianLoss:.6f}, "
                  f"验证Loss: {pingJunYanZhengLoss:.6f}, 验证MAPE: {yanZhengMAPE:.4f}%{best_mark}")

        # 保存权重：任意严格更低的验证 MAPE 即更新（与早停计数解耦）
        if yanZhengMAPE < zuiJiaMAPE:
            zuiJiaMAPE = yanZhengMAPE
            zuiJiaYanZhengLoss = pingJunYanZhengLoss
            zuiJiaMoXingZhuangTai = {k: v.cpu().clone() for k, v in moXing.state_dict().items()}

        # MAPE 早停计数：仅当 MAPE 相对参考值下降 ≥ mape_min_delta 才算改善
        if mape_can_kao == float('inf') or yanZhengMAPE < mape_can_kao - mape_min_delta:
            mape_can_kao = yanZhengMAPE
            weiGaiShanJiShu = 0
        else:
            weiGaiShanJiShu += 1

        # Loss 早停计数：仅当 Loss 相对参考值下降 ≥ loss_min_delta_rel 才算改善
        if loss_can_kao == float('inf') or pingJunYanZhengLoss < loss_can_kao * (1.0 - loss_min_delta_rel):
            loss_can_kao = pingJunYanZhengLoss
            loss_weiGaiShan = 0
        else:
            loss_weiGaiShan += 1

        # 双重早停：MAPE 与 Loss 同时停滞才结束（避免 MAPE 噪声导致过早停止）
        if zaoZhiTingZhiPatience > 0 and weiGaiShanJiShu >= zaoZhiTingZhiPatience and loss_weiGaiShan >= zaoZhiTingZhiPatience:
            print(
                f"\n[早停] 验证MAPE（连续{weiGaiShanJiShu}轮）与Loss（连续{loss_weiGaiShan}轮）同时无显著改善，"
                f"于第{dai+1}轮提前结束（已保存权重对应历史最优 MAPE={zuiJiaMAPE:.4f}%）"
            )
            break

    return xunLianLiShi, zuiJiaMoXingZhuangTai


def xunLianDuiBiMoXing(
    moXing,
    xunLianJiaZaiQi,
    yanZhengJiaZaiQi,
    sheBei='cuda',
    xueXiLv=0.001,
    xunLianDaiShu=50,
    scalers=None,
    zaoZhiTingZhiPatience=10,
    mape_min_delta=0.005,
    loss_min_delta_rel=0.005,
    quiet=False,
    label='对比模型',
):
    """
    训练对比基线模型（与 xunLianMoXing 接口兼容，但不含凌晨加权等主模型专属逻辑）。
    返回 (训练历史, 最佳模型状态字典, 测试指标元组)
    """
    moXing = moXing.to(sheBei)
    sunShiHanShu = nn.MSELoss()
    youHuaQi = optim.Adam(moXing.parameters(), lr=xueXiLv)
    xueXiLvDiaoDu = ReduceLROnPlateau(youHuaQi, mode='min', factor=0.5, patience=5)

    zuiJiaMAPE = float('inf')
    mape_can_kao = float('inf')
    zuiJiaYanZhengLoss = float('inf')
    weiGaiShanJiShu = 0
    loss_can_kao = float('inf')
    loss_weiGaiShan = 0
    zuiJiaMoXingZhuangTai = None
    xunLianLiShi = {'xunLianLoss': [], 'yanZhengLoss': [], 'yanZhengMAPE': []}
    zaoTingTingZao = False

    for dai in range(xunLianDaiShu):
        moXing.train()
        daiLoss = 0.0
        for fuHe, qiXiang, shiJian, muBiao in xunLianJiaZaiQi:
            fuHe, qiXiang, shiJian, muBiao = (
                fuHe.to(sheBei), qiXiang.to(sheBei), shiJian.to(sheBei), muBiao.to(sheBei)
            )
            youHuaQi.zero_grad()
            yuCe = moXing(fuHe, qiXiang, shiJian)
            loss = sunShiHanShu(yuCe, muBiao)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(moXing.parameters(), max_norm=1.0)
            youHuaQi.step()
            daiLoss += loss.item()

        pingJunXunLianLoss = daiLoss / len(xunLianJiaZaiQi)
        xunLianLiShi['xunLianLoss'].append(pingJunXunLianLoss)

        moXing.eval()
        yanZhengLossZong = 0.0
        yanZhengSunShi = nn.MSELoss()
        with torch.no_grad():
            for fuHe, qiXiang, shiJian, muBiao in yanZhengJiaZaiQi:
                fuHe, qiXiang, shiJian, muBiao = (
                    fuHe.to(sheBei), qiXiang.to(sheBei), shiJian.to(sheBei), muBiao.to(sheBei)
                )
                yanZhengLossZong += yanZhengSunShi(moXing(fuHe, qiXiang, shiJian), muBiao).item()

        pingJunYanZhengLoss = yanZhengLossZong / len(yanZhengJiaZaiQi)
        res = pingGuMoXing(moXing, yanZhengJiaZaiQi, sheBei, scalers)
        yanZhengMAPE = res[2]

        xunLianLiShi['yanZhengLoss'].append(pingJunYanZhengLoss)
        xunLianLiShi['yanZhengMAPE'].append(yanZhengMAPE)

        # 学习率调度用 Loss（更平滑）
        xueXiLvDiaoDu.step(pingJunYanZhengLoss)

        if not quiet:
            best_mark = ' ★' if yanZhengMAPE < zuiJiaMAPE else ''
            print(
                f"  [{label}] 第{dai+1}/{xunLianDaiShu}轮 - "
                f"训练Loss: {pingJunXunLianLoss:.6f}, 验证MAPE: {yanZhengMAPE:.4f}%{best_mark}",
                flush=True,
            )

        if yanZhengMAPE < zuiJiaMAPE:
            zuiJiaMAPE = yanZhengMAPE
            zuiJiaYanZhengLoss = pingJunYanZhengLoss
            zuiJiaMoXingZhuangTai = {k: v.cpu().clone() for k, v in moXing.state_dict().items()}

        if mape_can_kao == float('inf') or yanZhengMAPE < mape_can_kao - mape_min_delta:
            mape_can_kao = yanZhengMAPE
            weiGaiShanJiShu = 0
        else:
            weiGaiShanJiShu += 1

        if loss_can_kao == float('inf') or pingJunYanZhengLoss < loss_can_kao * (1.0 - loss_min_delta_rel):
            loss_can_kao = pingJunYanZhengLoss
            loss_weiGaiShan = 0
        else:
            loss_weiGaiShan += 1

        # 双重早停：MAPE 与 Loss 同时停滞才结束
        if zaoZhiTingZhiPatience > 0 and weiGaiShanJiShu >= zaoZhiTingZhiPatience and loss_weiGaiShan >= zaoZhiTingZhiPatience:
            zaoTingTingZao = True
            print(
                f"\n[{label} 早停] 验证MAPE（连续{weiGaiShanJiShu}轮）与Loss（连续{loss_weiGaiShan}轮）同时无显著改善，"
                f"于第{dai+1}轮结束（已保存历史最优 MAPE={zuiJiaMAPE:.4f}%）"
            )
            break

    # quiet 时每轮不打印；若未触发早停则整段无输出，结束时补一行便于与后续步骤衔接
    if quiet and not zaoTingTingZao:
        n_lun = len(xunLianLiShi.get('yanZhengMAPE', []))
        print(
            f"  [{label}] 训练结束，验证集历史最优 MAPE={zuiJiaMAPE:.4f}%（共 {n_lun} 轮，未触发早停）",
            flush=True,
        )

    return xunLianLiShi, zuiJiaMoXingZhuangTai


def ceShiYuCeXiangYingShiJian(moXing, shiLiShuJu, sheBei, chongFuCiShu=100):
    """
    测试单次预测响应时间
    CeShi = 测试, YuCe = 预测, XiangYing = 响应, ShiJian = 时间
    
    目标：单次预测响应时间 ≤ 1分钟
    shiLiShuJu: (fuHe, qiXiang, shiJian) 或 (fuHe, qiXiang, shiJian, _)
    """
    moXing.eval()
    fuHe, qiXiang, shiJian = shiLiShuJu[0], shiLiShuJu[1], shiLiShuJu[2]
    
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = moXing(
                fuHe.unsqueeze(0).to(sheBei),
                qiXiang.unsqueeze(0).to(sheBei),
                shiJian.unsqueeze(0).to(sheBei)
            )
        
        # 计时
        if sheBei == 'cuda':
            torch.cuda.synchronize()
        
        qiShiShiJian = time.time()
        for _ in range(chongFuCiShu):
            _ = moXing(
                fuHe.unsqueeze(0).to(sheBei),
                qiXiang.unsqueeze(0).to(sheBei),
                shiJian.unsqueeze(0).to(sheBei)
            )
        
        if sheBei == 'cuda':
            torch.cuda.synchronize()
        
        jieShuShiJian = time.time()
    
    pingJunShiJian = (jieShuShiJian - qiShiShiJian) / chongFuCiShu * 1000  # 毫秒
    return pingJunShiJian

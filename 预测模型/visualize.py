# -*- coding: utf-8 -*-
"""
模型评估可视化模块
Model Evaluation Visualization

生成训练曲线、预测对比、误差分布等图表
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def huaXunLianQuXian(xunLianLiShi, baoCunLuJing):
    """
    绘制训练/验证 Loss 与 MAPE 曲线
    Hua = 绘制, XunLian = 训练, QuXian = 曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(xunLianLiShi['xunLianLoss']) + 1)
    
    axes[0].plot(epochs, xunLianLiShi['xunLianLoss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, xunLianLiShi['yanZhengLoss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, xunLianLiShi['yanZhengMAPE'], 'g-', label='Val MAPE (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Validation MAPE (Original Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaYuCeDuiBi(moXing, jiaZaiQi, sheBei, scalers, baoCunLuJing, yangBenShu=500):
    """
    绘制预测值 vs 真实值 散点图 + 时序对比
    """
    moXing.eval()
    yuCeLieBiao, zhenShiLieBiao = [], []
    
    import torch
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe = fuHe.to(sheBei)
            qiXiang = qiXiang.to(sheBei)
            shiJian = shiJian.to(sheBei)
            yuCe = moXing(fuHe, qiXiang, shiJian)
            yuCeLieBiao.append(yuCe.cpu().numpy())
            zhenShiLieBiao.append(muBiao.cpu().numpy())
    
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0).flatten()
    zhenShiZhi = np.concatenate(zhenShiLieBiao, axis=0).flatten()
    
    if scalers and 'M019Value' in scalers:
        yuCeZhi = scalers['M019Value'].inverse_transform(yuCeZhi.reshape(-1, 1)).flatten()
        zhenShiZhi = scalers['M019Value'].inverse_transform(zhenShiZhi.reshape(-1, 1)).flatten()
    
    n = min(yangBenShu, len(yuCeZhi))
    yuCeZhi, zhenShiZhi = yuCeZhi[:n], zhenShiZhi[:n]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 散点图
    axes[0].scatter(zhenShiZhi, yuCeZhi, alpha=0.5, s=10)
    minv = min(zhenShiZhi.min(), yuCeZhi.min())
    maxv = max(zhenShiZhi.max(), yuCeZhi.max())
    axes[0].plot([minv, maxv], [minv, maxv], 'r--', label='Ideal y=x')
    axes[0].set_xlabel('Actual Load (MW)')
    axes[0].set_ylabel('Predicted Load (MW)')
    axes[0].set_title('Predicted vs. Actual Scatter')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 时序对比（取前 500 点）
    x = np.arange(n)
    axes[1].plot(x, zhenShiZhi, 'b-', alpha=0.7, label='Actual')
    axes[1].plot(x, yuCeZhi, 'r-', alpha=0.7, label='Predicted')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Load (MW)')
    axes[1].set_title('Predicted vs. Actual Time Series')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def _huoQuYuCeZhenShi(moXing, jiaZaiQi, sheBei, scalers):
    """获取预测值与真实值（反标准化后）"""
    moXing.eval()
    yuCeLieBiao, zhenShiLieBiao = [], []
    import torch
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe, qiXiang, shiJian = fuHe.to(sheBei), qiXiang.to(sheBei), shiJian.to(sheBei)
            yuCe = moXing(fuHe, qiXiang, shiJian)
            yuCeLieBiao.append(yuCe.cpu().numpy())
            zhenShiLieBiao.append(muBiao.cpu().numpy())
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0).flatten()
    zhenShiZhi = np.concatenate(zhenShiLieBiao, axis=0).flatten()
    if scalers and 'M019Value' in scalers:
        yuCeZhi = scalers['M019Value'].inverse_transform(yuCeZhi.reshape(-1, 1)).flatten()
        zhenShiZhi = scalers['M019Value'].inverse_transform(zhenShiZhi.reshape(-1, 1)).flatten()
    return yuCeZhi, zhenShiZhi


def huaWuChaFenBu(moXing, jiaZaiQi, sheBei, scalers, baoCunLuJing):
    """
    绘制误差分布：残差直方图、残差 vs 预测值
    """
    yuCeZhi, zhenShiZhi = _huoQuYuCeZhenShi(moXing, jiaZaiQi, sheBei, scalers)
    wuCha = zhenShiZhi - yuCeZhi
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(wuCha, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='r', linestyle='--')
    axes[0].set_xlabel('Residual (Actual - Predicted) MW')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Residual Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(yuCeZhi, wuCha, alpha=0.3, s=5)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Load (MW)')
    axes[1].set_ylabel('Residual (MW)')
    axes[1].set_title('Residual vs. Predicted')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaWuChaZhiFangTuHeMiDu(moXing, jiaZaiQi, sheBei, scalers, baoCunLuJing):
    """误差分布：直方图 + 核密度估计 (KDE)"""
    yuCeZhi, zhenShiZhi = _huoQuYuCeZhenShi(moXing, jiaZaiQi, sheBei, scalers)
    wuCha = zhenShiZhi - yuCeZhi
    
    from scipy import stats as scipy_stats
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(wuCha, bins=60, density=True, alpha=0.6, color='#3498db', edgecolor='black', label='Histogram')
    try:
        kde = scipy_stats.gaussian_kde(wuCha)
        x_kde = np.linspace(wuCha.min(), wuCha.max(), 200)
        ax.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')
    except Exception:
        pass
    ax.axvline(0, color='green', linestyle='--', alpha=0.8)
    ax.set_xlabel('Residual (Actual - Predicted) MW')
    ax.set_ylabel('Density')
    ax.set_title('Residual Distribution & KDE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaYuCeZhunQueDuJuZhen(moXing, jiaZaiQi, sheBei, scalers, baoCunLuJing, n_bins=5):
    """
    预测准确度矩阵（回归版混淆矩阵）
    将真实值与预测值分箱，展示各区间样本分布
    """
    yuCeZhi, zhenShiZhi = _huoQuYuCeZhenShi(moXing, jiaZaiQi, sheBei, scalers)
    all_vals = np.concatenate([zhenShiZhi, yuCeZhi])
    bins = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.array([all_vals.min() - 1e-6, all_vals.max() + 1e-6])
    row_idx = np.digitize(zhenShiZhi, bins) - 1
    col_idx = np.digitize(yuCeZhi, bins) - 1
    n_actual = len(bins) - 1
    row_idx = np.clip(row_idx, 0, n_actual - 1)
    col_idx = np.clip(col_idx, 0, n_actual - 1)
    mat = np.zeros((n_actual, n_actual))
    for r, c in zip(row_idx, col_idx):
        mat[r, c] += 1
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap='Blues', aspect='auto')
    labels = [f'Q{i+1}' for i in range(n_actual)]
    ax.set_xticks(range(n_actual))
    ax.set_yticks(range(n_actual))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted Load Quantile')
    ax.set_ylabel('Actual Load Quantile')
    ax.set_title('Prediction Accuracy Matrix (Actual vs. Predicted)')
    for i in range(n_actual):
        for j in range(n_actual):
            ax.text(j, i, f'{int(mat[i,j])}', ha='center', va='center', color='black' if mat[i,j] < mat.max()/2 else 'white')
    plt.colorbar(im, ax=ax, label='Count')
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaPingJiaZhiBiaoYiBiaoPan(mse, mae, mape, r2, mapeJiChu, baoCunLuJing, smape=None, wape=None, acc=None):
    """关键评价指标仪表盘"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    def _draw_gauge(ax, value, title, max_val, color, unit=''):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        # 半圆仪表
        theta = np.linspace(0, np.pi, 100)
        ax.plot(0.5 + 0.4 * np.cos(theta), 0.5 + 0.4 * np.sin(theta), 'k-', lw=2)
        pct = min(value / max_val, 1.0) if max_val > 0 else 0
        angle = np.pi * (1 - pct)
        ax.plot([0.5, 0.5 + 0.35 * np.cos(angle)], [0.5, 0.5 + 0.35 * np.sin(angle)], color=color, lw=4)
        ax.text(0.5, 0.15, f'{value:.4f}{unit}', ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.05, title, ha='center', fontsize=11)
    
    # 为各指标设定合理显示范围
    _draw_gauge(axes[0, 0], mse, 'MSE', max(mse * 2, 10), '#3498db')
    _draw_gauge(axes[0, 1], mae, 'MAE (MW)', max(mae * 2, 5), '#2ecc71')
    _draw_gauge(axes[1, 0], mape, 'MAPE (%)', max(mape * 2, 5), '#e74c3c', '%')
    # R² 范围 0~1
    r2_display = max(0, min(r2, 1))
    _draw_gauge(axes[1, 1], r2_display, 'R²', 1.0, '#9b59b6')
    
    title_parts = [f'MAPE: {mape:.4f}%']
    if acc is not None:
        title_parts.append(f'Acc: {acc:.4f}%')
    if smape is not None and wape is not None:
        title_parts.extend([f'SMAPE: {smape:.4f}%', f'WAPE: {wape:.4f}%'])
    if mapeJiChu is not None and mapeJiChu > 0:
        title_parts.append(f'Baseline MAPE: {mapeJiChu:.4f}%')
    fig.suptitle('Key Metrics (' + ' | '.join(title_parts) + ')', fontsize=11)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaPingGuZongJie(mse, mae, mape, mapeJiChu, baoCunLuJing, smape=None, wape=None, acc=None):
    """
    绘制评估指标总结（柱状图）
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    labels = ['MSE', 'MAE', 'MAPE (%)']
    values = [mse, mae, mape]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    if acc is not None:
        labels.append('Acc (%)')
        values.append(acc)
        colors.append('#1abc9c')
    if smape is not None and wape is not None:
        labels.extend(['SMAPE (%)', 'WAPE (%)'])
        values.extend([smape, wape])
        colors.extend(['#9b59b6', '#f39c12'])
    if mapeJiChu is not None and mapeJiChu > 0:
        labels.append('Baseline MAPE (%)')
        values.append(mapeJiChu)
        colors.append('#95a5a6')
    
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel('Value')
    ax.set_title('Model Evaluation Metrics Summary')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                f'{v:.4f}' if v < 100 else f'{v:.2f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaAnXiaoShiMAPE(moXing, jiaZaiQi, sheBei, scalers, baoCunLuJing):
    """
    按小时统计 MAPE（0-23 点）
    从 shiJian 的 hour_sin, hour_cos 反推小时
    """
    moXing.eval()
    yuCeLieBiao, zhenShiLieBiao, xiaoShiLieBiao = [], [], []
    
    import torch
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe = fuHe.to(sheBei)
            qiXiang = qiXiang.to(sheBei)
            shiJian = shiJian.to(sheBei)
            yuCe = moXing(fuHe, qiXiang, shiJian)
            # shiJian: [batch, seq, 5] 含 hour_sin, hour_cos，取最后一个时间步
            h_sin = shiJian[:, -1, 0].cpu().numpy()
            h_cos = shiJian[:, -1, 1].cpu().numpy()
            hour_rad = np.arctan2(h_sin, h_cos)
            hour = (hour_rad * 24 / (2 * np.pi)) % 24
            hour = np.clip(hour.astype(int), 0, 23)
            xiaoShiLieBiao.extend(hour.tolist())
            yuCeLieBiao.append(yuCe.cpu().numpy())
            zhenShiLieBiao.append(muBiao.cpu().numpy())
    
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0).flatten()
    zhenShiZhi = np.concatenate(zhenShiLieBiao, axis=0).flatten()
    xiaoShiArr = np.array(xiaoShiLieBiao)
    
    if scalers and 'M019Value' in scalers:
        yuCeZhi = scalers['M019Value'].inverse_transform(yuCeZhi.reshape(-1, 1)).flatten()
        zhenShiZhi = scalers['M019Value'].inverse_transform(zhenShiZhi.reshape(-1, 1)).flatten()
    
    mape_by_hour = []
    for h in range(24):
        mask = xiaoShiArr == h
        if mask.sum() > 0:
            m = np.mean(np.abs((zhenShiZhi[mask] - yuCeZhi[mask]) / (zhenShiZhi[mask] + 1e-8))) * 100
            mape_by_hour.append(m)
        else:
            mape_by_hour.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(24), mape_by_hour, color='#3498db', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Hour')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE by Hour (0-23)')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


RESULT_DIR_NAME = '模型预测结果'

# ──────────────────────────────────────────────────────────────
# 多模型对比可视化（仿照论文图6/7风格）
# ──────────────────────────────────────────────────────────────

# 模型显示颜色与线型配置（与论文图6颜色风格对应）
_MODEL_STYLES = {
    'TCN-BiLSTM-Attention': {'color': '#000000', 'linestyle': '-',  'linewidth': 2.0, 'label': 'TCN-BiLSTM-Attention'},
    'BiLSTM-Attention':     {'color': '#3b5ea6', 'linestyle': '-',  'linewidth': 1.5, 'label': 'BiLSTM-Attention'},
    'TCN':                   {'color': '#ffa500', 'linestyle': '-',  'linewidth': 1.5, 'label': 'TCN'},
    'BiLSTM':                {'color': '#27ae60', 'linestyle': '-',  'linewidth': 1.5, 'label': 'BiLSTM'},
    'LSTM':                  {'color': '#e74c3c', 'linestyle': '--', 'linewidth': 1.2, 'label': 'LSTM'},
    '真实值':                {'color': '#e74c3c', 'linestyle': '-',  'linewidth': 2.0, 'label': 'Actual'},
}


def _huoQuYuCeZhenShiForModel(moXing, jiaZaiQi, sheBei, scalers):
    """获取模型在 jiaZaiQi 上的预测值与真实值（反标准化后）"""
    moXing.eval()
    yuCeLieBiao, zhenShiLieBiao = [], []
    with torch.no_grad():
        for fuHe, qiXiang, shiJian, muBiao in jiaZaiQi:
            fuHe, qiXiang, shiJian = fuHe.to(sheBei), qiXiang.to(sheBei), shiJian.to(sheBei)
            yuCe = moXing(fuHe, qiXiang, shiJian)
            yuCeLieBiao.append(yuCe.cpu().numpy())
            zhenShiLieBiao.append(muBiao.numpy())
    yuCeZhi = np.concatenate(yuCeLieBiao, axis=0).flatten()
    zhenShiZhi = np.concatenate(zhenShiLieBiao, axis=0).flatten()
    if scalers and 'M019Value' in scalers:
        yuCeZhi = scalers['M019Value'].inverse_transform(yuCeZhi.reshape(-1, 1)).flatten()
        zhenShiZhi = scalers['M019Value'].inverse_transform(zhenShiZhi.reshape(-1, 1)).flatten()
    return yuCeZhi, zhenShiZhi


def huaDuoMoXingYuCeQuXian(
    models_dict: dict,
    jiaZaiQi,
    sheBei: str,
    scalers: dict,
    baoCunLuJing: str,
    yangBenShu: int = 192,
):
    """
    绘制多模型预测曲线对比图（仿论文图6）。

    models_dict: {'模型名称': moXing_instance, ...}
                 按 ['真实值', 'LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention'] 顺序绘制
    yangBenShu: 展示的时间步数（192=两天，96=一天）
    """
    # 先获取真实值（从任意模型的数据中获取）
    first_model = next(iter(models_dict.values()))
    _, zhenShiZhi = _huoQuYuCeZhenShiForModel(first_model, jiaZaiQi, sheBei, scalers)
    n = min(yangBenShu, len(zhenShiZhi))

    fig, ax = plt.subplots(figsize=(14, 5))

    # 绘制真实值
    style = _MODEL_STYLES.get('真实值', {'color': '#e74c3c', 'linestyle': '-', 'linewidth': 2.0})
    ax.plot(np.arange(n), zhenShiZhi[:n], color=style['color'],
            linestyle=style['linestyle'], linewidth=style['linewidth'],
            label='Actual', zorder=10)

    # 绘制各模型
    draw_order = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
    for name in draw_order:
        if name not in models_dict:
            continue
        moXing = models_dict[name]
        yuCeZhi, _ = _huoQuYuCeZhenShiForModel(moXing, jiaZaiQi, sheBei, scalers)
        style = _MODEL_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
        ax.plot(np.arange(n), yuCeZhi[:n], color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style.get('label', name), alpha=0.85)

    tick_interval = 24
    ax.set_xticks(np.arange(0, n + 1, tick_interval))
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Load / (kW·h)', fontsize=12)
    ax.set_title('Prediction Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaDuoMoXingJueDuiWuCha(
    models_dict: dict,
    jiaZaiQi,
    sheBei: str,
    scalers: dict,
    baoCunLuJing: str,
    yangBenShu: int = 192,
):
    """
    绘制各模型平均绝对误差（MAE 逐点）对比图（仿论文图7）。
    纵轴为逐点绝对百分比误差，便于直观看出各模型的误差分布差异。
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    draw_order = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
    for name in draw_order:
        if name not in models_dict:
            continue
        moXing = models_dict[name]
        yuCeZhi, zhenShiZhi = _huoQuYuCeZhenShiForModel(moXing, jiaZaiQi, sheBei, scalers)
        n = min(yangBenShu, len(yuCeZhi))
        ape = np.abs((zhenShiZhi[:n] - yuCeZhi[:n]) / (zhenShiZhi[:n] + 1e-8)) * 100
        style = _MODEL_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
        ax.plot(np.arange(1, n + 1), ape, color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style.get('label', name), alpha=0.85)

    tick_interval = 24
    ax.set_xticks(np.arange(0, min(yangBenShu, n) + 1, tick_interval))
    ax.set_xlabel('Sample Point', fontsize=12)
    ax.set_ylabel('Absolute Percentage Error (%)', fontsize=12)
    ax.set_title('Absolute Error Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaDuoMoXingZhiBiaoBiaoGe(
    metrics_dict: dict,
    baoCunLuJing: str,
):
    """
    绘制多模型评估指标对比表格图（仿论文表3格式）。

    metrics_dict: {'模型名': {'MAPE(%)': x, 'NRMSE(%)': y, 'R²': z}, ...}
    """
    model_order = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
    rows = []
    for name in model_order:
        if name not in metrics_dict:
            continue
        m = metrics_dict[name]
        rows.append([
            name,
            f"{m.get('MAPE(%)', float('nan')):.4f}",
            f"{m.get('NRMSE(%)', float('nan')):.4f}",
            f"{m.get('R²', float('nan')):.4f}",
        ])

    n_data_rows = len(rows)
    fig, ax = plt.subplots(figsize=(9, n_data_rows * 0.55 + 0.9))
    ax.axis('off')
    col_labels = ['Model', 'MAPE (%)', 'NRMSE (%)', r'$R^2$']
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.40, 0.20, 0.20, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#b2dfdb')
        if r == 0:
            cell.set_facecolor('#009688')   # 青绿色
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#e0f2f1')   # 浅青绿交替行
        else:
            cell.set_facecolor('#ffffff')
        # 高亮主模型行（深一点的青绿色）
        if r > 0 and rows[r - 1][0] == 'TCN-BiLSTM-Attention':
            cell.set_facecolor('#80cbc4')
            cell.set_text_props(fontweight='bold')

    fig.suptitle('Model Comparison Results', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaChaoCanShuPeiZhiBiaoGe(
    model_configs: dict,
    baoCunLuJing: str,
):
    """
    绘制各模型超参数配置表格图（仿论文表2格式）。

    model_configs: {
        '模型名': {
            '网络层数': ..., '神经元个数': ..., '迭代次数': ...,
            '批量大小': ..., '学习率': ...
        },
        ...
    }
    """
    model_order = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
    rows = []
    for name in model_order:
        if name not in model_configs:
            continue
        cfg = model_configs[name]
        rows.append([
            name,
            str(cfg.get('Layers', '-')),
            str(cfg.get('Hidden Dim', '-')),
            str(cfg.get('Epochs', '-')),
            str(cfg.get('Batch Size', '-')),
            str(cfg.get('Learning Rate', '-')),
        ])

    col_labels = ['Model', 'Layers', 'Hidden Dim', 'Epochs', 'Batch Size', 'Learning Rate']
    n_data_rows = len(rows)
    fig, ax = plt.subplots(figsize=(13, n_data_rows * 0.55 + 0.9))
    ax.axis('off')
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.28, 0.13, 0.18, 0.13, 0.13, 0.13],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#b2dfdb')
        if r == 0:
            cell.set_facecolor('#009688')   # 青绿色
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#e0f2f1')
        else:
            cell.set_facecolor('#ffffff')
        if r > 0 and rows[r - 1][0] == 'TCN-BiLSTM-Attention':
            cell.set_facecolor('#80cbc4')
            cell.set_text_props(fontweight='bold')

    fig.suptitle('Model Hyperparameter Configuration', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def huaLiangLiangBiJiao(
    name_a: str,
    name_b: str,
    models_dict: dict,
    jiaZaiQi,
    sheBei: str,
    scalers: dict,
    baoCunLuJing: str,
    yangBenShu: int = 192,
):
    """
    绘制两模型与真实值的对比曲线（仿论文图8/9：逐步比较相邻两个模型）。
    """
    first_model = next(iter(models_dict.values()))
    _, zhenShiZhi = _huoQuYuCeZhenShiForModel(first_model, jiaZaiQi, sheBei, scalers)
    n = min(yangBenShu, len(zhenShiZhi))

    fig, ax = plt.subplots(figsize=(13, 5))

    # 真实值
    ax.plot(np.arange(n), zhenShiZhi[:n], color='#e74c3c', linewidth=2.0, label='Actual', zorder=10)

    for name in [name_a, name_b]:
        if name not in models_dict:
            continue
        yuCeZhi, _ = _huoQuYuCeZhenShiForModel(models_dict[name], jiaZaiQi, sheBei, scalers)
        style = _MODEL_STYLES.get(name, {'color': 'gray', 'linestyle': '-', 'linewidth': 1.5})
        ax.plot(np.arange(n), yuCeZhi[:n], color=style['color'],
                linestyle=style['linestyle'], linewidth=style['linewidth'],
                label=style.get('label', name), alpha=0.9)

    tick_interval = 24
    ax.set_xticks(np.arange(0, n + 1, tick_interval))
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Load / (kW·h)', fontsize=12)
    ax.set_title(f'{name_a} vs {name_b} Prediction Comparison', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(baoCunLuJing, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {baoCunLuJing}")


def _qingKongJieGuoMuLu(output_dir):
    """清空结果目录中的旧文件（避免显示上次运行的结果）"""
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            p = os.path.join(output_dir, f)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _baoChunWenZiBaoGao(output_dir, mse, mae, mape, r2, mapeJiChu, jiangDi, pingJunHaoMiao, xunLianDaiShu, teShuChangJingJieGuo=None, smape=None, wape=None, acc=None):
    """保存文字报告到 Markdown 文件"""
    from datetime import datetime
    path = os.path.join(output_dir, '评估结果报告.md')
    lines = [
        '# 模型预测评估结果报告',
        '',
        f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        '> 以下指标均在**测试集**上计算（模型训练全程未见过测试集）',
        '',
        '## 一、评估指标',
        '',
        '| 指标 | 数值 |',
        '|------|------|',
        f'| MSE | {mse:.4f} |',
        f'| MAE | {mae:.4f} |',
        f'| MAPE | {mape:.4f}% |',
        f'| R² 决定系数 | {r2:.4f} |',
        ''
    ]
    if acc is not None:
        lines.insert(-2, f'| Acc 预测准确率 | {acc:.4f}% |')
    if smape is not None and wape is not None:
        lines.insert(-2, f'| SMAPE | {smape:.4f}% |')
        lines.insert(-2, f'| WAPE | {wape:.4f}% |')
    if mapeJiChu is not None:
        lines.extend([
            '## 二、基线对比',
            '',
            '| 模型 | MAPE |',
            '|------|------|',
            f'| 基线 LSTM | {mapeJiChu:.4f}% |',
            f'| TCN-BiLSTM-Attention | {mape:.4f}% |',
            f'| MAPE 降低 | {jiangDi:.1f}% |',
            ''
        ])
    if teShuChangJingJieGuo:
        lines.extend([
            '## 三、特殊场景 MAPE',
            '',
            '| 场景 | 样本数 | MAPE |',
            '|------|--------|------|',
        ])
        for name, (n, m) in teShuChangJingJieGuo.items():
            lines.append(f'| {name} | {n} | {m:.4f}% |')
        lines.append('')
    lines.extend([
        '## 四、其他信息',
        '',
        f'| 项目 | 数值 |',
        '|------|------|',
        f'| 训练轮数 | {xunLianDaiShu} |',
        f'| 单次预测耗时 | {pingJunHaoMiao:.2f} 毫秒 |',
        f'| 实时性要求 (≤1分钟) | {"满足" if pingJunHaoMiao < 60000 else "未满足"} |',
        '',
        '## 五、图表说明',
        '',
        '- `01_train_curves.png` - 训练/验证 Loss 与 MAPE 曲线',
        '- `02_pred_vs_actual.png` - 预测值 vs 真实值 散点图与时序对比',
        '- `03_error_distribution.png` - 残差分布直方图',
        '- `03b_error_hist_kde.png` - 误差分布直方图与核密度估计',
        '- `03c_prediction_matrix.png` - 预测准确度矩阵（回归版混淆矩阵）',
        '- `04_mape_by_hour.png` - 各时段 MAPE 分布',
        '- `05_metrics_summary.png` - 评估指标柱状图',
        '- `06_metrics_dashboard.png` - 关键评价指标仪表盘',
        ''
    ])
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  已保存: {path}")


def shengChengSuoYouTuBiao(
    moXing,
    xunLianLiShi,
    jiaZaiQi,
    sheBei,
    scalers,
    mse,
    mae,
    mape,
    r2=0,
    mapeJiChu=None,
    jiangDi=None,
    pingJunHaoMiao=None,
    xunLianDaiShu=50,
    teShuChangJingJieGuo=None,
    smape=None,
    wape=None,
    acc=None,
    nrmse=None,
    models_dict=None,
    metrics_dict=None,
    model_configs=None,
    output_dir=None,
    copy_to_dir=None,
):
    """
    生成所有评估可视化图表及文字报告，保存至 模型预测结果 文件夹。
    output_dir: 为 None 时使用默认 模型预测结果 根目录（与 RESULT_DIR_NAME 一致）。
    copy_to_dir: 若指定，在生成后把本目录下 .png 与 评估结果报告.md 再复制到该路径（不删除源文件）。
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n生成评估可视化...")
    huaXunLianQuXian(xunLianLiShi, os.path.join(output_dir, '01_train_curves.png'))
    huaYuCeDuiBi(moXing, jiaZaiQi, sheBei, scalers, os.path.join(output_dir, '02_pred_vs_actual.png'))
    huaWuChaFenBu(moXing, jiaZaiQi, sheBei, scalers, os.path.join(output_dir, '03_error_distribution.png'))
    huaWuChaZhiFangTuHeMiDu(moXing, jiaZaiQi, sheBei, scalers, os.path.join(output_dir, '03b_error_hist_kde.png'))
    huaYuCeZhunQueDuJuZhen(moXing, jiaZaiQi, sheBei, scalers, os.path.join(output_dir, '03c_prediction_matrix.png'))
    huaAnXiaoShiMAPE(moXing, jiaZaiQi, sheBei, scalers, os.path.join(output_dir, '04_mape_by_hour.png'))
    huaPingGuZongJie(mse, mae, mape, mapeJiChu, os.path.join(output_dir, '05_metrics_summary.png'), smape, wape, acc)
    huaPingJiaZhiBiaoYiBiaoPan(mse, mae, mape, r2, mapeJiChu, os.path.join(output_dir, '06_metrics_dashboard.png'), smape, wape, acc)

    # 多模型对比图（仿论文图6/7/8/9）
    if models_dict and len(models_dict) > 1:
        print("\n生成多模型对比图表...")
        huaDuoMoXingYuCeQuXian(
            models_dict, jiaZaiQi, sheBei, scalers,
            os.path.join(output_dir, '07_model_compare_curves.png'),
        )
        huaDuoMoXingJueDuiWuCha(
            models_dict, jiaZaiQi, sheBei, scalers,
            os.path.join(output_dir, '08_model_compare_error.png'),
        )
        # 逐步两两对比（仿论文图8/9）
        order = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
        available = [n for n in order if n in models_dict]
        for i in range(len(available) - 1):
            a, b = available[i], available[i + 1]
            fname = f'09_compare_{a.replace("-","_")}_vs_{b.replace("-","_")}.png'
            huaLiangLiangBiJiao(a, b, models_dict, jiaZaiQi, sheBei, scalers,
                                 os.path.join(output_dir, fname))

    if metrics_dict:
        huaDuoMoXingZhiBiaoBiaoGe(metrics_dict, os.path.join(output_dir, '10_metrics_table.png'))

    if model_configs:
        huaChaoCanShuPeiZhiBiaoGe(model_configs, os.path.join(output_dir, '11_hyperparams_table.png'))

    if pingJunHaoMiao is not None:
        _baoChunWenZiBaoGao(output_dir, mse, mae, mape, r2, mapeJiChu, jiangDi or 0, pingJunHaoMiao,
                             xunLianDaiShu, teShuChangJingJieGuo, smape, wape, acc)

    print(f"  所有结果已保存至: {output_dir}")

    if copy_to_dir:
        import shutil
        os.makedirs(copy_to_dir, exist_ok=True)
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if not os.path.isfile(fp):
                continue
            if f.endswith(('.png', '.md')):
                shutil.copy2(fp, os.path.join(copy_to_dir, f))
        print(f"  已同步复制到: {copy_to_dir}")

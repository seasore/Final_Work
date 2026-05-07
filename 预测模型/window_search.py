# -*- coding: utf-8 -*-
"""
滑动窗口大小对比实验
Sliding Window Size Comparison Experiment

仿照参考论文（基于MSCNN-BiGRU-Attention的短期电力负荷预测）3.5节：
逐一测试不同滑动窗口宽度对预测精度的影响，生成对比表格和曲线图。

候选窗口宽度（以15min为采样间隔）：
  24  → 6小时
  48  → 12小时
  72  → 18小时
  96  → 24小时（一天，与论文最优值一致）
  128 → 32小时
  192 → 48小时（两天）

独立运行：python window_search.py
main.py 跑完后再运行本脚本，不会影响已有结果。
"""

import os
import sys
import warnings
import time
import json

warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*WeightNorm.*', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader

from data import DianLiFuHeShuJuJi, _TIME_FEATURE_COLS
from model import TCNBiLstmZhuYiLiYuCeMoXing
from train import xunLianDuiBiMoXing, pingGuMoXing
from device_utils import get_best_device

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_CSV = os.path.join(_BASE_DIR, '数据', '训练集', 'train_data.csv')
_VAL_CSV = os.path.join(_BASE_DIR, '数据', '验证集', 'val_data.csv')
_SCALERS_PKL = os.path.join(_BASE_DIR, '数据预处理', 'scalers.pkl')

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '模型预测结果', 'window_search')

# 候选窗口宽度
WINDOW_CANDIDATES = [24, 48, 72, 96, 128, 192]

# 每种窗口的训练轮数（快速评估用，比主训练少）
TRAIN_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
YU_CE_BU_CHANG = 1
SHI_JIAN_TE_ZHENG_SHU = 5

# 默认模型超参数（从 best_params.json 读取，若不存在则使用以下默认值）
_DEFAULT_MODEL_PARAMS = {
    'tcnYinCangWeiDu': 64,
    'tcnCengShu': 4,
    'bilstmYinCangWeiDu': 64,
    'bilstmCengShu': 2,
    'zhuYiLiTouShu': 4,
    'tuoQiLv': 0.2,
}


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _load_best_model_params() -> dict:
    """从 best_params.json 加载主模型最优超参数"""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            params = d.get('best_params', {})
            if params:
                print(f"  已从 best_params.json 读取模型超参数")
                return params
        except Exception:
            pass
    print(f"  best_params.json 不存在或读取失败，使用默认超参数")
    return _DEFAULT_MODEL_PARAMS


def _load_data_for_window(xu_lie_chang_du: int):
    """加载指定窗口长度的数据加载器"""
    if not os.path.exists(_TRAIN_CSV) or not os.path.exists(_VAL_CSV):
        raise FileNotFoundError(
            f"请先运行数据预处理：\n  {_TRAIN_CSV}\n  {_VAL_CSV}"
        )

    train_df = pd.read_csv(_TRAIN_CSV)
    val_df = pd.read_csv(_VAL_CSV)

    scalers = {}
    if os.path.exists(_SCALERS_PKL):
        with open(_SCALERS_PKL, 'rb') as f:
            scalers = pickle.load(f)

    def _extract(df):
        fu_he = df['M019Value'].values.reshape(-1, 1).astype(np.float32)
        qi_xiang = df[['max_temp', 'min_temp', 'humidity', 'wind_speed']].values.astype(np.float32)
        shi_jian = df[_TIME_FEATURE_COLS].values.astype(np.float32)
        mu_biao = df['M019Value'].values.astype(np.float32)
        return fu_he, qi_xiang, shi_jian, mu_biao

    train_ds = DianLiFuHeShuJuJi(*_extract(train_df), xu_lie_chang_du, YU_CE_BU_CHANG)
    val_ds = DianLiFuHeShuJuJi(*_extract(val_df), xu_lie_chang_du, YU_CE_BU_CHANG)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    return train_loader, val_loader, scalers


def _build_model(model_params: dict) -> TCNBiLstmZhuYiLiYuCeMoXing:
    return TCNBiLstmZhuYiLiYuCeMoXing(
        fuHeTeZhengShu=1,
        qiXiangTeZhengShu=4,
        shiJianTeZhengShu=SHI_JIAN_TE_ZHENG_SHU,
        tcnYinCangWeiDu=model_params.get('tcnYinCangWeiDu', 64),
        tcnCengShu=model_params.get('tcnCengShu', 4),
        bilstmYinCangWeiDu=model_params.get('bilstmYinCangWeiDu', 64),
        bilstmCengShu=model_params.get('bilstmCengShu', 2),
        zhuYiLiTouShu=model_params.get('zhuYiLiTouShu', 4),
        yuCeBuChang=YU_CE_BU_CHANG,
        tuoQiLv=model_params.get('tuoQiLv', 0.2),
    )


# ──────────────────────────────────────────────────────────────
# 核心：逐窗口训练与评估
# ──────────────────────────────────────────────────────────────

def run_window_search(
    window_list: list = None,
    output_dir: str = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    对每种滑动窗口大小训练主模型并评估，输出对比表格与曲线图。

    返回: DataFrame，每行对应一种窗口大小的指标
    """
    if window_list is None:
        window_list = WINDOW_CANDIDATES
    if output_dir is None:
        output_dir = RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    sheBei = get_best_device(verbose=True)
    model_params = _load_best_model_params()

    print(f"\n滑动窗口大小对比实验")
    print(f"候选窗口: {window_list}（单位：15分钟采样点）")
    print(f"训练轮数: {TRAIN_EPOCHS}  |  设备: {sheBei}")
    print("=" * 65)

    records = []

    for w in window_list:
        hours = w * 15 / 60
        print(f"\n  窗口宽度={w}（{hours:.0f}小时）...")
        t0 = time.time()

        try:
            train_loader, val_loader, scalers = _load_data_for_window(w)
            moXing = _build_model(model_params)

            _, best_state = xunLianDuiBiMoXing(
                moXing, train_loader, val_loader,
                sheBei=sheBei,
                xueXiLv=LEARNING_RATE,
                xunLianDaiShu=TRAIN_EPOCHS,
                scalers=scalers,
                zaoZhiTingZhiPatience=8,
                quiet=quiet,
                label=f'窗口={w}',
            )
            moXing.load_state_dict(best_state)
            moXing.to(sheBei)

            res = pingGuMoXing(moXing, val_loader, sheBei, scalers)
            mape, nrmse, r2 = res[2], res[6], res[3]
            mae, mse = res[1], res[0]
            elapsed = time.time() - t0

            records.append({
                '滑窗宽度': w,
                '对应时长': f'{hours:.0f}h',
                'MAPE(%)': round(mape, 4),
                'NRMSE(%)': round(nrmse, 4),
                'R²': round(r2, 4),
                'MAE': round(mae, 4),
                '耗时(s)': round(elapsed),
            })
            marker = ' ◀ 当前最优' if mape == min(r['MAPE(%)'] for r in records) else ''
            print(f"    → MAPE={mape:.4f}%, NRMSE={nrmse:.4f}%, R²={r2:.4f}  ({elapsed:.0f}s){marker}")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"    → 失败: {e}")
            records.append({
                '滑窗宽度': w,
                '对应时长': f'{hours:.0f}h',
                'MAPE(%)': float('nan'),
                'NRMSE(%)': float('nan'),
                'R²': float('nan'),
                'MAE': float('nan'),
                '耗时(s)': round(elapsed),
            })

    df = pd.DataFrame(records)

    # 打印汇总表
    print("\n" + "=" * 65)
    print("滑动窗口对比汇总：")
    print(df[['滑窗宽度', '对应时长', 'MAPE(%)', 'NRMSE(%)', 'R²']].to_string(index=False))

    # 找最优
    valid = df.dropna(subset=['MAPE(%)'])
    if not valid.empty:
        best_row = valid.loc[valid['MAPE(%)'].idxmin()]
        print(f"\n  >> 最优滑窗宽度: {int(best_row['滑窗宽度'])}（对应{best_row['对应时长']}），"
              f"MAPE={best_row['MAPE(%)']:.4f}%")

    # 保存 CSV
    csv_path = os.path.join(output_dir, 'window_search_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  结果已保存: {csv_path}")

    # 生成图表
    _plot_window_results(df, output_dir)

    return df


# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────

def _plot_window_results(df: pd.DataFrame, output_dir: str):
    """绘制窗口大小对比曲线图（仿论文图7风格）+ 表格图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    valid = df.dropna(subset=['MAPE(%)'])
    x_labels = valid['滑窗宽度'].astype(str).tolist()
    mape_vals = valid['MAPE(%)'].tolist()
    nrmse_vals = valid['NRMSE(%)'].tolist()
    r2_vals = valid['R²'].tolist()

    # ── 曲线图（三指标）──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ('MAPE(%)', mape_vals, '#e74c3c'),
        ('NRMSE(%)', nrmse_vals, '#3498db'),
        ('R²', r2_vals, '#2ecc71'),
    ]

    for ax, (label, vals, color) in zip(axes, metrics):
        ax.plot(x_labels, vals, 'o-', color=color, linewidth=2.2, markersize=8)
        for xi, vi in zip(x_labels, vals):
            ax.annotate(f'{vi:.4f}', (xi, vi), textcoords="offset points",
                        xytext=(0, 9), ha='center', fontsize=8.5)
        # 标注最优点
        if label in ('MAPE(%)', 'NRMSE(%)'):
            best_idx = int(np.argmin(vals))
        else:
            best_idx = int(np.argmax(vals))
        ax.scatter([x_labels[best_idx]], [vals[best_idx]], color='gold', s=150,
                   zorder=5, edgecolors='black', linewidths=1.5,
                   label=f'最优: {x_labels[best_idx]}')
        ax.set_xlabel('滑窗宽度（采样点）', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} vs 滑窗宽度', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('不同滑动窗口宽度对预测精度的影响', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, 'window_comparison_curves.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname}")

    # ── 表格图（仿论文表3格式）──
    show_cols = ['滑窗宽度', '对应时长', 'MAPE(%)', 'NRMSE(%)', 'R²']
    rows = []
    best_mape = valid['MAPE(%)'].min() if not valid.empty else None
    for _, row in df.iterrows():
        rows.append([
            str(int(row['滑窗宽度'])),
            str(row['对应时长']),
            f"{row['MAPE(%)']:.4f}" if not np.isnan(row['MAPE(%)']) else '-',
            f"{row['NRMSE(%)']:.4f}" if not np.isnan(row['NRMSE(%)']) else '-',
            f"{row['R²']:.4f}" if not np.isnan(row['R²']) else '-',
        ])

    fig2, ax2 = plt.subplots(figsize=(9, len(rows) * 0.62 + 2.0))
    ax2.axis('off')
    table = ax2.table(
        cellText=rows,
        colLabels=show_cols,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.14, 0.16, 0.16, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        # 高亮最优行
        if r > 0 and best_mape is not None:
            try:
                row_mape = float(rows[r - 1][2])
                if abs(row_mape - best_mape) < 1e-6:
                    cell.set_facecolor('#d5e8d4')
                    cell.set_text_props(fontweight='bold')
            except ValueError:
                pass

    ax2.set_title('不同滑窗宽度下的预测误差对比', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fname2 = os.path.join(output_dir, 'window_comparison_table.png')
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname2}")


# ──────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='滑动窗口大小对比实验')
    parser.add_argument('--windows', nargs='+', type=int,
                        default=WINDOW_CANDIDATES,
                        help=f'候选窗口大小列表，默认: {WINDOW_CANDIDATES}')
    parser.add_argument('--epochs', type=int, default=TRAIN_EPOCHS,
                        help=f'每种窗口的训练轮数，默认: {TRAIN_EPOCHS}')
    parser.add_argument('--verbose', action='store_true',
                        help='显示每轮训练详情')
    args = parser.parse_args()

    TRAIN_EPOCHS = args.epochs
    run_window_search(
        window_list=args.windows,
        quiet=not args.verbose,
    )

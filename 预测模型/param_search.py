# -*- coding: utf-8 -*-
"""
超参数网格搜索与探索模块
Hyperparameter Grid Search & Exploration

仿照参考论文（基于MSCNN-BiGRU-Attention的短期电力负荷预测）第三节的实验方法：
逐一固定其他超参数，对目标超参数进行网格搜索，记录 MAPE(%)、NRMSE(%)、R² 并绘图。

探索的超参数（主模型 TCN-BiLSTM-Attention）：
  1. TCN 网络层数（tcnCengShu）：影响时序局部特征的感受野
  2. BiLSTM 网络层数（bilstmCengShu）：影响长程依赖建模深度
  3. TCN 隐藏维度（tcnYinCangWeiDu）：特征表达容量
  4. BiLSTM 隐藏维度（bilstmYinCangWeiDu）：时序建模容量
  5. 批次大小（piCiDaXiao）：影响梯度估计稳定性与收敛速度
  6. 学习率（xueXiLv）：影响收敛速度与最终精度

默认值（探索某参数时其余均取此默认值）：
  tcnCengShu=4, bilstmCengShu=2, tcnYinCangWeiDu=64, bilstmYinCangWeiDu=64,
  piCiDaXiao=64, xueXiLv=0.001, epochs=40
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*WeightNorm.*', category=FutureWarning)
import numpy as np
import pandas as pd
import pickle
import json
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import DianLiFuHeShuJuJi, _TIME_FEATURE_COLS
from model import TCNBiLstmZhuYiLiYuCeMoXing
from train import xunLianDuiBiMoXing, pingGuMoXing
from device_utils import get_best_device

# ──────────────────────────────────────────────────────────────
# 路径常量
# ──────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_CSV = os.path.join(_BASE_DIR, '数据', '训练集', 'train_data.csv')
_VAL_CSV = os.path.join(_BASE_DIR, '数据', '验证集', 'val_data.csv')
_SCALERS_PKL = os.path.join(_BASE_DIR, '数据预处理', 'scalers.pkl')

PARAM_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '模型预测结果', 'param_search')

# ──────────────────────────────────────────────────────────────
# 默认超参数（其他参数固定值）
# ──────────────────────────────────────────────────────────────
_DEFAULT_PARAMS = {
    'tcnCengShu': 4,
    'bilstmCengShu': 2,
    'tcnYinCangWeiDu': 64,
    'bilstmYinCangWeiDu': 64,
    'zhuYiLiTouShu': 4,
    'tuoQiLv': 0.2,
    'piCiDaXiao': 64,
    'xueXiLv': 0.001,
}

# 各参数的候选值（仿照论文逐一探索）
_SEARCH_SPACE = {
    'tcnCengShu':        [1, 2, 3, 4, 5, 6, 7, 8],
    'bilstmCengShu':     [1, 2, 3, 4],
    'tcnYinCangWeiDu':   [16, 32, 48, 64, 96, 128],
    'bilstmYinCangWeiDu': [16, 32, 48, 64, 96, 128],
    'piCiDaXiao':        [16, 32, 64, 128, 256],
    'xueXiLv':           [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
}

_XU_LIE_CHANG_DU = 96
_YU_CE_BU_CHANG = 1
_SHI_JIAN_TE_ZHENG_SHU = 5
_SEARCH_EPOCHS = 60   # 每次搜索的训练轮数（配合双重早停；坏配置会提前退出）
_SEARCH_SEEDS = [42, 123, 456]  # 多种子：每个候选值训练 N 次取平均，消除初始化噪声


# ──────────────────────────────────────────────────────────────
# 数据加载工具
# ──────────────────────────────────────────────────────────────

def _load_data(piCiDaXiao=64):
    """加载训练集与验证集，返回 DataLoader 和 scalers"""
    if not os.path.exists(_TRAIN_CSV) or not os.path.exists(_VAL_CSV):
        raise FileNotFoundError(f"请先运行数据预处理，确保存在:\n  {_TRAIN_CSV}\n  {_VAL_CSV}")

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

    train_ds = DianLiFuHeShuJuJi(*_extract(train_df), _XU_LIE_CHANG_DU, _YU_CE_BU_CHANG)
    val_ds = DianLiFuHeShuJuJi(*_extract(val_df), _XU_LIE_CHANG_DU, _YU_CE_BU_CHANG)

    train_loader = DataLoader(train_ds, batch_size=piCiDaXiao, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=piCiDaXiao, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, scalers


def _build_model(params: dict) -> TCNBiLstmZhuYiLiYuCeMoXing:
    """按参数字典构建主模型"""
    return TCNBiLstmZhuYiLiYuCeMoXing(
        fuHeTeZhengShu=1,
        qiXiangTeZhengShu=4,
        shiJianTeZhengShu=_SHI_JIAN_TE_ZHENG_SHU,
        tcnYinCangWeiDu=params.get('tcnYinCangWeiDu', _DEFAULT_PARAMS['tcnYinCangWeiDu']),
        tcnCengShu=params.get('tcnCengShu', _DEFAULT_PARAMS['tcnCengShu']),
        bilstmYinCangWeiDu=params.get('bilstmYinCangWeiDu', _DEFAULT_PARAMS['bilstmYinCangWeiDu']),
        bilstmCengShu=params.get('bilstmCengShu', _DEFAULT_PARAMS['bilstmCengShu']),
        zhuYiLiTouShu=params.get('zhuYiLiTouShu', _DEFAULT_PARAMS['zhuYiLiTouShu']),
        yuCeBuChang=_YU_CE_BU_CHANG,
        tuoQiLv=params.get('tuoQiLv', _DEFAULT_PARAMS['tuoQiLv']),
    )


# ──────────────────────────────────────────────────────────────
# 单参数网格搜索
# ──────────────────────────────────────────────────────────────

def _tanSuoDanCanShu(param_name: str, sheBei: str, quiet=True,
                     seeds: list = None) -> pd.DataFrame:
    """
    固定其他参数为默认值，对 param_name 进行逐值多种子搜索。
    每个候选值用 len(seeds) 个不同随机种子各训练一次，取指标均值，消除初始化噪声。

    返回 DataFrame: columns=[param_name, 'MAPE(%)', 'NRMSE(%)', 'R²', 'MAPE_std']
    """
    if seeds is None:
        seeds = _SEARCH_SEEDS
    candidates = _SEARCH_SPACE[param_name]
    records = []

    print(f"\n  探索参数: {param_name}, 候选值: {candidates}, 种子数: {len(seeds)}")

    for val in candidates:
        params = dict(_DEFAULT_PARAMS)
        params[param_name] = val

        piCiDaXiao = int(params['piCiDaXiao'])
        xueXiLv = float(params['xueXiLv'])

        mapes, nrmses, r2s = [], [], []
        t0 = time.time()

        for seed_idx, seed in enumerate(seeds):
            try:
                # 固定随机种子：PyTorch / NumPy / CUDA
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                train_loader, val_loader, scalers = _load_data(piCiDaXiao)
                moXing = _build_model(params)

                _, zuiJiaZhuangTai = xunLianDuiBiMoXing(
                    moXing, train_loader, val_loader,
                    sheBei=sheBei, xueXiLv=xueXiLv,
                    xunLianDaiShu=_SEARCH_EPOCHS,
                    scalers=scalers,
                    zaoZhiTingZhiPatience=10,
                    quiet=quiet,
                    label=f'{param_name}={val} seed={seed}',
                )
                moXing.load_state_dict(zuiJiaZhuangTai)
                moXing.to(sheBei)

                res = pingGuMoXing(moXing, val_loader, sheBei, scalers)
                mapes.append(res[2])
                nrmses.append(res[6])
                r2s.append(res[3])

                if not quiet:
                    print(f"      seed={seed}: MAPE={res[2]:.4f}%, NRMSE={res[6]:.4f}%, R²={res[3]:.4f}")

            except Exception as e:
                print(f"    {param_name}={val} seed={seed}: 失败 - {e}")

        elapsed = time.time() - t0
        if mapes:
            mean_mape = float(np.mean(mapes))
            std_mape  = float(np.std(mapes))
            mean_nrmse = float(np.mean(nrmses))
            mean_r2    = float(np.mean(r2s))
            records.append({
                param_name:  val,
                'MAPE(%)':   round(mean_mape, 4),
                'NRMSE(%)':  round(mean_nrmse, 4),
                'R²':        round(mean_r2, 4),
                'MAPE_std':  round(std_mape, 4),
            })
            print(f"    {param_name}={val}: MAPE={mean_mape:.4f}%±{std_mape:.4f}%, "
                  f"NRMSE={mean_nrmse:.4f}%, R²={mean_r2:.4f}  ({elapsed:.0f}s, {len(mapes)}/{len(seeds)} 成功)")
        else:
            records.append({param_name: val, 'MAPE(%)': float('nan'),
                             'NRMSE(%)': float('nan'), 'R²': float('nan'), 'MAPE_std': float('nan')})
            print(f"    {param_name}={val}: 全部失败")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────
# 参数名称到中文标签的映射
# ──────────────────────────────────────────────────────────────

_PARAM_LABELS = {
    'tcnCengShu':        'TCN Layers',
    'bilstmCengShu':     'BiLSTM Layers',
    'tcnYinCangWeiDu':   'TCN Hidden Dim',
    'bilstmYinCangWeiDu': 'BiLSTM Hidden Dim',
    'piCiDaXiao':        'Batch Size',
    'xueXiLv':           'Learning Rate',
    'zhuYiLiTouShu':     'Attention Heads',
    'tuoQiLv':           'Dropout Rate',
}


# ──────────────────────────────────────────────────────────────
# 绘制单参数探索曲线（论文图风格）
# ──────────────────────────────────────────────────────────────

def _huaTanSuoTuBiao(df: pd.DataFrame, param_name: str, output_dir: str):
    """
    绘制参数探索结果曲线图。
    若 DataFrame 含 'MAPE_std' 列（多种子结果），MAPE 子图将绘制误差棒（±std），
    直观展示各候选值结果的稳定性。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    param_label = _PARAM_LABELS.get(param_name, param_name)
    has_std = 'MAPE_std' in df.columns

    x_raw = df[param_name].tolist()
    x_str = [str(v) for v in x_raw]
    mape_vals = df['MAPE(%)'].tolist()
    nrmse_vals = df['NRMSE(%)'].tolist()
    r2_vals = df['R²'].tolist()
    mape_std_vals = df['MAPE_std'].tolist() if has_std else [0.0] * len(mape_vals)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    metrics = [
        ('MAPE (%)',   mape_vals,  False),
        ('NRMSE (%)',  nrmse_vals, False),
        (r'$R^2$',     r2_vals,    True),
    ]

    def _best_valid_idx(vals, higher_is_better):
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if not valid:
            return None, None
        idx, val = (max if higher_is_better else min)(valid, key=lambda t: t[1])
        return idx, val

    for ax_idx, (ax, (label, vals, higher_is_better), color) in enumerate(zip(axes, metrics, colors)):
        valid_pairs = [(x_str[i], v) for i, v in enumerate(vals) if not np.isnan(v)]
        if len(valid_pairs) < 1:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(f'{label} vs {param_label}', fontsize=12)
            ax.set_xlabel(param_label, fontsize=11)
            ax.set_ylabel(label, fontsize=11)
            continue

        vx, vy = zip(*valid_pairs)

        # MAPE 子图：若有多种子标准差则画误差棒
        if ax_idx == 0 and has_std:
            valid_std = [mape_std_vals[i] for i, v in enumerate(vals) if not np.isnan(v)]
            ax.errorbar(vx, vy, yerr=valid_std, fmt='o-', color=color,
                        linewidth=2, markersize=7, capsize=4, elinewidth=1.5,
                        ecolor=color, alpha=0.9)
            # 在误差棒旁标注 std
            for xi, vi, si in zip(vx, vy, valid_std):
                ax.annotate(f'±{si:.4f}', (xi, vi + si),
                            textcoords="offset points", xytext=(4, 4),
                            ha='left', fontsize=7, color='#888888')
        else:
            ax.plot(vx, vy, 'o-', color=color, linewidth=2, markersize=7)

        # 标注均值数字（奇偶交替）
        for i, (xi, vi) in enumerate(zip(vx, vy)):
            offset_y = 10 if i % 2 == 0 else -16
            ax.annotate(f'{vi:.4f}', (xi, vi), textcoords="offset points",
                        xytext=(0, offset_y), ha='center', fontsize=8, color='#333333')

        ax.set_xlabel(param_label, fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        title_suffix = ' (mean±std)' if ax_idx == 0 and has_std else ''
        ax.set_title(f'{label} vs {param_label}{title_suffix}', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        vmin, vmax = min(vy), max(vy)
        span = max(vmax - vmin, abs(vmin) * 0.02, 1e-6)
        ax.set_ylim(vmin - span * 0.4, vmax + span * 0.5)

        b_idx, b_val = _best_valid_idx(list(vy), higher_is_better)
        if b_idx is not None:
            ax.scatter([vx[b_idx]], [b_val], color='gold', s=140, zorder=6,
                       edgecolors='black', linewidths=1.5, label=f'Best: {vx[b_idx]}')
            ax.legend(fontsize=9, loc='best')

    n_seeds = len(_SEARCH_SEEDS)
    subtitle = f'Hyperparameter Search: {param_label}  (mean of {n_seeds} seeds)'
    fig.suptitle(subtitle, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    fname = os.path.join(output_dir, f'param_{param_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname}")


# ──────────────────────────────────────────────────────────────
# 综合探索入口
# ──────────────────────────────────────────────────────────────

def yunXingCanShuTanSuo(
    params_to_search: list = None,
    output_dir: str = None,
    quiet: bool = True,
    n_seeds: int = None,
) -> dict:
    """
    运行超参数探索实验，依次对各参数进行网格搜索，生成图表与汇总报告。

    params_to_search: 要探索的参数名列表，None 则探索全部
    output_dir: 图表保存目录，None 则使用默认
    quiet: 是否静默每轮训练输出
    n_seeds: 每个候选值使用的随机种子数（默认使用 _SEARCH_SEEDS 全部）

    返回: {param_name: DataFrame} 各参数的搜索结果
    """
    if params_to_search is None:
        params_to_search = list(_SEARCH_SPACE.keys())

    seeds = _SEARCH_SEEDS if n_seeds is None else _SEARCH_SEEDS[:n_seeds]

    if output_dir is None:
        output_dir = PARAM_RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    sheBei = get_best_device(verbose=True)
    print(f"\n超参数探索（设备: {sheBei}，每值训 {len(seeds)} 次取均值，种子: {seeds}）")
    print(f"保存目录: {output_dir}")
    print("=" * 60)

    all_results = {}
    best_vals = dict(_DEFAULT_PARAMS)  # 累积最优值

    for param_name in params_to_search:
        df = _tanSuoDanCanShu(param_name, sheBei, quiet=quiet, seeds=seeds)
        all_results[param_name] = df

        # 保存 CSV
        csv_path = os.path.join(output_dir, f'param_{param_name}.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 绘图
        _huaTanSuoTuBiao(df, param_name, output_dir)

        # 更新最优值（用 MAPE 最小的）
        valid = df.dropna(subset=['MAPE(%)'])
        if not valid.empty:
            best_row = valid.loc[valid['MAPE(%)'].idxmin()]
            raw = best_row[param_name]
            # pandas 读取后整数列可能变为 numpy float，保持原类型写入 JSON
            if param_name in ('tcnCengShu', 'bilstmCengShu', 'zhuYiLiTouShu', 'piCiDaXiao',
                              'tcnYinCangWeiDu', 'bilstmYinCangWeiDu'):
                best_vals[param_name] = int(raw)
            else:
                best_vals[param_name] = float(raw)
            print(f"  >> {param_name} 最优值: {best_row[param_name]}  "
                  f"(MAPE={best_row['MAPE(%)']:.4f}%)")

    # 生成最优参数汇总表格图
    _huaCanShuZuiYouZongJie(best_vals, output_dir)

    # 保存最优参数到 JSON（可被 main.py 读取）
    best_json_path = os.path.join(output_dir, 'param_search_best.json')
    with open(best_json_path, 'w', encoding='utf-8') as fp:
        json.dump(best_vals, fp, indent=2, ensure_ascii=False)
    print(f"\n最优参数已保存: {best_json_path}")

    return all_results


def _huaCanShuZuiYouZongJie(best_vals: dict, output_dir: str):
    """生成最优超参数汇总表格图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    labels = {
        'tcnCengShu': 'TCN Layers',
        'bilstmCengShu': 'BiLSTM Layers',
        'tcnYinCangWeiDu': 'TCN Hidden Dim',
        'bilstmYinCangWeiDu': 'BiLSTM Hidden Dim',
        'zhuYiLiTouShu': 'Attention Heads',
        'tuoQiLv': 'Dropout Rate',
        'piCiDaXiao': 'Batch Size',
        'xueXiLv': 'Learning Rate',
    }

    rows = [[labels.get(k, k), str(v)] for k, v in best_vals.items()]
    n_rows = len(rows)

    fig, ax = plt.subplots(figsize=(6, n_rows * 0.52 + 0.9))
    ax.axis('off')
    # 用 bbox 参数精确控制表格在 axes 中的占位，减少多余空白
    table = ax.table(
        cellText=rows,
        colLabels=['Hyperparameter', 'Best Value'],
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.62, 0.38],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#b2dfdb')
        if r == 0:
            cell.set_facecolor('#009688')   # 青绿色（Material Teal 500）
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#e0f2f1')   # 浅青绿（Teal 50）
        else:
            cell.set_facecolor('#ffffff')
    fig.suptitle('TCN-BiLSTM-Attention Best Hyperparameters', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, bottom=0.02, left=0.02, right=0.98)

    fname = os.path.join(output_dir, 'param_best_summary.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname}")


# ──────────────────────────────────────────────────────────────
# 训练轮数收敛曲线（仿照论文图：Loss/MAPE vs Epoch）
# ──────────────────────────────────────────────────────────────

def tanSuoXunLianDaiShu(
    epoch_list: list = None,
    output_dir: str = None,
    quiet: bool = True,
) -> pd.DataFrame:
    """
    探索训练轮数影响：用固定参数训练较多轮，记录每轮 MAPE，绘制收敛曲线。
    """
    if epoch_list is None:
        epoch_list = [20, 40, 60, 80, 100, 120]

    if output_dir is None:
        output_dir = PARAM_RESULT_DIR
    os.makedirs(output_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    sheBei = get_best_device(verbose=False)
    max_epochs = max(epoch_list)

    train_loader, val_loader, scalers = _load_data(_DEFAULT_PARAMS['piCiDaXiao'])
    moXing = _build_model(_DEFAULT_PARAMS)

    from train import xunLianMoXing
    xunLianLiShi, _ = xunLianMoXing(
        moXing, train_loader, val_loader,
        sheBei=sheBei,
        xueXiLv=_DEFAULT_PARAMS['xueXiLv'],
        xunLianDaiShu=max_epochs,
        scalers=scalers,
        zaoZhiTingZhiPatience=0,  # 不早停，观察完整曲线
        quiet=quiet,
    )

    epochs_x = list(range(1, max_epochs + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_x, xunLianLiShi['xunLianLoss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].plot(epochs_x, xunLianLiShi['yanZhengLoss'], 'r--', linewidth=2, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Train / Val Loss Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, xunLianLiShi['yanZhengMAPE'], 'g-', linewidth=2, label='Val MAPE (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Validation MAPE Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'param_epoch_convergence.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {fname}")

    # 各节点 MAPE
    records = []
    for e in epoch_list:
        idx = min(e - 1, len(xunLianLiShi['yanZhengMAPE']) - 1)
        records.append({'训练轮数': e, 'MAPE(%)': round(xunLianLiShi['yanZhengMAPE'][idx], 4)})
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_dir, 'param_epoch.csv'), index=False, encoding='utf-8-sig')

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TCN-BiLSTM-Attention 超参数探索')
    parser.add_argument('--params', nargs='+',
                        choices=list(_SEARCH_SPACE.keys()) + ['all'],
                        default=['all'],
                        help='要探索的超参数，all 表示全部')
    parser.add_argument('--epoch_curve', action='store_true', help='额外绘制轮数收敛曲线')
    parser.add_argument('--verbose', action='store_true', help='显示每轮训练信息')
    parser.add_argument('--n_seeds', type=int, default=None,
                        help=f'每个候选值使用的随机种子数（默认 {len(_SEARCH_SEEDS)} 个: {_SEARCH_SEEDS}）')
    args = parser.parse_args()

    params_to_run = None if 'all' in args.params else args.params
    yunXingCanShuTanSuo(params_to_search=params_to_run, quiet=not args.verbose,
                        n_seeds=args.n_seeds)

    if args.epoch_curve:
        tanSuoXunLianDaiShu(quiet=not args.verbose)

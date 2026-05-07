# -*- coding: utf-8 -*-
"""
compare_params.py
-----------------
对比 OAT 网格搜索 与 Optuna 贝叶斯搜索 找到的最优超参数，
并用两组参数分别快速评估模型在验证集上的 MAPE，生成对比表格图。

用法（main.py 运行完毕后执行）：
    python compare_params.py

如果只有 OAT 结果，Optuna 列会显示 N/A；反之亦然。
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*WeightNorm.*', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '模型预测结果')

OAT_JSON  = os.path.join(_RESULT_DIR, 'param_search', 'param_search_best.json')
OPT_JSON  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')

_PARAM_EN = {
    'tcnCengShu':        'TCN Layers',
    'bilstmCengShu':     'BiLSTM Layers',
    'tcnYinCangWeiDu':   'TCN Hidden Dim',
    'bilstmYinCangWeiDu': 'BiLSTM Hidden Dim',
    'zhuYiLiTouShu':     'Attention Heads',
    'tuoQiLv':           'Dropout Rate',
    'piCiDaXiao':        'Batch Size',
    'xueXiLv':           'Learning Rate',
}

_ALL_PARAM_KEYS = [
    'tcnCengShu', 'bilstmCengShu', 'tcnYinCangWeiDu', 'bilstmYinCangWeiDu',
    'zhuYiLiTouShu', 'tuoQiLv', 'piCiDaXiao', 'xueXiLv',
]

_DEFAULTS = {
    'tcnCengShu': 4, 'bilstmCengShu': 2,
    'tcnYinCangWeiDu': 64, 'bilstmYinCangWeiDu': 64,
    'zhuYiLiTouShu': 4, 'tuoQiLv': 0.2,
    'piCiDaXiao': 64, 'xueXiLv': 0.001,
}


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return d.get('best_params', d)  # 兼容两种格式
    except Exception:
        return None


def _quick_eval(params: dict, epochs=60, label='') -> float:
    """用给定参数快速训练并返回验证集 MAPE"""
    import pickle
    import pandas as pd
    from torch.utils.data import DataLoader
    from data import DianLiFuHeShuJuJi, _TIME_FEATURE_COLS
    from model import TCNBiLstmZhuYiLiYuCeMoXing
    from train import xunLianDuiBiMoXing, pingGuMoXing
    from device_utils import get_best_device

    train_csv = os.path.join(_BASE_DIR, '数据', '训练集', 'train_data.csv')
    val_csv   = os.path.join(_BASE_DIR, '数据', '验证集', 'val_data.csv')
    scaler_p  = os.path.join(_BASE_DIR, '数据预处理', 'scalers.pkl')

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    scalers  = pickle.load(open(scaler_p, 'rb')) if os.path.exists(scaler_p) else {}

    def _extract(df):
        fh = df['M019Value'].values.reshape(-1,1).astype('float32')
        qx = df[['max_temp','min_temp','humidity','wind_speed']].values.astype('float32')
        sj = df[_TIME_FEATURE_COLS].values.astype('float32')
        mb = df['M019Value'].values.astype('float32')
        return fh, qx, sj, mb

    bs = int(params.get('piCiDaXiao', 64))
    tds = DianLiFuHeShuJuJi(*_extract(train_df), 96, 1)
    vds = DianLiFuHeShuJuJi(*_extract(val_df),   96, 1)
    tl  = DataLoader(tds, batch_size=bs, shuffle=True,  num_workers=0, pin_memory=True)
    vl  = DataLoader(vds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    moXing = TCNBiLstmZhuYiLiYuCeMoXing(
        fuHeTeZhengShu=1, qiXiangTeZhengShu=4, shiJianTeZhengShu=5,
        tcnYinCangWeiDu   = int(params.get('tcnYinCangWeiDu',   64)),
        tcnCengShu        = int(params.get('tcnCengShu',          4)),
        bilstmYinCangWeiDu= int(params.get('bilstmYinCangWeiDu', 64)),
        bilstmCengShu     = int(params.get('bilstmCengShu',       2)),
        zhuYiLiTouShu     = int(params.get('zhuYiLiTouShu',       4)),
        yuCeBuChang=1,
        tuoQiLv=float(params.get('tuoQiLv', 0.2)),
    )

    sheBei = get_best_device(verbose=False)
    _, best_state = xunLianDuiBiMoXing(
        moXing, tl, vl,
        sheBei=sheBei,
        xueXiLv=float(params.get('xueXiLv', 0.001)),
        xunLianDaiShu=epochs,
        scalers=scalers,
        zaoZhiTingZhiPatience=10,
        quiet=True,
        label=label,
    )
    moXing.load_state_dict(best_state)
    moXing.to(sheBei)
    res = pingGuMoXing(moXing, vl, sheBei, scalers)
    return res[2], res[3], res[6]   # MAPE, R², NRMSE


def _draw_comparison_table(oat_params, opt_params,
                            oat_metrics=None, opt_metrics=None,
                            save_path=None):
    """生成参数对比表格图（青绿色配色）"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    rows = []
    for k in _ALL_PARAM_KEYS:
        label = _PARAM_EN.get(k, k)
        ov = str(oat_params.get(k, 'N/A')) if oat_params else 'N/A'
        bv = str(opt_params.get(k, 'N/A')) if opt_params else 'N/A'
        diff = '' if ov == bv else '  ◀'
        rows.append([label, ov, bv, diff])

    # 评估指标行
    if oat_metrics or opt_metrics:
        rows.append(['─' * 12, '─' * 8, '─' * 8, ''])
        om = oat_metrics or {}
        bm = opt_metrics or {}
        rows.append(['Val MAPE (%)',
                      f"{om.get('mape','N/A'):.4f}" if isinstance(om.get('mape'), float) else 'N/A',
                      f"{bm.get('mape','N/A'):.4f}" if isinstance(bm.get('mape'), float) else 'N/A',
                      ''])
        rows.append([r'Val $R^2$',
                      f"{om.get('r2','N/A'):.4f}" if isinstance(om.get('r2'), float) else 'N/A',
                      f"{bm.get('r2','N/A'):.4f}" if isinstance(bm.get('r2'), float) else 'N/A',
                      ''])
        rows.append(['Val NRMSE (%)',
                      f"{om.get('nrmse','N/A'):.4f}" if isinstance(om.get('nrmse'), float) else 'N/A',
                      f"{bm.get('nrmse','N/A'):.4f}" if isinstance(bm.get('nrmse'), float) else 'N/A',
                      ''])

    n = len(rows)
    fig, ax = plt.subplots(figsize=(10, n * 0.52 + 0.9))
    ax.axis('off')
    col_labels = ['Hyperparameter', 'OAT Grid Search', 'Bayesian (Optuna)', 'Diff']
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.0, 0.0, 1.0, 1.0],
        colWidths=[0.32, 0.22, 0.28, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#b2dfdb')
        if r == 0:
            cell.set_facecolor('#009688')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 1:
            cell.set_facecolor('#e0f2f1')
        else:
            cell.set_facecolor('#ffffff')
        # 高亮有差异的行
        if r > 0 and c < 4 and rows[r - 1][3] == '  ◀':
            cell.set_facecolor('#fff9c4')

    # 添加"更优"标注
    if oat_metrics and opt_metrics:
        oat_m = oat_metrics.get('mape', float('inf'))
        opt_m = opt_metrics.get('mape', float('inf'))
        winner = 'OAT Grid Search' if oat_m <= opt_m else 'Bayesian (Optuna)'
        fig.suptitle(
            f'Hyperparameter Search Comparison  |  Better MAPE: {winner}',
            fontsize=12, fontweight='bold', y=0.98,
        )
    else:
        fig.suptitle('Hyperparameter Search Comparison (OAT vs Optuna)',
                     fontsize=12, fontweight='bold', y=0.98)

    plt.subplots_adjust(top=0.90, bottom=0.02, left=0.02, right=0.98)

    if save_path is None:
        save_path = os.path.join(_RESULT_DIR, 'param_search_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


def main(quick_eval=False, eval_epochs=60):
    """
    主函数：加载两种寻参结果，打印对比，可选做快速验证训练。

    quick_eval=True  : 用两组参数各自训练 eval_epochs 轮，比较验证集 MAPE
    quick_eval=False : 只对比参数值本身（不训练，秒级完成）
    """
    oat_params = _load_json(OAT_JSON)
    opt_params = _load_json(OPT_JSON)

    print("\n" + "=" * 60)
    print("Hyperparameter Search Method Comparison")
    print("=" * 60)
    print(f"  OAT Grid Search result: {OAT_JSON}")
    print(f"  Bayesian (Optuna) result: {OPT_JSON}")
    print()

    if oat_params is None and opt_params is None:
        print("  Neither result file found. Run main.py first.")
        return

    # 打印参数对比表
    print(f"  {'Parameter':<22} {'OAT':>12}  {'Optuna':>12}  Diff")
    print("  " + "-" * 54)
    for k in _ALL_PARAM_KEYS:
        label = _PARAM_EN.get(k, k)
        ov = str(oat_params.get(k, 'N/A')) if oat_params else 'N/A'
        bv = str(opt_params.get(k, 'N/A')) if opt_params else 'N/A'
        flag = '' if ov == bv else '  ← different'
        print(f"  {label:<22} {ov:>12}  {bv:>12}{flag}")

    oat_metrics = None
    opt_metrics = None

    if quick_eval:
        print("\nQuick validation training to compare MAPE...")
        if oat_params:
            print(f"\n  [OAT] Training {eval_epochs} epochs...")
            m, r2, n = _quick_eval(oat_params, eval_epochs, label='OAT')
            oat_metrics = {'mape': m, 'r2': r2, 'nrmse': n}
            print(f"  [OAT]    Val MAPE={m:.4f}%  R²={r2:.4f}  NRMSE={n:.4f}%")

        if opt_params:
            print(f"\n  [Optuna] Training {eval_epochs} epochs...")
            m, r2, n = _quick_eval(opt_params, eval_epochs, label='Optuna')
            opt_metrics = {'mape': m, 'r2': r2, 'nrmse': n}
            print(f"  [Optuna] Val MAPE={m:.4f}%  R²={r2:.4f}  NRMSE={n:.4f}%")

        if oat_metrics and opt_metrics:
            if oat_metrics['mape'] <= opt_metrics['mape']:
                print(f"\n  ★ OAT is better  (MAPE {oat_metrics['mape']:.4f}% vs {opt_metrics['mape']:.4f}%)")
            else:
                print(f"\n  ★ Optuna is better  (MAPE {opt_metrics['mape']:.4f}% vs {oat_metrics['mape']:.4f}%)")

    # 生成对比表格图
    _draw_comparison_table(oat_params, opt_params, oat_metrics, opt_metrics)
    print("\nDone.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare OAT vs Optuna hyperparameter search results')
    parser.add_argument('--eval', action='store_true',
                        help='Quick train both param sets and compare MAPE on validation set')
    parser.add_argument('--eval_epochs', type=int, default=60,
                        help='Epochs for quick evaluation (default: 60)')
    args = parser.parse_args()
    main(quick_eval=args.eval, eval_epochs=args.eval_epochs)

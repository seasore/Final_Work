# -*- coding: utf-8 -*-
"""
主程序入口
Main Program Entry

TCN-BiLSTM-Attention 短期电力负荷预测系统

完整流程：
  1. [可选] 超参数网格探索（步骤 0）+ 可选轮数收敛曲线（仍属步骤 0）
  2. [可选] Optuna 贝叶斯优化（步骤 1，在加载数据、训练主模型之前执行，不是主模型跑完之后）
  3. 加载数据并用最终 best_params 训练主模型（打印为「步骤 2」）及基线、评估、出图

寻参完成后可仅从训练开始：python main.py --from_train
  4. 训练所有对比模型（LSTM / BiLSTM / TCN / BiLSTM-Attention）
  5. [可选] 时序 K 折交叉验证
  6. 测试集综合评估 + 特殊场景验证
  7. 生成所有对比图表与报告（论文图6/7/8/9 风格 + 表2/3 风格）
"""

import sys
import io
import argparse
import warnings
# 过滤 PyTorch weight_norm 弃用警告（不影响运行，仅减少输出噪音）
warnings.filterwarnings('ignore', message='.*weight_norm.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*WeightNorm.*', category=FutureWarning)

if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
import shutil
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BEST_PARAMS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_params.json')
BEST_MODEL_PT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pt')
# 对比模型权重目录（供后端 prediction_service 多曲线滚动预测）
BASELINE_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline_checkpoints')

from model import TCNBiLstmZhuYiLiYuCeMoXing
from baselines import chuangJianDuiBiMoXing
from device_utils import get_best_device
from data import jiaZaiYuChuLiShuJu
from train import (
    xunLianMoXing, xunLianDuiBiMoXing, pingGuMoXing,
    ceShiYuCeXiangYingShiJian,
)
from visualize import (
    shengChengSuoYouTuBiao, RESULT_DIR_NAME, _qingKongJieGuoMuLu,
)


# ──────────────────────────────────────────────────────────────
# 各对比模型默认超参数配置（用于论文表2）
# ──────────────────────────────────────────────────────────────
_BASELINE_CONFIGS = {
    'LSTM': {
        'params': {'yinCangWeiDu': 64, 'cengShu': 2, 'tuoQiLv': 0.2},
        'xueXiLv': 0.001,
        'xunLianDaiShu': 60,
        'Layers': 2, 'Hidden Dim': 'All 64', 'Epochs': 60, 'Batch Size': 64, 'Learning Rate': 0.001,
    },
    'BiLSTM': {
        'params': {'yinCangWeiDu': 64, 'cengShu': 2, 'tuoQiLv': 0.2},
        'xueXiLv': 0.001,
        'xunLianDaiShu': 60,
        'Layers': 2, 'Hidden Dim': 'All 64', 'Epochs': 60, 'Batch Size': 64, 'Learning Rate': 0.001,
    },
    'TCN': {
        'params': {'tcnYinCangWeiDu': 64, 'tcnCengShu': 4, 'tuoQiLv': 0.2},
        'xueXiLv': 0.001,
        'xunLianDaiShu': 60,
        'Layers': 4, 'Hidden Dim': 'All 64', 'Epochs': 60, 'Batch Size': 64, 'Learning Rate': 0.001,
    },
    'BiLSTM-Attention': {
        'params': {'yinCangWeiDu': 64, 'cengShu': 2, 'zhuYiLiTouShu': 4, 'tuoQiLv': 0.2},
        'xueXiLv': 0.001,
        'xunLianDaiShu': 60,
        'Layers': 2, 'Hidden Dim': 'All 64', 'Epochs': 60, 'Batch Size': 64, 'Learning Rate': 0.001,
    },
}


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────

def _jiaZaiBestParams():
    if not os.path.exists(BEST_PARAMS_JSON):
        return None
    try:
        with open(BEST_PARAMS_JSON, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if not d or 'best_params' not in d:
            return None
        return d['best_params']
    except (json.JSONDecodeError, KeyError):
        return None


def _json_sanitize(obj):
    """将 numpy 标量/数组等转为 json 可序列化类型（json 不认 np.float32 等）。"""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _baoChunBestParams(best_params):
    d = {'best_params': _json_sanitize(best_params)}
    with open(BEST_PARAMS_JSON, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def _pingGuMoXingComplete(moXing, jiaZaiQi, sheBei, scalers):
    """封装 pingGuMoXing，兼容 7 返回值"""
    res = pingGuMoXing(moXing, jiaZaiQi, sheBei, scalers)
    mse, mae, mape, r2 = res[0], res[1], res[2], res[3]
    smape = res[4] if len(res) > 4 else mape
    wape = res[5] if len(res) > 5 else mape
    nrmse = res[6] if len(res) > 6 else 0.0
    return mse, mae, mape, r2, smape, wape, nrmse


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def zhuHanShu(
    run_param_search=False,
    run_optuna=True,
    run_cv=True,
    optuna_trials=30,
    optuna_epochs=80,
    force_optuna=False,
    optuna_metric='mape',
    param_search_params=None,
    run_epoch_curve=False,
    param_search_n_seeds=None,
    baseline_epochs=60,
    main_epochs=80,
    from_train=False,
    early_stop_patience=10,
    mape_min_delta=0.005,
    epoch_curve_only=False,
    param_search_export=False,
):
    """
    主函数：超参数探索 -> Optuna -> 训练主模型 -> 训练对比模型 -> 评估 -> 生成图表

    参数:
        run_param_search: 是否运行参数网格探索（仿论文Section 3）
        run_optuna: 是否运行 Optuna 贝叶斯优化
        run_cv: 是否运行时序 K 折交叉验证
        optuna_trials: Optuna 试验次数
        optuna_epochs: Optuna 每次试验的训练轮数
        force_optuna: 强制重新运行 Optuna（忽略已保存的 best_params.json）
        optuna_metric: Optuna 优化目标 ('mape' 或 'wape')
        param_search_params: 要探索的超参数列表（None 表示全部）
        run_epoch_curve: 是否绘制训练轮数收敛曲线
        baseline_epochs: 对比模型训练轮数
        main_epochs: 主模型最终训练轮数
        from_train: 为 True 时跳过步骤 0–1，从加载数据与训练主模型开始（需已有 best_params.json 或 param_search_best.json）
        early_stop_patience: 主模型与基线早停耐心轮数
        mape_min_delta: 早停判定中「MAPE 改善」的最小幅度（百分点），避免验证 MAPE 噪声反复重置计数导致跑满轮数
        epoch_curve_only: 为 True 时只运行轮数收敛曲线（tanSuoXunLianDaiShu），不清空结果目录、不重跑网格/Optuna/主模型
        param_search_export: 为 True 时将评估图、报告、最佳权重与 used_best_params.json 同步保存到
            模型预测结果/param_search/main_from_train_<时间戳>/（子目录不受根目录清空影响）
    """
    if from_train:
        run_param_search = False
        run_optuna = False

    if epoch_curve_only:
        run_param_search = False
        run_optuna = False
        from_train = False

    sheBei = get_best_device(verbose=True)
    print("提示：如需中途停止，请在终端按 Ctrl+C")

    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULT_DIR_NAME)
    os.makedirs(result_dir, exist_ok=True)

    if epoch_curve_only:
        from param_search import tanSuoXunLianDaiShu, PARAM_RESULT_DIR
        print("\n" + "=" * 60)
        print("仅运行：训练轮数收敛曲线（默认固定超参、无早停、训满 max(epoch_list) 轮）")
        print("=" * 60)
        tanSuoXunLianDaiShu(quiet=False)
        print("\n输出文件：")
        print(f"  {os.path.join(PARAM_RESULT_DIR, 'param_epoch_convergence.png')}")
        print(f"  {os.path.join(PARAM_RESULT_DIR, 'param_epoch.csv')}")
        print("\n下一步可选：")
        print("  Optuna：python main.py --force_optuna")
        print("  仅训主模型：python main.py --from_train")
        return None, None, None, None

    _qingKongJieGuoMuLu(result_dir)

    xuLieChangDu = 96
    yuCeBuChang = 1
    piCiDaXiao = 64
    shiJianTeZhengShu = 5
    best_params = {}
    oat_best_params = {}

    if from_train:
        print("\n" + "=" * 60)
        print("从步骤2开始：已跳过超参数网格探索与 Optuna，直接加载数据并训练主模型")
        print("=" * 60)
        loaded = _jiaZaiBestParams()
        if loaded is not None:
            best_params = dict(loaded)
            print(f"  已加载超参数: {BEST_PARAMS_JSON}")
        else:
            oat_json = os.path.join(result_dir, 'param_search', 'param_search_best.json')
            if os.path.exists(oat_json):
                with open(oat_json, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)
                print(f"  已加载超参数: {oat_json}")
            else:
                print(
                    "  警告: 未找到 best_params.json 与 param_search/param_search_best.json，"
                    "将使用代码内置默认超参数。"
                )
        oat_best_params = dict(best_params)
        for k, v in sorted(best_params.items()):
            print(f"    {k}: {v}")

    # ──────────────────────────────
    # 步骤 0：超参数网格探索（可选，仿论文Section 3）
    # ──────────────────────────────
    if run_param_search:
        from param_search import yunXingCanShuTanSuo, tanSuoXunLianDaiShu, PARAM_RESULT_DIR
        print("\n" + "=" * 60)
        print("步骤 0: 超参数网格探索（仿论文 Section 3 逐一探索）")
        print("=" * 60)
        yunXingCanShuTanSuo(params_to_search=param_search_params, quiet=False,
                            n_seeds=param_search_n_seeds)
        if run_epoch_curve:
            tanSuoXunLianDaiShu(quiet=False)

        # 尝试读取探索最优值作为初始 best_params
        param_best_json = os.path.join(PARAM_RESULT_DIR, 'param_search_best.json')
        if os.path.exists(param_best_json):
            with open(param_best_json, 'r', encoding='utf-8') as f:
                param_best = json.load(f)
            best_params.update(param_best)
            print(f"\n  已从网格探索获得初始参数: {param_best_json}")

    # 本次未跑网格但磁盘上已有 OAT 结果时，读入以便步骤 1 的 Optuna 作 seed（例如上次中断在轮数曲线之后）
    if not from_train and not run_param_search and not best_params:
        oat_json_disk = os.path.join(result_dir, 'param_search', 'param_search_best.json')
        if os.path.exists(oat_json_disk):
            with open(oat_json_disk, 'r', encoding='utf-8') as f:
                best_params = json.load(f)
            print(f"\n  已从磁盘载入 OAT 最优参数（供 Optuna 作初始种子）: {oat_json_disk}")

    # ──────────────────────────────
    # 步骤 1：Optuna 贝叶斯超参数优化（在主模型训练之前执行，不是训练之后）
    # ──────────────────────────────
    if not from_train:
        oat_best_params = dict(best_params)   # 保留 OAT 结果，以便事后对比

    if run_optuna:
        cached = _jiaZaiBestParams() if not force_optuna else None
        if cached is not None and not force_optuna:
            best_params = cached
            print("\n" + "=" * 60)
            print("步骤 1: 使用已保存的 Optuna 最优超参数（跳过重跑）")
            print("=" * 60)
            for k, v in best_params.items():
                print(f"  {k}: {v}")
        else:
            try:
                from optuna_search import run_optuna_search
                print("\n" + "=" * 60)
                print("步骤 1: Optuna 贝叶斯超参数优化")
                print("=" * 60)
                # 将 OAT 结果作为 Optuna 种子，引导初始搜索方向
                seed = oat_best_params if oat_best_params else None
                study = run_optuna_search(
                    n_trials=optuna_trials,
                    n_trials_epochs=optuna_epochs,
                    optuna_metric=optuna_metric,
                    seed_params=seed,
                )
                best_params = study.best_params
                _baoChunBestParams(best_params)
                print(f"\n  已保存到: {BEST_PARAMS_JSON}")
            except ImportError:
                print(f"\n跳过 Optuna（未安装）: pip install optuna")
            except Exception as e:
                print(f"\nOptuna 运行异常，使用 OAT 参数继续: {e}")

    # 如果两种方法都运行了，打印对比提示
    if run_param_search and run_optuna and oat_best_params and best_params:
        print("\n" + "=" * 60)
        print("参数来源对比（OAT 网格 vs Optuna 贝叶斯）：")
        print(f"  {'超参数':<22} {'OAT 最优':>12}  {'Optuna 最优':>12}")
        print("  " + "-" * 48)
        all_keys = set(oat_best_params) | set(best_params)
        for k in sorted(all_keys):
            ov = oat_best_params.get(k, '-')
            bv = best_params.get(k, '-')
            flag = '  ←相同' if str(ov) == str(bv) else '  ← 不同！'
            print(f"  {k:<22} {str(ov):>12}  {str(bv):>12}{flag}")
        print("=" * 60)
        print("  将使用 Optuna 结果作为最终参数训练主模型")

    # ──────────────────────────────
    # 步骤 2：加载数据
    # ──────────────────────────────
    piCiDaXiao = int(best_params.get('piCiDaXiao', 64))
    print("\n加载预处理数据（训练/验证/测试三分）...", flush=True)
    xunLianJiaZaiQi, yanZhengJiaZaiQi, ceShiJiaZaiQi, scalers = jiaZaiYuChuLiShuJu(
        xuLieChangDu=xuLieChangDu,
        yuCeBuChang=yuCeBuChang,
        piCiDaXiao=piCiDaXiao,
    )
    print(
        f"  训练样本: {len(xunLianJiaZaiQi.dataset)}, "
        f"验证样本: {len(yanZhengJiaZaiQi.dataset)}, "
        f"测试样本: {len(ceShiJiaZaiQi.dataset)}"
    )

    # ──────────────────────────────
    # 步骤 3：训练主模型（TCN-BiLSTM-Attention）
    # ──────────────────────────────
    xueXiLv = best_params.get('xueXiLv', 0.001)
    print("\n" + "=" * 60)
    print("步骤 2: 训练主模型（TCN-BiLSTM-Attention）")
    print("=" * 60)
    moXing = TCNBiLstmZhuYiLiYuCeMoXing(
        fuHeTeZhengShu=1,
        qiXiangTeZhengShu=4,
        shiJianTeZhengShu=shiJianTeZhengShu,
        tcnYinCangWeiDu=best_params.get('tcnYinCangWeiDu', 64),
        tcnCengShu=best_params.get('tcnCengShu', 4),
        bilstmYinCangWeiDu=best_params.get('bilstmYinCangWeiDu', 64),
        bilstmCengShu=best_params.get('bilstmCengShu', 2),
        zhuYiLiTouShu=best_params.get('zhuYiLiTouShu', 4),
        yuCeBuChang=yuCeBuChang,
        tuoQiLv=best_params.get('tuoQiLv', 0.2),
    )
    print(f"  模型参数量: {sum(p.numel() for p in moXing.parameters()):,}")
    # 固定随机种子，保证主模型训练结果可复现
    _TRAIN_SEED = 42
    torch.manual_seed(_TRAIN_SEED)
    np.random.seed(_TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_TRAIN_SEED)
    print(f"\n  已固定随机种子: {_TRAIN_SEED}（保证可复现）", flush=True)
    print("\n  开始训练...", flush=True)
    xunLianLiShi, zuiJiaZhuangTai = xunLianMoXing(
        moXing, xunLianJiaZaiQi, yanZhengJiaZaiQi,
        sheBei=sheBei, xueXiLv=xueXiLv,
        xunLianDaiShu=main_epochs,
        scalers=scalers,
        lingChenQuanZhong=1.0,
        zaoZhiTingZhiPatience=early_stop_patience,
        mape_min_delta=mape_min_delta,
    )
    moXing.load_state_dict(zuiJiaZhuangTai)
    torch.save({'state_dict': zuiJiaZhuangTai, 'best_params': best_params}, BEST_MODEL_PT)
    print(f"\n  主模型已保存到: {BEST_MODEL_PT}")

    # ──────────────────────────────
    # 步骤 4：训练对比基线模型
    # ──────────────────────────────
    print("\n" + "=" * 60)
    print("步骤 3: 训练对比基线模型（LSTM / BiLSTM / TCN / BiLSTM-Attention）")
    print("=" * 60)
    # 与主模型共用 best_params 中的学习率，避免「主模型用寻参 LR、基线写死 0.001」导致的不可比
    duiBiXueXiLv = float(best_params.get('xueXiLv', 0.001))
    print(f"  基线学习率与主模型对齐: {duiBiXueXiLv}（来自 best_params）")

    models_dict = {'TCN-BiLSTM-Attention': moXing}
    metrics_dict = {}
    model_configs = {
        'TCN-BiLSTM-Attention': {
            'Layers': f"TCN:{best_params.get('tcnCengShu', 4)}, BiLSTM:{best_params.get('bilstmCengShu', 2)}",
            'Hidden Dim': f"TCN:{best_params.get('tcnYinCangWeiDu', 64)}, BiLSTM:{best_params.get('bilstmYinCangWeiDu', 64)}",
            'Epochs': main_epochs,
            'Batch Size': int(piCiDaXiao),
            'Learning Rate': xueXiLv,
        }
    }

    for name, cfg in _BASELINE_CONFIGS.items():
        print(f"\n  训练 {name}...", flush=True)
        bl_moXing = chuangJianDuiBiMoXing(name, cfg['params'], yuCeBuChang)
        _, bl_zuiJia = xunLianDuiBiMoXing(
            bl_moXing, xunLianJiaZaiQi, yanZhengJiaZaiQi,
            sheBei=sheBei,
            xueXiLv=duiBiXueXiLv,
            xunLianDaiShu=cfg.get('xunLianDaiShu', baseline_epochs),
            scalers=scalers,
            zaoZhiTingZhiPatience=early_stop_patience,
            mape_min_delta=mape_min_delta,
            quiet=True,
            label=name,
        )
        bl_moXing.load_state_dict(bl_zuiJia)
        bl_moXing.to(sheBei)
        models_dict[name] = bl_moXing
        try:
            os.makedirs(BASELINE_CHECKPOINT_DIR, exist_ok=True)
            safe_fn = name.replace('-', '_') + '.pt'
            torch.save(
                {
                    'state_dict': bl_zuiJia,
                    'model_name': name,
                    'params': dict(cfg['params']),
                },
                os.path.join(BASELINE_CHECKPOINT_DIR, safe_fn),
            )
            print(f"  已保存基线权重: {BASELINE_CHECKPOINT_DIR}/{safe_fn}", flush=True)
        except Exception as _e:
            print(f"  警告: 基线权重未写入磁盘 ({name}): {_e}", flush=True)
        model_configs[name] = {
            'Layers': cfg['Layers'],
            'Hidden Dim': cfg['Hidden Dim'],
            'Epochs': cfg['Epochs'],
            'Batch Size': int(piCiDaXiao),
            'Learning Rate': duiBiXueXiLv,
        }

    # ──────────────────────────────
    # 步骤 5：时序 K 折交叉验证（可选）
    # ──────────────────────────────
    if run_cv:
        try:
            from cross_validate import shiXuKZheJiaoChaYanZheng
            print("\n" + "=" * 60)
            print("步骤 4: 时序 K 折交叉验证")
            print("=" * 60)
            mape_list, mean_mape, std_mape = shiXuKZheJiaoChaYanZheng(
                n_splits=5, xunLianDaiShu=30, best_params=best_params, sheBei=sheBei
            )
            print(f"  交叉验证 MAPE: {mean_mape:.4f}% ± {std_mape:.4f}%")
        except Exception as e:
            print(f"\n交叉验证跳过: {e}")

    # ──────────────────────────────
    # 步骤 6：测试集评估（所有模型）
    # ──────────────────────────────
    print("\n" + "=" * 60)
    print("步骤 5: 测试集评估（所有模型）")
    print("=" * 60)

    all_names_ordered = ['LSTM', 'BiLSTM', 'TCN', 'BiLSTM-Attention', 'TCN-BiLSTM-Attention']
    print(f"\n{'模型':<25} {'MAPE(%)':>10} {'NRMSE(%)':>10} {'R²':>8}")
    print("-" * 60)
    for name in all_names_ordered:
        if name not in models_dict:
            continue
        m = models_dict[name]
        res = _pingGuMoXingComplete(m, ceShiJiaZaiQi, sheBei, scalers)
        mse_i, mae_i, mape_i, r2_i, smape_i, wape_i, nrmse_i = res
        metrics_dict[name] = {'MAPE(%)': mape_i, 'NRMSE(%)': nrmse_i, 'R²': r2_i,
                               'MSE': mse_i, 'MAE': mae_i, 'SMAPE': smape_i, 'WAPE': wape_i}
        marker = ' ◀ 主模型' if name == 'TCN-BiLSTM-Attention' else ''
        print(f"  {name:<23} {mape_i:>10.4f} {nrmse_i:>10.4f} {r2_i:>8.4f}{marker}")

    # 主模型指标
    main_metrics = metrics_dict.get('TCN-BiLSTM-Attention', {})
    mse = main_metrics.get('MSE', 0)
    mae = main_metrics.get('MAE', 0)
    mape = main_metrics.get('MAPE(%)', 0)
    r2 = main_metrics.get('R²', 0)
    smape = main_metrics.get('SMAPE', 0)
    wape = main_metrics.get('WAPE', 0)
    nrmse = main_metrics.get('NRMSE(%)', 0)
    acc = max(0, 100 - mape)

    # LSTM 基线 MAPE（用于 MAPE 降幅计算）
    mapeJiChu = metrics_dict.get('LSTM', {}).get('MAPE(%)', None)
    jiangDi = (mapeJiChu - mape) / mapeJiChu * 100 if mapeJiChu and mapeJiChu > 0 else None

    if jiangDi is not None:
        print(f"\n  相对 LSTM 基线 MAPE 降低: {jiangDi:.1f}%"
              f" {'(满足>15%要求)' if jiangDi >= 15 else '(未达15%)'}")

    # 特殊场景验证
    teShuJieGuo = {}
    _base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(_base_dir, '数据', '测试集', 'test_data.csv')
    if os.path.exists(test_csv) and scalers:
        test_df = pd.read_csv(test_csv)
        test_df['DATETIME'] = pd.to_datetime(test_df['DATETIME'])
        if 'is_holiday' in test_df.columns:
            try:
                from special_scenario_eval import pingGuTeShuChangJing
                teShuJieGuo = pingGuTeShuChangJing(
                    moXing, ceShiJiaZaiQi, test_df, xuLieChangDu, sheBei, scalers
                )
                if teShuJieGuo:
                    print("\n特殊场景 MAPE:")
                    for sname, (n_, m_) in teShuJieGuo.items():
                        print(f"  {sname}: n={n_}, MAPE={m_:.4f}%")
            except Exception as e:
                print(f"\n特殊场景评估跳过: {e}")

    # 预测响应时间
    print("\n预测响应时间测试:")
    ceShiYangBen = next(iter(ceShiJiaZaiQi))
    fuHe0, qiXiang0, shiJian0, _ = ceShiYangBen
    pingJunHaoMiao = ceShiYuCeXiangYingShiJian(
        moXing, (fuHe0[0], qiXiang0[0], shiJian0[0]), sheBei, chongFuCiShu=100
    )
    print(f"  单次预测平均耗时: {pingJunHaoMiao:.2f} 毫秒")
    print(f"  {'满足' if pingJunHaoMiao < 60000 else '未满足'}实时性要求 (≤1分钟)")

    # ──────────────────────────────
    # 步骤 7：生成所有图表和报告
    # ──────────────────────────────
    print("\n" + "=" * 60)
    print("步骤 6: 生成图表与评估报告")
    print("=" * 60)
    param_search_copy_dir = None
    if param_search_export:
        param_search_copy_dir = os.path.join(
            result_dir, 'param_search',
            f'main_from_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        )
        os.makedirs(os.path.join(result_dir, 'param_search'), exist_ok=True)
        print(f"  本次将额外保存到: {param_search_copy_dir}")

    shengChengSuoYouTuBiao(
        moXing, xunLianLiShi, ceShiJiaZaiQi, sheBei, scalers,
        mse, mae, mape, r2,
        mapeJiChu=mapeJiChu,
        jiangDi=jiangDi,
        pingJunHaoMiao=pingJunHaoMiao,
        xunLianDaiShu=main_epochs,
        teShuChangJingJieGuo=teShuJieGuo or None,
        smape=smape,
        wape=wape,
        acc=acc,
        nrmse=nrmse,
        models_dict=models_dict,
        metrics_dict=metrics_dict,
        model_configs=model_configs,
        copy_to_dir=param_search_copy_dir,
    )

    if param_search_copy_dir and os.path.isdir(param_search_copy_dir):
        try:
            if os.path.isfile(BEST_MODEL_PT):
                shutil.copy2(BEST_MODEL_PT, os.path.join(param_search_copy_dir, 'best_model.pt'))
            with open(os.path.join(param_search_copy_dir, 'used_best_params.json'),
                      'w', encoding='utf-8') as fp:
                json.dump(_json_sanitize(best_params), fp, indent=2, ensure_ascii=False)
            # 测试集主模型指标摘要，便于与旧实验对照
            with open(os.path.join(param_search_copy_dir, 'test_metrics_main.json'),
                      'w', encoding='utf-8') as fp:
                json.dump(_json_sanitize({
                    'MAPE(%)': mape, 'NRMSE(%)': nrmse, 'R²': r2,
                    'MSE': mse, 'MAE': mae, 'SMAPE': smape, 'WAPE': wape,
                }), fp, indent=2, ensure_ascii=False)
            print(f"  已写入: {param_search_copy_dir}/best_model.pt 与 used_best_params.json、test_metrics_main.json")
        except OSError as e:
            print(f"  警告: 复制到 param_search 子目录失败: {e}")

    return moXing, xunLianLiShi, models_dict, metrics_dict


# ──────────────────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TCN-BiLSTM-Attention 电力负荷预测完整训练流程')

    # Optuna 相关
    parser.add_argument('--no_optuna', action='store_true', help='跳过 Optuna')
    parser.add_argument('--force_optuna', action='store_true', help='强制重新运行 Optuna')
    parser.add_argument('--optuna_trials', type=int, default=30)
    parser.add_argument('--optuna_epochs', type=int, default=80,
                        help='Optuna 每个 trial 最大训练轮数（建议与主模型 main_epochs 对齐，双重早停保证坏参数提前退出）')
    parser.add_argument('--optuna_metric', type=str, default='mape', choices=['mape', 'wape'])

    # 参数探索相关（仿论文Section 3）
    parser.add_argument('--param_search', action='store_true',
                        help='运行超参数网格探索（仿论文Section 3）')
    parser.add_argument('--param_search_params', nargs='+',
                        default=None,
                        help='要探索的参数名，不指定则探索全部')
    parser.add_argument('--epoch_curve', action='store_true',
                        help='额外绘制训练轮数收敛曲线')
    parser.add_argument('--param_search_n_seeds', type=int, default=None,
                        help='网格搜索每个候选值的随机种子数（默认3，传1退化为单次）')
    parser.add_argument(
        '--epoch_curve_only',
        action='store_true',
        help='只重跑轮数收敛曲线（不跑网格/Optuna/主模型；不清空模型预测结果目录）',
    )

    # 训练相关
    parser.add_argument('--no_cv', action='store_true', help='跳过交叉验证')
    parser.add_argument('--baseline_epochs', type=int, default=60, help='对比模型训练轮数')
    parser.add_argument('--main_epochs', type=int, default=80, help='主模型训练轮数')
    parser.add_argument(
        '--from_train',
        action='store_true',
        help='跳过步骤0–1（网格/Optuna），从加载数据并训练主模型开始；优先读 best_params.json，否则 param_search/param_search_best.json',
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=10,
        help='验证 MAPE 早停耐心轮数（与 mape_min_delta 配合）',
    )
    parser.add_argument(
        '--mape_min_delta',
        type=float,
        default=0.005,
        help='早停判定中 MAPE 至少下降多少百分点才算改善（抑制噪声导致永不早停）',
    )
    parser.add_argument(
        '--param_search_export',
        action='store_true',
        help='将本运行图表/报告/权重/超参副本保存到 模型预测结果/param_search/main_from_train_<时间戳>/',
    )

    args = parser.parse_args()

    if args.epoch_curve_only and args.from_train:
        parser.error('--epoch_curve_only 与 --from_train 不能同时使用')

    zhuHanShu(
        run_param_search=args.param_search,
        run_optuna=not args.no_optuna,
        run_cv=not args.no_cv,
        optuna_trials=args.optuna_trials,
        optuna_epochs=args.optuna_epochs,
        force_optuna=args.force_optuna,
        optuna_metric=args.optuna_metric,
        param_search_params=args.param_search_params,
        run_epoch_curve=args.epoch_curve,
        param_search_n_seeds=args.param_search_n_seeds,
        baseline_epochs=args.baseline_epochs,
        main_epochs=args.main_epochs,
        from_train=args.from_train,
        early_stop_patience=args.early_stop_patience,
        mape_min_delta=args.mape_min_delta,
        epoch_curve_only=args.epoch_curve_only,
        param_search_export=args.param_search_export,
    )

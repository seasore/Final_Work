# -*- coding: utf-8 -*-
"""
贝叶斯超参数优化模块（Optuna）
Bayesian Hyperparameter Optimization Module

开题报告要求：采用贝叶斯优化或遗传算法等智能优化方法，自动搜索超参数的最优组合
选用 Optuna（贝叶斯优化），样本效率高，适合神经网络超参数搜索
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader

from data import DianLiFuHeShuJuJi, _TIME_FEATURE_COLS
from model import TCNBiLstmZhuYiLiYuCeMoXing
from train import xunLianMoXing, pingGuMoXing
from device_utils import get_best_device


def _create_objective(base_dir, n_trials_epochs=40, optuna_metric='mape'):
    """构建 Optuna 目标函数，支持剪枝以提前终止表现差的 trial"""
    
    def objective(trial):
        # 搜索空间与 param_search._SEARCH_SPACE 对齐，避免 OAT 种子 enqueue_trial 与分布冲突
        xueXiLv = trial.suggest_float('xueXiLv', 1e-4, 1e-2, log=True)
        tcnYinCangWeiDu = trial.suggest_categorical(
            'tcnYinCangWeiDu', [16, 32, 48, 64, 96, 128]
        )
        tcnCengShu = trial.suggest_int('tcnCengShu', 1, 8)
        bilstmYinCangWeiDu = trial.suggest_categorical(
            'bilstmYinCangWeiDu', [16, 32, 48, 64, 96, 128]
        )
        bilstmCengShu = trial.suggest_int('bilstmCengShu', 1, 4)
        zhuYiLiTouShu = trial.suggest_categorical('zhuYiLiTouShu', [2, 4, 8])
        tuoQiLv = trial.suggest_float('tuoQiLv', 0.1, 0.5)
        piCiDaXiao = trial.suggest_categorical('piCiDaXiao', [16, 32, 64, 128, 256])
        
        xuLieChangDu = 96
        yuCeBuChang = 1
        shiJianTeZhengShu = 5
        sheBei = get_best_device(verbose=False)
        
        train_path = os.path.join(base_dir, '数据', '训练集', 'train_data.csv')
        val_path = os.path.join(base_dir, '数据', '验证集', 'val_data.csv')
        scaler_path = os.path.join(base_dir, '数据预处理', 'scalers.pkl')
        
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError("请先运行数据预处理")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
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
        
        train_arrays = _extract_arrays(train_df)
        val_arrays = _extract_arrays(val_df)
        
        train_ds = DianLiFuHeShuJuJi(*train_arrays, xuLieChangDu, yuCeBuChang)
        val_ds = DianLiFuHeShuJuJi(*val_arrays, xuLieChangDu, yuCeBuChang)
        
        train_loader = DataLoader(train_ds, batch_size=piCiDaXiao, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=piCiDaXiao, shuffle=False, num_workers=0)
        
        moXing = TCNBiLstmZhuYiLiYuCeMoXing(
            fuHeTeZhengShu=1,
            qiXiangTeZhengShu=4,
            shiJianTeZhengShu=shiJianTeZhengShu,
            tcnYinCangWeiDu=tcnYinCangWeiDu,
            tcnCengShu=tcnCengShu,
            bilstmYinCangWeiDu=bilstmYinCangWeiDu,
            bilstmCengShu=bilstmCengShu,
            zhuYiLiTouShu=zhuYiLiTouShu,
            yuCeBuChang=yuCeBuChang,
            tuoQiLv=tuoQiLv
        )
        
        def _prune_callback(epoch, val_loss, metrics):
            val = metrics.get(optuna_metric, metrics.get('mape'))
            trial.report(val, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        
        xunLianLiShi, zuiJiaZhuangTai = xunLianMoXing(
            moXing, train_loader, val_loader,
            sheBei=sheBei, xueXiLv=xueXiLv, xunLianDaiShu=n_trials_epochs,
            scalers=scalers, lingChenQuanZhong=1.0,
            per_epoch_callback=_prune_callback, quiet=False
        )
        moXing.load_state_dict(zuiJiaZhuangTai)
        res = pingGuMoXing(moXing, val_loader, sheBei, scalers)
        mse, mae, mape, r2, smape, wape = res[0], res[1], res[2], res[3], res[4], res[5]
        if optuna_metric == 'wape':
            return wape
        return mape
    
    return objective


def run_optuna_search(
    n_trials=50,
    n_trials_epochs=80,
    study_name='tcn_bilstm_attention',
    verbose=True,
    optuna_metric='mape',
    seed_params=None,
):
    """
    运行 Optuna 贝叶斯超参数搜索。

    参数:
        n_trials: 尝试的超参数组合数量（建议 ≥ 50，搜索空间较大）
        n_trials_epochs: 每次尝试的训练轮数
        study_name: 研究名称
        optuna_metric: 优化目标 'mape' 或 'wape'
        seed_params: 可选 dict，将已知好参数作为首个 trial 的种子（如 OAT 结果）
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("请安装 Optuna: pip install optuna")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    objective = _create_objective(base_dir, n_trials_epochs, optuna_metric)

    # MedianPruner: 表现差的 trial 提前终止，节省时间
    # n_startup_trials=8 保证足够多早期探索再开始剪枝
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=8)
    sampler = optuna.samplers.TPESampler(seed=42)   # 固定随机种子，保证可复现
    study = optuna.create_study(
        direction='minimize',
        study_name=study_name,
        pruner=pruner,
        sampler=sampler,
    )

    # 用 OAT 网格搜索结果作为种子：将已知好参数加入 study，引导 TPE 采样方向
    if seed_params:
        _VALID_KEYS = {
            'xueXiLv', 'tcnYinCangWeiDu', 'tcnCengShu',
            'bilstmYinCangWeiDu', 'bilstmCengShu',
            'zhuYiLiTouShu', 'tuoQiLv', 'piCiDaXiao',
        }
        clean = {k: v for k, v in seed_params.items() if k in _VALID_KEYS}
        if clean:
            # 确保 categorical 参数类型一致
            for k in ('tcnYinCangWeiDu', 'bilstmYinCangWeiDu', 'zhuYiLiTouShu', 'piCiDaXiao'):
                if k in clean:
                    clean[k] = int(clean[k])
            for k in ('tcnCengShu', 'bilstmCengShu'):
                if k in clean:
                    clean[k] = int(clean[k])
            study.enqueue_trial(clean)
            print(f"  [Optuna] 已注入 OAT 种子参数: {clean}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print("\nOptuna best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print(f"  Best {optuna_metric.upper()}: {study.best_value:.4f}%")

    # 保存完整的 Optuna 搜索报告（CSV）
    result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '模型预测结果')
    os.makedirs(result_dir, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(result_dir, 'optuna_trials.csv'), index=False)

    return study

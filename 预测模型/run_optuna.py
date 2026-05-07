# -*- coding: utf-8 -*-
"""
运行 Optuna 贝叶斯超参数搜索
Run Optuna Bayesian Hyperparameter Search

用法: python run_optuna.py
可选: 修改 n_trials 和 n_trials_epochs
"""

import sys
import io
if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optuna_search import run_optuna_search

if __name__ == '__main__':
    print("=" * 50)
    print("Optuna 贝叶斯超参数搜索")
    print("=" * 50)
    study = run_optuna_search(
        n_trials=15,  # 尝试次数，可调
        n_trials_epochs=25  # 每次尝试的训练轮数
    )

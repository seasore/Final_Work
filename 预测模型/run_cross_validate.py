# -*- coding: utf-8 -*-
"""
运行时序 K 折交叉验证
Run Time Series K-Fold Cross-Validation

用法: python run_cross_validate.py
"""

import sys
import io
if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cross_validate import shiXuKZheJiaoChaYanZheng

if __name__ == '__main__':
    print("=" * 50)
    print("时序 K 折交叉验证")
    print("=" * 50)
    mape_list, mean_mape, std_mape = shiXuKZheJiaoChaYanZheng(
        n_splits=5,
        xunLianDaiShu=30  # 每折训练轮数，可调
    )
    print(f"\n各折 MAPE: {[f'{m:.4f}%' for m in mape_list]}")
    print(f"平均 MAPE: {mean_mape:.4f}% ± {std_mape:.4f}%")

# -*- coding: utf-8 -*-
"""
StandardScaler 若在 DataFrame 上 fit，会记录 feature_names_in_；
后续用 ndarray 调用 transform / inverse_transform 会触发 sklearn UserWarning。
此处用与拟合时一致的列名包一层 DataFrame，消除刷屏警告且语义一致。
"""
import numpy as np
import pandas as pd


def scaler_transform(scaler, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and len(names) == X.shape[1]:
        return scaler.transform(pd.DataFrame(X, columns=list(names)))
    return scaler.transform(X)


def scaler_inverse_transform(scaler, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None and len(names) == X.shape[1]:
        return scaler.inverse_transform(pd.DataFrame(X, columns=list(names)))
    return scaler.inverse_transform(X)

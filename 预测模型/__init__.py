# -*- coding: utf-8 -*-
"""
TCN-BiLSTM-Attention 短期电力负荷预测模型包

根据开题报告设计与实现，包含以下模块：
- TCN时间卷积网络：多尺度特征提取
- BiLSTM双向长短期记忆：时序建模
- 注意力机制：动态权重分配
- 多源特征融合：负荷+气象+时间
"""

from .model import TCNBiLstmZhuYiLiYuCeMoXing
from .tcn import ShiJianJuanJiWangLuo
from .bilstm import ShuangXiangChangDuanQiJiYiWangLuo
from .attention import DuoTouZhuYiLi, TeZhengZhuYiLi
from .fusion import DuoYuanTeZhengRongHe
from .data import DianLiFuHeShuJuJi, chuangJianShuJuJiaZaiQi, jiaZaiYuChuLiShuJu
from .train import (
    xunLianMoXing,
    xunLianDuiBiMoXing,
    pingGuMoXing,
    jiSuanMAPE,
    jiSuanMAE,
    jiSuanMSE,
    jiSuanRMSE,
    jiSuanNRMSE,
    ceShiYuCeXiangYingShiJian,
)
from .baselines import (
    LSTMJiChuMoXing,
    BiLSTMMoXing,
    TCNMoXing,
    BiLSTMAttentionMoXing,
    chuangJianDuiBiMoXing,
)

__all__ = [
    # 主模型
    'TCNBiLstmZhuYiLiYuCeMoXing',
    # 组件
    'ShiJianJuanJiWangLuo',
    'ShuangXiangChangDuanQiJiYiWangLuo',
    'DuoTouZhuYiLi',
    'TeZhengZhuYiLi',
    'DuoYuanTeZhengRongHe',
    # 对比基线模型
    'LSTMJiChuMoXing',
    'BiLSTMMoXing',
    'TCNMoXing',
    'BiLSTMAttentionMoXing',
    'chuangJianDuiBiMoXing',
    # 数据
    'DianLiFuHeShuJuJi',
    'chuangJianShuJuJiaZaiQi',
    'jiaZaiYuChuLiShuJu',
    # 训练与评估
    'xunLianMoXing',
    'xunLianDuiBiMoXing',
    'pingGuMoXing',
    'jiSuanMAPE',
    'jiSuanMAE',
    'jiSuanMSE',
    'jiSuanRMSE',
    'jiSuanNRMSE',
    'ceShiYuCeXiangYingShiJian',
]

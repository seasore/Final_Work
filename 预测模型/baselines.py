# -*- coding: utf-8 -*-
"""
对比基线模型定义
Baseline Comparison Models

按照论文对比实验设计，包含以下四种对比模型：
  1. LSTM            - 最基础基线（单向，无注意力）
  2. BiLSTM          - 双向LSTM
  3. TCN             - 纯时间卷积网络（无BiLSTM）
  4. BiLSTM-Attention - 双向LSTM + 多头自注意力

所有模型均使用相同的多源特征融合前端（DuoYuanTeZhengRongHe），
确保对比公平性。
"""

import torch
import torch.nn as nn

try:
    from .tcn import ShiJianJuanJiWangLuo
    from .bilstm import ShuangXiangChangDuanQiJiYiWangLuo
    from .attention import DuoTouZhuYiLi, TeZhengZhuYiLi
    from .fusion import DuoYuanTeZhengRongHe
except ImportError:
    from tcn import ShiJianJuanJiWangLuo
    from bilstm import ShuangXiangChangDuanQiJiYiWangLuo
    from attention import DuoTouZhuYiLi, TeZhengZhuYiLi
    from fusion import DuoYuanTeZhengRongHe

# 多源融合模块统一输出维度（与主模型保持一致）
_RONGHE_DIM = 64


class LSTMJiChuMoXing(nn.Module):
    """
    LSTM 最基础基线模型
    单向 LSTM，无注意力机制，代表最简单的序列建模方法。
    """

    def __init__(
        self,
        yinCangWeiDu: int = 64,
        cengShu: int = 2,
        tuoQiLv: float = 0.2,
        yuCeBuChang: int = 1,
    ):
        super().__init__()
        self.rongHe = DuoYuanTeZhengRongHe(
            fuHeTeZhengWeiDu=1,
            qiXiangTeZhengWeiDu=4,
            shiJianTeZhengWeiDu=5,
            shuChuWeiDu=_RONGHE_DIM,
        )
        self.lstm = nn.LSTM(
            input_size=_RONGHE_DIM,
            hidden_size=yinCangWeiDu,
            num_layers=cengShu,
            batch_first=True,
            dropout=tuoQiLv if cengShu > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(tuoQiLv)
        self.fc = nn.Linear(yinCangWeiDu, yuCeBuChang)

    def forward(self, f, q, s):
        x = self.rongHe(f, q, s)
        o, _ = self.lstm(x)
        return self.fc(self.dropout(o[:, -1, :]))


class BiLSTMMoXing(nn.Module):
    """
    BiLSTM 对比模型
    双向 LSTM，无注意力机制。
    相比 LSTM 增加反向建模能力，可捕获更丰富的双向时序依赖。
    """

    def __init__(
        self,
        yinCangWeiDu: int = 64,
        cengShu: int = 2,
        tuoQiLv: float = 0.2,
        yuCeBuChang: int = 1,
    ):
        super().__init__()
        self.rongHe = DuoYuanTeZhengRongHe(
            fuHeTeZhengWeiDu=1,
            qiXiangTeZhengWeiDu=4,
            shiJianTeZhengWeiDu=5,
            shuChuWeiDu=_RONGHE_DIM,
        )
        self.bilstm = ShuangXiangChangDuanQiJiYiWangLuo(
            shuRuWeiDu=_RONGHE_DIM,
            yinCangWeiDu=yinCangWeiDu,
            cengShu=cengShu,
            tuoQiLv=tuoQiLv,
        )
        bilstm_out = yinCangWeiDu * 2
        self.dropout = nn.Dropout(tuoQiLv)
        self.fc = nn.Linear(bilstm_out, yuCeBuChang)

    def forward(self, f, q, s):
        x = self.rongHe(f, q, s)
        o = self.bilstm(x)
        return self.fc(self.dropout(o[:, -1, :]))


class TCNMoXing(nn.Module):
    """
    TCN 对比模型（纯时间卷积，无双向 LSTM）
    使用多尺度膨胀卷积提取局部时序特征。
    相比 BiLSTM 侧重局部模式而非长程双向依赖。
    """

    def __init__(
        self,
        tcnYinCangWeiDu: int = 64,
        tcnCengShu: int = 4,
        tuoQiLv: float = 0.2,
        yuCeBuChang: int = 1,
    ):
        super().__init__()
        self.rongHe = DuoYuanTeZhengRongHe(
            fuHeTeZhengWeiDu=1,
            qiXiangTeZhengWeiDu=4,
            shiJianTeZhengWeiDu=5,
            shuChuWeiDu=_RONGHE_DIM,
        )
        self.tcn = ShiJianJuanJiWangLuo(
            shuRuTeZhengShu=_RONGHE_DIM,
            yinCangCengWeiDu=tcnYinCangWeiDu,
            juanJiHeDaXiao=3,
            tcnCengShu=tcnCengShu,
        )
        self.dropout = nn.Dropout(tuoQiLv)
        self.fc = nn.Linear(tcnYinCangWeiDu, yuCeBuChang)

    def forward(self, f, q, s):
        x = self.rongHe(f, q, s)
        o = self.tcn(x)
        return self.fc(self.dropout(o[:, -1, :]))


class BiLSTMAttentionMoXing(nn.Module):
    """
    BiLSTM-Attention 对比模型
    双向 LSTM + 多头自注意力 + 特征注意力。
    与主模型的区别：无 TCN 多尺度特征提取前端。
    """

    def __init__(
        self,
        yinCangWeiDu: int = 64,
        cengShu: int = 2,
        zhuYiLiTouShu: int = 4,
        tuoQiLv: float = 0.2,
        yuCeBuChang: int = 1,
    ):
        super().__init__()
        self.rongHe = DuoYuanTeZhengRongHe(
            fuHeTeZhengWeiDu=1,
            qiXiangTeZhengWeiDu=4,
            shiJianTeZhengWeiDu=5,
            shuChuWeiDu=_RONGHE_DIM,
        )
        self.bilstm = ShuangXiangChangDuanQiJiYiWangLuo(
            shuRuWeiDu=_RONGHE_DIM,
            yinCangWeiDu=yinCangWeiDu,
            cengShu=cengShu,
            tuoQiLv=tuoQiLv,
        )
        bilstm_out = yinCangWeiDu * 2
        # 确保 bilstm_out 整除 heads
        if bilstm_out % zhuYiLiTouShu != 0:
            zhuYiLiTouShu = 2
        self.attention = DuoTouZhuYiLi(
            shuRuWeiDu=bilstm_out,
            touShu=zhuYiLiTouShu,
            tuiChuLv=tuoQiLv,
        )
        self.feat_attn = TeZhengZhuYiLi(bilstm_out)
        self.dropout = nn.Dropout(tuoQiLv)
        self.fc = nn.Linear(bilstm_out, yuCeBuChang)

    def forward(self, f, q, s):
        x = self.rongHe(f, q, s)
        o = self.bilstm(x)
        o = self.attention(o) + o  # 残差
        o = self.feat_attn(o)
        return self.fc(self.dropout(o[:, -1, :]))


# ──────────────────────────────────────────────────────────────
# 工厂函数：按名称创建对比模型
# ──────────────────────────────────────────────────────────────

_MODEL_NAMES = {
    "LSTM": LSTMJiChuMoXing,
    "BiLSTM": BiLSTMMoXing,
    "TCN": TCNMoXing,
    "BiLSTM-Attention": BiLSTMAttentionMoXing,
}


def chuangJianDuiBiMoXing(mingCheng: str, params: dict = None, yuCeBuChang: int = 1):
    """
    按名称创建对比模型实例。

    mingCheng: 'LSTM' | 'BiLSTM' | 'TCN' | 'BiLSTM-Attention'
    params: 超参数字典（各模型支持的键见各类 __init__），None 则使用默认值
    """
    params = params or {}
    if mingCheng not in _MODEL_NAMES:
        raise ValueError(f"未知模型名称: {mingCheng}，可选: {list(_MODEL_NAMES)}")
    cls = _MODEL_NAMES[mingCheng]
    if mingCheng == "LSTM":
        return cls(
            yinCangWeiDu=params.get("yinCangWeiDu", 64),
            cengShu=params.get("cengShu", 2),
            tuoQiLv=params.get("tuoQiLv", 0.2),
            yuCeBuChang=yuCeBuChang,
        )
    elif mingCheng == "BiLSTM":
        return cls(
            yinCangWeiDu=params.get("yinCangWeiDu", 64),
            cengShu=params.get("cengShu", 2),
            tuoQiLv=params.get("tuoQiLv", 0.2),
            yuCeBuChang=yuCeBuChang,
        )
    elif mingCheng == "TCN":
        return cls(
            tcnYinCangWeiDu=params.get("tcnYinCangWeiDu", 64),
            tcnCengShu=params.get("tcnCengShu", 4),
            tuoQiLv=params.get("tuoQiLv", 0.2),
            yuCeBuChang=yuCeBuChang,
        )
    elif mingCheng == "BiLSTM-Attention":
        return cls(
            yinCangWeiDu=params.get("yinCangWeiDu", 64),
            cengShu=params.get("cengShu", 2),
            zhuYiLiTouShu=params.get("zhuYiLiTouShu", 4),
            tuoQiLv=params.get("tuoQiLv", 0.2),
            yuCeBuChang=yuCeBuChang,
        )

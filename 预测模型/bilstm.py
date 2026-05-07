# -*- coding: utf-8 -*-
"""
双向长短期记忆网络（BiLSTM）模块
Bidirectional Long Short-Term Memory Module

根据开题报告设计：
- 前向LSTM捕捉历史依赖关系，后向LSTM捕捉未来趋势信息
- 采用多层双向长短期记忆网络结构以增强对长期时间依赖关系的建模能力
- 引入Dropout机制防止过拟合，提升模型的泛化性能
"""

import torch
import torch.nn as nn


class ShuangXiangChangDuanQiJiYiWangLuo(nn.Module):
    """
    双向长短期记忆网络
    ShuangXiang = 双向, ChangDuanQiJiYi = 长短期记忆, WangLuo = 网络
    
    同时从前向和后向两个方向处理序列，能够捕捉更丰富的时序依赖关系。
    对于15分钟间隔的负荷数据，双向结构能更好地理解负荷的周期性变化。
    """
    
    def __init__(self, shuRuWeiDu, yinCangWeiDu, cengShu=2, tuoQiLv=0.2):
        """
        初始化BiLSTM网络
        
        参数:
            shuRuWeiDu: 输入特征维度（来自TCN的输出维度）
            yinCangWeiDu: 隐藏层维度（每个方向的LSTM单元数）
            cengShu: LSTM层数，多层结构增强长期依赖建模能力
            tuoQiLv: Dropout比率，防止过拟合，默认0.2
        """
        super(ShuangXiangChangDuanQiJiYiWangLuo, self).__init__()
        
        self.yinCangWeiDu = yinCangWeiDu
        self.cengShu = cengShu
        # 双向LSTM输出维度为 2 * 隐藏维度（前向+后向拼接）
        self.shuChuWeiDu = yinCangWeiDu * 2
        
        self.biLstm = nn.LSTM(
            input_size=shuRuWeiDu,
            hidden_size=yinCangWeiDu,
            num_layers=cengShu,
            batch_first=True,
            dropout=tuoQiLv if cengShu > 1 else 0,
            bidirectional=True  # 启用双向结构
        )
        
    def forward(self, shuRu):
        """
        前向传播
        
        参数:
            shuRu: 输入张量，形状 (批次大小, 序列长度, 输入维度)
            
        返回:
            lstmShuChu: 输出张量，形状 (批次大小, 序列长度, 隐藏维度*2)
        """
        # BiLSTM前向传播
        # 输出: (序列输出, (最终隐藏状态, 最终细胞状态))
        lstmShuChu, (zuiZhongYinCangZhuangTai, zuiZhongXiBaoZhuangTai) = self.biLstm(shuRu)
        
        # lstmShuChu 已包含前向和后向的拼接，形状为 (N, L, hidden_size*2)
        return lstmShuChu

# -*- coding: utf-8 -*-
"""
时间卷积网络（TCN）模块
Temporal Convolutional Network Module

根据开题报告设计：
- 采用扩张卷积结构，通过不同扩张因子的卷积层捕捉多时间尺度的局部特征
- 设计残差连接机制以确保梯度流动并提升模型训练稳定性
- 运用因果卷积确保预测的时间性，避免未来信息泄露
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class PengZhangJuanJiCeng(nn.Module):
    """
    膨胀卷积层（扩张卷积层）
    PengZhang = 膨胀, JuanJi = 卷积, Ceng = 层
    
    实现因果卷积：通过左侧填充确保输出只依赖过去和当前时刻的输入，
    避免未来信息泄露，保证预测的时间因果性。
    """
    
    def __init__(self, shuRuTongDaoShu, shuChuTongDaoShu, juanJiHeDaXiao, pengZhangYinZi):
        """
        初始化膨胀卷积层
        
        参数:
            shuRuTongDaoShu: 输入通道数（输入特征的维度）
            shuChuTongDaoShu: 输出通道数
            juanJiHeDaXiao: 卷积核大小（在时间维度上的感受野）
            pengZhangYinZi: 扩张因子，用于扩大感受野而不增加参数量
        """
        super(PengZhangJuanJiCeng, self).__init__()
        self.pengZhangYinZi = pengZhangYinZi
        self.juanJiHeDaXiao = juanJiHeDaXiao
        
        # 因果卷积：左侧填充 (kernel_size - 1) * dilation 个零，确保不看到未来信息
        zuoCeTianChong = (juanJiHeDaXiao - 1) * pengZhangYinZi
        
        self.juanJiCeng = nn.Conv1d(
            shuRuTongDaoShu,
            shuChuTongDaoShu,
            juanJiHeDaXiao,
            padding=zuoCeTianChong,
            dilation=pengZhangYinZi
        )
        # 使用权重归一化提升训练稳定性
        self.juanJiCeng = weight_norm(self.juanJiCeng)
        self.jiHuoHanShu = nn.ReLU()
        self.tuiChuCeng = nn.Dropout(0.1)
        
    def forward(self, shuRu):
        """
        前向传播
        
        参数:
            shuRu: 输入张量，形状为 (批次大小, 通道数, 序列长度)
            
        返回:
            输出张量，形状与输入相同（在因果卷积中会裁剪掉右侧多余的填充部分）
        """
        # 因果卷积：裁剪右侧的填充部分，确保输出长度与输入一致且无未来信息泄露
        juanJiShuChu = self.juanJiCeng(shuRu)
        # 裁剪右侧多余的填充
        if juanJiShuChu.size(2) > shuRu.size(2):
            juanJiShuChu = juanJiShuChu[:, :, :shuRu.size(2)]
        shuChu = self.tuiChuCeng(self.jiHuoHanShu(juanJiShuChu))
        return shuChu


class CanChaShiJianJuanJiKuai(nn.Module):
    """
    残差时间卷积块
    CanCha = 残差, ShiJianJuanJi = 时间卷积, Kuai = 块
    
    包含两个膨胀卷积层和残差连接，确保梯度能够有效流动，
    提升深层网络的训练稳定性。
    """
    
    def __init__(self, shuRuTongDaoShu, shuChuTongDaoShu, juanJiHeDaXiao, pengZhangYinZi):
        """
        初始化残差TCN块
        
        参数:
            shuRuTongDaoShu: 输入通道数
            shuChuTongDaoShu: 输出通道数
            juanJiHeDaXiao: 卷积核大小
            pengZhangYinZi: 扩张因子
        """
        super(CanChaShiJianJuanJiKuai, self).__init__()
        
        self.juanJiCeng1 = PengZhangJuanJiCeng(
            shuRuTongDaoShu, shuChuTongDaoShu, juanJiHeDaXiao, pengZhangYinZi
        )
        self.juanJiCeng2 = PengZhangJuanJiCeng(
            shuChuTongDaoShu, shuChuTongDaoShu, juanJiHeDaXiao, pengZhangYinZi
        )
        
        # 当输入输出通道数不同时，使用1x1卷积进行维度匹配
        if shuRuTongDaoShu != shuChuTongDaoShu:
            self.canChaTongDao = nn.Conv1d(shuRuTongDaoShu, shuChuTongDaoShu, 1)
        else:
            self.canChaTongDao = nn.Identity()
            
    def forward(self, shuRu):
        """
        前向传播（含残差连接）
        
        参数:
            shuRu: 输入张量
            
        返回:
            输出 = ReLU(卷积输出 + 残差连接)
        """
        juanJiShuChu = self.juanJiCeng2(self.juanJiCeng1(shuRu))
        canChaShuRu = self.canChaTongDao(shuRu)
        # 残差连接：确保梯度流动，提升训练稳定性
        return torch.relu(juanJiShuChu + canChaShuRu)


class ShiJianJuanJiWangLuo(nn.Module):
    """
    时间卷积网络（TCN）主模块
    ShiJianJuanJi = 时间卷积, WangLuo = 网络
    
    通过堆叠多个具有不同扩张因子的残差块，实现多尺度特征提取。
    扩张因子呈指数增长：1, 2, 4, 8...，使感受野指数级扩大。
    """
    
    def __init__(self, shuRuTeZhengShu, yinCangCengWeiDu, juanJiHeDaXiao=3, tcnCengShu=4):
        """
        初始化TCN网络
        
        参数:
            shuRuTeZhengShu: 输入特征维度（每个时间步的特征数）
            yinCangCengWeiDu: 隐藏层维度（TCN内部通道数）
            juanJiHeDaXiao: 卷积核大小，默认3
            tcnCengShu: TCN残差块数量，每层扩张因子翻倍
        """
        super(ShiJianJuanJiWangLuo, self).__init__()
        
        self.tcnCengShu = tcnCengShu
        tcnCengLieBiao = []
        
        # 构建多层残差TCN块，扩张因子依次为 1, 2, 4, 8, ...
        for i in range(tcnCengShu):
            pengZhangYinZi = 2 ** i  # 指数增长的扩张因子
            shuRuTongDao = shuRuTeZhengShu if i == 0 else yinCangCengWeiDu
            shuChuTongDao = yinCangCengWeiDu
            
            tcnCengLieBiao.append(
                CanChaShiJianJuanJiKuai(
                    shuRuTongDao, shuChuTongDao, juanJiHeDaXiao, pengZhangYinZi
                )
            )
        
        self.tcnWangLuo = nn.Sequential(*tcnCengLieBiao)
        
    def forward(self, shuRu):
        """
        前向传播
        
        参数:
            shuRu: 输入张量，形状 (批次大小, 序列长度, 特征维度)
            
        返回:
            输出张量，形状 (批次大小, 序列长度, 隐藏维度)
        """
        # TCN的Conv1d期望输入格式为 (N, C, L)，需要转置
        shuRuZhuanZhi = shuRu.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        shuChu = self.tcnWangLuo(shuRuZhuanZhi)
        # 转回 (N, L, C) 格式以供后续BiLSTM使用
        return shuChu.permute(0, 2, 1)

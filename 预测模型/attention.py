# -*- coding: utf-8 -*-
"""
注意力机制模块
Attention Mechanism Module

根据开题报告设计：
- 自注意力机制，使模型能够自动聚焦对预测结果影响最大的历史时间步
- 构建特征注意力模块，动态分配气象特征与历史负荷特征的权重
- 采用多头注意力机制，从多角度捕捉特征间的复杂关联
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DuoTouZhuYiLi(nn.Module):
    """
    多头自注意力机制
    DuoTou = 多头, ZhuYiLi = 注意力
    
    将输入投影到多个子空间，从不同角度捕捉序列中不同时间步的重要性，
    能够学习特征间的复杂关联关系。
    """
    
    def __init__(self, shuRuWeiDu, touShu=4, tuiChuLv=0.1):
        """
        初始化多头注意力
        
        参数:
            shuRuWeiDu: 输入维度，必须能被头数整除
            touShu: 注意力头数，从多角度捕捉特征关联
            tuiChuLv: Dropout比率
        """
        super(DuoTouZhuYiLi, self).__init__()
        assert shuRuWeiDu % touShu == 0, "输入维度必须能被头数整除"
        
        self.shuRuWeiDu = shuRuWeiDu
        self.touShu = touShu
        self.meiTouWeiDu = shuRuWeiDu // touShu
        
        # Q、K、V 的线性投影
        self.qingQiuTouXiangLiang = nn.Linear(shuRuWeiDu, shuRuWeiDu)
        self.jianChaTouXiangLiang = nn.Linear(shuRuWeiDu, shuRuWeiDu)
        self.jiaZhiTouXiangLiang = nn.Linear(shuRuWeiDu, shuRuWeiDu)
        self.shuChuTouXiangLiang = nn.Linear(shuRuWeiDu, shuRuWeiDu)
        
        self.tuiChuCeng = nn.Dropout(tuiChuLv)
        
    def forward(self, shuRu):
        """
        前向传播
        
        参数:
            shuRu: 输入张量，形状 (批次大小, 序列长度, 特征维度)
            
        返回:
            加权后的输出，形状与输入相同
        """
        piCiDaXiao, xuLieChangDu, teZhengWeiDu = shuRu.size()
        
        # 计算 Q, K, V
        qingQiu = self.qingQiuTouXiangLiang(shuRu)
        jianCha = self.jianChaTouXiangLiang(shuRu)
        jiaZhi = self.jiaZhiTouXiangLiang(shuRu)
        
        # 重塑为多头格式: (N, 头数, 序列长度, 每头维度)
        qingQiu = qingQiu.view(piCiDaXiao, xuLieChangDu, self.touShu, self.meiTouWeiDu).transpose(1, 2)
        jianCha = jianCha.view(piCiDaXiao, xuLieChangDu, self.touShu, self.meiTouWeiDu).transpose(1, 2)
        jiaZhi = jiaZhi.view(piCiDaXiao, xuLieChangDu, self.touShu, self.meiTouWeiDu).transpose(1, 2)
        
        # 缩放点积注意力: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        dianJi = torch.matmul(qingQiu, jianCha.transpose(-2, -1)) / math.sqrt(self.meiTouWeiDu)
        zhuYiLiQuanZhong = F.softmax(dianJi, dim=-1)
        zhuYiLiQuanZhong = self.tuiChuCeng(zhuYiLiQuanZhong)
        
        # 应用注意力权重
        zhuYiLiShuChu = torch.matmul(zhuYiLiQuanZhong, jiaZhi)
        
        # 合并多头: (N, 头数, L, d) -> (N, L, 特征维度)
        zhuYiLiShuChu = zhuYiLiShuChu.transpose(1, 2).contiguous().view(
            piCiDaXiao, xuLieChangDu, teZhengWeiDu
        )
        
        return self.shuChuTouXiangLiang(zhuYiLiShuChu)


class TeZhengZhuYiLi(nn.Module):
    """
    特征注意力模块
    TeZheng = 特征, ZhuYiLi = 注意力
    
    动态分配不同特征（气象、负荷等）的权重，
    使模型能够自动聚焦对预测影响最大的特征。
    """
    
    def __init__(self, teZhengWeiDu):
        """
        初始化特征注意力
        
        参数:
            teZhengWeiDu: 特征维度
        """
        super(TeZhengZhuYiLi, self).__init__()
        yinCangWei = max(1, teZhengWeiDu // 4)  # 防止维度为0
        self.teZhengQuanZhongWangLuo = nn.Sequential(
            nn.Linear(teZhengWeiDu, yinCangWei),
            nn.Tanh(),
            nn.Linear(yinCangWei, 1)
        )
        
    def forward(self, shuRu):
        """
        前向传播
        
        参数:
            shuRu: 输入张量，形状 (批次大小, 序列长度, 特征维度)
            
        返回:
            加权后的特征表示
        """
        # 计算每个特征维度的重要性分数
        zhongYaoXingFenShu = self.teZhengQuanZhongWangLuo(shuRu)  # (N, L, 1)
        teZhengQuanZhong = F.softmax(zhongYaoXingFenShu, dim=1)
        # 加权求和
        return shuRu * teZhengQuanZhong

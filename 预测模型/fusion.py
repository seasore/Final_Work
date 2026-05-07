# -*- coding: utf-8 -*-
"""
多源特征融合模块
Multi-Source Feature Fusion Module

根据开题报告设计：
- 整合历史负荷数据、气象数据（最高温度、最低温度、相对湿度、风速）及时间特征等多维信息
- 通过注意力机制动态分配特征权重，有效增强对各因素间耦合关系的探索能力
- 设计特征交互层可捕捉不同特征间的非线性交互关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeZhengJiaoHuCeng(nn.Module):
    """
    特征交互层
    TeZheng = 特征, JiaoHu = 交互, Ceng = 层
    
    捕捉不同特征（负荷、温度、湿度、风速等）之间的非线性交互关系，
    例如：高温+高湿度可能对负荷有协同影响。
    """
    
    def __init__(self, shuRuWeiDu, yinCangWeiDu=64):
        """
        初始化特征交互层
        
        参数:
            shuRuWeiDu: 输入特征总维度
            yinCangWeiDu: 交互表示的隐藏维度
        """
        super(TeZhengJiaoHuCeng, self).__init__()
        self.jiaoHuWangLuo = nn.Sequential(
            nn.Linear(shuRuWeiDu * 2, yinCangWeiDu),  # 两两交互
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(yinCangWeiDu, shuRuWeiDu)
        )
        
    def forward(self, shuRu):
        """
        前向传播（简化版：使用全连接捕捉特征交互）
        
        参数:
            shuRu: 输入 (批次, 序列长度, 特征维度)
            
        返回:
            融合交互信息后的特征
        """
        # 通过线性变换捕捉特征间交互
        return shuRu + self.jiaoHuWangLuo(torch.cat([shuRu, shuRu * shuRu], dim=-1))


class DuoYuanTeZhengRongHe(nn.Module):
    """
    多源特征融合模块
    DuoYuan = 多源, TeZheng = 特征, RongHe = 融合
    
    整合历史负荷、气象数据（最高温、最低温、湿度、风速）、时间特征，
    通过注意力动态分配各源数据的权重。
    """
    
    def __init__(self, fuHeTeZhengWeiDu, qiXiangTeZhengWeiDu, shiJianTeZhengWeiDu, shuChuWeiDu):
        """
        初始化多源特征融合
        
        参数:
            fuHeTeZhengWeiDu: 负荷特征维度
            qiXiangTeZhengWeiDu: 气象特征维度（温度、湿度、风速等）
            shiJianTeZhengWeiDu: 时间特征维度（小时、星期等）
            shuChuWeiDu: 融合后的输出维度
        """
        super(DuoYuanTeZhengRongHe, self).__init__()
        
        zongTeZhengWeiDu = fuHeTeZhengWeiDu + qiXiangTeZhengWeiDu + shiJianTeZhengWeiDu
        
        # 各源特征的投影层
        self.fuHeTouYing = nn.Linear(fuHeTeZhengWeiDu, shuChuWeiDu)
        self.qiXiangTouYing = nn.Linear(qiXiangTeZhengWeiDu, shuChuWeiDu)
        self.shiJianTouYing = nn.Linear(shiJianTeZhengWeiDu, shuChuWeiDu)
        
        # 源权重注意力：动态分配各源的重要性
        self.yuanQuanZhongWangLuo = nn.Sequential(
            nn.Linear(shuChuWeiDu * 3, shuChuWeiDu),
            nn.Tanh(),
            nn.Linear(shuChuWeiDu, 3)  # 3个源：负荷、气象、时间
        )
        
        # 特征交互层
        self.teZhengJiaoHu = TeZhengJiaoHuCeng(shuChuWeiDu)
        
        self.shuChuTouYing = nn.Linear(shuChuWeiDu, shuChuWeiDu)
        
    def forward(self, fuHeTeZheng, qiXiangTeZheng, shiJianTeZheng):
        """
        前向传播
        
        参数:
            fuHeTeZheng: 历史负荷特征 (批次, 序列长度, 负荷维度)
            qiXiangTeZheng: 气象特征 (批次, 序列长度, 气象维度)
            shiJianTeZheng: 时间特征 (批次, 序列长度, 时间维度)
            
        返回:
            融合后的特征表示 (批次, 序列长度, 输出维度)
        """
        # 投影到统一维度
        fuHeTouYing = self.fuHeTouYing(fuHeTeZheng)
        qiXiangTouYing = self.qiXiangTouYing(qiXiangTeZheng)
        shiJianTouYing = self.shiJianTouYing(shiJianTeZheng)
        
        # 拼接计算源权重
        pinJieTeZheng = torch.cat([fuHeTouYing, qiXiangTouYing, shiJianTouYing], dim=-1)
        yuanQuanZhong = F.softmax(self.yuanQuanZhongWangLuo(pinJieTeZheng), dim=-1)
        
        # 加权融合
        rongHeTeZheng = (
            yuanQuanZhong[:, :, 0:1] * fuHeTouYing +
            yuanQuanZhong[:, :, 1:2] * qiXiangTouYing +
            yuanQuanZhong[:, :, 2:3] * shiJianTouYing
        )
        
        # 特征交互
        rongHeTeZheng = self.teZhengJiaoHu(rongHeTeZheng)
        
        return self.shuChuTouYing(rongHeTeZheng)

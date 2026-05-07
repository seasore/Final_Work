# -*- coding: utf-8 -*-
"""
TCN-BiLSTM-Attention 短期电力负荷预测模型
TCN-BiLSTM-Attention Short-term Power Load Forecasting Model

根据开题报告完整实现：
- TCN多尺度特征提取
- BiLSTM双向时间建模
- 注意力机制动态权重分配
- 多源特征融合（负荷+气象+时间）

目标：MAPE降低超过15%，单次预测响应时间≤1分钟
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


class TCNBiLstmZhuYiLiYuCeMoXing(nn.Module):
    """
    TCN-BiLSTM-Attention 混合预测模型主类
    
    架构流程：
    输入 -> 多源特征融合 -> TCN多尺度提取 -> BiLSTM时序建模 -> 注意力加权 -> 全连接预测
    """
    
    def __init__(
        self,
        fuHeTeZhengShu=1,
        qiXiangTeZhengShu=4,
        shiJianTeZhengShu=4,
        tcnYinCangWeiDu=64,
        tcnCengShu=4,
        bilstmYinCangWeiDu=64,
        bilstmCengShu=2,
        zhuYiLiTouShu=4,
        yuCeBuChang=1,
        tuoQiLv=0.2
    ):
        """
        初始化TCN-BiLSTM-Attention预测模型
        
        参数:
            fuHeTeZhengShu: 负荷相关特征数（历史负荷等）
            qiXiangTeZhengShu: 气象特征数（最高温、最低温、湿度、风速等）
            shiJianTeZhengShu: 时间特征数（小时、星期、是否节假日等）
            tcnYinCangWeiDu: TCN隐藏层维度
            tcnCengShu: TCN残差块层数
            bilstmYinCangWeiDu: BiLSTM隐藏维度（单方向）
            bilstmCengShu: BiLSTM层数
            zhuYiLiTouShu: 多头注意力头数
            yuCeBuChang: 预测步长（预测未来几个时间步）
            tuoQiLv: Dropout比率
        """
        super(TCNBiLstmZhuYiLiYuCeMoXing, self).__init__()
        
        self.yuCeBuChang = yuCeBuChang
        self.zongTeZhengShu = fuHeTeZhengShu + qiXiangTeZhengShu + shiJianTeZhengShu
        
        # 融合后的特征维度
        rongHeWeiDu = tcnYinCangWeiDu
        
        # 1. 多源特征融合模块
        self.duoYuanRongHe = DuoYuanTeZhengRongHe(
            fuHeTeZhengWeiDu=fuHeTeZhengShu,
            qiXiangTeZhengWeiDu=qiXiangTeZhengShu,
            shiJianTeZhengWeiDu=shiJianTeZhengShu,
            shuChuWeiDu=rongHeWeiDu
        )
        
        # 2. TCN多尺度特征提取模块
        self.shiJianJuanJi = ShiJianJuanJiWangLuo(
            shuRuTeZhengShu=rongHeWeiDu,
            yinCangCengWeiDu=tcnYinCangWeiDu,
            juanJiHeDaXiao=3,
            tcnCengShu=tcnCengShu
        )
        
        # 3. BiLSTM双向时间建模模块
        self.shuangXiangLstm = ShuangXiangChangDuanQiJiYiWangLuo(
            shuRuWeiDu=tcnYinCangWeiDu,
            yinCangWeiDu=bilstmYinCangWeiDu,
            cengShu=bilstmCengShu,
            tuoQiLv=tuoQiLv
        )
        
        # 4. 多头自注意力模块
        bilstmShuChuWeiDu = bilstmYinCangWeiDu * 2  # 双向
        assert bilstmShuChuWeiDu % zhuYiLiTouShu == 0
        self.duoTouZhuYiLi = DuoTouZhuYiLi(
            shuRuWeiDu=bilstmShuChuWeiDu,
            touShu=zhuYiLiTouShu,
            tuiChuLv=tuoQiLv
        )
        
        # 5. 特征注意力（可选，增强关键时间步）
        self.teZhengZhuYiLi = TeZhengZhuYiLi(bilstmShuChuWeiDu)
        
        # 6. 预测输出层
        self.yuCeQuanLianJie = nn.Sequential(
            nn.Linear(bilstmShuChuWeiDu, bilstmShuChuWeiDu // 2),
            nn.ReLU(),
            nn.Dropout(tuoQiLv),
            nn.Linear(bilstmShuChuWeiDu // 2, yuCeBuChang)
        )
        
    def forward(self, fuHeTeZheng, qiXiangTeZheng, shiJianTeZheng):
        """
        前向传播
        
        参数:
            fuHeTeZheng: 负荷特征 (批次, 序列长度, 负荷特征数)
            qiXiangTeZheng: 气象特征 (批次, 序列长度, 气象特征数)
            shiJianTeZheng: 时间特征 (批次, 序列长度, 时间特征数)
            
        返回:
            yuCeZhi: 预测值 (批次, 预测步长)
        """
        # 1. 多源特征融合
        rongHeTeZheng = self.duoYuanRongHe(fuHeTeZheng, qiXiangTeZheng, shiJianTeZheng)
        
        # 2. TCN多尺度特征提取
        tcnShuChu = self.shiJianJuanJi(rongHeTeZheng)
        
        # 3. BiLSTM双向时序建模
        bilstmShuChu = self.shuangXiangLstm(tcnShuChu)
        
        # 4. 多头自注意力
        zhuYiLiShuChu = self.duoTouZhuYiLi(bilstmShuChu)
        zhuYiLiShuChu = zhuYiLiShuChu + bilstmShuChu  # 残差连接
        
        # 5. 特征注意力
        zhongYaoTeZheng = self.teZhengZhuYiLi(zhuYiLiShuChu)
        
        # 6. 取最后一个时间步进行预测（可改为取多个时间步）
        zuiHouShiJianBu = zhongYaoTeZheng[:, -1, :]
        
        # 7. 预测输出
        yuCeZhi = self.yuCeQuanLianJie(zuiHouShiJianBu)
        
        return yuCeZhi

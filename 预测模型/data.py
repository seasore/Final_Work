# -*- coding: utf-8 -*-
"""
数据加载器
Data Loader Module

根据开题报告：
- 数据时间跨度不少于一年，采样间隔为15分钟
- 包含历史负荷与气象数据（最高温度、最低温度、相对湿度、风速）
- 进行数据清洗、异常值处理及缺失值填补等预处理
"""

import os
import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 预处理数据路径（与 数据预处理 模块对齐）
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_CSV = os.path.join(_BASE_DIR, '数据', '训练集', 'train_data.csv')
_VAL_CSV = os.path.join(_BASE_DIR, '数据', '验证集', 'val_data.csv')
_TEST_CSV = os.path.join(_BASE_DIR, '数据', '测试集', 'test_data.csv')
_SCALERS_PKL = os.path.join(_BASE_DIR, '数据预处理', 'scalers.pkl')
_TIME_FEATURE_COLS = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend']


class DianLiFuHeShuJuJi(Dataset):
    """
    电力负荷数据集
    DianLi = 电力, FuHe = 负荷, ShuJu = 数据, Ji = 集
    
    支持滑动窗口构建时序样本，将多维特征分为负荷、气象、时间三类。
    """
    
    def __init__(
        self,
        fuHeShuJu,
        qiXiangShuJu,
        shiJianTeZheng,
        muBiaoZhi,
        xuLieChangDu=96,
        yuCeBuChang=1,
        biaoZhunHuaFangShi='z_score'
    ):
        """
        初始化数据集
        
        参数:
            fuHeShuJu: 负荷数据，形状 (样本数, 负荷特征数)，如历史负荷值
            qiXiangShuJu: 气象数据，形状 (样本数, 4)，[最高温、最低温、湿度、风速]
            shiJianTeZheng: 时间特征，形状 (样本数, 时间特征数)，如小时、星期、是否节假日
            muBiaoZhi: 预测目标值，形状 (样本数,)
            xuLieChangDu: 输入序列长度（时间步数），96表示24小时*4（15分钟间隔）
            yuCeBuChang: 预测步长
            biaoZhunHuaFangShi: 标准化方式，'z_score'或'minmax'
        """
        super(DianLiFuHeShuJuJi, self).__init__()
        
        self.xuLieChangDu = xuLieChangDu
        self.yuCeBuChang = yuCeBuChang
        
        # 构建滑动窗口样本
        self.fuHeYangBenLieBiao = []
        self.qiXiangYangBenLieBiao = []
        self.shiJianYangBenLieBiao = []
        self.muBiaoLieBiao = []
        
        zongYangBenShu = len(muBiaoZhi)
        
        for i in range(xuLieChangDu, zongYangBenShu - yuCeBuChang + 1):
            # 输入窗口
            qiShiXiaBiao = i - xuLieChangDu
            jieShuXiaBiao = i
            
            self.fuHeYangBenLieBiao.append(fuHeShuJu[qiShiXiaBiao:jieShuXiaBiao])
            self.qiXiangYangBenLieBiao.append(qiXiangShuJu[qiShiXiaBiao:jieShuXiaBiao])
            self.shiJianYangBenLieBiao.append(shiJianTeZheng[qiShiXiaBiao:jieShuXiaBiao])
            # 预测目标：取窗口后的值
            self.muBiaoLieBiao.append(muBiaoZhi[i:i + yuCeBuChang])
        
        self.fuHeYangBenLieBiao = np.array(self.fuHeYangBenLieBiao, dtype=np.float32)
        self.qiXiangYangBenLieBiao = np.array(self.qiXiangYangBenLieBiao, dtype=np.float32)
        self.shiJianYangBenLieBiao = np.array(self.shiJianYangBenLieBiao, dtype=np.float32)
        self.muBiaoLieBiao = np.array(self.muBiaoLieBiao, dtype=np.float32)
        
    def __len__(self):
        """返回数据集样本数量"""
        return len(self.muBiaoLieBiao)
    
    def __getitem__(self, suoYin):
        """获取单个样本"""
        return (
            torch.FloatTensor(self.fuHeYangBenLieBiao[suoYin]),
            torch.FloatTensor(self.qiXiangYangBenLieBiao[suoYin]),
            torch.FloatTensor(self.shiJianYangBenLieBiao[suoYin]),
            torch.FloatTensor(self.muBiaoLieBiao[suoYin])
        )


def chuangJianShuJuJiaZaiQi(
    fuHeShuJu,
    qiXiangShuJu,
    shiJianTeZheng,
    muBiaoZhi,
    xuLieChangDu=96,
    yuCeBuChang=1,
    piCiDaXiao=32,
    xunLianBiLi=0.8,
    yanZhengBiLi=0.1
):
    """
    创建数据加载器
    ChuangJian = 创建, ShuJu = 数据, JiaZaiQi = 加载器
    
    参数:
        fuHeShuJu: 负荷数据
        qiXiangShuJu: 气象数据
        shiJianTeZheng: 时间特征
        muBiaoZhi: 目标值
        xuLieChangDu: 序列长度
        yuCeBuChang: 预测步长
        piCiDaXiao: 批次大小
        xunLianBiLi: 训练集比例
        yanZhengBiLi: 验证集比例
        
    返回:
        (训练加载器, 验证加载器, 测试加载器, 标准化器字典)
    """
    zongChangDu = len(muBiaoZhi)
    xunLianJieDian = int(zongChangDu * xunLianBiLi)
    yanZhengJieDian = int(zongChangDu * (xunLianBiLi + yanZhengBiLi))
    
    # 划分数据集
    xunLianFuHe = fuHeShuJu[:xunLianJieDian]
    xunLianQiXiang = qiXiangShuJu[:xunLianJieDian]
    xunLianShiJian = shiJianTeZheng[:xunLianJieDian]
    xunLianMuBiao = muBiaoZhi[:xunLianJieDian]
    
    yanZhengFuHe = fuHeShuJu[xunLianJieDian:yanZhengJieDian]
    yanZhengQiXiang = qiXiangShuJu[xunLianJieDian:yanZhengJieDian]
    yanZhengShiJian = shiJianTeZheng[xunLianJieDian:yanZhengJieDian]
    yanZhengMuBiao = muBiaoZhi[xunLianJieDian:yanZhengJieDian]
    
    ceShiFuHe = fuHeShuJu[yanZhengJieDian:]
    ceShiQiXiang = qiXiangShuJu[yanZhengJieDian:]
    ceShiShiJian = shiJianTeZheng[yanZhengJieDian:]
    ceShiMuBiao = muBiaoZhi[yanZhengJieDian:]
    
    # 创建数据集
    xunLianJi = DianLiFuHeShuJuJi(
        xunLianFuHe, xunLianQiXiang, xunLianShiJian, xunLianMuBiao,
        xuLieChangDu, yuCeBuChang
    )
    yanZhengJi = DianLiFuHeShuJuJi(
        yanZhengFuHe, yanZhengQiXiang, yanZhengShiJian, yanZhengMuBiao,
        xuLieChangDu, yuCeBuChang
    )
    ceShiJi = DianLiFuHeShuJuJi(
        ceShiFuHe, ceShiQiXiang, ceShiShiJian, ceShiMuBiao,
        xuLieChangDu, yuCeBuChang
    )
    
    # 创建DataLoader
    piCiDaXiao = int(piCiDaXiao)   # JSON 加载后可能为 float，强制转 int
    xunLianJiaZaiQi = DataLoader(
        xunLianJi, batch_size=piCiDaXiao, shuffle=True, num_workers=0
    )
    yanZhengJiaZaiQi = DataLoader(
        yanZhengJi, batch_size=piCiDaXiao, shuffle=False, num_workers=0
    )
    ceShiJiaZaiQi = DataLoader(
        ceShiJi, batch_size=piCiDaXiao, shuffle=False, num_workers=0
    )
    
    return xunLianJiaZaiQi, yanZhengJiaZaiQi, ceShiJiaZaiQi


def jiaZaiYuChuLiShuJu(
    xuLieChangDu=96,
    yuCeBuChang=1,
    piCiDaXiao=64,
    xunLianLuJing=None,
    yanZhengLuJing=None,
    ceShiLuJing=None,
    biaoZhunHuaQiLuJing=None
):
    """
    从预处理后的 CSV 加载数据并构建 DataLoader
    三分：训练集(训练)、验证集(调参/早停)、测试集(最终评估)
    
    返回:
        (训练加载器, 验证加载器, 测试加载器, scalers字典)
    """
    train_path = xunLianLuJing or _TRAIN_CSV
    val_path = yanZhengLuJing or _VAL_CSV
    test_path = ceShiLuJing or _TEST_CSV
    scaler_path = biaoZhunHuaQiLuJing or _SCALERS_PKL
    
    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError(
            f"预处理数据不存在。请先运行: cd 数据预处理 && python data_preprocessing.py\n"
            f"  训练集: {train_path}\n  验证集: {val_path}\n  测试集: {test_path}"
        )
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    def _extract_arrays(df):
        fu_he = df['M019Value'].values.reshape(-1, 1).astype(np.float32)
        qi_xiang = df[['max_temp', 'min_temp', 'humidity', 'wind_speed']].values.astype(np.float32)
        shi_jian = df[_TIME_FEATURE_COLS].values.astype(np.float32)
        mu_biao = df['M019Value'].values.astype(np.float32)
        return fu_he, qi_xiang, shi_jian, mu_biao
    
    train_arrays = _extract_arrays(train_df)
    val_arrays = _extract_arrays(val_df)
    test_arrays = _extract_arrays(test_df)
    
    xunLianJi = DianLiFuHeShuJuJi(*train_arrays, xuLieChangDu, yuCeBuChang)
    yanZhengJi = DianLiFuHeShuJuJi(*val_arrays, xuLieChangDu, yuCeBuChang)
    ceShiJi = DianLiFuHeShuJuJi(*test_arrays, xuLieChangDu, yuCeBuChang)
    
    piCiDaXiao = int(piCiDaXiao)   # JSON 加载后可能为 float，强制转 int
    xunLianJiaZaiQi = DataLoader(xunLianJi, batch_size=piCiDaXiao, shuffle=True, num_workers=0)
    yanZhengJiaZaiQi = DataLoader(yanZhengJi, batch_size=piCiDaXiao, shuffle=False, num_workers=0)
    ceShiJiaZaiQi = DataLoader(ceShiJi, batch_size=piCiDaXiao, shuffle=False, num_workers=0)
    
    scalers = {}
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scalers = pickle.load(f)
    
    return xunLianJiaZaiQi, yanZhengJiaZaiQi, ceShiJiaZaiQi, scalers

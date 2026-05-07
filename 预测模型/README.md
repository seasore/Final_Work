# TCN-BiLSTM-Attention 预测模型设计与实现说明

## 一、设计概述

根据开题报告，本模块实现了基于 **TCN-BiLSTM-Attention** 混合架构的短期电力负荷预测模型。该架构融合了：

1. **时间卷积网络（TCN）**：并行多尺度特征提取
2. **双向长短期记忆网络（BiLSTM）**：双向时序建模
3. **注意力机制**：动态特征权重分配
4. **多源特征融合**：负荷+气象+时间多维信息整合

### 设计目标

- MAPE 降低超过 15%
- 单次预测响应时间 ≤ 1 分钟
- 适应极端天气、节假日等复杂场景

---

## 二、文件结构

```
预测模型/
├── __init__.py      # 包初始化
├── tcn.py           # TCN 时间卷积模块
├── bilstm.py        # BiLSTM 模块
├── attention.py     # 注意力机制模块
├── fusion.py       # 多源特征融合模块
├── model.py        # 主模型
├── data.py         # 数据加载器
├── train.py        # 训练流程
├── main.py         # 主程序入口
├── requirements.txt
└── README.md       # 本说明文档
```

---

## 三、使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行主程序（必须按顺序执行）

```bash
# 第一步：数据预处理（必须先运行）
cd 模型构建/数据预处理
python data_preprocessing.py

# 第二步：训练与评估
cd ../预测模型
python main.py
```

**重要**：必须先运行 `data_preprocessing.py`，否则 `main.py` 会因缺少 `train_data.csv`、`val_data.csv`、`scalers.pkl` 而报错。详见项目根目录 `运行顺序说明.md`。

### 使用模拟数据

```python
# 在 main.py 中调用
zhuHanShu(shiYongYuChuLiShuJu=False)
```

---

## 四、数据格式要求（与预处理输出一致）

- **负荷特征**：`(样本数, 1)` M019Value（已标准化）
- **气象特征**：`(样本数, 4)` [max_temp, min_temp, humidity, wind_speed]
- **时间特征**：`(样本数, 5)` [hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend]
- **采样间隔**：15 分钟

---

## 五、main.py 集成功能（开题报告补充）

运行 `python main.py` 将自动执行：

1. **Optuna 贝叶斯超参数搜索**：自动找最优学习率、隐藏维度、Dropout 等
2. **最终训练**：用最优参数训练
3. **时序 K 折交叉验证**：评估结果可靠性
4. **测试集评估**：MSE、MAE、MAPE、R²
5. **特殊场景验证**：节假日、极端高温、极端高湿的 MAPE

**可选参数**：`--no_optuna` 跳过 Optuna，`--no_cv` 跳过交叉验证

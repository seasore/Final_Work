# -*- coding: utf-8 -*-
"""
电力负荷数据预处理脚本
Data Preprocessing for Power Load Forecasting

根据开题报告要求实现：
1. 数据清洗（Data cleaning）
2. 异常值处理（Outlier handling）
3. 缺失值填补（Missing value imputation）
4. 标准化（Normalization）
5. 划分训练集、验证集、测试集（三分法）
"""

import os
import sys
import io

# 修复 Windows 控制台中文乱码（避免重复包装导致 closed file）
if sys.platform == 'win32' and getattr(sys.stdout, 'encoding', '').lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '数据')
PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(DATA_DIR, '训练集')
VAL_DIR = os.path.join(DATA_DIR, '验证集')
TEST_DIR = os.path.join(DATA_DIR, '测试集')


def load_raw_data():
    """加载原始负荷数据和气象数据"""
    fuhe_path = os.path.join(DATA_DIR, 'FuHe.xls')
    weather_path = os.path.join(DATA_DIR, '(ky_weather).xls')
    
    fuhe_df = pd.read_excel(fuhe_path)
    weather_df = pd.read_excel(weather_path)
    
    return fuhe_df, weather_df


def preprocess_load_data(fuhe_df):
    """
    负荷数据预处理
    - 数据清洗：去除重复、无效时间戳
    - 异常值处理：IQR方法
    - 缺失值填补：线性插值
    """
    df = fuhe_df.copy()
    
    # 1. 数据清洗
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.drop_duplicates(subset=['DATETIME'])
    df = df.sort_values('DATETIME').reset_index(drop=True)
    
    # 检查并处理负荷列
    load_col = 'M019Value'
    if load_col not in df.columns:
        raise ValueError(f"负荷数据列 {load_col} 不存在")
    
    # 将非数值转为NaN
    df[load_col] = pd.to_numeric(df[load_col], errors='coerce')
    
    # 2. 缺失值填补（线性插值）
    df[load_col] = df[load_col].interpolate(method='linear', limit_direction='both')
    # 若首尾仍有缺失，用前后向填充
    df[load_col] = df[load_col].ffill().bfill()
    
    # 3. 异常值处理（IQR方法）
    Q1 = df[load_col].quantile(0.25)
    Q3 = df[load_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 将异常值替换为边界值（或可用中位数替代）
    df.loc[df[load_col] < lower_bound, load_col] = lower_bound
    df.loc[df[load_col] > upper_bound, load_col] = upper_bound
    
    return df


def preprocess_weather_data(weather_df):
    """
    气象数据预处理
    列索引映射（因编码问题）：2=最高温, 3=最低温, 7=相对湿度, 10=平均风速
    - 缺失值填补
    - 异常值处理
    """
    df = weather_df.copy()
    
    # 列名映射（根据数据探索结果，使用英文便于跨平台）
    col_map = {
        2: 'max_temp',      # 最高温度
        3: 'min_temp',      # 最低温度
        7: 'humidity',      # 相对湿度
        10: 'wind_speed'    # 平均风速
    }
    
    # 提取所需列并重命名
    weather_clean = pd.DataFrame()
    weather_clean['DATE'] = pd.to_datetime(df['DATE'])
    
    for idx, name in col_map.items():
        col_data = df.iloc[:, idx].copy()
        # 处理<NULL>等非数值
        col_data = pd.to_numeric(col_data, errors='coerce')
        weather_clean[name] = col_data
    
    # 缺失值填补：按列用中位数或前向填充
    for col in list(col_map.values()):
        weather_clean[col] = weather_clean[col].fillna(weather_clean[col].median())
        weather_clean[col] = weather_clean[col].ffill().bfill()
    
    # 异常值处理（合理范围约束）
    # 温度：-20~50°C
    weather_clean['max_temp'] = weather_clean['max_temp'].clip(-20, 50)
    weather_clean['min_temp'] = weather_clean['min_temp'].clip(-20, 50)
    # 湿度：0~100%
    weather_clean['humidity'] = weather_clean['humidity'].clip(0, 100)
    # 风速：0~50 m/s
    weather_clean['wind_speed'] = weather_clean['wind_speed'].clip(0, 50)
    
    return weather_clean


def merge_load_and_weather(load_df, weather_df):
    """
    合并负荷数据与气象数据
    负荷为15分钟间隔，气象为日数据，按日期将气象数据扩展到每个15分钟点
    """
    load_df = load_df.copy()
    load_df['date'] = load_df['DATETIME'].dt.date
    
    weather_df = weather_df.copy()
    weather_df['date'] = weather_df['DATE'].dt.date
    
    weather_cols = ['date', 'max_temp', 'min_temp', 'humidity', 'wind_speed']
    merged = load_df.merge(
        weather_df[weather_cols],
        on='date',
        how='left'
    )
    merged = merged.drop(columns=['date'])
    
    # 若某日无气象数据，用前后日插值
    for col in ['max_temp', 'min_temp', 'humidity', 'wind_speed']:
        merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
        merged[col] = merged[col].ffill().bfill()
    
    return merged


def extract_time_features(df):
    """提取时间特征：小时、星期、是否周末（周期性编码）"""
    df = df.copy()
    df['hour'] = df['DATETIME'].dt.hour + df['DATETIME'].dt.minute / 60
    df['weekday'] = df['DATETIME'].dt.dayofweek
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    
    # 周期性编码
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    return df


def add_special_scenario_columns(df):
    """
    添加特殊场景标记列（开题报告：极端天气、节假日验证）
    - is_holiday: 法定节假日（含调休）
    - is_extreme_heat: 极端高温（最高温>=35°C）
    - is_extreme_humidity: 极端高湿（相对湿度>=90%，近似强降雨）
    """
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    
    # 法定节假日
    try:
        import chinese_calendar as calendar
        df['is_holiday'] = df['DATETIME'].apply(
            lambda t: 1 if calendar.is_holiday(t.date()) else 0
        ).astype(int)
    except ImportError:
        df['is_holiday'] = 0
    
    # 极端天气（使用原始气象值，标准化前添加）
    df['is_extreme_heat'] = (df['max_temp'] >= 35).astype(int)
    df['is_extreme_humidity'] = (df['humidity'] >= 90).astype(int)
    
    return df


FEATURE_COLS = ['M019Value', 'max_temp', 'min_temp', 'humidity', 'wind_speed']
TIME_FEATURE_COLS = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos', 'is_weekend']


def normalize_data(df, scalers=None, fit=True):
    """
    标准化
    对负荷、气象特征进行Z-score标准化
    """
    df_norm = df.copy()
    if scalers is None:
        scalers = {}
    for col in FEATURE_COLS:
        if col not in df_norm.columns:
            continue
        if fit:
            scaler = StandardScaler()
            df_norm[col] = scaler.fit_transform(df_norm[[col]])
            scalers[col] = scaler
        else:
            if col in scalers:
                df_norm[col] = scalers[col].transform(df_norm[[col]])
    return df_norm, scalers


def split_train_val_test(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=False):
    """
    按时间顺序三分：训练集、验证集、测试集
    - 训练集：用于训练模型
    - 验证集：训练过程中调参、早停、选最佳模型
    - 测试集：仅用于最终评估，训练全程不可见
    默认 70% / 15% / 15%
    """
    n = len(df)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_datasets(train_df, val_df, test_df):
    """保存训练集、验证集、测试集到对应文件夹"""
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    train_csv = os.path.join(TRAIN_DIR, 'train_data.csv')
    val_csv = os.path.join(VAL_DIR, 'val_data.csv')
    test_csv = os.path.join(TEST_DIR, 'test_data.csv')
    
    train_df.to_csv(train_csv, index=False, encoding='utf-8-sig')
    val_df.to_csv(val_csv, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_csv, index=False, encoding='utf-8-sig')
    
    train_xlsx = os.path.join(TRAIN_DIR, 'train_data.xlsx')
    val_xlsx = os.path.join(VAL_DIR, 'val_data.xlsx')
    test_xlsx = os.path.join(TEST_DIR, 'test_data.xlsx')
    for df, path, sheet in [(train_df, train_xlsx, '训练集'), (val_df, val_xlsx, '验证集'), (test_df, test_xlsx, '测试集')]:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name=sheet)
            worksheet = writer.sheets[sheet]
            worksheet.column_dimensions['A'].width = 22
    
    print(f"训练集已保存: {train_csv}，样本数: {len(train_df)}")
    print(f"验证集已保存: {val_csv}，样本数: {len(val_df)}")
    print(f"测试集已保存: {test_csv}，样本数: {len(test_df)}")


def run_preprocessing(train_ratio=0.8, run_correlation=True):
    """执行完整预处理流程"""
    print("=" * 50)
    print("电力负荷数据预处理")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n[1] 加载原始数据...")
    fuhe_df, weather_df = load_raw_data()
    print(f"  负荷数据: {len(fuhe_df)} 条")
    print(f"  气象数据: {len(weather_df)} 条")
    
    # 2. 负荷数据预处理
    print("\n[2] 负荷数据预处理（清洗、缺失值、异常值）...")
    load_df = preprocess_load_data(fuhe_df)
    print(f"  处理后负荷数据: {len(load_df)} 条")
    
    # 3. 气象数据预处理
    print("\n[3] 气象数据预处理...")
    weather_clean = preprocess_weather_data(weather_df)
    print(f"  处理后气象数据: {len(weather_clean)} 条")
    
    # 4. 合并
    print("\n[4] 合并负荷与气象数据...")
    merged = merge_load_and_weather(load_df, weather_clean)
    print(f"  合并后数据: {len(merged)} 条")
    
    # 5. 提取时间特征
    print("\n[5] 提取时间特征...")
    merged = extract_time_features(merged)
    
    # 5.5 添加特殊场景标记（节假日、极端天气，供验证时评估）
    print("\n[5.5] 添加特殊场景标记...")
    merged = add_special_scenario_columns(merged)
    
    # 6. 三分：训练集(70%)、验证集(15%)、测试集(15%)
    print("\n[6] 划分训练集、验证集、测试集...")
    train_df, val_df, test_df = split_train_val_test(
        merged, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )
    print(f"  训练集: {len(train_df)} 条 (70%)")
    print(f"  验证集: {len(val_df)} 条 (15%)")
    print(f"  测试集: {len(test_df)} 条 (15%)")
    
    # 7. 标准化（仅用训练集拟合scaler，验证集和测试集用相同参数变换）
    print("\n[7] 标准化...")
    scalers = {}
    for col in FEATURE_COLS:
        scaler = StandardScaler()
        scaler.fit(train_df[[col]])
        train_df[col] = scaler.transform(train_df[[col]])
        val_df[col] = scaler.transform(val_df[[col]])
        test_df[col] = scaler.transform(test_df[[col]])
        scalers[col] = scaler
    
    # 8. 保存
    print("\n[8] 保存数据集...")
    save_datasets(train_df, val_df, test_df)
    
    # 保存scaler供预测时使用
    import pickle
    scaler_path = os.path.join(PREPROCESS_DIR, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"  标准化器已保存: {scaler_path}")
    
    # 9. 可选：相关性分析
    if run_correlation:
        print("\n[9] 相关性分析（Pearson + 分组探索）...")
        try:
            run_correlation_analysis()
        except Exception as e:
            print(f"  相关性分析跳过: {e}")
    
    print("\n预处理完成！")
    return train_df, val_df, test_df, scalers


def run_correlation_analysis():
    """运行相关性分析（Pearson + 分组探索），量化负荷与特征的关联"""
    try:
        from correlation_analysis import run_correlation_analysis as _run
        return _run()
    except ImportError:
        print("提示：可单独运行 python correlation_analysis.py 进行相关性分析")


def load_preprocessed_data():
    """
    加载预处理后的训练集、验证集、测试集
    返回格式可直接用于预测模型的 data.py
    """
    train_path = os.path.join(TRAIN_DIR, 'train_data.csv')
    val_path = os.path.join(VAL_DIR, 'val_data.csv')
    test_path = os.path.join(TEST_DIR, 'test_data.csv')
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    train_df['DATETIME'] = pd.to_datetime(train_df['DATETIME'])
    val_df['DATETIME'] = pd.to_datetime(val_df['DATETIME'])
    test_df['DATETIME'] = pd.to_datetime(test_df['DATETIME'])
    
    def extract_arrays(df):
        fu_he = df['M019Value'].values.reshape(-1, 1)
        qi_xiang = df[['max_temp', 'min_temp', 'humidity', 'wind_speed']].values
        shi_jian = df[TIME_FEATURE_COLS].values
        mu_biao = df['M019Value'].values
        return fu_he, qi_xiang, shi_jian, mu_biao
    
    train_arrays = extract_arrays(train_df)
    val_arrays = extract_arrays(val_df)
    test_arrays = extract_arrays(test_df)
    
    return train_arrays, val_arrays, test_arrays, train_df, val_df, test_df


if __name__ == '__main__':
    run_preprocessing()

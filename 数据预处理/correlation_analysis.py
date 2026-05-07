# -*- coding: utf-8 -*-
"""
相关性分析与分组探索
Correlation Analysis and Grouped Exploration

根据开题报告：
- 采用 Pearson 相关分析法量化气象因素（温度、湿度、风速）对负荷的影响程度
- 分析不同季节、天气条件下负荷模式的差异，为模型场景分类提供依据
"""

import os
import sys
import io

# 仅当独立运行时修复控制台编码（被 data_preprocessing 调用时不再重复包装）
if __name__ == '__main__' and sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import chinese_calendar as calendar
    HAS_CHINESE_CALENDAR = True
except ImportError:
    HAS_CHINESE_CALENDAR = False

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '数据')
PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(DATA_DIR, '训练集')
OUTPUT_DIR = os.path.join(PREPROCESS_DIR, 'correlation_report')
# 毕业论文插图目录（与预处理结果一致，便于 Word/PPT 引用）
THESIS_ILLUSTR_DIR = os.path.join(BASE_DIR, '毕业论文', '插图')


def load_data_for_analysis():
    """加载预处理前的合并数据（未标准化，便于解释相关性）"""
    from data_preprocessing import (
        load_raw_data, preprocess_load_data, preprocess_weather_data,
        merge_load_and_weather, extract_time_features, split_train_val_test
    )
    fuhe_df, weather_df = load_raw_data()
    load_df = preprocess_load_data(fuhe_df)
    weather_clean = preprocess_weather_data(weather_df)
    merged = merge_load_and_weather(load_df, weather_clean)
    merged = extract_time_features(merged)
    train_df, _, _ = split_train_val_test(merged, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    return train_df


def add_holiday_column(df):
    """添加法定节假日标记（使用 chinese_calendar，含调休）"""
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    if HAS_CHINESE_CALENDAR:
        df['is_holiday'] = df['DATETIME'].apply(
            lambda t: 1 if calendar.is_holiday(t.date()) else 0
        )
        df['holiday_type'] = df['is_holiday'].map({0: '非节假日', 1: '法定节假日'})
    else:
        df['is_holiday'] = 0
        df['holiday_type'] = '非节假日'
    return df


def add_group_columns(df):
    """添加分组列：季节、工作日/周末、法定节假日"""
    df = df.copy()
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    month = df['DATETIME'].dt.month
    # 季节：3-5春, 6-8夏, 9-11秋, 12/1/2冬
    df['season'] = month.apply(lambda m: '春' if 3 <= m <= 5 else '夏' if 6 <= m <= 8 else '秋' if 9 <= m <= 11 else '冬')
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['day_type'] = df['is_weekend'].map({0: '工作日', 1: '周末'})
    # 法定节假日（含调休）
    df = add_holiday_column(df)
    return df


def pearson_correlation_with_load(df, target_col='M019Value', feature_cols=None):
    """
    Pearson 相关性分析：负荷与各特征的线性关联
    返回 (相关系数, p值) 字典
    """
    if feature_cols is None:
        feature_cols = ['max_temp', 'min_temp', 'humidity', 'wind_speed', 'hour', 'weekday']
    
    results = {}
    for col in feature_cols:
        if col not in df.columns:
            continue
        valid = df[[target_col, col]].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.pearsonr(valid[target_col], valid[col])
        results[col] = {'r': r, 'p': p, 'abs_r': abs(r)}
    return results


def grouped_correlation(df, group_col, target_col='M019Value', feature_cols=None):
    """
    分组相关性：不同场景下负荷与特征的关联可能不同
    例如：夏季温度与负荷相关性更强（空调），冬季可能不同
    """
    if feature_cols is None:
        feature_cols = ['max_temp', 'min_temp', 'humidity', 'wind_speed']
    
    groups = df.groupby(group_col)
    all_results = {}
    for name, g in groups:
        if len(g) < 100:  # 样本太少则跳过
            continue
        corr = pearson_correlation_with_load(g, target_col, feature_cols)
        all_results[name] = corr
    return all_results


def _save_correlation_heatmap(corr_matrix, col_names, output_dir, thesis_dir=None):
    """将相关系数矩阵绘制为热力图并保存为图片（英文标注、X 轴水平）。"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    display_names_en = {
        'M019Value': 'Load (M019Value)',
        'max_temp': 'Max. temperature',
        'min_temp': 'Min. temperature',
        'humidity': 'Relative humidity',
        'wind_speed': 'Wind speed',
        'hour': 'Hour',
        'weekday': 'Weekday',
    }
    labels = [display_names_en.get(c, str(c).replace('_', ' ')) for c in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(corr_matrix.values, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # 在每个格子中显示相关系数
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.iloc[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Pearson correlation coefficient', fontsize=11)
    ax.set_title('Pearson correlation matrix: load and features (training subset)', fontsize=14)
    fig.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    paths = [os.path.join(output_dir, 'correlation_matrix.png')]
    if thesis_dir:
        os.makedirs(thesis_dir, exist_ok=True)
        paths.append(os.path.join(thesis_dir, 'fig_correlation_matrix_pearson.png'))
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"相关系数矩阵热力图已保存: {paths[0]}" + (f"；论文插图: {paths[1]}" if len(paths) > 1 else ""))


def _save_pearson_bars_vs_load(corr_all, col_names, output_dir, thesis_dir=None):
    """
    负荷与各特征的 Pearson 相关柱状图（|r|×100%，便于与 PPT「相关度/%」对照）。
    数据与热力图同源：训练子集、预处理合并后未标准化。
    导出图稿为英文标注、X 轴水平，便于国际答辩/英文稿。
    """
    if not corr_all:
        return
    # 英文图用西文字体优先
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    labels_en = {
        'max_temp': 'Max. temperature',
        'min_temp': 'Min. temperature',
        'humidity': 'Rel. humidity',
        'wind_speed': 'Wind speed',
        'hour': 'Hour',
        'weekday': 'Weekday',
    }
    items = sorted(corr_all.items(), key=lambda x: -x[1]['abs_r'])
    labels = [labels_en.get(c, str(c).replace('_', ' ')) for c, _ in items]
    rs = [v['r'] for _, v in items]
    pct = [abs(r) * 100.0 for r in rs]
    colors = ['#4472C4' if r >= 0 else '#C55A11' for r in rs]

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    x = np.arange(len(labels))
    ax.bar(x, pct, color=colors, edgecolor='#333333', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel(r'Linear correlation strength $|r|\times 100$ (%)')
    ax.set_title(
        'Load–feature association via Pearson correlation (exploratory, training subset)',
        fontsize=12,
    )
    ax.set_ylim(0, max(15, max(pct) * 1.15) if pct else 1)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    # 柱顶标注符号化 r
    for i, r in enumerate(rs):
        ax.text(i, pct[i] + 0.8, f'{r:+.3f}', ha='center', va='bottom', fontsize=9, color='#333333')
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    paths = [os.path.join(output_dir, 'pearson_load_vs_features_bar.png')]
    if thesis_dir:
        os.makedirs(thesis_dir, exist_ok=True)
        paths.append(os.path.join(thesis_dir, 'fig_pearson_load_vs_features_bar.png'))
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Pearson 柱状图已保存: {paths[0]}" + (f"；论文插图: {paths[1]}" if len(paths) > 1 else ""))


def run_correlation_analysis():
    """执行完整相关性分析并输出报告"""
    print("=" * 55)
    print("负荷与特征相关性分析（Pearson + 分组探索）")
    print("=" * 55)
    
    df = load_data_for_analysis()
    df = add_group_columns(df)
    
    feature_cols = ['max_temp', 'min_temp', 'humidity', 'wind_speed', 'hour', 'weekday']
    col_names = {
        'max_temp': '最高温度', 'min_temp': '最低温度',
        'humidity': '相对湿度', 'wind_speed': '平均风速',
        'hour': '小时', 'weekday': '星期'
    }
    
    # 1. 整体 Pearson 相关性
    print("\n[1] 整体 Pearson 相关性（负荷 vs 各特征）")
    print("-" * 50)
    corr_all = pearson_correlation_with_load(df, feature_cols=feature_cols)
    
    lines = []
    lines.append("特征\t\t相关系数 r\tp值\t\t|r| 强度")
    lines.append("-" * 50)
    for col, v in sorted(corr_all.items(), key=lambda x: -x[1]['abs_r']):
        name = col_names.get(col, col)
        r, p = v['r'], v['p']
        strength = "强" if abs(r) > 0.5 else "中" if abs(r) > 0.3 else "弱"
        lines.append(f"{name}\t\t{r:.4f}\t\t{p:.2e}\t{strength}")
    
    for line in lines:
        print(line)
    
    # 2. 按季节分组的相关性
    print("\n[2] 按季节分组的相关性（不同季节关联可能不同）")
    print("-" * 50)
    corr_season = grouped_correlation(df, 'season', feature_cols=feature_cols)
    
    for season in ['春', '夏', '秋', '冬']:
        if season not in corr_season:
            continue
        print(f"\n  {season}季 (n≈{len(df[df['season']==season])}):")
        for col in ['max_temp', 'min_temp', 'humidity', 'wind_speed']:
            if col in corr_season[season]:
                r = corr_season[season][col]['r']
                print(f"    {col_names[col]}: r={r:.4f}")
    
    # 3. 按工作日/周末分组
    print("\n[3] 按工作日/周末分组的相关性")
    print("-" * 50)
    corr_day = grouped_correlation(df, 'day_type', feature_cols=feature_cols)
    for day_type in ['工作日', '周末']:
        if day_type not in corr_day:
            continue
        print(f"\n  {day_type} (n≈{len(df[df['day_type']==day_type])}):")
        for col in feature_cols:
            if col in corr_day[day_type]:
                r = corr_day[day_type][col]['r']
                print(f"    {col_names.get(col, col)}: r={r:.4f}")
    
    # 4. 法定节假日与负荷的相关性分析
    print("\n[4] 法定节假日与电力负荷的相关性分析")
    print("-" * 50)
    corr_holiday = None
    if HAS_CHINESE_CALENDAR and 'is_holiday' in df.columns:
        n_holiday = df['is_holiday'].sum()
        n_work = len(df) - n_holiday
        print(f"  法定节假日样本: {n_holiday} 条，非节假日样本: {n_work} 条")
        # 点二列相关：节假日(0/1)与负荷的关联
        r_holiday, p_holiday = stats.pearsonr(df['M019Value'], df['is_holiday'])
        print(f"  节假日与负荷的相关系数 r = {r_holiday:.4f} (p = {p_holiday:.2e})")
        if abs(r_holiday) > 0.1:
            direction = "正" if r_holiday > 0 else "负"
            print(f"  → 节假日与负荷呈{direction}相关")
        # 分组均值对比
        mean_holiday = df[df['is_holiday'] == 1]['M019Value'].mean()
        mean_non = df[df['is_holiday'] == 0]['M019Value'].mean()
        print(f"  节假日平均负荷: {mean_holiday:.2f}，非节假日平均负荷: {mean_non:.2f}")
        print(f"  差异: {mean_holiday - mean_non:+.2f} ({(mean_holiday/mean_non-1)*100:+.1f}%)")
        # 按节假日/非节假日分组的相关性
        corr_holiday = grouped_correlation(df, 'holiday_type', feature_cols=feature_cols)
        for htype in ['法定节假日', '非节假日']:
            if htype not in corr_holiday:
                continue
            n = len(df[df['holiday_type'] == htype])
            print(f"\n  {htype} (n={n}):")
            for col in feature_cols:
                if col in corr_holiday[htype]:
                    r = corr_holiday[htype][col]['r']
                    print(f"    {col_names.get(col, col)}: r={r:.4f}")
        lines.append(f"\n节假日与负荷: r={r_holiday:.4f}, 节假日均负荷={mean_holiday:.2f}, 非节假日均负荷={mean_non:.2f}")
    else:
        print("  未安装 chinese-calendar，无法识别法定节假日。请运行: pip install chinese-calendar")
    
    # 5. 保存报告
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, 'correlation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        f.write("\n\n--- 分组相关性（季节/工作日/节假日）详见控制台输出 ---\n")
    print(f"\n报告已保存: {report_path}")
    
    # 6. 保存相关性矩阵（供后续特征筛选）
    corr_matrix = df[['M019Value'] + feature_cols].corr()
    matrix_path = os.path.join(OUTPUT_DIR, 'correlation_matrix.csv')
    corr_matrix.to_csv(matrix_path, encoding='utf-8-sig')
    print(f"相关性矩阵已保存: {matrix_path}")
    
    # 7. 将相关系数矩阵保存为热力图图片，并导出柱状图与论文插图目录
    if HAS_MATPLOTLIB:
        _save_correlation_heatmap(corr_matrix, col_names, OUTPUT_DIR, thesis_dir=THESIS_ILLUSTR_DIR)
        _save_pearson_bars_vs_load(corr_all, col_names, OUTPUT_DIR, thesis_dir=THESIS_ILLUSTR_DIR)
    else:
        print("  提示：未安装 matplotlib，无法生成热力图。请运行: pip install matplotlib")
    
    print("\n分析完成！")
    return corr_all, corr_season, corr_day, corr_holiday


if __name__ == '__main__':
    run_correlation_analysis()

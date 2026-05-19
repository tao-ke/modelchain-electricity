"""
电力价格预测系统 - 高级用法示例

这个示例展示了电力价格预测系统的高级功能，包括：
- 自定义特征工程
- 超参数优化
- 集成学习
- 模型解释性分析
- 在线学习和模型更新
- 天气数据集成（Open-Meteo 历史+预报）
"""

import pandas as pd
import numpy as np
import warnings
import yaml
from pathlib import Path
import matplotlib.pylab as plt
import sys, os
import psycopg2

# TimescaleDB 数据库连接配置
DB_CONFIG = {
    'dbname': 'Electricity',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432'
}

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei',"SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 导入天气API模块
try:
    from weather_api import (
        fetch_historical_weather,
        fetch_forecast_weather,
        fetch_weather_for_period,
        resample_weather_to_15min,
        align_weather_to_price_data,
        get_location_from_console,
    )
    WEATHER_AVAILABLE = True
except ImportError:
    print("[警告] weather_api.py 未找到，天气特征功能不可用")
    WEATHER_AVAILABLE = False

def load_electricity_data(path):
    """加载电力价格数据"""
    try:
        df = pd.read_excel(path)
        print(f"[OK] 数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"[WARN] 数据加载失败: {e}")
        return None
    # 第1列是日期
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # 第2列到最后是96个时段
    time_cols = df.columns[1:97]

    # 宽转长
    df_long = df.melt(
        id_vars=[date_col],
        value_vars=time_cols,
        var_name="timestep",
        value_name="price"
    )

    # 统一时段字符串（防止列名里有空格）
    df_long["timestep"] = df_long["timestep"].astype(str).str.strip()

    # 合并日期+时段为时间戳
    df_long["time"] = pd.to_datetime(
        df_long[date_col].dt.strftime("%Y/%m/%d") + " " + df_long["timestep"],
        errors="coerce"
    )

    # 丢弃无法解析的行并按时间排序
    df_long = df_long.dropna(subset=["time"]).sort_values("time")

    # 输出两列
    electricity_price = df_long[["time", "price"]].copy()
    electricity_price = electricity_price.set_index("time")
    electricity_price.columns = ["price"]

    if not isinstance(electricity_price.index, pd.DatetimeIndex):
        electricity_price.index = pd.to_datetime(electricity_price.index, errors="coerce")
        electricity_price = electricity_price[~electricity_price.index.isna()]
    return electricity_price


def _parse_single_station_file(path):
    """解析单个电站文件，返回DataFrame，包含时间索引和price列。
    支持宽表（日期在第一列，后续列为时段）或已是长表（含'time'/'timestamp'和'price'列）。
    """
    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"[WARN] 读取文件失败: {path} -> {e}")
        return None

    cols = list(df.columns)
    # 已经是长表
    lower_cols = [c.lower() for c in cols]
    if 'time' in lower_cols or 'timestamp' in lower_cols:
        # 尝试找到时间和价格列
        time_col = cols[lower_cols.index('time')] if 'time' in lower_cols else cols[lower_cols.index('timestamp')]
        price_candidates = [c for c in cols if str(c).lower() in ('price', '电价', 'price(元)')]
        price_col = price_candidates[0] if price_candidates else cols[-1]

        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        out = df[[time_col, price_col]].copy()
        out.columns = ['time', 'price']
        out['time'] = pd.to_datetime(out['time'])
        out = out.set_index('time').sort_index()
        return out

    # 否则尝试宽表（第一列日期，其余为时段）
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    time_cols = df.columns[1:97]  # 96个时段
    if len(time_cols) == 0:
        return None

    df_long = df.melt(
        id_vars=[date_col],
        value_vars=time_cols,
        var_name="timestep",
        value_name="price"
    )

    df_long["timestep"] = df_long["timestep"].astype(str).str.strip()
    df_long["timestamp"] = pd.to_datetime(
        df_long[date_col].dt.strftime("%Y/%m/%d") + " " + df_long["timestep"],
        errors="coerce"
    )

    # 只保留需要的两列，避免列数不匹配
    out = df_long[["timestamp", "price"]].dropna(subset=["timestamp"]).copy()
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna(subset=["price"]).sort_values("timestamp")
    out = out.set_index("timestamp")
    out.index.name = "time"
    return out


def load_station_electricity_data(station_name, base_dir=None, years=(2024, 2025, 2026)):
    """从TimescaleDB读取实时电价数据，返回DataFrame（timestamp, price）。"""
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        query = """
            SELECT time, price
            FROM real_time_electricity_price
            WHERE station = %s
              AND time >= %s AND time < %s
            ORDER BY time
        """
        df = pd.read_sql_query(query, conn, params=(station_name, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        print(f"[WARN] 数据库中未找到 {station_name} 的实时电价数据 ({years})")
        return None

    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'price']).sort_values('timestamp').reset_index(drop=True)

    print(f"[OK] 实时电价加载成功: {station_name}, {len(df)} 条记录, "
          f"范围 {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df


def load_dayahead_price_data(station_name, base_dir=None, years=(2024, 2025, 2026)):
    """从TimescaleDB读取日前电价数据，返回DataFrame（timestamp, dayahead_price）。"""
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        query = """
            SELECT time, price AS dayahead_price
            FROM day_ahead_price
            WHERE station = %s
              AND time >= %s AND time < %s
            ORDER BY time
        """
        df = pd.read_sql_query(query, conn, params=(station_name, start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        print(f"  未找到日前电价数据: {station_name}")
        return None

    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['dayahead_price'] = pd.to_numeric(df['dayahead_price'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'dayahead_price']).sort_values('timestamp').reset_index(drop=True)

    print(f"  日前电价加载成功: {station_name}, {len(df)} 条记录")
    return df


def _merge_dayahead_data(data, station_name, base_dir='电价数据', years=(2024, 2025, 2026)):
    """加载日前电价并按 timestamp 精确合并到实时电价数据。"""
    da_data = load_dayahead_price_data(station_name, base_dir, years)
    if da_data is None or da_data.empty:
        print("  [日前电价] 无数据，跳过合并")
        return data
    if 'timestamp' not in data.columns:
        data = data.reset_index()
    data['_key'] = pd.to_datetime(data['timestamp']).astype('int64')
    da_data['_key'] = pd.to_datetime(da_data['timestamp']).astype('int64')
    merged = data.merge(da_data[['_key', 'dayahead_price']], on='_key', how='left')
    merged.drop(columns=['_key'], inplace=True)
    match_rate = merged['dayahead_price'].notna().mean()
    print(f"  [日前电价] 合并完成: 匹配率 {match_rate:.1%}")
    if match_rate == 0:
        return data
    merged['dayahead_price'] = merged['dayahead_price'].ffill().bfill()
    return merged


def load_demand_data(base_dir=None, years=(2024, 2025, 2026)):
    """从TimescaleDB读取全省负荷数据。

    Returns: DataFrame(timestamp, demand) 或 None
    """
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        query = """
            SELECT time, provincial_class_b_power_mw AS demand,
                   local_power_output_mw AS local_power_output
            FROM demand
            WHERE time >= %s AND time < %s
            ORDER BY time
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    finally:
        conn.close()

    if df.empty:
        print("  负荷数据解析后无有效数据")
        return None

    df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
    df['local_power_output'] = pd.to_numeric(df['local_power_output'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'demand']).sort_values('timestamp').reset_index(drop=True)

    local_info = ""
    if df['local_power_output'].notna().any():
        local_info = (f", 地方出力 {df['local_power_output'].min():.0f}~"
                      f"{df['local_power_output'].max():.0f} MW")
    print(f"  负荷数据加载成功: {len(df)} 条记录, "
          f"范围 {df['demand'].min():.0f}~{df['demand'].max():.0f} MW{local_info}")
    return df


def _merge_demand_data(data, base_dir='负荷数据', years=(2024, 2025, 2026)):
    """加载全省负荷数据并按 timestamp 合并到电价数据。

    负荷是全省实时总有功负荷，与电价同时段。特征工程中会对其做滞后处理，
    确保不引入未来信息。
    返回合并了 demand 列的 DataFrame。
    """
    demand_data = load_demand_data(base_dir, years)
    if demand_data is None or demand_data.empty:
        print("  [负荷] 无数据，跳过合并")
        return data

    if 'timestamp' not in data.columns:
        data = data.reset_index()
    data['_key'] = pd.to_datetime(data['timestamp']).astype('int64')
    demand_data['_key'] = pd.to_datetime(demand_data['timestamp']).astype('int64')

    merge_cols = ['_key', 'demand']
    if 'local_power_output' in demand_data.columns:
        merge_cols.append('local_power_output')

    merged = data.merge(
        demand_data[merge_cols],
        on='_key', how='left'
    )
    merged.drop(columns=['_key'], inplace=True)

    match_rate = merged['demand'].notna().mean()
    print(f"  [负荷] 合并完成: 匹配率 {match_rate:.1%}")

    if match_rate == 0:
        print("  [负荷] 匹配失败，时间戳可能不一致，跳过")
        return data

    merged['demand'] = merged['demand'].ffill().bfill()
    if 'local_power_output' in merged.columns:
        merged['local_power_output'] = merged['local_power_output'].ffill().bfill()
    return merged


def sanitize_features(X, clip_value=1e6):
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.clip(-clip_value, clip_value)
    return X

class AdvancedElectricityPredictor:
    """高级电力价格预测器"""
    
    def __init__(self, config_path=None, lat=None, lon=None):
        """初始化预测器

        Args:
            config_path: 配置文件路径
            lat, lon: 电站经纬度（用于获取天气数据）
        """
        self.config = self.load_config(config_path)
        self.models = {}
        self.feature_importance = {}
        self.performance_history = []
        self.lat = lat
        self.lon = lon
        self.weather_cache = {}
        
    def load_config(self, config_path):
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # 默认配置
            return {
                'feature_engineering': {
                    'lag_periods': [1, 2, 4, 96, 192, 672],
                    'rolling_windows': [4, 12, 96, 672],
                    'fourier_periods': [296, 672],
                    'enable_target_encoding': True,
                    'enable_interactions': True,
                    'enable_weather': True,       # 是否启用天气特征
                    'weather_lag_periods': [1, 4, 96],  # 天气滞后特征
                    'weather_rolling_windows': [4, 24, 96],  # 天气滚动窗口
                },
                'model_params': {
                    'lightgbm': {
                        'objective': 'huber',
                        'alpha': 0.9,
                        'metric': 'huber',
                        'num_leaves': 63,
                        'learning_rate': 0.03,
                        'feature_fraction': 0.6,
                        'bagging_fraction': 0.8,
                        'reg_alpha': 0.1,
                        'reg_lambda': 0.5,
                        'random_state': 42,
                        'bagging_freq': 5,
                        'min_child_samples': 40,
                        'min_child_weight': 0.001,
                        'verbosity': -1
                    }
                    
                    # 备选：如果想专门预测"高电价(尖峰)"可以启用分位数回归，例如预测 90% 分位数：
                    #params = {
                    #    'objective': 'quantile', 
                    #    'alpha': 0.9,     # 0.9表示预测P90上限，0.5即为中位数(MAE)
                    #    'metric': 'quantile', ...
                    #}
                    
                },
                'ensemble': {
                    'use_stacking': True,
                    'use_blending': True,
                    'cv_folds': 5
                },
                'two_stage': {
                    'enable': True,           # 是否启用两阶段模型
                    'spike_percentile': 80,   # 尖峰分位数阈值（高于此分位数为尖峰）
                    'use_blend': True,        # 是否使用概率混合（否则硬分类）
                },
                'prediction': {
                    # Delta mode makes regressors learn price - price_lag_1.
                    # It keeps price-level accuracy while focusing learning on
                    # the next-step movement.
                    'target_mode': 'delta',
                    'baseline_col': 'price_lag_1',
                    'direction_deadband': 0.005,
                    'direction_weight': 0.25,
                    'direction_correction': True,
                    'direction_correction_min_proba': 0.55,
                    'direction_max_rmse_increase': 0.03,
                    'use_direction_ensemble': True,
                    'optimize_ensemble_weights': True,
                },
                'optimization': {
                    'n_trials': 100,
                    'timeout': 3600
                }
            }
    
    def advanced_feature_engineering(self, data):
        """高级特征工程"""
        print("开始高级特征工程...")
        
        df = data.copy()
        # 兼容处理：如果没有 timestamp 列但索引是时间索引，则自动转换
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.reset_index().columns[0]: 'timestamp'})
            elif 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            else:
                raise KeyError("输入数据缺少 'timestamp' 列")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 1. 基础时间特征
        df = self._create_time_features(df)
        
        # 2. 滞后特征和滚动统计
        df = self._create_lag_and_rolling_features(df)
        
        # 3. 傅里叶变换特征
        if self.config['feature_engineering']['fourier_periods']:
            df = self._create_fourier_features(df)
        
        # 4. 目标编码（如果有类别特征）
        if self.config['feature_engineering']['enable_target_encoding']:
            df = self._create_target_encoding(df)
        
        # 5. 交叉特征
        if self.config['feature_engineering']['enable_interactions']:
            df = self._create_interaction_features(df)
        
        # 6. 天气特征（如果数据中包含天气列且启用了天气特征）
        if self.config['feature_engineering'].get('enable_weather', True):
            weather_cols_in_data = [c for c in df.columns if c in [
                'temp_air', 'wind_speed', 'wind_direction', 'pressure',
                'ghi', 'dni', 'dhi', 'dew_point', 'et0', 'vpd'
            ]]
            if weather_cols_in_data:
                df = self._create_weather_features(df)

        # 6.5 日前电价特征（如果数据中包含日前电价列）
        if 'dayahead_price' in df.columns:
            df = self._create_dayahead_features(df)

        # 6.6 负荷特征（如果数据中包含负荷列）
        if 'demand' in df.columns:
            df = self._create_demand_features(df)

        # 7. 技术指标特征
        df = self._create_technical_indicators(df)

        # 8. 异常检测特征
        df = self._create_anomaly_features(df)

        print(f"特征工程完成，最终特征数: {df.shape[1]}")
        return df
    
    def _create_time_features(self, df):
        """创建时间特征（含中国节假日、电价峰谷时段、季节需求）"""
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['period_of_day'] = df['hour'] * 4 + df['minute'] // 15
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # ---- 中国电力市场时段划分 ----
        h = df['hour'] + df['minute'] / 60.0
        # 尖峰: 11:00-13:00, 17:00-20:00 (夏季可扩展)
        df['is_sharp_peak'] = (((h >= 11) & (h < 13)) | ((h >= 17) & (h < 20))).astype(int)
        # 高峰: 8:00-11:00, 13:00-17:00, 20:00-22:00
        df['is_peak'] = (((h >= 8) & (h < 11)) | ((h >= 13) & (h < 17)) | ((h >= 20) & (h < 22))).astype(int)
        # 平段: 6:00-8:00, 22:00-24:00
        df['is_shoulder'] = (((h >= 6) & (h < 8)) | (h >= 22)).astype(int)
        # 谷段: 0:00-6:00
        df['is_valley'] = ((h >= 0) & (h < 6)).astype(int)

        # 4段离散编码: 0=谷 1=平 2=峰 3=尖峰
        df['price_period'] = 1  # 默认平段
        df.loc[df['is_valley'] == 1, 'price_period'] = 0
        df.loc[df['is_peak'] == 1, 'price_period'] = 2
        df.loc[df['is_sharp_peak'] == 1, 'price_period'] = 3

        # ---- 周期性编码 ----
        df['hour_sin'] = np.sin(2 * np.pi * df['period_of_day'] / 96)
        df['hour_cos'] = np.cos(2 * np.pi * df['period_of_day'] / 96)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

        # ---- 季节 / 制冷供暖需求 ----
        df['season'] = df['month'] % 12 // 3  # 0=冬 1=春 2=夏 3=秋
        df['is_cooling_season'] = df['month'].isin([6, 7, 8, 9]).astype(int)  # 迎峰度夏
        df['is_heating_season'] = df['month'].isin([12, 1, 2]).astype(int)     # 迎峰度冬
        df['is_shoulder_season'] = (~df['month'].isin([6, 7, 8, 9, 12, 1, 2])).astype(int)

        # ---- 中国法定节假日特征 ----
        df = self._add_holiday_features(df)

        return df

    @staticmethod
    def _add_holiday_features(df):
        """注入中国法定节假日特征（基于 chinesecalendar 库动态获取）。

        节假日对电价有显著影响：工业负荷下降，电价整体走低。
        关键区分：春节（影响最大）、国庆黄金周、小长假、普通周末。
        """
        try:
            from chinese_calendar import is_holiday, is_workday, is_in_lieu, get_holiday_detail
        except ImportError:
            raise ImportError(
                "需要安装 chinesecalendar 库以获取节假日信息，请执行: pip install chinesecalendar"
            )

        dates = pd.to_datetime(df['timestamp'].dt.date).dt.date
        n = len(df)

        # ---------- 逐日查询 chinesecalendar ----------
        is_holiday_arr = np.zeros(n, dtype=int)
        is_adjusted_workday = np.zeros(n, dtype=int)  # 调休上班（周末变工作日）
        holiday_name_arr = [None] * n
        for i, d in enumerate(dates):
            is_holiday_arr[i] = 1 if is_holiday(d) else 0
            is_adjusted_workday[i] = 1 if is_in_lieu(d) else 0
            _, name = get_holiday_detail(d)
            holiday_name_arr[i] = name if is_holiday(d) else None

        holiday_s = pd.Series(is_holiday_arr, index=df.index)

        # ---------- 节日名称分类 ----------
        is_spring_festival = np.zeros(n, dtype=int)
        is_national_day = np.zeros(n, dtype=int)
        for i, name in enumerate(holiday_name_arr):
            if name is None:
                continue
            if '春节' in name or 'Spring Festival' in name:
                is_spring_festival[i] = 1
            elif '国庆' in name or 'National Day' in name:
                is_national_day[i] = 1

        # ---------- 识别连续假日块，区分长/短假期 ----------
        is_long_holiday = np.zeros(n, dtype=int)
        is_short_holiday = np.zeros(n, dtype=int)
        holiday_type = np.zeros(n, dtype=int)  # 0=工作日 1=周末 2=短假 3=长假

        in_block = False
        block_start = 0
        for i in range(n):
            if is_holiday_arr[i]:
                if not in_block:
                    in_block = True
                    block_start = i
                if i == n - 1 or not is_holiday_arr[i + 1]:
                    block_len = i - block_start + 1
                    if block_len >= 5:
                        is_long_holiday[block_start:i + 1] = 1
                        holiday_type[block_start:i + 1] = 3
                    else:
                        is_short_holiday[block_start:i + 1] = 1
                        holiday_type[block_start:i + 1] = 2
                    in_block = False

        # 春节和国庆始终标记为长假
        is_long_holiday[is_spring_festival == 1] = 1
        holiday_type[is_spring_festival == 1] = 3
        is_long_holiday[is_national_day == 1] = 1
        holiday_type[is_national_day == 1] = 3

        # ---------- 周末 ----------
        is_weekend = ((df['day_of_week'] >= 5) & (is_holiday_arr == 0)).astype(int)
        # 调休上班日修正：周末但上班 -> 按工作日处理
        is_weekend[is_adjusted_workday == 1] = 0
        holiday_type[is_adjusted_workday == 1] = 0

        # 普通周末标记 holiday_type=1
        holiday_type[(is_weekend == 1) & (holiday_type == 0)] = 1

        # ---------- 写入基础列 ----------
        df['is_holiday'] = is_holiday_arr
        df['is_spring_festival'] = is_spring_festival
        df['is_national_day'] = is_national_day
        df['is_long_holiday'] = is_long_holiday
        df['is_short_holiday'] = is_short_holiday
        df['holiday_type'] = holiday_type
        df['is_weekend'] = is_weekend

        # ---------- 假期前/后效应 ----------
        df['is_pre_holiday'] = (
            holiday_s.shift(-1).fillna(0).astype(int)
            & (~holiday_s.astype(bool)).astype(int)
        )
        df['is_post_holiday'] = (
            holiday_s.shift(1).fillna(0).astype(int)
            & (~holiday_s.astype(bool)).astype(int)
        )

        # ---------- 距假期天数 [-7, 7] ----------
        holiday_proximity = np.full(n, 999, dtype=int)
        for i in range(n):
            if is_holiday_arr[i]:
                holiday_proximity[i] = 0
                continue
            best = 999
            for offset in range(1, 8):
                if i + offset < n and is_holiday_arr[i + offset]:
                    best = -offset  # 节前：负数
                    break
                if i - offset >= 0 and is_holiday_arr[i - offset]:
                    best = offset   # 节后：正数
                    break
            holiday_proximity[i] = best
        df['holiday_proximity'] = holiday_proximity.clip(-7, 7)

        # ---------- 工作日/营业时段 ----------
        is_workday = (
            (is_weekend == 0) & (is_holiday_arr == 0) & (is_adjusted_workday == 0)
        ).astype(int)
        is_workday[is_adjusted_workday == 1] = 1
        df['is_business_hour'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17) & (is_workday == 1)
        ).astype(int)

        return df
    def _create_lag_and_rolling_features(self, df):
        """创建滞后和滚动统计特征"""
        lags = self.config['feature_engineering']['lag_periods']
        windows = self.config['feature_engineering']['rolling_windows']
        
        for lag in lags:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
        
        # 【重要修复】：所有滚动全部往后错位一格 (shift(1))，严防使用当期 price 预测当期 price
        past_price = df['price'].shift(1)
        for window in windows:
            df[f'price_roll_mean_{window}'] = past_price.rolling(window).mean()
            df[f'price_roll_std_{window}'] = past_price.rolling(window).std()
            df[f'price_roll_min_{window}'] = past_price.rolling(window).min()
            df[f'price_roll_max_{window}'] = past_price.rolling(window).max()
            df[f'price_roll_skew_{window}'] = past_price.rolling(window).skew()
            df[f'price_roll_rank_{window}'] = past_price.rolling(window).rank() / window
        
        df['price_diff_1'] = past_price.diff(1)
        df['price_diff_96'] = past_price.diff(96)
        df['price_pct_change_1'] = past_price.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['price_pct_change_96'] = past_price.pct_change(96).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['price_direction_1'] = np.sign(df['price_diff_1']).fillna(0)
        df['price_direction_96'] = np.sign(df['price_diff_96']).fillna(0)
        df['price_accel_1'] = df['price_diff_1'] - df['price_diff_1'].shift(1)
        df['price_accel_4'] = df['price_diff_1'] - df['price_diff_1'].shift(4)

        # 波动率特征：近期价格波动越大，电价不确定性越高
        df['price_volatility_4h'] = past_price.rolling(16).std()
        df['price_volatility_24h'] = past_price.rolling(96).std()
        df['price_range_4h'] = past_price.rolling(16).max() - past_price.rolling(16).min()
        df['price_range_24h'] = past_price.rolling(96).max() - past_price.rolling(96).min()

        # 日内价格剖面偏离：当前价格 vs 该时段历史均价
        if 'period_of_day' in df.columns:
            hourly_mean = df.groupby('period_of_day')['price'].transform('mean')
            df['price_vs_tod_mean'] = past_price - hourly_mean.shift(1)

        # 反转风险：当前价格距离近期高/低点有多远
        high_96 = past_price.rolling(96).max()
        low_96 = past_price.rolling(96).min()
        df['dist_to_96high'] = (high_96 - past_price).clip(lower=0)
        df['dist_to_96low'] = (past_price - low_96).clip(lower=0)
        df['reversal_risk'] = df['dist_to_96high'] / (high_96 - low_96).clip(lower=0.01)

        return df

    def _create_fourier_features(self, df):
        """创建傅里叶变换特征"""
        periods = self.config['feature_engineering']['fourier_periods']
        # 【重要修复】：使用固定参照日，防止截断数据推断时发生相位漂移
        ref_time = pd.Timestamp("2020-01-01 00:00:00")
        df['hour_index'] = (df['timestamp'] - ref_time).dt.total_seconds() / 3600
        
        for period in periods:
            for n in range(1, 4):
                df[f'fourier_sin_{period}_{n}'] = np.sin(2 * np.pi * n * df['hour_index'] / period)
                df[f'fourier_cos_{period}_{n}'] = np.cos(2 * np.pi * n * df['hour_index'] / period)
        
        df.drop('hour_index', axis=1, inplace=True)
        return df
    
    def _create_target_encoding(self, df):
        """创建目标编码特征"""
        categorical_cols = ['hour', 'day_of_week', 'month', 'season',
                           'price_period', 'holiday_type']

        for col in categorical_cols:
            if col in df.columns:
                global_mean = df['price'].mean()
                df[f'{col}_target_encoded'] = (
                    df.groupby(col)['price']
                    .transform(lambda x: x.shift(1).expanding().mean())
                ).fillna(global_mean)

                counts = df[col].value_counts()
                df[f'{col}_count'] = df[col].map(counts)

        return df
    
    def _create_interaction_features(self, df):
        """创建交叉特征"""
        # 修改依据：因为配置由 24 换成了单日 96 步，所以交互特征必须对齐 lag_96
        numerical_cols = ['hour', 'day_of_week', 'month', 'price_lag_1', 'price_lag_96']
        for i, col1 in enumerate(numerical_cols):
            if col1 in df.columns:
                for col2 in numerical_cols[i+1:]:
                    if col2 in df.columns:
                        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        den = df[col2].astype(float).abs().replace(0, np.nan).clip(lower=1e-8)
                        df[f'{col1}_div_{col2}'] = (df[col1].astype(float) / den).replace([np.inf, -np.inf], np.nan).fillna(0)
        # 天气-时间交互特征
        if 'temp_air' in df.columns:
            if 'hour' in df.columns:
                df['temp_x_hour'] = df['temp_air'] * df['hour']
            if 'season' in df.columns:
                df['temp_x_season'] = df['temp_air'] * df['season']
            if 'is_weekend' in df.columns:
                df['temp_x_weekend'] = df['temp_air'] * df['is_weekend']
        # 日前电价交互特征
        if 'dayahead_price' in df.columns:
            if 'hour' in df.columns:
                df['da_x_hour'] = df['dayahead_price'] * df['hour']
            if 'is_weekend' in df.columns:
                df['da_x_weekend'] = df['dayahead_price'] * df['is_weekend']
        # 负荷交互特征（使用 past_demand 避免数据泄露）
        if 'demand' in df.columns:
            past_demand = df['demand'].shift(1)
            if 'hour' in df.columns:
                df['demand_x_hour'] = past_demand * df['hour']
            if 'season' in df.columns:
                df['demand_x_season'] = past_demand * df['season']
            if 'is_weekend' in df.columns:
                df['demand_x_weekend'] = past_demand * df['is_weekend']
            if 'temp_air' in df.columns:
                df['demand_x_temp'] = past_demand * df['temp_air']
        # 地方出力交互特征（使用 past_local 避免数据泄露）
        if 'local_power_output' in df.columns:
            past_local = df['local_power_output'].shift(1)
            if 'hour' in df.columns:
                df['local_power_x_hour'] = past_local * df['hour']
            if 'season' in df.columns:
                df['local_power_x_season'] = past_local * df['season']
            if 'is_weekend' in df.columns:
                df['local_power_x_weekend'] = past_local * df['is_weekend']
            if 'temp_air' in df.columns:
                df['local_power_x_temp'] = past_local * df['temp_air']
        # 节假日交互特征：节假日的峰谷效应与工作日完全不同
        if 'is_holiday' in df.columns:
            if 'hour' in df.columns:
                df['holiday_x_hour'] = df['is_holiday'] * df['hour']
            if 'is_sharp_peak' in df.columns:
                df['holiday_x_peak'] = df['is_holiday'] * df['is_sharp_peak']
        # 制冷/供暖季 + 尖峰时段 = 极端电价风险
        if 'is_cooling_season' in df.columns and 'is_sharp_peak' in df.columns:
            df['cooling_x_sharp'] = df['is_cooling_season'] * df['is_sharp_peak']
        if 'is_heating_season' in df.columns and 'is_sharp_peak' in df.columns:
            df['heating_x_sharp'] = df['is_heating_season'] * df['is_sharp_peak']
        return df

    def _create_technical_indicators(self, df):
        """创建技术指标特征"""
        # 【重要修复】：一切指标必须基于 past_price 结算
        past_price = df['price'].shift(1)

        # 复用 _create_lag_and_rolling_features 已算好的滚动统计，避免重复计算
        if 'price_roll_mean_12' in df.columns:
            df['sma_12'] = df['price_roll_mean_12']
        else:
            df['sma_12'] = past_price.rolling(12).mean()

        if 'price_roll_mean_24' in df.columns:
            rolling_mean = df['price_roll_mean_24']
            rolling_std = df['price_roll_std_24']
            df['sma_24'] = rolling_mean
        else:
            rolling_mean = past_price.rolling(24).mean()
            rolling_std = past_price.rolling(24).std().replace(0, np.nan)
            df['sma_24'] = rolling_mean

        df['ema_12'] = past_price.ewm(span=12).mean()
        df['ema_24'] = past_price.ewm(span=24).mean()

        df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
        df['bollinger_lower'] = rolling_mean - (rolling_std * 2)
        df['bollinger_ratio'] = ((past_price - rolling_mean) / (2 * rolling_std)).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        delta = past_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)  # gain=0 且 loss=0 时价格无变化，RSI=50（中性）
        return df

    def _create_anomaly_features(self, df):
        """创建异常检测特征"""
        past_price = df['price'].shift(1)

        # 复用 _create_lag_and_rolling_features 已算好的滚动统计
        if 'price_roll_mean_24' in df.columns:
            rolling_mean = df['price_roll_mean_24']
            rolling_std = df['price_roll_std_24']
        else:
            rolling_mean = past_price.rolling(24).mean()
            rolling_std = past_price.rolling(24).std()

        df['price_zscore'] = (past_price - rolling_mean) / rolling_std
        df['is_anomaly_zscore'] = (abs(df['price_zscore']) > 3).astype(int)
        
        Q1 = past_price.rolling(168).quantile(0.25)
        Q3 = past_price.rolling(168).quantile(0.75)
        IQR = Q3 - Q1
        df['is_anomaly_iqr'] = ((past_price < (Q1 - 1.5 * IQR)) | (past_price > (Q3 + 1.5 * IQR))).astype(int)
        return df

    def _create_weather_features(self, df):
        """从天气数据列创建天气相关特征"""
        weather_lags = self.config['feature_engineering'].get('weather_lag_periods', [1, 4, 96])
        weather_windows = self.config['feature_engineering'].get('weather_rolling_windows', [4, 24, 96])

        # 直接可用的天气变量（可能已通过merge加入）
        weather_vars = [
            'temp_air', 'wind_speed', 'wind_direction', 'pressure',
            'ghi', 'dni', 'dhi', 'dew_point', 'et0', 'vpd'
        ]
        available_vars = [v for v in weather_vars if v in df.columns]

        if not available_vars:
            return df

        # 温度相关衍生特征
        if 'temp_air' in df.columns:
            # 温度变化率（温度剧烈变化 → 负荷突变 → 电价波动）
            df['temp_change_rate'] = df['temp_air'].diff()
            df['temp_ramp_1h'] = df['temp_air'] - df['temp_air'].shift(4)
            df['temp_ramp_4h'] = df['temp_air'] - df['temp_air'].shift(16)

        # 辐射相关衍生特征
        if 'ghi' in df.columns:
            # 辐射与时段交互（白天才有辐射）
            if 'hour' in df.columns:
                df['ghi_is_day'] = (df['ghi'] > 50).astype(int)
                df['ghi_hour_interact'] = df['ghi'] * df['hour']
            # 辐射利用率（与理论最大值的比率）
            df['ghi_ratio'] = df['ghi'].clip(lower=0) / 1000  # 标准化到[0,1]区间

        # 风速衍生特征
        if 'wind_speed' in df.columns:
            df['wind_power_potential'] = df['wind_speed'] ** 3  # 风能与风速的三次方成正比

        # 1. 天气变量的滞后特征
        for var in available_vars:
            for lag in weather_lags:
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)

        # 2. 天气变量的滚动统计
        for var in available_vars:
            for window in weather_windows:
                df[f'{var}_roll_mean_{window}'] = df[var].shift(1).rolling(window).mean()
                df[f'{var}_roll_std_{window}'] = df[var].shift(1).rolling(window).std()

        # 3. 天气-价格交叉特征
        past_price = df['price'].shift(1)
        if 'temp_air' in df.columns:
            # 温度与价格的交互（高温时段电价通常更高）
            df['price_x_temp'] = past_price * df['temp_air']
            # 极端温度标记（高温/低温可能推高电价）
            temp_median = df['temp_air'].median()
            temp_std = df['temp_air'].std()
            df['is_extreme_temp'] = (
                (abs(df['temp_air'] - temp_median) > 2 * temp_std)
            ).astype(int)

        if 'ghi' in df.columns:
            # 低辐射+高电价 = 光伏出力不足时段的稀缺电价
            df['price_div_ghi'] = (
                past_price / df['ghi'].clip(lower=1)
            ).clip(-1e6, 1e6).fillna(0)

        if 'wind_speed' in df.columns:
            df['price_x_wind'] = past_price * df['wind_speed']

        # 4. 综合天气指标
        if all(v in df.columns for v in ['temp_air', 'wind_speed']):
            # 舒适度指数（简化版）
            df['comfort_index'] = (
                df['temp_air'] * 0.4
                + df['wind_speed'] * 0.3
            )
            # 制冷/供暖需求指标
            df['cooling_degree'] = (df['temp_air'] - 26).clip(lower=0)   # >26°C需要制冷
            df['heating_degree'] = (18 - df['temp_air']).clip(lower=0)   # <18°C需要供暖

        return df

    def _create_dayahead_features(self, df):
        """创建日前电价特征（精简版：仅保留最强信号）

        日前电价是市场出清价格，是实时电价最强的单一预测信号。
        特征设计原则：少而精，避免引入噪声。
        """
        if 'dayahead_price' not in df.columns:
            return df

        past_price = df['price'].shift(1)

        # 1. 日前电价与实时价格的价差（市场预期 vs 实际）
        df['da_vs_price'] = df['dayahead_price'] - past_price

        # 2. 日前电价自身滞后（昨日同期日前电价）
        df['da_price_lag_96'] = df['dayahead_price'].shift(96)

        # 3. 日前电价趋势（日前价格的日间变化）
        df['da_price_trend_96'] = df['dayahead_price'] - df['dayahead_price'].shift(96)

        # 4. 日前vs实时价差方向（正价差=实时可能高于日前）
        df['da_premium_signal'] = (df['dayahead_price'] > past_price).astype(int)

        # Day-ahead slope and spread dynamics are stronger signals for direction
        # than the absolute day-ahead level alone.
        df['da_diff_1'] = df['dayahead_price'] - df['dayahead_price'].shift(1)
        df['da_diff_4'] = df['dayahead_price'] - df['dayahead_price'].shift(4)
        df['da_diff_96'] = df['dayahead_price'] - df['dayahead_price'].shift(96)
        df['da_direction_1'] = np.sign(df['da_diff_1']).fillna(0)
        df['da_direction_4'] = np.sign(df['da_diff_4']).fillna(0)
        df['da_spread_lag1'] = past_price - df['dayahead_price'].shift(1)
        df['da_spread_change_1'] = df['da_spread_lag1'] - df['da_spread_lag1'].shift(1)
        df['da_spread_change_4'] = df['da_spread_lag1'] - df['da_spread_lag1'].shift(4)
        if 'price_diff_1' in df.columns:
            df['price_diff_x_da_diff_1'] = df['price_diff_1'] * df['da_diff_1']
            df['price_dir_match_da_1'] = (
                np.sign(df['price_diff_1']).fillna(0)
                == np.sign(df['da_diff_1']).fillna(0)
            ).astype(int)

        return df

    def _create_demand_features(self, df):
        """创建负荷相关特征（全省实时总有功负荷）

        所有负荷特征基于 past_demand（shift(1)），避免使用当期负荷造成数据泄露。
        """
        if 'demand' not in df.columns:
            return df

        demand_lags = self.config['feature_engineering'].get('lag_periods', [1, 2, 4, 96, 192, 672])
        demand_windows = self.config['feature_engineering'].get('rolling_windows', [4, 12, 96, 672])

        past_demand = df['demand'].shift(1)

        # 1. 负荷滞后特征
        for lag in demand_lags:
            df[f'demand_lag_{lag}'] = df['demand'].shift(lag)

        # 2. 负荷滚动统计（基于 past_demand）
        for window in demand_windows:
            df[f'demand_roll_mean_{window}'] = past_demand.rolling(window).mean()
            df[f'demand_roll_std_{window}'] = past_demand.rolling(window).std()
            df[f'demand_roll_max_{window}'] = past_demand.rolling(window).max()
            df[f'demand_roll_min_{window}'] = past_demand.rolling(window).min()

        # 3. 负荷变化趋势
        df['demand_trend_4'] = past_demand - past_demand.shift(4)
        df['demand_trend_96'] = past_demand - past_demand.shift(96)
        df['demand_pct_change_1'] = past_demand.pct_change(1).replace([np.inf, -np.inf], 0).clip(-2, 2)
        df['demand_pct_change_96'] = past_demand.pct_change(96).replace([np.inf, -np.inf], 0).clip(-2, 2)
        df['demand_accel_1'] = df['demand_pct_change_1'] - df['demand_pct_change_1'].shift(1)
        df['demand_accel_4'] = df['demand_pct_change_1'] - df['demand_pct_change_1'].shift(4)
        df['demand_direction_1'] = np.sign(df['demand_pct_change_1']).fillna(0)

        # 4. 负荷率（当前负荷相对近期峰值的占比）
        peak_96 = past_demand.rolling(96).max()
        df['load_factor'] = (past_demand / peak_96.clip(lower=1)).clip(0, 2)

        # 5. 负荷日内模式偏差
        if 'period_of_day' in df.columns:
            demand_tod_mean = df.groupby('period_of_day')['demand'].transform('mean')
            df['demand_vs_tod_mean'] = past_demand - demand_tod_mean.shift(1)

        # 6. 价格-负荷交叉特征（使用 past_demand 和 past_price）
        past_price = df['price'].shift(1)
        df['price_x_demand'] = past_price * past_demand
        df['price_per_mw'] = (past_price / past_demand.clip(lower=1)).clip(-1, 1)

        # 7. 供需紧张信号
        df['demand_surge'] = (past_demand > past_demand.rolling(96).mean() * 1.1).astype(int)
        df['demand_drop'] = (past_demand < past_demand.rolling(96).mean() * 0.9).astype(int)

        # 8. 负荷爬坡率（需求变化速度，快速变化时段电价更不稳定）
        df['demand_ramp_1h'] = past_demand - past_demand.shift(4)
        df['demand_ramp_4h'] = past_demand - past_demand.shift(16)
        df['demand_volatility_24h'] = past_demand.rolling(96).std()

        # 9. 负荷-天气交叉信号（高温+高负荷 = 空调负荷大 = 推高电价）
        if 'temp_air' in df.columns:
            past_temp = df['temp_air'].shift(1)
            df['demand_x_temp_high'] = (past_demand * (past_temp > 30).astype(int))
            df['demand_x_temp_low'] = (past_demand * (past_temp < 10).astype(int))

        # 10. 地方能源出力特征（local_power_output）
        if 'local_power_output' in df.columns:
            local = df['local_power_output']
            past_local = local.shift(1)

            # 地方出力滞后
            for lag in demand_lags:
                df[f'local_power_lag_{lag}'] = local.shift(lag)

            # 地方出力滚动统计
            for window in demand_windows:
                df[f'local_power_roll_mean_{window}'] = past_local.rolling(window).mean()
                df[f'local_power_roll_std_{window}'] = past_local.rolling(window).std()
                df[f'local_power_roll_max_{window}'] = past_local.rolling(window).max()
                df[f'local_power_roll_min_{window}'] = past_local.rolling(window).min()

            # 地方出力变化趋势
            df['local_power_trend_4'] = past_local - past_local.shift(4)
            df['local_power_trend_96'] = past_local - past_local.shift(96)
            df['local_power_pct_change_1'] = past_local.pct_change(1).replace([np.inf, -np.inf], 0).clip(-2, 2)
            df['local_power_pct_change_96'] = past_local.pct_change(96).replace([np.inf, -np.inf], 0).clip(-2, 2)

            # 地方出力占比（地方出力 / 全省负荷）
            df['local_power_ratio'] = (past_local / past_demand.clip(lower=1)).clip(0, 2)

            # 价格-地方出力交叉
            past_price = df['price'].shift(1)
            df['price_x_local_power'] = past_price * past_local
            df['price_per_local_mw'] = (past_price / past_local.clip(lower=1)).clip(-1, 1)

            # 地方出力骤变信号
            df['local_power_surge'] = (past_local > past_local.rolling(96).mean() * 1.15).astype(int)
            df['local_power_drop'] = (past_local < past_local.rolling(96).mean() * 0.85).astype(int)

            # 净负荷 = 全省负荷 - 地方出力（系统需要从主网购买的量）
            df['net_demand'] = past_demand - past_local
            df['net_demand_ratio'] = ((past_demand - past_local) / past_demand.clip(lower=1)).clip(0, 2)

        return df

    def _prediction_cfg(self):
        return self.config.get('prediction', {})

    def _direction_deadband(self):
        return float(self._prediction_cfg().get('direction_deadband', 0.005))

    def _baseline_col(self):
        return self._prediction_cfg().get('baseline_col', 'price_lag_1')

    def _baseline_values(self, X):
        baseline_col = getattr(self, 'target_baseline_col_', self._baseline_col())
        if not hasattr(X, 'columns') or baseline_col not in X.columns:
            return None
        baseline = pd.to_numeric(X[baseline_col], errors='coerce').to_numpy(dtype=float)
        if np.isnan(baseline).any():
            median = np.nanmedian(baseline)
            if np.isnan(median):
                median = 0.0
            baseline = np.nan_to_num(baseline, nan=median, posinf=median, neginf=median)
        return baseline

    def _prepare_regression_target(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        mode = self._prediction_cfg().get('target_mode', 'delta')
        baseline_col = self._baseline_col()
        baseline = self._baseline_values(X)

        if mode == 'delta' and baseline is not None:
            self.target_mode_ = 'delta'
            self.target_baseline_col_ = baseline_col
            return y_arr - baseline

        self.target_mode_ = 'price'
        self.target_baseline_col_ = baseline_col
        return y_arr

    def _restore_regression_target(self, X, pred):
        pred_arr = np.asarray(pred, dtype=float)
        mode = getattr(self, 'target_mode_', self._prediction_cfg().get('target_mode', 'price'))
        if mode != 'delta':
            return pred_arr

        baseline = self._baseline_values(X)
        if baseline is None:
            return pred_arr
        return pred_arr + baseline

    @staticmethod
    def _direction_labels_from_delta(delta, deadband):
        delta = np.asarray(delta, dtype=float)
        return np.where(delta > deadband, 1, np.where(delta < -deadband, -1, 0))

    def direction_accuracy(self, y_true, y_pred, baseline=None, deadband=None):
        """Direction accuracy against the previous observed price baseline."""
        if deadband is None:
            deadband = self._direction_deadband()

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        if baseline is None:
            if len(y_true) <= 1:
                return np.nan
            true_dir = self._direction_labels_from_delta(np.diff(y_true), deadband)
            pred_dir = self._direction_labels_from_delta(np.diff(y_pred), deadband)
        else:
            baseline = np.asarray(baseline, dtype=float)
            true_dir = self._direction_labels_from_delta(y_true - baseline, deadband)
            pred_dir = self._direction_labels_from_delta(y_pred - baseline, deadband)

        valid = np.isfinite(true_dir) & np.isfinite(pred_dir)
        return float(np.mean(true_dir[valid] == pred_dir[valid])) if valid.any() else np.nan

    def _train_direction_classifier(self, X_train, y_train, X_val=None, y_val=None):
        baseline = self._baseline_values(X_train)
        if baseline is None:
            return

        deadband = self._direction_deadband()
        labels = self._direction_labels_from_delta(np.asarray(y_train, dtype=float) - baseline, deadband)
        if len(np.unique(labels)) < 2:
            print("[WARN] 方向分类标签不足，跳过方向分类器")
            return

        self.direction_class_order_ = np.array([-1, 0, 1], dtype=int)

        def val_score(model):
            if X_val is None or y_val is None:
                return np.nan
            val_baseline = self._baseline_values(X_val)
            if val_baseline is None:
                return np.nan
            val_labels = self._direction_labels_from_delta(
                np.asarray(y_val, dtype=float) - val_baseline, deadband
            )
            val_pred = model.predict(X_val)
            return float(np.mean(val_pred == val_labels))

        direction_models = {}
        model_scores = {}

        try:
            import lightgbm as lgb

            clf_main = lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.7,
                class_weight='balanced',
                random_state=42,
                verbosity=-1,
            )
            clf_main.fit(X_train, labels)
            direction_models['lgb_main'] = clf_main
            model_scores['lgb_main'] = val_score(clf_main)

            clf_stable = lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.02,
                num_leaves=15,
                min_child_samples=80,
                subsample=0.75,
                colsample_bytree=0.55,
                class_weight='balanced',
                random_state=43,
                verbosity=-1,
            )
            clf_stable.fit(X_train, labels)
            direction_models['lgb_stable'] = clf_stable
            model_scores['lgb_stable'] = val_score(clf_stable)
        except ImportError:
            print("[WARN] LightGBM未安装，跳过LightGBM方向分类器")
        except Exception as e:
            print(f"[WARN] LightGBM方向分类器训练失败: {e}")

        if self._prediction_cfg().get('use_direction_ensemble', True):
            try:
                from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler

                rf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight='balanced_subsample',
                    random_state=44,
                    n_jobs=-1,
                )
                rf.fit(X_train, labels)
                direction_models['rf'] = rf
                model_scores['rf'] = val_score(rf)

                et = ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=20,
                    class_weight='balanced',
                    random_state=45,
                    n_jobs=-1,
                )
                et.fit(X_train, labels)
                direction_models['extra_trees'] = et
                model_scores['extra_trees'] = val_score(et)

                lr = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', LogisticRegression(
                        C=0.5,
                        max_iter=1000,
                        class_weight='balanced',
                        multi_class='auto',
                        random_state=46,
                    ))
                ])
                lr.fit(X_train, labels)
                direction_models['logistic'] = lr
                model_scores['logistic'] = val_score(lr)
            except Exception as e:
                print(f"[WARN] 备用方向分类器训练失败: {e}")

        if not direction_models:
            print("[WARN] 未训练出可用方向分类器")
            return

        self.direction_classifiers_ = direction_models
        self.direction_classifier_ = next(iter(direction_models.values()))

        if X_val is not None and y_val is not None:
            scores_msg = {
                name: None if np.isnan(score) else round(score * 100, 1)
                for name, score in model_scores.items()
            }
            self._optimize_direction_classifier_weights(X_val, y_val)
            cls, _ = self._direction_classifier_prediction(X_val)
            val_baseline = self._baseline_values(X_val)
            val_labels = self._direction_labels_from_delta(
                np.asarray(y_val, dtype=float) - val_baseline, deadband
            )
            val_acc = np.mean(cls == val_labels) * 100 if cls is not None else np.nan
            print(f"[OK] 方向分类器集成训练完成 | 验证集方向准确率: {val_acc:.1f}%, 单模型={scores_msg}")
        else:
            print(f"[OK] 方向分类器集成训练完成 | 模型数={len(direction_models)}")

    def _proba_to_class_order(self, model, proba):
        if hasattr(model, 'classes_'):
            classes = np.asarray(model.classes_, dtype=int)
        elif hasattr(model, 'named_steps') and 'model' in model.named_steps:
            classes = np.asarray(model.named_steps['model'].classes_, dtype=int)
        else:
            classes = getattr(self, 'direction_class_order_', np.array([-1, 0, 1], dtype=int))
        class_order = getattr(self, 'direction_class_order_', np.array([-1, 0, 1], dtype=int))
        aligned = np.zeros((proba.shape[0], len(class_order)), dtype=float)
        for i, cls in enumerate(classes):
            matches = np.where(class_order == cls)[0]
            if len(matches):
                aligned[:, matches[0]] = proba[:, i]
        row_sum = aligned.sum(axis=1, keepdims=True)
        aligned = np.divide(aligned, row_sum, out=np.ones_like(aligned) / len(class_order), where=row_sum > 0)
        return aligned

    def _direction_ensemble_proba(self, X):
        models = getattr(self, 'direction_classifiers_', None)
        if not models:
            if hasattr(self, 'direction_classifier_') and self.direction_classifier_ is not None:
                proba = self.direction_classifier_.predict_proba(X)
                return self._proba_to_class_order(self.direction_classifier_, proba)
            return None

        weights = getattr(self, 'direction_classifier_weights_', None) or {}
        usable = []
        for name, model in models.items():
            try:
                proba = self._proba_to_class_order(model, model.predict_proba(X))
                usable.append((name, proba))
            except Exception as e:
                print(f"[WARN] 方向分类器 {name} 预测失败: {e}")

        if not usable:
            return None

        total_weight = sum(weights.get(name, 0.0) for name, _ in usable)
        if total_weight <= 0:
            total_weight = len(usable)
            weights = {name: 1.0 for name, _ in usable}

        out = np.zeros_like(usable[0][1])
        for name, proba in usable:
            out += proba * (weights.get(name, 0.0) / total_weight)
        return out

    def _optimize_direction_classifier_weights(self, X_val, y_val):
        models = getattr(self, 'direction_classifiers_', None)
        if not models or X_val is None or y_val is None:
            return

        val_baseline = self._baseline_values(X_val)
        if val_baseline is None:
            return

        labels = self._direction_labels_from_delta(
            np.asarray(y_val, dtype=float) - val_baseline,
            self._direction_deadband()
        )

        names, probas = [], []
        for name, model in models.items():
            try:
                names.append(name)
                probas.append(self._proba_to_class_order(model, model.predict_proba(X_val)))
            except Exception as e:
                print(f"[WARN] 方向分类器权重优化跳过 {name}: {e}")

        if not probas:
            return

        class_order = getattr(self, 'direction_class_order_', np.array([-1, 0, 1], dtype=int))

        def score(weights):
            combined = np.zeros_like(probas[0])
            for w, proba in zip(weights, probas):
                combined += proba * w
            pred = class_order[np.argmax(combined, axis=1)]
            return float(np.mean(pred == labels))

        candidates = []
        n = len(names)
        candidates.append(np.ones(n) / n)
        for i in range(n):
            one_hot = np.zeros(n)
            one_hot[i] = 1.0
            candidates.append(one_hot)

        rng = np.random.default_rng(43)
        candidates.extend(rng.dirichlet(np.ones(n), size=300))

        best_acc, best_weights = -1.0, None
        for weights in candidates:
            acc = score(weights)
            if acc > best_acc:
                best_acc, best_weights = acc, weights

        self.direction_classifier_weights_ = {
            name: float(weight) for name, weight in zip(names, best_weights)
        }
        print(
            "[OK] 方向分类器权重已优化 | "
            f"验证集方向={best_acc * 100:.1f}%, 权重={self.direction_classifier_weights_}"
        )

    def _direction_classifier_prediction(self, X):
        proba = self._direction_ensemble_proba(X)
        if proba is None:
            return None, None

        try:
            classes = getattr(self, 'direction_class_order_', np.array([-1, 0, 1], dtype=int))
            best_idx = np.argmax(proba, axis=1)
            cls = classes[best_idx].astype(int)
            conf = proba[np.arange(len(best_idx)), best_idx]
            return cls, conf
        except Exception as e:
            print(f"[WARN] 方向修正失败: {e}")
            return None, None

    def _correct_delta_with_direction(self, delta, cls, conf, threshold):
        deadband = self._direction_deadband()
        pred_dir = self._direction_labels_from_delta(delta, deadband)

        corrected = np.asarray(delta, dtype=float).copy()
        strong_move = (conf >= threshold) & (cls != 0) & (pred_dir != cls)
        if strong_move.any():
            corrected[strong_move] = cls[strong_move] * np.maximum(
                np.abs(corrected[strong_move]), deadband
            )

        strong_flat = (conf >= threshold) & (cls == 0)
        if strong_flat.any():
            corrected[strong_flat] = np.clip(corrected[strong_flat], -deadband, deadband)

        return corrected

    def _apply_direction_correction(self, X, price_pred):
        cfg = self._prediction_cfg()
        if not cfg.get('direction_correction', True):
            return price_pred

        baseline = self._baseline_values(X)
        if baseline is None:
            return price_pred

        cls, conf = self._direction_classifier_prediction(X)
        if cls is None or conf is None:
            return price_pred

        min_proba = float(getattr(
            self,
            'direction_correction_min_proba_',
            cfg.get('direction_correction_min_proba', 0.55)
        ))
        delta = np.asarray(price_pred, dtype=float) - baseline
        corrected = self._correct_delta_with_direction(delta, cls, conf, min_proba)
        return baseline + corrected

    def _optimize_direction_correction(self, X_val, y_val, price_pred):
        if X_val is None or y_val is None or price_pred is None:
            return

        baseline = self._baseline_values(X_val)
        if baseline is None:
            return

        cls, conf = self._direction_classifier_prediction(X_val)
        if cls is None or conf is None:
            return

        y_val_arr = np.asarray(y_val, dtype=float)
        base_rmse = np.sqrt(np.mean((y_val_arr - price_pred) ** 2))
        base_dir = self.direction_accuracy(y_val_arr, price_pred, baseline=baseline)
        if np.isnan(base_dir):
            base_dir = 0.0

        max_rmse_increase = float(self._prediction_cfg().get('direction_max_rmse_increase', 0.03))
        best = {
            'threshold': float(self._prediction_cfg().get('direction_correction_min_proba', 0.55)),
            'rmse': base_rmse,
            'dir_acc': base_dir,
            'score': base_dir,
        }

        delta = np.asarray(price_pred, dtype=float) - baseline
        for threshold in np.arange(0.35, 0.801, 0.025):
            corrected = baseline + self._correct_delta_with_direction(delta, cls, conf, threshold)
            rmse = np.sqrt(np.mean((y_val_arr - corrected) ** 2))
            dir_acc = self.direction_accuracy(y_val_arr, corrected, baseline=baseline)
            if np.isnan(dir_acc):
                continue

            rmse_ratio = rmse / max(base_rmse, 1e-8)
            if rmse_ratio > 1 + max_rmse_increase:
                continue

            # Direction is primary here; RMSE is the tie breaker.
            score = dir_acc - 0.02 * max(0.0, rmse_ratio - 1.0)
            if (score > best['score']) or (
                np.isclose(score, best['score']) and rmse < best['rmse']
            ):
                best = {
                    'threshold': float(threshold),
                    'rmse': float(rmse),
                    'dir_acc': float(dir_acc),
                    'score': float(score),
                }

        self.direction_correction_min_proba_ = best['threshold']
        print(
            "[OK] 方向修正阈值已优化 | "
            f"threshold={best['threshold']:.3f}, "
            f"验证集方向={best['dir_acc'] * 100:.1f}%, RMSE={best['rmse']:.4f}"
        )

    def _optimize_ensemble_weights(self, X_val, y_val):
        if not self.models or X_val is None or y_val is None:
            return None

        names, raw_preds = [], []
        for name, model in self.models.items():
            try:
                names.append(name)
                raw_preds.append(np.asarray(model.predict(X_val), dtype=float))
            except Exception as e:
                print(f"[WARN] 验证集权重优化跳过 {name}: {e}")

        if not raw_preds:
            return None

        raw_preds = np.vstack(raw_preds)
        y_val_arr = np.asarray(y_val, dtype=float)
        baseline = self._baseline_values(X_val)
        scale = np.nanstd(y_val_arr)
        scale = scale if scale > 1e-8 else 1.0
        direction_weight = float(self._prediction_cfg().get('direction_weight', 0.25))

        def score(weights):
            raw = np.average(raw_preds, axis=0, weights=weights)
            price_pred = self._restore_regression_target(X_val, raw)
            price_pred = self._apply_direction_correction(X_val, price_pred)
            rmse = np.sqrt(np.mean((y_val_arr - price_pred) ** 2))
            dir_acc = self.direction_accuracy(y_val_arr, price_pred, baseline=baseline)
            if np.isnan(dir_acc):
                dir_acc = 0.0
            return rmse / scale - direction_weight * dir_acc, rmse, dir_acc

        candidates = []
        n = len(names)
        candidates.append(np.ones(n) / n)

        default_weights = {
            'lightgbm': 0.30, 'xgboost': 0.25, 'catboost': 0.25,
            'ridge': 0.10, 'random_forest': 0.10
        }
        default = np.array([default_weights.get(name, 1.0 / n) for name in names], dtype=float)
        candidates.append(default / default.sum())

        model_rmse = []
        for raw in raw_preds:
            price_pred = self._restore_regression_target(X_val, raw)
            model_rmse.append(np.sqrt(np.mean((y_val_arr - price_pred) ** 2)))
        inv = 1 / np.maximum(model_rmse, 1e-8)
        candidates.append(inv / inv.sum())

        for i in range(n):
            one_hot = np.zeros(n)
            one_hot[i] = 1.0
            candidates.append(one_hot)

        rng = np.random.default_rng(42)
        candidates.extend(rng.dirichlet(np.ones(n), size=300))

        best = None
        for weights in candidates:
            current = score(weights)
            if best is None or current[0] < best[0]:
                best = (*current, weights)

        _, rmse, dir_acc, weights = best
        weight_map = {name: float(w) for name, w in zip(names, weights)}
        print(
            "[OK] 集成权重已按验证集优化 | "
            f"RMSE={rmse:.4f}, 方向准确率={dir_acc * 100:.1f}%, 权重={weight_map}"
        )
        return weight_map

    def hyperparameter_optimization(self, X_train, y_train, X_val, y_val):
        """超参数优化"""
        print("开始超参数优化...")
        
        try:
            import optuna
        except ImportError:
            print("Optuna未安装，跳过超参数优化")
            return self.config['model_params']['lightgbm']
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': 42,
                'verbosity': -1
            }
            
            try:
                import lightgbm as lgb
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=300,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                
                return rmse
                
            except Exception as e:
                print(f"优化过程中出现错误: {e}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective, 
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization'].get('timeout', 3600)
        )
        
        print(f"最佳RMSE: {study.best_value:.6f}")
        print(f"最佳参数: {study.best_params}")
        
        # 更新配置
        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 42,
            'verbosity': -1
        })
        
        return best_params
    
    def train_ensemble_models(self, X_train, y_train, X_val=None, y_val=None,
                              train_direction=True, tune_ensemble=True):
        """训练集成模型"""
        print("开始训练集成模型...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        models = {}
        y_train_original = np.asarray(y_train, dtype=float)
        y_val_original = np.asarray(y_val, dtype=float) if y_val is not None else None
        y_train_model = self._prepare_regression_target(X_train, y_train_original)
        val_baseline = self._baseline_values(X_val) if X_val is not None else None
        y_val_model = (
            np.asarray(y_val_original, dtype=float) - val_baseline
            if y_val_original is not None and val_baseline is not None
            and getattr(self, 'target_mode_', 'price') == 'delta'
            else y_val_original
        )

        if getattr(self, 'target_mode_', 'price') == 'delta':
            print(f"[目标] 使用残差目标: price - {self.target_baseline_col_}")

        # 用来计算验证误差并打印
        def print_val_score(model_name, model_obj, X_eval, y_eval):
            if X_eval is not None and y_eval is not None:
                preds = self._restore_regression_target(X_eval, model_obj.predict(X_eval))
                rmse = np.sqrt(mean_squared_error(y_eval, preds))
                mae = mean_absolute_error(y_eval, preds)
                baseline = self._baseline_values(X_eval)
                dir_acc = self.direction_accuracy(y_eval, preds, baseline=baseline)
                dir_msg = f", 方向准确率: {dir_acc * 100:.1f}%" if not np.isnan(dir_acc) else ""
                print(f"[OK] {model_name} 训练完成 | 验证集 RMSE: {rmse:.4f}, MAE: {mae:.4f}{dir_msg}")
            else:
                print(f"[OK] {model_name} 训练完成")
        
        # 1. LightGBM
        try:
            import lightgbm as lgb
            
            # 如果有验证集，使用优化过的参数
            if X_val is not None and y_val is not None:
                params = self.hyperparameter_optimization(X_train, y_train_model, X_val, y_val_model)
            else:
                params = self.config['model_params']['lightgbm']
            
            train_data = lgb.Dataset(X_train, label=y_train_model)
            if X_val is not None and y_val_model is not None:
                val_data = lgb.Dataset(X_val, label=y_val_model, reference=train_data)
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                )
            else:
                model = lgb.train(params, train_data, num_boost_round=500)
            
            models['lightgbm'] = model
            print_val_score("LightGBM", model, X_val, y_val)
            
        except ImportError:
            print("[WARN] LightGBM未安装")
        
        # 2. XGBoost
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
                objective='reg:pseudohubererror' 
            )
            model.fit(X_train, y_train_model)
            models['xgboost'] = model
            print_val_score('XGBoost', model, X_val, y_val)
            
        except ImportError:
            print("[WARN] XGBoost未安装")
        
        # 3. CatBoost
        try:
            import catboost as cb
            
            model = cb.CatBoostRegressor(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                random_seed=42,
                verbose=False
            )
            model.fit(X_train, y_train_model)
            models['catboost'] = model
            print_val_score('CatBoost', model, X_val, y_val)
            
        except ImportError:
            print("[WARN] CatBoost未安装")
        
        # 4. 线性模型
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=42))
        ])
        pipeline.fit(X_train, y_train_model)
        models['ridge'] = pipeline
        print_val_score('Ridge回归', pipeline, X_val, y_val)
        
        # 5. 随机森林
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_model)
        models['random_forest'] = model
        print_val_score('随机森林', model, X_val, y_val)
        
        self.models = models
        if train_direction:
            self._train_direction_classifier(X_train, y_train_original, X_val, y_val_original)

        if tune_ensemble and self._prediction_cfg().get('optimize_ensemble_weights', True):
            self.ensemble_weights_ = self._optimize_ensemble_weights(X_val, y_val_original)

        if train_direction and X_val is not None and y_val_original is not None:
            val_pred_raw = self.create_ensemble_prediction(
                X_val, method='weighted_average', apply_direction=False
            )
            self._optimize_direction_correction(X_val, y_val_original, val_pred_raw)
            if tune_ensemble and self._prediction_cfg().get('optimize_ensemble_weights', True):
                self.ensemble_weights_ = self._optimize_ensemble_weights(X_val, y_val_original)

        return models

    def train_two_stage_models(self, X_train, y_train, X_val=None, y_val=None):
        """训练两阶段模型：阶段1分类(尖峰/非尖峰) + 阶段2分组建模

        阶段1: LightGBM 二分类器，预测是否为尖峰时段
        阶段2: 对尖峰和非尖峰分别训练各自的集成模型
        """
        from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, mean_absolute_error

        cfg = self.config.get('two_stage', {})
        percentile = cfg.get('spike_percentile', 80)
        use_blend = cfg.get('use_blend', True)

        # ---- 尖峰定义 + 最优阈值搜索 ----
        threshold = np.percentile(y_train, percentile)
        self.spike_threshold_ = threshold
        y_train_binary = (y_train > threshold).astype(int)
        spike_ratio = y_train_binary.mean()
        print(f"\n[两阶段模型] 尖峰阈值 = {threshold:.4f} (P{percentile}), "
              f"尖峰占比 = {spike_ratio:.1%}")

        if X_val is not None and y_val is not None:
            y_val_binary = (y_val > threshold).astype(int)
        else:
            y_val_binary = None

        # 计算类别权重（反比于类别频率，增强尖峰识别）
        n_pos = y_train_binary.sum()
        n_neg = len(y_train_binary) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # ---- 阶段1: 尖峰分类器 ----
        print(f"[阶段1] 训练尖峰/非尖峰分类器 (LightGBM, scale_pos_weight={scale_pos_weight:.1f})...")
        import lightgbm as lgb

        clf_params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.03,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 30,
            'reg_alpha': 0.1,
            'reg_lambda': 0.5,
            'random_state': 42,
            'verbosity': -1,
            'scale_pos_weight': scale_pos_weight,
        }

        train_data = lgb.Dataset(X_train, label=y_train_binary)
        if X_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val_binary, reference=train_data)
            self.spike_classifier_ = lgb.train(
                clf_params, train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.spike_classifier_ = lgb.train(clf_params, train_data, num_boost_round=300)

        # 评估分类器 + 最优判定阈值搜索（优化F1而非默认0.5）
        if X_val is not None and y_val_binary is not None:
            from sklearn.metrics import f1_score
            val_proba = self.spike_classifier_.predict(X_val)

            # 搜索最优分类阈值
            best_thresh, best_f1 = 0.5, 0.0
            for t in np.arange(0.25, 0.75, 0.025):
                pred_bin = (val_proba > t).astype(int)
                f1 = f1_score(y_val_binary, pred_bin, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thresh = f1, t
            self.spike_class_threshold_ = best_thresh
            val_pred_binary = (val_proba > best_thresh).astype(int)
            auc = roc_auc_score(y_val_binary, val_proba)
            print(f"[阶段1] 验证集 AUC = {auc:.4f}, 最优分类阈值 = {best_thresh:.3f} (F1={best_f1:.3f})")
            print(classification_report(y_val_binary, val_pred_binary,
                                        target_names=['非尖峰', '尖峰'], zero_division=0))
        else:
            self.spike_class_threshold_ = 0.5

        # ---- 阶段2: 分组回归 ----
        # 尖峰组
        spike_mask_train = y_train > threshold
        X_spike, y_spike = X_train[spike_mask_train], y_train[spike_mask_train]

        # 非尖峰组
        X_normal, y_normal = X_train[~spike_mask_train], y_train[~spike_mask_train]

        print(f"\n[阶段2] 分组训练: 尖峰样本={len(y_spike)}, 非尖峰样本={len(y_normal)}")

        # 保存原模型集，训练分组模型
        original_models = self.models.copy() if self.models else {}

        # 尖峰模型组
        if len(y_spike) >= 30:
            print("[阶段2-尖峰] 训练尖峰专属模型...")
            self.models = {}  # 清空，用 train_ensemble_models 填充
            self.train_ensemble_models(X_spike, y_spike, train_direction=False, tune_ensemble=False)
            self.models_spike_ = self.models.copy()
        else:
            print("[阶段2-尖峰] 尖峰样本不足30条，使用全量模型作为尖峰模型")
            self.models_spike_ = original_models.copy()

        # 非尖峰模型组
        self.models = {}  # 清空
        self.train_ensemble_models(X_normal, y_normal, train_direction=False, tune_ensemble=False)
        self.models_normal_ = self.models.copy()

        # 恢复全量模型（保留给单阶段模式使用）
        self.models_full_ = original_models.copy()
        self.models = original_models.copy()
        self.two_stage_trained_ = True

        # ---- 在验证集上评估两阶段效果 ----
        if X_val is not None and y_val is not None:
            preds_two_stage = self._predict_two_stage(X_val, use_blend=use_blend)
            preds_original = self.create_ensemble_prediction(X_val, method='weighted_average')

            rmse_2s = np.sqrt(mean_squared_error(y_val, preds_two_stage))
            mae_2s = mean_absolute_error(y_val, preds_two_stage)
            rmse_orig = np.sqrt(mean_squared_error(y_val, preds_original))
            mae_orig = mean_absolute_error(y_val, preds_original)

            # 分尖峰/非尖峰统计
            spike_mask_val = y_val > threshold
            if spike_mask_val.sum() > 0:
                spike_rmse_2s = np.sqrt(mean_squared_error(y_val[spike_mask_val], preds_two_stage[spike_mask_val]))
                spike_rmse_orig = np.sqrt(mean_squared_error(y_val[spike_mask_val], preds_original[spike_mask_val]))
                print(f"[两阶段评估] 尖峰段 RMSE: 单阶段={spike_rmse_orig:.4f} -> 两阶段={spike_rmse_2s:.4f}")

            if (~spike_mask_val).sum() > 0:
                normal_rmse_2s = np.sqrt(mean_squared_error(y_val[~spike_mask_val], preds_two_stage[~spike_mask_val]))
                normal_rmse_orig = np.sqrt(mean_squared_error(y_val[~spike_mask_val], preds_original[~spike_mask_val]))
                print(f"[两阶段评估] 常规段 RMSE: 单阶段={normal_rmse_orig:.4f} -> 两阶段={normal_rmse_2s:.4f}")

            print(f"[两阶段评估] 整体 RMSE: 单阶段={rmse_orig:.4f} -> 两阶段={rmse_2s:.4f} | "
                  f"MAE: 单阶段={mae_orig:.4f} -> 两阶段={mae_2s:.4f}")

    def _predict_two_stage(self, X, use_blend=True):
        """两阶段预测：分类 -> 选择回归模型

        Args:
            X: 特征矩阵
            use_blend: True=用尖峰概率混合两组预测, False=硬分类
        """
        if not hasattr(self, 'spike_classifier_') or self.spike_classifier_ is None:
            return self.create_ensemble_prediction(X, method='weighted_average')

        X = self._align_prediction_features(X)

        spike_proba = self.spike_classifier_.predict(X)

        # 保存原模型引用后切换
        saved_models = self.models

        if use_blend:
            # 软混合：用尖峰概率加权两组预测
            self.models = self.models_spike_
            preds_spike = self.create_ensemble_prediction(X, method='weighted_average', apply_direction=False)
            self.models = self.models_normal_
            preds_normal = self.create_ensemble_prediction(X, method='weighted_average', apply_direction=False)
            self.models = saved_models
            blended = preds_spike * spike_proba + preds_normal * (1 - spike_proba)
            return self._apply_direction_correction(X, blended)
        else:
            # 硬分类：使用最优阈值
            thresh = getattr(self, 'spike_class_threshold_', 0.5)
            is_spike = spike_proba > thresh
            preds = np.zeros(len(X))
            self.models = self.models_spike_
            if is_spike.sum() > 0:
                preds[is_spike] = self.create_ensemble_prediction(
                    X.loc[is_spike] if hasattr(X, 'loc') else X[is_spike],
                    method='weighted_average',
                    apply_direction=False
                )
            self.models = self.models_normal_
            if (~is_spike).sum() > 0:
                preds[~is_spike] = self.create_ensemble_prediction(
                    X.loc[~is_spike] if hasattr(X, 'loc') else X[~is_spike],
                    method='weighted_average',
                    apply_direction=False
                )
            self.models = saved_models
            return self._apply_direction_correction(X, preds)

    def _align_prediction_features(self, X):
        """将预测输入的特征列对齐到训练时的特征列。

        补齐缺失列（填0），丢弃多余列。若未记录训练特征列则直接返回。
        """
        if not hasattr(self, 'training_feature_cols') or not self.training_feature_cols:
            return X
        if not hasattr(X, 'columns'):
            return X
        train_cols = self.training_feature_cols
        if len(train_cols) == 0:
            return X
        missing = [c for c in train_cols if c not in X.columns]
        extra = [c for c in X.columns if c not in train_cols]
        if missing or extra:
            if missing:
                for c in missing:
                    X[c] = 0
            if extra:
                X = X.drop(columns=extra)
            X = X[train_cols]
        return X

    def create_ensemble_prediction(self, X_test, method='weighted_average', apply_direction=True):
        """创建集成预测"""
        if not self.models:
            raise ValueError("请先训练模型")

        # 对齐到训练时的特征列
        X_test = self._align_prediction_features(X_test)

        # 生成基础预测
        base_predictions = {}
        for name, model in self.models.items():
            try:
                base_predictions[name] = model.predict(X_test)
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                continue

        if not base_predictions:
            raise ValueError("无可用模型产生预测")

        if method == 'simple_average':
            predictions = np.array(list(base_predictions.values()))
            raw_pred = np.mean(predictions, axis=0)

        elif method == 'weighted_average':
            # Prefer validation-optimized weights, then fall back to defaults.
            n_models = len(base_predictions)
            weights = {}
            saved_weights = getattr(self, 'ensemble_weights_', None)
            if saved_weights:
                for name in base_predictions:
                    weights[name] = saved_weights.get(name, 0.0)
                if sum(weights.values()) <= 0:
                    weights = {}

            if not weights:
                default_weights = {
                    'lightgbm': 0.30, 'xgboost': 0.25, 'catboost': 0.25,
                    'ridge': 0.10, 'random_forest': 0.10
                }
                base_w = 1.0 / n_models
                for name in base_predictions:
                    weights[name] = default_weights.get(name, base_w)

            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

            raw_pred = np.zeros(X_test.shape[0])
            for name, pred in base_predictions.items():
                raw_pred += pred * weights[name]

        else:  # stacking
            # Ridge元模型堆叠
            meta_features = np.column_stack(list(base_predictions.values()))
            from sklearn.linear_model import Ridge
            # 使用各子模型预测的均值作为伪标签训练元模型
            pseudo_target = np.mean(meta_features, axis=1)
            meta_model = Ridge(alpha=0.1)
            meta_model.fit(meta_features, pseudo_target)
            raw_pred = meta_model.predict(meta_features)

        price_pred = self._restore_regression_target(X_test, raw_pred)
        if apply_direction:
            return self._apply_direction_correction(X_test, price_pred)
        return price_pred
    
    def model_interpretation(self, X_test, y_test):
        """模型解释性分析"""
        print("开始模型解释性分析...")
        
        interpretation_results = {}
        
        # 1. 特征重要性分析
        if 'lightgbm' in self.models:
            model = self.models['lightgbm']
            feature_importance = dict(zip(X_test.columns, model.feature_importance()))
            interpretation_results['feature_importance'] = feature_importance
            
            # 显示Top 10重要特征
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            print("\nTop 10重要特征:")
            for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                print(f"{i:2d}. {feature:<30}: {importance:>8.2f}")
        
        # 2. SHAP值分析（如果有SHAP库）
        try:
            import shap
            
            if 'lightgbm' in self.models:
                explainer = shap.TreeExplainer(self.models['lightgbm'])
                shap_values = explainer.shap_values(X_test.iloc[:100])  # 只分析前100个样本
                
                interpretation_results['shap_values'] = shap_values
                print("[OK] SHAP分析完成")
                
                # 保存SHAP图（需要matplotlib）
                try:
                    import matplotlib.pyplot as plt
                    
                    # 特征重要性图
                    shap.summary_plot(shap_values, X_test.iloc[:100], 
                                     plot_type="bar", show=False)
                    plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # 详细SHAP图
                    shap.summary_plot(shap_values, X_test.iloc[:100], show=False)
                    plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print("[OK] SHAP图表已保存")
                    
                except ImportError:
                    print("[WARN] matplotlib未安装，跳过图表保存")
                
        except ImportError:
            print("[WARN] SHAP库未安装，跳过SHAP分析")
        
        # 3. 部分依赖图分析
        try:
            from sklearn.inspection import partial_dependence
            
            if 'random_forest' in self.models:
                model = self.models['random_forest']
                
                # 选择重要特征进行分析
                important_features = ['price_lag_1', 'price_lag_24', 'hour', 'day_of_week']
                available_features = [f for f in important_features if f in X_test.columns]
                
                if available_features:
                    pd_results = partial_dependence(
                        model, X_test, features=available_features[:2],  # 只分析前2个特征
                        kind='average'
                    )
                    interpretation_results['partial_dependence'] = pd_results
                    print("[OK] 部分依赖分析完成")
        
        except Exception as e:
            print(f"[WARN] 部分依赖分析失败: {e}")
        
        return interpretation_results
    
    def online_learning_update(self, X_new, y_new):
        """在线学习模型更新"""
        print("执行在线学习更新...")
        
        # 检查数据质量
        if len(X_new) < 24:  # 至少需要24小时数据
            print("[WARN] 数据量不足，跳过更新")
            return
        
        # 计算当前性能
        if 'lightgbm' in self.models and len(X_new) > 0:
            # 检查特征数量是否匹配
            model_features = self.models['lightgbm'].num_feature()
            input_features = X_new.shape[1]
            
            if model_features != input_features:
                print(f"[警告] [WARN] 特征数量不匹配！")
                print(f"  模型期望: {model_features} 个特征")
                print(f"  实际输入: {input_features} 个特征")
                print(f"  差异: {input_features - model_features} 个特征")
                print(f"  建议: 删除旧模型并重新训练")
                print(f"  跳过在线更新，避免预测错误")
                return False
            
            try:
                current_pred = self.create_ensemble_prediction(X_new)
                current_rmse = np.sqrt(np.mean((y_new - current_pred) ** 2))
                
                # 记录性能历史
                self.performance_history.append({
                    'timestamp': pd.Timestamp.now(),
                    'rmse': current_rmse,
                    'data_size': len(X_new)
                })
                
                print(f"当前模型RMSE: {current_rmse:.4f}")
                
                # 判断是否需要重训练
                if len(self.performance_history) > 7:  # 有足够的历史数据
                    recent_rmse = [h['rmse'] for h in self.performance_history[-7:]]
                    avg_rmse = np.mean(recent_rmse)
                    
                    if current_rmse > avg_rmse * 1.1:  # 性能下降超过10%
                        print("检测到性能下降，建议重新训练模型")
                        return True  # 需要重训练
                
                print("[OK] 模型性能稳定")
                return False  # 不需要重训练
                
            except Exception as e:
                print(f"[错误] 在线预测失败: {e}")
                print(f"建议: 删除旧模型并重新训练")
                return False
        
        print("[OK] 模型性能稳定")
        return False  # 不需要重训练
    
    def save_models(self, save_dir='models', feature_cols=None):
        """保存训练好的模型
        
        参数:
            save_dir: 保存目录
            feature_cols: 训练时使用的特征列名列表（用于验证）
        """
        from pathlib import Path
        import pickle
        import joblib
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    model.save_model(str(save_path / f'{name}_model.txt'))
                elif name in ['xgboost', 'catboost']:
                    model.save_model(str(save_path / f'{name}_model.bin'))
                else:
                    # sklearn模型
                    joblib.dump(model, save_path / f'{name}_model.pkl')
                
                print(f"[OK] {name} 模型已保存")
            
            except Exception as e:
                print(f" 保存 {name} 模型失败: {e}")
        
        # 保存配置和其他信息
        with open(save_path / 'config.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        # 保存特征列信息（用于后续验证）
        if feature_cols is not None:
            feature_info = {
                'feature_cols': list(feature_cols),
                'num_features': len(feature_cols),
                'target_mode': getattr(self, 'target_mode_', self._prediction_cfg().get('target_mode', 'price')),
                'target_baseline_col': getattr(self, 'target_baseline_col_', self._baseline_col()),
                'saved_at': pd.Timestamp.now().isoformat()
            }
            with open(save_path / 'feature_info.pkl', 'wb') as f:
                pickle.dump(feature_info, f)
            print(f" 特征信息已保存: {len(feature_cols)} 个特征")

        if hasattr(self, 'ensemble_weights_') and self.ensemble_weights_:
            with open(save_path / 'ensemble_weights.pkl', 'wb') as f:
                pickle.dump(self.ensemble_weights_, f)
            print("[OK] 集成权重已保存")

        if hasattr(self, 'direction_classifier_') and self.direction_classifier_ is not None:
            joblib.dump(self.direction_classifier_, save_path / 'direction_classifier.pkl')
            print("[OK] 方向分类器已保存")

        if hasattr(self, 'direction_classifiers_') and self.direction_classifiers_:
            joblib.dump(self.direction_classifiers_, save_path / 'direction_classifiers.pkl')
            print(f"[OK] 方向分类器集成已保存: {len(self.direction_classifiers_)} 个模型")

        direction_info = {
            'direction_correction_min_proba': float(getattr(
                self,
                'direction_correction_min_proba_',
                self._prediction_cfg().get('direction_correction_min_proba', 0.55)
            )),
            'direction_deadband': self._direction_deadband(),
            'direction_classifier_weights': getattr(self, 'direction_classifier_weights_', {}),
            'direction_class_order': getattr(self, 'direction_class_order_', np.array([-1, 0, 1])).tolist(),
        }
        with open(save_path / 'direction_info.pkl', 'wb') as f:
            pickle.dump(direction_info, f)
        print("[OK] 方向修正参数已保存")

        # 保存两阶段模型组件
        if hasattr(self, 'two_stage_trained_') and self.two_stage_trained_:
            print(" 保存两阶段模型组件...")
            if hasattr(self, 'spike_classifier_') and self.spike_classifier_ is not None:
                self.spike_classifier_.save_model(str(save_path / 'spike_classifier.txt'))
                print("[OK] 尖峰分类器已保存")
            if hasattr(self, 'spike_threshold_'):
                np.save(str(save_path / 'spike_threshold.npy'), np.array([self.spike_threshold_]))
                print(f"[OK] 尖峰阈值 {self.spike_threshold_:.4f} 已保存")
            # 分组模型
            for group_name in ['spike', 'normal']:
                group_models = getattr(self, f'models_{group_name}_', {})
                for name, model in group_models.items():
                    try:
                        if name == 'lightgbm':
                            model.save_model(str(save_path / f'{group_name}_lightgbm_model.txt'))
                        elif name in ['xgboost', 'catboost']:
                            model.save_model(str(save_path / f'{group_name}_{name}_model.bin'))
                        else:
                            joblib.dump(model, save_path / f'{group_name}_{name}_model.pkl')
                    except Exception as e:
                        print(f"  保存 {group_name}/{name} 失败: {e}")
            print("[OK] 两阶段模型组件已保存")

        print(f"[OK] 所有模型已保存到 {save_dir}")
    
    def load_models(self, load_dir='models'):
        """从本地加载训练好的模型"""
        from pathlib import Path
        import joblib
        import pickle
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostRegressor
            
        load_path = Path(load_dir)
        if not load_path.exists():
            print(f" 模型目录 {load_dir} 不存在！")
            return False
                
        print(f"尝试从 {load_dir} 加载模型...")
        self.models = {}
            
        try:
            # 1. 尝试加载LightGBM
            lgb_path = load_path / 'lightgbm_model.txt'
            if lgb_path.exists():
                self.models['lightgbm'] = lgb.Booster(model_file=str(lgb_path))
                print("[OK] LightGBM 模型加载成功")
                    
            # 2. 尝试加载XGBoost
            xgb_path = load_path / 'xgboost_model.bin'
            if xgb_path.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                self.models['xgboost'] = model
                print("[OK] XGBoost 模型加载成功")
                    
            # 3. 尝试加载CatBoost
            cat_path = load_path / 'catboost_model.bin'
            if cat_path.exists():
                model = CatBoostRegressor()
                model.load_model(str(cat_path))
                self.models['catboost'] = model
                print("[OK] CatBoost 模型加载成功")
                    
            # 4. 尝试加载Scikit-Learn模型 (随机森林, 梯度提升等)
            for pkl_file in load_path.glob('*_model.pkl'):
                model_name = pkl_file.stem.replace('_model', '')
                self.models[model_name] = joblib.load(pkl_file)
                print(f"[OK] {model_name} 模型加载成功")
                    
            # 加载配置
            config_path = load_path / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config.update(yaml.safe_load(f) or {})
                print("[OK] 模型配置加载成功")
                
            # 5. 加载特征信息（新增）
            feature_info_path = load_path / 'feature_info.pkl'
            if feature_info_path.exists():
                with open(feature_info_path, 'rb') as f:
                    feature_info = pickle.load(f)
                self.training_feature_cols = feature_info.get('feature_cols', [])
                self.training_num_features = feature_info.get('num_features', 0)
                self.target_mode_ = feature_info.get(
                    'target_mode',
                    'price'
                )
                self.target_baseline_col_ = feature_info.get(
                    'target_baseline_col',
                    self._baseline_col()
                )
                print(f"[OK] 特征信息加载成功: {self.training_num_features} 个特征")
            else:
                print("[警告] 未找到特征信息文件，将无法验证特征匹配")
                self.training_feature_cols = []
                self.training_num_features = 0
                self.target_mode_ = 'price'
                self.target_baseline_col_ = self._baseline_col()

            weights_path = load_path / 'ensemble_weights.pkl'
            if weights_path.exists():
                with open(weights_path, 'rb') as f:
                    self.ensemble_weights_ = pickle.load(f)
                print("[OK] 集成权重加载成功")

            direction_clf_path = load_path / 'direction_classifier.pkl'
            if direction_clf_path.exists():
                self.direction_classifier_ = joblib.load(direction_clf_path)
                print("[OK] 方向分类器加载成功")

            direction_clfs_path = load_path / 'direction_classifiers.pkl'
            if direction_clfs_path.exists():
                self.direction_classifiers_ = joblib.load(direction_clfs_path)
                if self.direction_classifiers_:
                    self.direction_classifier_ = next(iter(self.direction_classifiers_.values()))
                print(f"[OK] 方向分类器集成加载成功: {len(self.direction_classifiers_)} 个模型")

            direction_info_path = load_path / 'direction_info.pkl'
            if direction_info_path.exists():
                with open(direction_info_path, 'rb') as f:
                    direction_info = pickle.load(f)
                self.direction_correction_min_proba_ = direction_info.get(
                    'direction_correction_min_proba',
                    self._prediction_cfg().get('direction_correction_min_proba', 0.55)
                )
                self.direction_classifier_weights_ = direction_info.get('direction_classifier_weights', {})
                self.direction_class_order_ = np.array(
                    direction_info.get('direction_class_order', [-1, 0, 1]),
                    dtype=int
                )
                print(f"[OK] 方向修正参数加载成功: threshold={self.direction_correction_min_proba_:.3f}")

            # 6. 尝试加载两阶段模型组件
            classifier_path = load_path / 'spike_classifier.txt'
            threshold_path = load_path / 'spike_threshold.npy'
            spike_lgb_path = load_path / 'spike_lightgbm_model.txt'
            normal_lgb_path = load_path / 'normal_lightgbm_model.txt'

            if classifier_path.exists() and threshold_path.exists():
                try:
                    self.spike_classifier_ = lgb.Booster(model_file=str(classifier_path))
                    self.spike_threshold_ = float(np.load(str(threshold_path))[0])
                    self.models_spike_ = {}
                    self.models_normal_ = {}

                    # 加载分组模型
                    for group_name, store_attr in [('spike', self.models_spike_), ('normal', self.models_normal_)]:
                        for f in load_path.glob(f'{group_name}_*_model.*'):
                            stem = f.stem
                            model_name = stem.replace(f'{group_name}_', '').replace('_model', '')
                            if f.suffix == '.txt':
                                store_attr[model_name] = lgb.Booster(model_file=str(f))
                            elif f.suffix == '.bin' and 'xgboost' in model_name:
                                m = xgb.XGBRegressor()
                                m.load_model(str(f))
                                store_attr[model_name] = m
                            elif f.suffix == '.bin' and 'catboost' in model_name:
                                m = CatBoostRegressor()
                                m.load_model(str(f))
                                store_attr[model_name] = m
                            elif f.suffix == '.pkl':
                                store_attr[model_name] = joblib.load(f)

                    if self.models_spike_ and self.models_normal_:
                        self.two_stage_trained_ = True
                        print(f"[OK] 两阶段模型加载成功 (尖峰阈值={self.spike_threshold_:.4f}, "
                              f"尖峰模型={len(self.models_spike_)}个, 常规模型={len(self.models_normal_)}个)")
                except Exception as e:
                    print(f"  加载两阶段模型失败: {e}")

            return len(self.models) > 0
                
        except Exception as e:
            print(f" 加载模型时发生成错误: {e}")
            return False

    def predict_by_time_range(self, data, start_time, end_time, target_col='price'):
        """预测指定时间范围内的数据
        
        参数:
        data: 完整数据集(需要包含timestamp列)
        start_time: 起始时间, 例如 '2026-01-01 00:00:00'
        end_time: 结束时间, 例如 '2026-01-31 23:00:00'
        target_col: 需要分离出的目标列名
        """
        # 将输入时间字符串转换为时间对象用于切片筛选
        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)
        
        # 筛选符合时间区间的数据
        mask = (data['timestamp'] >= start) & (data['timestamp'] <= end)
        range_data = data[mask].copy()
        
        if range_data.empty:
            print(f"[WARN] 在 {start} 到 {end} 范围内没有找到数据！")
            return None, None, None
            
        print(f"获取到 {start_time} 至 {end_time} 的数据，共 {len(range_data)} 条记录")
        
        # 将特征列与目标列（如有）、timestamp分离
        feature_cols = [col for col in range_data.columns if col not in ['timestamp', target_col]]
        X = range_data[feature_cols].copy()
        
        y = None
        if target_col in range_data.columns:
            y = range_data[target_col]
            
        # 核心：必须经过和训练时一样的特征清洗过程
        X = sanitize_features(X)

        # 优先使用两阶段预测，否则使用单阶段集成
        if hasattr(self, 'two_stage_trained_') and self.two_stage_trained_:
            predictions = self._predict_two_stage(X, use_blend=True)
        else:
            predictions = self.create_ensemble_prediction(X, method='weighted_average')

        return range_data['timestamp'], y, predictions
    
    def forecast_future(self, data_with_features, horizon_steps=672,
                        method='weighted_average', future_weather_df=None):
        """Recursive future forecast with optional weather injection.

        Args:
            data_with_features: feature-engineered historical data
            horizon_steps: forecast horizon (15min per step)
            method: ensemble method
            future_weather_df: forecast weather DataFrame aligned to 15min
        """
        print(f"\n开始递推未来 {horizon_steps} 个步长...")
        raw_df = data_with_features[['timestamp', 'price']].copy()

        # 如果有天气数据，将天气也加入raw_df
        weather_cols_in_data = [c for c in data_with_features.columns
                                if c in ['temp_air', 'relative_humidity', 'precipitation',
                                          'wind_speed', 'wind_direction', 'pressure',
                                          'cloud_cover', 'ghi', 'dni', 'dhi',
                                          'dew_point', 'et0', 'vpd']]
        has_weather = len(weather_cols_in_data) > 0
        has_future_weather = future_weather_df is not None and len(future_weather_df) > 0

        # 扩展raw_df以包含天气列
        for col in weather_cols_in_data:
            if col not in raw_df.columns:
                raw_df[col] = data_with_features[col].values
        raw_weather_cols = [c for c in weather_cols_in_data if c in raw_df.columns]

        # 如果没有未来天气，使用历史均值填充
        if has_weather and not has_future_weather:
            weather_means = {}
            for col in raw_weather_cols:
                weather_means[col] = raw_df[col].median()
            print("[递推] 无未来天气预报，使用历史天气中位数填充")
        else:
            weather_means = {}

        # 处理日前电价列：未来不可得，用最近一天的日前电价模式循环填充
        has_dayahead = 'dayahead_price' in data_with_features.columns
        last_da_cycle = None
        if has_dayahead:
            raw_df['dayahead_price'] = data_with_features['dayahead_price'].values
            last_da_cycle = data_with_features['dayahead_price'].iloc[-96:].values
            if len(last_da_cycle) < 96:
                last_da_cycle = data_with_features['dayahead_price'].values[-96:]
            print(f"[递推] 日前电价列已加入，未来步骤将循环最近 {len(last_da_cycle)} 个日前电价模式")

        # 处理负荷列：未来不可得，用最近一天的负荷模式循环填充
        has_demand = 'demand' in data_with_features.columns
        last_demand_cycle = None
        if has_demand:
            raw_df['demand'] = data_with_features['demand'].values
            last_demand_cycle = data_with_features['demand'].iloc[-96:].values
            if len(last_demand_cycle) < 96:
                last_demand_cycle = data_with_features['demand'].values[-96:]
            print(f"[递推] 负荷列已加入，未来步骤将循环最近 {len(last_demand_cycle)} 个负荷模式")

        # 处理地方出力列：未来不可得，用最近一天的地方出力模式循环填充
        has_local_power = 'local_power_output' in data_with_features.columns
        last_local_power_cycle = None
        if has_local_power:
            raw_df['local_power_output'] = data_with_features['local_power_output'].values
            last_local_power_cycle = data_with_features['local_power_output'].iloc[-96:].values
            if len(last_local_power_cycle) < 96:
                last_local_power_cycle = data_with_features['local_power_output'].values[-96:]
            print(f"[递推] 地方出力列已加入，未来步骤将循环最近 {len(last_local_power_cycle)} 个地方出力模式")

        preds, timestamps = [], []
        import sys, os

        for h in range(1, horizon_steps + 1):
            next_ts = raw_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=15)

            # 构建新行：价格占位，天气从预报中获取或均值填充
            new_row = {'timestamp': next_ts, 'price': np.nan}

            # 日前电价：循环最近的日前电价模式
            if has_dayahead and last_da_cycle is not None:
                da_idx = (h - 1) % len(last_da_cycle)
                new_row['dayahead_price'] = float(last_da_cycle[da_idx])

            # 负荷：循环最近的负荷模式
            if has_demand and last_demand_cycle is not None:
                demand_idx = (h - 1) % len(last_demand_cycle)
                new_row['demand'] = float(last_demand_cycle[demand_idx])

            # 地方出力：循环最近的地方出力模式
            if has_local_power and last_local_power_cycle is not None:
                local_idx = (h - 1) % len(last_local_power_cycle)
                new_row['local_power_output'] = float(last_local_power_cycle[local_idx])

            if has_future_weather:
                # 从未来天气DataFrame中查找对应时间点的天气
                matched = False
                for col in raw_weather_cols:
                    if col in future_weather_df.columns:
                        try:
                            val = future_weather_df.loc[next_ts, col]
                            if not pd.isna(val):
                                new_row[col] = float(val)
                                matched = True
                        except (KeyError, TypeError):
                            pass
                if not matched and weather_means:
                    for col in raw_weather_cols:
                        new_row[col] = weather_means.get(col, 0)
            elif weather_means:
                for col in raw_weather_cols:
                    new_row[col] = weather_means[col]

            raw_df.loc[len(raw_df)] = new_row

            # 只保留最近2000行用于特征计算
            current_chunk = raw_df.tail(2000).copy().reset_index(drop=True)

            # 静默执行特征工程
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                feat_df = self.advanced_feature_engineering(current_chunk)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout

            # 取最新一行作为预测输入
            X_row = feat_df.iloc[[-1]].copy()
            feature_cols = [c for c in feat_df.columns if c not in ['timestamp', 'price']]
            X_row_sanitized = sanitize_features(X_row[feature_cols])

            # 优先用两阶段预测，否则用单阶段集成
            if hasattr(self, 'two_stage_trained_') and self.two_stage_trained_:
                pred = self._predict_two_stage(X_row_sanitized, use_blend=True)[0]
            else:
                pred = self.create_ensemble_prediction(X_row_sanitized, method=method)[0]
            raw_df.loc[raw_df.index[-1], 'price'] = float(pred)

            timestamps.append(next_ts)
            preds.append(float(pred))

            if h % 96 == 0:
                print(f"-> 已推演 {h//96} 天 ({h}/{horizon_steps})")

        return timestamps, preds

    def plot_prediction_analysis(self, y_true, y_pred, model_name="Model"):
        """预测结果分析图"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 真实值vs预测值散点图
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('真实值')
        axes[0, 0].set_ylabel('预测值')
        axes[0, 0].set_title(f'{model_name} - 真实值vs预测值')
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title(f'{model_name} - 残差分析')
        
        # 3. 残差分布
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title(f'{model_name} - 残差分布')
        
        # 4. 时间序列对比
        indices = range(len(y_true))
        axes[1, 1].plot(indices, y_true, label='真实值', alpha=0.8)
        axes[1, 1].plot(indices, y_pred, label='预测值', alpha=0.8)
        axes[1, 1].set_xlabel('时间点')
        axes[1, 1].set_ylabel('价格')
        axes[1, 1].set_title(f'{model_name} - 时间序列对比')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()


def _merge_weather_data(data, lat, lon, predictor=None):
    """获取历史天气数据并与电价数据合并

    Args:
        data: 电价DataFrame (需有 'timestamp' 列或DatetimeIndex)
        lat, lon: 经纬度
        predictor: AdvancedElectricityPredictor 实例（用于缓存）
    Returns: 合并了天气列后的DataFrame
    """
    if not WEATHER_AVAILABLE:
        print("[天气] 天气模块不可用，跳过")
        return data

    # 确定日期范围
    if 'timestamp' in data.columns:
        ts = pd.to_datetime(data['timestamp'])
    elif isinstance(data.index, pd.DatetimeIndex):
        ts = data.index.to_series().reset_index(drop=True)
    else:
        print("[天气] 无法确定数据时间范围，跳过")
        return data

    start_date = ts.min()
    end_date = ts.max()

    # 获取历史天气数据（Open-Meteo 免费，无需token）
    weather = fetch_historical_weather(
        lat, lon,
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
    )

    if weather is None or len(weather) == 0:
        print("[天气] 未获取到天气数据，继续使用纯电价特征")
        return data

    # 重采样到15分钟
    weather_15min = resample_weather_to_15min(weather)
    if weather_15min is None or len(weather_15min) == 0:
        return data

    # 对齐到电价时间戳
    if 'timestamp' in data.columns:
        price_ts = data['timestamp']
    else:
        price_ts = data.index

    aligned_weather = align_weather_to_price_data(weather_15min, price_ts)

    # 合并到原数据
    weather_cols = [c for c in aligned_weather.columns
                     if c not in ('timestamp', 'price', 'time')]

    if 'timestamp' in data.columns:
        # 确保data也有时间戳索引用于合并
        data_indexed = data.set_index('timestamp')
        for col in weather_cols:
            data_indexed[col] = aligned_weather[col]
        data = data_indexed.reset_index()
    else:
        for col in weather_cols:
            data[col] = aligned_weather[col].values

    print(f"[天气] 已合并 {len(weather_cols)} 个天气变量到电价数据")
    return data


def train_station_model(station_name, lat=None, lon=None):
    """针对指定电站进行全量数据训练并保存模型"""
    print(f"\n[训练阶段] 开始针对电站: {station_name} 进行模型训练...")

    # 1. 初始化高级预测器
    predictor = AdvancedElectricityPredictor(lat=lat, lon=lon)

    # 2. 生成或加载数据
    print(f"[训练阶段] 准备 {station_name} 的历史数据...")
    data = load_station_electricity_data(station_name)
    if data.empty:
        raise ValueError(f"未获取到 {station_name} 的数据，无法完成训练。")

    # 2.5 获取并合并天气数据（如果经纬度已配置）
    if lat is not None and lon is not None and WEATHER_AVAILABLE:
        data = _merge_weather_data(data, lat, lon, predictor)
        # 验证天气数据是否成功合并
        weather_cols = [c for c in data.columns if c in [
            'temp_air', 'wind_speed', 'ghi', 'dni', 'dhi', 'pressure'
        ]]
        if weather_cols:
            print(f"[验证] [OK] 成功合并 {len(weather_cols)} 个天气特征: {weather_cols[:5]}...")
            print(f"[验证] 天气数据统计:")
            print(f"  - 温度范围: {data['temp_air'].min():.1f} ~ {data['temp_air'].max():.1f} °C")
            if 'ghi' in data.columns:
                print(f"  - 辐照度范围: {data['ghi'].min():.1f} ~ {data['ghi'].max():.1f} W/m²")
        else:
            print(f"[警告] [WARN] 天气数据合并失败！请检查:")
            print(f"  1. 经纬度是否正确: lat={lat}, lon={lon}")
            print(f"  2. 网络连接是否正常")
            print(f"  3. API token是否有效")
            print(f"  继续使用纯电价特征训练")

    # 2.6 合并日前电价和负荷数据
    data = _merge_dayahead_data(data, station_name)
    data = _merge_demand_data(data)

    # 3. 高级特征工程
    print("[训练阶段] 执行高级特征工程...")
    data_with_features = predictor.advanced_feature_engineering(data)
    
    # 4. 数据分割 (按照时间顺序 70% 训练 / 15% 验证 / 15% 测试)
    print("[训练阶段] 分割数据...")
    train_size = int(len(data_with_features) * 0.7)
    val_size = int(len(data_with_features) * 0.15)
    
    train_data = data_with_features[:train_size]
    val_data = data_with_features[train_size:train_size+val_size]
    test_data = data_with_features[train_size+val_size:]
    
    feature_cols = [col for col in data_with_features.columns 
                   if col not in ['timestamp', 'price']]
    
    X_train = train_data[feature_cols].fillna(0)
    y_train = train_data['price']
    X_val = val_data[feature_cols].fillna(0)
    y_val = val_data['price']
    X_test = test_data[feature_cols].fillna(0)
    y_test = test_data['price']

    X_train = sanitize_features(X_train)
    X_val   = sanitize_features(X_val)
    X_test  = sanitize_features(X_test)
    
    # 5. 训练集成模型
    print("[训练阶段] 开始训练子模型...")
    predictor.train_ensemble_models(X_train, y_train, X_val, y_val)

    # 5.5 两阶段模型（如果启用）
    if predictor.config.get('two_stage', {}).get('enable', True):
        print("[训练阶段] 训练两阶段模型（尖峰/非尖峰分离）...")
        predictor.train_two_stage_models(X_train, y_train, X_val, y_val)

    print("\n6. 生成集成预测...")
    if hasattr(predictor, 'two_stage_trained_') and predictor.two_stage_trained_:
        predictions = predictor._predict_two_stage(X_test, use_blend=True)
        print("  [使用两阶段预测]")
    else:
        predictions = predictor.create_ensemble_prediction(X_test, method='weighted_average')
    
    # 7. 评估性能
    print("\n7. 评估模型性能...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
        
    # 修复 MAPE 计算：排除零值和极小值
    mask = np.abs(y_test) > 1e-6  # 过滤接近 0 的值
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
        valid_samples = mask.sum()
    else:
        mape = np.nan
        valid_samples = 0
        
    # 计算 SMAPE（对称MAPE，更稳定）
    denominator = (np.abs(y_test) + np.abs(predictions)) / 2
    denominator[denominator == 0] = 1e-6  # 避免除零
    smape = np.mean(np.abs(y_test - predictions) / denominator) * 100
        
    # 计算解释方差（比 R² 更稳定）
    from sklearn.metrics import explained_variance_score
    explained_var = explained_variance_score(y_test, predictions)
        
    print(f" 集成模型性能:")
    print(f"   RMSE:            {rmse:.4f} 元")
    print(f"   MAE:             {mae:.4f} 元")
    if not np.isnan(mape):
        print(f"   MAPE:            {mape:.2f}% (基于 {valid_samples}/{len(y_test)} 个有效样本)")
    else:
        print(f"   MAPE:            N/A (数据中包含大量零值)")
    print(f"   SMAPE:           {smape:.2f}% (对称MAPE，推荐使用)")
    print(f"   R²:              {r2:.4f}")
    print(f"   解释方差:        {explained_var:.4f}")
        
    # 额外统计信息
    print(f"\n 预测统计:")
    print(f"   真实值范围:      {y_test.min():.4f} ~ {y_test.max():.4f} 元")
    print(f"   预测值范围:      {predictions.min():.4f} ~ {predictions.max():.4f} 元")
    print(f"   真实值均值:      {y_test.mean():.4f} 元")
    print(f"   预测值均值:      {predictions.mean():.4f} 元")
        
    # 误差分布
    errors = np.abs(y_test - predictions)
    print(f"\n 误差分布:")
    print(f"   中位数误差:      {np.median(errors):.4f} 元")
    print(f"   90%分位误差:     {np.percentile(errors, 90):.4f} 元")
    print(f"   95%分位误差:     {np.percentile(errors, 95):.4f} 元")
    print(f"   最大误差:        {errors.max():.4f} 元")
    
    # 8. 模型解释性分析
    print("\n8. 执行模型解释性分析...")
    interpretation = predictor.model_interpretation(X_test, y_test)
    
    # 额外输出天气特征重要性
    if 'lightgbm' in predictor.models:
        print("\n[天气特征分析]")
        model = predictor.models['lightgbm']
        feature_names = feature_cols
        importances = model.feature_importance()
        
        # 找出天气相关特征
        weather_features = [f for f in feature_names if any(w in f for w in [
            'temp', 'wind', 'ghi', 'dni', 'dhi',
            'pressure', 'dew', 'vpd', 'et0'
        ])]
        
        if weather_features:
            print(f"模型中包含 {len(weather_features)} 个天气特征")
            weather_importance = []
            for wf in weather_features:
                if wf in feature_names:
                    idx = feature_names.index(wf)
                    weather_importance.append((wf, importances[idx]))
            
            # 按重要性排序
            weather_importance.sort(key=lambda x: x[1], reverse=True)
            print("Top 10 天气特征重要性:")
            for i, (feat, imp) in enumerate(weather_importance[:10], 1):
                print(f"  {i:2d}. {feat:<30}: {imp:>8.2f}")
            
            # 计算天气特征总重要性占比
            total_importance = importances.sum()
            weather_total = sum(imp for _, imp in weather_importance)
            weather_pct = (weather_total / total_importance * 100) if total_importance > 0 else 0
            print(f"\n天气特征总重要性占比: {weather_pct:.2f}%")
            
            if weather_pct < 5:
                print("[警告] 天气特征重要性过低，可能原因:")
                print("  1. 天气数据与电价相关性弱")
                print("  2. 天气数据质量有问题（缺失值过多/异常值）")
                print("  3. 价格滞后特征过于强势，掩盖了天气信号")
                print("  建议: 尝试移除部分价格滞后特征，或增加天气特征的衍生项")
        else:
            print("[警告] 模型中未检测到天气特征！")
            print("请检查:")
            print("  1. 训练数据是否成功合并天气列")
            print("  2. 特征工程是否正确处理天气列")
    
    print("\n9.预测结果可视化...")
    predictor.plot_prediction_analysis(y_test, predictions, model_name=f"集成模型 - {station_name}")
    # 6. 为该电站保存专属模型 (文件夹名增加电站标识)
    model_dir = f'./model_sets/models_{station_name}'
    print(f"[训练阶段] 训练完成，将模型写入 {model_dir} ...")
    predictor.save_models(model_dir, feature_cols=feature_cols)
    print("[训练阶段] 所有流程完成！\n")
    
    # 为了方便连续推理，返回包含已训练模型的 predictor 和最新特征数据
    return predictor, data_with_features


def run_production_inference(station_name='百合站', target_start='2026-01-01 00:00:00',
                           target_end='2026-01-31 23:00:00', lat=None, lon=None, force_retrain=False):
    """结合了如果无历史模型则自动训练的推断工作流"""
    print(f"=== 电力价格预测系统 - 自动生产工作流 ({station_name}) ===")

    # 专属该电站的模型存放路径
    model_dir = f'./model_sets/models_{station_name}'

    # 1. 尝试直接加载该电站历史模型
    predictor = AdvancedElectricityPredictor(lat=lat, lon=lon)
    is_loaded = predictor.load_models(model_dir)

    # 需要用于推断的特征加工后的数据（如果在内存里已算完就无需再次计算）
    data_with_features = None

    # 2. 根据是否存在模型决定是否立刻训练
    if force_retrain:
        print(f"[强制重训] 忽略已有模型，重新训练...")
        # 删除旧模型文件夹
        import shutil
        if Path(model_dir).exists():
            shutil.rmtree(model_dir)
            print(f"已删除旧模型: {model_dir}")
        predictor, data_with_features = train_station_model(station_name, lat=lat, lon=lon)
    elif not is_loaded:
        print(f" 未在 {model_dir} 找到 {station_name} 的可用历史模型，触发首次全量训练机制...")
        predictor, data_with_features = train_station_model(station_name, lat=lat, lon=lon)
    else:
        print(f"[OK] 成功加载 {station_name} 的历史最佳模型。")
            
        # 检查特征数量是否匹配
        if hasattr(predictor.models.get('lightgbm'), 'num_feature'):
            model_features = predictor.models['lightgbm'].num_feature()
            print(f"[检查] 模型期望特征数: {model_features}")

    # 3. 准备推断数据 (如果上面训练流没有跑，则此处需加载最新的全量历史用于特征算子)
    if data_with_features is None:
        print(f"\n[推断阶段] 提取 {station_name} 最新数据环境以计算推断时刻的特征...")
        data = load_station_electricity_data(station_name)
        if lat is not None and lon is not None and WEATHER_AVAILABLE:
            data = _merge_weather_data(data, lat, lon, predictor)
        data = _merge_dayahead_data(data, station_name)
        data = _merge_demand_data(data)
        data_with_features = predictor.advanced_feature_engineering(data)

    # 4. 限定我们需要推断与输出报表的时间区间
    print(f"\n[推断阶段] 查询预测区间：{target_start} 到 {target_end}")
    
    if data_with_features.empty:
        print("未获取到推断基础数据，退出。")
        return
        
    timestamps, y_true, y_pred = predictor.predict_by_time_range(
        data=data_with_features, 
        start_time=target_start, 
        end_time=target_end
    )
    
    # 5. 推断结果和稳定性验证
    if y_pred is not None:
        if y_true is not None and not y_true.isnull().all():
            print("\n[监视器] 区间内包含真实标签数据，将用于在线更新检测与回溯对齐...")
            mask = (data_with_features['timestamp'] >= target_start) & (data_with_features['timestamp'] <= target_end)
            feature_cols = [c for c in data_with_features.columns if c not in ['timestamp', 'price']]
            X_new_infer = sanitize_features(data_with_features.loc[mask, feature_cols])
            
            # 检测是否触发再次重训阀门
            need_retrain = predictor.online_learning_update(X_new_infer, y_true)
            if need_retrain:
                print(f"[WARN] 警报：{station_name} 模型在指定历史区间推断大幅度退步。需要执行重新训练流程。")
            else:
                print(f"[OK] {station_name} 生产模型在推断该时间段时表现非常稳定。")
                
            predictor.plot_prediction_analysis(y_true, y_pred, model_name=f"集成学习 - {station_name}")
        else:
            print(f"注意：当前区间未包含 {station_name} 的真实标签值，仅输出纯预测值。")
            print("预测结果样例 (前5条):")
            for t, p in zip(timestamps.head(5), y_pred[:5]):
                print(f"[{t}] P预测 = {p:.4f}")

def run_production_inference_weekly(station_name='百合站', forecast_days=7, lat=None, lon=None):
    """每日自动运行的流水线"""
    print(f"=== 电力价格预测系统 - 自动工作流 ({station_name}) ===")

    model_dir = f'./model_sets/models_{station_name}'
    predictor = AdvancedElectricityPredictor(lat=lat, lon=lon)

    print(f"\\n[获取数据] 获取 {station_name} 截至今日的所有历史数据...")
    data = load_station_electricity_data(station_name)
    if data is None or data.empty:
        print("无数据，退出。")
        return

    # 合并历史天气数据
    if lat is not None and lon is not None and WEATHER_AVAILABLE:
        data = _merge_weather_data(data, lat, lon, predictor)
    # 合并日前电价和负荷数据
    data = _merge_dayahead_data(data, station_name)
    data = _merge_demand_data(data)

    data_with_features = predictor.advanced_feature_engineering(data)

    # 1. 尝试加载对应模型
    is_loaded = predictor.load_models(model_dir)

    if not is_loaded:
        print(f"[!] 未发现 {station_name} 的模型，开始首次初始化全量训练...")
        predictor, _ = train_station_model(station_name, lat=lat, lon=lon)
    else:
        print("[OK] 加载已保存的历史模型成功，免于重复训练。")

        recent_x = data_with_features.iloc[-100:].copy()
        feature_cols = [c for c in data_with_features.columns if c not in ['timestamp', 'price']]
        X_monitor = sanitize_features(recent_x[feature_cols])
        y_monitor = recent_x['price']

        need_retrain = predictor.online_learning_update(X_monitor, y_monitor)
        if need_retrain:
            print("[!] 警报：发现误差显著变大，建议触发夜间重新训练！")
        else:
            print("[OK] 当前模型依旧稳定，误差符合历史阈值！")

    # 2. 获取预测区间的天气数据（从数据末尾下一天开始）
    horizon = forecast_days * 96
    future_weather_15min = None
    if lat is not None and lon is not None and WEATHER_AVAILABLE:
        last_ts = data_with_features['timestamp'].iloc[-1]
        forecast_start = last_ts + pd.Timedelta(minutes=15)
        forecast_end = last_ts + pd.Timedelta(minutes=15) * horizon
        print(f"[天气] 预测区间: {forecast_start} ~ {forecast_end}")
        future_weather = fetch_weather_for_period(lat, lon, forecast_start, forecast_end)
        if future_weather is not None:
            future_weather_15min = resample_weather_to_15min(future_weather)
            # 截取所需时间窗口
            future_weather_15min = future_weather_15min[
                (future_weather_15min.index >= forecast_start)
                & (future_weather_15min.index <= forecast_end)
            ]
            print(f"[天气] 已获取预测区间天气，{len(future_weather_15min)}条15分钟记录")
        else:
            print("[天气] 未获取到预测区间天气数据，将使用历史天气均值填充")

    # 3. 对未来进行递推预测
    future_dates, future_preds = predictor.forecast_future(
        data_with_features,
        horizon_steps=horizon,
        future_weather_df=future_weather_15min,
    )

    # 4. 绘制未来曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    history_to_show = data_with_features.iloc[-72:]
    plt.plot(history_to_show['timestamp'], history_to_show['price'],
             label='过去真实电价', color='blue')
    plt.plot(future_dates, future_preds,
             label=f'未来推断电价 ({forecast_days}天)', color='red', linestyle='--')
    plt.title(f"{station_name} - 未来{forecast_days}天真实盲预测(含天气)")
    plt.xlabel("时间")
    plt.ylabel("预测电价")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # ---- 获取经纬度 ----
    lat, lon = None, None
    if WEATHER_AVAILABLE:
        print("\n[系统] 天气数据模块已就绪")
        use_weather = input("是否使用天气数据增强预测？(y/n，默认y): ").strip().lower()
        if use_weather != "n":
            lat, lon, _ = get_location_from_console()
            print(f"[系统] 将使用天气数据 (lat={lat}, lon={lon})")
        else:
            print("[系统] 跳过天气数据，使用纯电价特征")
    else:
        print("[系统] 天气模块不可用，使用纯电价特征模式")

    # 询问是否强制重训
    force_retrain = False
    if lat is not None and lon is not None:
        retrain_choice = input("\n是否删除旧模型并重新训练（使用天气数据）？(y/n，默认n): ").strip().lower()
        force_retrain = (retrain_choice == "y")
        if force_retrain:
            print("[系统] 将删除旧模型并使用天气数据重新训练")

    # 启动应用
    stations_to_run = ['百合站']  # 可扩充: ['百合站', '木瓜站', '青柠站']

    for station in stations_to_run:
        print("=" * 60)
        # 历史回溯模式:
        # run_production_inference(
        #     station_name=station,
        #     target_start='2026-01-01 00:00:00',
        #     target_end='2026-01-31 23:00:00',
        #     lat=lat, lon=lon,
        #     force_retrain=force_retrain,  # 新增参数
        # )
        run_production_inference_weekly(
            station_name=station, forecast_days=1,
            lat=lat, lon=lon,
        )
        print("=" * 60)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') 
    
    try:
        main()
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        print("\n💡 请确保:")
        print("1. 已安装所有必需的依赖库")
        print("2. 数据文件路径正确")
        print("3. 有足够的内存和计算资源")
        
    input("\n按Enter键退出...")

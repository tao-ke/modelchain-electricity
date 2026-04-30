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

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei',"SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# 导入天气API模块
try:
    from weather_api import (
        fetch_historical_weather,
        fetch_forecast_weather,
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
        print(f"✓ 数据加载成功，形状: {df.shape}")
    except Exception as e:
        print(f"⚠ 数据加载失败: {e}")
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
        print(f"⚠ 读取文件失败: {path} -> {e}")
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


def load_station_electricity_data(station_name, base_dir='电价数据', years=(2025, 2026)):
    """根据电站名搜索并合并 base_dir 下指定年份的excel文件，返回长表DataFrame（timestamp, price）。"""
    base = Path(base_dir)
    if not base.exists():
        print(f"⚠ 基目录不存在: {base_dir}")
        return None

    found_files = []
    station_lower = station_name.lower()
    for yr in years:
        search_dir = base / str(yr)
        if not search_dir.exists():
            continue
        for f in search_dir.rglob('*.xls*'):  # 建议用 xls*，兼容 xls/xlsx/xlsm
            if station_lower in f.stem.lower() or station_lower in f.name.lower():
                found_files.append(f)

    if not found_files:
        for f in base.rglob('*.xls*'):
            if station_lower in f.stem.lower() or station_lower in f.name.lower():
                found_files.append(f)

    if not found_files:
        print(f"未找到匹配文件: {station_name} 在 {base_dir} 的 {years} 目录下")
        return None

    print(f"找到 {len(found_files)} 个匹配文件，开始解析并合并...")
    parts = []
    for f in found_files:
        part = _parse_single_station_file(f)  # 这里返回 index=time, col=price
        if part is not None and not part.empty:
            parts.append(part)

    if not parts:
        print("解析后无有效数据")
        return None

    combined = pd.concat(parts)
    combined = combined[~combined.index.duplicated(keep='first')].sort_index()

    # 转成与 generate_advanced_sample_data 一致的格式
    combined = combined.reset_index().rename(columns={'time': 'timestamp'})
    if 'timestamp' not in combined.columns:
        combined = combined.rename(columns={combined.columns[0]: 'timestamp'})
    combined = combined[['timestamp', 'price']].copy()
    combined['timestamp'] = pd.to_datetime(combined['timestamp'], errors='coerce')
    combined['price'] = pd.to_numeric(combined['price'], errors='coerce')
    combined = combined.dropna(subset=['timestamp', 'price']).sort_values('timestamp').reset_index(drop=True)

    print(combined.head())
    return combined

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
                        'alpha': 1.2,
                        'metric': 'huber',
                        'num_leaves': 22,
                        'learning_rate': 0.17,
                        'feature_fraction': 0.7,
                        'bagging_fraction': 0.9,
                        'reg_alpha': 0.4,
                        'reg_lambda': 1.9,
                        'random_state': 42,
                        'bagging_freq': 6,
                        'min_child_samples': 97,
                        'min_child_weight': 0.07,
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
                'temp_air', 'relative_humidity', 'precipitation', 'wind_speed',
                'wind_direction', 'pressure', 'cloud_cover', 'ghi', 'dni', 'dhi',
                'dew_point', 'et0', 'vpd'
            ]]
            if weather_cols_in_data:
                df = self._create_weather_features(df)

        # 7. 技术指标特征
        df = self._create_technical_indicators(df)

        # 8. 异常检测特征
        df = self._create_anomaly_features(df)

        print(f"特征工程完成，最终特征数: {df.shape[1]}")
        return df
    
    def _create_time_features(self, df):
        """创建时间特征"""
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['period_of_day'] = df['hour'] * 4 + df['minute'] // 15 
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 96)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 96)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 季节特征
        df['season'] = df['month'] % 12 // 3
        
        # 工作日/节假日特征（简化版本）
        df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['day_of_week'] < 5)).astype(int)
        
        return df
    
    def _create_lag_and_rolling_features(self, df):
        """创建滞后和滚动统计特征"""
        lags = self.config['feature_engineering']['lag_periods']
        windows = self.config['feature_engineering']['rolling_windows']
        
        for lag in lags:
            df[f'price_lag_{lag}'] = df['price'].shift(lag)
            if 'demand' in df.columns:
                df[f'demand_lag_{lag}'] = df['demand'].shift(lag)
        
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
        df['price_diff_96'] = past_price.diff(96)  # 改为单日96
        df['price_pct_change_1'] = past_price.pct_change(1).replace([np.inf, -np.inf], np.nan).fillna(0)
        df['price_pct_change_96'] = past_price.pct_change(96).replace([np.inf, -np.inf], np.nan).fillna(0)
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
        # 简化的目标编码，实际应用中需要使用K折交叉验证
        categorical_cols = ['hour', 'day_of_week', 'month', 'season']
        
        for col in categorical_cols:
            if col in df.columns:
                # 全局均值编码（简化版本）
                global_mean = df['price'].mean()
                target_mean = df.groupby(col)['price'].mean()
                df[f'{col}_target_encoded'] = df[col].map(target_mean).fillna(global_mean)
                
                # 计数编码
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
        if 'temperature' in df.columns:
            df['temp_x_hour'] = df['temperature'] * df['hour']
            df['temp_x_season'] = df['temperature'] * df['season']
        # 天气-时间交互特征
        if 'temp_air' in df.columns:
            if 'hour' in df.columns:
                df['temp_x_hour'] = df['temp_air'] * df['hour']
            if 'season' in df.columns:
                df['temp_x_season'] = df['temp_air'] * df['season']
            if 'is_weekend' in df.columns:
                df['temp_x_weekend'] = df['temp_air'] * df['is_weekend']
        if 'cloud_cover' in df.columns:
            if 'hour' in df.columns:
                df['cloud_x_hour'] = df['cloud_cover'] * df['hour']
            if 'season' in df.columns:
                df['cloud_x_season'] = df['cloud_cover'] * df['season']
        return df

    def _create_technical_indicators(self, df):
        """创建技术指标特征"""
        # 【重要修复】：一切指标必须基于 past_price 结算
        past_price = df['price'].shift(1)
        df['sma_12'] = past_price.rolling(12).mean()
        df['sma_24'] = past_price.rolling(24).mean()
        df['ema_12'] = past_price.ewm(span=12).mean()
        df['ema_24'] = past_price.ewm(span=24).mean()
        
        rolling_mean = past_price.rolling(24).mean()
        rolling_std = past_price.rolling(24).std().replace(0, np.nan)
        df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
        df['bollinger_lower'] = rolling_mean - (rolling_std * 2)
        df['bollinger_ratio'] = ((past_price - rolling_mean) / (2 * rolling_std)).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        delta = past_price.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    def _create_anomaly_features(self, df):
        """创建异常检测特征"""
        past_price = df['price'].shift(1)
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
            'temp_air', 'relative_humidity', 'precipitation', 'wind_speed',
            'wind_direction', 'pressure', 'cloud_cover', 'ghi', 'dni', 'dhi',
            'dew_point', 'et0', 'vpd'
        ]
        available_vars = [v for v in weather_vars if v in df.columns]

        if not available_vars:
            return df

        # 温度相关衍生特征
        if 'temp_air' in df.columns:
            # 体感温度（热指数简化版）
            if 'relative_humidity' in df.columns:
                T = df['temp_air']
                RH = df['relative_humidity'].clip(0, 100)
                df['heat_index'] = (
                    -8.7847 + 1.6114*T + 2.3385*RH - 0.1461*T*RH
                    - 0.0123*T**2 - 0.0164*RH**2 + 0.0022*T**2*RH
                    + 0.00073*T*RH**2 - 0.000029*T**2*RH**2
                )
                # 温湿复合
                df['temp_humidity_interact'] = df['temp_air'] * df['relative_humidity'] / 100

            # 温度变化率
            df['temp_change_rate'] = df['temp_air'].diff()

        # 辐射相关衍生特征
        if 'ghi' in df.columns:
            # 辐射与时段交互（白天才有辐射）
            if 'hour' in df.columns:
                df['ghi_is_day'] = (df['ghi'] > 50).astype(int)
                df['ghi_hour_interact'] = df['ghi'] * df['hour']
            # 辐射利用率（与理论最大值的比率）
            df['ghi_ratio'] = df['ghi'].clip(lower=0) / 1000  # 标准化到[0,1]区间

        if 'cloud_cover' in df.columns:
            # 云量离散化
            df['cloud_cover_level'] = pd.cut(
                df['cloud_cover'].clip(0, 100),
                bins=[-0.01, 20, 60, 100],
                labels=[0, 1, 2]  # 0=晴, 1=多云, 2=阴
            ).astype(float)

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
        if all(v in df.columns for v in ['temp_air', 'cloud_cover', 'wind_speed']):
            # 舒适度指数（简化版）
            df['comfort_index'] = (
                df['temp_air'] * 0.4
                + (100 - df['cloud_cover']) * 0.3
                + df['wind_speed'] * 0.3
            )
            # 制冷/供暖需求指标
            df['cooling_degree'] = (df['temp_air'] - 26).clip(lower=0)   # >26°C需要制冷
            df['heating_degree'] = (18 - df['temp_air']).clip(lower=0)   # <18°C需要供暖

        if 'precipitation' in df.columns:
            # 降雨影响（大雨可能影响线路安全和用电行为）
            df['is_rainy'] = (df['precipitation'] > 0).astype(int)
            df['is_heavy_rain'] = (df['precipitation'] > 10).astype(int)

        return df

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
    
    def train_ensemble_models(self, X_train, y_train, X_val=None, y_val=None):
        """训练集成模型"""
        print("开始训练集成模型...")
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        models = {}

        # 用来计算验证误差并打印
        def print_val_score(model_name, model_obj, X_eval, y_eval):
            if X_eval is not None and y_eval is not None:
                preds = model_obj.predict(X_eval)
                rmse = np.sqrt(mean_squared_error(y_eval, preds))
                mae = mean_absolute_error(y_eval, preds)
                print(f"✓ {model_name} 训练完成 | 验证集 RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            else:
                print(f"✓ {model_name} 训练完成")
        
        # 1. LightGBM
        try:
            import lightgbm as lgb
            
            # 如果有验证集，使用优化过的参数
            if X_val is not None and y_val is not None:
                params = self.hyperparameter_optimization(X_train, y_train, X_val, y_val)
            else:
                params = self.config['model_params']['lightgbm']
            
            train_data = lgb.Dataset(X_train, label=y_train)
            if X_val is not None:
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
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
            print("⚠ LightGBM未安装")
        
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
            model.fit(X_train, y_train)
            models['xgboost'] = model
            print_val_score('XGBoost', model, X_val, y_val)
            
        except ImportError:
            print("⚠ XGBoost未安装")
        
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
            model.fit(X_train, y_train)
            models['catboost'] = model
            print_val_score('CatBoost', model, X_val, y_val)
            
        except ImportError:
            print("⚠ CatBoost未安装")
        
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
        pipeline.fit(X_train, y_train)
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
        model.fit(X_train, y_train)
        models['random_forest'] = model
        print_val_score('随机森林', model, X_val, y_val)
        
        self.models = models
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

        # ---- 尖峰定义 ----
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

        # ---- 阶段1: 尖峰分类器 ----
        print("[阶段1] 训练尖峰/非尖峰分类器 (LightGBM)...")
        import lightgbm as lgb

        clf_params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': -1,
            'is_unbalance': True,  # 尖峰通常是少数类
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

        # 评估分类器
        if X_val is not None and y_val_binary is not None:
            val_proba = self.spike_classifier_.predict(X_val)
            val_pred_binary = (val_proba > 0.5).astype(int)
            auc = roc_auc_score(y_val_binary, val_proba)
            print(f"[阶段1] 验证集 AUC = {auc:.4f}")
            print(classification_report(y_val_binary, val_pred_binary,
                                        target_names=['非尖峰', '尖峰'], zero_division=0))

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
            self.train_ensemble_models(X_spike, y_spike)
            self.models_spike_ = self.models.copy()
        else:
            print("[阶段2-尖峰] 尖峰样本不足30条，使用全量模型作为尖峰模型")
            self.models_spike_ = original_models.copy()

        # 非尖峰模型组
        self.models = {}  # 清空
        self.train_ensemble_models(X_normal, y_normal)
        self.models_normal_ = self.models.copy()

        # 恢复全量模型（保留给单阶段模式使用）
        self.models = self.models_spike_ if original_models else {}
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

        spike_proba = self.spike_classifier_.predict(X)

        # 保存原模型引用后切换
        saved_models = self.models

        if use_blend:
            # 软混合：用尖峰概率加权两组预测
            self.models = self.models_spike_
            preds_spike = self.create_ensemble_prediction(X, method='weighted_average')
            self.models = self.models_normal_
            preds_normal = self.create_ensemble_prediction(X, method='weighted_average')
            self.models = saved_models
            return preds_spike * spike_proba + preds_normal * (1 - spike_proba)
        else:
            # 硬分类：超过50%概率用尖峰模型
            is_spike = spike_proba > 0.5
            preds = np.zeros(len(X))
            self.models = self.models_spike_
            if is_spike.sum() > 0:
                preds[is_spike] = self.create_ensemble_prediction(X.loc[is_spike] if hasattr(X, 'loc') else X[is_spike], method='weighted_average')
            self.models = self.models_normal_
            if (~is_spike).sum() > 0:
                preds[~is_spike] = self.create_ensemble_prediction(X.loc[~is_spike] if hasattr(X, 'loc') else X[~is_spike], method='weighted_average')
            self.models = saved_models
            return preds

    def create_ensemble_prediction(self, X_test, method='stacking'):
        """创建集成预测"""
        if not self.models:
            raise ValueError("请先训练模型")
        
        # 生成基础预测
        base_predictions = {}
        for name, model in self.models.items():
            try:
                if name == 'lightgbm':
                    pred = model.predict(X_test)
                elif name == 'xgboost':
                    pred = model.predict(X_test)
                elif name == 'catboost':
                    pred = model.predict(X_test)
                else:
                    pred = model.predict(X_test)
                
                base_predictions[name] = pred
                
            except Exception as e:
                print(f"模型 {name} 预测失败: {e}")
                continue
        
        if method == 'simple_average':
            # 简单平均
            predictions = np.array(list(base_predictions.values()))
            return np.mean(predictions, axis=0)
        
        elif method == 'weighted_average':
            # 加权平均（可以基于验证集性能确定权重）
            weights = {
                'lightgbm': 0.3,
                'xgboost': 0.25,
                'catboost': 0.25,
                'ridge': 0.1,
                'random_forest': 0.1
            }
            
            weighted_pred = np.zeros(X_test.shape[0])
            total_weight = 0
            
            for name, pred in base_predictions.items():
                if name in weights:
                    weighted_pred += pred * weights[name]
                    total_weight += weights[name]
            
            return weighted_pred / total_weight if total_weight > 0 else weighted_pred
        
        else:  # stacking
            # 简化的Stacking（实际应用中需要单独的验证集）
            meta_features = np.column_stack(list(base_predictions.values()))
            
            from sklearn.linear_model import Ridge
            meta_model = Ridge(alpha=0.1)
            
            # 这里应该使用独立的验证集训练meta模型
            # 为了演示，我们直接返回加权平均
            return self.create_ensemble_prediction(X_test, 'weighted_average')
    
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
                print("✓ SHAP分析完成")
                
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
                    
                    print("✓ SHAP图表已保存")
                    
                except ImportError:
                    print("⚠ matplotlib未安装，跳过图表保存")
                
        except ImportError:
            print("⚠ SHAP库未安装，跳过SHAP分析")
        
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
                    print("✓ 部分依赖分析完成")
        
        except Exception as e:
            print(f"⚠ 部分依赖分析失败: {e}")
        
        return interpretation_results
    
    def online_learning_update(self, X_new, y_new):
        """在线学习模型更新"""
        print("执行在线学习更新...")
        
        # 检查数据质量
        if len(X_new) < 24:  # 至少需要24小时数据
            print("⚠ 数据量不足，跳过更新")
            return
        
        # 计算当前性能
        if 'lightgbm' in self.models and len(X_new) > 0:
            # 检查特征数量是否匹配
            model_features = self.models['lightgbm'].num_feature()
            input_features = X_new.shape[1]
            
            if model_features != input_features:
                print(f"[警告] ⚠ 特征数量不匹配！")
                print(f"  模型期望: {model_features} 个特征")
                print(f"  实际输入: {input_features} 个特征")
                print(f"  差异: {input_features - model_features} 个特征")
                print(f"  建议: 删除旧模型并重新训练")
                print(f"  跳过在线更新，避免预测错误")
                return False
            
            try:
                current_pred = self.models['lightgbm'].predict(X_new)
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
                
                print("✓ 模型性能稳定")
                return False  # 不需要重训练
                
            except Exception as e:
                print(f"[错误] 在线预测失败: {e}")
                print(f"建议: 删除旧模型并重新训练")
                return False
        
        print("✓ 模型性能稳定")
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
                
                print(f"✓ {name} 模型已保存")
            
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
                'saved_at': pd.Timestamp.now().isoformat()
            }
            with open(save_path / 'feature_info.pkl', 'wb') as f:
                pickle.dump(feature_info, f)
            print(f" 特征信息已保存: {len(feature_cols)} 个特征")

        # 保存两阶段模型组件
        if hasattr(self, 'two_stage_trained_') and self.two_stage_trained_:
            print(" 保存两阶段模型组件...")
            if hasattr(self, 'spike_classifier_') and self.spike_classifier_ is not None:
                self.spike_classifier_.save_model(str(save_path / 'spike_classifier.txt'))
                print("✓ 尖峰分类器已保存")
            if hasattr(self, 'spike_threshold_'):
                np.save(str(save_path / 'spike_threshold.npy'), np.array([self.spike_threshold_]))
                print(f"✓ 尖峰阈值 {self.spike_threshold_:.4f} 已保存")
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
            print("✓ 两阶段模型组件已保存")

        print(f"✓ 所有模型已保存到 {save_dir}")
    
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
                print("✓ LightGBM 模型加载成功")
                    
            # 2. 尝试加载XGBoost
            xgb_path = load_path / 'xgboost_model.bin'
            if xgb_path.exists():
                model = xgb.XGBRegressor()
                model.load_model(str(xgb_path))
                self.models['xgboost'] = model
                print("✓ XGBoost 模型加载成功")
                    
            # 3. 尝试加载CatBoost
            cat_path = load_path / 'catboost_model.bin'
            if cat_path.exists():
                model = CatBoostRegressor()
                model.load_model(str(cat_path))
                self.models['catboost'] = model
                print("✓ CatBoost 模型加载成功")
                    
            # 4. 尝试加载Scikit-Learn模型 (随机森林, 梯度提升等)
            for pkl_file in load_path.glob('*_model.pkl'):
                model_name = pkl_file.stem.replace('_model', '')
                self.models[model_name] = joblib.load(pkl_file)
                print(f"✓ {model_name} 模型加载成功")
                    
            # 加载配置
            config_path = load_path / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config.update(yaml.safe_load(f) or {})
                print("✓ 模型配置加载成功")
                
            # 5. 加载特征信息（新增）
            feature_info_path = load_path / 'feature_info.pkl'
            if feature_info_path.exists():
                with open(feature_info_path, 'rb') as f:
                    feature_info = pickle.load(f)
                self.training_feature_cols = feature_info.get('feature_cols', [])
                self.training_num_features = feature_info.get('num_features', 0)
                print(f"✓ 特征信息加载成功: {self.training_num_features} 个特征")
            else:
                print("[警告] 未找到特征信息文件，将无法验证特征匹配")
                self.training_feature_cols = []
                self.training_num_features = 0

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
                        print(f"✓ 两阶段模型加载成功 (尖峰阈值={self.spike_threshold_:.4f}, "
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
            print(f"⚠ 在 {start} 到 {end} 范围内没有找到数据！")
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

        preds, timestamps = [], []
        import sys, os

        for h in range(1, horizon_steps + 1):
            next_ts = raw_df['timestamp'].iloc[-1] + pd.Timedelta(minutes=15)

            # 构建新行：价格占位，天气从预报中获取或均值填充
            new_row = {'timestamp': next_ts, 'price': np.nan}

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

            new_row = pd.DataFrame([new_row])
            raw_df = pd.concat([raw_df, new_row], ignore_index=True)

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
            'temp_air', 'relative_humidity', 'precipitation', 'wind_speed',
            'ghi', 'dni', 'dhi', 'cloud_cover', 'pressure'
        ]]
        if weather_cols:
            print(f"[验证] ✓ 成功合并 {len(weather_cols)} 个天气特征: {weather_cols[:5]}...")
            print(f"[验证] 天气数据统计:")
            print(f"  - 温度范围: {data['temp_air'].min():.1f} ~ {data['temp_air'].max():.1f} °C")
            if 'ghi' in data.columns:
                print(f"  - 辐照度范围: {data['ghi'].min():.1f} ~ {data['ghi'].max():.1f} W/m²")
        else:
            print(f"[警告] ⚠ 天气数据合并失败！请检查:")
            print(f"  1. 经纬度是否正确: lat={lat}, lon={lon}")
            print(f"  2. 网络连接是否正常")
            print(f"  3. API token是否有效")
            print(f"  继续使用纯电价特征训练")

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
            'temp', 'humidity', 'precip', 'wind', 'ghi', 'dni', 'dhi', 
            'cloud', 'pressure', 'dew', 'vpd', 'et0'
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
        print(f"✓ 成功加载 {station_name} 的历史最佳模型。")
            
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
                print(f"⚠ 警报：{station_name} 模型在指定历史区间推断大幅度退步。需要执行重新训练流程。")
            else:
                print(f"✓ {station_name} 生产模型在推断该时间段时表现非常稳定。")
                
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

    # 2. 获取未来天气预报
    horizon = forecast_days * 96
    future_weather_15min = None
    if lat is not None and lon is not None and WEATHER_AVAILABLE:
        future_weather = fetch_forecast_weather(lat, lon, forecast_days=forecast_days)
        if future_weather is not None:
            future_weather_15min = resample_weather_to_15min(future_weather)
            print(f"[天气] 已获取未来天气预报，{len(future_weather_15min)}条15分钟记录")
        else:
            print("[天气] 未获取到未来天气预报，将使用历史天气均值填充")

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
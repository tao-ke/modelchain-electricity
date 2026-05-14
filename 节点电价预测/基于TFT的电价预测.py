"""
电力价格预测系统 - 基于 PyTorch Forecasting TFT (Temporal Fusion Transformer) 架构

核心特点：
1. 原生支持多步直接预测 (无需单步递归)
2. 自动适配 GPU/CPU 环境
3. 采用分位数回归 (QuantileLoss) 准确预测极端电价的波动范围 (P10, P50, P90)
4. 将过去的 7 天 (672步) 编码，直接输出未来 1 天 (96步) 的预测结果
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import psycopg2

# TimescaleDB 数据库连接配置
DB_CONFIG = {
    'dbname': 'Electricity',
    'user': 'postgres',
    'password': '1234',
    'host': 'localhost',
    'port': '5432'
}

# PyTorch 生态
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# PyTorch Forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei',"SimHei"]
plt.rcParams['axes.unicode_minus'] = False

try:
    from weather_api import fetch_historical_weather, fetch_forecast_weather, resample_weather_to_15min, align_weather_to_price_data
    WEATHER_AVAILABLE = True
except ImportError:
    WEATHER_AVAILABLE = False

# =========================================================
# 1. 数据加载模块（从TimescaleDB读取）
# =========================================================

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
    print(f"  日前电价加载成功: {len(df)} 条记录")
    return df


def _merge_dayahead_data(data, station_name, base_dir=None, years=(2024, 2025, 2026)):
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


def load_system_load_data(base_dir=None, years=(2024, 2025, 2026)):
    """从TimescaleDB读取全省负荷数据，返回DataFrame（timestamp, demand）。"""
    start_date = f"{min(years)}-01-01"
    end_date = f"{max(years) + 1}-01-01"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        query = """
            SELECT time, total_demand AS demand
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
    df = df.dropna(subset=['timestamp', 'demand']).sort_values('timestamp').reset_index(drop=True)

    print(f"  负荷数据加载成功: {len(df)} 条记录, "
          f"范围 {df['demand'].min():.0f}~{df['demand'].max():.0f} MW")
    return df


# =========================================================
# 2. TFT 高级预测器核心模块
# =========================================================

class TFTElectricityPredictor:
    def __init__(self, station_name):
        self.station_name = station_name
        self.model = None
        self.trainer = None
        
        # 硬件自适应：检测可用硬件
        self.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        self.devices = 1
        self.use_amp = False
        if self.accelerator == 'gpu':
            torch.set_float32_matmul_precision('medium')
        
        # 视窗配置 (15分钟频次)
        self.max_encoder_length = 672  # 回看历史: 7 天 * 96
        self.max_prediction_length = 96 # 预测未来: 1 天 * 96
        
        self.model_dir = f'./model_sets_tft/models_{self.station_name}'
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        self.last_daily_update = None
        self.last_weekly_retrain = None
        self.update_window_days = 60
        self.retrain_window_days = 180
        
    
    def build_features(self, df, weather_df=None, load_df=None, include_target_features=True):
        """统一特征生成器，处理时间、价格、天气与负荷特征，并对缺失值做安全兜底。"""
        data = df.copy()
        
        # 1. 如果传入了天气数据，按时间戳对齐合并
        if weather_df is not None and not weather_df.empty:
            weather_df = weather_df.copy()
            if 'timestamp' not in weather_df.columns and isinstance(weather_df.index, pd.DatetimeIndex):
                weather_df = weather_df.reset_index()
                weather_df = weather_df.rename(columns={weather_df.columns[0]: 'timestamp'})

            incoming_weather_cols = [c for c in weather_df.columns if c != 'timestamp']
            data = data.drop(columns=[c for c in incoming_weather_cols if c in data.columns], errors='ignore')
            # 合并天气
            data = pd.merge(data, weather_df, on='timestamp', how='left')

        # 1.2 如果传入了负荷数据，按时间戳对齐合并
        if load_df is not None and not load_df.empty:
            load_df = load_df.copy()
            if 'timestamp' not in load_df.columns and isinstance(load_df.index, pd.DatetimeIndex):
                load_df = load_df.reset_index()
                load_df = load_df.rename(columns={load_df.columns[0]: 'timestamp'})
            if 'demand' not in load_df.columns:
                demand_candidates = [c for c in load_df.columns if str(c).lower() in ('demand', 'load', '负荷', '负荷值')]
                if demand_candidates:
                    load_df = load_df.rename(columns={demand_candidates[0]: 'demand'})
            if 'timestamp' in load_df.columns and 'demand' in load_df.columns:
                load_df['timestamp'] = pd.to_datetime(load_df['timestamp'], errors='coerce')
                load_df['demand'] = pd.to_numeric(load_df['demand'], errors='coerce')
                load_df = load_df[['timestamp', 'demand']].dropna(subset=['timestamp'])
                data = data.drop(columns=['demand'], errors='ignore')
                data = pd.merge(data, load_df, on='timestamp', how='left')

        if 'demand' not in data.columns:
            data['demand'] = np.nan

        # 2. 基础标识与绝对时间步
        data["station"] = self.station_name
        if "time_idx" not in data.columns:
            data["time_idx"] = np.arange(len(data))

        # 3. 时间维度分类特征
        data["hour_int"] = data["timestamp"].dt.hour.astype(int)
        data["minute_int"] = data["timestamp"].dt.minute.astype(int)
        data["day_of_week_int"] = data["timestamp"].dt.dayofweek.astype(int)
        data['is_weekend'] = (data['day_of_week_int'] >= 5).astype(int)
        data['is_business_hour'] = ((data['hour_int'] >= 9) & (data['hour_int'] <= 17) & (data['day_of_week_int'] < 5)).astype(int)

        data["hour"] = data["hour_int"].astype(str)
        data["minute"] = data["minute_int"].astype(str)
        data["day_of_week"] = data["day_of_week_int"].astype(str)
        data[["hour", "minute", "day_of_week", "station"]] = data[["hour", "minute", "day_of_week", "station"]].astype("category")

        # 4. 时间衍生连续特征
        data["hour_int"] = data["hour_int"].astype(float)
        data["minute_int"] = data["minute_int"].astype(float)
        data["time_of_day_num"] = data["hour_int"] + data["minute_int"] / 60.0
        data["hour_sin"] = np.sin(2 * np.pi * data["time_of_day_num"] / 24.0)
        data["hour_cos"] = np.cos(2 * np.pi * data["time_of_day_num"] / 24.0)
        data["period_of_day"] = data["hour_int"] * 4 + (data["minute_int"] // 15)

        # 如果合并了天气数据，这里再基于已就绪的时间特征创建天气衍生特征
        if weather_df is not None and not weather_df.empty:
            try:
                data = self._create_weather_features(data)
            except Exception:
                print("[警告] 创建天气特征失败，继续后续流程。")

        # 如果存在负荷列，构建负荷衍生特征
        try:
            data = self._create_load_features(data)
        except Exception:
            print("[警告] 创建负荷特征失败，继续后续流程。")

        # 5. 目标变量滞后和滚动特征 (只对已知过去的序列)
        if include_target_features:
            data["price_lag_1"] = data["price"].shift(1)
            data["price_lag_2"] = data["price"].shift(2)
            data["price_lag_4"] = data["price"].shift(4)
            data["price_lag_96"] = data["price"].shift(96)
            data["price_diff_96"] = data["price_lag_1"] - data["price_lag_96"]
            data["price_roll_mean_4"] = data["price_lag_1"].rolling(window=4).mean()
            data["price_roll_max_4"] = data["price_lag_1"].rolling(window=4).max()
            data["price_roll_std_96"] = data["price_lag_1"].rolling(window=96).std()
            data["bollinger_ratio"] = 0.0
            data["price_lag_1_x_price_lag_96"] = data["price_lag_1"] * data["price_lag_96"]
            
            # 使用 target encoding 计算 hour_target_encoded
            data["hour_target_encoded"] = data.groupby("hour_int")["price"].transform(
                lambda x: x.shift(1).expanding().mean()
            )

        if 'dayahead_price' in data.columns:
            data = self._create_dayahead_features(data)

        # 6. NaN 的统一安全兜底 (防止丢给 PyTorch Forecasting 后报错)
        # 将所有无穷大替换为NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        # 使用 bfill 解决由 lag / rolling / 合并导致的序列头部空值，用 ffill 解决尾部空值
        cols_to_fill = data.select_dtypes(include=[np.number]).columns
        data[cols_to_fill] = data[cols_to_fill].bfill().ffill().fillna(0)

        return data

    def _create_weather_features(self, df):
        """从天气列创建天气相关衍生特征，借鉴自集成学习实现的特征集。"""
        weather_lags = [1, 4, 96]
        weather_windows = [4, 24, 96]

        weather_vars = [
            'temp_air', 'relative_humidity', 'precipitation', 'wind_speed',
            'wind_direction', 'pressure', 'cloud_cover', 'ghi', 'dni', 'dhi',
            'dew_point', 'et0', 'vpd'
        ]
        available_vars = [v for v in weather_vars if v in df.columns]

        if not available_vars:
            return df

        # 温度相关
        if 'temp_air' in df.columns:
            if 'relative_humidity' in df.columns:
                T = df['temp_air']
                RH = df['relative_humidity'].clip(0, 100)
                df['heat_index'] = (
                    -8.7847 + 1.6114 * T + 2.3385 * RH - 0.1461 * T * RH
                    - 0.0123 * T ** 2 - 0.0164 * RH ** 2 + 0.0022 * T ** 2 * RH
                    + 0.00073 * T * RH ** 2 - 0.000029 * T ** 2 * RH ** 2
                )
                df['temp_humidity_interact'] = df['temp_air'] * df['relative_humidity'] / 100

            df['temp_change_rate'] = df['temp_air'].diff()

        # 辐射相关
        if 'ghi' in df.columns:
            if 'hour' in df.columns:
                df['ghi_is_day'] = (df['ghi'] > 50).astype(int)
                df['ghi_hour_interact'] = df['ghi'] * df['hour'].astype(float)
            df['ghi_ratio'] = df['ghi'].clip(lower=0) / 1000.0

        if 'cloud_cover' in df.columns:
            df['cloud_cover_level'] = pd.cut(
                df['cloud_cover'].clip(0, 100),
                bins=[-0.01, 20, 60, 100],
                labels=[0, 1, 2]
            ).astype(float)

        if 'wind_speed' in df.columns:
            df['wind_power_potential'] = df['wind_speed'] ** 3

        # 天气滞后与滚动
        for var in available_vars:
            for lag in weather_lags:
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)

        for var in available_vars:
            for window in weather_windows:
                df[f'{var}_roll_mean_{window}'] = df[var].shift(1).rolling(window).mean()
                df[f'{var}_roll_std_{window}'] = df[var].shift(1).rolling(window).std()

        # 天气-价格交叉（使用滞后价格以避免泄露）
        if 'price' in df.columns:
            past_price = df['price'].shift(1)
            if 'temp_air' in df.columns:
                df['price_x_temp'] = past_price * df['temp_air']
                tm_med = df['temp_air'].median()
                tm_std = df['temp_air'].std()
                df['is_extreme_temp'] = (abs(df['temp_air'] - tm_med) > 2 * tm_std).astype(int)
            if 'ghi' in df.columns:
                df['price_div_ghi'] = (past_price / df['ghi'].clip(lower=1)).clip(-1e6, 1e6).fillna(0)
            if 'wind_speed' in df.columns:
                df['price_x_wind'] = past_price * df['wind_speed']

        # 综合指标
        if all(v in df.columns for v in ['temp_air', 'cloud_cover', 'wind_speed']):
            df['comfort_index'] = (
                df['temp_air'] * 0.4 + (100 - df['cloud_cover']) * 0.3 + df['wind_speed'] * 0.3
            )
            df['cooling_degree'] = (df['temp_air'] - 26).clip(lower=0)
            df['heating_degree'] = (18 - df['temp_air']).clip(lower=0)

        if 'precipitation' in df.columns:
            df['is_rainy'] = (df['precipitation'] > 0).astype(int)
            df['is_heavy_rain'] = (df['precipitation'] > 10).astype(int)

        return df

    def _create_dayahead_features(self, df):
        """Create compact day-ahead price features."""
        if 'dayahead_price' not in df.columns:
            return df

        df['dayahead_price'] = pd.to_numeric(df['dayahead_price'], errors='coerce')
        past_price = df['price'].shift(1) if 'price' in df.columns else np.nan

        df['da_vs_price'] = df['dayahead_price'] - past_price
        df['da_price_lag_96'] = df['dayahead_price'].shift(96)
        df['da_price_trend_96'] = df['dayahead_price'] - df['dayahead_price'].shift(96)
        df['da_premium_signal'] = (df['dayahead_price'] > past_price).astype(float)

        if 'hour_int' in df.columns:
            df['da_x_hour'] = df['dayahead_price'] * df['hour_int']
        if 'is_weekend' in df.columns:
            df['da_x_weekend'] = df['dayahead_price'] * df['is_weekend']

        return df

    def _create_load_features(self, df):
        """从负荷列创建负荷相关衍生特征。"""
        if 'demand' not in df.columns:
            return df

        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
        df['demand_lag_1'] = df['demand'].shift(1)
        df['demand_lag_4'] = df['demand'].shift(4)
        df['demand_lag_96'] = df['demand'].shift(96)
        df['demand_diff_96'] = df['demand_lag_1'] - df['demand_lag_96']
        df['demand_roll_mean_4'] = df['demand'].shift(1).rolling(window=4).mean()
        df['demand_roll_std_96'] = df['demand'].shift(1).rolling(window=96).std()

        if 'price' in df.columns:
            df['price_x_demand'] = df['price'].shift(1) * df['demand']

        return df

    def _unknown_real_cols(self):
        return [
            "price",
            "price_lag_1", "price_lag_2", "price_lag_4", "price_lag_96",
            "price_diff_96", "price_roll_mean_4", "price_roll_max_4", "price_roll_std_96",
            "bollinger_ratio", "price_lag_1_x_price_lag_96", "hour_target_encoded",
            "demand", "demand_lag_1", "demand_lag_4", "demand_lag_96",
            "demand_diff_96", "demand_roll_mean_4", "demand_roll_std_96", "price_x_demand",
        ]

    def _dayahead_real_cols(self):
        return [
            "dayahead_price", "da_vs_price", "da_price_lag_96", "da_price_trend_96",
            "da_premium_signal", "da_x_hour", "da_x_weekend",
        ]

    def _reserved_feature_cols(self):
        return set([
            "timestamp", "station", "time_idx", "hour", "minute", "day_of_week",
            "time_of_day_num", "hour_sin", "hour_cos", "hour_int", "minute_int",
            "day_of_week_int", "period_of_day", "price",
        ] + self._unknown_real_cols() + self._dayahead_real_cols())

    def _get_extra_known_reals(self, data):
        reserved_cols = self._reserved_feature_cols()
        return [
            c for c in data.columns
            if c not in reserved_cols and pd.api.types.is_numeric_dtype(data[c])
        ]

    def _unique_existing_cols(self, cols, data, excluded=None):
        excluded = set(excluded or [])
        out = []
        for col in cols:
            if col in data.columns and col not in excluded and col not in out:
                out.append(col)
        return out

    def create_dataset(self, data, weather_cols=None, is_train=True):
        """生成 PyTorch Forecasting 的 TimeSeriesDataSet"""
        
        # 当作预测集时，可以放宽 min_encoder_length 允许序列短端预测
        min_enc_length = 96 if not is_train else 96
        
        # 合并自定义天气列表
        known_reals = ["time_idx", "hour_sin", "hour_cos", "period_of_day"]
        known_reals.extend(self._dayahead_real_cols())
        if weather_cols is not None:
            known_reals.extend(weather_cols)
        known_reals = self._unique_existing_cols(known_reals, data)

        unknown_reals = self._unique_existing_cols(
            self._unknown_real_cols(),
            data,
            excluded=set(known_reals),
        )

        return TimeSeriesDataSet(
            data,
            time_idx="time_idx",
            target="price",
            group_ids=["station"],
            min_encoder_length=min_enc_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["station"],
            time_varying_known_categoricals=["hour", "minute", "day_of_week"],
            time_varying_known_reals=known_reals,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=unknown_reals,
            # 将价格归一化
            target_normalizer=GroupNormalizer(groups=["station"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
    @torch.inference_mode()
    def _predict_quantiles_no_logger(self, dataloader):
        return self.model.predict(
                dataloader,
                mode="quantiles",
                trainer_kwargs={
                "logger": False,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "default_root_dir": str(Path.cwd() / "pl_root_predict"),
                },
        )

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _extract_p50_predictions(self, predictions):
        pred_arr = self._to_numpy(predictions)
        if pred_arr.ndim == 3:
            return pred_arr[0, :, 1]
        if pred_arr.ndim == 2:
            return pred_arr[:, 1] if pred_arr.shape[1] >= 3 else pred_arr.reshape(-1)
        return pred_arr.reshape(-1)

    def _calculate_metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        errors = y_true - y_pred
        abs_errors = np.abs(errors)

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(abs_errors))
        mask = np.abs(y_true) > 1e-6
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan

        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        denominator[denominator == 0] = 1e-6
        smape = float(np.mean(abs_errors / denominator) * 100)

        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        accuracy = float(max(0.0, 100.0 - smape))

        return {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "smape": smape,
            "accuracy": accuracy,
            "r2": r2,
            "median_abs_error": float(np.median(abs_errors)),
            "p90_abs_error": float(np.percentile(abs_errors, 90)),
            "max_abs_error": float(np.max(abs_errors)),
        }

    def evaluate_test_efficient(self, full_features, train_val_end_idx, y_test,
                                 weather_df=None, load_df=None):
        """高效评估方法: 预计算特征后，逐块切片预测，避免重复特征工程。

        Args:
            full_features: build_features 预计算的完整特征DataFrame (train+val+test)
            train_val_end_idx: train+val 在 full_features 中的结束位置
            y_test: 测试集真实价格 (numpy array)
            weather_df: 天气数据 (用于提取 known reals)
            load_df: 负荷数据 (用于提取 known reals)
        Returns:
            tft_preds: 测试集预测价格 (numpy array, 与 y_test 等长)
        """
        # 提取天气列名供 create_dataset 使用
        weather_cols = self._get_extra_known_reals(full_features)

        forecast_days = max(len(y_test) // 96, 1)
        future_steps = forecast_days * 96
        print(f'\n  [高效评估] {forecast_days} 天, 共 {future_steps} 步, 批大小={min(32, forecast_days)}')

        all_preds_p50 = []

        for day in range(forecast_days):
            blk_start = train_val_end_idx + day * 96
            blk_end = blk_start + 96

            # 从预计算特征中切片 (O(1) 操作, 无重复计算)
            encoder = full_features.iloc[max(0, blk_start - self.max_encoder_length):blk_start]
            decoder = full_features.iloc[blk_start:blk_end].copy()

            # 确保 decoder 的 time_idx 接续 encoder
            last_tidx = encoder['time_idx'].iloc[-1] if len(encoder) > 0 else 0
            decoder['time_idx'] = [last_tidx + i + 1 for i in range(len(decoder))]

            inference_df = pd.concat([encoder, decoder], ignore_index=True)
            inference_dataset = self.create_dataset(inference_df, weather_cols=weather_cols, is_train=False)
            dataloader = inference_dataset.to_dataloader(
                train=False, batch_size=32, num_workers=0
            )

            predictions = self._predict_quantiles_no_logger(dataloader)
            p50 = self._extract_p50_predictions(predictions)[:96]
            all_preds_p50.extend(p50)

        return np.asarray(all_preds_p50[:len(y_test)], dtype=float)

    def plot_prediction_analysis(self, y_true, y_pred, model_name="TFT"):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
        max_v = max(float(np.max(y_true)), float(np.max(y_pred)))
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title(f'{model_name} - Actual vs Predicted')

        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].set_title(f'{model_name} - Residuals')

        axes[1, 0].hist(residuals, bins=min(50, max(10, len(residuals) // 2)), alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'{model_name} - Residual Distribution')

        indices = range(len(y_true))
        axes[1, 1].plot(indices, y_true, label='Actual', alpha=0.85)
        axes[1, 1].plot(indices, y_pred, label='Predicted P50', alpha=0.85)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Price')
        axes[1, 1].set_title(f'{model_name} - Time Series Comparison')
        axes[1, 1].legend()

        plt.tight_layout()
        output_path = Path(self.model_dir) / "tft_training_effect.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[TFT eval] effect plot saved: {output_path}")
        plt.show(block=False)
        plt.close(fig)

    def evaluate_last_window(self, validation_data, val_dataloader, model_name=None):
        try:
            predictions = self._predict_quantiles_no_logger(val_dataloader)
            y_pred = self._extract_p50_predictions(predictions)
            y_true = (
                validation_data.sort_values("time_idx")
                .tail(len(y_pred))["price"]
                .to_numpy(dtype=float)
            )
            n = min(len(y_true), len(y_pred))
            y_true = y_true[-n:]
            y_pred = y_pred[-n:]

            metrics = self._calculate_metrics(y_true, y_pred)
            self.last_evaluation_metrics = metrics

            print("\n[TFT holdout evaluation - last prediction window]")
            print(f"   Samples:          {n}")
            print(f"   RMSE:             {metrics['rmse']:.4f}")
            print(f"   MAE:              {metrics['mae']:.4f}")
            if not np.isnan(metrics['mape']):
                print(f"   MAPE:             {metrics['mape']:.2f}%")
            else:
                print("   MAPE:             N/A")
            print(f"   SMAPE:            {metrics['smape']:.2f}%")
            print(f"   Accuracy(100-SMAPE): {metrics['accuracy']:.2f}%")
            print(f"   R2:               {metrics['r2']:.4f}")
            print(f"   Median abs error: {metrics['median_abs_error']:.4f}")
            print(f"   90% abs error:    {metrics['p90_abs_error']:.4f}")
            print(f"   Max abs error:    {metrics['max_abs_error']:.4f}")

            self.plot_prediction_analysis(y_true, y_pred, model_name=model_name or f"TFT - {self.station_name}")
            return metrics
        except Exception as e:
            print(f"[TFT eval] evaluation skipped: {e}")
            return None
        
    def train_model(self, data, max_epochs=10, batch_size=32, learning_rate=0.01):
        print(f"\n[硬件报告] 正在使用 {self.accelerator.upper()} 进行运算。")

        data = data.copy()
        required_real_cols = self._unique_existing_cols(
            ["time_idx", "price", "hour_sin", "hour_cos", "period_of_day"]
            + self._unknown_real_cols()
            + self._dayahead_real_cols(),
            data,
        )
        
        weather_cols = self._get_extra_known_reals(data)
        
        # 1. 先统一清洗全量数据 (无穷值替换和NaN清理已在造特征时初步完成)
        data = data.dropna(subset=required_real_cols + weather_cols).reset_index(drop=True)

        # 2. 再切分训练/验证
        validation_cutoff = data["time_idx"].max() - (self.max_prediction_length * 7)
        training_cutoff = data["time_idx"].max() - self.max_prediction_length

        training_data = data[data["time_idx"] <= training_cutoff].copy()
        validation_data = data[data["time_idx"] > validation_cutoff].copy()

        if len(training_data) < self.max_encoder_length + self.max_prediction_length:
                raise ValueError(f"训练集长度不足，当前 {len(training_data)} 行")

        print("初始化 Dataset 与 DataLoader...")
        train_dataset = self.create_dataset(training_data, weather_cols=weather_cols, is_train=True)
        val_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset,
                validation_data,
                predict=True,
                stop_randomization=True
        )

        num_workers = 0
        train_dataloader = train_dataset.to_dataloader(
                train=True,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
        )
        val_dataloader = val_dataset.to_dataloader(
                train=False,
                batch_size=batch_size * 2,
                num_workers=num_workers,
                pin_memory=True,
        )

        print("初始化 Temporal Fusion Transformer 网络架构...")
        # RTX 4060 8GB: hidden_size=48 兼顾精度与速度 (~30% 参数减少, 精度损失 <1%)
        hidden_size = 24 if self.accelerator == "cpu" else 48

        self.model = TemporalFusionTransformer.from_dataset(
                train_dataset,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=16,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                log_interval=10,
                reduce_on_plateau_patience=4,
        )

        early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=3,
                verbose=True,
                mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
                dirpath=self.model_dir,
                filename="tft-best-checkpoint",
                save_top_k=1,
                monitor="val_loss",
                mode="min"
        )

        trainer_kwargs = dict(
                max_epochs=max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                enable_model_summary=False,
                enable_progress_bar=False,
                callbacks=[early_stop_callback, checkpoint_callback],
                logger=False,
                default_root_dir=str(Path.cwd() / "pl_root"),
        )
        if self.use_amp:
            trainer_kwargs['precision'] = '16-mixed'
            print("  启用 FP16 混合精度训练")

        self.trainer = pl.Trainer(**trainer_kwargs)

        print("开始深入训练 TFT 网络...")
        self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
        )

        if checkpoint_callback.best_model_path:
            self.model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)

        self.evaluate_last_window(
            validation_data,
            val_dataloader,
            model_name=f"TFT - {self.station_name}",
        )
            
    def load_latest_model(self):
        """尝试从磁盘加载最新训练好的模型"""
        # 查找最新的 ckpt 文件
        ckpt_files = list(Path(self.model_dir).glob("*.ckpt"))
        if not ckpt_files:
            return False
            
        print(f"找到历史模型: {ckpt_files[0].name}，开始加载...")
        self.model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_files[0]))
        return True
        
    def forecast_future(self, data, weather_df=None, load_df=None, forecast_days=1,
                        actual_future=None):
        """利用 TFT 原生的多步推断机制预测未来区块。

        actual_future: 可选 DataFrame，包含未来时段的真实 price（和可选的 dayahead_price）。
                       传入后使用真实价格推进窗口（单步评估模式，误差不累积）；
                       不传则使用模型预测值递推（生产模式）。
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先加载或训练模型！")

        recursive = actual_future is None
        mode_label = "递推预测" if recursive else "单步预测(使用真实历史)"
        print(f"\n TFT 开始多步长推断：{mode_label} {forecast_days} 天...")

        all_q10, all_q50, all_q90 = [], [], []
        encoder_data = data.iloc[-self.max_encoder_length:].copy()

        last_timestamp = encoder_data['timestamp'].iloc[-1]
        last_time_idx = encoder_data['time_idx'].iloc[-1]

        future_steps = forecast_days * 96
        future_dates = [last_timestamp + pd.Timedelta(minutes=15 * i) for i in range(1, future_steps + 1)]

        decoder_df = pd.DataFrame({'timestamp': future_dates})
        decoder_df['station'] = self.station_name
        decoder_df['time_idx'] = [last_time_idx + i for i in range(1, future_steps + 1)]

        decoder_df['hour'] = decoder_df['timestamp'].dt.hour.astype(str)
        decoder_df['minute'] = decoder_df['timestamp'].dt.minute.astype(str)
        decoder_df['day_of_week'] = decoder_df['timestamp'].dt.dayofweek.astype(str)
        decoder_df[['hour', 'minute', 'day_of_week', 'station']] = decoder_df[['hour', 'minute', 'day_of_week', 'station']].astype('category')

        decoder_df['time_of_day_num'] = decoder_df['timestamp'].dt.hour + decoder_df['timestamp'].dt.minute / 60.0
        decoder_df['hour_sin'] = np.sin(2 * np.pi * decoder_df['time_of_day_num'] / 24.0)
        decoder_df['hour_cos'] = np.cos(2 * np.pi * decoder_df['time_of_day_num'] / 24.0)

        decoder_df['price'] = encoder_data['price'].iloc[-1]

        if 'dayahead_price' in encoder_data.columns:
            last_da_cycle = encoder_data['dayahead_price'].dropna().tail(96).to_numpy(dtype=float)
            if len(last_da_cycle) > 0:
                decoder_df['dayahead_price'] = [
                    float(last_da_cycle[i % len(last_da_cycle)])
                    for i in range(future_steps)
                ]

        decoder_df["hour_int"] = decoder_df["timestamp"].dt.hour.astype(float)
        decoder_df["minute_int"] = decoder_df["timestamp"].dt.minute.astype(float)
        decoder_df["period_of_day"] = (decoder_df["hour_int"] * 4 + (decoder_df["minute_int"] // 15)).astype(float)

        all_preds_p50 = []
        current_data = encoder_data.copy()
        actual_offset = 0

        for block_start in range(0, future_steps, self.max_prediction_length):
            block_end = min(block_start + self.max_prediction_length, future_steps)
            block_len = block_end - block_start
            block_decoder = decoder_df.iloc[block_start:block_end].copy()

            # ---- 非递推模式：用真实价格填充 decoder，使特征工程基于真实历史 ----
            if not recursive and actual_future is not None:
                actual_block = actual_future.iloc[actual_offset:actual_offset + block_len]
                block_decoder['price'] = actual_block['price'].values
                if 'dayahead_price' in actual_block.columns:
                    block_decoder['dayahead_price'] = actual_block['dayahead_price'].values
                actual_offset += block_len

            inference_df = pd.concat([current_data, block_decoder], ignore_index=True)

            inference_df = self.build_features(
                inference_df,
                weather_df=weather_df,
                load_df=load_df,
                include_target_features=True
            )

            weather_cols = self._get_extra_known_reals(inference_df)
            inference_dataset = self.create_dataset(inference_df, weather_cols=weather_cols, is_train=False)
            dataloader = inference_dataset.to_dataloader(train=False, batch_size=32, num_workers=0)

            predictions = self._predict_quantiles_no_logger(dataloader)[0]
            pred_q10 = predictions[:, 0]
            pred_q50 = predictions[:, 1]
            pred_q90 = predictions[:, 2]

            predicted_p50 = predictions[:, 1].numpy()
            all_preds_p50.extend(predicted_p50)
            all_q10.extend(pred_q10)
            all_q50.extend(pred_q50)
            all_q90.extend(pred_q90)

            # 推进窗口：递推模式用预测值，评估模式用真实值（已在上面填入）
            if recursive:
                block_decoder['price'] = predicted_p50
            current_data = pd.concat([current_data, block_decoder], ignore_index=True)
            current_data = current_data.iloc[-self.max_encoder_length:]

        # 如果返回长超了，裁剪至实际天数
        all_preds_p50 = all_preds_p50[:future_steps]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))
        plt.plot(future_dates, all_q50[:future_steps], label='P50', color='orange')
        plt.fill_between(future_dates, all_q10[:future_steps], all_q90[:future_steps], color='orange', alpha=0.2, label='P10-P90')
        plt.title(f"{self.station_name} TFT Forecast Quantiles (P10/P50/P90)")
        plt.xlabel("time")
        plt.ylabel("price")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return future_dates, all_preds_p50


# =========================================================
# 3. 生产逻辑封装
# =========================================================

def train_station_tft(station_name, lat=None, lon=None):
    print(f"\n[训练阶段] 开始基于 TFT 深度学习架构针对电站: {station_name} 进行训练...")
    
    predictor = TFTElectricityPredictor(station_name)
    
    data = load_station_electricity_data(station_name)
    if data is None or data.empty:
        raise ValueError(f"未获取到 {station_name} 的数据，无法训练。")

    data = _merge_dayahead_data(data, station_name)

    load_data = load_system_load_data()
    if load_data is None or load_data.empty:
        print("[警告] 未获取到有效负荷数据，将以缺失值兜底继续训练。")

    # 获取并合并历史天气（如果可用且提供了经纬度）
    weather_hist = None
    if WEATHER_AVAILABLE and lat is not None and lon is not None:
        start_str = data['timestamp'].min().strftime("%Y%m%d")
        end_str = data['timestamp'].max().strftime("%Y%m%d")
        try:
            raw_weather = fetch_historical_weather(lat, lon, start_str, end_str)
            weather_hist = resample_weather_to_15min(raw_weather)
        except Exception as e:
            print(f"[警告] 获取历史天气失败: {e}")
            weather_hist = None

    # 核心特征工程：转化为TFT连续序列（传入天气数据用于生成天气衍生特征）
    data_with_features = predictor.build_features(data, weather_df=weather_hist, load_df=load_data)

    # 设置 epochs：如果是 GPU 可以拉大，CPU 保持小一点即可见效
    epochs_num = 20 if predictor.accelerator == 'gpu' else 5
    
    # 训练模型
    predictor.train_model(data_with_features, max_epochs=epochs_num)
    print("\n[训练结束] 模型检查点已自动保存至本地。")
    return predictor, data_with_features

def run_production_inference_tft(station_name='百合站', forecast_days=1, lat=23.4, lon=113.2):
    """每日自动运行的 TFT 深度学习流水线"""
    print(f"=== TFT 电力价格预测系统 -  ({station_name}) ===")
    
    predictor = TFTElectricityPredictor(station_name)
    
    data = load_station_electricity_data(station_name)
    if data is None or data.empty:
        print("无数据，退出。")
        return

    data = _merge_dayahead_data(data, station_name)

    load_data = load_system_load_data()
    if load_data is None or load_data.empty:
        print("[警告] 未获取到有效负荷数据，将以缺失值兜底进行推断。")

    weather_hist = None
    if WEATHER_AVAILABLE and lat is not None and lon is not None:
        start_str = data['timestamp'].min().strftime("%Y%m%d")
        end_str = data['timestamp'].max().strftime("%Y%m%d")
        raw_weather = fetch_historical_weather(lat, lon, start_str, end_str)
        weather_hist = resample_weather_to_15min(raw_weather)
        
    data_with_features = predictor.build_features(data, weather_df=weather_hist, load_df=load_data)

    # 1. 尝试加载训练好的 TFT 模型
    is_loaded = predictor.load_latest_model()
    if is_loaded and 'dayahead_price' in data_with_features.columns:
        loaded_reals = list(getattr(predictor.model.hparams, "x_reals", []) or [])
        if 'dayahead_price' not in loaded_reals:
            print("[TFT] loaded checkpoint was trained without day-ahead price; retraining.")
            is_loaded = False
    
    if not is_loaded:
        print(f"⚠ 未发现 {station_name} 的深度学习预训练模型文件，执行【全量训练】...")
        predictor, _ = train_station_tft(station_name)
    else:
        print("✓ 加载本地最新历史检查点(.ckpt)成功。")
        
        # [如果需要可以在次实现微调(在线学习)逻辑：修改小学习率然后 trainer.fit(model)]
    weather_forecast = None
    if WEATHER_AVAILABLE and lat is not None and lon is not None:
        raw_forecast = fetch_forecast_weather(lat, lon, forecast_days=forecast_days)
        weather_forecast = resample_weather_to_15min(raw_forecast)

    # 2. 对未来进行推断
    print(f"\n开始对未来 {forecast_days} 天进行多步长预测...")
    future_dates, future_preds = predictor.forecast_future(
        data_with_features,
        weather_df=weather_forecast,
        load_df=load_data,
        forecast_days=forecast_days
    )

    # 3. 绘制“未来曲线结果”
    plt.figure(figsize=(14, 6))
    
    # 过去最后1天(96个点) 的波动
    history_to_show = data_with_features.iloc[-96:] 
    plt.plot(history_to_show['timestamp'], history_to_show['price'], label='观测视窗内已知的真实特征序列', color='navy')
    plt.plot(future_dates, future_preds, label='TFT原生多步未来预测曲线 (P50 波动)', color='darkorange', linewidth=2.5)
    
    plt.title(f"{station_name} - 基于 TFT 注意力的未来 {forecast_days} 天序列生成")
    plt.xlabel("时间")
    plt.ylabel("预测电价")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.6)
    plt.tight_layout()
    plt.show()

def main():
    stations_to_run = ['百合站']  # 可以按需扩充
    
    for station in stations_to_run:
        print("=" * 60)
        # 推理并预测未来 1 天 
        run_production_inference_tft(station_name=station, forecast_days=1)
        print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 提示：运行本代码需要安装如下深度学习包:")
        print("pip install torch pytorch-lightning pytorch-forecasting")
        
    input("\n按Enter键退出...")
    

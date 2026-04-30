"""
天气数据获取模块
- 历史天气：Open-Meteo Archive API (免费无需token)
- 未来天气：Open-Meteo Forecast API (免费无需token)
- 自动对齐到15分钟分辨率，与电价数据匹配
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

# ============================================
# Open-Meteo API 接口 (免费，无需token)
# ============================================

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# 请求包含的气象变量
OPEN_METEO_VARIABLES = [
    "temperature_2m",            # 2米气温 (C)
    "relative_humidity_2m",      # 2米相对湿度 (%)
    "precipitation",             # 降水量 (mm)
    "wind_speed_10m",            # 10米风速 (m/s)
    "wind_direction_10m",        # 10米风向
    "surface_pressure",          # 地面气压 (hPa)
    "cloud_cover",               # 总云量 (%)
    "shortwave_radiation",       # 短波辐射 (W/m2)
    "direct_radiation",          # 直接辐射 (W/m2)
    "diffuse_radiation",         # 散射辐射 (W/m2)
    "dew_point_2m",              # 2米露点温度 (C)
    "et0_fao_evapotranspiration",  # ET0蒸发蒸腾量 (mm)
    "vapour_pressure_deficit",   # 蒸汽压亏缺 (kPa)
]


def fetch_historical_weather(lat, lon, start_date, end_date):
    """从Open-Meteo Archive获取历史天气数据

    Args:
        lat, lon: 经纬度
        start_date, end_date: 日期范围 (datetime 或 str 'YYYY-MM-DD' 或 'YYYYMMDD')
    Returns: DataFrame(小时分辨率, 含时间索引) 或 None
    """
    if isinstance(start_date, datetime):
        start_date = start_date.strftime("%Y-%m-%d")
    elif isinstance(start_date, str) and "-" not in start_date:
        start_date = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
    if isinstance(end_date, datetime):
        end_date = end_date.strftime("%Y-%m-%d")
    elif isinstance(end_date, str) and "-" not in end_date:
        end_date = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPEN_METEO_VARIABLES),
        "timezone": "Asia/Shanghai",
        "wind_speed_unit": "ms",
    }

    try:
        print(f"[Open-Meteo Archive] 获取 {start_date} ~ {end_date} 历史天气...")
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)

        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            if not hourly:
                print("[Open-Meteo Archive] 无数据返回")
                return None

            time_list = pd.to_datetime(hourly["time"])
            df = pd.DataFrame(index=time_list)

            var_name_map = {
                "temperature_2m": "temp_air",
                "relative_humidity_2m": "relative_humidity",
                "precipitation": "precipitation",
                "wind_speed_10m": "wind_speed",
                "wind_direction_10m": "wind_direction",
                "surface_pressure": "pressure",
                "cloud_cover": "cloud_cover",
                "shortwave_radiation": "ghi",
                "direct_radiation": "dni",
                "diffuse_radiation": "dhi",
                "dew_point_2m": "dew_point",
                "et0_fao_evapotranspiration": "et0",
                "vapour_pressure_deficit": "vpd",
            }

            for api_name, local_name in var_name_map.items():
                if api_name in hourly:
                    df[local_name] = pd.to_numeric(hourly[api_name], errors="coerce")

            df.index.name = "time"
            df = _standardize_weather_df(df)
            print(f"[Open-Meteo Archive] 获取成功: {len(df)} 条记录, "
                  f"{df.index[0]} ~ {df.index[-1]}")
            return df

        else:
            print(f"[Open-Meteo Archive] HTTP {resp.status_code}: {resp.text[:300]}")
            return None

    except Exception as e:
        print(f"[Open-Meteo Archive] 获取失败: {e}")
        return None


def fetch_forecast_weather(lat, lon, forecast_days=7):
    """从Open-Meteo获取未来天气预报

    Args:
        lat, lon: 经纬度
        forecast_days: 预报天数（默认7天，最多16天）
    Returns: DataFrame(小时分辨率) 或 None
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(OPEN_METEO_VARIABLES),
        "timezone": "Asia/Shanghai",
        "wind_speed_unit": "ms",
        "forecast_days": min(forecast_days, 16),
    }

    try:
        print(f"[Open-Meteo Forecast] 获取未来 {forecast_days} 天预报...")
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            hourly = data.get("hourly", {})
            if not hourly:
                print("[Open-Meteo Forecast] 无数据返回")
                return None

            time_list = pd.to_datetime(hourly["time"])
            df = pd.DataFrame(index=time_list)

            var_name_map = {
                "temperature_2m": "temp_air",
                "relative_humidity_2m": "relative_humidity",
                "precipitation": "precipitation",
                "wind_speed_10m": "wind_speed",
                "wind_direction_10m": "wind_direction",
                "surface_pressure": "pressure",
                "cloud_cover": "cloud_cover",
                "shortwave_radiation": "ghi",
                "direct_radiation": "dni",
                "diffuse_radiation": "dhi",
                "dew_point_2m": "dew_point",
                "et0_fao_evapotranspiration": "et0",
                "vapour_pressure_deficit": "vpd",
            }

            for api_name, local_name in var_name_map.items():
                if api_name in hourly:
                    df[local_name] = pd.to_numeric(hourly[api_name], errors="coerce")

            df.index.name = "time"
            df = _standardize_weather_df(df)
            print(f"[Open-Meteo Forecast] 获取成功: {len(df)} 条记录, "
                  f"{df.index[0]} ~ {df.index[-1]}")
            return df

        else:
            print(f"[Open-Meteo Forecast] HTTP {resp.status_code}: {resp.text[:300]}")
            return None

    except Exception as e:
        print(f"[Open-Meteo Forecast] 获取失败: {e}")
        return None


def _standardize_weather_df(df):
    """标准化天气DataFrame：确保有统一列名，填充缺失列"""
    if df is None or len(df) == 0:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["time", "timestamp", "datetime"]:
            if col in df.columns:
                df["time"] = pd.to_datetime(df[col])
                df = df.set_index("time")
                break

    df = df.sort_index()

    default_fill = {
        "temp_air": 20.0, "relative_humidity": 60.0,
        "precipitation": 0.0, "wind_speed": 3.0,
        "wind_direction": 180.0, "pressure": 1013.0,
        "cloud_cover": 50.0,
        "ghi": 0.0, "dni": 0.0, "dhi": 0.0,
        "dew_point": 15.0, "et0": 0.0, "vpd": 0.5,
    }
    for col, default in default_fill.items():
        if col not in df.columns:
            df[col] = default

    return df


def resample_weather_to_15min(df):
    """将小时级天气数据重采样为15分钟分辨率

    连续量用线性插值，离散量用前向填充。
    末尾扩展一小时，确保最后一天覆盖 23:15/23:30/23:45。
    """
    if df is None or len(df) == 0:
        return None

    # 末尾扩展一小时，保证 15min 网格覆盖到最后一天的 23:45
    last_ts = df.index[-1]
    extra_row = pd.DataFrame(index=[last_ts + pd.Timedelta(hours=1)])
    extra_row.index.name = df.index.name
    df = pd.concat([df, extra_row]).ffill()

    continuous_cols = [
        "temp_air", "relative_humidity", "wind_speed", "pressure",
        "dew_point", "vpd", "et0",
    ]
    forward_fill_cols = [
        "precipitation", "cloud_cover", "wind_direction",
        "ghi", "dni", "dhi",
    ]

    cont_cols = [c for c in continuous_cols if c in df.columns]
    ff_cols = [c for c in forward_fill_cols if c in df.columns]

    df_15min = df.resample("15min").asfreq()

    for col in cont_cols:
        df_15min[col] = df_15min[col].interpolate(method="linear")

    for col in ff_cols:
        df_15min[col] = df_15min[col].ffill()

    for rad_col in ["ghi", "dni", "dhi"]:
        if rad_col in df_15min.columns:
            df_15min[rad_col] = df_15min[rad_col].fillna(0).clip(lower=0)

    df_15min = df_15min.ffill().bfill()
    # 删除末尾多余的整点行（扩展1h引入的下个00:00），只保留到当天23:45
    df_15min = df_15min.iloc[:-1] if len(df_15min) >= 1 else df_15min
    return df_15min


def align_weather_to_price_data(weather_df, price_timestamps):
    """将天气数据对齐到电价数据的时间戳

    Args:
        weather_df: 天气DataFrame（需有时间索引）
        price_timestamps: 电价数据的timestamp列
    Returns: DataFrame索引与price_timestamps对齐
    """
    if weather_df is None or len(weather_df) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex(price_timestamps))

    if isinstance(price_timestamps, pd.DataFrame):
        price_timestamps = pd.to_datetime(price_timestamps.iloc[:, 0])
    elif isinstance(price_timestamps, pd.Series):
        price_timestamps = pd.to_datetime(price_timestamps)

    if weather_df.index.inferred_freq is None or weather_df.index.freq != pd.Timedelta("15min"):
        weather_15min = resample_weather_to_15min(weather_df)
    else:
        weather_15min = weather_df

    weather_15min = weather_15min.sort_index()
    weather_cols = list(weather_15min.columns)

    aligned = pd.DataFrame({"timestamp": pd.to_datetime(price_timestamps)})
    aligned = aligned.sort_values("timestamp")

    aligned_ts_vals = aligned["timestamp"].values
    weather_ts = weather_15min.index

    idx = np.searchsorted(weather_ts, aligned_ts_vals)
    idx = np.clip(idx, 0, len(weather_ts) - 1)

    for col in weather_cols:
        aligned[col] = np.nan

    for i, (ts_idx, ts) in enumerate(zip(idx, aligned_ts_vals)):
        nearest_ts = weather_ts[ts_idx]
        if abs((ts - nearest_ts) / np.timedelta64(1, 's')) <= 1800:
            for col in weather_cols:
                aligned.iloc[i, aligned.columns.get_loc(col)] = weather_15min[col].iloc[ts_idx]

    aligned = aligned.set_index("timestamp")
    aligned = aligned.fillna(aligned.median(numeric_only=True))
    return aligned


def get_location_from_console():
    """从控制台获取经纬度信息

    Returns: (lat, lon, location_name)
    """
    print("\n" + "=" * 55)
    print("[地理位置配置]")
    print("请输入电站所在位置的经纬度（用于获取天气数据）")
    print("常见参考：广州花都 ~ 23.4, 113.2")
    print("         北京    ~ 39.9, 116.4")
    print("         上海    ~ 31.2, 121.5")
    print("=" * 55)

    while True:
        lat_input = input("纬度 (latitude, 如 23.4): ").strip()
        lon_input = input("经度 (longitude, 如 113.2): ").strip()

        try:
            lat = float(lat_input)
            lon = float(lon_input)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                break
            else:
                print("经纬度范围有误，纬度[-90,90]，经度[-180,180]，请重新输入")
        except ValueError:
            print("输入格式有误，请输入数字")

    location_name = input("电站/地点名称（可选，回车跳过）: ").strip()
    if not location_name:
        location_name = f"lat{lat}_lon{lon}"

    print(f"已配置: {location_name} (lat={lat}, lon={lon})")
    return lat, lon, location_name

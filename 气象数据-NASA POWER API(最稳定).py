import pandas as pd
import numpy as np
import requests
import json
import matplotlib
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime, timedelta
import warnings
import threading
import sys
import platform
import subprocess
import tempfile
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================
# 第一部分：中文字体自动配置模块
# ============================================

def setup_chinese_font():
    """
    自动配置Matplotlib中文字体
    返回：是否成功配置中文字体
    """
    system = platform.system()
    font_config_success = False

    print(f"操作系统: {system}")

    # 清除Matplotlib字体缓存
    try:
        import matplotlib
        matplotlib.font_manager._rebuild()
        print("✓ 已清除Matplotlib字体缓存")
    except:
        pass

    # 预定义的中文字体列表（按优先级排序）
    chinese_fonts = {
        'Windows': [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',  # 黑体
            'SimSun',  # 宋体
            'NSimSun',  # 新宋体
            'FangSong',  # 仿宋
            'KaiTi',  # 楷体
            'DengXian',  # 等线
            'YouYuan',  # 幼圆
        ],
        'Darwin': [  # macOS
            'PingFang SC',  # 苹方
            'Hiragino Sans GB',  # 冬青黑体
            'STHeiti',  # 华文黑体
            'STKaiti',  # 华文楷体
            'STSong',  # 华文宋体
            'Arial Unicode MS',  # Arial Unicode
            'Apple LiGothic',  # 苹果俪中黑
            'Apple LiSung',  # 苹果俪宋
        ],
        'Linux': [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',  # 文泉驿正黑
            'DejaVu Sans',  # DejaVu Sans
            'Noto Sans CJK SC',  # Noto Sans CJK
            'Source Han Sans SC',  # 思源黑体
            'Droid Sans Fallback',  # Droid Sans
        ]
    }

    # 获取当前系统的字体列表
    try:
        import matplotlib.font_manager as fm
        system_fonts = [f.name for f in fm.fontManager.ttflist]
        print(f"系统可用的字体数量: {len(system_fonts)}")
    except:
        system_fonts = []
        print("无法获取系统字体列表")

    # 尝试使用预设的字体
    fonts_to_try = chinese_fonts.get(system, [])

    # 添加通用的中文字体
    fonts_to_try.extend([
        'Microsoft YaHei',
        'SimHei',
        'SimSun',
        'PingFang SC',
        'WenQuanYi Micro Hei',
        'Arial Unicode MS',
    ])

    # 去重
    fonts_to_try = list(dict.fromkeys(fonts_to_try))

    print("尝试的字体顺序:", fonts_to_try[:10])  # 只显示前10个

    # 尝试设置字体
    for font_name in fonts_to_try:
        try:
            # 检查字体是否存在
            if system_fonts and font_name not in system_fonts:
                continue

            # 设置Matplotlib字体
            matplotlib.rcParams['font.sans-serif'] = [font_name]
            matplotlib.rcParams['axes.unicode_minus'] = False

            # 测试字体是否可用
            test_fig, test_ax = plt.subplots(figsize=(1, 1))
            test_ax.text(0.5, 0.5, '测试中文', fontproperties=font_name, ha='center', va='center')
            plt.close(test_fig)

            print(f"✓ 成功设置中文字体: {font_name}")
            font_config_success = True
            break

        except Exception as e:
            continue

    # 如果预设字体都失败，尝试从系统中查找中文字体
    if not font_config_success and system_fonts:
        print("尝试从系统中查找中文字体...")

        # 常见中文字体的关键词
        chinese_keywords = ['YaHei', 'Sim', 'Song', 'Kai', 'Fang', 'Hei',
                            'PingFang', 'Hiragino', 'ST', 'WenQuan',
                            'Noto', 'Source Han', 'Microsoft', 'MS']

        for font in system_fonts:
            for keyword in chinese_keywords:
                if keyword.lower() in font.lower():
                    try:
                        matplotlib.rcParams['font.sans-serif'] = [font]
                        matplotlib.rcParams['axes.unicode_minus'] = False
                        print(f"✓ 找到并使用中文字体: {font}")
                        font_config_success = True
                        break
                    except:
                        continue
            if font_config_success:
                break

    # 如果还不行，尝试下载并使用开源字体
    if not font_config_success:
        print("尝试下载开源中文字体...")
        if download_open_source_font():
            font_config_success = True

    # 最后的手段：使用绝对路径的字体文件
    if not font_config_success:
        print("尝试使用绝对路径字体文件...")
        font_config_success = setup_font_by_absolute_path()

    if not font_config_success:
        print("⚠ 警告：无法找到合适的中文字体，图表可能显示为方框")
        print("建议手动安装中文字体，如：")
        print("  1. 微软雅黑 (Windows)")
        print("  2. 苹方 (macOS)")
        print("  3. 文泉驿微米黑 (Linux)")

    return font_config_success


def download_open_source_font():
    """
    下载开源中文字体
    返回：是否成功
    """
    try:
        # 尝试使用SimHei字体（黑体）
        # 这里可以扩展为从网络下载字体文件
        print("尝试使用内置的SimHei字体...")

        # 设置SimHei字体
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        # 测试
        test_fig, test_ax = plt.subplots(figsize=(1, 1))
        test_ax.text(0.5, 0.5, '测试', ha='center', va='center')
        plt.close(test_fig)

        print("✓ 使用SimHei字体成功")
        return True

    except Exception as e:
        print(f"下载字体失败: {e}")
        return False


def setup_font_by_absolute_path():
    """
    通过绝对路径设置字体
    返回：是否成功
    """
    system = platform.system()

    # 不同系统的常见字体路径
    font_paths = {
        'Windows': [
            r'C:\Windows\Fonts\msyh.ttc',  # 微软雅黑
            r'C:\Windows\Fonts\simhei.ttf',  # 黑体
            r'C:\Windows\Fonts\simsun.ttc',  # 宋体
        ],
        'Darwin': [  # macOS
            '/System/Library/Fonts/PingFang.ttc',  # 苹方
            '/System/Library/Fonts/STHeiti Light.ttc',  # 华文黑体
            '/Library/Fonts/Arial Unicode.ttf',  # Arial Unicode
        ],
        'Linux': [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans
        ]
    }

    paths_to_try = font_paths.get(system, [])

    for font_path in paths_to_try:
        if os.path.exists(font_path):
            try:
                # 将字体文件添加到Matplotlib
                import matplotlib.font_manager as fm
                fm.fontManager.addfont(font_path)

                # 获取字体名称
                font_name = fm.FontProperties(fname=font_path).get_name()

                # 设置字体
                matplotlib.rcParams['font.sans-serif'] = [font_name]
                matplotlib.rcParams['axes.unicode_minus'] = False

                print(f"✓ 通过绝对路径设置字体成功: {font_name}")
                return True

            except Exception as e:
                print(f"字体路径 {font_path} 设置失败: {e}")

    return False


# 在程序开始时设置中文字体
print("正在配置中文字体...")
font_success = setup_chinese_font()

# ============================================
# 第二部分：中英文列名定义
# ============================================

# 定义中英文对照的列名
COLUMN_NAMES = {
    'time': '时间_time',
    'temp_air': '气温_temp_air_°C',
    'ghi': '水平面总辐射_ghi_W/m²',
    'dni': '法向直射辐射_dni_W/m²',
    'dhi': '水平散射辐射_dhi_W/m²',
    'clearness_index': '晴空指数_clearness_index',
    'wind_speed': '风速_wind_speed_m/s',
    'relative_humidity': '相对湿度_relative_humidity_%',
    'pressure': '气压_pressure_kPa',
    'precipitation': '降水量_precipitation_mm',
    'ac_power_kw': '交流出力_ac_power_kW',
    'dc_power_kw': '直流出力_dc_power_kW',
    'cell_temperature': '电池温度_cell_temperature_°C',
    'poa_global': '斜面总辐射_poa_global_W/m²',
    'poa_direct': '斜面直射辐射_poa_direct_W/m²',
    'poa_diffuse': '斜面散射辐射_poa_diffuse_W/m²',
    'aoi': '入射角_aoi_°',
    'zenith': '天顶角_zenith_°',
    'azimuth': '方位角_azimuth_°',
    'elevation': '高度角_elevation_°',
    'dew_point': '露点温度_dew_point_°C',
    'visibility': '能见度_visibility_km',
    'cloud_cover': '云量_cloud_cover_%',
    'snow_depth': '积雪深度_snow_depth_cm',
    'solar_zenith': '太阳天顶角_solar_zenith_°',
    'solar_azimuth': '太阳方位角_solar_azimuth_°',
    'sunrise': '日出时间_sunrise',
    'sunset': '日落时间_sunset',
    'daylight_hours': '日照时数_daylight_hours_h',
    'temperature_max': '最高温度_temperature_max_°C',
    'temperature_min': '最低温度_temperature_min_°C',
    'temperature_mean': '平均温度_temperature_mean_°C',
    'ghi_daily': '日总辐射_ghi_daily_kWh/m²',
    'energy_daily': '日发电量_energy_daily_kWh',
    'energy_monthly': '月发电量_energy_monthly_kWh',
    'energy_yearly': '年发电量_energy_yearly_kWh',
    'capacity_factor': '容量因子_capacity_factor_%',
    'performance_ratio': '性能比_performance_ratio',
    'specific_yield': '单位发电量_specific_yield_kWh/kWp',
    'array_temperature': '组件温度_array_temperature_°C',
    'inverter_efficiency': '逆变器效率_inverter_efficiency_%',
    'system_losses': '系统损耗_system_losses_%',
    'soiling_losses': '污染损耗_soiling_losses_%',
    'shading_losses': '遮挡损耗_shading_losses_%',
    'mismatch_losses': '失配损耗_mismatch_losses_%',
    'wiring_losses': '线损_wiring_losses_%',
    'transformer_losses': '变压器损耗_transformer_losses_%',
    'availability': '可利用率_availability_%',
    'downtime': '停机时间_downtime_h',
    'maintenance_hours': '维护时间_maintenance_hours_h',
    'revenue': '收益_revenue_¥',
    'cost': '成本_cost_¥',
    'roi': '投资回报率_roi_%',
    'payback_period': '投资回收期_payback_period_years',
    'lcoe': '平准化度电成本_lcoe_¥/kWh',
    'irr': '内部收益率_irr_%',
    'npv': '净现值_npv_¥',
}

# 反向映射：从中文列名到英文列名
REVERSE_COLUMN_NAMES = {v: k for k, v in COLUMN_NAMES.items()}


def get_chinese_column_name(english_name):
    """获取中文列名，如果不存在则返回英文列名"""
    return COLUMN_NAMES.get(english_name, english_name)


def get_english_column_name(chinese_name):
    """从中文列名获取英文列名，如果不存在则返回原列名"""
    return REVERSE_COLUMN_NAMES.get(chinese_name, chinese_name)


def rename_dataframe_columns(df, to_chinese=True):
    """
    重命名DataFrame的列名

    参数:
        df: pandas DataFrame
        to_chinese: 如果为True，则将列名改为中文；如果为False，则改为英文

    返回:
        重命名后的DataFrame
    """
    df_renamed = df.copy()

    if to_chinese:
        # 将列名改为中文
        new_columns = {}
        for col in df.columns:
            if col in COLUMN_NAMES:
                new_columns[col] = COLUMN_NAMES[col]
            else:
                # 如果列名不在字典中，保持原样
                new_columns[col] = col

        # 重命名列
        df_renamed = df_renamed.rename(columns=new_columns)

        # 重命名索引（如果是时间索引）
        if df_renamed.index.name in COLUMN_NAMES:
            df_renamed.index.name = COLUMN_NAMES[df_renamed.index.name]
    else:
        # 将列名改为英文
        new_columns = {}
        for col in df.columns:
            if col in REVERSE_COLUMN_NAMES:
                new_columns[col] = REVERSE_COLUMN_NAMES[col]
            else:
                # 如果列名不在字典中，保持原样
                new_columns[col] = col

        # 重命名列
        df_renamed = df_renamed.rename(columns=new_columns)

        # 重命名索引
        if df_renamed.index.name in REVERSE_COLUMN_NAMES:
            df_renamed.index.name = REVERSE_COLUMN_NAMES[df_renamed.index.name]

    return df_renamed


# ============================================
# 第三部分：数据获取和GUI模块
# ============================================

# 全局变量
data_fetch_complete = False
fetch_progress = 0
fetch_status = ""
current_weather_data = None
current_data_path = None


def select_save_directory():
    """弹出对话框让用户选择保存目录"""
    root = tk.Tk()
    root.withdraw()

    # 设置初始目录
    initial_dir = os.path.expanduser("~")
    if os.path.exists("D:/"):
        initial_dir = "D:/"
    elif os.path.exists("/Users/"):
        initial_dir = "/Users/"

    # 弹出文件夹选择对话框
    selected_dir = filedialog.askdirectory(
        title="请选择保存数据的目录",
        initialdir=initial_dir
    )

    root.destroy()
    return selected_dir


def create_progress_window():
    """创建进度窗口"""
    progress_window = tk.Tk()
    progress_window.title("数据获取进度")
    progress_window.geometry("500x300")
    progress_window.configure(bg='white')

    # 窗口居中
    progress_window.update_idletasks()
    width = progress_window.winfo_width()
    height = progress_window.winfo_height()
    x = (progress_window.winfo_screenwidth() // 2) - (width // 2)
    y = (progress_window.winfo_screenheight() // 2) - (height // 2)
    progress_window.geometry(f'{width}x{height}+{x}+{y}')

    # 设置窗口图标
    try:
        progress_window.iconbitmap(default='')  # 可以设置图标文件路径
    except:
        pass

    # 标题
    title_label = tk.Label(
        progress_window,
        text="🌞 NASA POWER 数据获取",
        font=("微软雅黑", 16, "bold") if font_success else ("Arial", 16, "bold"),
        bg='white',
        fg='#1a73e8'
    )
    title_label.pack(pady=20)

    # 地点信息
    location_label = tk.Label(
        progress_window,
        text=f"📍 地点: 广州花都 (纬度: 23.4°, 经度: 113.2°)",
        font=("微软雅黑", 10) if font_success else ("Arial", 10),
        bg='white'
    )
    location_label.pack(pady=5)

    # 年份信息
    year_label = tk.Label(
        progress_window,
        text="📅 数据年份: 2025年",
        font=("微软雅黑", 10) if font_success else ("Arial", 10),
        bg='white'
    )
    year_label.pack(pady=5)

    # 进度条
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        progress_window,
        variable=progress_var,
        maximum=100,
        length=400,
        mode='determinate'
    )
    progress_bar.pack(pady=20)

    # 进度百分比
    percent_label = tk.Label(
        progress_window,
        text="0%",
        font=("微软雅黑", 12, "bold") if font_success else ("Arial", 12, "bold"),
        bg='white',
        fg='#1a73e8'
    )
    percent_label.pack()

    # 状态标签
    status_label = tk.Label(
        progress_window,
        text="正在初始化...",
        font=("微软雅黑", 10) if font_success else ("Arial", 10),
        bg='white',
        fg='#666666'
    )
    status_label.pack(pady=10)

    # 详细状态
    detail_label = tk.Label(
        progress_window,
        text="",
        font=("微软雅黑", 9) if font_success else ("Arial", 9),
        bg='white',
        fg='#888888',
        wraplength=450
    )
    detail_label.pack(pady=5)

    # 关闭按钮
    close_button = tk.Button(
        progress_window,
        text="关闭",
        font=("微软雅黑", 10) if font_success else ("Arial", 10),
        bg='#f44336',
        fg='white',
        state='disabled',
        command=progress_window.destroy
    )
    close_button.pack(pady=20)

    def update_progress():
        """更新进度条和状态"""
        global data_fetch_complete, fetch_progress, fetch_status, current_data_path

        if data_fetch_complete:
            progress_var.set(100)
            percent_label.config(text="100%")
            status_label.config(text="✅ 数据获取完成！", fg='#4CAF50')

            if current_data_path:
                detail_label.config(text=f"保存位置: {current_data_path}")

            close_button.config(state='normal', bg='#4CAF50', text="完成")

            # 5秒后自动关闭
            progress_window.after(5000, progress_window.destroy)
        else:
            progress_var.set(fetch_progress)
            percent_label.config(text=f"{int(fetch_progress)}%")
            status_label.config(text=fetch_status)

            # 每100ms更新一次
            progress_window.after(100, update_progress)

    progress_window.after(100, update_progress)
    return progress_window


def get_nasa_power_data_2025_async(lat, lon, save_dir, filename):
    """异步获取NASA POWER数据"""
    global data_fetch_complete, fetch_progress, fetch_status, current_weather_data, current_data_path

    data_fetch_complete = False
    fetch_progress = 0
    fetch_status = ""
    current_weather_data = None
    current_data_path = None

    try:
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        current_data_path = full_path

        fetch_status = "正在连接NASA POWER服务器..."
        fetch_progress = 10

        base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
        params = {
            'parameters': 'T2M,ALLSKY_SFC_SW_DWN,ALLSKY_KT,WS10M,RH2M,PS,PRECTOTCORR',
            'community': 'RE',
            'longitude': lon,
            'latitude': lat,
            'start': '20250101',
            'end': '20251231',
            'format': 'JSON',
            'user': 'anonymous'
        }

        fetch_status = "正在从NASA服务器下载数据..."
        fetch_progress = 20

        response = requests.get(base_url, params=params, timeout=120)

        if response.status_code == 200:
            fetch_status = "数据下载完成，正在处理..."
            fetch_progress = 40

            data = response.json()
            properties = data['properties']['parameter']
            timestamps = data['properties']['parameter']['T2M'].keys()

            fetch_status = "正在创建时间序列..."
            fetch_progress = 50

            time_index = pd.to_datetime(list(timestamps), format='%Y%m%d%H')
            time_index = time_index.tz_localize('UTC').tz_convert('Asia/Shanghai')

            weather_data = pd.DataFrame({
                'time_utc': pd.to_datetime(list(timestamps), format='%Y%m%d%H'),
                'temp_air': list(properties['T2M'].values()),
                'ghi': list(properties['ALLSKY_SFC_SW_DWN'].values()),
                'clearness_index': list(properties['ALLSKY_KT'].values()),
                'wind_speed': list(properties['WS10M'].values()),
                'relative_humidity': list(properties['RH2M'].values()),
                'pressure': list(properties['PS'].values()),
                'precipitation': list(properties['PRECTOTCORR'].values()),
            })

            weather_data.set_index('time_utc', inplace=True)
            weather_data.index = weather_data.index.tz_localize('UTC').tz_convert('Asia/Shanghai')
            weather_data.index.name = 'time'

            fetch_status = "正在计算太阳位置和辐射分量..."
            fetch_progress = 60

            from pvlib import solarposition
            solar_pos = solarposition.get_solarposition(
                time=weather_data.index,
                latitude=lat,
                longitude=lon,
                altitude=50
            )

            fetch_status = "正在计算辐射分量..."
            fetch_progress = 70

            KT = weather_data['clearness_index']
            zenith_rad = np.radians(solar_pos['zenith'])
            cos_zenith = np.cos(zenith_rad)
            cos_zenith = np.where(cos_zenith > 0.1, cos_zenith, 0.1)

            weather_data['dni'] = np.where(
                KT > 0.6,
                weather_data['ghi'] * 0.8 / cos_zenith,
                weather_data['ghi'] * (1.0 - 0.09 * KT) / cos_zenith
            )

            weather_data['dhi'] = weather_data['ghi'] - weather_data['dni'] * cos_zenith
            weather_data['dni'] = weather_data['dni'].clip(lower=0)
            weather_data['dhi'] = weather_data['dhi'].clip(lower=0)

            fetch_status = "正在保存数据到文件..."
            fetch_progress = 80

            # 将列名改为中英文
            weather_data_chinese = rename_dataframe_columns(weather_data, to_chinese=True)

            # 保存CSV文件（包含中英文列名）
            weather_data_chinese.to_csv(full_path, encoding='utf-8-sig')

            fetch_status = "正在生成15分钟分辨率数据..."
            fetch_progress = 90

            # 创建15分钟数据
            weather_15min = pd.DataFrame()
            for col in weather_data.columns:
                if col != 'clearness_index':
                    weather_15min[col] = weather_data[col].resample('15min').interpolate(method='linear')
                else:
                    weather_15min[col] = weather_data[col].resample('15min').ffill()

            # 将列名改为中英文
            weather_15min_chinese = rename_dataframe_columns(weather_15min, to_chinese=True)

            # 保存15分钟数据
            filename_15min = filename.replace('.csv', '_15min.csv')
            full_path_15min = os.path.join(save_dir, filename_15min)
            weather_15min_chinese.to_csv(full_path_15min, encoding='utf-8-sig')

            current_weather_data = weather_15min
            fetch_status = "✅ 数据获取和处理完成！"
            fetch_progress = 100

        else:
            fetch_status = f"❌ API请求失败，状态码: {response.status_code}"
            fetch_progress = 100

    except requests.exceptions.Timeout:
        fetch_status = "❌ 请求超时，请检查网络连接"
        fetch_progress = 100
    except requests.exceptions.ConnectionError:
        fetch_status = "❌ 网络连接错误，请检查网络"
        fetch_progress = 100
    except Exception as e:
        fetch_status = f"❌ 获取数据时出错: {str(e)}"
        fetch_progress = 100
    finally:
        data_fetch_complete = True


def create_visualization(weather_data, save_dir):
    """
    创建可视化图表
    """
    if weather_data is None or len(weather_data) == 0:
        return None

    # 创建图形
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 设置全局标题
    if font_success:
        fig.suptitle('广州花都2025年气象数据（15分钟分辨率）', fontsize=16, y=1.02)
    else:
        fig.suptitle('Guangzhou Huadu 2025 Weather Data (15-min resolution)', fontsize=16, y=1.02)

    # 颜色方案
    colors = ['#FF6B6B', '#FFD166', '#06D6A0', '#118AB2', '#073B4C', '#7209B7']

    # 获取日期格式化的X轴标签
    dates = weather_data.index

    # 1. 温度图
    ax1 = axes[0, 0]
    ax1.plot(dates, weather_data['temp_air'], color=colors[0], alpha=0.7, linewidth=0.5)
    if font_success:
        ax1.set_title('气温变化', fontsize=12, fontweight='bold')
        ax1.set_ylabel('温度 (°C)', fontsize=10)
    else:
        ax1.set_title('Temperature', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temp (°C)', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # 2. 辐射图
    ax2 = axes[0, 1]
    ax2.plot(dates, weather_data['ghi'], color=colors[1], alpha=0.7, linewidth=0.5)
    if font_success:
        ax2.set_title('水平面总辐射', fontsize=12, fontweight='bold')
    else:
        ax2.set_title('Global Horizontal Irradiance', fontsize=12, fontweight='bold')
    ax2.set_ylabel('GHI (W/m²)', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(axis='x', rotation=45, labelsize=8)

    # 3. 风速图
    ax3 = axes[1, 0]
    ax3.plot(dates, weather_data['wind_speed'], color=colors[2], alpha=0.7, linewidth=0.5)
    if font_success:
        ax3.set_title('风速变化', fontsize=12, fontweight='bold')
    else:
        ax3.set_title('Wind Speed', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Wind (m/s)', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(axis='x', rotation=45, labelsize=8)

    # 4. 湿度图
    ax4 = axes[1, 1]
    ax4.plot(dates, weather_data['relative_humidity'], color=colors[3], alpha=0.7, linewidth=0.5)
    if font_success:
        ax4.set_title('相对湿度', fontsize=12, fontweight='bold')
    else:
        ax4.set_title('Relative Humidity', fontsize=12, fontweight='bold')
    ax4.set_ylabel('RH (%)', fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)

    # 5. 降水量图
    ax5 = axes[2, 0]
    # 只绘制有降水的点
    precipitation_data = weather_data['precipitation'].copy()
    precipitation_data[precipitation_data == 0] = np.nan
    ax5.bar(dates, precipitation_data, color=colors[4], alpha=0.7, width=0.02)
    if font_success:
        ax5.set_title('降水量', fontsize=12, fontweight='bold')
    else:
        ax5.set_title('Precipitation', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Precip (mm)', fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)

    # 6. 晴空指数图
    ax6 = axes[2, 1]
    # 使用散点图显示晴空指数
    ax6.scatter(dates, weather_data['clearness_index'], color=colors[5], alpha=0.3, s=1)
    if font_success:
        ax6.set_title('晴空指数', fontsize=12, fontweight='bold')
    else:
        ax6.set_title('Clearness Index', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Clearness (0-1)', fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()

    # 保存图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = os.path.join(save_dir, f"weather_visualization_{timestamp}.png")

    # 调整保存参数以获得更好的质量
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)

    return plot_path


def calculate_pv_generation(weather_data, save_dir):
    """
    基于气象数据计算光伏发电量
    """
    if weather_data is None or len(weather_data) == 0:
        return None, None, None, None

    try:
        from pvlib import pvsystem, modelchain, location

        print("正在计算光伏出力...")

        # 定义地点
        lat, lon = 23.4, 113.2
        tz = 'Asia/Shanghai'
        site = location.Location(lat, lon, tz=tz)

        # 准备气象数据
        weather_for_pvlib = pd.DataFrame({
            'ghi': weather_data['ghi'],
            'dni': weather_data['dni'],
            'dhi': weather_data['dhi'],
            'temp_air': weather_data['temp_air'],
            'wind_speed': weather_data['wind_speed']
        })

        # 定义光伏系统参数（假设1MW系统）
        system = pvsystem.PVSystem(
            surface_tilt=20,  # 倾角20度
            surface_azimuth=180,  # 正南
            module_parameters={'pdc0': 1000, 'gamma_pdc': -0.004},
            inverter_parameters={'pdc0': 1100, 'eta_inv_nom': 0.96},
            modules_per_string=20,
            strings_per_inverter=50
        )

        # 创建模型链
        mc = modelchain.ModelChain(system, site)

        # 运行模型
        mc.run_model(weather_for_pvlib)

        # 获取出力结果
        pv_output = pd.DataFrame({
            'ac_power_kw': mc.results.ac / 1000,  # 转换为kW
            'dc_power_kw': mc.results.dc / 1000,
            'cell_temperature': mc.results.cell_temperature
        })

        pv_output.index = weather_data.index

        # 计算日发电量
        daily_energy = pv_output['ac_power_kw'].resample('D').sum()

        # 保存光伏出力数据（中英文列名）
        pv_output_chinese = rename_dataframe_columns(pv_output, to_chinese=True)
        daily_energy_chinese = rename_dataframe_columns(
            daily_energy.to_frame(name='energy_daily'), to_chinese=True
        )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        pv_output_path = os.path.join(save_dir, f"pv_output_2025_{timestamp}.csv")
        pv_output_chinese.to_csv(pv_output_path, encoding='utf-8-sig')

        daily_energy_path = os.path.join(save_dir, f"daily_energy_2025_{timestamp}.csv")
        daily_energy_chinese.to_csv(daily_energy_path, encoding='utf-8-sig')

        print(f"✓ 光伏出力数据已保存到: {pv_output_path}")
        print(f"✓ 日发电量数据已保存到: {daily_energy_path}")

        # 打印统计
        print(f"\n光伏系统统计:")
        print(f"  年总发电量: {daily_energy.sum():.0f} kWh")
        print(f"  平均日发电量: {daily_energy.mean():.0f} kWh")
        print(f"  最大日发电量: {daily_energy.max():.0f} kWh (日期: {daily_energy.idxmax().date()})")
        print(f"  最小日发电量: {daily_energy.min():.0f} kWh (日期: {daily_energy.idxmin().date()})")

        return pv_output, daily_energy, pv_output_path, daily_energy_path

    except Exception as e:
        print(f"✗ 计算光伏出力时出错: {str(e)}")
        return None, None, None, None


def generate_report(weather_data, save_dir, data_path):
    """生成数据报告"""
    if weather_data is None or len(weather_data) == 0:
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(save_dir, f"data_report_{timestamp}.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        if font_success:
            f.write("广州花都2025年气象数据报告\n")
        else:
            f.write("Guangzhou Huadu 2025 Weather Data Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"地点: 广州花都 (纬度: 23.4°, 经度: 113.2°)\n")
        f.write(f"数据年份: 2025\n")
        f.write(f"时间分辨率: 15分钟\n")
        f.write(f"数据点数: {len(weather_data)}\n")
        f.write(f"时间范围: {weather_data.index[0]} 到 {weather_data.index[-1]}\n")
        f.write(f"数据文件: {data_path}\n\n")

        f.write("=" * 70 + "\n")
        f.write("1. 数据统计摘要\n")
        f.write("=" * 70 + "\n\n")

        # 将列名临时改为中文以便报告阅读
        weather_data_chinese = rename_dataframe_columns(weather_data, to_chinese=True)
        stats = weather_data_chinese.describe()
        f.write(stats.to_string())
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("2. 月度统计\n")
        f.write("=" * 70 + "\n\n")

        # 月度统计
        monthly_stats = weather_data.resample('M').agg({
            'temp_air': ['mean', 'min', 'max'],
            'ghi': ['mean', 'max'],
            'wind_speed': 'mean',
            'relative_humidity': 'mean',
            'precipitation': 'sum'
        })

        # 重命名列名以便阅读
        monthly_stats_renamed = monthly_stats.copy()
        if not monthly_stats_renamed.columns.nlevels > 1:
            monthly_stats_renamed.columns = [get_chinese_column_name(col) for col in monthly_stats_renamed.columns]

        f.write(monthly_stats_renamed.round(2).to_string())
        f.write("\n\n")

        f.write("=" * 70 + "\n")
        f.write("3. 数据质量检查\n")
        f.write("=" * 70 + "\n\n")

        missing = weather_data.isnull().sum()
        f.write("缺失值统计:\n")
        for col, count in missing.items():
            percentage = (count / len(weather_data)) * 100
            col_name = get_chinese_column_name(col)
            f.write(f"  {col_name}: {count} 个 ({percentage:.2f}%)\n")

    return report_path


def show_completion_dialog(save_dir, files_created):
    """显示完成对话框"""
    completion_window = tk.Tk()
    completion_window.title("数据获取完成")
    completion_window.geometry("600x500")
    completion_window.configure(bg='#f0f8ff')

    completion_window.update_idletasks()
    width = completion_window.winfo_width()
    height = completion_window.winfo_height()
    x = (completion_window.winfo_screenwidth() // 2) - (width // 2)
    y = (completion_window.winfo_screenheight() // 2) - (height // 2)
    completion_window.geometry(f'{width}x{height}+{x}+{y}')

    # 字体设置
    font_family = "微软雅黑" if font_success else "Arial"

    icon_label = tk.Label(
        completion_window,
        text="✅",
        font=("Arial", 48),
        bg='#f0f8ff',
        fg='#4CAF50'
    )
    icon_label.pack(pady=20)

    title_label = tk.Label(
        completion_window,
        text="数据获取成功！" if font_success else "Data Acquisition Successful!",
        font=(font_family, 20, "bold"),
        bg='#f0f8ff',
        fg='#1a73e8'
    )
    title_label.pack(pady=10)

    path_label = tk.Label(
        completion_window,
        text=f"保存目录: {save_dir}",
        font=(font_family, 10),
        bg='#f0f8ff',
        fg='#666666',
        wraplength=550
    )
    path_label.pack(pady=5)

    # 创建滚动条框架用于显示文件列表
    files_frame = tk.Frame(completion_window, bg='#f0f8ff')
    files_frame.pack(pady=10, fill='both', expand=True)

    files_label = tk.Label(
        files_frame,
        text="生成的文件列表:",
        font=(font_family, 12, "bold"),
        bg='#f0f8ff',
        fg='#333333'
    )
    files_label.pack(anchor='w', padx=20)

    # 创建Canvas和Scrollbar
    canvas = tk.Canvas(files_frame, bg='#f0f8ff', height=200)
    scrollbar = tk.Scrollbar(files_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas, bg='#f0f8ff')

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # 添加文件到滚动区域
    for i, (filename, size_kb) in enumerate(files_created.items()):
        file_label = tk.Label(
            scrollable_frame,
            text=f"📄 {filename} ({size_kb:.1f} KB)",
            font=(font_family, 9),
            bg='#f0f8ff',
            fg='#555555',
            anchor='w',
            justify='left'
        )
        file_label.pack(fill='x', padx=20, pady=2)

    canvas.pack(side="left", fill="both", expand=True, padx=(20, 0))
    scrollbar.pack(side="right", fill="y")

    button_frame = tk.Frame(completion_window, bg='#f0f8ff')
    button_frame.pack(pady=20)

    def open_folder():
        try:
            if sys.platform == 'win32':
                os.startfile(save_dir)
            elif sys.platform == 'darwin':
                os.system(f'open "{save_dir}"')
            else:
                os.system(f'xdg-open "{save_dir}"')
        except:
            pass

    def close_all():
        completion_window.destroy()
        try:
            tk.Tk().destroy()
        except:
            pass

    open_button = tk.Button(
        button_frame,
        text="📁 打开文件夹" if font_success else "📁 Open Folder",
        font=(font_family, 10),
        bg='#4CAF50',
        fg='white',
        padx=20,
        pady=5,
        command=open_folder
    )
    open_button.pack(side='left', padx=10)

    close_button = tk.Button(
        button_frame,
        text="关闭" if font_success else "Close",
        font=(font_family, 10),
        bg='#f44336',
        fg='white',
        padx=20,
        pady=5,
        command=close_all
    )
    close_button.pack(side='left', padx=10)

    completion_window.mainloop()


def main():
    """主程序"""
    global data_fetch_complete, fetch_progress, fetch_status, current_weather_data, current_data_path

    print("=" * 70)
    if font_success:
        print("广州花都2025年气象数据获取系统")
    else:
        print("Guangzhou Huadu 2025 Weather Data Acquisition System")
    print("=" * 70)

    if font_success:
        print("\n请选择数据保存目录...")
    else:
        print("\nPlease select a directory to save data...")

    save_dir = select_save_directory()

    if not save_dir:
        if font_success:
            print("用户取消了操作")
        else:
            print("Operation cancelled by user")
        return

    progress_window = create_progress_window()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if font_success:
        filename = f"广州花都_2025气象数据_{timestamp}.csv"
    else:
        filename = f"Guangzhou_Huadu_2025_Weather_Data_{timestamp}.csv"

    lat, lon = 23.4, 113.2
    fetch_thread = threading.Thread(
        target=get_nasa_power_data_2025_async,
        args=(lat, lon, save_dir, filename)
    )
    fetch_thread.daemon = True
    fetch_thread.start()

    progress_window.mainloop()

    if current_weather_data is not None and current_data_path is not None:
        if font_success:
            print("\n数据获取成功！正在生成报告和可视化...")
        else:
            print("\nData acquisition successful! Generating report and visualization...")

        # 可视化
        plot_path = create_visualization(current_weather_data, save_dir)

        # 计算光伏发电量
        pv_output, daily_energy, pv_path, energy_path = calculate_pv_generation(current_weather_data, save_dir)

        # 生成报告
        report_path = generate_report(current_weather_data, save_dir, current_data_path)

        # 获取15分钟数据文件名
        filename_15min = filename.replace('.csv', '_15min.csv')
        data_15min_path = os.path.join(save_dir, filename_15min)

        # 计算文件大小
        files_created = {}
        file_paths = [current_data_path, data_15min_path, plot_path, report_path]

        if pv_path:
            file_paths.append(pv_path)
        if energy_path:
            file_paths.append(energy_path)

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    files_created[os.path.basename(file_path)] = size_kb
                except:
                    pass

        # 显示完成对话框
        show_completion_dialog(save_dir, files_created)

        if font_success:
            print("\n✅ 所有处理完成！")
            print(f"数据已保存到: {save_dir}")

            # 显示CSV文件示例
            print("\nCSV文件列标题示例:")
            try:
                sample_df = pd.read_csv(current_data_path, nrows=0, encoding='utf-8-sig')
                print("列名 (中文_英文_单位):")
                for col in sample_df.columns:
                    print(f"  - {col}")
            except:
                pass
        else:
            print("\n✅ All processing completed!")
            print(f"Data saved to: {save_dir}")

    else:
        if font_success:
            print("\n❌ 数据获取失败")
        else:
            print("\n❌ Data acquisition failed")
        print(f"Error: {fetch_status}")


if __name__ == "__main__":
    # 检查依赖库
    required_libs = ['pandas', 'numpy', 'requests', 'matplotlib', 'tkinter']

    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)

    if missing_libs:
        print("❌ 缺少必要的库:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print("\n请运行以下命令安装:")
        print("pip install pandas numpy matplotlib requests pvlib")

        if 'tkinter' in missing_libs:
            print("\n对于tkinter:")
            print("  Windows/Mac: 通常已预装")
            print("  Linux: sudo apt-get install python3-tk")

        input("按Enter键退出..." if font_success else "Press Enter to exit...")
        sys.exit(1)

    # 运行主程序
    try:
        main()
    except KeyboardInterrupt:
        if font_success:
            print("\n\n程序被用户中断")
        else:
            print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\n\n程序运行出错: {e}" if font_success else f"\n\nProgram error: {e}")
        import traceback

        traceback.print_exc()
        input("按Enter键退出..." if font_success else "Press Enter to exit...")

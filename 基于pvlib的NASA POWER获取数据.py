import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pvlib import solarposition, irradiance, atmosphere
import warnings
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import calendar
import threading
import time

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class ExportConfigDialog:
    """导出配置对话框"""

    def __init__(self, parent):
        self.parent = parent

        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("太阳能数据导出配置")
        self.dialog.geometry("500x500")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()  # 使对话框模态

        # 初始化变量
        self.start_date = None
        self.end_date = None
        self.output_dir = None
        self.result = None

        self.create_widgets()
        self.center_dialog()

    def create_widgets(self):
        """创建对话框部件"""
        # 标题
        title_label = tk.Label(self.dialog, text="太阳能数据导出配置", font=("Arial", 14, "bold"))
        title_label.pack(pady=15)

        # 说明文本
        info_text = "请选择要导出的时间范围和保存路径"
        info_label = tk.Label(self.dialog, text=info_text, fg="blue")
        info_label.pack(pady=5)

        # 开始日期选择框架
        start_frame = tk.LabelFrame(self.dialog, text="开始日期", padx=10, pady=10)
        start_frame.pack(pady=10, padx=20, fill=tk.X)

        # 年份选择
        start_year_frame = tk.Frame(start_frame)
        start_year_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(start_year_frame, text="年:", width=5, anchor="w").pack(side=tk.LEFT)
        self.start_year_var = tk.StringVar(value="2025")
        # 允许选择2000-2030年的数据
        start_year_spinbox = tk.Spinbox(start_year_frame, from_=2000, to=2030, width=8,
                                        textvariable=self.start_year_var, command=self.update_start_day_range)
        start_year_spinbox.pack(side=tk.LEFT)

        # 月份选择
        start_month_frame = tk.Frame(start_frame)
        start_month_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(start_month_frame, text="月:", width=5, anchor="w").pack(side=tk.LEFT)
        self.start_month_var = tk.StringVar(value="1")
        start_month_spinbox = tk.Spinbox(start_month_frame, from_=1, to=12, width=5,
                                         textvariable=self.start_month_var, command=self.update_start_day_range)
        start_month_spinbox.pack(side=tk.LEFT)

        # 日期选择
        start_day_frame = tk.Frame(start_frame)
        start_day_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(start_day_frame, text="日:", width=5, anchor="w").pack(side=tk.LEFT)
        self.start_day_var = tk.StringVar(value="1")
        self.start_day_spinbox = tk.Spinbox(start_day_frame, from_=1, to=31, width=5,
                                            textvariable=self.start_day_var)
        self.start_day_spinbox.pack(side=tk.LEFT)

        # 结束日期选择框架
        end_frame = tk.LabelFrame(self.dialog, text="结束日期", padx=10, pady=10)
        end_frame.pack(pady=10, padx=20, fill=tk.X)

        # 年份选择
        end_year_frame = tk.Frame(end_frame)
        end_year_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(end_year_frame, text="年:", width=5, anchor="w").pack(side=tk.LEFT)
        self.end_year_var = tk.StringVar(value="2025")
        end_year_spinbox = tk.Spinbox(end_year_frame, from_=2000, to=2030, width=8,
                                      textvariable=self.end_year_var, command=self.update_end_day_range)
        end_year_spinbox.pack(side=tk.LEFT)

        # 月份选择
        end_month_frame = tk.Frame(end_frame)
        end_month_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(end_month_frame, text="月:", width=5, anchor="w").pack(side=tk.LEFT)
        self.end_month_var = tk.StringVar(value="12")
        end_month_spinbox = tk.Spinbox(end_month_frame, from_=1, to=12, width=5,
                                       textvariable=self.end_month_var, command=self.update_end_day_range)
        end_month_spinbox.pack(side=tk.LEFT)

        # 日期选择
        end_day_frame = tk.Frame(end_frame)
        end_day_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(end_day_frame, text="日:", width=5, anchor="w").pack(side=tk.LEFT)
        self.end_day_var = tk.StringVar(value="31")
        self.end_day_spinbox = tk.Spinbox(end_day_frame, from_=1, to=31, width=5,
                                          textvariable=self.end_day_var)
        self.end_day_spinbox.pack(side=tk.LEFT)

        # 当前日期按钮
        current_date_frame = tk.Frame(self.dialog)
        current_date_frame.pack(pady=5)

        current_date_button = tk.Button(current_date_frame, text="设为当前日期",
                                        command=self.set_to_current_date, width=15)
        current_date_button.pack(side=tk.LEFT, padx=5)

        # 全年按钮
        full_year_button = tk.Button(current_date_frame, text="设为全年",
                                     command=self.set_to_full_year, width=15)
        full_year_button.pack(side=tk.LEFT, padx=5)

        # 保存路径选择
        path_frame = tk.LabelFrame(self.dialog, text="保存路径", padx=10, pady=10)
        path_frame.pack(pady=20, padx=20, fill=tk.X)

        self.path_var = tk.StringVar()
        path_entry = tk.Entry(path_frame, textvariable=self.path_var, width=50)
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        browse_button = tk.Button(path_frame, text="浏览...", command=self.browse_path, width=8)
        browse_button.pack(side=tk.RIGHT)

        # 按钮框架
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=20)

        ok_button = tk.Button(button_frame, text="开始导出", command=self.on_ok, width=12, bg="#4CAF50", fg="white")
        ok_button.pack(side=tk.LEFT, padx=10)

        cancel_button = tk.Button(button_frame, text="取消", command=self.on_cancel, width=12)
        cancel_button.pack(side=tk.RIGHT, padx=10)

        # 初始化日期范围
        self.update_start_day_range()
        self.update_end_day_range()

    def update_start_day_range(self):
        """更新开始日期的日期范围"""
        try:
            year = int(self.start_year_var.get())
            month = int(self.start_month_var.get())
            max_day = calendar.monthrange(year, month)[1]
            self.start_day_spinbox.config(to=max_day)

            # 如果当前日期大于最大日期，调整为最大日期
            current_day = int(self.start_day_var.get())
            if current_day > max_day:
                self.start_day_var.set(str(max_day))
        except:
            pass

    def update_end_day_range(self):
        """更新结束日期的日期范围"""
        try:
            year = int(self.end_year_var.get())
            month = int(self.end_month_var.get())
            max_day = calendar.monthrange(year, month)[1]
            self.end_day_spinbox.config(to=max_day)

            # 如果当前日期大于最大日期，调整为最大日期
            current_day = int(self.end_day_var.get())
            if current_day > max_day:
                self.end_day_var.set(str(max_day))
        except:
            pass

    def set_to_current_date(self):
        """设置为当前日期"""
        now = datetime.now()
        self.start_year_var.set(str(now.year))
        self.start_month_var.set(str(now.month))
        self.start_day_var.set(str(now.day))

        self.end_year_var.set(str(now.year))
        self.end_month_var.set(str(now.month))
        self.end_day_var.set(str(now.day))

        self.update_start_day_range()
        self.update_end_day_range()

    def set_to_full_year(self):
        """设置为全年"""
        year = int(self.start_year_var.get())
        self.start_year_var.set(str(year))
        self.start_month_var.set("1")
        self.start_day_var.set("1")

        self.end_year_var.set(str(year))
        self.end_month_var.set("12")
        self.end_day_var.set("31")

        self.update_start_day_range()
        self.update_end_day_range()

    def browse_path(self):
        """浏览保存路径"""
        path = filedialog.askdirectory(title="选择保存文件夹")
        if path:
            self.path_var.set(path)

    def on_ok(self):
        """确定按钮点击事件"""
        try:
            # 获取开始日期
            start_year = int(self.start_year_var.get())
            start_month = int(self.start_month_var.get())
            start_day = int(self.start_day_var.get())
            self.start_date = datetime(start_year, start_month, start_day)

            # 获取结束日期
            end_year = int(self.end_year_var.get())
            end_month = int(self.end_month_var.get())
            end_day = int(self.end_day_var.get())
            self.end_date = datetime(end_year, end_month, end_day)

            # 获取保存路径
            self.output_dir = self.path_var.get()

            # 验证日期范围
            if self.start_date > self.end_date:
                messagebox.showerror("错误", "开始日期不能晚于结束日期！")
                return

            if not self.output_dir:
                messagebox.showerror("错误", "请选择保存路径！")
                return

            self.result = (self.start_date, self.end_date, self.output_dir)
            self.dialog.destroy()

        except ValueError as e:
            messagebox.showerror("错误", f"日期格式错误: {e}")
        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")

    def on_cancel(self):
        """取消按钮点击事件"""
        self.result = None
        self.dialog.destroy()

    def center_dialog(self):
        """居中对话框"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')

    def get_result(self):
        """获取对话框结果"""
        self.parent.wait_window(self.dialog)
        return self.result


class ProgressDialog:
    """进度条对话框"""

    def __init__(self, parent, title="处理中..."):
        self.parent = parent

        # 创建对话框
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x200")
        self.dialog.resizable(False, False)
        self.dialog.grab_set()  # 使对话框模态

        # 创建部件
        self.create_widgets()
        self.center_dialog()

    def create_widgets(self):
        """创建对话框部件"""
        # 标题
        self.title_label = tk.Label(self.dialog, text="正在处理数据，请稍候...", font=("Arial", 12, "bold"))
        self.title_label.pack(pady=20)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.dialog,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(pady=10)

        # 进度文本
        self.progress_text = tk.Label(self.dialog, text="0%")
        self.progress_text.pack(pady=5)

        # 状态文本
        self.status_label = tk.Label(self.dialog, text="准备开始...", wraplength=350)
        self.status_label.pack(pady=10)

    def update_progress(self, progress, status=""):
        """更新进度"""
        self.progress_var.set(progress)
        self.progress_text.config(text=f"{int(progress)}%")
        if status:
            self.status_label.config(text=status)
        self.dialog.update()

    def set_title(self, title):
        """设置标题"""
        self.dialog.title(title)
        self.title_label.config(text=title)

    def close(self):
        """关闭对话框"""
        self.dialog.destroy()

    def center_dialog(self):
        """居中对话框"""
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')


def get_nasa_power_data(start_date, end_date, progress_callback=None):
    """
    从NASA POWER API获取指定日期范围的数据
    """
    # 广州花都经纬度
    latitude, longitude = 23.4, 113.2

    # 构建API URL
    base_url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

    # 格式化日期
    start_str = start_date.strftime('%Y%m%d')
    end_str = end_date.strftime('%Y%m%d')

    params = {
        'parameters': 'T2M,ALLSKY_SFC_SW_DWN,ALLSKY_KT,WS10M,RH2M,PS,PRECTOTCORR',
        'community': 'RE',
        'longitude': longitude,
        'latitude': latitude,
        'start': start_str,
        'end': end_str,
        'format': 'JSON',
        'user': 'anonymous'
    }

    if progress_callback:
        progress_callback(10, f"正在从NASA POWER获取数据 ({start_str} 到 {end_str})...")

    print(f"正在从NASA POWER获取数据 ({start_str} 到 {end_str})...")
    print(f"请求URL: {base_url}")
    print(f"请求参数: {params}")

    try:
        response = requests.get(base_url, params=params, timeout=120)

        if response.status_code == 200:
            data = response.json()
            print("✓ 数据获取成功")

            if progress_callback:
                progress_callback(20, "数据获取成功，开始处理...")

            return data
        else:
            print(f"✗ API请求失败: {response.status_code}")
            print(f"响应内容: {response.text[:500]}")
            return None

    except Exception as e:
        print(f"✗ 获取数据时出错: {e}")
        return None


def process_nasa_data(raw_data, start_date, end_date, progress_callback=None):
    """
    处理NASA POWER数据
    """
    if raw_data is None:
        return None

    try:
        if progress_callback:
            progress_callback(30, "正在处理NASA POWER数据...")

        # 提取参数
        parameters = raw_data['properties']['parameter']

        # 获取时间戳
        timestamps = list(parameters['T2M'].keys())

        if not timestamps:
            print("✗ 没有获取到时间戳数据")
            return None

        print(f"处理时间戳数量: {len(timestamps)}")

        # 创建时间索引
        time_index = pd.to_datetime(timestamps, format='%Y%m%d%H')

        # 创建DataFrame
        weather_data = pd.DataFrame({
            'temp_air': [parameters['T2M'][ts] for ts in timestamps],
            'ghi': [parameters['ALLSKY_SFC_SW_DWN'][ts] for ts in timestamps],
            'clearness_index': [parameters['ALLSKY_KT'][ts] for ts in timestamps],
            'wind_speed': [parameters['WS10M'][ts] for ts in timestamps],
            'relative_humidity': [parameters['RH2M'][ts] for ts in timestamps],
            'pressure': [parameters['PS'][ts] for ts in timestamps],
            'precipitation': [parameters['PRECTOTCORR'][ts] for ts in timestamps]
        }, index=time_index)

        weather_data.index.name = 'time'

        print(f"✓ 数据形状: {weather_data.shape}")
        print(f"✓ 时间范围: {weather_data.index[0]} 到 {weather_data.index[-1]}")

        # 检查数据是否包含请求的时间范围
        if weather_data.index[0] > pd.Timestamp(start_date) or weather_data.index[-1] < pd.Timestamp(end_date):
            print(f"⚠ 警告: 返回数据的时间范围与请求不完全一致")

        if progress_callback:
            progress_callback(40, f"数据处理完成，共{len(weather_data)}个数据点")

        return weather_data

    except Exception as e:
        print(f"✗ 数据处理出错: {e}")
        return None


def calculate_solar_position(weather_data, latitude, longitude, progress_callback=None):
    """
    使用pvlib计算太阳位置
    """
    if progress_callback:
        progress_callback(50, "正在计算太阳位置...")

    print("正在计算太阳位置...")

    try:
        # 计算太阳位置
        solar_pos = solarposition.get_solarposition(
            time=weather_data.index,
            latitude=latitude,
            longitude=longitude,
            altitude=50,  # 海拔约50米
            pressure=weather_data['pressure'].values * 0.1  # 转换为hPa
        )

        print(f"✓ 太阳位置计算完成，形状: {solar_pos.shape}")

        if progress_callback:
            progress_callback(60, "太阳位置计算完成")

        return solar_pos

    except Exception as e:
        print(f"✗ 计算太阳位置出错: {e}")
        return None


def calculate_irradiance_components(weather_data, solar_pos, latitude, longitude, progress_callback=None):
    """
    使用pvlib计算DNI和DHI
    """
    if progress_callback:
        progress_callback(65, "正在计算辐照度分量...")

    print("正在计算辐照度分量...")

    try:
        # 使用pvlib的disc模型计算DNI
        disc_result = irradiance.disc(
            ghi=weather_data['ghi'].values,
            solar_zenith=solar_pos['zenith'].values,
            datetime_or_doy=weather_data.index,
            pressure=weather_data['pressure'].values * 0.1
        )

        # 获取DNI
        weather_data['dni'] = disc_result['dni']

        # 计算DHI = GHI - DNI * cos(zenith)
        cos_zenith = np.cos(np.radians(solar_pos['zenith'].values))
        weather_data['dhi'] = weather_data['ghi'] - weather_data['dni'] * cos_zenith

        # 确保非负
        weather_data['dni'] = weather_data['dni'].clip(lower=0)
        weather_data['dhi'] = weather_data['dhi'].clip(lower=0)

        print("✓ 辐照度分量计算完成")
        print(f"  DNI范围: {weather_data['dni'].min():.1f} 到 {weather_data['dni'].max():.1f} W/m²")
        print(f"  DHI范围: {weather_data['dhi'].min():.1f} 到 {weather_data['dhi'].max():.1f} W/m²")

        if progress_callback:
            progress_callback(70, "辐照度分量计算完成")

        return weather_data

    except Exception as e:
        print(f"✗ 计算辐照度分量出错: {e}")
        return weather_data


def create_15min_data(weather_data, progress_callback=None):
    """
    创建15分钟分辨率数据
    """
    if progress_callback:
        progress_callback(75, "正在创建15分钟分辨率数据...")

    print("正在创建15分钟分辨率数据...")

    try:
        # 重采样到15分钟频率
        weather_15min = weather_data.resample('15min').asfreq()

        # 对数值列进行线性插值
        numeric_cols = weather_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            weather_15min[col] = weather_15min[col].interpolate(method='linear')

        print(f"✓ 15分钟数据创建完成")
        print(f"  原始数据形状: {weather_data.shape}")
        print(f"  15分钟数据形状: {weather_15min.shape}")

        if progress_callback:
            progress_callback(80, f"15分钟数据创建完成，共{len(weather_15min)}个数据点")

        return weather_15min

    except Exception as e:
        print(f"✗ 创建15分钟数据出错: {e}")
        return weather_data.resample('15min').interpolate(method='linear')


def prepare_data_for_export(df, is_hourly=True):
    """
    准备导出数据，将列名改为中英文对照
    """
    if df is None or len(df) == 0:
        print("✗ 没有数据可导出")
        return None

    # 创建数据副本
    df_export = df.copy()

    # 重置索引，将时间列变成普通列
    df_export = df_export.reset_index()

    # 列名映射：英文 -> 中英文对照
    column_mapping = {
        'time': 'time(时间)',
        'temp_air': 'temp_air(气温_℃)',
        'ghi': 'ghi(水平面总辐射_W/m²)',
        'clearness_index': 'clearness_index(晴空指数)',
        'wind_speed': 'wind_speed(风速_m/s)',
        'relative_humidity': 'relative_humidity(相对湿度_%)',
        'pressure': 'pressure(气压_hPa)',
        'precipitation': 'precipitation(降水量_mm)',
        'dni': 'dni(法向直射辐射_W/m²)',
        'dhi': 'dhi(水平散射辐射_W/m²)'
    }

    # 只重命名存在的列
    existing_columns = {}
    for eng_name, chi_name in column_mapping.items():
        if eng_name in df_export.columns:
            existing_columns[eng_name] = chi_name

    # 重命名列
    df_export.rename(columns=existing_columns, inplace=True)

    return df_export


def save_data_to_csv(weather_data, weather_15min, start_date, end_date, output_dir, progress_callback=None):
    """
    保存数据到CSV文件
    """
    if progress_callback:
        progress_callback(85, "正在准备导出数据...")

    if weather_data is None or len(weather_data) == 0:
        print("✗ 没有数据可保存")
        return None, None

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 准备导出数据（中英文列名）
    weather_data_export = prepare_data_for_export(weather_data, is_hourly=True)
    weather_15min_export = prepare_data_for_export(weather_15min, is_hourly=False)

    if weather_data_export is None or weather_15min_export is None:
        return None, None

    # 生成文件名（包含日期范围）
    date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if progress_callback:
        progress_callback(90, "正在保存小时数据...")

    # 保存小时数据
    hour_filename = f"guangzhou_solar_data_hourly_{date_range_str}_{timestamp}.csv"
    hour_path = os.path.join(output_dir, hour_filename)

    try:
        weather_data_export.to_csv(hour_path, index=False, encoding='utf-8-sig')
        print(f"✓ 小时数据已保存: {hour_path}")
        print(f"  文件大小: {os.path.getsize(hour_path) / 1024:.1f} KB")

    except Exception as e:
        print(f"✗ 保存小时数据失败: {e}")
        hour_path = None

    if progress_callback:
        progress_callback(95, "正在保存15分钟数据...")

    # 保存15分钟数据
    min15_filename = f"guangzhou_solar_data_15min_{date_range_str}_{timestamp}.csv"
    min15_path = os.path.join(output_dir, min15_filename)

    try:
        weather_15min_export.to_csv(min15_path, index=False, encoding='utf-8-sig')
        print(f"✓ 15分钟数据已保存: {min15_path}")
        print(f"  文件大小: {os.path.getsize(min15_path) / 1024:.1f} KB")

    except Exception as e:
        print(f"✗ 保存15分钟数据失败: {e}")
        min15_path = None

    if progress_callback:
        progress_callback(100, "数据导出完成！")

    return hour_path, min15_path


def process_data_async(start_date, end_date, output_dir, progress_dialog, result_callback):
    """
    异步处理数据
    """
    try:
        # 地点参数
        latitude, longitude = 23.4, 113.2

        # 进度回调函数
        def update_progress(progress, status=""):
            progress_dialog.update_progress(progress, status)

        # 1. 获取NASA POWER数据
        raw_data = get_nasa_power_data(start_date, end_date, update_progress)
        if raw_data is None:
            result_callback(False, "无法获取NASA POWER数据，请检查网络连接或日期范围是否有效。")
            return

        # 2. 处理数据
        weather_data = process_nasa_data(raw_data, start_date, end_date, update_progress)
        if weather_data is None or len(weather_data) == 0:
            result_callback(False, "数据处理失败，可能没有该日期范围的数据。")
            return

        # 3. 计算太阳位置
        solar_pos = calculate_solar_position(weather_data, latitude, longitude, update_progress)
        if solar_pos is None:
            print("计算太阳位置失败，使用默认值继续")

        # 4. 计算辐照度分量
        weather_data = calculate_irradiance_components(weather_data, solar_pos, latitude, longitude, update_progress)

        # 5. 创建15分钟数据
        weather_15min = create_15min_data(weather_data, update_progress)

        # 6. 保存数据
        hour_path, min15_path = save_data_to_csv(weather_data, weather_15min, start_date, end_date, output_dir,
                                                 update_progress)

        if hour_path and min15_path:
            # 获取文件大小
            hour_size = os.path.getsize(hour_path) / 1024
            min15_size = os.path.getsize(min15_path) / 1024

            result_callback(True,
                            f"数据导出成功！\n\n"
                            f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}\n"
                            f"小时数据: {os.path.basename(hour_path)} ({hour_size:.1f} KB)\n"
                            f"15分钟数据: {os.path.basename(min15_path)} ({min15_size:.1f} KB)\n\n"
                            f"数据统计:\n"
                            f"- 小时数据点: {len(weather_data)}\n"
                            f"- 15分钟数据点: {len(weather_15min)}\n"
                            f"- 实际时间范围: {weather_data.index[0].strftime('%Y-%m-%d %H:%M')} 到 {weather_data.index[-1].strftime('%Y-%m-%d %H:%M')}\n"
                            f"- 保存路径: {output_dir}"
                            )
        else:
            result_callback(False, "数据保存失败")

    except Exception as e:
        result_callback(False, f"处理过程中发生错误: {str(e)}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("广州花都太阳能数据获取与处理系统")
    print("=" * 60)

    # 创建隐藏的根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    root.title("太阳能数据导出系统")

    # 1. 显示导出配置对话框
    print("显示导出配置对话框...")
    config_dialog = ExportConfigDialog(root)
    config_result = config_dialog.get_result()

    if config_result is None:
        print("用户取消了导出操作")
        root.destroy()
        return

    start_date, end_date, output_dir = config_result

    print(f"导出配置:")
    print(f"  开始日期: {start_date.strftime('%Y-%m-%d')}")
    print(f"  结束日期: {end_date.strftime('%Y-%m-%d')}")
    print(f"  保存路径: {output_dir}")

    # 验证日期范围
    if start_date > end_date:
        messagebox.showerror("错误", "开始日期不能晚于结束日期！")
        root.destroy()
        return

    # 2. 显示进度条对话框
    print("显示进度条对话框...")
    title = f"正在导出 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据"
    progress_dialog = ProgressDialog(root, title)

    # 3. 在后台线程中处理数据
    def on_processing_complete(success, message):
        # 关闭进度条对话框
        progress_dialog.close()

        # 显示结果对话框
        if success:
            messagebox.showinfo("导出成功", message)
            print("\n" + "=" * 60)
            print("数据导出成功！")
            print("=" * 60)
        else:
            messagebox.showerror("导出失败", message)
            print("\n" + "=" * 60)
            print("数据导出失败！")
            print("=" * 60)

        # 关闭主窗口
        root.quit()
        root.destroy()

    # 启动处理线程
    process_thread = threading.Thread(
        target=process_data_async,
        args=(start_date, end_date, output_dir, progress_dialog, on_processing_complete)
    )
    process_thread.daemon = True
    process_thread.start()

    # 启动主事件循环
    root.mainloop()


if __name__ == "__main__":
    main()
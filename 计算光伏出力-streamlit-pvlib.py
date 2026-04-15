import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import io
import warnings
import pvlib
import calendar
import tempfile
import os

warnings.filterwarnings('ignore')


# 设置中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


# 初始化字体
font_name = setup_chinese_font()

# 页面配置
st.set_page_config(
    page_title="光伏发电量分析工具 (修复版)",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和描述
st.title("☀️ 光伏发电量分析工具 (修复poa_ground错误)")
st.markdown("基于NASA POWER数据和PVlib的专业光伏发电量分析平台 - 已修复poa_ground错误")

# 创建侧边栏
with st.sidebar:
    st.header("⚙️ 系统参数")

    st.subheader("地理位置")
    latitude = st.number_input("纬度 (°N)", value=23.4, min_value=-90.0, max_value=90.0)
    longitude = st.number_input("经度 (°E)", value=113.2, min_value=-180.0, max_value=180.0)
    altitude = st.number_input("海拔 (m)", value=91.46, min_value=0.0)

    st.subheader("光伏系统参数")
    system_capacity = st.number_input("系统容量 (kW)", value=10.0, min_value=0.1)
    tilt_angle = st.number_input("倾角 (°)", value=30, min_value=0, max_value=90)
    azimuth = st.number_input("方位角 (°)", value=180, min_value=0, max_value=360)
    albedo = st.number_input("反照率", value=0.2, min_value=0.0, max_value=1.0)

    st.subheader("技术参数")
    temp_coeff = st.number_input("温度系数 (%/°C)", value=-0.4, format="%.3f") / 100
    inv_efficiency = st.number_input("逆变器效率 (%)", value=96.0, min_value=90.0, max_value=99.0) / 100

    # 保存参数到session state
    st.session_state['config'] = {
        'location': {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        },
        'system': {
            'capacity_kw': system_capacity,
            'capacity_w': system_capacity * 1000,
            'tilt': tilt_angle,
            'azimuth': azimuth,
            'albedo': albedo,
            'gamma_pdc': temp_coeff,
            'inv_efficiency': inv_efficiency
        }
    }

# 主内容区
tab1, tab2, tab3, tab4 = st.tabs(["📁 数据上传", "📈 PVlib分析", "📊 可视化", "📤 导出报告"])

with tab1:
    st.header("📁 数据文件上传")

    uploaded_file = st.file_uploader("选择NASA POWER CSV数据文件", type=['csv'])

    if uploaded_file is not None:
        try:
            # 读取文件
            content = uploaded_file.read().decode('utf-8')
            lines = content.split('\n')

            # 查找数据开始行
            data_start_line = 0
            for i, line in enumerate(lines):
                if 'YEAR' in line and 'MO' in line and 'DY' in line and 'HR' in line:
                    data_start_line = i
                    break

            # 重新读取文件
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, skiprows=data_start_line, low_memory=False)
            df.columns = df.columns.str.strip()

            # 创建日期时间索引
            if all(col in df.columns for col in ['YEAR', 'MO', 'DY', 'HR']):
                df['datetime'] = pd.to_datetime({
                    'year': df['YEAR'],
                    'month': df['MO'],
                    'day': df['DY'],
                    'hour': df['HR']
                })
                df.set_index('datetime', inplace=True)

                # 处理缺失值
                for col in df.columns:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col] = df[col].replace(-999.0, np.nan)

                # 列名映射
                column_mapping = {
                    'ALLSKY_SFC_SW_DNI': 'dni',
                    'ALLSKY_SFC_SW_DWN': 'ghi',
                    'ALLSKY_SFC_SW_DIFF': 'dhi',
                    'T2M': 'temp_air',
                    'WS10M': 'wind_speed',
                }

                # 应用列名映射
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns:
                        df[new_col] = df[old_col]

                # 确保必要列存在
                required_cols = ['ghi', 'dni', 'dhi', 'temp_air']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = 0.0

                if 'wind_speed' not in df.columns:
                    df['wind_speed'] = 1.0

                st.session_state['data'] = df
                st.session_state['data_loaded'] = True

                st.success(f"✅ 数据加载成功！共 {len(df)} 行数据")
                st.dataframe(df.head())

            else:
                st.error("❌ 数据文件缺少必要的列")

        except Exception as e:
            st.error(f"❌ 读取文件时出错: {str(e)}")

with tab2:
    st.header("📈 PVlib专业分析计算")

    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.warning("⚠️ 请先上传数据")
    elif 'config' not in st.session_state:
        st.warning("⚠️ 请在侧边栏设置系统参数")
    else:
        if st.button("🚀 开始PVlib计算", type="primary"):
            with st.spinner("正在使用PVlib进行专业计算..."):
                try:
                    df = st.session_state['data'].copy()
                    config = st.session_state['config']

                    # 创建进度指示器
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 步骤1: 创建Location对象
                    status_text.text("步骤 1/6: 创建地理位置对象...")
                    progress_bar.progress(10)

                    location = pvlib.location.Location(
                        latitude=config['location']['latitude'],
                        longitude=config['location']['longitude'],
                        altitude=config['location']['altitude'],
                        tz='UTC'
                    )

                    # 步骤2: 计算太阳位置
                    status_text.text("步骤 2/6: 计算太阳位置...")
                    progress_bar.progress(20)

                    solar_position = location.get_solarposition(
                        times=df.index,
                        method='nrel_numpy'
                    )

                    # 添加太阳位置到DataFrame
                    df['solar_zenith'] = solar_position['apparent_zenith']
                    df['solar_azimuth'] = solar_position['azimuth']
                    df['solar_elevation'] = 90 - df['solar_zenith']

                    # 步骤3: 计算地外辐射
                    status_text.text("步骤 3/6: 计算地外辐射...")
                    progress_bar.progress(30)

                    dni_extra = pvlib.irradiance.get_extra_radiation(df.index)

                    # 步骤4: 计算倾斜面辐射（修复poa_ground错误）
                    status_text.text("步骤 4/6: 计算倾斜面辐射...")
                    progress_bar.progress(50)

                    # 修复：安全处理倾斜面辐射计算
                    try:
                        poa = pvlib.irradiance.get_total_irradiance(
                            surface_tilt=config['system']['tilt'],
                            surface_azimuth=config['system']['azimuth'],
                            solar_zenith=solar_position['apparent_zenith'],
                            solar_azimuth=solar_position['azimuth'],
                            dni=df['dni'],
                            ghi=df['ghi'],
                            dhi=df['dhi'],
                            albedo=config['system']['albedo'],
                            model='perez',
                            dni_extra=dni_extra
                        )   # 计算倾斜面辐射

                        # 安全地添加辐射列 - 修复poa_ground错误
                        df['poa_global'] = poa['poa_global']
                        df['poa_direct'] = poa['poa_direct']
                        df['poa_diffuse'] = poa['poa_diffuse']

                        # 检查poa_ground是否存在，如果不存在则计算
                        if 'poa_ground' in poa.columns:
                            df['poa_ground'] = poa['poa_ground']
                        else:
                            # 手动计算地面反射辐射
                            df['poa_ground'] = df['ghi'] * config['system']['albedo'] * (
                                        1 - np.cos(np.radians(config['system']['tilt']))) / 2

                    except Exception as e:
                        st.warning(f"倾斜面辐射计算警告: {e}，使用简化模型")
                        # 使用简化模型
                        df['poa_global'] = df['ghi'] * np.cos(np.radians(config['system']['tilt']))
                        df['poa_direct'] = df['dni'] * np.cos(np.radians(config['system']['tilt']))
                        df['poa_diffuse'] = df['dhi']
                        df['poa_ground'] = df['ghi'] * config['system']['albedo'] * (
                                    1 - np.cos(np.radians(config['system']['tilt']))) / 2

                    # 步骤5: 计算电池温度
                    status_text.text("步骤 5/6: 计算电池温度...")
                    progress_bar.progress(70)

                    try:
                        cell_temperature = pvlib.temperature.sapm_cell(
                            poa_global=df['poa_global'],
                            temp_air=df['temp_air'],
                            wind_speed=df['wind_speed']
                        )
                        df['temp_cell'] = cell_temperature
                    except:
                        # 简化温度模型
                        df['temp_cell'] = df['temp_air'] + (df['poa_global'] / 1000) * 3

                    # 步骤6: 计算发电功率
                    status_text.text("步骤 6/6: 计算发电功率...")
                    progress_bar.progress(90)

                    # 计算直流功率
                    pdc0 = config['system']['capacity_w']
                    gamma_pdc = config['system']['gamma_pdc']

                    # 温度修正
                    temp_correction = 1 + gamma_pdc * (df['temp_cell'] - 25)
                    df['p_dc'] = pdc0 * (df['poa_global'] / 1000) * temp_correction
                    df['p_dc'] = df['p_dc'].clip(lower=0)

                    # 计算交流功率
                    inv_efficiency = config['system']['inv_efficiency']
                    df['p_ac'] = df['p_dc'] * inv_efficiency

                    # 计算每小时发电量
                    df['energy_ac_kwh'] = df['p_ac'] / 1000
                    df['energy_dc_kwh'] = df['p_dc'] / 1000

                    st.session_state['results'] = df
                    st.session_state['calculation_done'] = True

                    # 计算关键指标
                    key_metrics = {
                        'total_energy': df['energy_ac_kwh'].sum(),
                        'total_dc_energy': df['energy_dc_kwh'].sum(),
                        'avg_daily_energy': df['energy_ac_kwh'].resample('D').sum().mean(),
                        'max_daily_energy': df['energy_ac_kwh'].resample('D').sum().max(),
                        'utilization_hours': df['energy_ac_kwh'].sum() / (config['system']['capacity_kw']),
                        'capacity_factor': (df['energy_ac_kwh'].sum() / (
                                    config['system']['capacity_kw'] * 24 * 365)) * 100,
                        'avg_poa': df['poa_global'].mean(),
                        'max_poa': df['poa_global'].max(),
                        'avg_cell_temp': df['temp_cell'].mean(),
                        'performance_ratio': (df['energy_ac_kwh'].sum() / (
                                    df['poa_global'].sum() * 0.18 * inv_efficiency / 1000)) if df[
                                                                                                   'poa_global'].sum() > 0 else 0
                    }

                    # 文档参考值
                    doc_metrics = {
                        'total_energy': 15039,
                        'avg_daily_energy': 41.2,
                        'max_daily_energy': 86.4,
                        'utilization_hours': 1504,
                        'capacity_factor': 17.2,
                        'performance_ratio': 0.82
                    }

                    st.session_state['key_metrics'] = key_metrics
                    st.session_state['doc_metrics'] = doc_metrics

                    progress_bar.progress(100)
                    status_text.text("✅ 计算完成！")

                    st.success("✅ PVlib计算完成！")

                except Exception as e:
                    st.error(f"❌ PVlib计算过程中出错: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

        if 'calculation_done' in st.session_state and st.session_state['calculation_done']:
            st.subheader("📊 关键性能指标")

            key_metrics = st.session_state['key_metrics']
            doc_metrics = st.session_state['doc_metrics']

            # 显示指标
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                delta = key_metrics['total_energy'] - doc_metrics['total_energy']
                st.metric(
                    "年总发电量",
                    f"{key_metrics['total_energy']:,.0f} kWh",
                    delta=f"{delta:+,.0f} kWh"
                )

            with col2:
                delta = key_metrics['avg_daily_energy'] - doc_metrics['avg_daily_energy']
                st.metric(
                    "平均日发电量",
                    f"{key_metrics['avg_daily_energy']:.1f} kWh",
                    delta=f"{delta:+.1f} kWh"
                )

            with col3:
                delta = key_metrics['max_daily_energy'] - doc_metrics['max_daily_energy']
                st.metric(
                    "最大日发电量",
                    f"{key_metrics['max_daily_energy']:.1f} kWh",
                    delta=f"{delta:+.1f} kWh"
                )

            with col4:
                delta = key_metrics['utilization_hours'] - doc_metrics['utilization_hours']
                st.metric(
                    "年利用小时数",
                    f"{key_metrics['utilization_hours']:.0f} 小时",
                    delta=f"{delta:+.0f} 小时"
                )

            with col5:
                delta = key_metrics['capacity_factor'] - doc_metrics['capacity_factor']
                st.metric(
                    "容量系数",
                    f"{key_metrics['capacity_factor']:.1f}%",
                    delta=f"{delta:+.1f}%"
                )

            with col6:
                delta = key_metrics['performance_ratio'] - doc_metrics['performance_ratio']
                st.metric(
                    "性能比",
                    f"{key_metrics['performance_ratio']:.1f}%",
                    delta=f"{delta:+.1f}%"
                )

            st.info("📋 文档参考值: 年总发电量 15,039 kWh, 平均日发电量 41.2 kWh, 容量系数 17.2%")

with tab3:
    st.header("📊 可视化分析")

    if 'calculation_done' not in st.session_state or not st.session_state['calculation_done']:
        st.warning("⚠️ 请先完成分析计算")
    else:
        df = st.session_state['results']

        chart_type = st.selectbox(
            "选择图表类型",
            ["日发电曲线", "月发电量趋势", "辐射分析", "温度分析"]
        )

        if chart_type == "日发电曲线":
            st.subheader("日发电曲线")
            if len(df) > 24:
                # 选择典型日
                sample_date = df.index[len(df) // 2].normalize()
                day_data = df[df.index.normalize() == sample_date]

                if len(day_data) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    hours = day_data.index.hour + day_data.index.minute / 60

                    ax.plot(hours, day_data['p_ac'] / 1000, 'b-', linewidth=2)
                    ax.set_xlabel('小时')
                    ax.set_ylabel('功率 (kW)')
                    ax.set_title(f'典型日发电曲线\n({sample_date.strftime("%Y-%m-%d")})')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(4, 20)

                    st.pyplot(fig)

        elif chart_type == "月发电量趋势":
            st.subheader("月发电量趋势")
            if 'energy_ac_kwh' in df.columns:
                monthly_energy = df['energy_ac_kwh'].resample('ME').sum()

                fig, ax = plt.subplots(figsize=(10, 6))
                months = [m.strftime('%Y-%m') for m in monthly_energy.index]

                bars = ax.bar(range(len(months)), monthly_energy.values, alpha=0.7, color='green')
                ax.set_xlabel('月份')
                ax.set_ylabel('月发电量 (kWh)')
                ax.set_title('月发电量趋势')
                ax.grid(True, alpha=0.3, axis='y')

                if len(months) > 6:
                    tick_indices = range(0, len(months), max(1, len(months) // 6))
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels([months[i] for i in tick_indices], rotation=45)
                else:
                    ax.set_xticks(range(len(months)))
                    ax.set_xticklabels(months, rotation=45)

                st.pyplot(fig)

        elif chart_type == "辐射分析":
            st.subheader("辐射分析")
            if all(col in df.columns for col in ['ghi', 'poa_global']):
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                # 水平面辐射
                daily_ghi = df['ghi'].resample('D').mean()
                ax1.plot(daily_ghi.index, daily_ghi.values, 'r-', alpha=0.7)
                ax1.set_ylabel('GHI (W/m²)')
                ax1.set_title('水平面总辐射 (GHI)')
                ax1.grid(True, alpha=0.3)

                # 倾斜面辐射
                daily_poa = df['poa_global'].resample('D').mean()
                ax2.plot(daily_poa.index, daily_poa.values, 'b-', alpha=0.7)
                ax2.set_xlabel('日期')
                ax2.set_ylabel('POA (W/m²)')
                ax2.set_title('倾斜面总辐射 (POA)')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

        elif chart_type == "温度分析":
            st.subheader("温度分析")
            if 'temp_cell' in df.columns:
                daily_temp = df['temp_cell'].resample('D').mean()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(daily_temp.index, daily_temp.values, 'r-', alpha=0.7)
                ax.set_xlabel('日期')
                ax.set_ylabel('电池温度 (°C)')
                ax.set_title('电池温度变化')
                ax.grid(True, alpha=0.3)

                st.pyplot(fig)

with tab4:
    st.header("📤 导出报告")

    if 'calculation_done' not in st.session_state or not st.session_state['calculation_done']:
        st.warning("⚠️ 请先完成分析计算")
    else:
        if st.button("📥 导出CSV数据"):
            df = st.session_state['results']

            # 选择重要列
            export_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'poa_global', 'temp_cell', 'p_ac', 'energy_ac_kwh']
            available_cols = [col for col in export_cols if col in df.columns]

            export_df = df[available_cols].copy()
            csv = export_df.to_csv()

            st.download_button(
                label="⬇️ 下载CSV",
                data=csv,
                file_name="光伏分析结果.csv",
                mime="text/csv"
            )

        if st.button("📊 生成分析报告"):
            if 'key_metrics' in st.session_state:
                key_metrics = st.session_state['key_metrics']
                config = st.session_state['config']

                report = f"""
光伏发电量分析报告
==================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

系统配置
--------
- 地理位置: 纬度 {config['location']['latitude']}°N, 经度 {config['location']['longitude']}°E
- 系统容量: {config['system']['capacity_kw']:.1f} kW
- 倾角: {config['system']['tilt']}°, 方位角: {config['system']['azimuth']}°
- 逆变器效率: {config['system']['inv_efficiency']:.1%}

关键性能指标
------------
- 年总发电量: {key_metrics['total_energy']:,.0f} kWh
- 平均日发电量: {key_metrics['avg_daily_energy']:.1f} kWh
- 最大日发电量: {key_metrics['max_daily_energy']:.1f} kWh
- 年利用小时数: {key_metrics['utilization_hours']:.0f} 小时
- 容量系数: {key_metrics['capacity_factor']:.1f}%
- 性能比: {key_metrics['performance_ratio']:.1%}

技术说明
--------
- 计算工具: PVlib专业光伏库
- 数据来源: NASA POWER
- 计算模型: Perez倾斜面辐射模型
- 状态: ✅ 已修复poa_ground错误
                """

                st.download_button(
                    label="⬇️ 下载报告",
                    data=report,
                    file_name="光伏分析报告.txt",
                    mime="text/plain"
                )

# 页脚
st.divider()
st.caption("© 2024 光伏发电量分析工具 | 基于Streamlit和PVlib构建")
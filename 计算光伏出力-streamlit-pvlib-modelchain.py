import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import pvlib
from pvlib.modelchain import ModelChain
from pvlib.location import Location
from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import warnings
import io

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
st.title("☀️ 光伏发电量分析工具")
st.markdown("基于Streamlit和PVlib ModelChain的专业光伏发电量分析平台 - 修复版")

# 创建侧边栏
with st.sidebar:
    st.header("⚙️ 系统参数配置")

    st.subheader("📍 地理位置")
    latitude = st.number_input("纬度 (°N)", value=23.4, min_value=-90.0, max_value=90.0)
    longitude = st.number_input("经度 (°E)", value=113.2, min_value=-180.0, max_value=180.0)
    altitude = st.number_input("海拔 (m)", value=91.46, min_value=0.0)

    st.subheader("🔧 光伏系统参数")
    system_capacity = st.number_input("系统容量 (kW)", value=10.0, min_value=0.1)
    tilt_angle = st.number_input("倾角 (°)", value=30, min_value=0, max_value=90)
    azimuth = st.number_input("方位角 (°)", value=180, min_value=0, max_value=360)
    albedo = st.number_input("反照率", value=0.2, min_value=0.0, max_value=1.0)

    st.subheader("🔬 技术参数")
    module_type = st.selectbox("组件技术", ['monoSi', 'multiSi', 'thin_film'])
    temp_model = st.selectbox("温度模型类型",
                              ['open_rack_glass_polymer', 'close_mount_glass_polymer', 'open_rack_glass_glass'])
    temp_coeff = st.number_input("温度系数 (%/°C)", value=-0.4, format="%.3f") / 100
    inv_efficiency = st.number_input("逆变器效率 (%)", value=96.0, min_value=90.0, max_value=99.0) / 100

    # 逆变器选择
    st.subheader("🔌 逆变器参数")
    inverter_option = st.radio("逆变器配置", ["自动选择", "手动配置"])

    if inverter_option == "手动配置":
        inv_power = st.number_input("逆变器额定功率 (kW)", value=10.0, min_value=1.0)
    else:
        inv_power = system_capacity

    # 不确定系数
    st.subheader("📊 不确定系数")
    uncertainty_factor = st.number_input("不确定系数", value=0.8, min_value=0.0, max_value=1.0, step=0.05)

    # 保存参数到session state
    st.session_state['config'] = {
        'location': {
            'latitude': latitude,
            'longitude': longitude,
            'altitude': altitude
        },
        'system': {
            'capacity_kw': system_capacity,
            'tilt': tilt_angle,
            'azimuth': azimuth,
            'albedo': albedo,
            'gamma_pdc': temp_coeff,
            'inv_efficiency': inv_efficiency,
            'module_type': module_type,
            'temp_model': temp_model,
            'inverter_power': inv_power,
            'uncertainty_factor': uncertainty_factor
        }
    }

# 创建选项卡
tab1, tab2, tab3, tab4 = st.tabs(["📁 数据上传", "🚀 ModelChain计算", "📊 结果分析", "📤 导出报告"])

with tab1:
    st.header("📁 NASA POWER数据上传")

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

                df.index = df.index.tz_localize('Asia/Shanghai')



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

                # 显示数据预览
                with st.expander("📊 数据预览"):
                    st.dataframe(df.head())

                # 显示数据统计
                st.subheader("📈 数据统计信息")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("GHI平均值", f"{df['ghi'].mean():.1f} W/m²")
                with col2:
                    st.metric("DNI平均值", f"{df['dni'].mean():.1f} W/m²")
                with col3:
                    st.metric("温度平均值", f"{df['temp_air'].mean():.1f} °C")
                with col4:
                    st.metric("数据时间范围", f"{len(df)} 小时")

            else:
                st.error("❌ 数据文件缺少必要的列")

        except Exception as e:
            st.error(f"❌ 读取文件时出错: {str(e)}")

with tab2:
    st.header("🚀 ModelChain光伏建模计算")

    if 'data_loaded' not in st.session_state or not st.session_state['data_loaded']:
        st.warning("⚠️ 请先上传数据文件")
    elif 'config' not in st.session_state:
        st.warning("⚠️ 请在侧边栏设置系统参数")
    else:
        if st.button("🚀 开始ModelChain计算", type="primary"):
            with st.spinner("正在使用PVlib ModelChain进行专业计算..."):
                try:
                    df = st.session_state['data'].copy()
                    config = st.session_state['config']

                    # 创建进度指示器
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # 步骤1: 创建Location对象
                    status_text.text("步骤 1/6: 创建地理位置对象...")
                    progress_bar.progress(10)

                    location = Location(
                        latitude=config['location']['latitude'],
                        longitude=config['location']['longitude'],
                        altitude=config['location']['altitude'],
                        tz='Asia/Shanghai'
                    )

                    # 步骤2: 获取或创建组件参数
                    status_text.text("步骤 2/6: 配置组件参数...")
                    progress_bar.progress(20)

                    try:
                        # 尝试从数据库获取组件参数
                        sandia_modules = retrieve_sam('SandiaMod')
                        # 使用通用组件
                        module_name = 'Canadian_Solar_CS5P_220M___2009_'
                        if module_name in sandia_modules.columns:
                            module_parameters = sandia_modules[module_name]
                            st.info(f"✅ 使用组件: {module_name}")
                        else:
                            # 如果指定组件不存在，使用第一个可用组件
                            module_parameters = sandia_modules.iloc[:, 0]
                            st.warning(f"⚠️ 使用备用组件: {sandia_modules.columns[0]}")
                    except Exception as e:
                        # 创建简化组件参数
                        st.warning(f"⚠️ 无法加载组件数据库，使用简化参数: {e}")
                        module_parameters = {
                            'pdc0': config['system']['capacity_kw'] * 1000,  # 额定功率
                            'gamma_pdc': config['system']['gamma_pdc'],  # 温度系数
                            'Vmpo': 30.7,  # 最大功率点电压
                            'Impo': 7.16,  # 最大功率点电流
                            'Voc': 37.3,  # 开路电压
                            'Isc': 7.61,  # 短路电流
                        }

                    # 步骤3: 创建逆变器参数
                    status_text.text("步骤 3/6: 配置逆变器参数...")
                    progress_bar.progress(30)

                    # 创建自定义逆变器参数
                    inv_power_w = config['system']['inverter_power'] * 1000

                    inverter_parameters = {
                        'Paco': inv_power_w,  # 额定交流功率
                        'Pdco': inv_power_w,  # 额定直流功率
                        'Vdco': 400.0,  # 额定直流电压
                        'Pso': 0.0,  # 自耗功率
                        'C0': 1.0,  # 效率曲线参数
                        'C1': 0.0,
                        'C2': 0.0,
                        'C3': 0.0,
                        'Pnt': 0.0,  # 夜间损耗
                        'Vdcmax': 600.0,  # 最大直流电压
                        'Idcmax': inv_power_w / 400,  # 最大直流电流
                        'Mppt_low': 200.0,  # MPPT电压下限
                        'Mppt_high': 500.0,  # MPPT电压上限
                        'Pacmax': inv_power_w,  # 最大交流功率
                    }

                    st.info(f"✅ 使用逆变器功率: {config['system']['inverter_power']} kW")

                    # 步骤4: 创建光伏系统
                    status_text.text("步骤 4/6: 创建光伏系统...")
                    progress_bar.progress(50)

                    # 计算需要的组件数量
                    module_power = module_parameters.get('pdc0', 220)  # 默认220W
                    num_modules = int(np.ceil(config['system']['capacity_kw'] * 1000 / module_power))
                    strings_per_inverter = 1
                    modules_per_string = num_modules

                    # 修复：使用正确的温度模型参数键
                    temp_model_key = config['system']['temp_model']
                    st.info(f"✅ 使用温度模型: {temp_model_key}")

                    system = PVSystem(
                        surface_tilt=config['system']['tilt'],
                        surface_azimuth=config['system']['azimuth'],
                        module_parameters=module_parameters,
                        inverter_parameters=inverter_parameters,
                        modules_per_string=modules_per_string,
                        strings_per_inverter=strings_per_inverter,
                        temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm'][temp_model_key]
                    )

                    # 步骤5: 创建ModelChain
                    status_text.text("步骤 5/6: 创建ModelChain...")
                    progress_bar.progress(70)

                    mc = ModelChain(
                        system,
                        location,
                        aoi_model='physical',
                        spectral_model='no_loss',
                        temperature_model='sapm'
                    )

                    # 步骤6: 准备并运行计算
                    status_text.text("步骤 6/6: 运行ModelChain计算...")
                    progress_bar.progress(80)

                    # 准备天气数据
                    weather_data = pd.DataFrame({
                        'ghi': df['ghi'],
                        'dni': df['dni'],
                        'dhi': df['dhi'],
                        'temp_air': df['temp_air'],
                        'wind_speed': df['wind_speed']
                    }, index=df.index)

                    # 处理缺失值
                    weather_data = weather_data.fillna(method='ffill').fillna(0)

                    # 运行ModelChain
                    mc.run_model(weather_data)
                    progress_bar.progress(90)

                    # 获取结果
                    results = mc.results

                    # 计算发电量
                    df['poa_global'] = results.effective_irradiance    # 提取有效辐照度
                    df['temp_cell'] = results.cell_temperature   # 计算电池温度
                    df['p_dc'] = results.dc['p_mp'] if hasattr(results.dc, 'p_mp') else results.dc

                    # 计算交流功率（考虑逆变器效率）
                    df['p_ac'] = df['p_dc'] * config['system']['inv_efficiency']

                    # 计算发电量（kWh）
                    df['energy_dc_kwh'] = df['p_dc'] / 1000
                    df['energy_ac_kwh'] = df['p_ac'] / 1000

                    # 计算考虑不确定系数的发电量
                    uncertainty_factor = config['system']['uncertainty_factor']
                    df['p_dc_uncertainty'] = df['energy_dc_kwh'] * uncertainty_factor  # 不确定直流发电量
                    df['p_ac_uncertainty'] = df['energy_ac_kwh'] * uncertainty_factor  # 不确定交流发电量

                    # 保存结果
                    st.session_state['results'] = df
                    st.session_state['modelchain'] = mc
                    st.session_state['calculation_done'] = True

                    # 计算关键指标
                    total_energy = df['energy_ac_kwh'].sum()
                    total_dc_energy = df['energy_dc_kwh'].sum()
                    avg_daily_energy = df['energy_ac_kwh'].resample('D').sum().mean()
                    max_daily_energy = df['energy_ac_kwh'].resample('D').sum().max()
                    utilization_hours = total_energy / config['system']['capacity_kw']
                    capacity_factor = (total_energy / (config['system']['capacity_kw'] * 24 * 365)) * 100
                    uncertainty_total_energy = total_energy * uncertainty_factor
                    uncertainty_utilization_hours = utilization_hours * uncertainty_factor

                    key_metrics = {
                        'total_energy': total_energy,
                        'total_dc_energy': total_dc_energy,
                        'avg_daily_energy': avg_daily_energy,
                        'max_daily_energy': max_daily_energy,
                        'utilization_hours': utilization_hours,
                        'capacity_factor': capacity_factor,
                        'avg_poa': df['poa_global'].mean(),
                        'max_poa': df['poa_global'].max(),
                        'avg_cell_temp': df['temp_cell'].mean(),
                        'performance_ratio': (total_energy / ((df['poa_global'] / 1000 * config['system']['capacity_kw']).sum())) if df['poa_global'].sum() > 0 else 0,
                        'uncertainty_total_energy': uncertainty_total_energy,
                        'uncertainty_utilization_hours': uncertainty_utilization_hours
                    }

                    st.session_state['key_metrics'] = key_metrics

                    progress_bar.progress(100)
                    status_text.text("✅ ModelChain计算完成！")

                    st.success(f"""
                    ✅ ModelChain计算成功完成！

                    **计算结果摘要:**
                    - 年总发电量: {key_metrics['total_energy']:,.0f} kWh
                    - 年利用小时数: {key_metrics['utilization_hours']:.0f} 小时
                    - 容量系数: {key_metrics['capacity_factor']:.1f}%
                    - 平均日发电量: {key_metrics['avg_daily_energy']:.1f} kWh
                    """)

                except Exception as e:
                    st.error(f"❌ ModelChain计算过程中出错: {str(e)}")
                    import traceback

                    st.code(traceback.format_exc())

with tab3:
    st.header("📊 计算结果分析")

    if 'calculation_done' not in st.session_state or not st.session_state['calculation_done']:
        st.warning("⚠️ 请先完成ModelChain计算")
    else:
        df = st.session_state['results']
        key_metrics = st.session_state['key_metrics']
        config = st.session_state['config']

        # 显示关键指标
        st.subheader("📈 关键性能指标")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "年总发电量",
                f"{key_metrics['total_energy']:,.0f} kWh",
                delta=f"直流: {key_metrics['total_dc_energy']:,.0f} kWh"
            )

        with col2:
            st.metric(
                "容量系数",
                f"{key_metrics['capacity_factor']:.1f}%"
            )

        with col3:
            st.metric(
                "年利用小时数",
                f"{key_metrics['utilization_hours']:.0f} 小时"
            )

        with col4:
            st.metric(
                "性能比",
                f"{key_metrics['performance_ratio']:.1%}"
            )

        col5, col6, col7, col8 = st.columns(4)

        with col5:
            st.metric(
                "平均日发电量",
                f"{key_metrics['avg_daily_energy']:.1f} kWh"
            )

        with col6:
            st.metric(
                "最大日发电量",
                f"{key_metrics['max_daily_energy']:.1f} kWh"
            )

        with col7:
            st.metric(
                "平均辐射",
                f"{key_metrics['avg_poa']:.1f} W/m²"
            )

        with col8:
            st.metric(
                "平均电池温度",
                f"{key_metrics['avg_cell_temp']:.1f} °C"
            )

        st.subheader("📈 考虑不确定参数后的关键性能指标")

        col9, col10 = st.columns(2)

        with col9:
            st.metric(
                "考虑不确定系数后的年度总发电量",
                f"{key_metrics['uncertainty_total_energy']:,.0f} kWh"
            )

        with col10:
            st.metric(
                "考虑不确定系数后的年利用小时数",
                f"{key_metrics['uncertainty_utilization_hours']:.0f} 小时"
            )

        # 可视化分析
        st.subheader("📊 发电量分析图表")

        chart_type = st.selectbox(
            "选择图表类型",
            ["发电功率曲线", "月发电量趋势", "辐射分析", "温度分析", "性能分析", "数据对比"]
        )

        # 修复：使用标志变量而不是continue
        chart_plotted = False

        if chart_type == "发电功率曲线":
            # 添加时间范围选择器
            st.subheader("📅 时间范围选择")
            
            # 获取数据的时间范围
            data_start = df.index[0].date()
            data_end = df.index[-1].date()
            
            # 创建两列用于日期选择
            date_col1, date_col2 = st.columns(2)
            
            with date_col1:
                start_date = st.date_input(
                    "开始日期",
                    value=data_start,
                    min_value=data_start,
                    max_value=data_end,
                    key="start_date_picker"
                )
            
            with date_col2:
                end_date = st.date_input(
                    "结束日期",
                    value=data_end,
                    min_value=data_start,
                    max_value=data_end,
                    key="end_date_picker"
                )

            # 根据选择的日期范围筛选数据
            # 使用带时区的datetime，与DataFrame索引的时区保持一致
            start_datetime = pd.Timestamp(datetime.combine(start_date, datetime.min.time())).tz_localize('Asia/Shanghai')
            end_datetime = pd.Timestamp(datetime.combine(end_date, datetime.max.time())).tz_localize('Asia/Shanghai')
            
            # 确保开始日期不早于结束日期
            if start_date > end_date:
                st.error("❌ 开始日期不能晚于结束日期！")
            else:
                filtered_df = df.loc[start_datetime:end_datetime]
                
                # 显示筛选后的数据范围
                st.info(f"📊 显示数据范围：{start_date} 至 {end_date}，共 {len(filtered_df)} 小时")
                
                # 绘制图表
                fig, ax = plt.subplots(figsize=(14, 6))
                
                # 使用筛选后的数据
                hours_index = range(len(filtered_df))
                
                # 绘制功率曲线
                ax.plot(hours_index, filtered_df['p_ac'] / 1000, 'b-', linewidth=0.8, alpha=0.7, label='交流功率')
                ax.plot(hours_index, filtered_df['p_dc'] / 1000, 'r--', linewidth=0.8, alpha=0.7, label='直流功率')
                
                ax.set_xlabel('时间序列 (小时)')
                ax.set_ylabel('功率 (kW)')
                ax.set_title(f'发电功率曲线 ({start_date} 至 {end_date}, 共{len(filtered_df)}小时)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 设置X轴刻度，根据数据长度动态调整
                num_hours = len(filtered_df)
                if num_hours > 0:
                    # 根据数据范围大小决定刻度间隔
                    if num_hours <= 168:  # 一周以内，显示每天
                        tick_interval = 24
                        date_format = '%m-%d'
                    elif num_hours <= 720:  # 一个月以内，显示每周
                        tick_interval = 168
                        date_format = '%m-%d'
                    else:  # 更长，显示每月
                        tick_interval = max(1, num_hours // 12)
                        date_format = '%Y-%m'
                    
                    tick_positions = list(range(0, num_hours, tick_interval))
                    tick_labels = []
                    
                    for pos in tick_positions:
                        if pos < len(filtered_df):
                            tick_labels.append(filtered_df.index[pos].strftime(date_format))
                        else:
                            tick_labels.append('')
                    
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        elif chart_type == "月发电量趋势":
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_energy = df['energy_ac_kwh'].resample('ME').sum()
            months = [m.strftime('%Y-%m') for m in monthly_energy.index]

            bars = ax.bar(range(len(months)), monthly_energy.values, alpha=0.7, color='green')
            ax.set_xlabel('月份')
            ax.set_ylabel('月发电量 (kWh)')
            ax.set_title('月发电量趋势')
            ax.grid(True, alpha=0.3, axis='y')

            for i, v in enumerate(monthly_energy.values):
                ax.text(i, v, f'{v:,.0f}', ha='center', va='bottom', fontsize=9)

            if len(months) > 6:
                tick_indices = range(0, len(months), max(1, len(months) // 6))
                ax.set_xticks(tick_indices)
                ax.set_xticklabels([months[i] for i in tick_indices], rotation=45)
            else:
                ax.set_xticks(range(len(months)))
                ax.set_xticklabels(months, rotation=45)

        elif chart_type == "辐射分析":
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_ghi = df['ghi'].resample('D').mean()
            daily_poa = df['poa_global'].resample('D').mean()

            ax.plot(daily_ghi.index, daily_ghi.values, 'r-', alpha=0.7, label='水平面辐射 (GHI)')
            ax.plot(daily_poa.index, daily_poa.values, 'b-', alpha=0.7, label='倾斜面辐射 (POA)')
            ax.set_xlabel('日期')
            ax.set_ylabel('辐射强度 (W/m²)')
            ax.set_title('辐射分析')
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif chart_type == "温度分析":
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_temp_air = df['temp_air'].resample('D').mean()
            daily_temp_cell = df['temp_cell'].resample('D').mean()

            ax.plot(daily_temp_air.index, daily_temp_air.values, 'g-', alpha=0.7, label='环境温度')
            ax.plot(daily_temp_cell.index, daily_temp_cell.values, 'r-', alpha=0.7, label='电池温度')
            ax.set_xlabel('日期')
            ax.set_ylabel('温度 (°C)')
            ax.set_title('温度分析')
            ax.legend()
            ax.grid(True, alpha=0.3)

        elif chart_type == "性能分析":
            fig, ax = plt.subplots(figsize=(12, 6))
            daily_pr = (df['energy_ac_kwh'].resample('D').sum() /
                        ((df['poa_global'] / 1000 * config['system']['capacity_kw']).resample('D').sum()))
            daily_pr = daily_pr.replace([np.inf, -np.inf], np.nan).fillna(0)

            ax.plot(daily_pr.index, daily_pr.values * 100, 'purple', alpha=0.7, linewidth=2)
            ax.axhline(y=daily_pr.mean() * 100, color='r', linestyle='--', alpha=0.5,
                       label=f'平均值: {daily_pr.mean() * 100:.1f}%')
            ax.set_xlabel('日期')
            ax.set_ylabel('性能比 (%)')
            ax.set_title('日性能比趋势')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)

        elif chart_type == "数据对比":
            # 修复：这里使用子图而不是单个图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # 上子图：辐射
            ax1.plot(df['poa_global'].resample('D').mean(), 'b-', alpha=0.7)
            ax1.set_ylabel('辐射强度 (W/m²)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_title('倾斜面辐射与发电量对比')
            ax1.grid(True, alpha=0.3)

            # 下子图：发电量
            ax2 = ax1.twinx()
            ax2.plot(df['energy_ac_kwh'].resample('D').sum(), 'r-', alpha=0.7)
            ax2.set_ylabel('日发电量 (kWh)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            # 直接显示这个图表
            st.pyplot(fig)
            chart_plotted = True

        # 如果图表还没有显示，则显示它
        if not chart_plotted and 'fig' in locals():
            st.pyplot(fig)

        # 详细数据表格
        st.subheader("📋 详细数据预览")
        display_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'poa_global', 'temp_cell', 'p_ac', 'energy_ac_kwh']
        available_cols = [col for col in display_cols if col in df.columns]

        st.dataframe(df[available_cols].head(50))

with tab4:
    st.header("📤 数据导出与报告")

    if 'calculation_done' not in st.session_state or not st.session_state['calculation_done']:
        st.warning("⚠️ 请先完成分析计算")
    else:
        df = st.session_state['results']
        key_metrics = st.session_state['key_metrics']
        config = st.session_state['config']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 导出CSV数据")
            if st.button("生成CSV数据"):
                # 选择重要列
                export_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed',
                               'poa_global', 'temp_cell', 'p_dc', 'p_ac',
                               'energy_dc_kwh', 'energy_ac_kwh', 'p_dc_uncertainty', 'p_ac_uncertainty']
                available_cols = [col for col in export_cols if col in df.columns]

                export_df = df[available_cols].copy()
                csv = export_df.to_csv()
                
                st.download_button(
                    label="⬇️ 下载CSV文件",
                    data=csv,
                    file_name=f"光伏发电量分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            st.subheader("📄 生成分析报告")
            if st.button("生成详细报告"):
                report = f"""
光伏发电量分析报告 (ModelChain版本)
==================================

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

系统配置
--------
- 地理位置: 纬度 {config['location']['latitude']}°N, 经度 {config['location']['longitude']}°E
- 海拔高度: {config['location']['altitude']} 米
- 系统容量: {config['system']['capacity_kw']:.1f} kW
- 安装参数: 倾角 {config['system']['tilt']}°, 方位角 {config['system']['azimuth']}°
- 反照率: {config['system']['albedo']}
- 温度系数: {config['system']['gamma_pdc'] * 100:.3f} %/°C
- 逆变器效率: {config['system']['inv_efficiency']:.1%}
- 温度模型: {config['system']['temp_model']}
- 组件技术: {config['system']['module_type']}
- 逆变器功率: {config['system']['inverter_power']} kW

关键性能指标
------------
- 年总发电量: {key_metrics['total_energy']:,.0f} kWh
- 年直流发电量: {key_metrics['total_dc_energy']:,.0f} kWh
- 平均日发电量: {key_metrics['avg_daily_energy']:.1f} kWh
- 最大日发电量: {key_metrics['max_daily_energy']:.1f} kWh
- 年利用小时数: {key_metrics['utilization_hours']:.0f} 小时
- 容量系数: {key_metrics['capacity_factor']:.1f}%
- 性能比: {key_metrics['performance_ratio']:.1%}

辐射与温度条件
-------------
- 平均倾斜面辐射: {key_metrics['avg_poa']:.1f} W/m²
- 最大倾斜面辐射: {key_metrics['max_poa']:.1f} W/m²
- 平均电池温度: {key_metrics['avg_cell_temp']:.1f} °C

月度发电量
---------
"""

                # 添加月度数据
                monthly_energy = df['energy_ac_kwh'].resample('ME').sum()
                for month, energy in monthly_energy.items():
                    report += f"- {month.strftime('%Y-%m')}: {energy:,.0f} kWh\n"

                report += f"""
技术说明
--------
- 计算工具: PVlib ModelChain
- 数据来源: NASA POWER
- 计算模型: 单二极管模型 + SAPM温度模型
- 计算时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 数据点数: {len(df)} 小时
"""

                st.download_button(
                    label="⬇️ 下载分析报告",
                    data=report,
                    file_name=f"光伏分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

        # 系统信息显示
        st.subheader("⚙️ 系统配置信息")
        st.json(st.session_state['config'])

# 页脚
st.divider()
st.caption("© 2024 光伏发电量分析工具 | 基于Streamlit和PVlib ModelChain构建")

# 使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 使用步骤：
    1. **数据上传**: 上传NASA POWER格式的CSV文件
    2. **参数配置**: 在侧边栏设置系统参数
    3. **模型计算**: 点击"开始ModelChain计算"按钮
    4. **结果分析**: 查看各种分析图表和指标
    5. **数据导出**: 下载计算结果和分析报告

    ### 主要修复：
    - ✅ 修复了温度模型参数错误（'monoSi'无效键问题）
    - ✅ 修复了图表显示中的`continue`语句错误
    - ✅ 增加了温度模型类型选择
    - ✅ 改进了错误处理和信息提示

    ### 温度模型说明：
    PVlib的温度模型参数使用特定的安装类型，而不是组件类型：
    - **open_rack_glass_polymer**: 开架式安装，玻璃聚合物背板
    - **close_mount_glass_polymer**: 密闭安装，玻璃聚合物背板
    - **open_rack_glass_glass**: 开架式安装，双玻组件

    ### 数据要求：
    - CSV文件必须包含: YEAR, MO, DY, HR, ALLSKY_SFC_SW_DNI, ALLSKY_SFC_SW_DWN等列
    - 时间格式: 年月日时
    - 缺失值标记: -999.0

    ### 技术支持：
    如遇问题，请检查：
    1. 确保上传正确的CSV格式文件
    2. 检查系统参数设置是否合理
    3. 确认Python环境已安装所需库
    """)
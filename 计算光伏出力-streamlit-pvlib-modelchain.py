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
import os

warnings.filterwarnings('ignore')

# ⚠️ 重要：st.set_page_config() 必须是第一个 Streamlit 命令！
st.set_page_config(
    page_title="光伏发电量分析工具",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
def setup_chinese_font():
    """设置中文字体支持"""
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Microsoft YaHei', 'SimHei', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


# 初始化字体
font_name = setup_chinese_font()

# 从Excel文件加载组件参数
MODULES_EXCEL_PATH = os.path.join(os.path.dirname(__file__), '组件参数库.xlsx')

def load_modules_from_excel():
    """从Excel文件加载组件参数库"""
    try:
        if not os.path.exists(MODULES_EXCEL_PATH):
            # 如果文件不存在，创建默认Excel文件
            st.warning(f"⚠️ 未找到组件参数文件，正在创建默认文件: {MODULES_EXCEL_PATH}")
            default_data = {
                '组件型号': ['LONGi LR5-72HPH-545M (545W)', 'LONGi LR5-72HPH-550M (550W)', 'LONGi LR5-72HPH-555M (555W)',
                            'LONGi LR5-72HPH-560M (560W)', 'LONGi LR5-72HPH-565M (565W)'],
                'pdc0': [545.0, 550.0, 555.0, 560.0, 565.0],
                'Voc': [49.65, 49.80, 49.95, 50.10, 50.30],
                'Isc': [13.92, 13.98, 14.04, 14.10, 14.16],
                'Vmp': [41.80, 41.95, 42.10, 42.25, 42.42],
                'Imp': [13.04, 13.12, 13.19, 13.26, 13.32],
                'efficiency': [21.1, 21.3, 21.5, 21.7, 21.9]
            }
            default_df = pd.DataFrame(default_data)
            default_df.to_excel(MODULES_EXCEL_PATH, index=False, engine='openpyxl')
            st.success(f"✅ 已创建默认组件参数文件，正在重新加载...")
            # 重新加载刚创建的文件
            df_modules = pd.read_excel(MODULES_EXCEL_PATH, engine='openpyxl')
        else:
            # 读取Excel文件
            df_modules = pd.read_excel(MODULES_EXCEL_PATH, engine='openpyxl')
        
        # 转换为字典格式
        modules_dict = {}
        for idx, row in df_modules.iterrows():
            module_name = row['组件型号']
            modules_dict[module_name] = {
                'pdc0': float(row['pdc0']),
                'Voc': float(row['Voc']),
                'Isc': float(row['Isc']),
                'Vmp': float(row['Vmp']),
                'Imp': float(row['Imp']),
                'efficiency': float(row['efficiency'])
            }
        
        st.success(f"✅ 成功加载 {len(modules_dict)} 个组件参数")
        return modules_dict
        
    except Exception as e:
        st.error(f"❌ 加载组件参数失败: {str(e)}")
        return {}

# 加载组件参数库
LONGI_MODULES = load_modules_from_excel()

# 从Excel文件加载站点参数
STATIONS_EXCEL_PATH = os.path.join(os.path.dirname(__file__), '站点参数库.xlsx')

def load_stations_from_excel():
    """从Excel文件加载站点参数库"""
    try:
        if not os.path.exists(STATIONS_EXCEL_PATH):
            # 如果文件不存在，创建默认Excel文件
            st.warning(f"⚠️ 未找到站点参数文件，正在创建默认文件: {STATIONS_EXCEL_PATH}")
            default_data = {
                '站点名称': ['荣拓', '丰泰'],
                '经度': [113.2, 114.1],
                '纬度': [23.4, 22.8],
                '海拔': [91.46, 50.0],
                '容量': [10.0, 20.0],
                '光伏组件': ['LONGi LR5-72HPH-550M (550W)', 'LONGi LR5-72HPH-560M (560W)'],
                '不确定系数': [0.8, 0.8]
            }
            default_df = pd.DataFrame(default_data)
            default_df.to_excel(STATIONS_EXCEL_PATH, index=False, engine='openpyxl')
            st.success(f"✅ 已创建默认站点参数文件，正在重新加载...")
            # 重新加载刚创建的文件
            df_stations = pd.read_excel(STATIONS_EXCEL_PATH, engine='openpyxl')
        else:
            # 读取Excel文件
            df_stations = pd.read_excel(STATIONS_EXCEL_PATH, engine='openpyxl')
        
        # 转换为字典格式
        stations_dict = {}
        for idx, row in df_stations.iterrows():
            station_name = row['站点名称']
            module_name = row.get('光伏组件', '')
            
            # 验证组件型号是否在组件库中存在
            if module_name and module_name not in LONGI_MODULES:
                st.warning(f"⚠️ 站点'{station_name}'的组件'{module_name}'不在组件库中，将使用默认组件")
                module_name = 'LONGi LR5-72HPH-550M (550W)'
            
            stations_dict[station_name] = {
                'longitude': float(row['经度']),
                'latitude': float(row['纬度']),
                'altitude': float(row['海拔']),
                'capacity': float(row['容量']),
                'module': module_name if module_name else 'LONGi LR5-72HPH-550M (550W)',
                'uncertainty_factor': float(row.get('不确定系数', 0.8))
            }
        
        return stations_dict
        
    except Exception as e:
        st.error(f"❌ 加载站点参数失败: {str(e)}")
        return {}

# 加载站点参数库
STATIONS = load_stations_from_excel()

# 初始化session_state中的组件参数（首次运行时）
if 'module_power' not in st.session_state:
    st.session_state['module_power'] = 550.0
    st.session_state['module_voc'] = 49.80
    st.session_state['module_isc'] = 13.98
    st.session_state['module_vmp'] = 41.95
    st.session_state['module_imp'] = 13.12
    st.session_state['module_selector'] = 'LONGi LR5-72HPH-550M (550W)'

# 标题和描述
st.title("☀️ 光伏发电量分析工具")
st.markdown("基于Streamlit和PVlib ModelChain的专业光伏发电量分析平台")

# 创建侧边栏
with st.sidebar:
    st.header("⚙️ 系统参数配置")
    
    # 站点选择
    st.subheader("🏭 站点选择")
    
    # 定义回调函数：当站点选择改变时更新参数
    def update_station_params():
        selected_station = st.session_state['station_selector']
        if selected_station in STATIONS:
            params = STATIONS[selected_station]
            st.session_state['station_longitude'] = params['longitude']
            st.session_state['station_latitude'] = params['latitude']
            st.session_state['station_altitude'] = params['altitude']
            st.session_state['station_capacity'] = params['capacity']
            st.session_state['station_uncertainty_factor'] = params.get('uncertainty_factor', 0.8)
            # 如果站点指定了组件，且该组件在组件库中，则更新组件选择和参数
            if params['module'] in LONGI_MODULES:
                st.session_state['module_selector'] = params['module']
                # 直接更新组件参数
                comp_params = LONGI_MODULES[params['module']]
                st.session_state['module_power'] = comp_params['pdc0']
                st.session_state['module_voc'] = comp_params['Voc']
                st.session_state['module_isc'] = comp_params['Isc']
                st.session_state['module_vmp'] = comp_params['Vmp']
                st.session_state['module_imp'] = comp_params['Imp']
                st.success(f"✅ 已自动选择组件: {params['module']}")
            else:
                st.warning(f"⚠️ 站点'{selected_station}'的组件'{params['module']}'不在组件库中，请手动选择组件")
    
    # 站点选择下拉框
    station_selector = st.selectbox(
        "选择站点",
        options=["自定义"] + list(STATIONS.keys()),
        index=0,  # 默认选择自定义
        key='station_selector',
        on_change=update_station_params,
        help="选择预设站点或自定义输入"
    )
    
    # 显示站点信息
    if station_selector in STATIONS:
        station_info = STATIONS[station_selector]
        st.info(f"📍 经度: {station_info['longitude']} | 纬度: {station_info['latitude']}\n🏔️ 海拔: {station_info['altitude']}m | ⚡ 容量: {station_info['capacity']}kW")

    st.subheader("📍 地理位置")
    
    # 根据站点选择决定输入框是否可编辑
    is_custom = station_selector == "自定义"
    
    # 当选择自定义时，显示站点名称输入框
    custom_station_name = None
    if is_custom:
        custom_station_name = st.text_input(
            " 站点名称",
            placeholder="请输入站点名称（如：新站点）",
            help="输入站点名称后可点击保存按钮加入站点参数库"
        )
    
    # 当选择站点时，允许编辑以修改站点参数
    allow_edit = not is_custom
    
    latitude = st.number_input(
        "纬度 (°N)", 
        value=st.session_state.get('station_latitude', 23.4) if not is_custom else 23.4, 
        min_value=-90.0, 
        max_value=90.0,
        disabled=False  # 允许编辑
    )
    longitude = st.number_input(
        "经度 (°E)", 
        value=st.session_state.get('station_longitude', 113.2) if not is_custom else 113.2, 
        min_value=-180.0, 
        max_value=180.0,
        disabled=False  # 允许编辑
    )
    altitude = st.number_input(
        "海拔 (m)", 
        value=st.session_state.get('station_altitude', 91.46) if not is_custom else 91.46, 
        min_value=0.0,
        disabled=False  # 允许编辑
    )

    st.subheader("🔧 光伏系统参数")
    system_capacity = st.number_input(
        "系统容量 (kW)", 
        value=st.session_state.get('station_capacity', 10.0) if not is_custom else 10.0, 
        min_value=0.1,
        disabled=False  # 允许编辑
    )
    tilt_angle = st.number_input("倾角 (°)", value=30, min_value=0, max_value=90)
    azimuth = st.number_input("方位角 (°)", value=180, min_value=0, max_value=360)
    albedo = st.number_input("反照率", value=0.2, min_value=0.0, max_value=1.0)

    st.subheader("🔬 技术参数")
    temp_model = st.selectbox("温度模型类型",
                              ['open_rack_glass_polymer', 'close_mount_glass_polymer', 'open_rack_glass_glass'])
    temp_coeff = st.number_input("温度系数 (%/°C)", value=-0.4, format="%.3f") / 100
    inv_efficiency = st.number_input("逆变器效率 (%)", value=96.0, min_value=90.0, max_value=99.0) / 100

    # 组件技术参数输入
    st.subheader("🔆 组件技术参数")
    
    # 定义回调函数：当组件选择改变时更新参数
    def update_module_params():
        selected_module = st.session_state['module_selector']
        if selected_module in LONGI_MODULES:
            params = LONGI_MODULES[selected_module]
            st.session_state['module_power'] = params['pdc0']
            st.session_state['module_voc'] = params['Voc']
            st.session_state['module_isc'] = params['Isc']
            st.session_state['module_vmp'] = params['Vmp']
            st.session_state['module_imp'] = params['Imp']
            st.success(f"✅ 组件参数已更新: {selected_module}")
    
    # 组件选择下拉框
    module_selector = st.selectbox(
        "选择组件型号",
        options=list(LONGI_MODULES.keys()),
        index=1,  # 默认选择550W
        key='module_selector',
        on_change=update_module_params
    )
    
    # 显示当前选择组件的效率
    if module_selector in LONGI_MODULES:
        st.info(f"💡 组件效率: {LONGI_MODULES[module_selector]['efficiency']}%")
    
    # 组件技术参数输入（选定组件型号后锁定参数）
    is_module_locked = module_selector in LONGI_MODULES
    
    module_power = st.number_input(
        "组件额定功率 (W)", 
        min_value=100.0, 
        step=10.0,
        key='module_power',
        disabled=is_module_locked
    )
    voc = st.number_input(
        "开路电压 Voc (V)", 
        min_value=10.0, 
        step=0.1,
        key='module_voc',
        disabled=is_module_locked
    )
    isc = st.number_input(
        "短路电流 Isc (A)", 
        min_value=1.0, 
        step=0.1,
        key='module_isc',
        disabled=is_module_locked
    )
    vmp = st.number_input(
        "最大功率点电压 Vmp (V)", 
        min_value=10.0, 
        step=0.1,
        key='module_vmp',
        disabled=is_module_locked
    )
    imp = st.number_input(
        "最大功率点电流 Imp (A)", 
        min_value=1.0, 
        step=0.01,
        key='module_imp',
        disabled=is_module_locked
    )
    
    # 提示用户如何修改参数
    if is_module_locked:
        st.caption("🔒 组件参数已锁定，如需修改请更改组件型号或在\"🔆 组件参数管理\"中编辑参数库")

    # 逆变器选择
    st.subheader("🔌 逆变器参数")
    inverter_option = st.radio("逆变器配置", ["自动选择", "手动配置"])

    if inverter_option == "手动配置":
        inv_power = st.number_input("逆变器额定功率 (kW)", value=10.0, min_value=1.0)
    else:
        inv_power = system_capacity

    # 不确定系数（由站点决定）
    st.subheader("📊 不确定系数")
    uncertainty_factor = st.number_input(
        "不确定系数", 
        value=st.session_state.get('station_uncertainty_factor', 0.8) if not is_custom else 0.8, 
        min_value=0.0, 
        max_value=1.0, 
        step=0.05,
        disabled=False  # 允许编辑
    )
    
    # 添加保存按钮
    if is_custom:
        # 自定义模式：添加新站点
        if st.button("➕ 添加到站点参数库", type="primary", help="将当前配置的站点加入站点参数库"):
            if not custom_station_name or custom_station_name.strip() == "":
                st.error("❌ 请先输入站点名称")
            else:
                try:
                    # 读取当前站点参数库
                    if os.path.exists(STATIONS_EXCEL_PATH):
                        df_stations = pd.read_excel(STATIONS_EXCEL_PATH, engine='openpyxl')
                    else:
                        df_stations = pd.DataFrame()
                    
                    # 检查站点名称是否已存在
                    if custom_station_name in df_stations['站点名称'].values:
                        st.error(f"❌ 站点'{custom_station_name}'已存在，请使用其他名称")
                    else:
                        # 添加新站点
                        new_row = pd.DataFrame([{
                            '站点名称': custom_station_name,
                            '经度': longitude,
                            '纬度': latitude,
                            '海拔': altitude,
                            '容量': system_capacity,
                            '光伏组件': module_selector,
                            '不确定系数': uncertainty_factor
                        }])
                        df_stations = pd.concat([df_stations, new_row], ignore_index=True)
                        
                        # 保存回Excel
                        df_stations.to_excel(STATIONS_EXCEL_PATH, index=False, engine='openpyxl')
                        
                        # 更新STATIONS字典
                        STATIONS[custom_station_name] = {
                            'longitude': longitude,
                            'latitude': latitude,
                            'altitude': altitude,
                            'capacity': system_capacity,
                            'module': module_selector,
                            'uncertainty_factor': uncertainty_factor
                        }
                        
                        st.success(f"✅ 站点'{custom_station_name}'已添加到站点参数库！")
                        st.info("💡 正在刷新页面以加载新站点...")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ 添加失败: {str(e)}")
    elif station_selector in STATIONS:
        # 预设站点模式：保存修改
        if st.button("💾 保存站点修改", type="secondary", help="将当前修改保存到站点参数库"):
            try:
                # 读取当前站点参数库
                if os.path.exists(STATIONS_EXCEL_PATH):
                    df_stations = pd.read_excel(STATIONS_EXCEL_PATH, engine='openpyxl')
                else:
                    df_stations = pd.DataFrame()
                
                # 查找要更新的站点
                station_idx = df_stations[df_stations['站点名称'] == station_selector].index
                
                if len(station_idx) > 0:
                    # 更新现有站点
                    idx = station_idx[0]
                    df_stations.at[idx, '经度'] = longitude
                    df_stations.at[idx, '纬度'] = latitude
                    df_stations.at[idx, '海拔'] = altitude
                    df_stations.at[idx, '容量'] = system_capacity
                    df_stations.at[idx, '光伏组件'] = module_selector
                    df_stations.at[idx, '不确定系数'] = uncertainty_factor
                    
                    # 保存回Excel
                    df_stations.to_excel(STATIONS_EXCEL_PATH, index=False, engine='openpyxl')
                    
                    # 更新STATIONS字典
                    STATIONS[station_selector] = {
                        'longitude': longitude,
                        'latitude': latitude,
                        'altitude': altitude,
                        'capacity': system_capacity,
                        'module': module_selector,
                        'uncertainty_factor': uncertainty_factor
                    }
                    
                    # 更新session_state
                    st.session_state['station_longitude'] = longitude
                    st.session_state['station_latitude'] = latitude
                    st.session_state['station_altitude'] = altitude
                    st.session_state['station_capacity'] = system_capacity
                    st.session_state['station_uncertainty_factor'] = uncertainty_factor
                    
                    st.success(f"✅ 站点'{station_selector}'修改已保存！")
                else:
                    st.error(f"❌ 未找到站点'{station_selector}'")
                    
            except Exception as e:
                st.error(f"❌ 保存失败: {str(e)}")

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
            'temp_model': temp_model,
            'inverter_power': inv_power,
            'uncertainty_factor': uncertainty_factor,
            # 新增组件技术参数
            'module_name': module_selector,
            'module_power': module_power,
            'Voc': voc,
            'Isc': isc,
            'Vmp': vmp,
            'Imp': imp
        }
    }

# 创建选项卡
weather_tab, module_tab, calc_tab, result_tab, export_tab, forecast_tab = st.tabs(["🌤️ 气象数据上传", "🔆 组件参数管理", "🚀 ModelChain计算", "📊 结果分析", "📤 导出报告", "🔮 光伏出力预测"])

with weather_tab:
    st.header("🌤️ 气象数据获取")
    
    # 创建两个子选项卡：下载 和 上传
    download_tab, upload_tab = st.tabs(["🌐 从NASA POWER下载", "📁 上传本地文件"])
    
    # ========== 子选项卡1: 从NASA POWER下载 ==========
    with download_tab:
        st.subheader("🌐 从NASA POWER API下载气象数据")
        st.markdown("""
        根据当前选择的**站点经纬度**和**时间范围**，直接从NASA POWER API下载气象数据。
        
        > 📌 NASA POWER提供全球太阳能和气象数据，涵盖1981年至今。
        """)
        
        # 检查是否有站点信息
        if 'station_longitude' not in st.session_state or 'station_latitude' not in st.session_state:
            st.warning("⚠️ 请先在侧边栏选择一个站点或设置经纬度")
        else:
            # 显示当前经纬度
            lat = st.session_state.get('station_latitude', 23.4)
            lon = st.session_state.get('station_longitude', 113.2)
            st.info(f"📍 当前经纬度: 纬度 {lat}°N, 经度 {lon}°E")
            
            # 时间范围选择
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "开始日期",
                    value=datetime.now() - timedelta(days=30),
                    min_value=datetime(1981, 1, 1),
                    max_value=datetime.now() - timedelta(days=2)
                )
            with col2:
                end_date = st.date_input(
                    "结束日期",
                    value=datetime.now() - timedelta(days=1),
                    min_value=datetime(1981, 1, 1),
                    max_value=datetime.now() - timedelta(days=1)
                )
            
            # 检查日期有效性
            if start_date > end_date:
                st.error("❌ 开始日期不能晚于结束日期")
            else:
                download_btn = st.button("📥 下载气象数据", type="primary")
                
                if download_btn:
                    try:
                        import requests
                        
                        with st.spinner("正在从NASA POWER下载数据，请稍候..."):
                            # 构建NASA POWER API URL
                            url = (
                                f"https://power.larc.nasa.gov/api/temporal/hourly/point"
                                f"?start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}"
                                f"&latitude={lat}&longitude={lon}"
                                f"&community=RE"
                                f"&parameters=ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF,T2M,WS10M"
                                f"&format=CSV"
                            )
                            
                            response = requests.get(url, timeout=120)
                            response.raise_for_status()
                            
                            # 解析CSV内容
                            content = response.text
                            lines = content.split('\n')
                            
                            # 查找数据开始行
                            data_start_line = 0
                            for i, line in enumerate(lines):
                                if 'YEAR' in line and 'MO' in line and 'DY' in line and 'HR' in line:
                                    data_start_line = i
                                    break
                            
                            # 读取数据
                            df = pd.read_csv(io.StringIO('\n'.join(lines[data_start_line:])), low_memory=False)
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
                                
                                # 保存结果
                                st.session_state['data'] = df
                                st.session_state['data_loaded'] = True
                                
                                st.success(f"✅ 下载成功！获取 {len(df)} 行数据 ({start_date} ~ {end_date})")
                                
                                # 显示数据预览
                                with st.expander("📊 数据预览"):
                                    st.dataframe(df.head())
                                
                                # 显示数据统计
                                st.subheader("📈 数据统计信息")
                                col_a, col_b, col_c, col_d = st.columns(4)
                                with col_a:
                                    st.metric("GHI平均值", f"{df['ghi'].mean():.1f} W/m²")
                                with col_b:
                                    st.metric("DNI平均值", f"{df['dni'].mean():.1f} W/m²")
                                with col_c:
                                    st.metric("温度平均值", f"{df['temp_air'].mean():.1f} °C")
                                with col_d:
                                    st.metric("数据时间范围", f"{len(df)} 小时")
                            else:
                                st.error("❌ 下载的数据格式不正确")
                                
                    except requests.exceptions.Timeout:
                        st.error("❌ 请求超时，NASA POWER API响应较慢，请稍后重试")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ 下载失败: {str(e)}")
                    except Exception as e:
                        st.error(f"❌ 解析数据时出错: {str(e)}")
                        with st.expander("🔍 查看详细错误"):
                            st.exception(e)
    
    # ========== 子选项卡2: 上传本地文件 ==========
    with upload_tab:
        st.subheader("📁 上传本地气象数据文件")
        st.markdown("上传NASA POWER下载的CSV格式气象数据文件")

        uploaded_file = st.file_uploader("选择NASA POWER CSV数据文件", type=['csv'], key='weather_data_upload')

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

with module_tab:
    st.header("🔆 组件参数管理")
    
    # 显示当前组件库信息
    st.info(f"📊 当前组件库包含 **{len(LONGI_MODULES)}** 个组件型号")
    
    # 上传组件参数Excel文件
    st.markdown("---")
    st.markdown("#### 📤 上传组件参数文件")
    
    uploaded_module_file = st.file_uploader(
        "选择组件参数Excel文件 (.xlsx)",
        type=['xlsx'],
        key='module_data_upload',
        help="Excel文件需包含列：组件型号、pdc0、Voc、Isc、Vmp、Imp、efficiency"
    )
    
    if uploaded_module_file is not None:
        try:
            # 读取Excel文件
            df_new_modules = pd.read_excel(uploaded_module_file, engine='openpyxl')
            
            # 检查必要的列
            required_columns = ['组件型号', 'pdc0', 'Voc', 'Isc', 'Vmp', 'Imp', 'efficiency']
            missing_columns = [col for col in required_columns if col not in df_new_modules.columns]
            
            if missing_columns:
                st.error(f"❌ Excel文件缺少必要的列: {', '.join(missing_columns)}")
                st.info(f"💡 必需的列名: {', '.join(required_columns)}")
            else:
                # 显示预览
                st.success(f"✅ 文件读取成功！检测到 **{len(df_new_modules)}** 个组件")
                
                with st.expander("📊 预览上传的数据"):
                    st.dataframe(df_new_modules)
                
                # 选择合并模式
                merge_mode = st.radio(
                    "选择合并模式",
                    ["追加到现有库", "替换整个组件库"],
                    key='merge_mode',
                    help="追加：保留现有组件，添加新组件\n替换：删除所有现有组件，只使用上传的组件"
                )
                
                # 确认按钮
                if st.button("📥 导入组件参数", type="primary", key='import_modules_btn'):
                    try:
                        if merge_mode == "追加到现有库":
                            # 追加模式：合并现有组件和新组件
                            existing_df = pd.DataFrame([
                                {'组件型号': name, **params}
                                for name, params in LONGI_MODULES.items()
                            ])
                            
                            # 合并并去重（以组件型号为准）
                            combined_df = pd.concat([existing_df, df_new_modules], ignore_index=True)
                            combined_df = combined_df.drop_duplicates(subset=['组件型号'], keep='last')
                            
                            # 保存为Excel
                            combined_df.to_excel(MODULES_EXCEL_PATH, index=False, engine='openpyxl')
                            st.success(f"✅ 成功追加 {len(df_new_modules)} 个组件！组件库现包含 {len(combined_df)} 个组件")
                        else:
                            # 替换模式：只保存新上传的组件
                            df_new_modules.to_excel(MODULES_EXCEL_PATH, index=False, engine='openpyxl')
                            st.success(f"✅ 成功替换组件库！现包含 {len(df_new_modules)} 个组件")
                        
                        st.warning("⚠️ 请刷新页面以加载新的组件参数")
                        
                    except Exception as save_error:
                        st.error(f"❌ 保存文件失败: {str(save_error)}")
                
        except Exception as e:
            st.error(f"❌ 读取Excel文件失败: {str(e)}")
    
    # 下载组件参数库
    st.markdown("---")
    st.markdown("#### 💾 导出组件参数库")
    
    if not LONGI_MODULES:
        st.info("💡 当前组件库为空，请先上传组件参数文件")
    else:
        st.info(f"📊 当前组件库包含 **{len(LONGI_MODULES)}** 个组件型号")
        
        if st.button("📥 下载组件参数库", key='download_module_library_btn'):
            # 将组件字典转换为DataFrame
            library_data = []
            for name, params in LONGI_MODULES.items():
                library_data.append({
                    '组件型号': name,
                    'pdc0': params.get('pdc0', 0),
                    'Voc': params.get('Voc', 0),
                    'Isc': params.get('Isc', 0),
                    'Vmp': params.get('Vmp', 0),
                    'Imp': params.get('Imp', 0),
                    'efficiency': params.get('efficiency', 0)
                })
            
            library_df = pd.DataFrame(library_data)
            
            # 转换为Excel字节流
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                library_df.to_excel(writer, index=False, sheet_name='组件参数库')
            output.seek(0)
            
            # 提供下载
            st.download_button(
                label="⬇️ 下载组件参数库.xlsx",
                data=output,
                file_name=f"组件参数库_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download_module_library'
            )
            
            st.success("✅ 组件参数库导出成功！")
    
    # ========== 站点参数管理 ==========
    st.markdown("---")
    st.markdown("## 🏭 站点参数管理")
    
    # 显示当前站点库信息
    st.info(f"📊 当前站点库包含 **{len(STATIONS)}** 个站点")
    
    # 上传站点参数Excel文件
    st.markdown("#### 📤 上传站点参数文件")
    
    uploaded_station_file = st.file_uploader(
        "选择站点参数Excel文件 (.xlsx)",
        type=['xlsx'],
        key='station_data_upload',
        help="Excel文件需包含列：站点名称、经度、纬度、海拔、容量、光伏组件"
    )
    
    if uploaded_station_file is not None:
        try:
            # 读取Excel文件
            df_new_stations = pd.read_excel(uploaded_station_file, engine='openpyxl')
            
            # 检查必要的列
            required_columns = ['站点名称', '经度', '纬度', '海拔', '容量']
            missing_columns = [col for col in required_columns if col not in df_new_stations.columns]
            
            if missing_columns:
                st.error(f"❌ Excel文件缺少必要的列: {', '.join(missing_columns)}")
                st.info(f"💡 必需的列名: {', '.join(required_columns)}")
            else:
                # 显示预览
                st.success(f"✅ 文件读取成功！检测到 **{len(df_new_stations)}** 个站点")
                
                with st.expander("📊 预览上传的数据"):
                    st.dataframe(df_new_stations)
                
                # 选择合并模式
                station_merge_mode = st.radio(
                    "选择合并模式",
                    ["追加到现有库", "替换整个站点库"],
                    key='station_merge_mode',
                    help="追加：保留现有站点，添加新站点\n替换：删除所有现有站点，只使用上传的站点"
                )
                
                # 确认按钮
                if st.button("📥 导入站点参数", type="primary", key='import_stations_btn'):
                    try:
                        if station_merge_mode == "追加到现有库":
                            # 追加模式：合并现有站点和新站点
                            existing_df = pd.DataFrame([
                                {'站点名称': name, **params}
                                for name, params in STATIONS.items()
                            ])
                            # 重命名列以匹配格式
                            existing_df = existing_df.rename(columns={
                                'longitude': '经度',
                                'latitude': '纬度',
                                'altitude': '海拔',
                                'capacity': '容量',
                                'module': '光伏组件'
                            })
                            
                            # 合并并去重（以站点名称为准）
                            combined_df = pd.concat([existing_df, df_new_stations], ignore_index=True)
                            combined_df = combined_df.drop_duplicates(subset=['站点名称'], keep='last')
                            
                            # 保存为Excel
                            combined_df.to_excel(STATIONS_EXCEL_PATH, index=False, engine='openpyxl')
                            st.success(f"✅ 成功追加 {len(df_new_stations)} 个站点！站点库现包含 {len(combined_df)} 个站点")
                        else:
                            # 替换模式：只保存新上传的站点
                            df_new_stations.to_excel(STATIONS_EXCEL_PATH, index=False, engine='openpyxl')
                            st.success(f"✅ 成功替换站点库！现包含 {len(df_new_stations)} 个站点")
                        
                        st.warning("⚠️ 请刷新页面以加载新的站点参数")
                        
                    except Exception as save_error:
                        st.error(f"❌ 保存文件失败: {str(save_error)}")
                
        except Exception as e:
            st.error(f"❌ 读取Excel文件失败: {str(e)}")
    
    # 下载站点参数库
    st.markdown("---")
    st.markdown("#### 💾 导出站点参数库")
    
    if not STATIONS:
        st.info("💡 当前站点库为空，请先上传站点参数文件")
    else:
        st.info(f"📊 当前站点库包含 **{len(STATIONS)}** 个站点")
        
        if st.button("📥 下载站点参数库", key='download_station_library_btn'):
            # 将站点字典转换为DataFrame
            library_data = []
            for name, params in STATIONS.items():
                library_data.append({
                    '站点名称': name,
                    '经度': params.get('longitude', 0),
                    '纬度': params.get('latitude', 0),
                    '海拔': params.get('altitude', 0),
                    '容量': params.get('capacity', 0),
                    '光伏组件': params.get('module', ''),
                    '不确定系数': params.get('uncertainty_factor', 0.8)
                })
            
            library_df = pd.DataFrame(library_data)
            
            # 转换为Excel字节流
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                library_df.to_excel(writer, index=False, sheet_name='站点参数库')
            output.seek(0)
            
            # 提供下载
            st.download_button(
                label="⬇️ 下载站点参数库.xlsx",
                data=output,
                file_name=f"站点参数库_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download_station_library'
            )
            
            st.success("✅ 站点参数库导出成功！")

with forecast_tab:
    st.header("🔮 光伏出力预测 (基于天气预报)")
    
    if not STATIONS:
        st.warning("⚠️ 请先在\"🔆 组件参数管理\"选项卡中添加站点参数")
    elif 'config' not in st.session_state:
        st.warning("⚠️ 请先在侧边栏设置系统参数（经纬度、组件型号等）")
    else:
        st.markdown("""
        本功能将根据您选择的**站点**，结合在线天气预报数据（Open-Meteo），
        对未来指定天数的光伏发电功率进行预测。支持同时预测多个站点。
        """)
        
        # 添加站点选择器（支持多选）
        st.subheader("🏭 选择预测站点")
        selected_stations = st.multiselect(
            "选择要预测的站点（可多选）",
            options=list(STATIONS.keys()),
            default=[list(STATIONS.keys())[0]] if STATIONS else [],
            help="按住 Ctrl 键可多选站点"
        )
        
        if selected_stations:
            # 添加预测天数选择器
            col1, col2 = st.columns([2, 1])
            with col1:
                predict_btn = st.button("🚀 开始获取天气并预测", type="primary")
            with col2:
                forecast_days = st.slider("预测天数", min_value=1, max_value=7, value=3, step=1, 
                                          help="选择要预测的天数（1-7天）",
                                          key="forecast_days_slider")
            
            # 将预测天数和站点保存到 session_state
            st.session_state['forecast_days'] = forecast_days
            st.session_state['selected_stations'] = selected_stations
            
            if predict_btn:
                all_results = {}
                
                # 对每个选中的站点进行预测
                for station_idx, station_name in enumerate(selected_stations):
                    station_info = STATIONS[station_name]
                    
                    with st.spinner(f"正在预测站点 {station_idx + 1}/{len(selected_stations)}: {station_name}..."):
                        try:
                            import requests
                            
                            lat = station_info['latitude']
                            lon = station_info['longitude']
                            altitude = station_info['altitude']
                            capacity = station_info['capacity']
                            module_name = station_info['module']
                            
                            # 获取组件参数
                            if module_name in LONGI_MODULES:
                                comp_params = LONGI_MODULES[module_name]
                            else:
                                st.warning(f"⚠️ 站点'{station_name}'的组件不在库中，使用默认参数")
                                comp_params = LONGI_MODULES.get('LONGi LR5-72HPH-550M (550W)', {})
                            
                            # 使用 Open-Meteo API 获取天气预报数据
                            url = (
                                f"https://api.open-meteo.com/v1/forecast?"
                                f"latitude={lat}&longitude={lon}"
                                f"&hourly=shortwave_radiation,direct_radiation,diffuse_radiation,"
                                f"temperature_2m,wind_speed_10m,cloud_cover"
                                f"&timezone=Asia%2FShanghai"
                                f"&forecast_days={forecast_days}"
                            )
                            
                            response = requests.get(url, timeout=30)
                            response.raise_for_status()
                            weather_json = response.json()
                            
                            # 解析数据
                            hourly = weather_json['hourly']
                            times = pd.to_datetime(hourly['time']).tz_localize('Asia/Shanghai')
                            
                            raw_data = pd.DataFrame({
                                'ghi': hourly['shortwave_radiation'],
                                'dni': hourly['direct_radiation'],
                                'dhi': hourly['diffuse_radiation'],
                                'temp_air': hourly['temperature_2m'],
                                'wind_speed': hourly['wind_speed_10m'],
                                'cloud_cover': hourly.get('cloud_cover', [0]*len(times)),
                            }, index=times)
                            
                            # 风速单位转换：km/h -> m/s
                            raw_data['wind_speed'] = raw_data['wind_speed'] / 3.6
                            
                            if raw_data.empty:
                                st.error(f"❌ 站点'{station_name}'未能获取到天气预报数据")
                                continue
                            
                            # 准备 ModelChain 计算
                            location = Location(
                                latitude=lat,
                                longitude=lon,
                                altitude=altitude,
                                tz='Asia/Shanghai'
                            )
                            
                            module_parameters = {
                                'pdc0': comp_params.get('pdc0', 550.0),
                                'gamma_pdc': st.session_state['config']['system']['gamma_pdc'],
                                'Vmpo': comp_params.get('Vmp', 41.95),
                                'Impo': comp_params.get('Imp', 13.12),
                                'Voc': comp_params.get('Voc', 49.80),
                                'Isc': comp_params.get('Isc', 13.98),
                            }
                            
                            inv_power_w = capacity * 1000
                            inverter_parameters = {
                                'Paco': inv_power_w,
                                'pdc0': inv_power_w,
                                'Vdco': 400.0,
                                'Pso': 0.0,
                                'C0': 1.0, 'C1': 0.0, 'C2': 0.0, 'C3': 0.0,
                                'Pnt': 0.0, 'Vdcmax': 600.0,
                                'Idcmax': inv_power_w / 400,
                                'Mppt_low': 200.0, 'Mppt_high': 500.0,
                                'Pacmax': inv_power_w,
                            }
                            
                            num_modules = int(np.ceil(capacity * 1000 / module_parameters['pdc0']))
                            system = PVSystem(
                                surface_tilt=st.session_state['config']['system']['tilt'],
                                surface_azimuth=st.session_state['config']['system']['azimuth'],
                                module_parameters=module_parameters,
                                inverter_parameters=inverter_parameters,
                                modules_per_string=num_modules,
                                strings_per_inverter=1,
                                temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm'][st.session_state['config']['system']['temp_model']]
                            )
                            
                            mc = ModelChain(
                                system, location,
                                aoi_model='physical', spectral_model='no_loss',
                                temperature_model='sapm', dc_model='pvwatts', ac_model='pvwatts'
                            )
                            
                            # 运行预测模型
                            mc.run_model(raw_data)
                            results = mc.results
                            
                            # 处理 pvwatts 模型和 sandia 模型的结果差异
                            if hasattr(results.dc, 'p_mp'):
                                dc_power = results.dc['p_mp']
                            else:
                                dc_power = results.dc
                            
                            # 整理预测结果
                            forecast_df = pd.DataFrame({
                                '预测时间': raw_data.index,
                                'GHI (W/m²)': raw_data['ghi'],
                                'DNI (W/m²)': raw_data['dni'],
                                'DHI (W/m²)': raw_data['dhi'],
                                '气温 (°C)': raw_data['temp_air'],
                                '风速 (m/s)': raw_data['wind_speed'],
                                '云量 (%)': raw_data['cloud_cover'],
                                '直流功率 (kW)': dc_power / 1000,
                                '交流功率 (kW)': results.ac / 1000
                            })
                            
                            forecast_df.set_index('预测时间', inplace=True)
                            
                            # 过滤掉当前时间之前的数据
                            current_time = pd.Timestamp.now(tz='Asia/Shanghai')
                            forecast_df = forecast_df[forecast_df.index >= current_time]
                            
                            all_results[station_name] = forecast_df
                            
                        except Exception as e:
                            st.error(f"❌ 站点'{station_name}'预测失败: {str(e)}")
                
                # 保存所有站点的结果
                if all_results:
                    st.session_state['forecast_multi_results'] = all_results
                    st.success(f"✅ 成功完成 {len(all_results)} 个站点的预测！请查看下方图表。")
            
            # 显示多站点预测结果
            if 'forecast_multi_results' in st.session_state:
                multi_results = st.session_state['forecast_multi_results']
                forecast_days_display = st.session_state.get('forecast_days', 3)
                
                # 为每个站点显示结果
                for station_name, forecast_df in multi_results.items():
                    st.divider()
                    st.subheader(f"📍 站点: {station_name}")
                    
                    # 绘图展示
                    st.subheader(f"📈 未来 {forecast_days_display} 天功率预测曲线")
                    
                    # 添加曲线类型选择器
                    curve_type_key = f"forecast_curve_type_{station_name}"
                    curve_type = st.radio(
                        "选择展示曲线", 
                        ["交流功率", "直流功率"], 
                        horizontal=True, 
                        key=curve_type_key
                    )
                    
                    # 根据选择确定数据列和颜色
                    if curve_type == "交流功率":
                        col_name = '交流功率 (kW)'
                        color = '#FF9800'
                        label = '预测交流功率'
                    else:
                        col_name = '直流功率 (kW)'
                        color = '#2196F3'
                        label = '预测直流功率'
                        
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(forecast_df.index, forecast_df[col_name], label=label, color=color, linewidth=2)
                    ax.set_xlabel('时间', fontsize=12)
                    ax.set_ylabel('功率 (kW)', fontsize=12)
                    ax.set_title(f'{station_name} - 光伏发电功率预测', fontsize=14)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    
                    # 预测数据统计
                    st.subheader("📊 预测数据概览")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("预计总发电量 (kWh)", f"{forecast_df[col_name].sum():.1f}")
                    c2.metric("峰值功率 (kW)", f"{forecast_df[col_name].max():.2f}")
                    c3.metric("平均功率 (kW)", f"{forecast_df[col_name].mean():.2f}")
                    
                    # 获取站点不确定系数
                    uncertainty = station_info.get('uncertainty_factor', 0.8)
                    
                    # 按日统计数据
                    st.subheader(f"📅 每日发电量统计 (不确定系数: {uncertainty})")
                    daily_stats = forecast_df[col_name].resample('D').agg(
                        每日发电量_kWh=('sum'),
                        每日平均功率_kW=('mean')
                    ).round(2)
                    
                    # 添加考虑不确定系数后的发电量
                    daily_stats['考虑不确定系数后_每日发电量_kWh'] = (daily_stats['每日发电量_kWh'] * uncertainty).round(2)
                    
                    # 将索引转换为列并重命名
                    daily_stats = daily_stats.reset_index()
                    daily_stats.columns = ['日期', '每日发电量_kWh', '每日平均功率_kW', '考虑不确定系数后_每日发电量_kWh']
                    
                    # 格式化日期显示
                    daily_stats['日期'] = daily_stats['日期'].dt.strftime('%Y-%m-%d')
                    
                    # 显示每日统计表格
                    st.dataframe(daily_stats, use_container_width=True, hide_index=True)
                    
                    # 提供每日统计下载
                    daily_csv = daily_stats.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label=f"⬇️ 下载 {station_name} 每日统计数据",
                        data=daily_csv,
                        file_name=f"{station_name}_每日发电统计_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    with st.expander("📋 查看详细预测数据"):
                        # 去掉时区信息，格式化时间显示
                        display_df = forecast_df.reset_index().copy()
                        display_df['预测时间'] = display_df['预测时间'].dt.strftime('%Y-%m-%d %H:%M:%S')
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # 提供下载
                        csv = display_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label=f"⬇️ 下载 {station_name} 预测结果 CSV",
                            data=csv,
                            file_name=f"光伏预测_{station_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
        else:
            st.info("💡 请至少选择一个站点进行预测")

with calc_tab:
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

                    # 使用用户输入的组件参数
                    module_parameters = {
                        'pdc0': config['system']['module_power'],  # 额定功率 (W)
                        'gamma_pdc': config['system']['gamma_pdc'],  # 温度系数
                        'Vmpo': config['system']['Vmp'],  # 最大功率点电压
                        'Impo': config['system']['Imp'],  # 最大功率点电流
                        'Voc': config['system']['Voc'],  # 开路电压
                        'Isc': config['system']['Isc'],  # 短路电流
                    }
                    st.info(f"✅ 使用用户自定义组件参数: {config['system']['module_power']}W")

                    # 步骤3: 创建逆变器参数
                    status_text.text("步骤 3/6: 配置逆变器参数...")
                    progress_bar.progress(30)

                    # 创建自定义逆变器参数
                    inv_power_w = config['system']['inverter_power'] * 1000

                    inverter_parameters = {
                        'Paco': inv_power_w,  # 额定交流功率
                        'pdc0': inv_power_w,  # 额定直流功率（pvwatts模型必须使用小写pdc0）
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
                        temperature_model='sapm',
                        dc_model='pvwatts',
                        ac_model='pvwatts'
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

with result_tab:
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
            ["发电直流功率曲线", "发电交流功率曲线", "月发电量趋势", "辐射分析", "温度分析", "性能分析", "数据对比"]
        )

        # 修复：使用标志变量而不是continue
        chart_plotted = False

        if chart_type == "发电直流功率曲线":
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
                # ax.plot(hours_index, filtered_df['p_ac'] / 1000, 'b-', linewidth=0.8, alpha=0.7, label='交流功率')
                ax.plot(hours_index, filtered_df['p_dc'] / 1000, 'r-', linewidth=0.8, alpha=0.7, label='直流功率')
                
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

        elif chart_type == "发电交流功率曲线":
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
            start_datetime = pd.Timestamp(datetime.combine(start_date, datetime.min.time())).tz_localize(
                'Asia/Shanghai')
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
                # ax.plot(hours_index, filtered_df['p_dc'] / 1000, 'r--', linewidth=0.8, alpha=0.7, label='直流功率')

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

with export_tab:
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
- 逆变器效率: {config['system']['inv_efficiency']:.1%}
- 温度模型: {config['system']['temp_model']}
- 逆变器功率: {config['system']['inverter_power']} kW

组件技术参数
------------
- 组件型号: {config['system']['module_name']}
- 组件额定功率: {config['system'].get('module_power', 'N/A')} W
- 开路电压 (Voc): {config['system'].get('Voc', 'N/A')} V
- 短路电流 (Isc): {config['system'].get('Isc', 'N/A')} A
- 最大功率点电压 (Vmp): {config['system'].get('Vmp', 'N/A')} V
- 最大功率点电流 (Imp): {config['system'].get('Imp', 'N/A')} A
- 温度系数: {config['system'].get('gamma_pdc', 'N/A') * 100 if isinstance(config['system'].get('gamma_pdc'), (int, float)) else 'N/A'} %/°C

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
        # st.subheader("⚙️ 系统配置信息")
        # st.json(st.session_state['config'])

# 页脚
st.divider()
st.caption("© 2026 光伏发电量分析工具 | 基于Streamlit和PVlib ModelChain构建")

# 使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    ### 使用步骤：
    1. **数据上传**: 上传NASA POWER格式的CSV文件
    2. **参数配置**: 在侧边栏设置系统参数
    3. **模型计算**: 点击"开始ModelChain计算"按钮
    4. **结果分析**: 查看各种分析图表和指标
    5. **数据导出**: 下载计算结果和分析报告


    ### 温度模型说明：
    PVlib的温度模型参数使用特定的安装类型，而不是组件类型：
    - **open_rack_glass_polymer**: 开架式安装，玻璃聚合物背板
    - **close_mount_glass_polymer**: 密闭安装，玻璃聚合物背板
    - **open_rack_glass_glass**: 开架式安装，双玻组件

    ### 数据要求：
    - CSV文件必须包含: YEAR, MO, DY, HR, ALLSKY_SFC_SW_DNI, ALLSKY_SFC_SW_DWN等列
    - 时间格式: 年月日时
    - 缺失值标记: -999.0

    """)
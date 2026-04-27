import pulp
import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import math

# 创建Tkinter根窗口但不显示
root = tk.Tk()
root.withdraw()

# 参数设置
num = 96
dt = 0.25  # 小时
P = 250000  # 储能逆变器功率，单位 kW
battery_capacity = 500000  # 电池容量上限，单位 kWh
initial_soc = 0  # 初始电量，假设为0 kWh
efficiency = 0.85  # 放电效率
print(f"电池容量: {battery_capacity:.2f} kWh")
print(f"初始电量: {initial_soc:.2f} kWh")
print(f"放电效率: {efficiency * 100:.0f}%")

print("=" * 60)
print("储能优化调度模型（多天优化）")
print("=" * 60)

# 选择输入文件
print("\n请选择输入Excel文件（包含多天电价数据）...")
input_file = filedialog.askopenfilename(
    title="选择输入Excel文件",
    filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
)

if not input_file:
    print("未选择输入文件，程序退出。")
    exit(0)

print(f"已选择输入文件: {input_file}")

# 选择输出文件
print("\n请选择输出Excel文件的保存位置和名称...")
output_file = filedialog.asksaveasfilename(
    title="保存优化结果",
    defaultextension=".xlsx",
    filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
    initialfile="优化结果.xlsx"
)

if not output_file:
    print("未选择输出文件，将使用默认名称 '优化结果.xlsx'")
    output_file = "优化结果.xlsx"

print(f"结果将保存到: {output_file}")

# 提取文件名
input_filename = os.path.basename(input_file)
output_filename = os.path.basename(output_file)

# 尝试从Excel文件读取数据
try:
    # 检查Excel文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: Excel文件 '{input_file}' 不存在。")
        exit(1)

    # 从Excel读取数据
    print(f"\n正在读取Excel文件: {input_filename}")

    # 尝试不同的工作表名称
    sheet_names = ["Sheet1", "电价数据", "数据", "电价"]
    df = None

    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            print(f"成功读取工作表: {sheet_name}")
            break
        except:
            continue

    # 如果所有尝试的工作表名称都失败，尝试第一个工作表
    if df is None:
        try:
            xls = pd.ExcelFile(input_file)
            sheet_name = xls.sheet_names[0]
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            print(f"成功读取工作表: {sheet_name}")
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            exit(1)

    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    # 检查数据列数（每行一天，96列）
    if len(df.columns) < num:
        print(f"错误: Excel文件只有{len(df.columns)}列数据，需要{num}列（96个时段）。")
        exit(1)
    
    # 检查数据行数（天数）
    num_days = len(df)
    if num_days == 0:
        print("错误: Excel文件没有数据行。")
        exit(1)
    
    print(f"\n检测到 {num_days} 天的电价数据")

    # 提取所有天的电价数据
    # 检查第一列是否为非数字列（如"日期"列）
    start_col = 0
    date_list = None
    
    if len(df.columns) > num and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        # 第一列不是数字，保存日期信息
        raw_dates = df.iloc[:, 0].values
        # 将日期格式化为只包含年月日的字符串
        date_list = []
        for d in raw_dates:
            if pd.isna(d):
                date_list.append('')
            elif isinstance(d, (pd.Timestamp, pd.DatetimeIndex)):
                date_list.append(d.strftime('%Y-%m-%d'))
            elif isinstance(d, str):
                # 如果是字符串，尝试提取年月日部分
                if 'T' in d:
                    date_list.append(d.split('T')[0])
                else:
                    date_list.append(str(d)[:10])
            else:
                date_list.append(str(d))
        start_col = 1
        print(f"检测到日期列 '{df.columns[0]}'，将使用该列作为日期标识")
        print(f"日期范围: {date_list[0]} 至 {date_list[-1]}")
    else:
        # 没有日期列，使用默认的第X天格式
        date_list = [f'第{i+1}天' for i in range(num_days)]
        print("未检测到日期列，将使用默认格式（第1天、第2天...）")
    
    # 提取电价数据（跳过非数字列）
    all_prices = df.iloc[:, start_col:start_col+num].values
    
    # 转换为数值类型
    all_prices = all_prices.astype(float)
    
    # 验证数据
    if np.any(all_prices < 0):
        print("警告: 电价数据包含负值。")

    # 显示数据统计信息
    print("\n" + "=" * 60)
    print("数据统计信息")
    print("=" * 60)
    print(f"电价数据统计（所有天）:")
    print(f"  最小值: {np.min(all_prices):.3f} 元/kWh")
    print(f"  最大值: {np.max(all_prices):.3f} 元/kWh")
    print(f"  平均值: {np.mean(all_prices):.3f} 元/kWh")

    # 显示第一天前5个时段
    print(f"\n第1天前5个时段的电价:")
    for i in range(min(5, num)):
        hour = i * dt
        time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"
        print(f"  {time_str}: 电价={all_prices[0, i]:.3f} 元/kWh")

except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    exit(1)

# 定义优化函数
def optimize_single_day(price, day_index, start_soc):
    """优化单天的储能调度"""
    print(f"\n{'=' * 60}")
    print(f"正在优化第 {day_index + 1} 天 (初始电量: {start_soc:.2f} kWh)")
    print("=" * 60)
    
    # 创建问题
    prob = pulp.LpProblem(f"BESS_Optimization_Day{day_index + 1}", pulp.LpMaximize)
    
    # 定义变量
    charge_power = pulp.LpVariable.dicts("charge", range(num), lowBound=0)
    discharge_power = pulp.LpVariable.dicts("discharge", range(num), lowBound=0)
    soc = pulp.LpVariable.dicts("soc", range(num + 1), lowBound=0, upBound=battery_capacity)
    
    # 二进制变量
    z_ch = pulp.LpVariable.dicts("is_charging", range(num), cat="Binary")
    z_dis = pulp.LpVariable.dicts("is_discharging", range(num), cat="Binary")
    y_ch = pulp.LpVariable.dicts("charge_start", range(num), cat="Binary")
    y_dis = pulp.LpVariable.dicts("discharge_start", range(num), cat="Binary")
    
    epsilon = 1e-3
    min_duration = 4
    min_on_power = 0.05 * P
    
    # 目标函数
    prob += pulp.lpSum([(discharge_power[i] - charge_power[i]) * price[i] * dt for i in range(num)])
    
    # 初始电量约束
    prob += soc[0] == start_soc
    
    # 电池状态转移方程
    for i in range(num):
        prob += soc[i + 1] == soc[i] + charge_power[i] * dt - discharge_power[i] * dt / efficiency
    
    # 功率-状态关联约束
    for i in range(num):
        prob += charge_power[i] <= P * z_ch[i]
        prob += discharge_power[i] <= P * z_dis[i]
        prob += charge_power[i] >= min_on_power * z_ch[i]
        prob += discharge_power[i] >= min_on_power * z_dis[i]
        prob += z_ch[i] + z_dis[i] <= 1
        
        if i == 0:
            prob += y_ch[i] == z_ch[i]
            prob += y_dis[i] == z_dis[i]
        else:
            prob += y_ch[i] >= z_ch[i] - z_ch[i-1]
            prob += y_ch[i] <= z_ch[i]
            prob += y_ch[i] <= 1 - z_ch[i-1]
            prob += y_dis[i] >= z_dis[i] - z_dis[i-1]
            prob += y_dis[i] <= z_dis[i]
            prob += y_dis[i] <= 1 - z_dis[i-1]
    
    # 最小连续时长约束
    for i in range(num):
        if i <= num - min_duration:
            prob += pulp.lpSum(z_ch[j] for j in range(i, i + min_duration)) >= min_duration * y_ch[i]
            prob += pulp.lpSum(z_dis[j] for j in range(i, i + min_duration)) >= min_duration * y_dis[i]
        else:
            prob += y_ch[i] == 0
            prob += y_dis[i] == 0
    
    # 最终电量约束
    # prob += soc[num] == initial_soc
    
    # 累计能量平衡约束（允许电量留存）
    for k in range(1, num+1):
        prob += efficiency * pulp.lpSum([charge_power[i] * dt for i in range(k)]) + start_soc >= pulp.lpSum(
            [discharge_power[i] * dt for i in range(k)])
    
    # 连续充电能量约束
    M = int(math.ceil(battery_capacity / (P * dt))) + 1
    for s in range(0, 97 - M):
        end_idx = min(s + M - 1, 95)
        prob += pulp.lpSum([charge_power[j] * dt for j in range(s, end_idx + 1)]) <= battery_capacity
    
    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)
    result = prob.solve(solver)
    
    return prob, result

# 开始优化
print("\n" + "=" * 60)
print("开始优化多天储能调度")
print("=" * 60)

# 存储所有天的结果
all_results = []
all_summaries = []
all_price_analysis = []  # 存储每天的电价分析数据

# 电价区间定义
price_bins = [0, 0.3, 0.7, 1.0, 1.5, float('inf')]
price_labels = ['<0.3', '0.3-0.7', '0.7-1.0', '1.0-1.5', '>1.5']

current_soc = initial_soc

for day_idx in range(num_days):
    price = all_prices[day_idx, :]
    
    # 保存当天的初始电量（用于后续约束检查和结果记录）
    day_start_soc = current_soc
    
    # 优化当天
    prob, result = optimize_single_day(price, day_idx, current_soc)
    
    if pulp.LpStatus[prob.status] == "Optimal":
        print(f"第 {day_idx + 1} 天求解完成!")
        
        # 将变量列表转换为字典以便按名称访问
        var_dict = {v.name: v for v in prob.variables()}
        
        # 获取充放电功率
        charge_power_values = [pulp.value(var_dict[f"charge_{i}"]) for i in range(num)]
        discharge_power_values = [pulp.value(var_dict[f"discharge_{i}"]) for i in range(num)]
        soc_values = [pulp.value(var_dict[f"soc_{i}"]) for i in range(num + 1)]
        
        # 更新下一天的初始电量
        current_soc = pulp.value(var_dict[f"soc_{num}"])
        
        # 计算能量和费用
        charge_energy = [charge_power_values[i] * dt for i in range(num)]
        discharge_energy = [discharge_power_values[i] * dt for i in range(num)]
        charge_cost = [charge_energy[i] * price[i] for i in range(num)]
        discharge_revenue = [discharge_energy[i] * price[i] for i in range(num)]
        
        # 创建详细结果
        for i in range(num):
            hour = i * dt
            time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"
            
            if charge_power_values[i] > 1e-4 and discharge_power_values[i] > 1e-4:
                period_type = "同时充放电"
            elif charge_power_values[i] > 1e-4:
                period_type = "充电"
            elif discharge_power_values[i] > 1e-4:
                period_type = "放电"
            else:
                period_type = "空闲"
            
            all_results.append({
                '日期': date_list[day_idx],
                '时段': i + 1,
                '时间': time_str,
                '电价_元/kWh': price[i],
                '充电功率_kW': charge_power_values[i],
                '放电功率_kW': discharge_power_values[i],
                '净功率_kW': discharge_power_values[i] - charge_power_values[i],
                '电池电量_kWh': soc_values[i + 1],
                '充电量_kWh': charge_energy[i],
                '放电量_kWh': discharge_energy[i],
                '充电电费_元': -charge_cost[i],
                '放电电费_元': discharge_revenue[i],
                '时段类型': period_type
            })
        
        # 每天结束后添加空行（最后一天除外）
        if day_idx < num_days - 1:
            all_results.append({
                '日期': '',
                '时段': '',
                '时间': '',
                '电价_元/kWh': '',
                '充电功率_kW': '',
                '放电功率_kW': '',
                '净功率_kW': '',
                '电池电量_kWh': '',
                '充电量_kWh': '',
                '放电量_kWh': '',
                '充电电费_元': '',
                '放电电费_元': '',
                '时段类型': ''
            })
        
        # 计算汇总统计
        total_revenue = pulp.value(prob.objective)
        total_charge = sum(charge_energy)
        total_discharge = sum(discharge_energy)
        charge_hours = sum(1 for i in range(num) if charge_power_values[i] > 1e-4)
        discharge_hours = sum(1 for i in range(num) if discharge_power_values[i] > 1e-4)
        
        # 电池实际存入和放出
        total_battery_charge = sum(charge_energy)
        total_battery_discharge = sum([discharge_energy[i] / efficiency for i in range(num)])
        
        # 系统效率
        system_efficiency = (total_discharge / total_charge * 100) if total_charge > 0 else 0
        
        # 同时充放电时段数
        simultaneous_charge_discharge = sum(1 for i in range(num) 
                                           if charge_power_values[i] > 1e-4 and discharge_power_values[i] > 1e-4)
        
        # 计算最大电池电量
        max_soc = max(soc_values)
        
        # 计算连续充放电段
        continuous_charge_segments = []
        current_segment = []
        current_segment_start = None
        
        for i in range(num):
            if charge_power_values[i] > 1e-4:
                if current_segment_start is None:
                    current_segment_start = i
                current_segment.append(i)
            else:
                if current_segment:
                    segment_energy = sum([charge_energy[j] for j in current_segment])
                    continuous_charge_segments.append({
                        'start': current_segment_start,
                        'end': i - 1,
                        'energy': segment_energy
                    })
                    current_segment = []
                    current_segment_start = None
        
        if current_segment:
            segment_energy = sum([charge_energy[j] for j in current_segment])
            continuous_charge_segments.append({
                'start': current_segment_start,
                'end': num - 1,
                'energy': segment_energy
            })
        
        # 连续放电段
        continuous_discharge_segments = []
        current_segment = []
        current_segment_start = None
        
        for i in range(num):
            if discharge_power_values[i] > 1e-4:
                if current_segment_start is None:
                    current_segment_start = i
                current_segment.append(i)
            else:
                if current_segment:
                    segment_energy = sum([discharge_energy[j] for j in current_segment])
                    continuous_discharge_segments.append({
                        'start': current_segment_start,
                        'end': i - 1,
                        'energy': segment_energy
                    })
                    current_segment = []
                    current_segment_start = None
        
        if current_segment:
            segment_energy = sum([discharge_energy[j] for j in current_segment])
            continuous_discharge_segments.append({
                'start': current_segment_start,
                'end': num - 1,
                'energy': segment_energy
            })
        
        # 最大连续充放电能量
        max_continuous_charge = max([seg['energy'] for seg in continuous_charge_segments]) if continuous_charge_segments else 0
        max_continuous_discharge = max([seg['energy'] for seg in continuous_discharge_segments]) if continuous_discharge_segments else 0
        
        # 最大可能连续充电时段数M
        M = int(math.ceil(battery_capacity / (P * dt))) + 1
        
        # 约束检查
        power_constraint_violations = sum(1 for i in range(num) 
                                         if charge_power_values[i] > P + 1e-6 or discharge_power_values[i] > P + 1e-6)
        
        cumulative_charge = 0
        cumulative_discharge = 0
        balance_constraint_violations = 0
        
        for k in range(1, num + 1):
            cumulative_charge = sum(charge_energy[:k])
            cumulative_discharge = sum(discharge_energy[:k])
            
            # 使用相对容差，避免大数值下的浮点误差误报
            tolerance = max(1e-6, 1e-5 * max(cumulative_charge, cumulative_discharge, 1.0))
            
            # 校验逻辑：累计充电 + 初始电量 >= 累计放电
            available_energy = efficiency * cumulative_charge + day_start_soc
            if available_energy < cumulative_discharge - tolerance:
                balance_constraint_violations += 1
        
        continuous_charge_violations = 0
        for s in range(0, 97 - M):
            end_idx = min(s + M - 1, 95)
            window_charge = sum(charge_energy[s:end_idx + 1])
            if window_charge > battery_capacity + 1e-6:
                continuous_charge_violations += 1
        
        soc_constraint_violations = sum(1 for i in range(num + 1) 
                                       if soc_values[i] > battery_capacity + 1e-6)
        
        all_summaries.append({
            '日期': date_list[day_idx],
            '求解状态': pulp.LpStatus[prob.status],
            '日收益(元)': f"{total_revenue:.2f}",
            '初始电量(kWh)': f"{day_start_soc:.2f}",
            '最终电量(kWh)': f"{current_soc:.2f}",
            '最大电池电量(kWh)': f"{max_soc:.2f}",
            '从电网总充电量(kWh)': f"{total_charge:.2f}",
            '向电网总放电量(kWh)': f"{total_discharge:.2f}",
            '电池实际存入(kWh)': f"{total_battery_charge:.2f}",
            '电池实际放出(kWh)': f"{total_battery_discharge:.2f}",
            '系统整体效率(%)': f"{system_efficiency:.1f}",
            '充电时段数': charge_hours,
            '放电时段数': discharge_hours,
            '同时充放电时段数': simultaneous_charge_discharge,
            '储能逆变器功率(kW)': P,
            '时间间隔(小时)': dt,
            '电池容量(kWh)': f"{battery_capacity:.2f}",
            '放电效率(%)': f"{efficiency * 100:.0f}",
            '最大可能连续充电时段数M': M,
            '最大连续充电能量(kWh)': f"{max_continuous_charge:.2f}",
            '最大连续放电能量(kWh)': f"{max_continuous_discharge:.2f}",
            '连续充电段数': len(continuous_charge_segments),
            '连续放电段数': len(continuous_discharge_segments),
            '功率约束检查': '通过' if power_constraint_violations == 0 else f'不通过({power_constraint_violations}个违规)',
            '累计能量平衡约束检查': '通过' if balance_constraint_violations == 0 else f'不通过({balance_constraint_violations}个违规)',
            '连续充电能量约束检查': '通过' if continuous_charge_violations == 0 else f'不通过({continuous_charge_violations}个违规)',
            '电池容量约束检查': '通过' if soc_constraint_violations == 0 else f'不通过({soc_constraint_violations}个违规)'
        })
        
        # 电价分析
        day_price_analysis = {}
        for i in range(len(price_bins) - 1):
            lower = price_bins[i]
            upper = price_bins[i + 1]
            label = price_labels[i]
            
            if upper == float('inf'):
                indices = np.where(price >= lower)[0]
            else:
                indices = np.where((price >= lower) & (price < upper))[0]
            
            if len(indices) > 0:
                charge_in_range = sum([charge_energy[idx] for idx in indices])
                discharge_in_range = sum([discharge_energy[idx] for idx in indices])
                net_revenue = sum([(discharge_energy[idx] - charge_energy[idx]) * price[idx] for idx in indices])
                
                day_price_analysis[label] = {
                    '时段数': len(indices),
                    '占比(%)': f"{(len(indices) / num * 100):.1f}",
                    '平均电价(元/kWh)': f"{np.mean(price[indices]):.3f}",
                    '充电量(kWh)': f"{charge_in_range:.2f}",
                    '放电量(kWh)': f"{discharge_in_range:.2f}",
                    '净收益(元)': f"{net_revenue:.2f}"
                }
            else:
                day_price_analysis[label] = {
                    '时段数': 0,
                    '占比(%)': '0.0',
                    '平均电价(元/kWh)': '0.000',
                    '充电量(kWh)': '0.00',
                    '放电量(kWh)': '0.00',
                    '净收益(元)': '0.00'
                }
        
        all_price_analysis.append(day_price_analysis)
    else:
        print(f"第 {day_idx + 1} 天优化失败! 状态: {pulp.LpStatus[prob.status]}")

# 导出结果到Excel
print("\n正在导出详细结果到Excel文件...")
results_df = pd.DataFrame(all_results)

# 将结果摘要表转置：每列为一天的数据，每行为指标
summary_df = pd.DataFrame(all_summaries)
# 提取日期列作为新列名，并格式化为只包含年月日
date_columns = []
for d in summary_df['日期'].values:
    if pd.isna(d) or d == '':
        date_columns.append('')
    elif isinstance(d, (pd.Timestamp, pd.DatetimeIndex)):
        date_columns.append(d.strftime('%Y-%m-%d'))
    elif isinstance(d, str):
        # 如果是字符串，提取年月日部分
        if 'T' in d:
            date_columns.append(d.split('T')[0])
        elif ' ' in d:
            date_columns.append(d.split(' ')[0])
        else:
            date_columns.append(d[:10])
    else:
        date_columns.append(str(d))

# 删除日期列
summary_df = summary_df.drop('日期', axis=1)
# 转置DataFrame
summary_transposed = summary_df.T
# 设置列名为格式化后的日期
summary_transposed.columns = date_columns
# 设置索引名称为"项目"
summary_transposed.index.name = '项目'
# 重置索引使"项目"成为第一列
summary_transposed = summary_transposed.reset_index()

# 将结果保存到Excel文件
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='详细结果', index=False)
    summary_transposed.to_excel(writer, sheet_name='结果摘要', index=False)
    
    # 创建电价分析表（转置格式）
    if all_price_analysis:
        # 构建电价分析DataFrame
        price_analysis_rows = []
        for label in price_labels:
            row = {'电价区间(元/kWh)': label}
            for day_idx in range(num_days):
                if day_idx < len(all_price_analysis):
                    day_data = all_price_analysis[day_idx].get(label, {})
                    current_date = date_list[day_idx]
                    row[f'{current_date}-时段数'] = day_data.get('时段数', 0)
                    row[f'{current_date}-占比(%)'] = day_data.get('占比(%)', '0.0')
                    row[f'{current_date}-平均电价(元/kWh)'] = day_data.get('平均电价(元/kWh)', '0.000')
                    row[f'{current_date}-充电量(kWh)'] = day_data.get('充电量(kWh)', '0.00')
                    row[f'{current_date}-放电量(kWh)'] = day_data.get('放电量(kWh)', '0.00')
                    row[f'{current_date}-净收益(元)'] = day_data.get('净收益(元)', '0.00')
            price_analysis_rows.append(row)
        
        price_analysis_df = pd.DataFrame(price_analysis_rows)
        price_analysis_df.to_excel(writer, sheet_name='电价分析', index=False)

print(f"详细结果已保存到 '{output_file}'")

# 显示文件大小
try:
    file_size = os.path.getsize(output_file) / 1024  # KB
    print(f"\n输出文件大小: {file_size:.1f} KB")
except:
    pass

print("\n" + "=" * 60)
print("优化完成!")
print(f"优化了 {num_days} 天的数据")
print(f"结果已保存到: {output_file}")
print("=" * 60)

# 显示完成消息
messagebox.showinfo("优化完成", f"储能优化调度已完成！\n\n优化了 {num_days} 天的数据\n\n结果已保存到:\n{output_file}\n\n点击确定关闭程序。")

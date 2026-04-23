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
P = 14900  # 储能逆变器功率，单位 kW
battery_capacity = 28702.4  # 电池容量上限，单位 kWh
initial_soc = 0  # 初始电量，假设为0 kWh
efficiency = 0.85  # 放电效率
print(f"电池容量: {battery_capacity:.2f} kWh")
print(f"初始电量: {initial_soc:.2f} kWh")
print(f"放电效率: {efficiency * 100:.0f}%")

print("=" * 60)
print("储能优化调度模型（仅考虑电价）")
print("=" * 60)

# 选择输入文件
print("\n请选择输入Excel文件（包含电价数据）...")
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

    # 检查数据行数
    if len(df) < num:
        print(f"错误: Excel文件只有{len(df)}行数据，需要{num}行。")
        exit(1)
    elif len(df) > num:
        print(f"警告: Excel文件有{len(df)}行数据，将只使用前{num}行。")

    # 检查列名
    if len(df.columns) < 1:
        print("错误: Excel文件至少需要1列电价数据。")
        exit(1)

    # 获取列名
    price_col = df.columns[0]  # 电价列名

    print(f"识别到电价列: '{price_col}'")

    # 读取电价（第一列）
    price = df.iloc[:num, 0].values

    # 验证数据长度
    if len(price) != num:
        print("错误: 数据长度不匹配。")
        exit(1)

    # 验证数据范围
    if np.any(price < 0):
        print("警告: 电价数据包含负值。")

    # 显示数据统计信息
    print("\n" + "=" * 60)
    print("数据统计信息")
    print("=" * 60)
    print(f"电价数据统计:")
    print(f"  最小值: {np.min(price):.3f} 元/kWh")
    print(f"  最大值: {np.max(price):.3f} 元/kWh")
    print(f"  平均值: {np.mean(price):.3f} 元/kWh")

    # 显示前5个数据点
    print(f"\n前5个时段的电价:")
    for i in range(min(5, num)):
        hour = i * dt
        time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"
        print(f"  {time_str}: 电价={price[i]:.3f} 元/kWh")

except Exception as e:
    print(f"读取Excel文件时出错: {e}")
    exit(1)

# 建立线性规划问题
print("\n" + "=" * 60)
print("建立优化模型")
print("=" * 60)
print("正在建立优化模型...")

# 创建问题
prob = pulp.LpProblem("BESS_Optimization", pulp.LpMaximize)

# 定义变量
charge_power = pulp.LpVariable.dicts("charge", range(num), lowBound=0)  # 充电功率（从电网吸收）
discharge_power = pulp.LpVariable.dicts("discharge", range(num), lowBound=0)  # 放电功率（向电网放出）
soc = pulp.LpVariable.dicts("soc", range(num + 1), lowBound=0, upBound=battery_capacity)  # 电池电量

# 二进制变量
z_ch = pulp.LpVariable.dicts("is_charging", range(num), cat="Binary")
z_dis = pulp.LpVariable.dicts("is_discharging", range(num), cat="Binary")
y_ch = pulp.LpVariable.dicts("charge_start", range(num), cat="Binary")
y_dis = pulp.LpVariable.dicts("discharge_start", range(num), cat="Binary")

epsilon = 1e-3
min_duration = 4
min_on_power = 0.05 * P

# 目标函数：最大化收益
# 收益 = 放电收入 - 充电成本
prob += pulp.lpSum([(discharge_power[i] - charge_power[i]) * price[i] * dt for i in range(num)])

# 初始电量约束
prob += soc[0] == initial_soc

# 电池状态转移方程
# 充电无损耗：电池电量增加 = 充电功率 * dt
# 放电有损耗：电池电量减少 = 放电功率 * dt / efficiency
for i in range(num):
    prob += soc[i + 1] == soc[i] + charge_power[i] * dt - discharge_power[i] * dt / efficiency

# 功率约束（仅考虑逆变器功率限制）
#print("\n添加功率约束...")
#for i in range(num):
    # 充放电功率不超过逆变器额定功率
#    prob += charge_power[i] <= P
#    prob += discharge_power[i] <= P
for i in range(num):
    # 功率-状态关联
    prob += charge_power[i] <= P * z_ch[i]
    prob += discharge_power[i] <= P * z_dis[i]
    prob += charge_power[i] >= min_on_power * z_ch[i]
    prob += discharge_power[i] >= min_on_power * z_dis[i]

    # 互斥：同一时段不能既充又放
    prob += z_ch[i] + z_dis[i] <= 1

    # 启动定义（充电）
    if i == 0:
        prob += y_ch[i] == z_ch[i]
    else:
        prob += y_ch[i] >= z_ch[i] - z_ch[i-1]
        prob += y_ch[i] <= z_ch[i]
        prob += y_ch[i] <= 1 - z_ch[i-1]

    # 启动定义（放电）
    if i == 0:
        prob += y_dis[i] == z_dis[i]
    else:
        prob += y_dis[i] >= z_dis[i] - z_dis[i-1]
        prob += y_dis[i] <= z_dis[i]
        prob += y_dis[i] <= 1 - z_dis[i-1]

# 最小连续时长 + 尾部禁止启动（充电）
for i in range(num):
    if i <= num - min_duration:
        prob += pulp.lpSum(z_ch[j] for j in range(i, i + min_duration)) >= min_duration * y_ch[i]
    else:
        prob += y_ch[i] == 0

# 最小连续时长 + 尾部禁止启动（放电）
for i in range(num):
    if i <= num - min_duration:
        prob += pulp.lpSum(z_dis[j] for j in range(i, i + min_duration)) >= min_duration * y_dis[i]
    else:
        prob += y_dis[i] == 0

# 最终电量约束：最终电量等于初始电量
prob += soc[num] == initial_soc

# 累计能量平衡约束
print("\n添加累计能量平衡约束...")
# 对于 k = 1, 2, ..., 95: 0.85 * Σ_{i=1}^{k} v_i ≥ Σ_{i=1}^{k} u_i
# 各个时刻的累计吸收能量（折算效率后）必须大于等于累计溢出/放出的能量，且全天结束后完全平衡：总放电量 =0.85× 总充电量。
for k in range(1, num):  # k = 1, 2, ..., 95
    prob += efficiency * pulp.lpSum([charge_power[i] * dt for i in range(k)]) >= pulp.lpSum(
        [discharge_power[i] * dt for i in range(k)])
    if k % 20 == 0 or k == 1 or k == 95:
        print(f"  k={k}: 0.85 * Σ(充电量[1:{k}]) ≥ Σ(放电量[1:{k}])")

# 对于 k = 96: 0.85 * Σ_{i=1}^{96} v_i = Σ_{i=1}^{96} u_i
prob += efficiency * pulp.lpSum([charge_power[i] * dt for i in range(num)]) == pulp.lpSum(
    [discharge_power[i] * dt for i in range(num)])
print(f"  k=96: 0.85 * Σ(充电量[1:96]) = Σ(放电量[1:96])")

# 连续充电能量约束
print("\n添加连续充电能量约束...")
# 计算最大可能连续充电时段数 M
M = int(math.ceil(battery_capacity / (P * dt))) + 1
print(f"  最大可能连续充电时段数 M = {M}")
print(f"  计算: M = ceil({battery_capacity} / ({P} * {dt})) + 1")

# 对于所有可能的连续充电时段窗口
for s in range(0, 97 - M):  # s = 1, 2, ..., 97-M，转换为0索引
    end_idx = min(s + M - 1, 95)  # 当 s+M-1 > 96 时，取 s+M-1 = 96
    # 连续M个时段的充电能量和不超过电池容量
    prob += pulp.lpSum([charge_power[j] * dt for j in range(s, end_idx + 1)]) <= battery_capacity
    if s == 0 or s == 96 - M:
        print(f"  窗口{s + 1}-{end_idx + 1}: Σ(充电量[{s + 1}:{end_idx + 1}]) * 0.25 ≤ {battery_capacity}")

# 求解
print("\n正在求解优化问题...")
solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=120)  # 增加求解时间
result = prob.solve(solver)
print(f"求解完成! 状态: {pulp.LpStatus[prob.status]}")

# 提取结果
if pulp.LpStatus[prob.status] == "Optimal":
    # 获取充放电功率
    charge_power_values = [pulp.value(charge_power[i]) for i in range(num)]
    discharge_power_values = [pulp.value(discharge_power[i]) for i in range(num)]
    soc_values = [pulp.value(soc[i]) for i in range(num + 1)]

    # 计算能量和费用
    charge_energy = [charge_power_values[i] * dt for i in range(num)]  # 从电网吸收的充电量
    discharge_energy = [discharge_power_values[i] * dt for i in range(num)]  # 向电网放出的放电量

    # 计算电池电量变化
    battery_charge_energy = charge_energy  # 充电无损耗，全部存入电池
    battery_discharge_energy = [discharge_energy[i] / efficiency for i in range(num)]  # 电池实际放出的能量

    # 计算电费
    charge_cost = [charge_energy[i] * price[i] for i in range(num)]  # 充电电费
    discharge_revenue = [discharge_energy[i] * price[i] for i in range(num)]  # 放电电费

    # 总充电量（从电网吸收）
    total_charge = sum(charge_energy)
    # 总放电量（向电网放出）
    total_discharge = sum(discharge_energy)
    # 电池实际存入的总能量
    total_battery_charge = sum(battery_charge_energy)
    # 电池实际放出的总能量
    total_battery_discharge = sum(battery_discharge_energy)

    # 收益
    total_revenue = pulp.value(prob.objective)

    # 检查所有约束
    print("\n" + "=" * 60)
    print("约束检查")
    print("=" * 60)

    # 1. 检查功率约束
    power_constraint_violations = 0
    for i in range(num):
        if charge_power_values[i] > P + 1e-6:
            power_constraint_violations += 1
        if discharge_power_values[i] > P + 1e-6:
            power_constraint_violations += 1
    print(
        f"功率约束检查: {'通过' if power_constraint_violations == 0 else f'不通过，{power_constraint_violations}个违规'}")

    # 2. 检查累计能量平衡约束
    cumulative_charge = 0
    cumulative_discharge = 0
    balance_constraint_violations = 0

    for k in range(1, num + 1):
        cumulative_charge = sum(charge_energy[:k])
        cumulative_discharge = sum(discharge_energy[:k])

        if k < num:
            # 对于 k = 1,...,95: 0.85 * Σ充电量 ≥ Σ放电量
            if efficiency * cumulative_charge < cumulative_discharge - 1e-6:
                balance_constraint_violations += 1
        else:
            # 对于 k = 96: 0.85 * Σ充电量 = Σ放电量
            if abs(efficiency * cumulative_charge - cumulative_discharge) > 1e-6:
                balance_constraint_violations += 1

    print(
        f"累计能量平衡约束检查: {'通过' if balance_constraint_violations == 0 else f'不通过，{balance_constraint_violations}个违规'}")
    if balance_constraint_violations == 0:
        print(f"  最终能量平衡: 0.85 * {total_charge:.2f} = {total_discharge:.2f}")

    # 3. 检查连续充电能量约束
    continuous_charge_violations = 0
    for s in range(0, 97 - M):
        end_idx = min(s + M - 1, 95)
        window_charge = sum(charge_energy[s:end_idx + 1])
        if window_charge > battery_capacity + 1e-6:
            continuous_charge_violations += 1

    print(
        f"连续充电能量约束检查: {'通过' if continuous_charge_violations == 0 else f'不通过，{continuous_charge_violations}个违规'}")
    print(f"  最大可能连续充电时段数 M = {M}")

    # 4. 检查电池容量约束
    soc_constraint_violations = 0
    max_soc = 0
    for i in range(num + 1):
        if soc_values[i] > battery_capacity + 1e-6:
            soc_constraint_violations += 1
        max_soc = max(max_soc, soc_values[i])

    print(
        f"电池容量约束检查: {'通过' if soc_constraint_violations == 0 else f'不通过，{soc_constraint_violations}个违规'}")
    print(f"  最大电池电量: {max_soc:.2f} kWh")

    # 输出结果摘要
    print("\n" + "=" * 60)
    print("优化结果摘要")
    print("=" * 60)
    print(f"求解状态: {pulp.LpStatus[prob.status]}")
    print(f"日收益: {total_revenue:.2f} 元")
    print(f"从电网总充电量: {total_charge:.2f} kWh")
    print(f"向电网总放电量: {total_discharge:.2f} kWh")
    print(f"电池实际存入: {total_battery_charge:.2f} kWh")
    print(f"电池实际放出: {total_battery_discharge:.2f} kWh")
    print(f"初始电量: {initial_soc:.2f} kWh")
    print(f"最终电量: {soc_values[-1]:.2f} kWh")
    print(f"最大电池电量: {max_soc:.2f} kWh")

    if total_charge > 0:
        # 计算系统整体效率：放电量/充电量
        system_efficiency = total_discharge / total_charge * 100
        print(f"系统整体效率: {system_efficiency:.1f}% (目标: {efficiency * 100:.0f}%)")

    charge_hours = sum(1 for i in range(num) if charge_power_values[i] > 1e-4)
    discharge_hours = sum(1 for i in range(num) if discharge_power_values[i] > 1e-4)
    print(f"充电时段数: {charge_hours} (共{num}个时段)")
    print(f"放电时段数: {discharge_hours} (共{num}个时段)")

    # 检查同时充放电
    simultaneous_charge_discharge = 0
    for i in range(num):
        if charge_power_values[i] > 0.001 and discharge_power_values[i] > 0.001:
            simultaneous_charge_discharge += 1
    print(f"同时充放电时段数: {simultaneous_charge_discharge}")

    # 检查连续充电情况
    print("\n正在分析连续充电情况...")
    continuous_charge_segments = []
    current_segment = []
    current_segment_start = None

    for i in range(num):
        if charge_power_values[i] > 1e-4:  # 正在充电
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

    # 处理最后一个时段
    if current_segment:
        segment_energy = sum([charge_energy[j] for j in current_segment])
        continuous_charge_segments.append({
            'start': current_segment_start,
            'end': num - 1,
            'energy': segment_energy
        })

    # 检查连续放电情况
    print("\n正在分析连续放电情况...")
    continuous_discharge_segments = []
    current_segment = []
    current_segment_start = None

    for i in range(num):
        if discharge_power_values[i] > 1e-4:  # 正在放电
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

    # 处理最后一个时段
    if current_segment:
        segment_energy = sum([discharge_energy[j] for j in current_segment])
        continuous_discharge_segments.append({
            'start': current_segment_start,
            'end': num - 1,
            'energy': segment_energy
        })

    # 找到最大连续充电能量
    max_continuous_charge = max(
        [seg['energy'] for seg in continuous_charge_segments]) if continuous_charge_segments else 0
    # 找到最大连续放电能量
    max_continuous_discharge = max(
        [seg['energy'] for seg in continuous_discharge_segments]) if continuous_discharge_segments else 0

    print(f"最大连续充电能量: {max_continuous_charge:.2f} kWh")
    print(f"最大连续放电能量: {max_continuous_discharge:.2f} kWh")

    if continuous_charge_segments:
        print(f"检测到 {len(continuous_charge_segments)} 个连续充电段:")
        for idx, seg in enumerate(continuous_charge_segments[:5]):
            start_hour = seg['start'] * dt
            end_hour = seg['end'] * dt
            start_time = f"{int(start_hour):02d}:{int((start_hour - int(start_hour)) * 60):02d}"
            end_time = f"{int(end_hour):02d}:{int((end_hour - int(end_hour)) * 60):02d}"
            print(
                f"  第{idx + 1}段: 时段{seg['start'] + 1}-{seg['end'] + 1} ({start_time}~{end_time}), 能量={seg['energy']:.2f} kWh")

    if continuous_discharge_segments:
        print(f"检测到 {len(continuous_discharge_segments)} 个连续放电段:")
        for idx, seg in enumerate(continuous_discharge_segments[:5]):
            start_hour = seg['start'] * dt
            end_hour = seg['end'] * dt
            start_time = f"{int(start_hour):02d}:{int((start_hour - int(start_hour)) * 60):02d}"
            end_time = f"{int(end_hour):02d}:{int((end_hour - int(end_hour)) * 60):02d}"
            print(
                f"  第{idx + 1}段: 时段{seg['start'] + 1}-{seg['end'] + 1} ({start_time}~{end_time}), 能量={seg['energy']:.2f} kWh")

    # 特别输出7:00-9:45时段的数据
    print("\n" + "=" * 60)
    print("7:00-9:45时段详细结果")
    print("=" * 60)
    print("时段  时间   电价    充电功率 放电功率 净功率  电池电量")
    print("-" * 60)
    for i in range(28, 40):  # 7:00-9:45对应时段29-40
        hour = i * dt
        time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"
        print(
            f"{i + 1:2d}  {time_str}  {price[i]:5.3f}  "
            f"{charge_power_values[i]:7.2f}  {discharge_power_values[i]:7.2f}  "
            f"{discharge_power_values[i] - charge_power_values[i]:7.2f}  {soc_values[i + 1]:9.2f}")

    # 特别输出10:00-11:45时段的数据
    print("\n" + "=" * 60)
    print("10:00-11:45时段详细结果")
    print("=" * 60)
    print("时段  时间   电价    充电功率 放电功率 净功率  电池电量")
    print("-" * 60)
    for i in range(40, 48):  # 10:00-11:45对应时段41-48
        hour = i * dt
        time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"
        print(
            f"{i + 1:2d}  {time_str}  {price[i]:5.3f}  "
            f"{charge_power_values[i]:7.2f}  {discharge_power_values[i]:7.2f}  "
            f"{discharge_power_values[i] - charge_power_values[i]:7.2f}  {soc_values[i + 1]:9.2f}")

    # 导出结果到Excel
    print("\n正在导出详细结果到Excel文件...")

    # 创建详细结果数据
    results = []
    for i in range(num):
        hour = i * dt
        time_str = f"{int(hour):02d}:{int((hour - int(hour)) * 60):02d}"

        # 电池实际放出的能量
        actual_battery_discharge = discharge_energy[i] / efficiency

        # 确定时段类型
        if charge_power_values[i] > 1e-4 and discharge_power_values[i] > 1e-4:
            period_type = "同时充放电"
        elif charge_power_values[i] > 1e-4:
            period_type = "充电"
        elif discharge_power_values[i] > 1e-4:
            period_type = "放电"
        else:
            period_type = "空闲"

        results.append({
            '时段': i + 1,
            '时间': time_str,
            '电价_元/kWh': price[i],
            '充电功率_kW': charge_power_values[i],
            '放电功率_kW': discharge_power_values[i],
            '净功率_kW': discharge_power_values[i] - charge_power_values[i],
            '电池电量_kWh': soc_values[i + 1],
            '充电量_kWh': charge_energy[i],  # 从电网吸收
            '放电量_kWh': discharge_energy[i],  # 向电网放出
            '充电电费_元': -charge_cost[i],  # 充电成本用负数表示
            '放电电费_元': discharge_revenue[i],
            '时段类型': period_type
        })

    results_df = pd.DataFrame(results)

    # 将结果保存到Excel文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='详细结果', index=False)

        # 创建结果摘要
        summary_data = {
            '项目': [
                '求解状态',
                '日收益(元)',
                '初始电量(kWh)',
                '最终电量(kWh)',
                '最大电池电量(kWh)',
                '从电网总充电量(kWh)',
                '向电网总放电量(kWh)',
                '电池实际存入(kWh)',
                '电池实际放出(kWh)',
                '系统整体效率(%)',
                '充电时段数',
                '放电时段数',
                '同时充放电时段数',
                '储能逆变器功率(kW)',
                '时间间隔(小时)',
                '电池容量(kWh)',
                '放电效率(%)',
                '最大可能连续充电时段数M',
                '最大连续充电能量(kWh)',
                '最大连续放电能量(kWh)',
                '连续充电段数',
                '连续放电段数',
                '功率约束检查',
                '累计能量平衡约束检查',
                '连续充电能量约束检查',
                '电池容量约束检查',
                '输入文件',
                '输出文件',
                '求解时间'
            ],
            '数值': [
                pulp.LpStatus[prob.status],
                f"{total_revenue:.2f}",
                f"{initial_soc:.2f}",
                f"{soc_values[-1]:.2f}",
                f"{max_soc:.2f}",
                f"{total_charge:.2f}",
                f"{total_discharge:.2f}",
                f"{total_battery_charge:.2f}",
                f"{total_battery_discharge:.2f}",
                f"{total_discharge / total_charge * 100:.1f}" if total_charge > 0 else "N/A",
                charge_hours,
                discharge_hours,
                simultaneous_charge_discharge,
                P,
                dt,
                f"{battery_capacity:.2f}",
                f"{efficiency * 100:.0f}",
                M,
                f"{max_continuous_charge:.2f}",
                f"{max_continuous_discharge:.2f}",
                len(continuous_charge_segments),
                len(continuous_discharge_segments),
                '通过' if power_constraint_violations == 0 else f'不通过({power_constraint_violations}个违规)',
                '通过' if balance_constraint_violations == 0 else f'不通过({balance_constraint_violations}个违规)',
                '通过' if continuous_charge_violations == 0 else f'不通过({continuous_charge_violations}个违规)',
                '通过' if soc_constraint_violations == 0 else f'不通过({soc_constraint_violations}个违规)',
                input_filename,
                output_filename,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='结果摘要', index=False)

        # 创建电价分析
        price_bins = [0, 0.3, 0.7, 1.0, 1.5, float('inf')]
        price_labels = ['<0.3', '0.3-0.7', '0.7-1.0', '1.0-1.5', '>1.5']

        price_analysis_data = []
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

                price_analysis_data.append({
                    '电价区间(元/kWh)': label,
                    '时段数': len(indices),
                    '占比(%)': f"{(len(indices) / num * 100):.1f}",
                    '平均电价(元/kWh)': f"{np.mean(price[indices]):.3f}",
                    '充电量(kWh)': f"{charge_in_range:.2f}",
                    '放电量(kWh)': f"{discharge_in_range:.2f}",
                    '净收益(元)': f"{net_revenue:.2f}"
                })

        price_analysis_df = pd.DataFrame(price_analysis_data)
        price_analysis_df.to_excel(writer, sheet_name='电价分析', index=False)

        # 创建连续充电段分析
        if continuous_charge_segments:
            continuous_charge_data = []
            for idx, seg in enumerate(continuous_charge_segments):
                start_hour = seg['start'] * dt
                end_hour = seg['end'] * dt
                start_time = f"{int(start_hour):02d}:{int((start_hour - int(start_hour)) * 60):02d}"
                end_time = f"{int(end_hour):02d}:{int((end_hour - int(end_hour)) * 60):02d}"

                continuous_charge_data.append({
                    '序号': idx + 1,
                    '起始时段': seg['start'] + 1,
                    '结束时段': seg['end'] + 1,
                    '起始时间': start_time,
                    '结束时间': end_time,
                    '持续时段数': seg['end'] - seg['start'] + 1,
                    '充电能量(kWh)': f"{seg['energy']:.2f}",
                    '起始电量(kWh)': f"{soc_values[seg['start']]:.2f}",
                    '结束电量(kWh)': f"{soc_values[seg['end'] + 1]:.2f}",
                    '是否超限': '是' if seg['energy'] > battery_capacity + 1e-6 else '否'
                })

            continuous_charge_df = pd.DataFrame(continuous_charge_data)
            continuous_charge_df.to_excel(writer, sheet_name='连续充电段', index=False)

        # 创建连续放电段分析
        if continuous_discharge_segments:
            continuous_discharge_data = []
            for idx, seg in enumerate(continuous_discharge_segments):
                start_hour = seg['start'] * dt
                end_hour = seg['end'] * dt
                start_time = f"{int(start_hour):02d}:{int((start_hour - int(start_hour)) * 60):02d}"
                end_time = f"{int(end_hour):02d}:{int((end_hour - int(end_hour)) * 60):02d}"

                continuous_discharge_data.append({
                    '序号': idx + 1,
                    '起始时段': seg['start'] + 1,
                    '结束时段': seg['end'] + 1,
                    '起始时间': start_time,
                    '结束时间': end_time,
                    '持续时段数': seg['end'] - seg['start'] + 1,
                    '放电能量(kWh)': f"{sum([discharge_energy[j] for j in range(seg['start'], seg['end'] + 1)]):.2f}",
                    '起始电量(kWh)': f"{soc_values[seg['start']]:.2f}",
                    '结束电量(kWh)': f"{soc_values[seg['end'] + 1]:.2f}"
                })

            continuous_discharge_df = pd.DataFrame(continuous_discharge_data)
            continuous_discharge_df.to_excel(writer, sheet_name='连续放电段', index=False)

        # 创建约束检查表
        constraints_data = {
            '约束类型': [
                '功率约束',
                '累计能量平衡约束(k=1,...,95)',
                '累计能量平衡约束(k=96)',
                '连续充电能量约束',
                '电池容量约束'
            ],
            '数学表达式': [
                '0 ≤ v_i ≤ P, 0 ≤ u_i ≤ P',
                '0.85 * Σ_{i=1}^{k} v_i ≥ Σ_{i=1}^{k} u_i, k=1,...,95',
                '0.85 * Σ_{i=1}^{96} v_i = Σ_{i=1}^{96} u_i',
                'Σ_{j=s}^{s+M-1} v_j * 0.25 ≤ E_max, ∀s=1,...,97-M',
                '0 ≤ soc_i ≤ E_max'
            ],
            '检查结果': [
                '通过' if power_constraint_violations == 0 else f'不通过({power_constraint_violations}个违规)',
                '通过' if balance_constraint_violations == 0 else f'不通过({balance_constraint_violations}个违规)',
                '通过' if abs(efficiency * total_charge - total_discharge) <= 1e-6 else '不通过',
                '通过' if continuous_charge_violations == 0 else f'不通过({continuous_charge_violations}个违规)',
                '通过' if soc_constraint_violations == 0 else f'不通过({soc_constraint_violations}个违规)'
            ]
        }

        constraints_df = pd.DataFrame(constraints_data)
        constraints_df.to_excel(writer, sheet_name='约束检查', index=False)

    print(f"详细结果已保存到 '{output_file}'")

else:
    print(f"优化失败! 状态: {pulp.LpStatus[prob.status]}")
    # 创建空结果
    results_df = pd.DataFrame(columns=['时段', '时间', '电价_元/kWh',
                                       '充电功率_kW', '放电功率_kW', '净功率_kW', '电池电量_kWh',
                                       '充电量_kWh', '放电量_kWh', '充电电费_元', '放电电费_元',
                                       '时段类型'])

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='详细结果', index=False)

        summary_data = {
            '项目': ['求解状态'],
            '数值': [pulp.LpStatus[prob.status]]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='结果摘要', index=False)

# 显示文件大小
try:
    file_size = os.path.getsize(output_file) / 1024  # KB
    print(f"\n输出文件大小: {file_size:.1f} KB")
except:
    pass

print("\n" + "=" * 60)
print("优化完成!")
print(f"结果已保存到: {output_file}")
print("=" * 60)

# 显示完成消息
messagebox.showinfo("优化完成", f"储能优化调度已完成！\n\n结果已保存到:\n{output_file}\n\n点击确定关闭程序。")
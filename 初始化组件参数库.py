"""
初始化脚本：生成组件参数库 Excel 文件
运行一次即可，生成后可以在 Excel 中编辑添加更多组件型号
"""
import pandas as pd
import os

# 创建组件参数数据
modules_data = {
    '组件型号': [
        'LR5-72HPH-545M (545W)',
        'LR5-72HPH-550M (550W)', 
        'LR5-72HPH-555M (555W)',
        'LR5-72HPH-560M (560W)',
        'LR5-72HPH-565M (565W)'
    ],
    'pdc0': [545.0, 550.0, 555.0, 560.0, 565.0],  # 额定功率 (W)
    'Voc': [49.65, 49.80, 49.95, 50.10, 50.30],   # 开路电压 (V)
    'Isc': [13.92, 13.98, 14.04, 14.10, 14.16],   # 短路电流 (A)
    'Vmp': [41.80, 41.95, 42.10, 42.25, 42.42],   # 最大功率点电压 (V)
    'Imp': [13.04, 13.12, 13.19, 13.26, 13.32],   # 最大功率点电流 (A)
    'efficiency': [21.1, 21.3, 21.5, 21.7, 21.9]   # 组件效率 (%)
}

# 创建 DataFrame
df = pd.DataFrame(modules_data)

# 保存为 Excel 文件
output_path = os.path.join(os.path.dirname(__file__), '组件参数库.xlsx')
df.to_excel(output_path, index=False, engine='openpyxl')

print(f"✅ 组件参数库已生成: {output_path}")
print(f"📊 包含 {len(df)} 个组件型号")
print("\n可以在 Excel 中打开此文件添加更多组件型号，格式如下：")
print("  - 组件型号: 字符串，唯一标识")
print("  - pdc0: 额定功率 (W)")
print("  - Voc: 开路电压 (V)")
print("  - Isc: 短路电流 (A)")
print("  - Vmp: 最大功率点电压 (V)")
print("  - Imp: 最大功率点电流 (A)")
print("  - efficiency: 组件效率 (%)")

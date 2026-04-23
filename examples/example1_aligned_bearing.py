"""
示例1：对齐轴承基本分析（偏心模式）
=====================================
复现论文 Jakeman (1984) Table 1 Case 1 的曲轴轴承工况。
轴承参数：D=63.5mm, L=23.68mm, Cd=0.0635mm, N=2000rpm, η=0.014 Pa·s, ε=0.6
"""

from jakeman_bearing import analyze_bearing
from jakeman_bearing.visualization import (
    plot_pressure_3d,
    plot_pressure_contour,
    plot_cavitation_map,
    plot_film_thickness,
)

# 零参数调用即复现论文 Table 1 Case 1（所有参数均为默认值）
result = analyze_bearing()

# 打印结果摘要
print(result.summary())

# 生成可视化图表
plot_pressure_3d(result, save_path="example1_pressure_3d.png")
plot_pressure_contour(result, save_path="example1_pressure_contour.png")
plot_cavitation_map(result, save_path="example1_cavitation.png")
plot_film_thickness(result, save_path="example1_film_thickness.png")

# 保存结果到 CSV
result.to_csv("example1_results.csv")

print(f"\n承载力: {result.load_capacity_n:.1f} N")
print(f"偏位角: {result.attitude_angle_deg:.2f}°")
print(f"最小油膜厚度: {result.min_film_thickness_m * 1e6:.2f} μm")
print(f"功率损耗: {result.power_loss_w:.1f} W")

"""
示例2：不对中轴承分析
======================
演示含轴线不对中的尾管轴承分析。
轴承参数：D=200mm, L=200mm (L/D=1), 双轴向槽 90°/270°, 张角30°
不对中角度：γ=0.0003 rad（垂直面）
"""

from jakeman_bearing import analyze_bearing
from jakeman_bearing.visualization import (
    plot_pressure_3d,
    plot_cavitation_map,
    plot_film_thickness,
    plot_journal_center,
)

# 不对中尾管轴承（L/D=1，双轴向槽）
result = analyze_bearing(
    diameter=0.200,                     # 200 mm
    length=0.200,                       # 200 mm (L/D=1)
    clearance=0.000280,                 # 0.28 mm 直径间隙
    speed_rps=300.0 / 60.0,            # 300 rpm
    viscosity=0.050,                    # 50 mPa·s
    eccentricity_ratio=0.5,
    misalignment_vertical=0.0003,       # γ = 0.3 mrad（垂直面不对中）
    misalignment_horizontal=0.0,
    groove_type="axial_dual",
    groove_positions=[90.0, 270.0],     # 双轴向槽位置
    groove_width=30.0,                  # 张角 30°
    supply_pressure=101325.0,           # 1 atm 供油压力
    n_circumferential=72,
    n_axial=14,
)

print(result.summary())

# 可视化
plot_pressure_3d(result, save_path="example2_pressure_3d.png")
plot_cavitation_map(result, save_path="example2_cavitation.png")
plot_film_thickness(result, save_path="example2_film_thickness.png")
plot_journal_center(result, save_path="example2_journal_center.png")

print(f"\n承载力: {result.load_capacity_n:.1f} N")
print(f"偏位角: {result.attitude_angle_deg:.2f}°")
print(f"垂直力矩 My: {result.moment_vertical_nm:.2f} N·m")
print(f"水平力矩 Mx: {result.moment_horizontal_nm:.2f} N·m")
print(f"最小油膜厚度: {result.min_film_thickness_m * 1e6:.2f} μm")

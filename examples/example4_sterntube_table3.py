"""
示例4：论文 Table 3 尾管轴承实例复现
=======================================
复现 Jakeman (1984) 论文 Table 3 的尾管轴承工况。
轴承参数：D=800mm, L=1200mm, Cd=1.4mm, N=80rpm, η=0.125 Pa·s
不对中角度：γ=0.0002 rad（垂直面）
双轴向槽：90°/270°，张角30°

论文目标结果：
  承载力 W ≈ 770,150 N
  力矩   M ≈ 6.591×10⁷ N·mm = 65,910 N·m
"""

from jakeman_bearing import analyze_bearing
from jakeman_bearing.visualization import (
    plot_pressure_3d,
    plot_pressure_contour,
    plot_cavitation_map,
    plot_film_thickness,
    plot_pressure_profile,
)

# 论文 Table 3 尾管轴承（载荷模式）
result = analyze_bearing(
    diameter=0.800,                     # 800 mm
    length=1.200,                       # 1200 mm
    clearance=0.001400,                 # 1.4 mm 直径间隙
    speed_rps=80.0 / 60.0,             # 80 rpm
    viscosity=0.125,                    # 125 mPa·s（海水润滑）
    eccentricity_ratio=None,
    load=770150.0,                      # 目标载荷 ≈ 770,150 N
    load_direction=270.0,               # 垂直向下
    misalignment_vertical=0.0002,       # γ = 0.2 mrad
    misalignment_horizontal=0.0,
    groove_type="axial_dual",
    groove_positions=[90.0, 270.0],
    groove_width=30.0,
    supply_pressure=101325.0,
    n_circumferential=72,
    n_axial=14,
    axial_grading_factor=1.5,           # 端部网格加密
)

print(result.summary())

# 与论文结果对比
W_paper = 770150.0
M_paper = 65910.0  # N·m (6.591×10⁷ N·mm)
M_computed = (result.moment_vertical_nm**2 + result.moment_horizontal_nm**2) ** 0.5

W_error = abs(result.load_capacity_n - W_paper) / W_paper * 100
M_error = abs(M_computed - M_paper) / M_paper * 100

print("\n── 与论文 Table 3 对比 ──")
print(f"  承载力: 计算={result.load_capacity_n:.0f} N, 论文≈{W_paper:.0f} N, 误差={W_error:.1f}%")
print(f"  合成力矩: 计算={M_computed:.0f} N·m, 论文≈{M_paper:.0f} N·m, 误差={M_error:.1f}%")
print(f"  偏位角: {result.attitude_angle_deg:.2f}°")
print(f"  最小油膜厚度: {result.min_film_thickness_m * 1e6:.1f} μm")

# 可视化
plot_pressure_3d(result, save_path="example4_pressure_3d.png")
plot_pressure_contour(result, save_path="example4_pressure_contour.png")
plot_cavitation_map(result, save_path="example4_cavitation.png")
plot_film_thickness(result, save_path="example4_film_thickness.png")
plot_pressure_profile(result, save_path="example4_pressure_profile.png")

# 保存结果
result.to_csv("example4_table3_results.csv")
print("\n结果已保存至 example4_table3_results.csv")

"""
示例3：给定载荷求偏心（载荷模式）
====================================
演示载荷模式：已知轴承承受的载荷，自动用 Brent 法反求偏心比。
轴承参数：D=100mm, L=100mm, Cd=0.1mm, N=1500rpm, η=0.030 Pa·s
目标载荷：W=5000 N（垂直向下，270°方向）
"""

from jakeman_bearing import analyze_bearing
from jakeman_bearing.visualization import (
    plot_pressure_contour,
    plot_cavitation_map,
    plot_journal_center,
)

# 载荷模式：设置 load 参数，不设置 eccentricity_ratio
result = analyze_bearing(
    diameter=0.100,                     # 100 mm
    length=0.100,                       # 100 mm
    clearance=0.000100,                 # 0.1 mm 直径间隙
    speed_rps=1500.0 / 60.0,           # 1500 rpm
    viscosity=0.030,                    # 30 mPa·s
    eccentricity_ratio=None,            # 不指定偏心比
    load=5000.0,                        # 目标载荷 5000 N
    load_direction=270.0,               # 载荷方向：正下方
    groove_type="axial_dual",
    groove_positions=[90.0, 270.0],
    groove_width=30.0,
    supply_pressure=101325.0,
    n_circumferential=72,
    n_axial=10,
)

print(result.summary())

# 可视化
plot_pressure_contour(result, save_path="example3_pressure_contour.png")
plot_cavitation_map(result, save_path="example3_cavitation.png")
plot_journal_center(result, save_path="example3_journal_center.png")

# 反求得到的偏心比
epsilon = result.input_params.eccentricity_ratio
print(f"\n目标载荷: 5000.0 N")
print(f"计算承载力: {result.load_capacity_n:.2f} N")
print(f"反求偏心比 ε: {epsilon:.4f}")
print(f"偏位角: {result.attitude_angle_deg:.2f}°")
print(f"最小油膜厚度: {result.min_film_thickness_m * 1e6:.2f} μm")

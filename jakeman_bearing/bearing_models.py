"""
bearing_models.py — 滑动轴承分析数据模型

包含三个核心 dataclass：
- GrooveConfig: 供油槽配置
- BearingInput: 轴承分析输入参数（默认值 = 论文 Table 1 Case 1）
- BearingOutput: 轴承分析输出结果
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class GrooveConfig:
    """供油槽配置。

    groove_type:
        "circumferential_360" — 360°环形槽
        "axial_dual"          — 双轴向槽
        "axial_single"        — 单轴向槽
        "none"                — 无供油槽
    """

    groove_type: str
    angular_positions_deg: list[float]
    angular_width_deg: float
    supply_pressure_pa: float
    axial_position_ratio: float = 0.5
    axial_width_ratio: float = 0.05


@dataclass
class BearingInput:
    """轴承分析输入参数。

    所有默认值对应论文 Table 1 Case 1（曲轴轴承对齐工况，ε=0.6）。
    零参数构造即可复现论文结果。
    """

    # ── 几何参数 ──
    diameter_m: float = 0.0635
    length_m: float = 0.02368
    clearance_m: float = 0.0000635

    # ── 工况参数 ──
    speed_rps: float = 2000.0 / 60.0
    viscosity_pa_s: float = 0.014

    # ── 偏心模式 ──
    eccentricity_ratio: float | None = 0.6
    attitude_angle_rad: float = 0.0

    # ── 载荷模式 ──
    load_n: float | None = None
    load_direction_deg: float = 270.0

    # ── 不对中参数 ──
    misalignment_vertical_rad: float = 0.0
    misalignment_horizontal_rad: float = 0.0

    # ── 压力参数 ──
    cavitation_pressure_pa: float = 0.0
    ambient_pressure_pa: float = 0.0

    # ── 供油槽 ──
    groove: GrooveConfig | None = None

    # ── 网格参数 ──
    n_circumferential: int = 72
    n_axial: int = 10
    axial_grading_factor: float = 1.0

    # ── 求解器参数 ──
    over_relaxation_factor: float = 1.7
    max_iterations: int = 10000
    convergence_tol: float = 1e-4

    def validate(self) -> None:
        """验证所有输入参数，无效时抛出 ValueError。"""
        # 偏心比范围
        if self.eccentricity_ratio is not None:
            if self.eccentricity_ratio <= 0 or self.eccentricity_ratio >= 1:
                raise ValueError(
                    f"eccentricity_ratio={self.eccentricity_ratio}: "
                    f"必须在开区间 (0, 1) 内"
                )

        # 几何参数正值
        for name in ("diameter_m", "length_m", "clearance_m"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"{name}={val}: 必须为正数")

        # 工况参数正值
        for name in ("speed_rps", "viscosity_pa_s"):
            val = getattr(self, name)
            if val <= 0:
                raise ValueError(f"{name}={val}: 必须为正数")

        # 偏心/载荷互斥
        if self.eccentricity_ratio is not None and self.load_n is not None:
            raise ValueError("不能同时指定 eccentricity_ratio 和 load_n")
        if self.eccentricity_ratio is None and self.load_n is None:
            raise ValueError("必须指定 eccentricity_ratio 或 load_n 之一")

        # 网格数下限
        if self.n_circumferential < 4:
            raise ValueError(
                f"n_circumferential={self.n_circumferential}: 网格数必须 ≥ 4"
            )
        if self.n_axial < 4:
            raise ValueError(
                f"n_axial={self.n_axial}: 网格数必须 ≥ 4"
            )

        # SOR 因子范围
        if self.over_relaxation_factor <= 0 or self.over_relaxation_factor >= 2:
            raise ValueError(
                f"over_relaxation_factor={self.over_relaxation_factor}: "
                f"必须在 (0, 2) 内"
            )

    @staticmethod
    def default_groove() -> GrooveConfig:
        """返回论文 Table 1 的默认供油槽配置（360°环形槽）。"""
        return GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,       # 0.2067 MPa
            axial_position_ratio=0.5,        # 轴承中部
            axial_width_ratio=0.2145,        # 5.08mm / 23.68mm
        )


@dataclass
class BearingOutput:
    """轴承分析输出结果。"""

    # 压力场与空化
    pressure_field_pa: np.ndarray
    cavitation_matrix: np.ndarray
    film_thickness_field_m: np.ndarray

    # 承载力
    load_capacity_n: float
    load_vertical_n: float
    load_horizontal_n: float
    attitude_angle_deg: float

    # 力矩
    moment_vertical_nm: float
    moment_horizontal_nm: float

    # 油膜
    min_film_thickness_m: float
    min_film_location: tuple[float, float]

    # 流量
    side_leakage_flow_m3s: float

    # 功率
    power_loss_w: float
    friction_force_n: float

    # 动态系数
    stiffness_coefficients: np.ndarray
    damping_coefficients: np.ndarray

    # 求解器信息
    iterations: int
    converged: bool
    residual: float

    # 输入回显
    input_params: BearingInput

    def summary(self) -> str:
        """生成包含所有性能参数的文本摘要。"""
        inp = self.input_params
        lines = [
            "=" * 60,
            "  滑动轴承流体动力学分析结果摘要",
            "=" * 60,
            "",
            "── 承载力 ──",
            f"  合成承载力:   {self.load_capacity_n:.4f} N",
            f"  垂直分量 Fy:  {self.load_vertical_n:.4f} N",
            f"  水平分量 Fx:  {self.load_horizontal_n:.4f} N",
            f"  偏位角:       {self.attitude_angle_deg:.4f} °",
            "",
            "── 力矩 ──",
            f"  垂直力矩 My:  {self.moment_vertical_nm:.6f} N·m",
            f"  水平力矩 Mx:  {self.moment_horizontal_nm:.6f} N·m",
            "",
            "── 油膜 ──",
            f"  最小油膜厚度: {self.min_film_thickness_m:.6e} m",
            f"  最薄处位置:   θ={self.min_film_location[0]:.2f}°, "
            f"s={self.min_film_location[1]:.4f} m",
            "",
            "── 流量 ──",
            f"  侧漏流量:     {self.side_leakage_flow_m3s:.6e} m³/s",
            "",
            "── 功率 ──",
            f"  功率损耗:     {self.power_loss_w:.4f} W",
            f"  摩擦力:       {self.friction_force_n:.4f} N",
            "",
            "── 输入参数回显 ──",
            f"  轴承内径:     {inp.diameter_m} m",
            f"  轴承长度:     {inp.length_m} m",
            f"  直径间隙:     {inp.clearance_m} m",
            f"  转速:         {inp.speed_rps} r/s",
            f"  粘度:         {inp.viscosity_pa_s} Pa·s",
            f"  偏心比:       {inp.eccentricity_ratio}",
            f"  载荷:         {inp.load_n} N",
            f"  不对中 γ:     {inp.misalignment_vertical_rad} rad",
            f"  不对中 λ:     {inp.misalignment_horizontal_rad} rad",
            f"  周向网格:     {inp.n_circumferential}",
            f"  轴向网格:     {inp.n_axial}",
            "",
            "── 求解器信息 ──",
            f"  迭代次数:     {self.iterations}",
            f"  收敛状态:     {'已收敛' if self.converged else '未收敛'}",
            f"  最终残差:     {self.residual:.6e}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_csv(self, filepath: str) -> None:
        """保存压力场和性能参数为 CSV，精度满足往返误差 ≤ 1e-10。

        使用 repr() 格式化 float64 值以保证完整精度。
        """
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # 性能参数段
            writer.writerow(["# Performance Parameters"])
            writer.writerow(["load_capacity_n", repr(self.load_capacity_n)])
            writer.writerow(["load_vertical_n", repr(self.load_vertical_n)])
            writer.writerow(["load_horizontal_n", repr(self.load_horizontal_n)])
            writer.writerow(["attitude_angle_deg", repr(self.attitude_angle_deg)])
            writer.writerow(["moment_vertical_nm", repr(self.moment_vertical_nm)])
            writer.writerow(["moment_horizontal_nm", repr(self.moment_horizontal_nm)])
            writer.writerow(["min_film_thickness_m", repr(self.min_film_thickness_m)])
            writer.writerow([
                "min_film_location",
                repr(self.min_film_location[0]),
                repr(self.min_film_location[1]),
            ])
            writer.writerow(["side_leakage_flow_m3s", repr(self.side_leakage_flow_m3s)])
            writer.writerow(["power_loss_w", repr(self.power_loss_w)])
            writer.writerow(["friction_force_n", repr(self.friction_force_n)])
            writer.writerow(["iterations", self.iterations])
            writer.writerow(["converged", self.converged])
            writer.writerow(["residual", repr(self.residual)])

            # 压力场段
            writer.writerow([])
            writer.writerow(["# Pressure Field (rows=circumferential, cols=axial)"])
            rows, cols = self.pressure_field_pa.shape
            writer.writerow(["shape", rows, cols])
            for i in range(rows):
                writer.writerow([repr(float(v)) for v in self.pressure_field_pa[i, :]])

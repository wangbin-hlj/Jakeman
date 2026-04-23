# jakeman_bearing — 滑动轴承流体动力学分析包
# 基于 Jakeman (1984) 流量连续性方法

from __future__ import annotations

import copy
import math

import numpy as np

from jakeman_bearing.bearing_geometry import (
    compute_eccentricity_components,
    compute_element_corner_thicknesses,
)
from jakeman_bearing.bearing_models import BearingInput, BearingOutput, GrooveConfig
from jakeman_bearing.bearing_postprocess import (
    compute_load_capacity,
    compute_moments,
    compute_power_loss,
    compute_side_leakage,
    find_min_film_thickness,
)
from jakeman_bearing.bearing_practical import solve_for_load
from jakeman_bearing.bearing_solver import BearingSolver
from jakeman_bearing.visualization import (
    plot_pressure_3d,
    plot_pressure_contour,
    plot_cavitation_map,
    plot_film_thickness,
    plot_pressure_profile,
    plot_journal_center,
)


def analyze_bearing(
    diameter: float = 0.0635,
    length: float = 0.02368,
    clearance: float = 0.0000635,
    speed_rps: float = 2000.0 / 60.0,
    viscosity: float = 0.014,
    eccentricity_ratio: float | None = 0.6,
    load: float | None = None,
    load_direction: float = 270.0,
    misalignment_vertical: float = 0.0,
    misalignment_horizontal: float = 0.0,
    groove_type: str | None = "circumferential_360",
    groove_positions: list[float] | None = None,
    groove_width: float = 360.0,
    supply_pressure: float = 206700.0,
    n_circumferential: int = 72,
    n_axial: int = 10,
    **kwargs,
) -> BearingOutput:
    """分析滑动轴承性能的公共接口。

    自动检测分析模式：
    - 提供 eccentricity_ratio → 偏心模式（直接求解）
    - 提供 load → 载荷模式（Brent 法反求偏心）

    Parameters
    ----------
    diameter : float
        轴承内径 (m)，默认 0.0635 m。
    length : float
        轴承长度 (m)，默认 0.02368 m。
    clearance : float
        直径间隙 Cd (m)，默认 0.0000635 m。
    speed_rps : float
        转速 (r/s)，默认 2000/60 r/s。
    viscosity : float
        动力粘度 (Pa·s)，默认 0.014 Pa·s。
    eccentricity_ratio : float | None
        偏心比 ε ∈ (0, 1)。与 load 互斥。
    load : float | None
        目标载荷 (N)。与 eccentricity_ratio 互斥。
    load_direction : float
        载荷方向角 (°)，默认 270°。
    misalignment_vertical : float
        垂直面不对中角 γ (rad)，默认 0。
    misalignment_horizontal : float
        水平面不对中角 λ (rad)，默认 0。
    groove_type : str | None
        供油槽类型："circumferential_360"、"axial_dual"、"axial_single"、"none" 或 None。
    groove_positions : list[float] | None
        供油槽角度位置 (°)。若为 None，按 groove_type 使用默认值。
    groove_width : float
        供油槽角度宽度 (°)，默认 360°。
    supply_pressure : float
        供油压力 (Pa)，默认 206700 Pa。
    n_circumferential : int
        周向网格数，默认 72。
    n_axial : int
        轴向网格数，默认 10。
    **kwargs
        传递给 BearingInput 的额外参数。

    Returns
    -------
    BearingOutput
        完整的轴承分析结果。

    Raises
    ------
    ValueError
        参数无效时，错误信息包含步骤名称。
    RuntimeError
        求解器失败时，错误信息包含步骤名称。
    """
    # ── 步骤1: 构建 GrooveConfig ──────────────────────────────
    try:
        resolved_groove_type = groove_type if groove_type is not None else "none"

        if resolved_groove_type == "none":
            groove_config = GrooveConfig(
                groove_type="none",
                angular_positions_deg=[],
                angular_width_deg=0,
                supply_pressure_pa=0,
            )
        elif resolved_groove_type == "circumferential_360":
            positions = groove_positions if groove_positions is not None else [0]
            groove_config = GrooveConfig(
                groove_type="circumferential_360",
                angular_positions_deg=positions,
                angular_width_deg=groove_width,
                supply_pressure_pa=supply_pressure,
                axial_position_ratio=0.5,
                axial_width_ratio=0.2145,
            )
        elif resolved_groove_type == "axial_dual":
            positions = groove_positions if groove_positions is not None else [90, 270]
            groove_config = GrooveConfig(
                groove_type="axial_dual",
                angular_positions_deg=positions,
                angular_width_deg=groove_width,
                supply_pressure_pa=supply_pressure,
            )
        elif resolved_groove_type == "axial_single":
            positions = groove_positions if groove_positions is not None else [90]
            groove_config = GrooveConfig(
                groove_type="axial_single",
                angular_positions_deg=positions,
                angular_width_deg=groove_width,
                supply_pressure_pa=supply_pressure,
            )
        else:
            raise ValueError(
                f"未知的 groove_type='{groove_type}'，"
                f"有效值为 'circumferential_360'、'axial_dual'、'axial_single'、'none'"
            )
    except ValueError as e:
        raise ValueError(f"[供油槽配置] {e}") from e

    # ── 步骤2: 验证模式互斥 ───────────────────────────────────
    if eccentricity_ratio is not None and load is not None:
        raise ValueError(
            "[参数验证] 不能同时指定 eccentricity_ratio 和 load，请只提供其中一个"
        )
    if eccentricity_ratio is None and load is None:
        raise ValueError(
            "[参数验证] 必须指定 eccentricity_ratio 或 load 之一"
        )

    # ── 步骤3: 构建 BearingInput ──────────────────────────────
    try:
        if load is not None:
            # 载荷模式：eccentricity_ratio=None, load_n=load
            bearing_input = BearingInput(
                diameter_m=diameter,
                length_m=length,
                clearance_m=clearance,
                speed_rps=speed_rps,
                viscosity_pa_s=viscosity,
                eccentricity_ratio=None,
                load_n=load,
                load_direction_deg=load_direction,
                misalignment_vertical_rad=misalignment_vertical,
                misalignment_horizontal_rad=misalignment_horizontal,
                groove=groove_config,
                n_circumferential=n_circumferential,
                n_axial=n_axial,
                **kwargs,
            )
        else:
            # 偏心模式：eccentricity_ratio 已提供
            bearing_input = BearingInput(
                diameter_m=diameter,
                length_m=length,
                clearance_m=clearance,
                speed_rps=speed_rps,
                viscosity_pa_s=viscosity,
                eccentricity_ratio=eccentricity_ratio,
                load_n=None,
                load_direction_deg=load_direction,
                misalignment_vertical_rad=misalignment_vertical,
                misalignment_horizontal_rad=misalignment_horizontal,
                groove=groove_config,
                n_circumferential=n_circumferential,
                n_axial=n_axial,
                **kwargs,
            )
        bearing_input.validate()
    except ValueError as e:
        raise ValueError(f"[参数验证] {e}") from e

    # ── 步骤4: 执行分析 ───────────────────────────────────────
    if load is not None:
        # 载荷模式：调用 solve_for_load
        try:
            output = solve_for_load(
                bearing_input=bearing_input,
                target_load_n=load,
                load_direction_deg=load_direction,
            )
        except ValueError as e:
            raise ValueError(f"[载荷反求] {e}") from e
        except RuntimeError as e:
            raise RuntimeError(f"[载荷反求] {e}") from e
        return output

    else:
        # 偏心模式：直接求解并完整后处理
        try:
            solver = BearingSolver(bearing_input)
        except ValueError as e:
            raise ValueError(f"[参数验证] {e}") from e

        try:
            result = solver.solve()
        except RuntimeError as e:
            raise RuntimeError(f"[压力场求解] {e}") from e

        # 后处理
        try:
            # 1. 承载力
            Fy, Fx, F_total, attitude_deg = compute_load_capacity(
                result.pressure_field_pa,
                solver.theta_centers,
                solver.delta_a,
                solver.delta_c,
            )

            # 2. 力矩
            My, Mx = compute_moments(
                result.pressure_field_pa,
                solver.theta_centers,
                solver.s_centers,
                solver.delta_a,
                solver.delta_c,
            )

            # 3. 最小油膜厚度
            min_h, min_loc = find_min_film_thickness(
                result.film_thickness_field_m,
                solver.theta_centers,
                solver.s_centers,
            )

            # 4. 侧漏流量 — 需要重新计算 H 系数
            radial_clearance = bearing_input.clearance_m / 2.0
            epsilon = bearing_input.eccentricity_ratio
            attitude_rad = bearing_input.attitude_angle_rad
            ecy = epsilon * radial_clearance * math.cos(attitude_rad)
            ecx = epsilon * radial_clearance * math.sin(attitude_rad)

            gamma = bearing_input.misalignment_vertical_rad
            lam = bearing_input.misalignment_horizontal_rad
            _esy_e, _esx_e, e_edges, psi_edges = compute_eccentricity_components(
                ecy, ecx, gamma, lam, solver.s_edges
            )
            ha, hb, hc, hd = compute_element_corner_thicknesses(
                bearing_input.clearance_m, e_edges, psi_edges,
                solver.theta_edges, solver.s_edges,
            )
            _Hci, Hai, _Hco, Hao = solver._compute_H_coefficients(ha, hb, hc, hd)

            side_leakage = compute_side_leakage(
                result.pressure_field_pa,
                Hao, Hai,
                bearing_input.ambient_pressure_pa,
                solver.delta_a,
                solver.delta_c,
            )

            # 5. 功率损耗
            power_loss, friction_force = compute_power_loss(
                result.film_thickness_field_m,
                bearing_input.viscosity_pa_s,
                solver.U,
                solver.delta_a,
                solver.delta_c,
                result.cavitation_matrix,
            )
        except Exception as e:
            raise RuntimeError(f"[后处理计算] {e}") from e

        # 构建完整输出
        output = BearingOutput(
            pressure_field_pa=result.pressure_field_pa,
            cavitation_matrix=result.cavitation_matrix,
            film_thickness_field_m=result.film_thickness_field_m,
            load_capacity_n=F_total,
            load_vertical_n=Fy,
            load_horizontal_n=Fx,
            attitude_angle_deg=attitude_deg,
            moment_vertical_nm=My,
            moment_horizontal_nm=Mx,
            min_film_thickness_m=min_h,
            min_film_location=min_loc,
            side_leakage_flow_m3s=side_leakage,
            power_loss_w=power_loss,
            friction_force_n=friction_force,
            stiffness_coefficients=np.zeros((2, 2), dtype=np.float64),
            damping_coefficients=np.zeros((2, 2), dtype=np.float64),
            iterations=result.iterations,
            converged=result.converged,
            residual=result.residual,
            input_params=bearing_input,
        )
        return output

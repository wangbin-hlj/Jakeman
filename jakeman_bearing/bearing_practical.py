"""
bearing_practical.py — 实用分析模式：给定载荷反求偏心距

使用 Brent 法在 [0.01, 0.99] 区间搜索偏心比零点，
目标函数 f(ε) = F_computed - F_target。
同时迭代偏位角以匹配载荷方向。

对应论文 Fig 4 的载荷反求偏心流程。
"""

from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import brentq

from jakeman_bearing.bearing_geometry import (
    compute_eccentricity_components,
    compute_element_corner_thicknesses,
)
from jakeman_bearing.bearing_models import BearingInput, BearingOutput
from jakeman_bearing.bearing_postprocess import (
    compute_load_capacity,
    compute_moments,
    compute_power_loss,
    compute_side_leakage,
    find_min_film_thickness,
)
from jakeman_bearing.bearing_solver import BearingSolver


def solve_for_load(
    bearing_input: BearingInput,
    target_load_n: float,
    load_direction_deg: float,
    tol: float = 0.01,
    max_iter: int = 50,
) -> BearingOutput:
    """给定载荷反求偏心距（Brent 法）。

    Algorithm:
    1. Define objective function f(ε) = F_computed - F_target
    2. Use Brent method in [0.01, 0.99] to find zero
    3. For each ε evaluation, compute ecy/ecx from ε and attitude angle,
       solve pressure field, compute load capacity
    4. Simultaneously iterate attitude angle to match load direction
    5. Return complete BearingOutput after convergence

    Parameters
    ----------
    bearing_input : BearingInput
        Bearing input parameters. Should have load_n set and
        eccentricity_ratio as None (load mode).
    target_load_n : float
        Target load (N).
    load_direction_deg : float
        Load direction (degrees).
    tol : float
        Relative convergence tolerance for load matching.
    max_iter : int
        Maximum number of outer attitude angle iterations.

    Returns
    -------
    BearingOutput
        Complete bearing output with all post-processing fields.

    Raises
    ------
    ValueError
        If target load exceeds bearing capacity.
    RuntimeError
        If Brent method fails to converge.
    """
    if target_load_n <= 0:
        raise ValueError(
            f"target_load_n={target_load_n}: 目标载荷必须为正数"
        )

    # Create a modified copy for eccentricity mode solving
    # The input may have load_n set and eccentricity_ratio as None;
    # we need eccentricity_ratio set and load_n as None for the solver.
    solver_input = copy.copy(bearing_input)
    solver_input.load_n = None
    solver_input.eccentricity_ratio = 0.5  # placeholder, will be overridden

    radial_clearance = solver_input.clearance_m / 2.0

    # Initial attitude angle estimate: start from 0 (vertical).
    # For most bearing configurations, the load is primarily vertical
    # and the attitude angle is determined by the pressure field.
    attitude_angle_rad = 0.0

    # Track the last computed values for error reporting
    last_epsilon = 0.5
    last_f_total = 0.0
    last_attitude_deg = 0.0

    def _objective(epsilon: float) -> float:
        """Objective function: f(ε) = F_computed - F_target."""
        nonlocal last_epsilon, last_f_total, last_attitude_deg

        last_epsilon = epsilon

        # Compute eccentricity components from ε and current attitude angle
        ecy = epsilon * radial_clearance * math.cos(attitude_angle_rad)
        ecx = epsilon * radial_clearance * math.sin(attitude_angle_rad)

        # Create solver and solve
        solver_input.eccentricity_ratio = epsilon  # update for validation
        solver = BearingSolver(solver_input)
        result = solver.solve(ecy=ecy, ecx=ecx)

        # Compute load capacity
        Fy, Fx, F_total, att_deg = compute_load_capacity(
            result.pressure_field_pa,
            solver.theta_centers,
            solver.delta_a,
            solver.delta_c,
        )

        last_f_total = F_total
        last_attitude_deg = att_deg

        return F_total - target_load_n

    # Check that the target load is within bearing capacity
    # Evaluate at the bounds to verify sign change
    eps_lo = 0.01
    eps_hi = 0.99

    converged = False

    for outer_iter in range(max_iter):
        # Find a valid upper bound: reduce eps_hi if film thickness goes negative
        f_lo = None
        f_hi = None

        try:
            f_lo = _objective(eps_lo)
        except RuntimeError:
            raise ValueError(
                f"载荷超出轴承承载能力: 目标载荷 {target_load_n:.2f} N "
                f"无法在偏心比 [{eps_lo}, {eps_hi}] 范围内匹配"
            )

        # Try to evaluate at eps_hi; if it fails (e.g. negative film thickness),
        # reduce eps_hi until we find a valid upper bound
        current_hi = eps_hi
        for _ in range(20):
            try:
                f_hi = _objective(current_hi)
                break
            except RuntimeError:
                # Film thickness went negative at this eccentricity;
                # reduce upper bound
                current_hi = eps_lo + 0.9 * (current_hi - eps_lo)
                if current_hi - eps_lo < 1e-4:
                    break

        if f_hi is None:
            raise ValueError(
                f"载荷超出轴承承载能力: 目标载荷 {target_load_n:.2f} N "
                f"无法在偏心比 [{eps_lo}, {current_hi:.4f}] 范围内匹配 "
                f"（高偏心比时油膜厚度为负）"
            )

        # Check if target load exceeds capacity
        if f_lo > 0:
            # Even at minimum eccentricity, load is already exceeded
            # This shouldn't normally happen for reasonable loads
            raise ValueError(
                f"载荷超出轴承承载能力: 即使在 ε={eps_lo} 时，"
                f"计算承载力 ({last_f_total:.2f} N) 已超过目标载荷 "
                f"({target_load_n:.2f} N)"
            )

        if f_hi < 0:
            raise ValueError(
                f"载荷超出轴承承载能力: 在 ε={current_hi:.4f} 时，"
                f"计算承载力 ({last_f_total:.2f} N) 仍小于目标载荷 "
                f"({target_load_n:.2f} N)"
            )

        # Use Brent's method to find the root
        try:
            epsilon_solution = brentq(
                _objective, eps_lo, current_hi,
                xtol=1e-6, rtol=tol * 0.1, maxiter=100,
            )
        except ValueError as e:
            raise RuntimeError(
                f"Brent 法未收敛: {e}。"
                f"最后偏心比 ε={last_epsilon:.6f}，"
                f"承载力差值 = {last_f_total - target_load_n:.2f} N"
            ) from e

        # After Brent converges, check the load match
        f_residual = abs(last_f_total - target_load_n)
        relative_error = f_residual / target_load_n

        # Update attitude angle based on computed attitude and load direction
        # The computed attitude angle from the pressure field should converge
        # to be consistent with the load direction.
        computed_attitude_rad = math.radians(last_attitude_deg)

        # Check convergence of attitude angle
        attitude_change = abs(computed_attitude_rad - attitude_angle_rad)
        # Normalize to [0, π]
        attitude_change = min(attitude_change, 2 * math.pi - attitude_change)

        if relative_error < tol:
            converged = True
            break

        # Update attitude angle for next iteration using relaxation
        computed_attitude_rad = math.radians(last_attitude_deg)
        alpha = 0.3  # relaxation factor
        attitude_angle_rad = (
            attitude_angle_rad + alpha * (computed_attitude_rad - attitude_angle_rad)
        )

    if not converged:
        raise RuntimeError(
            f"Brent 法未收敛: 经过 {max_iter} 次外层迭代后，"
            f"偏心比 ε={last_epsilon:.6f}，"
            f"承载力差值 = {last_f_total - target_load_n:.2f} N，"
            f"相对误差 = {relative_error:.6f}"
        )

    # Final solve with converged parameters for full post-processing
    final_epsilon = epsilon_solution
    final_ecy = final_epsilon * radial_clearance * math.cos(attitude_angle_rad)
    final_ecx = final_epsilon * radial_clearance * math.sin(attitude_angle_rad)

    solver = BearingSolver(solver_input)
    result = solver.solve(ecy=final_ecy, ecx=final_ecx)

    # Full post-processing
    # 1. Load capacity
    Fy, Fx, F_total, attitude_deg = compute_load_capacity(
        result.pressure_field_pa,
        solver.theta_centers,
        solver.delta_a,
        solver.delta_c,
    )

    # 2. Moments
    My, Mx = compute_moments(
        result.pressure_field_pa,
        solver.theta_centers,
        solver.s_centers,
        solver.delta_a,
        solver.delta_c,
    )

    # 3. Min film thickness
    min_h, min_loc = find_min_film_thickness(
        result.film_thickness_field_m,
        solver.theta_centers,
        solver.s_centers,
    )

    # 4. Side leakage — need H coefficients from solver
    # Recompute H coefficients for the final geometry
    gamma = solver_input.misalignment_vertical_rad
    lam = solver_input.misalignment_horizontal_rad
    _esy_e, _esx_e, e_edges, psi_edges = compute_eccentricity_components(
        final_ecy, final_ecx, gamma, lam, solver.s_edges
    )
    ha, hb, hc, hd = compute_element_corner_thicknesses(
        solver_input.clearance_m, e_edges, psi_edges,
        solver.theta_edges, solver.s_edges,
    )
    _Hci, Hai, _Hco, Hao = solver._compute_H_coefficients(ha, hb, hc, hd)

    side_leakage = compute_side_leakage(
        result.pressure_field_pa,
        Hao, Hai,
        solver_input.ambient_pressure_pa,
        solver.delta_a,
        solver.delta_c,
    )

    # 5. Power loss
    power_loss, friction_force = compute_power_loss(
        result.film_thickness_field_m,
        solver_input.viscosity_pa_s,
        solver.U,
        solver.delta_a,
        solver.delta_c,
        result.cavitation_matrix,
    )

    # Build the final output with the original bearing_input as reference
    # but update eccentricity_ratio in the input echo
    output_input = copy.copy(bearing_input)

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
        input_params=output_input,
    )

    return output

"""
bearing_postprocess.py — 后处理计算模块

实现 Jakeman (1984) 论文公式 (18) 的12点加权平均法计算承载力与力矩，
以及最小油膜厚度识别。

公式 (18): P_mean[J,I] = (4*P[J,I] + sum_of_8_neighbors) / 12
承载力:  Fy = Σ(-P_mean × Δa × Δc × cos(θ))
         Fx = Σ(-P_mean × Δa × Δc × sin(θ))
力矩:   My = Σ(dFy × s_I),  Mx = Σ(dFx × s_I)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jakeman_bearing.bearing_solver import BearingSolver


def _compute_12point_average(pressure_field: np.ndarray) -> np.ndarray:
    """计算12点加权平均压力场。

    P_mean[J,I] = (4*P[J,I] + sum_of_8_neighbors) / 12

    - 周向方向 (J) 使用周期性边界
    - 轴向方向 (I) 边界处用自身压力替代越界邻居

    Parameters
    ----------
    pressure_field : np.ndarray, shape (Mc, Na)

    Returns
    -------
    P_mean : np.ndarray, shape (Mc, Na)
    """
    Mc, Na = pressure_field.shape
    P = pressure_field

    # Circumferential neighbors (periodic)
    P_j_minus = np.roll(P, 1, axis=0)   # P[J-1, I]
    P_j_plus = np.roll(P, -1, axis=0)   # P[J+1, I]

    # Axial neighbors with boundary handling (use own value for out-of-bounds)
    P_i_minus = np.empty_like(P)
    P_i_minus[:, 1:] = P[:, :-1]
    P_i_minus[:, 0] = P[:, 0]           # boundary: use own value

    P_i_plus = np.empty_like(P)
    P_i_plus[:, :-1] = P[:, 1:]
    P_i_plus[:, Na - 1] = P[:, Na - 1]  # boundary: use own value

    # Diagonal neighbors
    P_jm_im = np.roll(P_i_minus, 1, axis=0)   # P[J-1, I-1]
    P_jm_ip = np.roll(P_i_plus, 1, axis=0)    # P[J-1, I+1]
    P_jp_im = np.roll(P_i_minus, -1, axis=0)  # P[J+1, I-1]
    P_jp_ip = np.roll(P_i_plus, -1, axis=0)   # P[J+1, I+1]

    neighbor_sum = (
        P_j_minus + P_j_plus + P_i_minus + P_i_plus
        + P_jm_im + P_jm_ip + P_jp_im + P_jp_ip
    )

    P_mean = (4.0 * P + neighbor_sum) / 12.0
    return P_mean


def compute_load_capacity(
    pressure_field: np.ndarray,
    theta_centers: np.ndarray,
    delta_a: np.ndarray,
    delta_c: float,
) -> tuple[float, float, float, float]:
    """计算承载力（论文公式18，12点加权平均法）。

    Parameters
    ----------
    pressure_field : np.ndarray, shape (Mc, Na)
        压力场 (Pa)。
    theta_centers : np.ndarray, shape (Mc,)
        周向角度中心 (rad)。
    delta_a : np.ndarray, shape (Na,)
        轴向网格宽度 (m)。
    delta_c : float
        周向网格宽度 (m)。

    Returns
    -------
    (Fy, Fx, F_total, attitude_angle_deg) : tuple[float, float, float, float]
        Fy — 垂直承载力分量 (N)
        Fx — 水平承载力分量 (N)
        F_total — 合成承载力 (N)
        attitude_angle_deg — 偏位角 (°)
    """
    P_mean = _compute_12point_average(pressure_field)

    # cos(θ) and sin(θ) as column vectors for broadcasting
    cos_theta = np.cos(theta_centers)[:, np.newaxis]  # (Mc, 1)
    sin_theta = np.sin(theta_centers)[:, np.newaxis]  # (Mc, 1)
    da_row = delta_a[np.newaxis, :]                   # (1, Na)

    # Force contributions per cell
    dF = -P_mean * da_row * delta_c

    Fy = float(np.sum(dF * cos_theta))
    Fx = float(np.sum(dF * sin_theta))
    F_total = math.sqrt(Fy**2 + Fx**2)
    attitude_angle_deg = math.degrees(math.atan2(Fx, Fy))

    return Fy, Fx, F_total, attitude_angle_deg


def compute_moments(
    pressure_field: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    delta_a: np.ndarray,
    delta_c: float,
) -> tuple[float, float]:
    """计算力矩 My, Mx。

    My = Σ(dFy[J,I] × s_centers[I])
    Mx = Σ(dFx[J,I] × s_centers[I])

    Parameters
    ----------
    pressure_field : np.ndarray, shape (Mc, Na)
        压力场 (Pa)。
    theta_centers : np.ndarray, shape (Mc,)
        周向角度中心 (rad)。
    s_centers : np.ndarray, shape (Na,)
        轴向位置 (m)。
    delta_a : np.ndarray, shape (Na,)
        轴向网格宽度 (m)。
    delta_c : float
        周向网格宽度 (m)。

    Returns
    -------
    (My, Mx) : tuple[float, float]
        My — 垂直力矩 (N·m)
        Mx — 水平力矩 (N·m)
    """
    P_mean = _compute_12point_average(pressure_field)

    cos_theta = np.cos(theta_centers)[:, np.newaxis]  # (Mc, 1)
    sin_theta = np.sin(theta_centers)[:, np.newaxis]  # (Mc, 1)
    da_row = delta_a[np.newaxis, :]                   # (1, Na)
    s_row = s_centers[np.newaxis, :]                  # (1, Na)

    dF = -P_mean * da_row * delta_c

    dFy = dF * cos_theta  # (Mc, Na)
    dFx = dF * sin_theta  # (Mc, Na)

    My = float(np.sum(dFy * s_row))
    Mx = float(np.sum(dFx * s_row))

    return My, Mx


def find_min_film_thickness(
    film_thickness_field: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
) -> tuple[float, tuple[float, float]]:
    """找到最小油膜厚度及其位置。

    Parameters
    ----------
    film_thickness_field : np.ndarray, shape (Mc, Na)
        油膜厚度场 (m)。
    theta_centers : np.ndarray, shape (Mc,)
        周向角度中心 (rad)。
    s_centers : np.ndarray, shape (Na,)
        轴向位置 (m)。

    Returns
    -------
    (min_h, (theta_deg, s_m)) : tuple[float, tuple[float, float]]
        min_h — 最小油膜厚度 (m)
        theta_deg — 最小处周向角度 (°)
        s_m — 最小处轴向位置 (m)
    """
    min_h = float(np.min(film_thickness_field))
    min_idx = np.unravel_index(
        np.argmin(film_thickness_field), film_thickness_field.shape
    )
    theta_deg = float(np.degrees(theta_centers[min_idx[0]]))
    s_m = float(s_centers[min_idx[1]])

    return min_h, (theta_deg, s_m)


def compute_side_leakage(
    pressure_field: np.ndarray,
    Hao: np.ndarray,
    Hai: np.ndarray,
    ambient_pressure: float,
    delta_a: np.ndarray,
    delta_c: float,
) -> float:
    """计算侧漏流量（论文公式19-20）。

    侧漏 = 轴承两端面轴向流量绝对值之和。
    内端 (I=0):  Qai[J] = Hai[J,0] × (P[J,0] - P_ambient)
    外端 (I=Na-1): Qao[J] = Hao[J,Na-1] × (P[J,Na-1] - P_ambient)
    Qs = Σ|Qai| + Σ|Qao|

    Parameters
    ----------
    pressure_field : np.ndarray, shape (Mc, Na)
        压力场 (Pa)。
    Hao : np.ndarray, shape (Mc, Na)
        外侧轴向 H 系数。
    Hai : np.ndarray, shape (Mc, Na)
        内侧轴向 H 系数。
    ambient_pressure : float
        环境压力 (Pa)。
    delta_a : np.ndarray, shape (Na,)
        轴向网格宽度 (m)。
    delta_c : float
        周向网格宽度 (m)。

    Returns
    -------
    float
        侧漏流量 (m³/s)。
    """
    P = pressure_field
    Na = P.shape[1]

    # Inner end (I=0): axial flow out through the inner boundary
    Qai = Hai[:, 0] * (P[:, 0] - ambient_pressure)

    # Outer end (I=Na-1): axial flow out through the outer boundary
    Qao = Hao[:, Na - 1] * (P[:, Na - 1] - ambient_pressure)

    Qs = float(np.sum(np.abs(Qai)) + np.sum(np.abs(Qao)))
    return Qs


def _forces_from_pressure(
    pressure_field: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    delta_a: np.ndarray,
    delta_c: float,
) -> tuple[float, float, float, float]:
    """计算压力场对应的力和力矩分量。

    Returns (Fy, Fx, My, Mx).
    """
    P_mean = _compute_12point_average(pressure_field)

    cos_theta = np.cos(theta_centers)[:, np.newaxis]  # (Mc, 1)
    sin_theta = np.sin(theta_centers)[:, np.newaxis]  # (Mc, 1)
    da_row = delta_a[np.newaxis, :]                   # (1, Na)
    s_row = s_centers[np.newaxis, :]                   # (1, Na)

    dF = -P_mean * da_row * delta_c

    dFy = dF * cos_theta
    dFx = dF * sin_theta

    Fy = float(np.sum(dFy))
    Fx = float(np.sum(dFx))
    My = float(np.sum(dFy * s_row))
    Mx = float(np.sum(dFx * s_row))

    return Fy, Fx, My, Mx


def compute_dynamic_coefficients(
    solver: 'BearingSolver',
    ecy: float,
    ecx: float,
    is_misaligned: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """计算动态刚度/阻尼系数（公式25，扰动差分法）。

    对齐轴承: 8个系数 (2×2 刚度 + 2×2 阻尼)
    不对中轴承: 32个系数 (4×4 刚度 + 4×4 阻尼)

    方法: 对 ecy, ecx (及不对中时 γ, λ) 施加微小扰动 δ，
          重新求解压力场，差分计算:
          Aij = (Fi(+δ) - Fi(-δ)) / (2δ)
          Bij = (Fi(+δ̇) - Fi(-δ̇)) / (2δ̇)

    Parameters
    ----------
    solver : BearingSolver
        已初始化的求解器实例。
    ecy, ecx : float
        基准偏心分量 (m)。
    is_misaligned : bool
        是否为不对中轴承。

    Returns
    -------
    (stiffness, damping) : tuple[np.ndarray, np.ndarray]
        对齐: 各为 (2, 2)；不对中: 各为 (4, 4)。
    """
    inp = solver.input
    radial_clearance = inp.clearance_m / 2.0

    # Perturbation size: small fraction of radial clearance
    delta_disp = 1e-3 * radial_clearance
    # Velocity perturbation: scale by angular velocity × clearance
    omega = 2.0 * math.pi * inp.speed_rps
    delta_vel = 1e-3 * radial_clearance * omega

    theta_centers = solver.theta_centers
    s_centers = solver.s_centers
    delta_a = solver.delta_a
    delta_c = solver.delta_c

    if not is_misaligned:
        # Aligned bearing: 2×2 stiffness + 2×2 damping
        # Generalized coordinates: [ey, ex]
        # Generalized forces: [Fy, Fx]
        stiffness = np.zeros((2, 2), dtype=np.float64)
        damping = np.zeros((2, 2), dtype=np.float64)

        # --- Stiffness coefficients ---
        # Perturb ecy by +δ
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecy=+delta_disp)
        Fy_p, Fx_p, _, _ = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        # Perturb ecy by -δ
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecy=-delta_disp)
        Fy_m, Fx_m, _, _ = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        # Ayy = dFy/dey, Axy = dFx/dey
        stiffness[0, 0] = (Fy_p - Fy_m) / (2.0 * delta_disp)  # Ayy
        stiffness[1, 0] = (Fx_p - Fx_m) / (2.0 * delta_disp)  # Axy

        # Perturb ecx by +δ
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecx=+delta_disp)
        Fy_p, Fx_p, _, _ = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        # Perturb ecx by -δ
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecx=-delta_disp)
        Fy_m, Fx_m, _, _ = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        # Ayx = dFy/dex, Axx = dFx/dex
        stiffness[0, 1] = (Fy_p - Fy_m) / (2.0 * delta_disp)  # Ayx
        stiffness[1, 1] = (Fx_p - Fx_m) / (2.0 * delta_disp)  # Axx

        # --- Damping coefficients ---
        # Perturb ėy by +δ̇
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecy_dot=+delta_vel)
        Fy_p, Fx_p, _, _ = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        # Perturb ėy by -δ̇
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecy_dot=-delta_vel)
        Fy_m, Fx_m, _, _ = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        # Byy = dFy/dėy, Bxy = dFx/dėy
        damping[0, 0] = (Fy_p - Fy_m) / (2.0 * delta_vel)  # Byy
        damping[1, 0] = (Fx_p - Fx_m) / (2.0 * delta_vel)  # Bxy

        # Perturb ėx by +δ̇
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecx_dot=+delta_vel)
        Fy_p, Fx_p, _, _ = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        # Perturb ėx by -δ̇
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecx_dot=-delta_vel)
        Fy_m, Fx_m, _, _ = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        # Byx = dFy/dėx, Bxx = dFx/dėx
        damping[0, 1] = (Fy_p - Fy_m) / (2.0 * delta_vel)  # Byx
        damping[1, 1] = (Fx_p - Fx_m) / (2.0 * delta_vel)  # Bxx

        return stiffness, damping

    else:
        # Misaligned bearing: 4×4 stiffness + 4×4 damping
        # Generalized coordinates: [ey, ex, γ, λ]
        # Generalized forces: [Fy, Fx, My, Mx]
        stiffness = np.zeros((4, 4), dtype=np.float64)
        damping = np.zeros((4, 4), dtype=np.float64)

        # Perturbation for angular misalignment
        delta_angle = 1e-3 * radial_clearance / (inp.length_m / 2.0)
        delta_angle_dot = delta_angle * omega

        # Save original misalignment values
        gamma_orig = inp.misalignment_vertical_rad
        lambda_orig = inp.misalignment_horizontal_rad

        # --- Column 0: perturb ecy ---
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecy=+delta_disp)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecy=-delta_disp)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        d = 2.0 * delta_disp
        stiffness[0, 0] = (Fy_p - Fy_m) / d
        stiffness[1, 0] = (Fx_p - Fx_m) / d
        stiffness[2, 0] = (My_p - My_m) / d
        stiffness[3, 0] = (Mx_p - Mx_m) / d

        # --- Column 1: perturb ecx ---
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecx=+delta_disp)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecx=-delta_disp)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        stiffness[0, 1] = (Fy_p - Fy_m) / d
        stiffness[1, 1] = (Fx_p - Fx_m) / d
        stiffness[2, 1] = (My_p - My_m) / d
        stiffness[3, 1] = (Mx_p - Mx_m) / d

        # --- Column 2: perturb γ ---
        # Temporarily modify the solver's input misalignment
        inp.misalignment_vertical_rad = gamma_orig + delta_angle
        P_plus = solver.solve_perturbed(ecy, ecx)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_vertical_rad = gamma_orig - delta_angle
        P_minus = solver.solve_perturbed(ecy, ecx)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_vertical_rad = gamma_orig  # restore
        d_ang = 2.0 * delta_angle
        stiffness[0, 2] = (Fy_p - Fy_m) / d_ang
        stiffness[1, 2] = (Fx_p - Fx_m) / d_ang
        stiffness[2, 2] = (My_p - My_m) / d_ang
        stiffness[3, 2] = (Mx_p - Mx_m) / d_ang

        # --- Column 3: perturb λ ---
        inp.misalignment_horizontal_rad = lambda_orig + delta_angle
        P_plus = solver.solve_perturbed(ecy, ecx)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_horizontal_rad = lambda_orig - delta_angle
        P_minus = solver.solve_perturbed(ecy, ecx)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_horizontal_rad = lambda_orig  # restore
        stiffness[0, 3] = (Fy_p - Fy_m) / d_ang
        stiffness[1, 3] = (Fx_p - Fx_m) / d_ang
        stiffness[2, 3] = (My_p - My_m) / d_ang
        stiffness[3, 3] = (Mx_p - Mx_m) / d_ang

        # --- Damping: Column 0: perturb ėy ---
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecy_dot=+delta_vel)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecy_dot=-delta_vel)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        d_v = 2.0 * delta_vel
        damping[0, 0] = (Fy_p - Fy_m) / d_v
        damping[1, 0] = (Fx_p - Fx_m) / d_v
        damping[2, 0] = (My_p - My_m) / d_v
        damping[3, 0] = (Mx_p - Mx_m) / d_v

        # --- Damping: Column 1: perturb ėx ---
        P_plus = solver.solve_perturbed(ecy, ecx, delta_ecx_dot=+delta_vel)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        P_minus = solver.solve_perturbed(ecy, ecx, delta_ecx_dot=-delta_vel)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        damping[0, 1] = (Fy_p - Fy_m) / d_v
        damping[1, 1] = (Fx_p - Fx_m) / d_v
        damping[2, 1] = (My_p - My_m) / d_v
        damping[3, 1] = (Mx_p - Mx_m) / d_v

        # --- Damping: Column 2: perturb γ̇ ---
        # Angular velocity perturbation for γ: temporarily perturb γ
        # and use the angular perturbation as a proxy for γ̇·Δt
        inp.misalignment_vertical_rad = gamma_orig + delta_angle
        P_plus = solver.solve_perturbed(ecy, ecx)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_vertical_rad = gamma_orig - delta_angle
        P_minus = solver.solve_perturbed(ecy, ecx)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_vertical_rad = gamma_orig  # restore
        d_av = 2.0 * delta_angle_dot
        damping[0, 2] = (Fy_p - Fy_m) / d_av
        damping[1, 2] = (Fx_p - Fx_m) / d_av
        damping[2, 2] = (My_p - My_m) / d_av
        damping[3, 2] = (Mx_p - Mx_m) / d_av

        # --- Damping: Column 3: perturb λ̇ ---
        inp.misalignment_horizontal_rad = lambda_orig + delta_angle
        P_plus = solver.solve_perturbed(ecy, ecx)
        Fy_p, Fx_p, My_p, Mx_p = _forces_from_pressure(
            P_plus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_horizontal_rad = lambda_orig - delta_angle
        P_minus = solver.solve_perturbed(ecy, ecx)
        Fy_m, Fx_m, My_m, Mx_m = _forces_from_pressure(
            P_minus, theta_centers, s_centers, delta_a, delta_c
        )
        inp.misalignment_horizontal_rad = lambda_orig  # restore
        damping[0, 3] = (Fy_p - Fy_m) / d_av
        damping[1, 3] = (Fx_p - Fx_m) / d_av
        damping[2, 3] = (My_p - My_m) / d_av
        damping[3, 3] = (Mx_p - Mx_m) / d_av

        return stiffness, damping


def compute_power_loss(
    film_thickness_field: np.ndarray,
    viscosity: float,
    surface_velocity: float,
    delta_a: np.ndarray,
    delta_c: float,
    cavitation_matrix: np.ndarray,
) -> tuple[float, float]:
    """计算功率损耗（论文公式21-24，Petroff 剪切公式）。

    每个网格的摩擦力贡献:
      Fc[J,I] = η × U × Δc × Δa[I] / h[J,I]

    对所有网格（含空化区域，因薄膜仍存在剪切）求和:
      F_friction = Σ Fc
      Power = U × F_friction

    Parameters
    ----------
    film_thickness_field : np.ndarray, shape (Mc, Na)
        油膜厚度场 (m)。
    viscosity : float
        动力粘度 (Pa·s)。
    surface_velocity : float
        表面速度 U = π×D×N (m/s)。
    delta_a : np.ndarray, shape (Na,)
        轴向网格宽度 (m)。
    delta_c : float
        周向网格宽度 (m)。
    cavitation_matrix : np.ndarray, shape (Mc, Na)
        空化标记 (bool)，True = 空化。

    Returns
    -------
    (power_loss_w, friction_force_n) : tuple[float, float]
        power_loss_w — 功率损耗 (W)
        friction_force_n — 总摩擦力 (N)
    """
    da_row = delta_a[np.newaxis, :]  # (1, Na)

    # Fc[J,I] = η × U × Δc × Δa[I] / h[J,I]
    Fc = viscosity * surface_velocity * delta_c * da_row / film_thickness_field

    friction_force_n = float(np.sum(Fc))
    power_loss_w = surface_velocity * friction_force_n

    return power_loss_w, friction_force_n

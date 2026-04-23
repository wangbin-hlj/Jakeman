"""
bearing_geometry.py — 油膜几何计算模块

实现 Jakeman (1984) 论文公式 (1)-(5)：
  (4) esy(s) = ecy + s × γ
  (5) esx(s) = ecx + s × λ
  (2) e(s)   = sqrt(esy² + esx²)
  (3) ψ(s)   = atan2(esx, esy)
  (1) h      = Cd/2 + e(s) × cos(θ - ψ(s))

所有角度使用弧度，clearance_m 为直径间隙 Cd。
"""

from __future__ import annotations

import numpy as np


def compute_eccentricity_components(
    ecy: float,
    ecx: float,
    gamma: float,
    lam: float,
    s_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算轴向各位置的偏心分量及合成偏心距（公式 2-5）。

    Parameters
    ----------
    ecy : float
        中心垂直偏心分量 (m)。
    ecx : float
        中心水平偏心分量 (m)。
    gamma : float
        垂直面倾斜角 γ (rad)。
    lam : float
        水平面倾斜角 λ (rad)。
    s_positions : np.ndarray
        轴向位置数组 (m)，以轴承中心为原点 (-L/2 ~ +L/2)。

    Returns
    -------
    (esy, esx, e, psi) : tuple of np.ndarray
        esy — 各轴向位置的垂直偏心分量 (m)
        esx — 各轴向位置的水平偏心分量 (m)
        e   — 合成偏心距 (m)
        psi — 偏位角 (rad)
    """
    s = np.asarray(s_positions, dtype=np.float64)
    esy = ecy + s * gamma          # 公式 (4)
    esx = ecx + s * lam            # 公式 (5)
    e = np.sqrt(esy**2 + esx**2)   # 公式 (2)
    psi = np.arctan2(esx, esy)     # 公式 (3)
    return esy, esx, e, psi


def compute_film_thickness(
    clearance_m: float,
    e_array: np.ndarray,
    psi_array: np.ndarray,
    theta_positions: np.ndarray,
) -> np.ndarray:
    """计算油膜厚度场（公式 1）。

    h = Cd/2 + e(s) × cos(θ - ψ(s))

    Parameters
    ----------
    clearance_m : float
        直径间隙 Cd (m)。径向间隙 = Cd/2。
    e_array : np.ndarray
        合成偏心距数组，shape (n_axial,)。
    psi_array : np.ndarray
        偏位角数组 (rad)，shape (n_axial,)。
    theta_positions : np.ndarray
        周向角度位置数组 (rad)，shape (n_theta,)。

    Returns
    -------
    film_thickness : np.ndarray
        油膜厚度二维数组，shape (n_theta, n_axial)。
    """
    theta = np.asarray(theta_positions, dtype=np.float64)  # (n_theta,)
    e = np.asarray(e_array, dtype=np.float64)              # (n_axial,)
    psi = np.asarray(psi_array, dtype=np.float64)          # (n_axial,)

    radial_clearance = clearance_m / 2.0

    # Broadcasting: theta[:, None] is (n_theta, 1), psi[None, :] is (1, n_axial)
    h = radial_clearance + e[np.newaxis, :] * np.cos(
        theta[:, np.newaxis] - psi[np.newaxis, :]
    )
    return h


def compute_element_corner_thicknesses(
    clearance_m: float,
    e_array: np.ndarray,
    psi_array: np.ndarray,
    theta_edges: np.ndarray,
    s_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """计算每个网格元素四角的油膜厚度。

    对于网格元素 (J, I)：
        ha = h(θ_J,     s_I)     — 上游内侧角
        hb = h(θ_J,     s_{I+1}) — 上游外侧角
        hc = h(θ_{J+1}, s_I)     — 下游内侧角
        hd = h(θ_{J+1}, s_{I+1}) — 下游外侧角

    Parameters
    ----------
    clearance_m : float
        直径间隙 Cd (m)。
    e_array : np.ndarray
        合成偏心距，在 s_edges 位置计算，shape (Na+1,)。
    psi_array : np.ndarray
        偏位角 (rad)，在 s_edges 位置计算，shape (Na+1,)。
    theta_edges : np.ndarray
        周向边界角度 (rad)，shape (Mc+1,)。
    s_edges : np.ndarray
        轴向边界位置 (m)，shape (Na+1,)。
        注意：e_array 和 psi_array 应在这些 s_edges 位置上计算。

    Returns
    -------
    (ha, hb, hc, hd) : tuple of np.ndarray
        每个数组 shape (Mc, Na)。
    """
    radial_clearance = clearance_m / 2.0

    theta_e = np.asarray(theta_edges, dtype=np.float64)  # (Mc+1,)
    e = np.asarray(e_array, dtype=np.float64)            # (Na+1,)
    psi = np.asarray(psi_array, dtype=np.float64)        # (Na+1,)

    # h(θ, s) = Cd/2 + e(s) * cos(θ - ψ(s))
    # Compute full thickness at all edge intersections: shape (Mc+1, Na+1)
    h_all = radial_clearance + e[np.newaxis, :] * np.cos(
        theta_e[:, np.newaxis] - psi[np.newaxis, :]
    )

    # Extract the four corners for each element (J, I)
    ha = h_all[:-1, :-1]   # (θ_J,     s_I)       — upstream inner
    hb = h_all[:-1, 1:]    # (θ_J,     s_{I+1})   — upstream outer
    hc = h_all[1:, :-1]    # (θ_{J+1}, s_I)       — downstream inner
    hd = h_all[1:, 1:]     # (θ_{J+1}, s_{I+1})   — downstream outer

    # Return copies to ensure contiguous arrays
    return ha.copy(), hb.copy(), hc.copy(), hd.copy()

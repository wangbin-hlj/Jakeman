"""
bearing_solver.py — 核心求解器

基于 Jakeman (1984) 流量连续性方法的压力场迭代求解器。
实现论文公式 (6)-(17)：
  (8-11) 压力流函数 Hci, Hai, Hco, Hao
  (12-16) 速度诱导流量 K
  (7) 压力求解公式
  (17) 空化流量 Qvo

与传统 Reynolds 方程 FDM 不同，本方法对每个网格元素直接列质量守恒方程，
通过引入空化流量项 Qvo 自动确定空化边界。
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from jakeman_bearing.bearing_geometry import (
    compute_eccentricity_components,
    compute_element_corner_thicknesses,
    compute_film_thickness,
)
from jakeman_bearing.bearing_models import BearingInput, BearingOutput, GrooveConfig


class BearingSolver:
    """基于 Jakeman 流量连续性方法的压力场求解器。"""

    def __init__(self, bearing_input: BearingInput):
        self.input = bearing_input
        bearing_input.validate()
        self._setup_grid()
        self._setup_groove_mask()

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------

    def _setup_grid(self) -> None:
        """初始化网格：θ 坐标、s 坐标、Δa、Δc、表面速度 U。"""
        inp = self.input
        Mc = inp.n_circumferential
        Na = inp.n_axial
        D = inp.diameter_m
        L = inp.length_m

        two_pi = 2.0 * math.pi

        # θ cell centres: θ_J = (J - 0.5) * 2π / Mc,  J = 1..Mc
        # Using 0-based indexing: θ_j = (j + 0.5) * 2π / Mc,  j = 0..Mc-1
        self.theta_centers = np.array(
            [(j + 0.5) * two_pi / Mc for j in range(Mc)], dtype=np.float64
        )

        # θ edges: θ_edge_j = j * 2π / Mc,  j = 0..Mc  (Mc+1 values)
        self.theta_edges = np.linspace(0.0, two_pi, Mc + 1)

        # Axial coordinates: -L/2 to +L/2
        half_L = L / 2.0

        if inp.axial_grading_factor <= 1.0 or abs(inp.axial_grading_factor - 1.0) < 1e-12:
            # Uniform grid
            self.s_edges = np.linspace(-half_L, half_L, Na + 1)
            self.delta_a = np.full(Na, L / Na, dtype=np.float64)
        else:
            # Graded mesh: end-refined (smaller spacing at ends, larger in middle)
            # Use a symmetric power-law grading
            gf = inp.axial_grading_factor
            # Generate normalised half-grid [0, 1] with Na/2 cells (or (Na+1)/2)
            # We build the full grid symmetrically
            half_n = Na // 2
            remainder = Na % 2

            # Normalised positions for each half
            # Using geometric grading: ratio between successive cells
            # r = gf^(1/(half_n-1)) if half_n > 1
            # Each half sums to half_sum (0.5 for even Na, adjusted for odd Na)
            if remainder == 1:
                # Odd Na: reserve space for middle cell
                # Each half sums to half_n / Na, middle cell gets 1/Na * gf^(half_n-1)
                # equivalent to the largest cell in the grading sequence
                half_sum = 0.5 * half_n / (half_n + 0.5)
            else:
                half_sum = 0.5

            if half_n > 1:
                r = gf ** (1.0 / (half_n - 1))
                # First cell size (normalised so sum = half_sum)
                if abs(r - 1.0) < 1e-12:
                    d0 = half_sum / half_n
                else:
                    d0 = half_sum * (1.0 - r) / (1.0 - r ** half_n)
                sizes_left = np.array([d0 * r ** k for k in range(half_n)])
            else:
                sizes_left = np.array([half_sum])

            if remainder == 1:
                # Odd Na: middle cell gets the remainder
                sizes_right = sizes_left[::-1]
                mid_size = 1.0 - 2.0 * np.sum(sizes_left)
                sizes_mid = np.array([mid_size])
                all_sizes = np.concatenate([sizes_left, sizes_mid, sizes_right])
            else:
                sizes_right = sizes_left[::-1]
                all_sizes = np.concatenate([sizes_left, sizes_right])

            # Scale to actual length
            all_sizes = all_sizes * L
            edges = np.zeros(Na + 1)
            edges[0] = -half_L
            for i in range(Na):
                edges[i + 1] = edges[i] + all_sizes[i]
            # Fix last edge exactly
            edges[-1] = half_L
            self.s_edges = edges
            self.delta_a = np.diff(edges)

        # s cell centres
        self.s_centers = 0.5 * (self.s_edges[:-1] + self.s_edges[1:])

        # Circumferential spacing in metres: Δc = π * D / Mc
        self.delta_c = math.pi * D / Mc

        # Surface velocity: U = π * D * N
        self.U = math.pi * D * inp.speed_rps

        self.Mc = Mc
        self.Na = Na

    # ------------------------------------------------------------------
    # Groove mask
    # ------------------------------------------------------------------

    def _setup_groove_mask(self) -> None:
        """根据 GrooveConfig 生成供油槽掩码矩阵 (Mc, Na)。"""
        inp = self.input
        Mc = self.Mc
        Na = self.Na

        groove = inp.groove if inp.groove is not None else BearingInput.default_groove()

        self.groove_mask = np.zeros((Mc, Na), dtype=bool)
        self.supply_pressure = groove.supply_pressure_pa

        if groove.groove_type == "none":
            return

        if groove.groove_type == "circumferential_360":
            # 360° circumferential groove at a given axial position ratio
            # Mark axial columns near axial_position_ratio
            axial_ratio = groove.axial_position_ratio  # 0~1, 0.5 = centre
            width_ratio = groove.axial_width_ratio

            L = inp.length_m
            half_L = L / 2.0
            # Groove centre in s coordinates
            groove_s_center = -half_L + axial_ratio * L
            groove_half_width = width_ratio * L / 2.0

            for i in range(Na):
                s_lo = self.s_edges[i]
                s_hi = self.s_edges[i + 1]
                s_mid = 0.5 * (s_lo + s_hi)
                if abs(s_mid - groove_s_center) <= groove_half_width:
                    self.groove_mask[:, i] = True

        elif groove.groove_type in ("axial_dual", "axial_single"):
            # Axial grooves at specified angular positions
            angular_positions_rad = [
                math.radians(a) % (2.0 * math.pi)
                for a in groove.angular_positions_deg
            ]
            half_width_rad = math.radians(groove.angular_width_deg) / 2.0

            for j in range(Mc):
                theta_j = self.theta_centers[j]
                for ang_pos in angular_positions_rad:
                    # Angular distance (periodic)
                    diff = abs(theta_j - ang_pos)
                    diff = min(diff, 2.0 * math.pi - diff)
                    if diff <= half_width_rad:
                        self.groove_mask[j, :] = True
                        break

    # ------------------------------------------------------------------
    # H coefficients (formulas 8-11)
    # ------------------------------------------------------------------

    def _compute_H_coefficients(
        self,
        ha: np.ndarray,
        hb: np.ndarray,
        hc: np.ndarray,
        hd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算压力流函数 Hci, Hai, Hco, Hao（公式 8-11）。

        Hci = (ha+hb)³ × Δa / (96 × η × Δc)
        Hai = (ha+hc)³ × Δc / (96 × η × Δa_ai)
        Hco = (hc+hd)³ × Δa / (96 × η × Δc)
        Hao = (hb+hd)³ × Δc / (96 × η × Δa_ao)

        对于 Hai/Hao，压力梯度跨越的轴向距离是相邻网格中心间距
        （非均匀网格下 ≠ 当前网格宽度）：
          Δa_ai[I] = (Δa[I] + Δa[I-1]) / 2   (I>0 时)
          Δa_ao[I] = (Δa[I] + Δa[I+1]) / 2   (I<Na-1 时)
        边界处使用 Δa[I]/2（半网格宽度，后续由 ×2 修正）。

        Returns (Hci, Hai, Hco, Hao) each shape (Mc, Na).
        """
        eta = self.input.viscosity_pa_s
        dc = self.delta_c
        da = self.delta_a  # shape (Na,) — may vary if graded
        Na = self.Na

        coeff = 96.0 * eta

        # da is (Na,), broadcast with (Mc, Na) arrays
        da_row = da[np.newaxis, :]  # (1, Na)

        # 轴向方向压力梯度距离：相邻网格中心间距
        # Hai: 从当前网格中心到内侧(I-1)邻居中心的距离
        # Hao: 从当前网格中心到外侧(I+1)邻居中心的距离
        # 边界处使用 Δa[I]/2（半网格到边界的距离，后续 ×2 修正等效）
        da_ai = np.empty(Na, dtype=np.float64)
        da_ao = np.empty(Na, dtype=np.float64)
        da_ai[0] = da[0] / 2.0  # 边界：中心到端面
        da_ao[Na - 1] = da[Na - 1] / 2.0  # 边界：中心到端面
        for i in range(1, Na):
            da_ai[i] = (da[i] + da[i - 1]) / 2.0
        for i in range(Na - 1):
            da_ao[i] = (da[i] + da[i + 1]) / 2.0

        da_ai_row = da_ai[np.newaxis, :]  # (1, Na)
        da_ao_row = da_ao[np.newaxis, :]  # (1, Na)

        # Hci/Hco: 周向方向，流量面积 ∝ Δa，梯度距离 = Δc
        Hci = (ha + hb) ** 3 * da_row / (coeff * dc)
        Hco = (hc + hd) ** 3 * da_row / (coeff * dc)

        # Hai/Hao: 轴向方向，流量面积 ∝ Δc，梯度距离 = 中心间距
        Hai = (ha + hc) ** 3 * dc / (coeff * da_ai_row)
        Hao = (hb + hd) ** 3 * dc / (coeff * da_ao_row)

        return Hci, Hai, Hco, Hao

    # ------------------------------------------------------------------
    # K flow (formulas 12-16, steady state)
    # ------------------------------------------------------------------

    def _compute_K_flow(
        self,
        ha: np.ndarray,
        hb: np.ndarray,
        hc: np.ndarray,
        hd: np.ndarray,
    ) -> np.ndarray:
        """计算速度诱导流量 K（公式 12-16，稳态）。

        K 定义为 Couette 流的净流入量：
        K = (ha + hb - hc - hd) × U × Δa / 4

        其中 (ha, hb) 为上游边油膜厚度，(hc, hd) 为下游边油膜厚度。
        在收敛楔区域 (ha+hb > hc+hd)，K > 0 表示净流入，驱动压力升高。

        Returns K array shape (Mc, Na).
        """
        da_row = self.delta_a[np.newaxis, :]  # (1, Na)
        K = (ha + hb - hc - hd) * self.U * da_row / 4.0
        return K

    # ------------------------------------------------------------------
    # SOR iteration solver — THE CORE
    # ------------------------------------------------------------------

    def _precompute_effective_H(
        self,
        Hci: np.ndarray,
        Hai: np.ndarray,
        Hco: np.ndarray,
        Hao: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Precompute effective H arrays with boundary/groove doubling.

        Returns (Hci_eff, Hai_eff, Hco_eff, Hao_eff) each (Mc, Na).
        """
        Mc, Na = self.Mc, self.Na
        gm = self.groove_mask

        Hci_e = Hci.copy()
        Hai_e = Hai.copy()
        Hco_e = Hco.copy()
        Hao_e = Hao.copy()

        # Circumferential groove doubling (only for non-groove cells)
        for J in range(Mc):
            if gm[J, :].all():
                continue  # skip rows that are entirely groove
            J_prev = (J - 1) % Mc
            J_next = (J + 1) % Mc
            for I in range(Na):
                if gm[J, I]:
                    continue  # skip groove cells themselves
                if gm[J_prev, I]:
                    Hci_e[J, I] *= 2.0
                if gm[J_next, I]:
                    Hco_e[J, I] *= 2.0

        # Axial boundary / groove doubling (only for non-groove cells)
        # I == 0: Hai doubles (boundary half-grid)
        for J in range(Mc):
            if not gm[J, 0]:
                Hai_e[J, 0] *= 2.0
        # I == Na-1: Hao doubles (boundary half-grid)
        for J in range(Mc):
            if not gm[J, Na - 1]:
                Hao_e[J, Na - 1] *= 2.0
        # Interior groove neighbors
        for J in range(Mc):
            for I in range(1, Na):
                if gm[J, I]:
                    continue  # skip groove cells
                if gm[J, I - 1]:
                    Hai_e[J, I] *= 2.0
            for I in range(Na - 1):
                if gm[J, I]:
                    continue  # skip groove cells
                if gm[J, I + 1]:
                    Hao_e[J, I] *= 2.0

        return Hci_e, Hai_e, Hco_e, Hao_e

    def solve_perturbed(
        self,
        ecy: float,
        ecx: float,
        delta_ecy: float = 0.0,
        delta_ecx: float = 0.0,
        delta_ecy_dot: float = 0.0,
        delta_ecx_dot: float = 0.0,
    ) -> np.ndarray:
        """求解扰动后的压力场，用于动态系数计算。

        Parameters
        ----------
        ecy, ecx : float
            基准偏心分量 (m)。
        delta_ecy, delta_ecx : float
            位移扰动 (m)。
        delta_ecy_dot, delta_ecx_dot : float
            速度扰动 (m/s)。

        Returns
        -------
        np.ndarray
            压力场 (Mc, Na)。
        """
        inp = self.input
        Mc = self.Mc
        Na = self.Na

        # Perturbed eccentricity
        ecy_pert = ecy + delta_ecy
        ecx_pert = ecx + delta_ecx

        gamma = inp.misalignment_vertical_rad
        lam = inp.misalignment_horizontal_rad

        # Compute geometry with perturbed eccentricity
        _esy, _esx, e_edges, psi_edges = compute_eccentricity_components(
            ecy_pert, ecx_pert, gamma, lam, self.s_edges
        )
        ha, hb, hc, hd = compute_element_corner_thicknesses(
            inp.clearance_m, e_edges, psi_edges,
            self.theta_edges, self.s_edges,
        )

        # H coefficients and K flow
        Hci, Hai, Hco, Hao = self._compute_H_coefficients(ha, hb, hc, hd)
        K = self._compute_K_flow(ha, hb, hc, hd)

        # Add squeeze film contribution for velocity perturbations
        if abs(delta_ecy_dot) > 0.0 or abs(delta_ecx_dot) > 0.0:
            # Squeeze film flow: ∂h/∂t = ∂h/∂ey × ėy + ∂h/∂ex × ėx
            # For h = Cd/2 + e·cos(θ - ψ), the partial derivatives w.r.t.
            # the eccentricity components are:
            #   ∂h/∂ey = cos(θ - ψ) · ∂e/∂ey + e·sin(θ-ψ)·∂ψ/∂ey
            # For aligned bearing (γ=λ=0): e=sqrt(ey²+ex²), ψ=atan2(ex,ey)
            #   ∂h/∂ey = cos(θ-ψ)·cos(ψ) + sin(θ-ψ)·sin(ψ) = cos(θ)
            #   ∂h/∂ex = cos(θ-ψ)·sin(ψ) - sin(θ-ψ)·cos(ψ) = sin(θ) ... wait
            # Actually more directly: h = Cd/2 + ey·cos(θ) + ex·sin(θ) for aligned
            # So ∂h/∂ey = cos(θ), ∂h/∂ex = sin(θ)
            # For misaligned: h = Cd/2 + esy(s)·cos(θ) + esx(s)·sin(θ)
            # where esy = ecy + s·γ, esx = ecx + s·λ
            # ∂h/∂ecy = cos(θ), ∂h/∂ecx = sin(θ)
            #
            # The squeeze film term modifies K:
            # K_squeeze = ∂h/∂t × Δa × Δc / 2
            # where ∂h/∂t at each cell center ≈ cos(θ_J)·ėy + sin(θ_J)·ėx
            # The factor comes from integrating the squeeze velocity over the element area.
            # In the Jakeman formulation, the squeeze flow contribution to the
            # continuity equation for each element is:
            # Q_squeeze = -(∂h/∂t) × Δa × Δc
            # This gets added to K (which represents net flow into the element).
            da_row = self.delta_a[np.newaxis, :]  # (1, Na)
            cos_theta = np.cos(self.theta_centers)[:, np.newaxis]  # (Mc, 1)
            sin_theta = np.sin(self.theta_centers)[:, np.newaxis]  # (Mc, 1)

            dh_dt = cos_theta * delta_ecy_dot + sin_theta * delta_ecx_dot
            K_squeeze = dh_dt * da_row * self.delta_c
            K = K + K_squeeze

        # Precompute effective H with boundary/groove doubling
        Hci_e, Hai_e, Hco_e, Hao_e = self._precompute_effective_H(
            Hci, Hai, Hco, Hao
        )
        H_sum = Hci_e + Hai_e + Hco_e + Hao_e

        # Initialize pressure field
        Pc = inp.cavitation_pressure_pa
        P_ambient = inp.ambient_pressure_pa
        Ps = self.supply_pressure

        P = np.full((Mc, Na), P_ambient, dtype=np.float64)
        Qvo = np.zeros((Mc, Na), dtype=np.float64)
        P[self.groove_mask] = Ps

        # SOR iteration (same as solve but returns only pressure)
        ORF = inp.over_relaxation_factor
        max_iter = inp.max_iterations
        tol = inp.convergence_tol
        gm = self.groove_mask

        P_flat = P.ravel()
        Qvo_flat = Qvo.ravel()
        gm_flat = gm.ravel()
        Hci_flat = Hci_e.ravel()
        Hai_flat = Hai_e.ravel()
        Hco_flat = Hco_e.ravel()
        Hao_flat = Hao_e.ravel()
        Hsum_flat = H_sum.ravel()
        K_flat = K.ravel()

        for iteration in range(1, max_iter + 1):
            max_change = 0.0

            for J in range(Mc):
                J_prev = (J - 1) % Mc
                J_next = (J + 1) % Mc
                base = J * Na
                base_prev = J_prev * Na
                base_next = J_next * Na

                for I in range(Na):
                    idx = base + I
                    if gm_flat[idx]:
                        continue

                    Pci = P_flat[base_prev + I]
                    Pco = P_flat[base_next + I]
                    Pai = P_ambient if I == 0 else P_flat[idx - 1]
                    Pao = P_ambient if I == Na - 1 else P_flat[idx + 1]

                    hci = Hci_flat[idx]
                    hai = Hai_flat[idx]
                    hco = Hco_flat[idx]
                    hao = Hao_flat[idx]
                    hs = Hsum_flat[idx]

                    Qvi = Qvo_flat[base_prev + I]

                    P_old = P_flat[idx]
                    k_val = K_flat[idx]

                    weighted_P = hci * Pci + hai * Pai + hco * Pco + hao * Pao
                    P_new = (k_val + Qvi + weighted_P) / hs

                    if P_new <= Pc:
                        P_flat[idx] = Pc
                        Qvo_flat[idx] = Pc * hs - k_val - Qvi - weighted_P
                    else:
                        Qvo_flat[idx] = 0.0
                        P_flat[idx] = P_old + ORF * (P_new - P_old)

                    change = abs(P_flat[idx] - P_old)
                    if change > max_change:
                        max_change = change

            # 相对收敛判据
            P_max = 0.0
            for k in range(Mc * Na):
                if not gm_flat[k]:
                    pv = abs(P_flat[k])
                    if pv > P_max:
                        P_max = pv
            if P_max > 0.0 and max_change / P_max < tol:
                break
            elif P_max == 0.0 and max_change < tol:
                break

        return P_flat.reshape(Mc, Na)

    def solve(
        self,
        ecy: float | None = None,
        ecx: float | None = None,
    ) -> BearingOutput:
        """执行压力场迭代求解。

        Parameters
        ----------
        ecy, ecx : float | None
            中心偏心分量 (m)。若为 None，则从 eccentricity_ratio 和
            attitude_angle_rad 计算。

        Returns
        -------
        BearingOutput
        """
        inp = self.input
        Mc = self.Mc
        Na = self.Na

        # ----------------------------------------------------------
        # 1. Determine eccentricity components
        # ----------------------------------------------------------
        if ecy is None or ecx is None:
            epsilon = inp.eccentricity_ratio
            if epsilon is None:
                raise ValueError(
                    "必须提供 ecy/ecx 或设置 eccentricity_ratio"
                )
            radial_clearance = inp.clearance_m / 2.0
            attitude = inp.attitude_angle_rad
            ecy = epsilon * radial_clearance * math.cos(attitude)
            ecx = epsilon * radial_clearance * math.sin(attitude)

        # ----------------------------------------------------------
        # 2. Compute geometry
        # ----------------------------------------------------------
        gamma = inp.misalignment_vertical_rad
        lam = inp.misalignment_horizontal_rad

        # Eccentricity at s_edges (Na+1 values)
        _esy, _esx, e_edges, psi_edges = compute_eccentricity_components(
            ecy, ecx, gamma, lam, self.s_edges
        )

        # Corner thicknesses ha, hb, hc, hd — each (Mc, Na)
        ha, hb, hc, hd = compute_element_corner_thicknesses(
            inp.clearance_m, e_edges, psi_edges,
            self.theta_edges, self.s_edges,
        )

        # 油膜厚度负值检测（提前报错，避免 H 系数计算异常）
        min_corner_h = min(float(np.min(ha)), float(np.min(hb)),
                          float(np.min(hc)), float(np.min(hd)))
        if min_corner_h <= 0:
            raise RuntimeError(
                f"油膜厚度出现非正值 (min={min_corner_h:.6e} m)。"
                f"请检查偏心比是否过大或不对中参数是否合理。"
            )

        # ----------------------------------------------------------
        # 3. Compute H coefficients and K flow
        # ----------------------------------------------------------
        Hci, Hai, Hco, Hao = self._compute_H_coefficients(ha, hb, hc, hd)
        K = self._compute_K_flow(ha, hb, hc, hd)

        # Precompute effective H with boundary/groove doubling
        Hci_e, Hai_e, Hco_e, Hao_e = self._precompute_effective_H(
            Hci, Hai, Hco, Hao
        )
        H_sum = Hci_e + Hai_e + Hco_e + Hao_e

        # ----------------------------------------------------------
        # 4. Initialize pressure field and Qvo
        # ----------------------------------------------------------
        Pc = inp.cavitation_pressure_pa
        P_ambient = inp.ambient_pressure_pa
        Ps = self.supply_pressure

        P = np.full((Mc, Na), P_ambient, dtype=np.float64)
        Qvo = np.zeros((Mc, Na), dtype=np.float64)
        cavitation = np.zeros((Mc, Na), dtype=bool)

        # Set groove cells to supply pressure
        P[self.groove_mask] = Ps

        # ----------------------------------------------------------
        # 5. SOR iteration (optimised with flat arrays)
        # ----------------------------------------------------------
        ORF = inp.over_relaxation_factor
        max_iter = inp.max_iterations
        tol = inp.convergence_tol
        gm = self.groove_mask

        # Flatten to 1-D for faster indexing in tight loop
        P_flat = P.ravel()
        Qvo_flat = Qvo.ravel()
        cav_flat = cavitation.ravel()
        gm_flat = gm.ravel()
        Hci_flat = Hci_e.ravel()
        Hai_flat = Hai_e.ravel()
        Hco_flat = Hco_e.ravel()
        Hao_flat = Hao_e.ravel()
        Hsum_flat = H_sum.ravel()
        K_flat = K.ravel()

        converged = False
        final_residual = 0.0
        iteration = 0
        prev_residual = float('inf')
        stall_count = 0

        for iteration in range(1, max_iter + 1):
            max_change = 0.0

            for J in range(Mc):
                J_prev = (J - 1) % Mc
                J_next = (J + 1) % Mc
                base = J * Na
                base_prev = J_prev * Na
                base_next = J_next * Na

                for I in range(Na):
                    idx = base + I
                    if gm_flat[idx]:
                        continue

                    # Neighbor pressures
                    Pci = P_flat[base_prev + I]
                    Pco = P_flat[base_next + I]
                    Pai = P_ambient if I == 0 else P_flat[idx - 1]
                    Pao = P_ambient if I == Na - 1 else P_flat[idx + 1]

                    # Effective H for this cell (precomputed)
                    hci = Hci_flat[idx]
                    hai = Hai_flat[idx]
                    hco = Hco_flat[idx]
                    hao = Hao_flat[idx]
                    hs = Hsum_flat[idx]

                    # Upstream cavitation flow
                    Qvi = Qvo_flat[base_prev + I]

                    P_old = P_flat[idx]
                    k_val = K_flat[idx]

                    weighted_P = (
                        hci * Pci + hai * Pai + hco * Pco + hao * Pao
                    )
                    P_new = (k_val + Qvi + weighted_P) / hs

                    if P_new <= Pc:
                        P_flat[idx] = Pc
                        cav_flat[idx] = True
                        Qvo_flat[idx] = Pc * hs - k_val - Qvi - weighted_P
                    else:
                        cav_flat[idx] = False
                        Qvo_flat[idx] = 0.0
                        P_flat[idx] = P_old + ORF * (P_new - P_old)

                    change = abs(P_flat[idx] - P_old)
                    if change > max_change:
                        max_change = change

            # 相对收敛判据
            P_max = 0.0
            for k in range(Mc * Na):
                if not gm_flat[k]:
                    pv = abs(P_flat[k])
                    if pv > P_max:
                        P_max = pv
            if P_max > 0.0:
                relative_residual = max_change / P_max
            else:
                relative_residual = max_change

            final_residual = relative_residual

            if relative_residual < tol:
                converged = True
                break

            # 自适应松弛因子：检测收敛停滞时自动降低 ORF
            if iteration > 100 and relative_residual > prev_residual * 0.95:
                stall_count += 1
            else:
                stall_count = 0
            prev_residual = min(prev_residual, relative_residual)

            if stall_count >= 20 and ORF > 1.05:
                ORF = max(1.0, ORF * 0.7)
                stall_count = 0
                prev_residual = relative_residual

        # Reshape back
        P = P_flat.reshape(Mc, Na)
        cavitation = cav_flat.reshape(Mc, Na)
        Qvo = Qvo_flat.reshape(Mc, Na)

        # ----------------------------------------------------------
        # NaN / Inf detection
        # ----------------------------------------------------------
        if not np.all(np.isfinite(P)):
            raise RuntimeError(
                "求解器检测到 NaN 或 Inf 压力值。请检查输入参数或减小松弛因子。"
            )

        if not converged:
            warnings.warn(
                f"求解器未收敛: 迭代 {iteration} 次后残差 = {final_residual:.4e} "
                f"(容差 = {tol})",
                RuntimeWarning,
                stacklevel=2,
            )

        # ----------------------------------------------------------
        # 6. Compute film thickness field at cell centres
        # ----------------------------------------------------------
        _esy_c, _esx_c, e_centers, psi_centers = compute_eccentricity_components(
            ecy, ecx, gamma, lam, self.s_centers
        )
        film_thickness = compute_film_thickness(
            inp.clearance_m, e_centers, psi_centers, self.theta_centers
        )

        # Min film thickness
        min_h = float(np.min(film_thickness))
        min_idx = np.unravel_index(np.argmin(film_thickness), film_thickness.shape)
        min_theta_deg = float(np.degrees(self.theta_centers[min_idx[0]]))
        min_s = float(self.s_centers[min_idx[1]])

        # ----------------------------------------------------------
        # 7. Build BearingOutput (post-processing fields as zeros)
        # ----------------------------------------------------------
        output = BearingOutput(
            pressure_field_pa=P,
            cavitation_matrix=cavitation,
            film_thickness_field_m=film_thickness,
            load_capacity_n=0.0,
            load_vertical_n=0.0,
            load_horizontal_n=0.0,
            attitude_angle_deg=0.0,
            moment_vertical_nm=0.0,
            moment_horizontal_nm=0.0,
            min_film_thickness_m=min_h,
            min_film_location=(min_theta_deg, min_s),
            side_leakage_flow_m3s=0.0,
            power_loss_w=0.0,
            friction_force_n=0.0,
            stiffness_coefficients=np.zeros((2, 2), dtype=np.float64),
            damping_coefficients=np.zeros((2, 2), dtype=np.float64),
            iterations=iteration,
            converged=converged,
            residual=final_residual,
            input_params=inp,
        )

        return output

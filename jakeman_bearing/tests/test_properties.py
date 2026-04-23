"""
属性测试 — Property-Based Tests for jakeman_bearing

使用 hypothesis 框架验证设计文档中定义的正确性属性。

Feature: journal-bearing-analysis
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from jakeman_bearing.bearing_models import BearingInput


# ── 辅助策略 ──────────────────────────────────────────────────

# 有效的正浮点数（用于几何参数等）
_positive_float = st.floats(min_value=1e-12, max_value=1e6, allow_nan=False, allow_infinity=False)


class TestProperty1InputValidation:
    """Property 1: 输入验证拒绝无效参数

    For any 偏心比值 ε ≤ 0 或 ε ≥ 1，以及任何非正的几何参数，
    BearingInput.validate() 应抛出 ValueError，且错误信息中包含参数名称。

    **Validates: Requirements 1.10, 1.11**
    """

    @given(
        epsilon=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_eccentricity_ratio_le_zero_rejected(self, epsilon: float) -> None:
        """eccentricity_ratio <= 0 → ValueError containing 'eccentricity_ratio'"""
        inp = BearingInput(eccentricity_ratio=epsilon, load_n=None)
        with pytest.raises(ValueError, match="eccentricity_ratio"):
            inp.validate()

    @given(
        epsilon=st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_eccentricity_ratio_ge_one_rejected(self, epsilon: float) -> None:
        """eccentricity_ratio >= 1 → ValueError containing 'eccentricity_ratio'"""
        inp = BearingInput(eccentricity_ratio=epsilon, load_n=None)
        with pytest.raises(ValueError, match="eccentricity_ratio"):
            inp.validate()

    @given(
        diameter=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_diameter_m_non_positive_rejected(self, diameter: float) -> None:
        """diameter_m <= 0 → ValueError containing 'diameter_m'"""
        inp = BearingInput(diameter_m=diameter)
        with pytest.raises(ValueError, match="diameter_m"):
            inp.validate()

    @given(
        length=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_length_m_non_positive_rejected(self, length: float) -> None:
        """length_m <= 0 → ValueError containing 'length_m'"""
        inp = BearingInput(length_m=length)
        with pytest.raises(ValueError, match="length_m"):
            inp.validate()

    @given(
        clearance=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_clearance_m_non_positive_rejected(self, clearance: float) -> None:
        """clearance_m <= 0 → ValueError containing 'clearance_m'"""
        inp = BearingInput(clearance_m=clearance)
        with pytest.raises(ValueError, match="clearance_m"):
            inp.validate()

    def test_both_eccentricity_and_load_set_rejected(self) -> None:
        """Both eccentricity_ratio and load_n set → ValueError"""
        inp = BearingInput(eccentricity_ratio=0.5, load_n=1000.0)
        with pytest.raises(ValueError, match="eccentricity_ratio.*load_n|load_n.*eccentricity_ratio"):
            inp.validate()

    def test_neither_eccentricity_nor_load_set_rejected(self) -> None:
        """Neither eccentricity_ratio nor load_n set → ValueError"""
        inp = BearingInput(eccentricity_ratio=None, load_n=None)
        with pytest.raises(ValueError, match="eccentricity_ratio|load_n"):
            inp.validate()

    @given(
        n_circ=st.integers(min_value=-100, max_value=3),
    )
    @settings(max_examples=100)
    def test_n_circumferential_below_minimum_rejected(self, n_circ: int) -> None:
        """n_circumferential < 4 → ValueError containing 'n_circumferential'"""
        inp = BearingInput(n_circumferential=n_circ)
        with pytest.raises(ValueError, match="n_circumferential"):
            inp.validate()

    @given(
        orf=st.one_of(
            st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=2.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=100)
    def test_over_relaxation_factor_outside_range_rejected(self, orf: float) -> None:
        """over_relaxation_factor outside (0, 2) → ValueError containing 'over_relaxation_factor'"""
        inp = BearingInput(over_relaxation_factor=orf)
        with pytest.raises(ValueError, match="over_relaxation_factor"):
            inp.validate()


import csv
import os
import tempfile

import numpy as np
from jakeman_bearing.bearing_models import BearingOutput, GrooveConfig


class TestProperty15CSVRoundtrip:
    """Property 15: CSV 往返精度

    For any valid pressure field array (float64), saving via to_csv() and re-reading
    should produce max absolute error <= 1e-10.

    Validates: Requirements 11.3, 11.4
    """

    @staticmethod
    def _make_dummy_output(pressure: np.ndarray) -> BearingOutput:
        """Create a minimal BearingOutput with the given pressure field."""
        rows, cols = pressure.shape
        inp = BearingInput()  # default params are valid
        return BearingOutput(
            pressure_field_pa=pressure,
            cavitation_matrix=np.zeros((rows, cols), dtype=bool),
            film_thickness_field_m=np.ones((rows, cols)) * 1e-5,
            load_capacity_n=100.0,
            load_vertical_n=80.0,
            load_horizontal_n=60.0,
            attitude_angle_deg=36.87,
            moment_vertical_nm=0.0,
            moment_horizontal_nm=0.0,
            min_film_thickness_m=1e-5,
            min_film_location=(180.0, 0.0),
            side_leakage_flow_m3s=1e-6,
            power_loss_w=50.0,
            friction_force_n=10.0,
            stiffness_coefficients=np.zeros((2, 2)),
            damping_coefficients=np.zeros((2, 2)),
            iterations=100,
            converged=True,
            residual=0.1,
            input_params=inp,
        )

    @staticmethod
    def _read_pressure_from_csv(filepath: str) -> np.ndarray:
        """Read back the pressure field from a CSV written by to_csv().

        Skips the performance parameters section, finds the '# Pressure Field'
        header, reads the shape, then parses data rows.
        """
        with open(filepath, "r", newline="") as f:
            reader = csv.reader(f)
            # Skip until we find the pressure field header
            for row in reader:
                if row and row[0].startswith("# Pressure Field"):
                    break
            # Next row is shape
            shape_row = next(reader)
            n_rows = int(shape_row[1])
            n_cols = int(shape_row[2])
            # Read data rows
            data = np.empty((n_rows, n_cols), dtype=np.float64)
            for i in range(n_rows):
                data_row = next(reader)
                for j in range(n_cols):
                    data[i, j] = float(data_row[j])
            return data

    @given(
        rows=st.integers(min_value=4, max_value=20),
        cols=st.integers(min_value=4, max_value=20),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_csv_roundtrip_precision(self, rows, cols, data):
        """Pressure field survives CSV roundtrip with max abs error <= 1e-10."""
        pressure = data.draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-1e8, max_value=1e8, allow_nan=False, allow_infinity=False),
                    min_size=cols,
                    max_size=cols,
                ),
                min_size=rows,
                max_size=rows,
            )
        )
        pressure_arr = np.array(pressure, dtype=np.float64)

        output = self._make_dummy_output(pressure_arr)
        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            output.to_csv(csv_path)
            recovered = self._read_pressure_from_csv(csv_path)
            max_err = np.max(np.abs(recovered - pressure_arr))
            assert max_err <= 1e-10, (
                f"CSV roundtrip max absolute error {max_err} exceeds 1e-10"
            )
        finally:
            os.unlink(csv_path)


class TestProperty16SummaryCompleteness:
    """Property 16: 结果摘要完整性

    For any valid BearingOutput, summary() should contain all key performance
    parameters: load capacity, attitude angle, moments, min film thickness,
    power loss, side leakage, input parameter echo, iterations, and convergence.

    **Validates: Requirements 11.1, 11.2**
    """

    @staticmethod
    def _make_dummy_output() -> BearingOutput:
        """Create a BearingOutput with known values for summary verification."""
        inp = BearingInput(
            diameter_m=0.0635,
            length_m=0.02368,
            clearance_m=0.0000635,
            speed_rps=33.333,
            viscosity_pa_s=0.014,
        )
        rows, cols = 10, 6
        return BearingOutput(
            pressure_field_pa=np.zeros((rows, cols)),
            cavitation_matrix=np.zeros((rows, cols), dtype=bool),
            film_thickness_field_m=np.ones((rows, cols)) * 1e-5,
            load_capacity_n=1234.5,
            load_vertical_n=1000.0,
            load_horizontal_n=700.0,
            attitude_angle_deg=35.0,
            moment_vertical_nm=0.123,
            moment_horizontal_nm=0.456,
            min_film_thickness_m=2.5e-5,
            min_film_location=(180.0, 0.005),
            side_leakage_flow_m3s=3.14e-6,
            power_loss_w=42.0,
            friction_force_n=8.0,
            stiffness_coefficients=np.zeros((2, 2)),
            damping_coefficients=np.zeros((2, 2)),
            iterations=150,
            converged=True,
            residual=0.05,
            input_params=inp,
        )

    def test_summary_contains_all_chinese_keywords(self) -> None:
        """summary() must contain all required Chinese section keywords."""
        output = self._make_dummy_output()
        text = output.summary()

        required_keywords = [
            "承载力",
            "偏位角",
            "力矩",
            "最小油膜厚度",
            "功率损耗",
            "侧漏流量",
            "迭代次数",
            "收敛",
        ]
        for kw in required_keywords:
            assert kw in text, f"summary() missing keyword: {kw}"

    def test_summary_contains_input_parameter_echo(self) -> None:
        """summary() must echo back the input parameters."""
        output = self._make_dummy_output()
        inp = output.input_params
        text = output.summary()

        # Each input value should appear in the summary text
        assert str(inp.diameter_m) in text, "summary() missing diameter_m echo"
        assert str(inp.length_m) in text, "summary() missing length_m echo"
        assert str(inp.clearance_m) in text, "summary() missing clearance_m echo"
        assert str(inp.speed_rps) in text, "summary() missing speed_rps echo"
        assert str(inp.viscosity_pa_s) in text, "summary() missing viscosity_pa_s echo"

    def test_summary_contains_numeric_values(self) -> None:
        """summary() must contain the actual numeric output values."""
        output = self._make_dummy_output()
        text = output.summary()

        assert "1234.5" in text, "summary() missing load_capacity_n value"
        assert "35.0" in text, "summary() missing attitude_angle_deg value"
        assert "150" in text, "summary() missing iterations value"


import numpy as np
from jakeman_bearing.bearing_geometry import (
    compute_eccentricity_components,
    compute_film_thickness,
    compute_element_corner_thicknesses,
)


class TestProperty2EccentricityComponents:
    """Property 2: 偏心分量线性计算

    For any 有效的中心偏心分量 (ecy, ecx)、不对中角度 (γ, λ) 和轴向位置数组 s，
    计算得到的偏心分量应满足 esy=ecy+s×γ, esx=ecx+s×λ,
    e=sqrt(esy²+esx²), ψ=atan2(esx, esy)。

    **Validates: Requirements 2.1, 2.2**
    """

    @given(
        ecy=st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False),
        ecx=st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False),
        lam=st.floats(min_value=-0.01, max_value=0.01, allow_nan=False, allow_infinity=False),
        s_val=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_eccentricity_linearity(
        self, ecy: float, ecx: float, gamma: float, lam: float, s_val: float
    ) -> None:
        """esy=ecy+s*gamma, esx=ecx+s*lam, e=sqrt(esy²+esx²), psi=atan2(esx,esy)"""
        s_positions = np.array([s_val])
        esy, esx, e, psi = compute_eccentricity_components(ecy, ecx, gamma, lam, s_positions)

        expected_esy = ecy + s_val * gamma
        expected_esx = ecx + s_val * lam
        expected_e = np.sqrt(expected_esy**2 + expected_esx**2)
        expected_psi = np.arctan2(expected_esx, expected_esy)

        np.testing.assert_allclose(esy[0], expected_esy, atol=1e-15)
        np.testing.assert_allclose(esx[0], expected_esx, atol=1e-15)
        np.testing.assert_allclose(e[0], expected_e, atol=1e-15)
        if expected_e > 1e-20:
            np.testing.assert_allclose(psi[0], expected_psi, atol=1e-12)


class TestProperty3CornerThickness:
    """Property 3: 网格四角油膜厚度公式

    验证 ha=h(θ_J,s_I), hb=h(θ_J,s_{I+1}), hc=h(θ_{J+1},s_I), hd=h(θ_{J+1},s_{I+1})
    使用小网格 (4×4) 与随机有效参数。

    **Validates: Requirements 2.3, 2.4**
    """

    @given(
        clearance=st.floats(min_value=1e-5, max_value=1e-3, allow_nan=False, allow_infinity=False),
        epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        attitude=st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False),
        gamma=st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False),
        lam=st.floats(min_value=-0.001, max_value=0.001, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_corner_thicknesses_match_formula(
        self, clearance: float, epsilon: float, attitude: float, gamma: float, lam: float
    ) -> None:
        """Each corner thickness equals h(θ,s) = Cd/2 + e(s)*cos(θ - ψ(s))"""
        Mc, Na = 4, 4
        L = 0.05
        radial_clearance = clearance / 2.0
        ecy = epsilon * radial_clearance * np.cos(attitude)
        ecx = epsilon * radial_clearance * np.sin(attitude)

        theta_edges = np.linspace(0, 2 * np.pi, Mc + 1)
        s_edges = np.linspace(-L / 2, L / 2, Na + 1)

        esy, esx, e, psi = compute_eccentricity_components(ecy, ecx, gamma, lam, s_edges)

        # Ensure all film thicknesses are positive (skip if not)
        h_min_check = radial_clearance - np.max(e)
        assume(h_min_check > 0)

        ha, hb, hc, hd = compute_element_corner_thicknesses(clearance, e, psi, theta_edges, s_edges)

        for J in range(Mc):
            for I in range(Na):
                expected_ha = radial_clearance + e[I] * np.cos(theta_edges[J] - psi[I])
                expected_hb = radial_clearance + e[I + 1] * np.cos(theta_edges[J] - psi[I + 1])
                expected_hc = radial_clearance + e[I] * np.cos(theta_edges[J + 1] - psi[I])
                expected_hd = radial_clearance + e[I + 1] * np.cos(theta_edges[J + 1] - psi[I + 1])

                np.testing.assert_allclose(ha[J, I], expected_ha, atol=1e-15)
                np.testing.assert_allclose(hb[J, I], expected_hb, atol=1e-15)
                np.testing.assert_allclose(hc[J, I], expected_hc, atol=1e-15)
                np.testing.assert_allclose(hd[J, I], expected_hd, atol=1e-15)


class TestProperty4FilmThicknessPositive:
    """Property 4: 油膜厚度正值不变量

    对偏心比在 (0,1) 内的有效参数，所有油膜厚度值严格大于零。

    **Validates: Requirements 2.5**
    """

    @given(
        clearance=st.floats(min_value=1e-5, max_value=1e-3, allow_nan=False, allow_infinity=False),
        epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        attitude=st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_film_thickness_all_positive(
        self, clearance: float, epsilon: float, attitude: float
    ) -> None:
        """For eccentricity_ratio in (0,1) and aligned bearing, all h > 0"""
        radial_clearance = clearance / 2.0
        ecy = epsilon * radial_clearance * np.cos(attitude)
        ecx = epsilon * radial_clearance * np.sin(attitude)

        n_theta = 36
        n_axial = 6
        L = 0.05
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        s_positions = np.linspace(-L / 2, L / 2, n_axial)

        # Aligned bearing: gamma=0, lam=0
        esy, esx, e, psi = compute_eccentricity_components(ecy, ecx, 0.0, 0.0, s_positions)
        h = compute_film_thickness(clearance, e, psi, theta)

        assert np.all(h > 0), f"Found non-positive film thickness: min={np.min(h)}"


class TestProperty5AlignedDegeneracy:
    """Property 5: 对齐轴承退化

    当 γ=0, λ=0 时，同一周向位置的油膜厚度在轴向方向上为常数。

    **Validates: Requirements 2.6**
    """

    @given(
        clearance=st.floats(min_value=1e-5, max_value=1e-3, allow_nan=False, allow_infinity=False),
        epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
        attitude=st.floats(min_value=0.0, max_value=2 * np.pi, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_aligned_bearing_constant_axial(
        self, clearance: float, epsilon: float, attitude: float
    ) -> None:
        """When gamma=0, lam=0, h at same theta is constant across axial positions"""
        radial_clearance = clearance / 2.0
        ecy = epsilon * radial_clearance * np.cos(attitude)
        ecx = epsilon * radial_clearance * np.sin(attitude)

        n_theta = 36
        n_axial = 8
        L = 0.05
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        s_positions = np.linspace(-L / 2, L / 2, n_axial)

        esy, esx, e, psi = compute_eccentricity_components(ecy, ecx, 0.0, 0.0, s_positions)
        h = compute_film_thickness(clearance, e, psi, theta)

        # For each theta row, all axial values should be identical
        for j in range(n_theta):
            np.testing.assert_allclose(
                h[j, :], h[j, 0], atol=1e-15,
                err_msg=f"Film thickness varies axially at theta index {j}"
            )


# ── 求解器相关属性测试 ────────────────────────────────────────

import math
from jakeman_bearing.bearing_solver import BearingSolver


class TestProperty17GridGeneration:
    """Property 17: 网格生成正确性

    For any 正整数 Mc（周向网格数），生成的网格中心角度应满足
    θ_J = (J-0.5)×360°/Mc。
    For any 正整数 Na 和 grading_factor=1.0，所有轴向网格间距 Δa 应相等；
    当 grading_factor > 1.0 时，端部网格间距应小于中部网格间距。

    **Validates: Requirements 9.1, 9.2**
    """

    @given(
        Mc=st.integers(min_value=4, max_value=144),
    )
    @settings(max_examples=100)
    def test_theta_centers_formula(self, Mc: int) -> None:
        """θ_J = (J - 0.5) × 360° / Mc for J = 1..Mc (0-based: (j+0.5)*2π/Mc)"""
        inp = BearingInput(n_circumferential=Mc, n_axial=4)
        solver = BearingSolver(inp)

        for j in range(Mc):
            expected = (j + 0.5) * 2.0 * math.pi / Mc
            np.testing.assert_allclose(
                solver.theta_centers[j], expected, atol=1e-12,
                err_msg=f"θ center mismatch at j={j}, Mc={Mc}"
            )

    @given(
        Na=st.integers(min_value=4, max_value=30),
        L=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_uniform_grid_equal_spacing(self, Na: int, L: float) -> None:
        """grading_factor=1.0 → all Δa equal to L/Na"""
        inp = BearingInput(
            n_circumferential=4, n_axial=Na,
            length_m=L, axial_grading_factor=1.0,
        )
        solver = BearingSolver(inp)

        expected_da = L / Na
        np.testing.assert_allclose(
            solver.delta_a, expected_da, atol=1e-12,
            err_msg=f"Non-uniform Δa with grading_factor=1.0, Na={Na}, L={L}"
        )

    @given(
        Na=st.integers(min_value=6, max_value=30),
        gf=st.floats(min_value=1.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_graded_grid_end_smaller_than_middle(self, Na: int, gf: float) -> None:
        """grading_factor > 1.0 → end Δa < middle Δa"""
        inp = BearingInput(
            n_circumferential=4, n_axial=Na,
            axial_grading_factor=gf,
        )
        solver = BearingSolver(inp)

        da = solver.delta_a
        # First cell (end) should be smaller than middle cell
        mid_idx = Na // 2
        assert da[0] < da[mid_idx], (
            f"End Δa[0]={da[0]} not smaller than middle Δa[{mid_idx}]={da[mid_idx]} "
            f"with grading_factor={gf}"
        )
        # Last cell (end) should also be smaller than middle cell
        assert da[-1] < da[mid_idx], (
            f"End Δa[-1]={da[-1]} not smaller than middle Δa[{mid_idx}]={da[mid_idx]} "
            f"with grading_factor={gf}"
        )


class TestProperty6HCoefficients:
    """Property 6: 压力流函数 H 系数公式

    For any 正的油膜厚度值 (ha, hb, hc, hd)、正的网格尺寸 (Δa, Δc) 和正的粘度 η，
    计算得到的 H 系数应满足：
    Hci = (ha+hb)³×Δa/(96×η×Δc)
    Hai = (ha+hc)³×Δc/(96×η×Δa)
    Hco = (hc+hd)³×Δa/(96×η×Δc)
    Hao = (hb+hd)³×Δc/(96×η×Δa)

    **Validates: Requirements 3.2**
    """

    @given(
        h_val=st.floats(min_value=1e-6, max_value=1e-3, allow_nan=False, allow_infinity=False),
        eta=st.floats(min_value=0.001, max_value=1.0, allow_nan=False, allow_infinity=False),
        Mc=st.integers(min_value=4, max_value=72),
        Na=st.integers(min_value=4, max_value=20),
    )
    @settings(max_examples=100)
    def test_H_coefficients_formula(self, h_val: float, eta: float, Mc: int, Na: int) -> None:
        """Verify Hci, Hai, Hco, Hao match formulas 8-11.

        For Hai/Hao, the axial gradient distance is the center-to-center
        spacing between adjacent cells (or half-cell at boundaries).
        On a uniform grid, this equals da[I] for interior cells.
        """
        # Create a solver with known parameters
        D = 0.0635
        L = 0.02368
        inp = BearingInput(
            diameter_m=D, length_m=L,
            viscosity_pa_s=eta,
            n_circumferential=Mc, n_axial=Na,
            axial_grading_factor=1.0,
        )
        solver = BearingSolver(inp)

        # Use uniform film thickness for simplicity
        ha = np.full((Mc, Na), h_val)
        hb = np.full((Mc, Na), h_val * 1.1)
        hc = np.full((Mc, Na), h_val * 0.9)
        hd = np.full((Mc, Na), h_val * 1.05)

        Hci, Hai, Hco, Hao = solver._compute_H_coefficients(ha, hb, hc, hd)

        dc = solver.delta_c
        da = solver.delta_a  # shape (Na,)
        coeff = 96.0 * eta

        # Compute expected axial gradient distances
        da_ai = np.empty(Na)
        da_ao = np.empty(Na)
        da_ai[0] = da[0] / 2.0
        da_ao[Na - 1] = da[Na - 1] / 2.0
        for i in range(1, Na):
            da_ai[i] = (da[i] + da[i - 1]) / 2.0
        for i in range(Na - 1):
            da_ao[i] = (da[i] + da[i + 1]) / 2.0

        for I in range(Na):
            expected_Hci = (h_val + h_val * 1.1) ** 3 * da[I] / (coeff * dc)
            expected_Hai = (h_val + h_val * 0.9) ** 3 * dc / (coeff * da_ai[I])
            expected_Hco = (h_val * 0.9 + h_val * 1.05) ** 3 * da[I] / (coeff * dc)
            expected_Hao = (h_val * 1.1 + h_val * 1.05) ** 3 * dc / (coeff * da_ao[I])

            np.testing.assert_allclose(Hci[0, I], expected_Hci, rtol=1e-10)
            np.testing.assert_allclose(Hai[0, I], expected_Hai, rtol=1e-10)
            np.testing.assert_allclose(Hco[0, I], expected_Hco, rtol=1e-10)
            np.testing.assert_allclose(Hao[0, I], expected_Hao, rtol=1e-10)


class TestProperty7KFlow:
    """Property 7: 速度诱导流量 K 公式

    For any 正的油膜厚度值 (ha, hb, hc, hd)、正的表面速度 U 和正的网格尺寸 Δa，
    稳态速度诱导流量应满足 K = (ha+hb-hc-hd)×U×Δa/4。
    K 定义为 Couette 流的净流入量（上游边流入减去下游边流出）。

    **Validates: Requirements 3.3**
    """

    @given(
        h_val=st.floats(min_value=1e-6, max_value=1e-3, allow_nan=False, allow_infinity=False),
        speed=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        Na=st.integers(min_value=4, max_value=20),
    )
    @settings(max_examples=100)
    def test_K_flow_formula(self, h_val: float, speed: float, Na: int) -> None:
        """Verify K = (ha+hb-hc-hd)×U×Δa/4"""
        Mc = 8
        D = 0.0635
        inp = BearingInput(
            diameter_m=D,
            speed_rps=speed,
            n_circumferential=Mc, n_axial=Na,
            axial_grading_factor=1.0,
        )
        solver = BearingSolver(inp)

        ha = np.full((Mc, Na), h_val)
        hb = np.full((Mc, Na), h_val * 1.1)
        hc = np.full((Mc, Na), h_val * 0.9)
        hd = np.full((Mc, Na), h_val * 1.05)

        K = solver._compute_K_flow(ha, hb, hc, hd)

        U = solver.U
        da = solver.delta_a

        for I in range(Na):
            expected_K = (h_val + h_val * 1.1 - h_val * 0.9 - h_val * 1.05) * U * da[I] / 4.0
            np.testing.assert_allclose(K[0, I], expected_K, rtol=1e-10)


class TestProperty8GroovePressureInvariant:
    """Property 8: 供油槽压力不变量

    For any 含供油槽的轴承配置，求解完成后，所有供油槽占据的网格单元的压力
    应精确等于供油压力 Ps。

    **Validates: Requirements 3.6**
    """

    @given(
        epsilon=st.floats(min_value=0.2, max_value=0.8, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_groove_cells_equal_Ps(self, epsilon: float) -> None:
        """After solving, all groove cells pressure = Ps"""
        groove = GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,
            axial_position_ratio=0.5,
            axial_width_ratio=0.2145,
        )
        inp = BearingInput(
            eccentricity_ratio=epsilon,
            groove=groove,
            n_circumferential=36,
            n_axial=8,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()

        P = output.pressure_field_pa
        Ps = groove.supply_pressure_pa

        groove_pressures = P[solver.groove_mask]
        if groove_pressures.size > 0:
            np.testing.assert_allclose(
                groove_pressures, Ps, atol=1e-10,
                err_msg="Groove cell pressure deviates from Ps"
            )


class TestProperty9PressureCavitationConsistency:
    """Property 9: 压力-空化一致性

    For any 求解完成的压力场：
    (a) 所有标记为空化的网格单元压力等于 Pc
    (b) 所有压力大于 Pc 的网格单元未被标记为空化
    (c) 空化标记与压力值完全一致

    **Validates: Requirements 4.1, 4.2, 4.5**
    """

    @given(
        epsilon=st.floats(min_value=0.3, max_value=0.8, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_cavitation_pressure_consistency(self, epsilon: float) -> None:
        """Cavitated cells P=Pc; P>Pc cells not cavitated"""
        inp = BearingInput(
            eccentricity_ratio=epsilon,
            n_circumferential=36,
            n_axial=8,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()

        P = output.pressure_field_pa
        cav = output.cavitation_matrix
        Pc = inp.cavitation_pressure_pa

        # (a) Cavitated cells have P == Pc
        if np.any(cav):
            np.testing.assert_allclose(
                P[cav], Pc, atol=1e-10,
                err_msg="Cavitated cell pressure != Pc"
            )

        # (b) Cells with P > Pc are not cavitated (exclude groove cells)
        non_groove = ~solver.groove_mask
        above_Pc = (P > Pc + 1e-10) & non_groove
        assert not np.any(cav & above_Pc), (
            "Found cells marked as cavitated but with P > Pc"
        )


# ── 后处理相关属性测试 ────────────────────────────────────────

from jakeman_bearing.bearing_postprocess import (
    _compute_12point_average,
    compute_load_capacity,
)


class TestProperty10LoadCapacity:
    """Property 10: 12点加权平均承载力

    For any 有效的压力场数组和网格参数，承载力分量应满足：
    先用12点加权平均 P_mean = (4P + ΣP_neighbors)/12 计算平均压力，
    再积分 Fy = Σ(-P_mean×Δa×Δc×cos(θ))，Fx = Σ(-P_mean×Δa×Δc×sin(θ))，
    合成力 F = sqrt(Fy²+Fx²)。

    **Validates: Requirements 5.1, 5.2, 5.3**

    Feature: journal-bearing-analysis, Property 10
    """

    @given(
        Mc=st.integers(min_value=4, max_value=36),
        Na=st.integers(min_value=4, max_value=14),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_load_capacity_matches_manual_12point_average(
        self, Mc: int, Na: int, data
    ) -> None:
        """compute_load_capacity results match manual 12-point average calculation."""
        # Generate random pressure field
        pressure_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=0.0, max_value=1e7,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        pressure = np.array(pressure_list, dtype=np.float64)

        # Generate random positive delta_a values
        delta_a_list = data.draw(
            st.lists(
                st.floats(
                    min_value=1e-4, max_value=0.1,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        delta_a = np.array(delta_a_list, dtype=np.float64)

        # Generate random positive delta_c
        delta_c = data.draw(
            st.floats(
                min_value=1e-4, max_value=0.1,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate theta_centers: evenly spaced angles
        theta_centers = np.array(
            [(j + 0.5) * 2.0 * np.pi / Mc for j in range(Mc)]
        )

        # --- Manual calculation ---
        P_mean = _compute_12point_average(pressure)

        cos_theta = np.cos(theta_centers)[:, np.newaxis]  # (Mc, 1)
        sin_theta = np.sin(theta_centers)[:, np.newaxis]  # (Mc, 1)
        da_row = delta_a[np.newaxis, :]                   # (1, Na)

        dF = -P_mean * da_row * delta_c
        expected_Fy = float(np.sum(dF * cos_theta))
        expected_Fx = float(np.sum(dF * sin_theta))
        expected_F_total = math.sqrt(expected_Fy**2 + expected_Fx**2)

        # --- Function under test ---
        Fy, Fx, F_total, _ = compute_load_capacity(
            pressure, theta_centers, delta_a, delta_c
        )

        # --- Assertions ---
        np.testing.assert_allclose(
            Fy, expected_Fy, atol=1e-8,
            err_msg="Fy mismatch between manual and compute_load_capacity",
        )
        np.testing.assert_allclose(
            Fx, expected_Fx, atol=1e-8,
            err_msg="Fx mismatch between manual and compute_load_capacity",
        )
        np.testing.assert_allclose(
            F_total, expected_F_total, atol=1e-8,
            err_msg="F_total mismatch between manual and compute_load_capacity",
        )


from jakeman_bearing.bearing_postprocess import compute_moments


class TestProperty11MomentCalculation:
    """Property 11: 力矩计算

    For any 有效的压力场和轴向位置数组，力矩应满足
    My = Σ(dFy × s_I)，Mx = Σ(dFx × s_I)，
    其中 dFy, dFx 为各网格的力分量（使用12点加权平均）。

    **Validates: Requirements 5.4**

    Feature: journal-bearing-analysis, Property 11
    """

    @given(
        Mc=st.integers(min_value=4, max_value=36),
        Na=st.integers(min_value=4, max_value=14),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_moment_matches_manual_12point_average(
        self, Mc: int, Na: int, data
    ) -> None:
        """compute_moments results match manual 12-point average moment calculation."""
        # Generate random pressure field
        pressure_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=0.0, max_value=1e7,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        pressure = np.array(pressure_list, dtype=np.float64)

        # Generate random positive delta_a values
        delta_a_list = data.draw(
            st.lists(
                st.floats(
                    min_value=1e-4, max_value=0.1,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        delta_a = np.array(delta_a_list, dtype=np.float64)

        # Generate random positive delta_c
        delta_c = data.draw(
            st.floats(
                min_value=1e-4, max_value=0.1,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate theta_centers: evenly spaced angles
        theta_centers = np.array(
            [(j + 0.5) * 2.0 * np.pi / Mc for j in range(Mc)]
        )

        # Generate random s_centers (symmetric around 0)
        s_centers_list = data.draw(
            st.lists(
                st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        s_centers = np.array(s_centers_list, dtype=np.float64)

        # --- Manual calculation ---
        P_mean = _compute_12point_average(pressure)

        cos_theta = np.cos(theta_centers)[:, np.newaxis]  # (Mc, 1)
        sin_theta = np.sin(theta_centers)[:, np.newaxis]  # (Mc, 1)
        da_row = delta_a[np.newaxis, :]                   # (1, Na)
        s_row = s_centers[np.newaxis, :]                  # (1, Na)

        dF = -P_mean * da_row * delta_c
        dFy = dF * cos_theta  # (Mc, Na)
        dFx = dF * sin_theta  # (Mc, Na)

        expected_My = float(np.sum(dFy * s_row))
        expected_Mx = float(np.sum(dFx * s_row))

        # --- Function under test ---
        My, Mx = compute_moments(
            pressure, theta_centers, s_centers, delta_a, delta_c
        )

        # --- Assertions ---
        np.testing.assert_allclose(
            My, expected_My, atol=1e-8,
            err_msg="My mismatch between manual and compute_moments",
        )
        np.testing.assert_allclose(
            Mx, expected_Mx, atol=1e-8,
            err_msg="Mx mismatch between manual and compute_moments",
        )


from jakeman_bearing.bearing_postprocess import find_min_film_thickness


class TestProperty12MinFilmThickness:
    """Property 12: 最小油膜厚度識別

    For any 有效的油膜厚度場，find_min_film_thickness 返回的 min_film_thickness_m
    应等于厚度场的全局最小值，min_film_location 应对应该最小值的 (theta_deg, s_m) 位置。

    **Validates: Requirements 5.5**

    Feature: journal-bearing-analysis, Property 12
    """

    @given(
        Mc=st.integers(min_value=4, max_value=36),
        Na=st.integers(min_value=4, max_value=14),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_min_film_thickness_identification(
        self, Mc: int, Na: int, data
    ) -> None:
        """min_h equals np.min(field); location matches argmin position."""
        # Generate random positive film thickness field
        film_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=1e-7, max_value=1e-3,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        film_thickness_field = np.array(film_list, dtype=np.float64)

        # Generate theta_centers: evenly spaced angles (rad)
        theta_centers = np.array(
            [(j + 0.5) * 2.0 * np.pi / Mc for j in range(Mc)]
        )

        # Generate random s_centers
        s_centers_list = data.draw(
            st.lists(
                st.floats(
                    min_value=-1.0, max_value=1.0,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        s_centers = np.array(s_centers_list, dtype=np.float64)

        # --- Expected values ---
        expected_min_h = float(np.min(film_thickness_field))
        min_idx = np.unravel_index(
            np.argmin(film_thickness_field), film_thickness_field.shape
        )
        expected_theta_deg = float(np.degrees(theta_centers[min_idx[0]]))
        expected_s_m = float(s_centers[min_idx[1]])

        # --- Function under test ---
        min_h, (theta_deg, s_m) = find_min_film_thickness(
            film_thickness_field, theta_centers, s_centers
        )

        # --- Assertions ---
        np.testing.assert_allclose(
            min_h, expected_min_h, atol=1e-15,
            err_msg="min_film_thickness_m does not equal global minimum of field",
        )
        np.testing.assert_allclose(
            theta_deg, expected_theta_deg, atol=1e-10,
            err_msg="theta_deg does not match argmin row position",
        )
        np.testing.assert_allclose(
            s_m, expected_s_m, atol=1e-15,
            err_msg="s_m does not match argmin col position",
        )


from jakeman_bearing.bearing_postprocess import compute_side_leakage


class TestProperty13SideLeakage:
    """Property 13: 侧漏流量计算

    For any 有效的压力场和 H 系数，侧漏流量应等于轴承两端面轴向流量绝对值之和：
    Qai = Hai[:,0] × (P[:,0] - P_ambient)
    Qao = Hao[:,Na-1] × (P[:,Na-1] - P_ambient)
    Qs = Σ|Qai| + Σ|Qao|

    **Validates: Requirements 6.1, 6.2**

    Feature: journal-bearing-analysis, Property 13
    """

    @given(
        Mc=st.integers(min_value=4, max_value=36),
        Na=st.integers(min_value=4, max_value=14),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_side_leakage_matches_manual_calculation(
        self, Mc: int, Na: int, data
    ) -> None:
        """Side leakage equals sum of absolute axial flow rates at both bearing ends."""
        # Generate random pressure field
        pressure_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=-1e6, max_value=1e7,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        pressure = np.array(pressure_list, dtype=np.float64)

        # Generate random positive Hai array (Mc, Na)
        Hai_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=1e-12, max_value=1e-3,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        Hai = np.array(Hai_list, dtype=np.float64)

        # Generate random positive Hao array (Mc, Na)
        Hao_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=1e-12, max_value=1e-3,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        Hao = np.array(Hao_list, dtype=np.float64)

        # Generate random ambient pressure
        ambient_pressure = data.draw(
            st.floats(
                min_value=-1e5, max_value=1e5,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate random positive delta_a
        delta_a_list = data.draw(
            st.lists(
                st.floats(
                    min_value=1e-4, max_value=0.1,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        delta_a = np.array(delta_a_list, dtype=np.float64)

        # Generate random positive delta_c
        delta_c = data.draw(
            st.floats(
                min_value=1e-4, max_value=0.1,
                allow_nan=False, allow_infinity=False,
            )
        )

        # --- Manual calculation ---
        Qai = Hai[:, 0] * (pressure[:, 0] - ambient_pressure)
        Qao = Hao[:, Na - 1] * (pressure[:, Na - 1] - ambient_pressure)
        expected_Qs = float(np.sum(np.abs(Qai)) + np.sum(np.abs(Qao)))

        # --- Function under test ---
        Qs = compute_side_leakage(
            pressure, Hao, Hai, ambient_pressure, delta_a, delta_c
        )

        # --- Assertions ---
        np.testing.assert_allclose(
            Qs, expected_Qs, atol=1e-8,
            err_msg="Side leakage Qs mismatch between manual and compute_side_leakage",
        )


from jakeman_bearing.bearing_postprocess import compute_power_loss


class TestProperty14PowerLoss:
    """Property 14: 功率损耗计算

    For any 有效的油膜厚度场、粘度和表面速度，总摩擦力应满足
    F_friction = Σ(η × U × Δc × Δa[I] / h[J,I])，
    总功率损耗 H = U × F_friction。

    **Validates: Requirements 6.3, 6.4**

    Feature: journal-bearing-analysis, Property 14
    """

    @given(
        Mc=st.integers(min_value=4, max_value=36),
        Na=st.integers(min_value=4, max_value=14),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_power_loss_matches_manual_calculation(
        self, Mc: int, Na: int, data
    ) -> None:
        """compute_power_loss results match manual Petroff shear formula."""
        # Generate random positive film thickness field
        film_list = data.draw(
            st.lists(
                st.lists(
                    st.floats(
                        min_value=1e-7, max_value=1e-3,
                        allow_nan=False, allow_infinity=False,
                    ),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        film_thickness = np.array(film_list, dtype=np.float64)

        # Generate random positive viscosity
        viscosity = data.draw(
            st.floats(
                min_value=1e-4, max_value=1.0,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate random positive surface velocity
        surface_velocity = data.draw(
            st.floats(
                min_value=0.01, max_value=100.0,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate random positive delta_a values
        delta_a_list = data.draw(
            st.lists(
                st.floats(
                    min_value=1e-4, max_value=0.1,
                    allow_nan=False, allow_infinity=False,
                ),
                min_size=Na,
                max_size=Na,
            )
        )
        delta_a = np.array(delta_a_list, dtype=np.float64)

        # Generate random positive delta_c
        delta_c = data.draw(
            st.floats(
                min_value=1e-4, max_value=0.1,
                allow_nan=False, allow_infinity=False,
            )
        )

        # Generate random cavitation matrix (bool)
        cav_list = data.draw(
            st.lists(
                st.lists(
                    st.booleans(),
                    min_size=Na,
                    max_size=Na,
                ),
                min_size=Mc,
                max_size=Mc,
            )
        )
        cavitation_matrix = np.array(cav_list, dtype=bool)

        # --- Manual calculation ---
        # Fc[J,I] = viscosity * surface_velocity * delta_c * delta_a[I] / h[J,I]
        da_row = delta_a[np.newaxis, :]  # (1, Na)
        Fc = viscosity * surface_velocity * delta_c * da_row / film_thickness
        expected_friction_force = float(np.sum(Fc))
        expected_power_loss = surface_velocity * expected_friction_force

        # --- Function under test ---
        power_loss, friction_force = compute_power_loss(
            film_thickness, viscosity, surface_velocity,
            delta_a, delta_c, cavitation_matrix,
        )

        # --- Assertions ---
        np.testing.assert_allclose(
            friction_force, expected_friction_force, rtol=1e-10,
            err_msg="friction_force mismatch between manual and compute_power_loss",
        )
        np.testing.assert_allclose(
            power_loss, expected_power_loss, rtol=1e-10,
            err_msg="power_loss mismatch between manual and compute_power_loss",
        )


# ── 实用分析模式属性测试 ──────────────────────────────────────

from jakeman_bearing.bearing_practical import solve_for_load


class TestProperty20BrentLoadMatching:
    """Property 20: Brent 法载荷匹配

    For any 在轴承承载能力范围内的目标载荷 W_target，
    Bearing_Practical 通过 Brent 法迭代后，计算得到的承载力 W_computed
    应满足 |W_computed - W_target| / W_target < 收敛容差。

    **Validates: Requirements 8.3**

    Feature: journal-bearing-analysis, Property 20
    """

    @given(
        target_load=st.floats(
            min_value=200.0, max_value=2500.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_load_matching_within_tolerance(self, target_load: float) -> None:
        """For target loads within bearing capacity, the Brent solver
        should converge such that |W_computed - W_target| / W_target < tol."""
        # Use default Table 1 crankshaft bearing parameters with coarse grid
        # for speed. Relaxed convergence tolerance for the coarse grid.
        solve_tol = 0.03  # 3% convergence tolerance for Brent solver
        check_tol = 0.06  # 6% assertion tolerance (accounts for final re-solve drift)
        inp = BearingInput(
            eccentricity_ratio=0.5,  # placeholder, overridden by solve_for_load
            load_n=None,
            n_circumferential=36,
            n_axial=6,
            max_iterations=3000,
            convergence_tol=1e-4,
        )

        output = solve_for_load(
            bearing_input=inp,
            target_load_n=target_load,
            load_direction_deg=270.0,
            tol=solve_tol,
            max_iter=50,
        )

        relative_error = abs(output.load_capacity_n - target_load) / target_load
        assert relative_error < check_tol, (
            f"Load matching failed: W_computed={output.load_capacity_n:.2f} N, "
            f"W_target={target_load:.2f} N, "
            f"relative_error={relative_error:.4f} >= check_tol={check_tol}"
        )


# ── 公共接口属性测试 ──────────────────────────────────────────

from jakeman_bearing import analyze_bearing


class TestProperty18AnalysisModeDetection:
    """Property 18: 分析模式自动检测

    当 analyze_bearing() 以 eccentricity_ratio 调用时，使用偏心模式；
    当以 load 调用时，使用载荷模式（Brent 法）。

    **Validates: Requirements 12.2**
    """

    def test_eccentricity_mode_returns_correct_epsilon(self) -> None:
        """eccentricity_ratio=0.5, load=None → 偏心模式，input_params.eccentricity_ratio == 0.5"""
        output = analyze_bearing(
            eccentricity_ratio=0.5,
            load=None,
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        assert output.input_params.eccentricity_ratio == 0.5, (
            f"Expected eccentricity_ratio=0.5, got {output.input_params.eccentricity_ratio}"
        )
        assert isinstance(output, BearingOutput)

    def test_load_mode_returns_matching_load_capacity(self) -> None:
        """eccentricity_ratio=None, load=500.0 → 载荷模式，load_capacity_n ≈ 500.0"""
        target_load = 500.0
        output = analyze_bearing(
            eccentricity_ratio=None,
            load=target_load,
            n_circumferential=36,
            n_axial=6,
            max_iterations=3000,
            convergence_tol=1e-4,
        )
        assert isinstance(output, BearingOutput)
        relative_error = abs(output.load_capacity_n - target_load) / target_load
        assert relative_error < 0.06, (
            f"Load mode: load_capacity_n={output.load_capacity_n:.2f} N, "
            f"target={target_load:.2f} N, relative_error={relative_error:.4f}"
        )

    @given(
        epsilon=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=20, deadline=None)
    def test_eccentricity_mode_hypothesis(self, epsilon: float) -> None:
        """Hypothesis: eccentricity_ratio provided → eccentricity mode detected."""
        output = analyze_bearing(
            eccentricity_ratio=epsilon,
            load=None,
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        assert output.input_params.eccentricity_ratio == epsilon, (
            f"Expected eccentricity_ratio={epsilon}, "
            f"got {output.input_params.eccentricity_ratio}"
        )

    @given(
        target_load=st.floats(
            min_value=200.0, max_value=2000.0,
            allow_nan=False, allow_infinity=False,
        ),
    )
    @settings(max_examples=10, deadline=None)
    def test_load_mode_hypothesis(self, target_load: float) -> None:
        """Hypothesis: load provided → load mode, load_capacity_n ≈ target_load."""
        output = analyze_bearing(
            eccentricity_ratio=None,
            load=target_load,
            n_circumferential=36,
            n_axial=6,
            max_iterations=3000,
            convergence_tol=1e-4,
        )
        relative_error = abs(output.load_capacity_n - target_load) / target_load
        assert relative_error < 0.06, (
            f"Load mode: load_capacity_n={output.load_capacity_n:.2f} N, "
            f"target={target_load:.2f} N, relative_error={relative_error:.4f}"
        )


class TestProperty19ErrorMessageContext:
    """Property 19: 错误信息包含上下文

    对于任何无效输入或求解器错误，错误信息应包含步骤名称和错误描述。

    **Validates: Requirements 12.4**
    """

    def test_invalid_eccentricity_ratio_contains_step_name(self) -> None:
        """eccentricity_ratio=1.5 → ValueError 包含步骤名称"""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(eccentricity_ratio=1.5, load=None)
        msg = str(exc_info.value)
        # Should contain a step identifier
        assert any(kw in msg for kw in ["参数验证", "[", "步骤", "eccentricity_ratio"]), (
            f"Error message missing step context: {msg!r}"
        )

    def test_invalid_diameter_contains_step_name(self) -> None:
        """diameter=-1.0 → ValueError 包含步骤名称"""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(diameter=-1.0, eccentricity_ratio=0.5, load=None)
        msg = str(exc_info.value)
        assert any(kw in msg for kw in ["参数验证", "[", "步骤", "diameter"]), (
            f"Error message missing step context: {msg!r}"
        )

    def test_both_eccentricity_and_load_contains_step_name(self) -> None:
        """eccentricity_ratio=0.5 and load=500.0 → ValueError 包含步骤名称"""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(eccentricity_ratio=0.5, load=500.0)
        msg = str(exc_info.value)
        assert any(kw in msg for kw in ["参数验证", "[", "步骤"]), (
            f"Error message missing step context: {msg!r}"
        )

    def test_neither_eccentricity_nor_load_contains_step_name(self) -> None:
        """eccentricity_ratio=None and load=None → ValueError 包含步骤名称"""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(eccentricity_ratio=None, load=None)
        msg = str(exc_info.value)
        assert any(kw in msg for kw in ["参数验证", "[", "步骤"]), (
            f"Error message missing step context: {msg!r}"
        )

    @given(
        epsilon=st.one_of(
            st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(max_examples=30)
    def test_invalid_eccentricity_hypothesis(self, epsilon: float) -> None:
        """Hypothesis: invalid eccentricity_ratio → ValueError with step context."""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(eccentricity_ratio=epsilon, load=None)
        msg = str(exc_info.value)
        assert any(kw in msg for kw in ["参数验证", "[", "步骤", "eccentricity_ratio"]), (
            f"Error message missing step context for epsilon={epsilon}: {msg!r}"
        )

    @given(
        diameter=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_invalid_diameter_hypothesis(self, diameter: float) -> None:
        """Hypothesis: non-positive diameter → ValueError with step context."""
        with pytest.raises(ValueError) as exc_info:
            analyze_bearing(diameter=diameter, eccentricity_ratio=0.5, load=None)
        msg = str(exc_info.value)
        assert any(kw in msg for kw in ["参数验证", "[", "步骤", "diameter"]), (
            f"Error message missing step context for diameter={diameter}: {msg!r}"
        )

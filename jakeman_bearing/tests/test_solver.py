"""
单元测试 — 求解器 BearingSolver

测试网格生成、H系数、K流量、SOR迭代求解。

Feature: journal-bearing-analysis
"""

import math

import numpy as np
import pytest

from jakeman_bearing.bearing_models import BearingInput, GrooveConfig
from jakeman_bearing.bearing_solver import BearingSolver


class TestGridSetup:
    """Test _setup_grid produces correct coordinates and spacings."""

    def test_theta_centers_default(self) -> None:
        """Default Mc=72: θ centers at (j+0.5)*2π/72"""
        inp = BearingInput()
        solver = BearingSolver(inp)
        assert len(solver.theta_centers) == 72
        expected_0 = 0.5 * 2.0 * math.pi / 72
        np.testing.assert_allclose(solver.theta_centers[0], expected_0, atol=1e-12)

    def test_theta_edges_span_2pi(self) -> None:
        """θ edges span [0, 2π]"""
        inp = BearingInput()
        solver = BearingSolver(inp)
        np.testing.assert_allclose(solver.theta_edges[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(solver.theta_edges[-1], 2.0 * math.pi, atol=1e-12)

    def test_uniform_delta_a(self) -> None:
        """Uniform grid: all Δa = L/Na"""
        L = 0.02368
        Na = 10
        inp = BearingInput(length_m=L, n_axial=Na, axial_grading_factor=1.0)
        solver = BearingSolver(inp)
        expected = L / Na
        np.testing.assert_allclose(solver.delta_a, expected, atol=1e-14)

    def test_s_edges_span(self) -> None:
        """s edges span [-L/2, +L/2]"""
        L = 0.05
        inp = BearingInput(length_m=L, n_axial=8)
        solver = BearingSolver(inp)
        np.testing.assert_allclose(solver.s_edges[0], -L / 2, atol=1e-14)
        np.testing.assert_allclose(solver.s_edges[-1], L / 2, atol=1e-14)

    def test_delta_c_formula(self) -> None:
        """Δc = π × D / Mc"""
        D = 0.0635
        Mc = 72
        inp = BearingInput(diameter_m=D, n_circumferential=Mc)
        solver = BearingSolver(inp)
        expected = math.pi * D / Mc
        np.testing.assert_allclose(solver.delta_c, expected, atol=1e-14)

    def test_surface_velocity(self) -> None:
        """U = π × D × N"""
        D = 0.0635
        N = 2000.0 / 60.0
        inp = BearingInput(diameter_m=D, speed_rps=N)
        solver = BearingSolver(inp)
        expected = math.pi * D * N
        np.testing.assert_allclose(solver.U, expected, atol=1e-12)


class TestGrooveMask:
    """Test _setup_groove_mask for different groove types."""

    def test_no_groove(self) -> None:
        """groove_type='none' → all False"""
        groove = GrooveConfig(
            groove_type="none",
            angular_positions_deg=[],
            angular_width_deg=0,
            supply_pressure_pa=0,
        )
        inp = BearingInput(groove=groove, n_circumferential=8, n_axial=4)
        solver = BearingSolver(inp)
        assert not np.any(solver.groove_mask)

    def test_circumferential_360_marks_axial_columns(self) -> None:
        """circumferential_360 groove marks some axial columns for all θ"""
        groove = GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,
            axial_position_ratio=0.5,
            axial_width_ratio=0.2,
        )
        inp = BearingInput(groove=groove, n_circumferential=8, n_axial=10)
        solver = BearingSolver(inp)
        # At least one column should be marked
        assert np.any(solver.groove_mask)
        # All marked columns should have all θ rows marked
        for i in range(10):
            col = solver.groove_mask[:, i]
            if np.any(col):
                assert np.all(col), f"Column {i} partially marked for 360° groove"


class TestHCoefficients:
    """Test _compute_H_coefficients with known values."""

    def test_known_values(self) -> None:
        """Hand-calculated H coefficients for uniform h"""
        inp = BearingInput(
            n_circumferential=4, n_axial=4,
            viscosity_pa_s=0.01,
            axial_grading_factor=1.0,
        )
        solver = BearingSolver(inp)

        h = 5e-5  # uniform film thickness
        ha = np.full((4, 4), h)
        hb = np.full((4, 4), h)
        hc = np.full((4, 4), h)
        hd = np.full((4, 4), h)

        Hci, Hai, Hco, Hao = solver._compute_H_coefficients(ha, hb, hc, hd)

        eta = 0.01
        dc = solver.delta_c
        da = solver.delta_a[0]  # uniform grid

        expected_Hci = (2 * h) ** 3 * da / (96 * eta * dc)
        # For interior cells (I=1,2): da_ai = da, so Hai same as before
        # For boundary cell I=0: da_ai = da/2, so Hai is doubled
        expected_Hai_interior = (2 * h) ** 3 * dc / (96 * eta * da)
        expected_Hai_boundary = (2 * h) ** 3 * dc / (96 * eta * (da / 2.0))

        # Interior cell (I=1)
        np.testing.assert_allclose(Hci[0, 1], expected_Hci, rtol=1e-10)
        np.testing.assert_allclose(Hai[0, 1], expected_Hai_interior, rtol=1e-10)
        np.testing.assert_allclose(Hco[0, 1], expected_Hci, rtol=1e-10)
        np.testing.assert_allclose(Hao[0, 1], expected_Hai_interior, rtol=1e-10)

        # Boundary cell (I=0): Hai uses da/2 as gradient distance
        np.testing.assert_allclose(Hai[0, 0], expected_Hai_boundary, rtol=1e-10)
        # Boundary cell (I=3): Hao uses da/2 as gradient distance
        expected_Hao_boundary = (2 * h) ** 3 * dc / (96 * eta * (da / 2.0))
        np.testing.assert_allclose(Hao[0, 3], expected_Hao_boundary, rtol=1e-10)


class TestKFlow:
    """Test _compute_K_flow with known values."""

    def test_uniform_thickness_zero_K(self) -> None:
        """When ha=hb=hc=hd, K should be zero (no net Couette flow difference)"""
        inp = BearingInput(n_circumferential=4, n_axial=4)
        solver = BearingSolver(inp)

        h = 5e-5
        ha = np.full((4, 4), h)
        hb = np.full((4, 4), h)
        hc = np.full((4, 4), h)
        hd = np.full((4, 4), h)

        K = solver._compute_K_flow(ha, hb, hc, hd)
        np.testing.assert_allclose(K, 0.0, atol=1e-20)


class TestSolverBasic:
    """Test solve() produces valid output for default parameters."""

    def test_default_solve_runs(self) -> None:
        """Default parameters produce valid output (iterations > 0, finite pressures)"""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=500,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            output = solver.solve()
        assert output.iterations > 0
        assert np.all(np.isfinite(output.pressure_field_pa))

    def test_pressure_field_shape(self) -> None:
        """Pressure field shape matches (Mc, Na)"""
        Mc, Na = 36, 6
        inp = BearingInput(
            n_circumferential=Mc, n_axial=Na,
            max_iterations=1000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()
        assert output.pressure_field_pa.shape == (Mc, Na)

    def test_cavitation_matrix_shape(self) -> None:
        """Cavitation matrix shape matches (Mc, Na)"""
        Mc, Na = 36, 6
        inp = BearingInput(
            n_circumferential=Mc, n_axial=Na,
            max_iterations=1000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()
        assert output.cavitation_matrix.shape == (Mc, Na)
        assert output.cavitation_matrix.dtype == bool

    def test_groove_pressure_preserved(self) -> None:
        """Groove cells maintain supply pressure after solving"""
        groove = GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,
            axial_position_ratio=0.5,
            axial_width_ratio=0.2145,
        )
        inp = BearingInput(
            groove=groove,
            n_circumferential=36,
            n_axial=8,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()

        P = output.pressure_field_pa
        groove_P = P[solver.groove_mask]
        if groove_P.size > 0:
            np.testing.assert_allclose(groove_P, 206700, atol=1e-10)

    def test_min_film_thickness_positive(self) -> None:
        """Minimum film thickness should be positive"""
        inp = BearingInput(
            n_circumferential=36, n_axial=6,
            max_iterations=1000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()
        assert output.min_film_thickness_m > 0

    def test_no_nan_in_pressure(self) -> None:
        """Pressure field should contain no NaN values"""
        inp = BearingInput(
            n_circumferential=36, n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        output = solver.solve()
        assert np.all(np.isfinite(output.pressure_field_pa))

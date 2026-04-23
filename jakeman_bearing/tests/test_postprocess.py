"""
test_postprocess.py — 后处理模块单元测试

测试 compute_load_capacity, compute_moments, find_min_film_thickness
使用已知压力场手算对比。
"""

import math

import numpy as np
import pytest

from jakeman_bearing.bearing_postprocess import (
    _compute_12point_average,
    compute_load_capacity,
    compute_moments,
    find_min_film_thickness,
)


class TestCompute12PointAverage:
    """Test the 12-point weighted average helper."""

    def test_uniform_pressure_unchanged(self) -> None:
        """Uniform pressure field → P_mean == P everywhere."""
        Mc, Na = 8, 6
        P = np.full((Mc, Na), 1000.0)
        P_mean = _compute_12point_average(P)
        np.testing.assert_allclose(P_mean, 1000.0, atol=1e-10)

    def test_single_spike_smoothed(self) -> None:
        """A single spike cell should be smoothed by the 12-point average."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        # Place a spike at an interior cell
        j, i = 3, 3
        P[j, i] = 1200.0
        P_mean = _compute_12point_average(P)
        # The spike cell: (4*1200 + 0) / 12 = 400
        assert abs(P_mean[j, i] - 400.0) < 1e-10
        # A direct neighbor: (4*0 + 1200) / 12 = 100
        assert abs(P_mean[j - 1, i] - 100.0) < 1e-10

    def test_periodic_circumferential_boundary(self) -> None:
        """Verify periodic wrapping in circumferential direction."""
        Mc, Na = 4, 4
        P = np.zeros((Mc, Na))
        # Set pressure at J=0, I=2
        P[0, 2] = 120.0
        P_mean = _compute_12point_average(P)
        # J=Mc-1 (last row) should see P[0,2] as its J+1 neighbor
        # P_mean[Mc-1, 2] = (4*0 + 0+0+0+0 + 0+120+0+0) / 12 = 10
        # neighbors of (3,2): J-1=2, J+1=0(periodic), I-1=1, I+1=3
        # diag: (2,1),(2,3),(0,1),(0,3)
        # only P[0,2] is nonzero and it's the J+1 neighbor
        assert abs(P_mean[Mc - 1, 2] - 120.0 / 12.0) < 1e-10

    def test_axial_boundary_uses_own_value(self) -> None:
        """At I=0, the I-1 neighbor should use the cell's own pressure."""
        Mc, Na = 4, 4
        P = np.zeros((Mc, Na))
        P[1, 0] = 60.0
        P_mean = _compute_12point_average(P)
        # For cell (1,0): neighbors I-1 → use own value = 60
        # 4 direct neighbors: J-1=(0,0)=0, J+1=(2,0)=0, I-1=self=60, I+1=(1,1)=0
        # 4 diagonal: (0,-1)→(0,0)=0, (0,1)=0, (2,-1)→(2,0)=0, (2,1)=0
        # P_mean = (4*60 + 60) / 12 = 300/12 = 25
        assert abs(P_mean[1, 0] - 25.0) < 1e-10


class TestComputeLoadCapacity:
    """Test compute_load_capacity with known pressure fields."""

    def test_zero_pressure_zero_load(self) -> None:
        """Zero pressure → zero load capacity."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        theta = np.linspace(0, 2 * np.pi, Mc, endpoint=False) + np.pi / Mc
        da = np.full(Na, 0.001)
        dc = 0.01

        Fy, Fx, F_total, angle = compute_load_capacity(P, theta, da, dc)
        assert abs(Fy) < 1e-15
        assert abs(Fx) < 1e-15
        assert abs(F_total) < 1e-15

    def test_uniform_pressure_zero_net_load(self) -> None:
        """Uniform pressure around full 360° → net load ≈ 0."""
        Mc = 72
        Na = 10
        P = np.full((Mc, Na), 1e6)
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        da = np.full(Na, 0.002)
        dc = np.pi * 0.0635 / Mc

        Fy, Fx, F_total, angle = compute_load_capacity(P, theta, da, dc)
        # Net force should be very small (numerical noise only)
        assert F_total < 1.0, f"Uniform pressure should give ~0 net load, got {F_total}"

    def test_known_single_cell_pressure(self) -> None:
        """Single cell with known pressure → verify force direction."""
        Mc, Na = 4, 4
        P = np.zeros((Mc, Na))
        # Put pressure at θ=0 (top), which should push downward (negative Fy)
        P[0, :] = 1000.0
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        da = np.full(Na, 0.01)
        dc = 0.01

        Fy, Fx, F_total, angle = compute_load_capacity(P, theta, da, dc)
        # Pressure at θ≈π/4 → cos(θ) > 0, so Fy should be negative
        assert Fy < 0, "Pressure near θ=0 should produce negative Fy"

    def test_attitude_angle_range(self) -> None:
        """Attitude angle should be in degrees."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        P[2, :] = 5e5  # pressure at some angle
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        da = np.full(Na, 0.002)
        dc = 0.01

        _, _, _, angle = compute_load_capacity(P, theta, da, dc)
        assert -180.0 <= angle <= 180.0


class TestComputeMoments:
    """Test compute_moments with known pressure fields."""

    def test_zero_pressure_zero_moments(self) -> None:
        """Zero pressure → zero moments."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        theta = np.linspace(0, 2 * np.pi, Mc, endpoint=False) + np.pi / Mc
        s = np.linspace(-0.01, 0.01, Na)
        da = np.full(Na, 0.001)
        dc = 0.01

        My, Mx = compute_moments(P, theta, s, da, dc)
        assert abs(My) < 1e-15
        assert abs(Mx) < 1e-15

    def test_symmetric_pressure_zero_moments(self) -> None:
        """Pressure symmetric about s=0 → moments ≈ 0."""
        Mc, Na = 8, 10
        P = np.full((Mc, Na), 1e6)
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        L = 0.02
        s = np.linspace(-L / 2, L / 2, Na)
        da = np.full(Na, L / Na)
        dc = 0.005

        My, Mx = compute_moments(P, theta, s, da, dc)
        # Symmetric pressure × symmetric s → moments should be ~0
        assert abs(My) < 1e-6, f"Symmetric pressure should give ~0 My, got {My}"
        assert abs(Mx) < 1e-6, f"Symmetric pressure should give ~0 Mx, got {Mx}"

    def test_asymmetric_pressure_nonzero_moment(self) -> None:
        """Pressure only on one side of s=0 → nonzero moment."""
        Mc, Na = 4, 6
        P = np.zeros((Mc, Na))
        # Pressure only on positive s side
        P[0, Na // 2:] = 1e6
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        L = 0.02
        s = np.linspace(-L / 2, L / 2, Na)
        da = np.full(Na, L / Na)
        dc = 0.005

        My, Mx = compute_moments(P, theta, s, da, dc)
        # Should have nonzero moment since pressure is asymmetric
        assert abs(My) > 0 or abs(Mx) > 0


class TestFindMinFilmThickness:
    """Test find_min_film_thickness."""

    def test_uniform_thickness(self) -> None:
        """Uniform thickness → min equals that value."""
        Mc, Na = 8, 6
        h = np.full((Mc, Na), 1e-5)
        theta = np.linspace(0, 2 * np.pi, Mc, endpoint=False)
        s = np.linspace(-0.01, 0.01, Na)

        min_h, (theta_deg, s_m) = find_min_film_thickness(h, theta, s)
        assert abs(min_h - 1e-5) < 1e-15

    def test_known_minimum_location(self) -> None:
        """Place minimum at known location and verify it's found."""
        Mc, Na = 8, 6
        h = np.full((Mc, Na), 1e-4)
        h[3, 2] = 1e-6  # minimum here
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        s = np.linspace(-0.01, 0.01, Na)

        min_h, (theta_deg, s_m) = find_min_film_thickness(h, theta, s)
        assert abs(min_h - 1e-6) < 1e-15
        expected_theta_deg = math.degrees(theta[3])
        assert abs(theta_deg - expected_theta_deg) < 1e-10
        assert abs(s_m - s[2]) < 1e-15

    def test_returns_degrees(self) -> None:
        """Theta in result should be in degrees."""
        Mc, Na = 36, 10
        h = np.ones((Mc, Na)) * 1e-4
        h[18, 5] = 5e-6
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        s = np.linspace(-0.05, 0.05, Na)

        _, (theta_deg, _) = find_min_film_thickness(h, theta, s)
        # θ[18] = (18.5) * 360/36 = 185°
        assert abs(theta_deg - 185.0) < 1e-10


from jakeman_bearing.bearing_postprocess import compute_side_leakage, compute_power_loss


class TestComputeSideLeakage:
    """Test compute_side_leakage with known pressure/H-coefficient fields."""

    def test_zero_pressure_zero_leakage(self) -> None:
        """Zero pressure field with zero ambient → zero leakage."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        Hao = np.ones((Mc, Na)) * 1e-8
        Hai = np.ones((Mc, Na)) * 1e-8
        da = np.full(Na, 0.002)
        dc = 0.01

        Qs = compute_side_leakage(P, Hao, Hai, 0.0, da, dc)
        assert abs(Qs) < 1e-20

    def test_uniform_pressure_equal_ambient_zero_leakage(self) -> None:
        """When all pressures equal ambient → zero leakage."""
        Mc, Na = 8, 6
        P_amb = 101325.0
        P = np.full((Mc, Na), P_amb)
        Hao = np.ones((Mc, Na)) * 5e-9
        Hai = np.ones((Mc, Na)) * 5e-9
        da = np.full(Na, 0.002)
        dc = 0.01

        Qs = compute_side_leakage(P, Hao, Hai, P_amb, da, dc)
        assert abs(Qs) < 1e-15

    def test_known_end_pressures(self) -> None:
        """Set known pressures at I=0 and I=Na-1, compute expected leakage by hand."""
        Mc, Na = 4, 5
        P = np.zeros((Mc, Na))
        # Inner end pressures (I=0): [100, 200, 300, 400]
        P[:, 0] = np.array([100.0, 200.0, 300.0, 400.0])
        # Outer end pressures (I=Na-1): [50, 150, 250, 350]
        P[:, Na - 1] = np.array([50.0, 150.0, 250.0, 350.0])

        Hai = np.ones((Mc, Na)) * 2e-8
        Hao = np.ones((Mc, Na)) * 3e-8
        ambient = 0.0
        da = np.full(Na, 0.001)
        dc = 0.005

        Qs = compute_side_leakage(P, Hao, Hai, ambient, da, dc)

        # Hand calculation:
        # Qai[j] = Hai[j, 0] * (P[j, 0] - 0) = 2e-8 * P[j, 0]
        # sum|Qai| = 2e-8 * (100 + 200 + 300 + 400) = 2e-8 * 1000 = 2e-5
        # Qao[j] = Hao[j, Na-1] * (P[j, Na-1] - 0) = 3e-8 * P[j, Na-1]
        # sum|Qao| = 3e-8 * (50 + 150 + 250 + 350) = 3e-8 * 800 = 2.4e-5
        # Qs = 2e-5 + 2.4e-5 = 4.4e-5
        expected = 2e-8 * 1000.0 + 3e-8 * 800.0
        assert abs(Qs - expected) < 1e-15

    def test_leakage_non_negative(self) -> None:
        """Side leakage should always be non-negative (sum of absolute values)."""
        Mc, Na = 6, 4
        # Pressures below ambient → negative (P - P_amb), but |Qai| still positive
        P = np.full((Mc, Na), 50000.0)
        P[:, 0] = 80000.0
        P[:, Na - 1] = 30000.0
        Hao = np.ones((Mc, Na)) * 1e-7
        Hai = np.ones((Mc, Na)) * 1e-7
        ambient = 60000.0
        da = np.full(Na, 0.002)
        dc = 0.01

        Qs = compute_side_leakage(P, Hao, Hai, ambient, da, dc)
        assert Qs >= 0.0


class TestComputePowerLoss:
    """Test compute_power_loss with known film thickness fields."""

    def test_uniform_thickness_known_power(self) -> None:
        """Uniform film thickness → hand-calculate expected friction force and power."""
        Mc, Na = 4, 3
        h = np.full((Mc, Na), 1e-4)  # 0.1 mm uniform
        viscosity = 0.01  # Pa·s
        U = 2.0  # m/s
        da = np.full(Na, 0.005)  # 5 mm each
        dc = 0.01  # 10 mm
        cav = np.zeros((Mc, Na), dtype=bool)

        power, friction = compute_power_loss(h, viscosity, U, da, dc, cav)

        # Hand calculation:
        # Fc[j,i] = η * U * Δc * Δa[i] / h[j,i]
        #         = 0.01 * 2.0 * 0.01 * 0.005 / 1e-4 = 0.01
        # Total cells = 4 * 3 = 12
        # friction = 12 * 0.01 = 0.12 N
        # power = U * friction = 2.0 * 0.12 = 0.24 W
        expected_friction = Mc * Na * (viscosity * U * dc * 0.005 / 1e-4)
        expected_power = U * expected_friction
        assert abs(friction - expected_friction) < 1e-12
        assert abs(power - expected_power) < 1e-12

    def test_power_equals_U_times_friction(self) -> None:
        """Verify power_loss = surface_velocity × friction_force."""
        Mc, Na = 6, 5
        rng = np.random.default_rng(42)
        h = rng.uniform(1e-5, 1e-3, size=(Mc, Na))
        viscosity = 0.014
        U = 3.5
        da = np.full(Na, 0.002)
        dc = 0.008
        cav = np.zeros((Mc, Na), dtype=bool)

        power, friction = compute_power_loss(h, viscosity, U, da, dc, cav)
        assert abs(power - U * friction) < 1e-10

    def test_thinner_film_more_friction(self) -> None:
        """Thinner film should produce more friction than thicker film (same other params)."""
        Mc, Na = 8, 6
        h_thin = np.full((Mc, Na), 5e-5)
        h_thick = np.full((Mc, Na), 2e-4)
        viscosity = 0.01
        U = 1.0
        da = np.full(Na, 0.002)
        dc = 0.005
        cav = np.zeros((Mc, Na), dtype=bool)

        _, friction_thin = compute_power_loss(h_thin, viscosity, U, da, dc, cav)
        _, friction_thick = compute_power_loss(h_thick, viscosity, U, da, dc, cav)
        assert friction_thin > friction_thick

    def test_zero_velocity_zero_power(self) -> None:
        """Zero surface velocity → zero power loss."""
        Mc, Na = 4, 4
        h = np.full((Mc, Na), 1e-4)
        viscosity = 0.01
        U = 0.0
        da = np.full(Na, 0.002)
        dc = 0.005
        cav = np.zeros((Mc, Na), dtype=bool)

        power, friction = compute_power_loss(h, viscosity, U, da, dc, cav)
        assert abs(power) < 1e-20
        assert abs(friction) < 1e-20


import warnings

from jakeman_bearing.bearing_models import BearingInput
from jakeman_bearing.bearing_postprocess import (
    _forces_from_pressure,
    compute_dynamic_coefficients,
)
from jakeman_bearing.bearing_solver import BearingSolver


class TestForcesFromPressure:
    """Test the _forces_from_pressure helper."""

    def test_zero_pressure_zero_forces(self) -> None:
        """Zero pressure → zero forces and moments."""
        Mc, Na = 8, 6
        P = np.zeros((Mc, Na))
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        s = np.linspace(-0.01, 0.01, Na)
        da = np.full(Na, 0.001)
        dc = 0.01

        Fy, Fx, My, Mx = _forces_from_pressure(P, theta, s, da, dc)
        assert abs(Fy) < 1e-15
        assert abs(Fx) < 1e-15
        assert abs(My) < 1e-15
        assert abs(Mx) < 1e-15

    def test_consistent_with_load_and_moments(self) -> None:
        """Forces from _forces_from_pressure should match compute_load_capacity + compute_moments."""
        Mc, Na = 8, 6
        rng = np.random.default_rng(123)
        P = rng.uniform(0, 1e6, size=(Mc, Na))
        theta = np.array([(j + 0.5) * 2 * np.pi / Mc for j in range(Mc)])
        s = np.linspace(-0.01, 0.01, Na)
        da = np.full(Na, 0.002)
        dc = 0.005

        Fy, Fx, My, Mx = _forces_from_pressure(P, theta, s, da, dc)
        Fy2, Fx2, _, _ = compute_load_capacity(P, theta, da, dc)
        My2, Mx2 = compute_moments(P, theta, s, da, dc)

        np.testing.assert_allclose(Fy, Fy2, atol=1e-10)
        np.testing.assert_allclose(Fx, Fx2, atol=1e-10)
        np.testing.assert_allclose(My, My2, atol=1e-10)
        np.testing.assert_allclose(Mx, Mx2, atol=1e-10)


class TestSolvePerturbed:
    """Test BearingSolver.solve_perturbed."""

    def _make_solver(self, Mc: int = 36, Na: int = 6) -> BearingSolver:
        inp = BearingInput(
            n_circumferential=Mc,
            n_axial=Na,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        return BearingSolver(inp)

    def test_returns_correct_shape(self) -> None:
        """solve_perturbed returns pressure field of shape (Mc, Na)."""
        Mc, Na = 36, 6
        solver = self._make_solver(Mc, Na)
        ecy = 0.6 * solver.input.clearance_m / 2.0
        ecx = 0.0
        P = solver.solve_perturbed(ecy, ecx)
        assert P.shape == (Mc, Na)

    def test_zero_perturbation_matches_solve(self) -> None:
        """With zero perturbation, solve_perturbed should match solve()."""
        Mc, Na = 36, 6
        solver = self._make_solver(Mc, Na)
        ecy = 0.6 * solver.input.clearance_m / 2.0
        ecx = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            output = solver.solve(ecy=ecy, ecx=ecx)

        P_pert = solver.solve_perturbed(ecy, ecx)
        np.testing.assert_allclose(P_pert, output.pressure_field_pa, atol=1e-6)

    def test_displacement_perturbation_changes_pressure(self) -> None:
        """A displacement perturbation should change the pressure field."""
        solver = self._make_solver()
        ecy = 0.6 * solver.input.clearance_m / 2.0
        ecx = 0.0
        delta = 1e-3 * solver.input.clearance_m / 2.0

        P_base = solver.solve_perturbed(ecy, ecx)
        P_pert = solver.solve_perturbed(ecy, ecx, delta_ecy=delta)

        # Pressure fields should differ
        assert not np.allclose(P_base, P_pert, atol=1e-10)

    def test_velocity_perturbation_changes_pressure(self) -> None:
        """A velocity perturbation should change the pressure field."""
        solver = self._make_solver()
        ecy = 0.6 * solver.input.clearance_m / 2.0
        ecx = 0.0
        omega = 2.0 * math.pi * solver.input.speed_rps
        delta_vel = 1e-3 * solver.input.clearance_m / 2.0 * omega

        P_base = solver.solve_perturbed(ecy, ecx)
        P_pert = solver.solve_perturbed(ecy, ecx, delta_ecy_dot=delta_vel)

        assert not np.allclose(P_base, P_pert, atol=1e-10)

    def test_finite_pressures(self) -> None:
        """solve_perturbed should produce finite pressures."""
        solver = self._make_solver()
        ecy = 0.5 * solver.input.clearance_m / 2.0
        ecx = 0.0
        delta = 1e-3 * solver.input.clearance_m / 2.0

        P = solver.solve_perturbed(ecy, ecx, delta_ecy=delta)
        assert np.all(np.isfinite(P))


class TestComputeDynamicCoefficients:
    """Test compute_dynamic_coefficients for aligned and misaligned bearings."""

    def test_aligned_returns_2x2(self) -> None:
        """Aligned bearing should return 2×2 stiffness and damping matrices."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        ecy = 0.6 * inp.clearance_m / 2.0
        ecx = 0.0

        stiffness, damping = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=False
        )
        assert stiffness.shape == (2, 2)
        assert damping.shape == (2, 2)

    def test_aligned_coefficients_finite(self) -> None:
        """All aligned dynamic coefficients should be finite."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        ecy = 0.6 * inp.clearance_m / 2.0
        ecx = 0.0

        stiffness, damping = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=False
        )
        assert np.all(np.isfinite(stiffness))
        assert np.all(np.isfinite(damping))

    def test_aligned_stiffness_nonzero(self) -> None:
        """Stiffness matrix should have nonzero entries for a loaded bearing."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        ecy = 0.6 * inp.clearance_m / 2.0
        ecx = 0.0

        stiffness, _ = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=False
        )
        # At least some stiffness coefficients should be nonzero
        assert np.any(np.abs(stiffness) > 0)

    def test_aligned_damping_nonzero(self) -> None:
        """Damping matrix should have nonzero entries."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
        )
        solver = BearingSolver(inp)
        ecy = 0.6 * inp.clearance_m / 2.0
        ecx = 0.0

        _, damping = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=False
        )
        assert np.any(np.abs(damping) > 0)

    def test_misaligned_returns_4x4(self) -> None:
        """Misaligned bearing should return 4×4 stiffness and damping matrices."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
            misalignment_vertical_rad=0.0001,
        )
        solver = BearingSolver(inp)
        ecy = 0.5 * inp.clearance_m / 2.0
        ecx = 0.0

        stiffness, damping = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=True
        )
        assert stiffness.shape == (4, 4)
        assert damping.shape == (4, 4)

    def test_misaligned_coefficients_finite(self) -> None:
        """All misaligned dynamic coefficients should be finite."""
        inp = BearingInput(
            n_circumferential=36,
            n_axial=6,
            max_iterations=2000,
            convergence_tol=1e-4,
            misalignment_vertical_rad=0.0001,
        )
        solver = BearingSolver(inp)
        ecy = 0.5 * inp.clearance_m / 2.0
        ecx = 0.0

        stiffness, damping = compute_dynamic_coefficients(
            solver, ecy, ecx, is_misaligned=True
        )
        assert np.all(np.isfinite(stiffness))
        assert np.all(np.isfinite(damping))

"""
test_practical.py — 实用分析模式单元测试

测试 Brent 法端到端载荷反求偏心 (solve_for_load)。

Feature: journal-bearing-analysis
"""

import pytest

from jakeman_bearing.bearing_models import BearingInput, GrooveConfig
from jakeman_bearing.bearing_practical import solve_for_load


def _make_coarse_input() -> BearingInput:
    """Create a BearingInput with coarse grid for fast tests."""
    return BearingInput(
        eccentricity_ratio=0.5,
        load_n=None,
        n_circumferential=36,
        n_axial=6,
        convergence_tol=1e-4,
        max_iterations=3000,
    )


@pytest.mark.slow
class TestSolveForLoad:
    """End-to-end tests for Brent method load-to-eccentricity solving."""

    def test_basic_load_matching(self) -> None:
        """Solve for a known target load and verify output load matches."""
        inp = _make_coarse_input()
        target = 500.0
        result = solve_for_load(inp, target_load_n=target, load_direction_deg=270.0)
        # The returned load_capacity_n should be close to the target
        assert abs(result.load_capacity_n - target) / target < 0.05

    def test_output_fields_populated(self) -> None:
        """solve_for_load returns a BearingOutput with all post-processing fields populated."""
        inp = _make_coarse_input()
        result = solve_for_load(inp, target_load_n=500.0, load_direction_deg=270.0)

        assert result.load_capacity_n > 0
        assert result.moment_vertical_nm != 0 or result.moment_horizontal_nm != 0
        assert result.side_leakage_flow_m3s > 0
        assert result.power_loss_w > 0
        assert result.min_film_thickness_m > 0

    def test_negative_load_raises_value_error(self) -> None:
        """Passing a negative target_load_n raises ValueError."""
        inp = _make_coarse_input()
        with pytest.raises(ValueError, match="目标载荷必须为正数"):
            solve_for_load(inp, target_load_n=-100.0, load_direction_deg=270.0)

    def test_excessive_load_raises_value_error(self) -> None:
        """An extremely large target load raises ValueError about exceeding capacity."""
        inp = _make_coarse_input()
        with pytest.raises(ValueError, match="载荷超出轴承承载能力"):
            solve_for_load(inp, target_load_n=1e9, load_direction_deg=270.0)

    def test_converged_output(self) -> None:
        """The returned output has converged=True."""
        inp = _make_coarse_input()
        result = solve_for_load(inp, target_load_n=500.0, load_direction_deg=270.0)
        assert result.converged is True

    def test_eccentricity_increases_with_load(self) -> None:
        """Higher load produces higher eccentricity (smaller min film thickness)."""
        inp_lo = _make_coarse_input()
        inp_hi = _make_coarse_input()

        result_lo = solve_for_load(inp_lo, target_load_n=300.0, load_direction_deg=270.0)
        result_hi = solve_for_load(inp_hi, target_load_n=1500.0, load_direction_deg=270.0)

        # Higher load → thinner minimum film
        assert result_hi.min_film_thickness_m < result_lo.min_film_thickness_m

    def test_input_params_preserved(self) -> None:
        """The output's input_params preserves the original bearing geometry."""
        inp = _make_coarse_input()
        result = solve_for_load(inp, target_load_n=500.0, load_direction_deg=270.0)

        assert result.input_params.diameter_m == inp.diameter_m
        assert result.input_params.length_m == inp.length_m
        assert result.input_params.clearance_m == inp.clearance_m
        assert result.input_params.speed_rps == inp.speed_rps
        assert result.input_params.viscosity_pa_s == inp.viscosity_pa_s

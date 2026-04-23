"""
单元测试 — BearingInput / BearingOutput / GrooveConfig 数据模型

测试构造、默认值、验证逻辑和输出方法。
"""

import numpy as np
import pytest

from jakeman_bearing.bearing_models import BearingInput, BearingOutput, GrooveConfig


# ── GrooveConfig 构造测试 ─────────────────────────────────────


class TestGrooveConfig:
    """Test GrooveConfig construction with all groove types."""

    def test_circumferential_360(self):
        g = GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,
        )
        assert g.groove_type == "circumferential_360"
        assert g.angular_width_deg == 360
        assert g.supply_pressure_pa == 206700
        assert g.axial_position_ratio == 0.5  # default
        assert g.axial_width_ratio == 0.05    # default

    def test_axial_dual(self):
        g = GrooveConfig(
            groove_type="axial_dual",
            angular_positions_deg=[90, 270],
            angular_width_deg=30,
            supply_pressure_pa=101325,
        )
        assert g.groove_type == "axial_dual"
        assert g.angular_positions_deg == [90, 270]
        assert g.angular_width_deg == 30

    def test_axial_single(self):
        g = GrooveConfig(
            groove_type="axial_single",
            angular_positions_deg=[180],
            angular_width_deg=20,
            supply_pressure_pa=50000,
        )
        assert g.groove_type == "axial_single"
        assert g.angular_positions_deg == [180]

    def test_none_type(self):
        g = GrooveConfig(
            groove_type="none",
            angular_positions_deg=[],
            angular_width_deg=0,
            supply_pressure_pa=0,
        )
        assert g.groove_type == "none"
        assert g.angular_positions_deg == []

    def test_custom_axial_ratios(self):
        g = GrooveConfig(
            groove_type="circumferential_360",
            angular_positions_deg=[0],
            angular_width_deg=360,
            supply_pressure_pa=206700,
            axial_position_ratio=0.3,
            axial_width_ratio=0.1,
        )
        assert g.axial_position_ratio == 0.3
        assert g.axial_width_ratio == 0.1


# ── BearingInput 默认值与验证测试 ─────────────────────────────


class TestBearingInputDefaults:
    """Test BearingInput default values match Table 1 Case 1."""

    def test_default_geometry(self):
        inp = BearingInput()
        assert inp.diameter_m == 0.0635
        assert inp.length_m == 0.02368
        assert inp.clearance_m == 0.0000635

    def test_default_operating_conditions(self):
        inp = BearingInput()
        assert inp.speed_rps == pytest.approx(2000.0 / 60.0)
        assert inp.viscosity_pa_s == 0.014

    def test_default_eccentricity(self):
        inp = BearingInput()
        assert inp.eccentricity_ratio == 0.6
        assert inp.load_n is None

    def test_default_misalignment_zero(self):
        inp = BearingInput()
        assert inp.misalignment_vertical_rad == 0.0
        assert inp.misalignment_horizontal_rad == 0.0

    def test_default_mesh(self):
        inp = BearingInput()
        assert inp.n_circumferential == 72
        assert inp.n_axial == 10
        assert inp.axial_grading_factor == 1.0

    def test_default_solver(self):
        inp = BearingInput()
        assert inp.over_relaxation_factor == 1.7
        assert inp.max_iterations == 10000
        assert inp.convergence_tol == 1e-4

    def test_default_pressure(self):
        inp = BearingInput()
        assert inp.cavitation_pressure_pa == 0.0
        assert inp.ambient_pressure_pa == 0.0

    def test_default_groove_is_none(self):
        inp = BearingInput()
        assert inp.groove is None


class TestBearingInputValidate:
    """Test BearingInput.validate() passes and fails correctly."""

    def test_validate_passes_with_defaults(self):
        inp = BearingInput()
        inp.validate()  # should not raise

    def test_validate_passes_load_mode(self):
        inp = BearingInput(eccentricity_ratio=None, load_n=5000.0)
        inp.validate()  # should not raise

    def test_eccentricity_zero_rejected(self):
        inp = BearingInput(eccentricity_ratio=0.0)
        with pytest.raises(ValueError, match="eccentricity_ratio"):
            inp.validate()

    def test_eccentricity_one_rejected(self):
        inp = BearingInput(eccentricity_ratio=1.0)
        with pytest.raises(ValueError, match="eccentricity_ratio"):
            inp.validate()

    def test_negative_diameter_rejected(self):
        inp = BearingInput(diameter_m=-0.01)
        with pytest.raises(ValueError, match="diameter_m"):
            inp.validate()

    def test_zero_length_rejected(self):
        inp = BearingInput(length_m=0.0)
        with pytest.raises(ValueError, match="length_m"):
            inp.validate()

    def test_negative_clearance_rejected(self):
        inp = BearingInput(clearance_m=-1e-6)
        with pytest.raises(ValueError, match="clearance_m"):
            inp.validate()

    def test_zero_speed_rejected(self):
        inp = BearingInput(speed_rps=0.0)
        with pytest.raises(ValueError, match="speed_rps"):
            inp.validate()

    def test_negative_viscosity_rejected(self):
        inp = BearingInput(viscosity_pa_s=-0.01)
        with pytest.raises(ValueError, match="viscosity_pa_s"):
            inp.validate()

    def test_both_eccentricity_and_load_rejected(self):
        inp = BearingInput(eccentricity_ratio=0.5, load_n=1000.0)
        with pytest.raises(ValueError):
            inp.validate()

    def test_neither_eccentricity_nor_load_rejected(self):
        inp = BearingInput(eccentricity_ratio=None, load_n=None)
        with pytest.raises(ValueError):
            inp.validate()

    def test_n_circumferential_too_small(self):
        inp = BearingInput(n_circumferential=3)
        with pytest.raises(ValueError, match="n_circumferential"):
            inp.validate()

    def test_n_axial_too_small(self):
        inp = BearingInput(n_axial=2)
        with pytest.raises(ValueError, match="n_axial"):
            inp.validate()

    def test_sor_factor_zero_rejected(self):
        inp = BearingInput(over_relaxation_factor=0.0)
        with pytest.raises(ValueError, match="over_relaxation_factor"):
            inp.validate()

    def test_sor_factor_two_rejected(self):
        inp = BearingInput(over_relaxation_factor=2.0)
        with pytest.raises(ValueError, match="over_relaxation_factor"):
            inp.validate()


class TestBearingInputDefaultGroove:
    """Test BearingInput.default_groove() returns correct GrooveConfig."""

    def test_default_groove_type(self):
        g = BearingInput.default_groove()
        assert isinstance(g, GrooveConfig)
        assert g.groove_type == "circumferential_360"

    def test_default_groove_supply_pressure(self):
        g = BearingInput.default_groove()
        assert g.supply_pressure_pa == 206700

    def test_default_groove_axial_position(self):
        g = BearingInput.default_groove()
        assert g.axial_position_ratio == 0.5

    def test_default_groove_axial_width(self):
        g = BearingInput.default_groove()
        assert g.axial_width_ratio == pytest.approx(0.2145)

    def test_default_groove_angular_width(self):
        g = BearingInput.default_groove()
        assert g.angular_width_deg == 360


# ── BearingOutput 构造与方法测试 ──────────────────────────────


def _make_minimal_output() -> BearingOutput:
    """Helper: create a BearingOutput with minimal valid data."""
    rows, cols = 8, 4
    return BearingOutput(
        pressure_field_pa=np.zeros((rows, cols)),
        cavitation_matrix=np.zeros((rows, cols), dtype=bool),
        film_thickness_field_m=np.ones((rows, cols)) * 1e-5,
        load_capacity_n=500.0,
        load_vertical_n=400.0,
        load_horizontal_n=300.0,
        attitude_angle_deg=36.87,
        moment_vertical_nm=0.01,
        moment_horizontal_nm=0.02,
        min_film_thickness_m=1e-5,
        min_film_location=(180.0, 0.005),
        side_leakage_flow_m3s=1e-7,
        power_loss_w=25.0,
        friction_force_n=5.0,
        stiffness_coefficients=np.eye(2),
        damping_coefficients=np.eye(2),
        iterations=200,
        converged=True,
        residual=0.5,
        input_params=BearingInput(),
    )


class TestBearingOutputConstruction:
    """Test BearingOutput construction with minimal valid data."""

    def test_construction_succeeds(self):
        out = _make_minimal_output()
        assert out.load_capacity_n == 500.0
        assert out.converged is True
        assert out.iterations == 200

    def test_pressure_field_shape(self):
        out = _make_minimal_output()
        assert out.pressure_field_pa.shape == (8, 4)

    def test_input_params_echoed(self):
        out = _make_minimal_output()
        assert isinstance(out.input_params, BearingInput)
        assert out.input_params.diameter_m == 0.0635


class TestBearingOutputSummary:
    """Test BearingOutput.summary() returns non-empty string with key info."""

    def test_summary_non_empty(self):
        out = _make_minimal_output()
        text = out.summary()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_summary_contains_load(self):
        out = _make_minimal_output()
        text = out.summary()
        assert "500.0" in text

    def test_summary_contains_convergence(self):
        out = _make_minimal_output()
        text = out.summary()
        assert "已收敛" in text


class TestValidationErrorMessages:
    """Test that validation error messages contain parameter names."""

    @pytest.mark.parametrize(
        "field,value,expected_name",
        [
            ("eccentricity_ratio", 0.0, "eccentricity_ratio"),
            ("eccentricity_ratio", 1.0, "eccentricity_ratio"),
            ("eccentricity_ratio", -0.5, "eccentricity_ratio"),
            ("eccentricity_ratio", 1.5, "eccentricity_ratio"),
            ("diameter_m", 0.0, "diameter_m"),
            ("diameter_m", -1.0, "diameter_m"),
            ("length_m", 0.0, "length_m"),
            ("clearance_m", -0.001, "clearance_m"),
            ("speed_rps", 0.0, "speed_rps"),
            ("viscosity_pa_s", -0.01, "viscosity_pa_s"),
            ("n_circumferential", 3, "n_circumferential"),
            ("n_axial", 1, "n_axial"),
            ("over_relaxation_factor", 0.0, "over_relaxation_factor"),
            ("over_relaxation_factor", 2.0, "over_relaxation_factor"),
        ],
    )
    def test_error_message_contains_param_name(self, field, value, expected_name):
        kwargs = {field: value}
        inp = BearingInput(**kwargs)
        with pytest.raises(ValueError, match=expected_name):
            inp.validate()

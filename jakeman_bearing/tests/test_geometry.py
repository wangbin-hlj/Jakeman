"""
单元测试 — 几何计算手算对比

用已知参数手算对比 compute_film_thickness 和 compute_element_corner_thicknesses。

Feature: journal-bearing-analysis
"""

import numpy as np
import pytest

from jakeman_bearing.bearing_geometry import (
    compute_eccentricity_components,
    compute_element_corner_thicknesses,
    compute_film_thickness,
)


class TestGeometryHandCalculations:
    """Hand-calculation verification for geometry functions.

    Known params: Cd=0.0001 m, epsilon=0.5, aligned (gamma=lam=0).
    Radial clearance = Cd/2 = 0.00005 m.
    e = epsilon * Cd/2 = 0.5 * 0.00005 = 0.000025 m.
    psi = 0 (attitude angle = 0, so ecy=e, ecx=0).

    At theta=0:  h = 0.00005 + 0.000025*cos(0)  = 0.000075 m
    At theta=pi: h = 0.00005 + 0.000025*cos(pi) = 0.000025 m
    """

    Cd = 0.0001          # diameter clearance (m)
    epsilon = 0.5
    radial_clearance = Cd / 2.0   # 0.00005 m
    e_val = epsilon * radial_clearance  # 0.000025 m
    ecy = e_val          # aligned, attitude=0 → ecy=e, ecx=0
    ecx = 0.0
    gamma = 0.0
    lam = 0.0

    def test_eccentricity_components_aligned(self) -> None:
        """Aligned bearing: esy=ecy, esx=0, e=ecy, psi=0 for all s."""
        s = np.array([-0.01, 0.0, 0.01])
        esy, esx, e, psi = compute_eccentricity_components(
            self.ecy, self.ecx, self.gamma, self.lam, s
        )
        np.testing.assert_allclose(esy, self.ecy, atol=1e-15)
        np.testing.assert_allclose(esx, 0.0, atol=1e-15)
        np.testing.assert_allclose(e, self.e_val, atol=1e-15)
        np.testing.assert_allclose(psi, 0.0, atol=1e-15)

    def test_film_thickness_at_theta_zero(self) -> None:
        """At theta=0: h = Cd/2 + e*cos(0) = 0.00005 + 0.000025 = 0.000075"""
        theta = np.array([0.0])
        s = np.array([0.0])
        _, _, e, psi = compute_eccentricity_components(
            self.ecy, self.ecx, self.gamma, self.lam, s
        )
        h = compute_film_thickness(self.Cd, e, psi, theta)
        np.testing.assert_allclose(h[0, 0], 0.000075, atol=1e-12)

    def test_film_thickness_at_theta_pi(self) -> None:
        """At theta=pi: h = Cd/2 + e*cos(pi) = 0.00005 - 0.000025 = 0.000025"""
        theta = np.array([np.pi])
        s = np.array([0.0])
        _, _, e, psi = compute_eccentricity_components(
            self.ecy, self.ecx, self.gamma, self.lam, s
        )
        h = compute_film_thickness(self.Cd, e, psi, theta)
        np.testing.assert_allclose(h[0, 0], 0.000025, atol=1e-12)

    def test_element_corner_thicknesses_aligned(self) -> None:
        """Corner thicknesses for a 2x2 grid, aligned bearing."""
        theta_edges = np.array([0.0, np.pi, 2 * np.pi])
        s_edges = np.array([-0.01, 0.0, 0.01])

        _, _, e, psi = compute_eccentricity_components(
            self.ecy, self.ecx, self.gamma, self.lam, s_edges
        )
        ha, hb, hc, hd = compute_element_corner_thicknesses(
            self.Cd, e, psi, theta_edges, s_edges
        )

        # Element (0,0): theta_edges[0]=0, theta_edges[1]=pi, s_edges[0]=-0.01, s_edges[1]=0
        # ha = h(0, s=-0.01) = 0.00005 + 0.000025*cos(0) = 0.000075
        # hb = h(0, s=0)     = 0.000075
        # hc = h(pi, s=-0.01)= 0.00005 + 0.000025*cos(pi) = 0.000025
        # hd = h(pi, s=0)    = 0.000025
        np.testing.assert_allclose(ha[0, 0], 0.000075, atol=1e-12)
        np.testing.assert_allclose(hb[0, 0], 0.000075, atol=1e-12)
        np.testing.assert_allclose(hc[0, 0], 0.000025, atol=1e-12)
        np.testing.assert_allclose(hd[0, 0], 0.000025, atol=1e-12)

    def test_film_thickness_symmetry(self) -> None:
        """For aligned bearing, h(theta) = h(2*pi - theta)."""
        theta = np.linspace(0, 2 * np.pi, 73, endpoint=True)
        s = np.array([0.0])
        _, _, e, psi = compute_eccentricity_components(
            self.ecy, self.ecx, self.gamma, self.lam, s
        )
        h = compute_film_thickness(self.Cd, e, psi, theta)
        # h at theta and 2*pi - theta should be equal
        for i in range(len(theta) // 2):
            j = len(theta) - 1 - i
            np.testing.assert_allclose(
                h[i, 0], h[j, 0], atol=1e-12,
                err_msg=f"Symmetry broken at theta={theta[i]:.4f} vs {theta[j]:.4f}"
            )

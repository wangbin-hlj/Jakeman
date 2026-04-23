"""
visualization.py — Matplotlib visualization functions for journal bearing analysis.

All functions accept a BearingOutput object and an optional save_path.
If save_path is provided, the figure is saved (PNG or PDF based on extension).
If save_path is None, plt.show() is called.
plt.close() is always called after saving to avoid memory leaks.
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from jakeman_bearing.bearing_models import BearingOutput


def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to file or show it, then close."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def _build_axes(output: BearingOutput) -> tuple[np.ndarray, np.ndarray]:
    """Return (theta_deg, s_mm) coordinate arrays matching the pressure field shape."""
    inp = output.input_params
    Mc, Na = output.pressure_field_pa.shape

    theta_deg = np.linspace(0.5 * 360.0 / Mc, 360.0 - 0.5 * 360.0 / Mc, Mc)
    s_mm = np.linspace(
        -inp.length_m / 2 * 1e3 + inp.length_m / Na * 1e3 / 2,
        inp.length_m / 2 * 1e3 - inp.length_m / Na * 1e3 / 2,
        Na,
    )
    return theta_deg, s_mm


def plot_pressure_3d(output: BearingOutput, save_path: str | None = None) -> None:
    """3D surface plot of oil film pressure distribution.

    X = circumferential angle (deg), Y = axial position (mm), Z = pressure (MPa).
    """
    theta_deg, s_mm = _build_axes(output)
    pressure_mpa = output.pressure_field_pa / 1e6

    # Clamp negative (cavitation) pressures to zero for display
    pressure_mpa_display = np.maximum(pressure_mpa, 0.0)

    Theta, S = np.meshgrid(theta_deg, s_mm, indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Theta, S, pressure_mpa_display,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    ax.set_xlabel("Circumferential Angle (deg)", labelpad=10)
    ax.set_ylabel("Axial Position (mm)", labelpad=10)
    ax.set_zlabel("Pressure (MPa)", labelpad=10)
    ax.set_title("Oil Film Pressure Distribution (3D)")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Pressure (MPa)")

    _save_or_show(fig, save_path)


def plot_pressure_contour(output: BearingOutput, save_path: str | None = None) -> None:
    """Pressure contour plot (top view).

    X = circumferential angle (deg), Y = axial position (mm), colour = pressure (MPa).
    """
    theta_deg, s_mm = _build_axes(output)
    pressure_mpa = output.pressure_field_pa / 1e6
    pressure_mpa_display = np.maximum(pressure_mpa, 0.0)

    Theta, S = np.meshgrid(theta_deg, s_mm, indexing="ij")

    fig, ax = plt.subplots(figsize=(10, 5))

    levels = 20
    cf = ax.contourf(Theta, S, pressure_mpa_display, levels=levels, cmap="jet")
    cs = ax.contour(Theta, S, pressure_mpa_display, levels=levels,
                    colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    fig.colorbar(cf, ax=ax, label="Pressure (MPa)")

    ax.set_xlabel("Circumferential Angle (deg)")
    ax.set_ylabel("Axial Position (mm)")
    ax.set_title("Oil Film Pressure Contour")
    ax.set_xlim(0, 360)

    _save_or_show(fig, save_path)


def plot_cavitation_map(output: BearingOutput, save_path: str | None = None) -> None:
    """Cavitation map — 2D heatmap marking cavitation and pressurized zones.

    Blue = pressurized, Red = cavitation.
    """
    theta_deg, s_mm = _build_axes(output)

    # cavitation_matrix: True where cavitation occurs
    cav = output.cavitation_matrix.astype(float)  # 1 = cavitation, 0 = pressurized

    Theta, S = np.meshgrid(theta_deg, s_mm, indexing="ij")

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = matplotlib.colors.ListedColormap(["steelblue", "tomato"])
    bounds = [-0.5, 0.5, 1.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    pcm = ax.pcolormesh(Theta, S, cav, cmap=cmap, norm=norm, shading="auto")

    cbar = fig.colorbar(pcm, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["Pressurized", "Cavitation"])

    ax.set_xlabel("Circumferential Angle (deg)")
    ax.set_ylabel("Axial Position (mm)")
    ax.set_title("Cavitation Map")
    ax.set_xlim(0, 360)

    _save_or_show(fig, save_path)


def plot_film_thickness(output: BearingOutput, save_path: str | None = None) -> None:
    """Film thickness distribution contour plot with annotation of the thinnest location."""
    theta_deg, s_mm = _build_axes(output)
    film_um = output.film_thickness_field_m * 1e6  # convert to micrometres

    Theta, S = np.meshgrid(theta_deg, s_mm, indexing="ij")

    fig, ax = plt.subplots(figsize=(10, 5))

    levels = 20
    cf = ax.contourf(Theta, S, film_um, levels=levels, cmap="viridis_r")
    cs = ax.contour(Theta, S, film_um, levels=levels,
                    colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    fig.colorbar(cf, ax=ax, label="Film Thickness (μm)")

    # Annotate minimum film thickness location
    theta_min, s_min = output.min_film_location  # (deg, m)
    s_min_mm = s_min * 1e3
    h_min_um = output.min_film_thickness_m * 1e6

    ax.plot(theta_min, s_min_mm, "r*", markersize=12, label=f"Min: {h_min_um:.2f} μm")
    ax.annotate(
        f"Min = {h_min_um:.2f} μm\n({theta_min:.1f}°, {s_min_mm:.2f} mm)",
        xy=(theta_min, s_min_mm),
        xytext=(theta_min + 20, s_min_mm),
        fontsize=8,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Circumferential Angle (deg)")
    ax.set_ylabel("Axial Position (mm)")
    ax.set_title("Oil Film Thickness Distribution")
    ax.set_xlim(0, 360)

    _save_or_show(fig, save_path)


def plot_pressure_profile(
    output: BearingOutput,
    axial_index: int | None = None,
    save_path: str | None = None,
) -> None:
    """Circumferential pressure profile at one axial cross-section.

    Parameters
    ----------
    output : BearingOutput
    axial_index : int | None
        Index of the axial slice to plot. Defaults to the mid-plane slice.
    save_path : str | None
    """
    theta_deg, s_mm = _build_axes(output)
    pressure_mpa = output.pressure_field_pa / 1e6

    Na = pressure_mpa.shape[1]
    if axial_index is None:
        axial_index = Na // 2

    axial_index = int(np.clip(axial_index, 0, Na - 1))
    profile = pressure_mpa[:, axial_index]
    s_label = s_mm[axial_index]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(theta_deg, profile, color="royalblue", linewidth=1.8)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.fill_between(theta_deg, profile, 0,
                    where=(profile > 0), alpha=0.25, color="royalblue",
                    label="Pressurized zone")
    ax.fill_between(theta_deg, profile, 0,
                    where=(profile <= 0), alpha=0.25, color="tomato",
                    label="Cavitation zone")

    ax.set_xlabel("Circumferential Angle (deg)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title(f"Circumferential Pressure Profile  (axial position = {s_label:.2f} mm)")
    ax.set_xlim(0, 360)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    _save_or_show(fig, save_path)


def plot_journal_center(output: BearingOutput, save_path: str | None = None) -> None:
    """Journal centre position plot — scatter within the bearing clearance circle.

    Shows the journal centre position relative to the bearing centre,
    normalised by the radial clearance.
    """
    inp = output.input_params
    radial_clearance = inp.clearance_m / 2.0

    # Recover eccentricity components from attitude angle and eccentricity ratio
    epsilon = inp.eccentricity_ratio
    if epsilon is None:
        # Derive from load capacity result
        epsilon = output.min_film_thickness_m  # fallback — use attitude angle
        # Better: compute from min film thickness
        # h_min = Cr(1 - epsilon)  =>  epsilon = 1 - h_min/Cr
        epsilon = 1.0 - output.min_film_thickness_m / radial_clearance

    attitude_rad = np.deg2rad(output.attitude_angle_deg)

    # Journal centre in Cartesian coordinates (normalised by Cr)
    ex_norm = epsilon * np.sin(attitude_rad)
    ey_norm = -epsilon * np.cos(attitude_rad)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw clearance circle
    phi = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(phi), np.sin(phi), "k-", linewidth=1.5, label="Clearance circle")

    # Bearing centre
    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2, label="Bearing centre")

    # Journal centre
    ax.plot(ex_norm, ey_norm, "ro", markersize=10, label="Journal centre")

    # Eccentricity vector
    ax.annotate(
        "",
        xy=(ex_norm, ey_norm),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # Annotations
    ax.text(
        ex_norm + 0.05, ey_norm + 0.05,
        f"ε = {epsilon:.3f}\nφ = {output.attitude_angle_deg:.1f}°",
        fontsize=9, color="red",
    )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal Eccentricity / Cr")
    ax.set_ylabel("Vertical Eccentricity / Cr")
    ax.set_title("Journal Centre Position")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)

    _save_or_show(fig, save_path)

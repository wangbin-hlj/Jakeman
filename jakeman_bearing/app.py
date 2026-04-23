"""
app.py — 船舶滑动轴承流体动力学分析 Web 界面

启动方式: streamlit run jakeman_bearing/app.py
"""

from __future__ import annotations

import io
import tempfile

import matplotlib
matplotlib.use("Agg")  # 必须在 import pyplot 之前设置

import matplotlib.pyplot as plt
import streamlit as st

from jakeman_bearing import analyze_bearing
from jakeman_bearing.visualization import (
    plot_cavitation_map,
    plot_film_thickness,
    plot_pressure_3d,
    plot_pressure_profile,
)

# ── 预设工况参数 ──────────────────────────────────────────────────────────────

PRESETS = {
    "论文 Table 1 — 曲轴轴承": {
        "diameter_mm": 63.5,
        "length_mm": 23.68,
        "clearance_um": 63.5,
        "speed_rpm": 2000.0,
        "viscosity": 0.014,
        "mode": "偏心比模式",
        "eccentricity_ratio": 0.6,
        "load_n": 10000.0,
        "misalignment_v_mrad": 0.0,
        "misalignment_h_mrad": 0.0,
        "groove_type": "circumferential_360",
        "groove_positions_str": "0",
        "groove_width_deg": 360.0,
        "supply_pressure_pa": 206700.0,
        "n_circumferential": 72,
        "n_axial": 10,
        "sor_factor": 1.7,
    },
    "论文 Table 3 — 尾管轴承": {
        "diameter_mm": 800.0,
        "length_mm": 1200.0,
        "clearance_um": 1400.0,
        "speed_rpm": 80.0,
        "viscosity": 0.125,
        "mode": "载荷模式",
        "eccentricity_ratio": 0.6,
        "load_n": 770150.0,
        "misalignment_v_mrad": 0.2,
        "misalignment_h_mrad": 0.0,
        "groove_type": "axial_dual",
        "groove_positions_str": "90, 270",
        "groove_width_deg": 30.0,
        "supply_pressure_pa": 0.0,
        "n_circumferential": 72,
        "n_axial": 14,
        "sor_factor": 1.7,
    },
    "论文 Table 2 Case 1 — 不对中尾管(ε=0.4, γ̄=0.369)": {
        "diameter_mm": 100.0,
        "length_mm": 100.0,
        "clearance_um": 100.0,
        "speed_rpm": 600.0,
        "viscosity": 0.01,
        "mode": "偏心比模式",
        "eccentricity_ratio": 0.4,
        "load_n": 10000.0,
        "misalignment_v_mrad": 0.369,  # γ̄ × Cd/L = 0.369 mrad
        "misalignment_h_mrad": 0.0,
        "groove_type": "axial_dual",
        "groove_positions_str": "90, 270",
        "groove_width_deg": 30.0,
        "supply_pressure_pa": 0.0,
        "n_circumferential": 72,
        "n_axial": 20,
        "sor_factor": 1.7,
    },
    "论文 Table 2 Case 2 — 不对中尾管(ε=0.4, γ̄=0.591)": {
        "diameter_mm": 100.0,
        "length_mm": 100.0,
        "clearance_um": 100.0,
        "speed_rpm": 600.0,
        "viscosity": 0.01,
        "mode": "偏心比模式",
        "eccentricity_ratio": 0.4,
        "load_n": 10000.0,
        "misalignment_v_mrad": 0.591,
        "misalignment_h_mrad": 0.0,
        "groove_type": "axial_dual",
        "groove_positions_str": "90, 270",
        "groove_width_deg": 30.0,
        "supply_pressure_pa": 0.0,
        "n_circumferential": 72,
        "n_axial": 20,
        "sor_factor": 1.7,
    },
    "论文 Table 2 Case 3 — 不对中尾管(ε=0.8, γ̄=0.112)": {
        "diameter_mm": 100.0,
        "length_mm": 100.0,
        "clearance_um": 100.0,
        "speed_rpm": 600.0,
        "viscosity": 0.01,
        "mode": "偏心比模式",
        "eccentricity_ratio": 0.8,
        "load_n": 10000.0,
        "misalignment_v_mrad": 0.112,
        "misalignment_h_mrad": 0.0,
        "groove_type": "axial_dual",
        "groove_positions_str": "90, 270",
        "groove_width_deg": 30.0,
        "supply_pressure_pa": 0.0,
        "n_circumferential": 72,
        "n_axial": 20,
        "sor_factor": 1.7,
    },
    "论文 Table 2 Case 4 — 不对中尾管(ε=0.8, γ̄=0.179)": {
        "diameter_mm": 100.0,
        "length_mm": 100.0,
        "clearance_um": 100.0,
        "speed_rpm": 600.0,
        "viscosity": 0.01,
        "mode": "偏心比模式",
        "eccentricity_ratio": 0.8,
        "load_n": 10000.0,
        "misalignment_v_mrad": 0.179,
        "misalignment_h_mrad": 0.0,
        "groove_type": "axial_dual",
        "groove_positions_str": "90, 270",
        "groove_width_deg": 30.0,
        "supply_pressure_pa": 0.0,
        "n_circumferential": 72,
        "n_axial": 20,
        "sor_factor": 1.7,
    },
    "自定义": None,
}

GROOVE_TYPE_OPTIONS = [
    "circumferential_360",
    "axial_dual",
    "axial_single",
    "none",
]

GROOVE_TYPE_LABELS = {
    "circumferential_360": "360° 环形槽",
    "axial_dual": "双轴向槽",
    "axial_single": "单轴向槽",
    "none": "无供油槽",
}


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _fig_to_png(fig: plt.Figure) -> bytes:
    """将 matplotlib Figure 渲染为 PNG 字节流。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def _capture_viz(viz_func, output, **kwargs) -> tuple[plt.Figure, bytes]:
    """调用可视化函数，捕获当前 Figure 并返回 (fig, png_bytes)。"""
    plt.close("all")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    viz_func(output, save_path=tmp_path, **kwargs)
    with open(tmp_path, "rb") as f:
        png = f.read()
    # Re-open as figure for potential reuse
    fig = plt.figure()
    return fig, png


def _result_to_csv_bytes(result) -> bytes:
    """将 BearingOutput 序列化为 CSV 字节流。"""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    result.to_csv(tmp_path)
    with open(tmp_path, "rb") as f:
        return f.read()


def _parse_groove_positions(s: str) -> list[float]:
    """解析逗号分隔的槽位置字符串，返回浮点列表。"""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


# ── 页面配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="船舶轴承分析",
    page_icon="🚢",
    layout="wide",
)

# ── 全局字体缩小 ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 全局字体缩小 */
html, body, [class*="css"] { font-size: 13px !important; }
h1 { font-size: 1.3rem !important; margin-bottom: 0.2rem !important; }
h2, h3 { font-size: 1.0rem !important; margin: 0.3rem 0 0.2rem 0 !important; }
/* metric 卡片紧凑 */
[data-testid="metric-container"] {
    padding: 4px 8px !important;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background: #fafafa;
}
[data-testid="metric-container"] label { font-size: 0.72rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-size: 1.0rem !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.68rem !important; }
/* 侧边栏字体 */
section[data-testid="stSidebar"] { font-size: 12px !important; }
section[data-testid="stSidebar"] label { font-size: 0.75rem !important; }
section[data-testid="stSidebar"] .stNumberInput input { font-size: 0.78rem !important; }
/* tab 字体 */
.stTabs [data-baseweb="tab"] { font-size: 0.78rem !important; padding: 4px 10px !important; }
/* divider 间距 */
hr { margin: 0.4rem 0 !important; }
/* expander 间距 */
.streamlit-expanderHeader { font-size: 0.8rem !important; }
/* 减少 subheader 上下间距 */
.stMarkdown p { margin: 0.1rem 0 !important; }
/* 图片不要额外边距 */
.stImage { margin: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("#### 🚢 船舶滑动轴承流体动力学分析 &nbsp;<small style='font-size:0.7rem;color:#888'>基于 Jakeman (1984) 流量连续性方法</small>", unsafe_allow_html=True)

# ── Session State 初始化 ──────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "last_preset" not in st.session_state:
    st.session_state.last_preset = None

# ── 侧边栏：预设工况选择器 ────────────────────────────────────────────────────

st.sidebar.header("⚙️ 输入参数")

preset_name = st.sidebar.selectbox(
    "📋 预设工况",
    list(PRESETS.keys()),
    key="preset_selector",
)

# 预设参数 key → 控件 key 的映射
_PARAM_TO_WIDGET = {
    "diameter_mm": "inp_diameter",
    "length_mm": "inp_length",
    "clearance_um": "inp_clearance",
    "speed_rpm": "inp_speed",
    "viscosity": "inp_viscosity",
    "mode": "inp_mode",
    "eccentricity_ratio": "inp_eccentricity",
    "load_n": "inp_load",
    "misalignment_v_mrad": "inp_mis_v",
    "misalignment_h_mrad": "inp_mis_h",
    "groove_type": "inp_groove_type",
    "groove_positions_str": "inp_groove_pos",
    "groove_width_deg": "inp_groove_width",
    "supply_pressure_pa": "inp_supply_pressure",
    "n_circumferential": "inp_nc",
    "n_axial": "inp_na",
    "sor_factor": "inp_sor",
}

# 当预设切换时，更新 session_state 中的默认值和控件值
if preset_name != st.session_state.last_preset:
    st.session_state.last_preset = preset_name
    if PRESETS[preset_name] is not None:
        for k, v in PRESETS[preset_name].items():
            st.session_state[f"param_{k}"] = v
            # 同时更新控件的 session_state key，确保 UI 立即刷新
            widget_key = _PARAM_TO_WIDGET.get(k)
            if widget_key:
                st.session_state[widget_key] = v
        st.rerun()

st.sidebar.divider()


def _get(key: str, default):
    """从 session_state 读取参数，不存在时返回 default。"""
    return st.session_state.get(f"param_{key}", default)


# ── 侧边栏：轴承几何 ──────────────────────────────────────────────────────────

with st.sidebar.expander("📐 轴承几何", expanded=True):
    diameter_mm = st.number_input(
        "轴承内径 D (mm)",
        min_value=1.0, max_value=5000.0,
        value=float(_get("diameter_mm", 63.5)),
        step=0.1, format="%.2f",
        key="inp_diameter",
    )
    diameter_err = st.empty()

    length_mm = st.number_input(
        "轴承长度 L (mm)",
        min_value=1.0, max_value=10000.0,
        value=float(_get("length_mm", 23.68)),
        step=0.1, format="%.2f",
        key="inp_length",
    )
    length_err = st.empty()

    clearance_um = st.number_input(
        "直径间隙 Cd (μm)",
        min_value=1.0, max_value=10000.0,
        value=float(_get("clearance_um", 63.5)),
        step=0.5, format="%.1f",
        key="inp_clearance",
    )
    clearance_err = st.empty()

# ── 侧边栏：工况参数 ──────────────────────────────────────────────────────────

with st.sidebar.expander("⚙️ 工况参数", expanded=True):
    speed_rpm = st.number_input(
        "转速 (rpm)",
        min_value=0.1, max_value=100000.0,
        value=float(_get("speed_rpm", 2000.0)),
        step=10.0, format="%.1f",
        key="inp_speed",
    )
    speed_err = st.empty()

    viscosity = st.number_input(
        "动力粘度 η (Pa·s)",
        min_value=0.0001, max_value=100.0,
        value=float(_get("viscosity", 0.014)),
        step=0.001, format="%.4f",
        key="inp_viscosity",
    )
    viscosity_err = st.empty()

# ── 侧边栏：偏心/载荷模式 ─────────────────────────────────────────────────────

with st.sidebar.expander("🎯 偏心 / 载荷模式", expanded=True):
    mode = st.radio(
        "分析模式",
        ["偏心比模式", "载荷模式"],
        index=0 if _get("mode", "偏心比模式") == "偏心比模式" else 1,
        key="inp_mode",
        horizontal=True,
    )

    if mode == "偏心比模式":
        eccentricity_ratio = st.slider(
            "偏心比 ε",
            min_value=0.01, max_value=0.99,
            value=float(_get("eccentricity_ratio", 0.6)),
            step=0.01, format="%.2f",
            key="inp_eccentricity",
        )
        ecc_err = st.empty()
        load_n = None
    else:
        load_n = st.number_input(
            "目标载荷 W (N)",
            min_value=1.0, max_value=1e9,
            value=float(_get("load_n", 10000.0)),
            step=100.0, format="%.1f",
            key="inp_load",
        )
        load_err = st.empty()
        eccentricity_ratio = None

# ── 侧边栏：不对中参数 ────────────────────────────────────────────────────────

with st.sidebar.expander("📏 不对中参数", expanded=False):
    misalignment_v_mrad = st.number_input(
        "垂直面倾斜角 γ (mrad)",
        min_value=-10.0, max_value=10.0,
        value=float(_get("misalignment_v_mrad", 0.0)),
        step=0.01, format="%.3f",
        key="inp_mis_v",
    )
    misalignment_h_mrad = st.number_input(
        "水平面倾斜角 λ (mrad)",
        min_value=-10.0, max_value=10.0,
        value=float(_get("misalignment_h_mrad", 0.0)),
        step=0.01, format="%.3f",
        key="inp_mis_h",
    )

# ── 侧边栏：供油槽配置 ────────────────────────────────────────────────────────

with st.sidebar.expander("🛢️ 供油槽配置", expanded=False):
    groove_type_key = _get("groove_type", "circumferential_360")
    groove_type_idx = GROOVE_TYPE_OPTIONS.index(groove_type_key) if groove_type_key in GROOVE_TYPE_OPTIONS else 0

    groove_type = st.selectbox(
        "槽类型",
        GROOVE_TYPE_OPTIONS,
        index=groove_type_idx,
        format_func=lambda x: GROOVE_TYPE_LABELS[x],
        key="inp_groove_type",
    )

    if groove_type != "none":
        groove_positions_str = st.text_input(
            "槽角度位置 (°, 逗号分隔)",
            value=str(_get("groove_positions_str", "0")),
            key="inp_groove_pos",
            help="例如: 0  或  90, 270",
        )
        groove_pos_err = st.empty()

        groove_width_deg = st.number_input(
            "槽张角 (°)",
            min_value=1.0, max_value=360.0,
            value=float(_get("groove_width_deg", 360.0)),
            step=1.0, format="%.1f",
            key="inp_groove_width",
        )

        supply_pressure_pa = st.number_input(
            "供油压力 (Pa)",
            min_value=0.0, max_value=1e8,
            value=float(_get("supply_pressure_pa", 206700.0)),
            step=1000.0, format="%.0f",
            key="inp_supply_pressure",
        )
    else:
        groove_positions_str = "0"
        groove_width_deg = 360.0
        supply_pressure_pa = 0.0

# ── 侧边栏：高级参数 ──────────────────────────────────────────────────────────

with st.sidebar.expander("🔧 高级参数", expanded=False):
    n_circumferential = st.slider(
        "周向网格数 Mc",
        min_value=12, max_value=144,
        value=int(_get("n_circumferential", 72)),
        step=4,
        key="inp_nc",
    )
    n_axial = st.slider(
        "轴向网格数 Na",
        min_value=4, max_value=40,
        value=int(_get("n_axial", 10)),
        step=2,
        key="inp_na",
    )
    sor_factor = st.slider(
        "SOR 松弛因子",
        min_value=1.0, max_value=1.99,
        value=float(_get("sor_factor", 1.7)),
        step=0.05, format="%.2f",
        key="inp_sor",
    )

st.sidebar.divider()

# ── 侧边栏：计算按钮 ──────────────────────────────────────────────────────────

run_button = st.sidebar.button(
    "🚀 开始计算",
    type="primary",
    use_container_width=True,
)

# ── 输入验证 ──────────────────────────────────────────────────────────────────

validation_errors: list[str] = []

if diameter_mm <= 0:
    diameter_err.error("轴承内径必须为正数")
    validation_errors.append("diameter")

if length_mm <= 0:
    length_err.error("轴承长度必须为正数")
    validation_errors.append("length")

if clearance_um <= 0:
    clearance_err.error("直径间隙必须为正数")
    validation_errors.append("clearance")

if speed_rpm <= 0:
    speed_err.error("转速必须为正数")
    validation_errors.append("speed")

if viscosity <= 0:
    viscosity_err.error("粘度必须为正数")
    validation_errors.append("viscosity")

if mode == "偏心比模式" and not (0 < eccentricity_ratio < 1):
    ecc_err.error("偏心比必须在开区间 (0, 1) 内")
    validation_errors.append("eccentricity")

if groove_type != "none":
    try:
        _parse_groove_positions(groove_positions_str)
    except ValueError:
        groove_pos_err.error("槽位置格式无效，请输入逗号分隔的数字，例如: 90, 270")
        validation_errors.append("groove_pos")

# ── 执行计算 ──────────────────────────────────────────────────────────────────

if run_button:
    if validation_errors:
        st.error(f"请修正以上 {len(validation_errors)} 处输入错误后再计算。")
    else:
        try:
            groove_positions = _parse_groove_positions(groove_positions_str) if groove_type != "none" else None

            with st.spinner("正在迭代求解，请稍候..."):
                result = analyze_bearing(
                    diameter=diameter_mm / 1000.0,
                    length=length_mm / 1000.0,
                    clearance=clearance_um / 1e6,
                    speed_rps=speed_rpm / 60.0,
                    viscosity=viscosity,
                    eccentricity_ratio=eccentricity_ratio,
                    load=load_n,
                    misalignment_vertical=misalignment_v_mrad / 1000.0,
                    misalignment_horizontal=misalignment_h_mrad / 1000.0,
                    groove_type=groove_type,
                    groove_positions=groove_positions,
                    groove_width=groove_width_deg,
                    supply_pressure=supply_pressure_pa,
                    n_circumferential=n_circumferential,
                    n_axial=n_axial,
                    over_relaxation_factor=sor_factor,
                )
            st.session_state.result = result

            if not result.converged:
                st.warning(
                    f"⚠️ 求解器未完全收敛（残差 = {result.residual:.3e}，"
                    f"迭代 {result.iterations} 次）。结果仅供参考。"
                )

        except (ValueError, RuntimeError) as exc:
            st.error(f"计算失败：{exc}")
            st.session_state.result = None

# ── 主区域：结果展示 ──────────────────────────────────────────────────────────

result = st.session_state.result

if result is not None:
    # ── 性能参数：8个指标紧凑排成一行 ──────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("承载力", f"{result.load_capacity_n:.0f} N")
    c2.metric("偏位角", f"{result.attitude_angle_deg:.1f}°")
    c3.metric("最小膜厚", f"{result.min_film_thickness_m * 1e6:.1f} μm")
    c4.metric("功率损耗", f"{result.power_loss_w:.0f} W")
    c5.metric("侧漏流量", f"{result.side_leakage_flow_m3s:.2e} m³/s")
    c6.metric("垂直力矩", f"{result.moment_vertical_nm:.1f} N·m")
    c7.metric("水平力矩", f"{result.moment_horizontal_nm:.1f} N·m")
    c8.metric(
        "收敛",
        "✅" if result.converged else "⚠️",
        delta=f"{result.iterations}次 {result.residual:.1e}",
    )

    # ── 预渲染所有图表 ────────────────────────────────────────────────────────
    with st.spinner("渲染图表..."):
        _, png1 = _capture_viz(plot_pressure_3d, result)
        _, png2 = _capture_viz(plot_cavitation_map, result)
        _, png3 = _capture_viz(plot_film_thickness, result)
        na = result.pressure_field_pa.shape[1]
        axial_idx = st.session_state.get("axial_slice", na // 2)
        _, png4 = _capture_viz(plot_pressure_profile, result, axial_index=axial_idx)

    # ── 图表 + 下载：左右布局 ────────────────────────────────────────────────
    chart_col, dl_col = st.columns([5, 1])

    with chart_col:
        tab1, tab2, tab3, tab4 = st.tabs([
            "压力分布 3D", "空化区域", "油膜厚度", "截面压力",
        ])
        with tab1:
            st.image(png1, use_container_width=True)
        with tab2:
            st.image(png2, use_container_width=True)
        with tab3:
            st.image(png3, use_container_width=True)
        with tab4:
            new_idx = st.slider(
                "轴向截面", min_value=0, max_value=na - 1,
                value=axial_idx, key="axial_slice",
            )
            if new_idx != axial_idx:
                _, png4 = _capture_viz(plot_pressure_profile, result, axial_index=new_idx)
            st.image(png4, use_container_width=True)

    with dl_col:
        st.markdown("**下载**")
        csv_bytes = _result_to_csv_bytes(result)
        st.download_button("📥 CSV", data=csv_bytes,
                           file_name="bearing_result.csv", mime="text/csv",
                           key="dl_csv", use_container_width=True)
        st.download_button("📥 压力图", data=png1,
                           file_name="pressure_3d.png", mime="image/png",
                           key="dl_p3d", use_container_width=True)
        st.download_button("📥 空化图", data=png2,
                           file_name="cavitation_map.png", mime="image/png",
                           key="dl_cav", use_container_width=True)
        st.download_button("📥 膜厚图", data=png3,
                           file_name="film_thickness.png", mime="image/png",
                           key="dl_film", use_container_width=True)
        st.download_button("📥 截面图", data=png4,
                           file_name="pressure_profile.png", mime="image/png",
                           key="dl_prof", use_container_width=True)
        st.markdown("---")
        with st.expander("📋 摘要"):
            st.text(result.summary())

else:
    st.info("👈 在左侧边栏选择预设工况或输入参数，点击 **🚀 开始计算**。")

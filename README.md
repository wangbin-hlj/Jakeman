# 船舶滑动轴承流体动力学分析程序

基于 Jakeman (1984) 流量连续性方法的船舶滑动轴承水动力润滑分析工具，提供 Web 交互界面和 Python API。

> **参考文献**: Jakeman, R.W. (1984). *A numerical analysis method based on flow continuity for hydrodynamic journal bearings*. Tribology International.

## 功能

- Jakeman 流量连续性方法，空化边界自动确定，严格保证质量守恒
- 支持轴线不对中分析（垂直面 γ、水平面 λ）
- 多种供油槽配置：360° 环形槽、双轴向槽、单轴向槽
- 偏心模式（给定 ε）和载荷模式（给定 W，Brent 法反求 ε）
- 自适应 SOR 松弛因子，自动处理各种工况的收敛问题
- 相对收敛判据，不同尺寸轴承精度一致
- 动态刚度/阻尼系数计算
- Streamlit Web 界面（主要使用方式）
- 完整的可视化（3D 压力图、空化图、油膜厚度图等）

---

## 环境要求

- **Python 3.10 或更高版本**
- 操作系统：Windows / macOS / Linux 均可

## 部署安装

### 第一步：安装 Python

如果系统没有 Python，从 [python.org](https://www.python.org/downloads/) 下载安装。安装时勾选 "Add Python to PATH"。

验证安装：
```bash
python --version
```
应显示 `Python 3.10.x` 或更高版本。

### 第二步：从 GitHub 下载程序

**方式一：git 克隆（推荐，方便后续更新）**

```bash
git clone https://github.com/wangbin-hlj/Jakeman.git
cd Jakeman
```

**方式二：直接下载 ZIP**

1. 打开 [https://github.com/wangbin-hlj/Jakeman](https://github.com/wangbin-hlj/Jakeman)
2. 点击绿色 **Code** 按钮 → **Download ZIP**
3. 解压到任意目录，进入解压后的文件夹

### 第三步：安装依赖

```bash
python -m pip install -r requirements.txt
```

如果 `pip` 命令找不到，用 `python -m pip` 代替。

依赖项：
| 包 | 用途 |
|---|------|
| numpy | 数值计算 |
| matplotlib | 图表绘制 |
| scipy | Brent 法优化（载荷模式） |
| streamlit >= 1.30.0 | Web 界面 |
| pytest | 测试框架（开发用） |
| hypothesis | 属性测试（开发用） |

### 第四步：验证安装

```bash
python -c "from jakeman_bearing import analyze_bearing; r = analyze_bearing(); print(f'承载力: {r.load_capacity_n:.1f} N, 收敛: {r.converged}')"
```

应输出类似 `承载力: 669.5 N, 收敛: True`。

---

## 运行 Web 界面

这是主要的使用方式，不需要写任何代码。

### 启动

```bash
python -m streamlit run jakeman_bearing/app.py
```

启动后浏览器会自动打开 `http://localhost:8501`。如果没有自动打开，手动在浏览器输入该地址。

### 使用流程

1. **选择预设工况**（左侧边栏顶部下拉框）— 选择后所有参数自动填充
   - 论文 Table 1 — 曲轴轴承（默认，偏心模式）
   - 论文 Table 2 Case 1~4 — 不对中尾管轴承（偏心模式）
   - 论文 Table 3 — 尾管轴承实例（载荷模式）
   - 自定义 — 输入自己的轴承参数
2. **调整参数**（可选）— 在各分组中修改需要的参数
3. **点击"🚀 开始计算"** — 等待求解完成
4. **查看结果** — 上方 8 个性能指标卡片 + 下方图表标签页（压力分布、空化区域、油膜厚度、截面压力）
5. **下载数据** — 右侧下载 CSV 数据文件或各图表 PNG

### 分析模式

- **偏心比模式**：直接指定偏心比 ε，用于参数研究和论文验证
- **载荷模式**：指定目标载荷 W（N），程序自动反求偏心比。实际工程设计中最常用

### 停止 Web 服务

在启动 Web 界面的终端窗口按 `Ctrl+C`。

---

## Python API 使用

适合批量计算、参数扫描或集成到其他程序中。

### 简单接口

```python
from jakeman_bearing import analyze_bearing

# 零参数 — 复现论文 Table 1 Case 1
result = analyze_bearing()
print(result.summary())

# 尾管轴承载荷模式（工程设计常用）
result = analyze_bearing(
    diameter=0.8,              # 轴承内径 800mm
    length=1.2,                # 轴承长度 1200mm
    clearance=0.0014,          # 直径间隙 1.4mm
    speed_rps=80/60,           # 转速 80rpm
    viscosity=0.125,           # 粘度 0.125 Pa·s
    eccentricity_ratio=None,   # 不指定偏心比
    load=770000,               # 目标载荷 770kN
    misalignment_vertical=0.0002,
    groove_type="axial_dual",
    groove_positions=[90, 270],
    groove_width=30,
    supply_pressure=0,
)
print(f"反求偏心比: {result.input_params.eccentricity_ratio:.4f}")
print(f"最小油膜厚度: {result.min_film_thickness_m*1e3:.3f} mm")
```

### analyze_bearing() 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `diameter` | 0.0635 | 轴承内径 (m) |
| `length` | 0.02368 | 轴承长度 (m) |
| `clearance` | 0.0000635 | 直径间隙 Cd (m) |
| `speed_rps` | 33.33 | 转速 (r/s) |
| `viscosity` | 0.014 | 动力粘度 (Pa·s) |
| `eccentricity_ratio` | 0.6 | 偏心比 ε，与 load 互斥 |
| `load` | None | 目标载荷 (N)，与 eccentricity_ratio 互斥 |
| `load_direction` | 270.0 | 载荷方向 (°)，270°=正下方 |
| `misalignment_vertical` | 0.0 | 垂直面不对中角 γ (rad) |
| `misalignment_horizontal` | 0.0 | 水平面不对中角 λ (rad) |
| `groove_type` | "circumferential_360" | 供油槽类型 |
| `groove_positions` | None | 槽角度位置列表 (°) |
| `groove_width` | 360.0 | 槽张角 (°) |
| `supply_pressure` | 206700.0 | 供油压力 (Pa) |
| `n_circumferential` | 72 | 周向网格数 |
| `n_axial` | 10 | 轴向网格数 |

供油槽类型：`"circumferential_360"`（环形槽）、`"axial_dual"`（双轴向槽）、`"axial_single"`（单轴向槽）、`"none"`（无）

### 结果对象 BearingOutput

| 属性 | 说明 |
|------|------|
| `load_capacity_n` | 合成承载力 (N) |
| `attitude_angle_deg` | 偏位角 (°) |
| `moment_vertical_nm` | 垂直力矩 My (N·m) |
| `min_film_thickness_m` | 最小油膜厚度 (m) |
| `side_leakage_flow_m3s` | 侧漏流量 (m³/s) |
| `power_loss_w` | 功率损耗 (W) |
| `pressure_field_pa` | 压力场数组 (Mc×Na) |
| `cavitation_matrix` | 空化标记 (bool) |
| `converged` | 是否收敛 |
| `iterations` | 迭代次数 |
| `residual` | 最终相对残差 |

```python
result.summary()           # 文本摘要
result.to_csv("out.csv")   # 保存 CSV
```

### 可视化

```python
from jakeman_bearing.visualization import (
    plot_pressure_3d,
    plot_pressure_contour,
    plot_cavitation_map,
    plot_film_thickness,
    plot_pressure_profile,
    plot_journal_center,
)

plot_pressure_3d(result, save_path="pressure.png")
plot_cavitation_map(result)  # 不传 save_path 则直接显示
```

### 高级接口

```python
from jakeman_bearing.bearing_models import BearingInput, GrooveConfig
from jakeman_bearing.bearing_solver import BearingSolver

groove = GrooveConfig("axial_dual", [90, 270], 30, 101325)
inp = BearingInput(
    diameter_m=0.2, length_m=0.2, clearance_m=0.00028,
    speed_rps=5.0, viscosity_pa_s=0.05,
    eccentricity_ratio=0.5, groove=groove,
)
inp.validate()
solver = BearingSolver(inp)
result = solver.solve()
```

---

## 示例脚本

```bash
python examples/example1_aligned_bearing.py      # 对齐轴承
python examples/example2_misaligned_bearing.py    # 不对中轴承
python examples/example3_load_mode.py             # 载荷模式
python examples/example4_sterntube_table3.py      # 论文 Table 3
```

---

## 运行测试

```bash
python -m pytest jakeman_bearing/tests/           # 全部 186 个测试
python -m pytest jakeman_bearing/tests/ -m "not slow"  # 跳过慢速测试
python -m pytest jakeman_bearing/tests/test_validation.py -v  # 论文验证
```

---

## 项目结构

```
Jakeman/
├── jakeman_bearing/
│   ├── __init__.py              # 公共接口 analyze_bearing()
│   ├── bearing_models.py        # 数据模型
│   ├── bearing_geometry.py      # 油膜几何计算
│   ├── bearing_solver.py        # 核心求解器（自适应 SOR 迭代）
│   ├── bearing_postprocess.py   # 后处理（承载力、流量、功率、动态系数）
│   ├── bearing_practical.py     # 载荷模式（Brent 法）
│   ├── visualization.py         # Matplotlib 可视化
│   ├── app.py                   # Streamlit Web 界面
│   └── tests/                   # 测试（186 个测试用例）
├── examples/                    # 示例脚本
├── requirements.txt             # Python 依赖
├── 程序设计说明书.md              # 算法原理详细说明
└── README.md                    # 本文件
```

---

## 求解器特性

| 特性 | 说明 |
|------|------|
| 收敛判据 | 相对残差（残差/压力场最大值），不同尺寸轴承精度一致 |
| 默认容差 | 1×10⁻⁴（0.01% 相对精度） |
| 自适应 SOR | 检测收敛停滞时自动降低松弛因子，确保各种工况收敛 |
| 载荷模式 | Brent 法自动搜索偏心比，自动处理不对中轴承的搜索上界 |
| 默认迭代上限 | 10000 次 |

---

## 常见问题

**Q: `streamlit` 命令找不到？**
用 `python -m streamlit run jakeman_bearing/app.py` 代替。

**Q: `pip` 命令找不到？**
用 `python -m pip install -r requirements.txt` 代替。

**Q: 计算很慢？**
减小网格数（周向 36~72，轴向 8~14）可以加快计算。载荷模式比偏心模式慢（需要多次迭代搜索偏心比）。

**Q: 求解器未收敛？**
程序有自适应松弛因子，大多数情况会自动处理。如果仍未收敛，尝试在高级参数中减小松弛因子（1.3~1.5）。

**Q: 载荷模式报错"载荷过大"？**
目标载荷超出了该轴承配置的承载能力上限。检查几何参数、转速和粘度是否合理。不对中轴承的承载能力上限低于对齐轴承。

**Q: 预设工况的作用？**
预设工况来自 Jakeman (1984) 论文，用于快速验证求解器正确性，同时为工程师提供输入参数的参考。实际工程计算时选择"自定义"输入自己的轴承参数。

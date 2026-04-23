# 实现计划：滑动轴承流体动力学分析程序（Jakeman 流量连续性方法）

## 概述

基于设计文档的四阶段开发计划，将 Jakeman (1984) 流量连续性方法实现为 Python 包 `jakeman_bearing/`。每个阶段递增构建，确保核心算法优先实现并通过论文数据验证。

## 任务

- [x] 1. 搭建项目结构与核心数据模型
  - [x] 1.1 创建项目目录结构与包入口
    - 创建 `jakeman_bearing/` 包目录及 `__init__.py`
    - 创建 `jakeman_bearing/tests/` 测试目录及 `__init__.py`
    - _需求: 12.1, 12.3_

  - [x] 1.2 实现 `bearing_models.py` — 数据模型
    - 实现 `GrooveConfig` dataclass：groove_type, angular_positions_deg, angular_width_deg, supply_pressure_pa, axial_position_ratio, axial_width_ratio
    - 实现 `BearingInput` dataclass：几何参数、工况参数、偏心/载荷模式、不对中参数、压力参数、供油槽、网格参数、求解器参数
    - 实现 `BearingInput.validate()` 方法：偏心比范围检查、几何参数正值检查、偏心/载荷互斥检查、网格数下限检查、SOR因子范围检查，错误信息包含参数名和有效范围
    - 实现 `BearingOutput` dataclass：压力场、空化标记、油膜厚度场、承载力、力矩、油膜、流量、功率、动态系数、求解器信息、输入回显
    - 实现 `BearingOutput.summary()` 方法：输出包含所有性能参数的文本摘要
    - 实现 `BearingOutput.to_csv()` 方法：保存压力场和性能参数为 CSV，精度满足往返误差 ≤ 1e-10
    - _需求: 1.1–1.12, 9.5, 11.1–11.4_

  - [x] 1.3 编写属性测试 — 输入验证（Property 1）
    - **Property 1: 输入验证拒绝无效参数**
    - 使用 hypothesis 生成无效偏心比（≤0 或 ≥1）和非正几何参数，验证 validate() 抛出 ValueError 且错误信息包含参数名
    - **验证: 需求 1.10, 1.11**

  - [x] 1.4 编写属性测试 — CSV 往返精度（Property 15）
    - **Property 15: CSV 往返精度**
    - 使用 hypothesis 生成随机 float64 数组，验证 to_csv() 保存后重新读取的最大绝对误差 ≤ 1e-10
    - **验证: 需求 11.3, 11.4**

  - [x] 1.5 编写属性测试 — 结果摘要完整性（Property 16）
    - **Property 16: 结果摘要完整性**
    - 验证 summary() 返回字符串包含承载力、偏位角、力矩、最小油膜厚度、功率损耗、侧漏流量、输入参数回显、迭代次数和收敛状态
    - **验证: 需求 11.1, 11.2**

  - [x] 1.6 编写单元测试 — 数据模型
    - 在 `tests/test_models.py` 中测试 BearingInput/BearingOutput/GrooveConfig 的构造、默认值和验证逻辑
    - _需求: 1.1–1.12_

- [x] 2. 实现油膜几何计算模块
  - [x] 2.1 实现 `bearing_geometry.py`
    - 实现 `compute_eccentricity_components()`：根据公式(4-5)计算轴向各位置偏心分量 esy(s)=ecy+s×γ, esx(s)=ecx+s×λ
    - 实现 `compute_film_thickness()`：根据公式(1)计算油膜厚度 h=Cd/2+e(s)×cos(θ-ψ(s))
    - 实现 `compute_element_corner_thicknesses()`：计算每个网格元素四角油膜厚度 ha, hb, hc, hd
    - _需求: 2.1–2.6_

  - [x] 2.2 编写属性测试 — 偏心分量线性计算（Property 2）
    - **Property 2: 偏心分量线性计算**
    - 使用 hypothesis 生成随机 ecy, ecx, γ, λ, s 值，验证 esy(s)=ecy+s×γ, esx(s)=ecx+s×λ, e(s)=sqrt(esy²+esx²), ψ(s)=atan2(esx,esy)
    - **验证: 需求 2.1, 2.2**

  - [x] 2.3 编写属性测试 — 网格四角油膜厚度公式（Property 3）
    - **Property 3: 网格四角油膜厚度公式**
    - 验证 ha=h(θ_J,s_I), hb=h(θ_J,s_{I+1}), hc=h(θ_{J+1},s_I), hd=h(θ_{J+1},s_{I+1})
    - **验证: 需求 2.3, 2.4**

  - [x] 2.4 编写属性测试 — 油膜厚度正值不变量（Property 4）
    - **Property 4: 油膜厚度正值不变量**
    - 对偏心比在 (0,1) 内的有效参数，验证所有油膜厚度值严格大于零
    - **验证: 需求 2.5**

  - [x] 2.5 编写属性测试 — 对齐轴承退化（Property 5）
    - **Property 5: 对齐轴承退化**
    - 当 γ=0, λ=0 时，验证同一周向位置的油膜厚度在轴向方向上为常数
    - **验证: 需求 2.6**

  - [x] 2.6 编写单元测试 — 几何计算
    - 在 `tests/test_geometry.py` 中用已知参数手算对比油膜厚度
    - _需求: 2.1–2.6_

- [x] 3. 实现核心求解器
  - [x] 3.1 实现 `bearing_solver.py` — BearingSolver 类框架
    - 实现 `__init__()`：接受 BearingInput，调用 _setup_grid 和 _setup_groove_mask
    - 实现 `_setup_grid()`：生成 θ 坐标（周期性 0°~360°）、s 坐标（-L/2~+L/2）、Δa（支持均匀和加密网格）、Δc
    - 实现 `_setup_groove_mask()`：根据 GrooveConfig 生成供油槽掩码矩阵，支持 circumferential_360、axial_dual、axial_single 三种类型
    - _需求: 3.1, 3.4–3.6, 9.1–9.4_

  - [x] 3.2 编写属性测试 — 网格生成正确性（Property 17）
    - **Property 17: 网格生成正确性**
    - 验证 θ_J=(J-0.5)×360°/Mc；grading_factor=1.0 时 Δa 相等；grading_factor>1.0 时端部 Δa 小于中部
    - **验证: 需求 9.1, 9.2**

  - [x] 3.3 实现 `bearing_solver.py` — H 系数与 K 流量计算
    - 实现 `_compute_H_coefficients()`：计算 Hci, Hai, Hco, Hao（公式8-11），边界相邻网格 H 系数 ×2
    - 实现 `_compute_K_flow()`：计算速度诱导流量 K（公式12-16），稳态 K=(hc+hd-ha-hb)×U×Δa/4
    - _需求: 3.2, 3.3, 3.7_

  - [x] 3.4 编写属性测试 — H 系数公式（Property 6）
    - **Property 6: 压力流函数 H 系数公式**
    - 使用 hypothesis 生成随机正的 h, Δa, Δc, η 值，验证 Hci=(ha+hb)³×Δa/(96×η×Δc) 等
    - **验证: 需求 3.2**

  - [x] 3.5 编写属性测试 — K 流量公式（Property 7）
    - **Property 7: 速度诱导流量 K 公式**
    - 验证 K=(hc+hd-ha-hb)×U×Δa/4
    - **验证: 需求 3.3**

  - [x] 3.6 实现 `bearing_solver.py` — SOR 迭代求解核心 `solve()` 方法
    - 调用 bearing_geometry 计算 ha, hb, hc, hd
    - 计算 H 系数和 K 流量
    - SOR 迭代循环：对每个非供油槽网格用公式(7)计算 P_new；空化判断（P_new≤Pc 时标记空化、设 P=Pc、计算 Qvo 公式17）；正常承压区 Qvo=0；Qvo 传递给下游网格作为 Qvi；SOR 松弛
    - 周向周期性边界条件（J=1 上游连接 J=Mc）
    - 轴向两端环境压力边界条件
    - 收敛判据：max|P_new - P_old| < tol
    - 未收敛时返回 BearingOutput（converged=False）附带警告
    - 数值异常检测（NaN/Inf → RuntimeError）
    - _需求: 3.1, 3.4–3.11, 4.1–4.6_

  - [x] 3.7 编写属性测试 — 供油槽压力不变量（Property 8）
    - **Property 8: 供油槽压力不变量**
    - 求解完成后，验证所有供油槽网格压力精确等于 Ps
    - **验证: 需求 3.6**

  - [x] 3.8 编写属性测试 — 压力-空化一致性（Property 9）
    - **Property 9: 压力-空化一致性**
    - 验证：空化网格压力=Pc；压力>Pc 的网格未标记空化；空化标记与压力值完全一致
    - **验证: 需求 4.1, 4.2, 4.5**

  - [x] 3.9 编写单元测试 — 求解器
    - 在 `tests/test_solver.py` 中测试简单工况的压力场求解（如短轴承近似解对比）
    - _需求: 3.1–3.11, 4.1–4.6_

- [x] 4. 检查点 — 核心算法验证
  - 确保所有已实现的测试通过，如有问题请向用户确认。

- [x] 5. 实现后处理模块
  - [x] 5.1 实现 `bearing_postprocess.py` — 承载力与力矩
    - 实现 `compute_load_capacity()`：12点加权平均法计算 P_mean，积分计算 Fy, Fx, F_total, attitude_angle_deg（公式18）
    - 实现 `compute_moments()`：计算力矩 My, Mx
    - _需求: 5.1–5.5_

  - [x] 5.2 编写属性测试 — 12点加权平均承载力（Property 10）
    - **Property 10: 12点加权平均承载力**
    - 使用 hypothesis 生成随机压力场，验证 P_mean=(4P+ΣP_neighbors)/12，Fy=Σ(-P_mean×Δa×Δc×cos(θ))，Fx=Σ(-P_mean×Δa×Δc×sin(θ))
    - **验证: 需求 5.1, 5.2, 5.3**

  - [x] 5.3 编写属性测试 — 力矩计算（Property 11）
    - **Property 11: 力矩计算**
    - 验证 My=Σ(dFy×s_I), Mx=Σ(dFx×s_I)
    - **验证: 需求 5.4**

  - [x] 5.4 编写属性测试 — 最小油膜厚度识别（Property 12）
    - **Property 12: 最小油膜厚度识别**
    - 验证 min_film_thickness_m 等于厚度场全局最小值，min_film_location 对应正确位置
    - **验证: 需求 5.5**

  - [x] 5.5 实现 `bearing_postprocess.py` — 流量与功率损耗
    - 实现 `compute_side_leakage()`：计算侧漏流量 Qs=Σ|Qai|（公式19-20）
    - 实现 `compute_power_loss()`：计算功率损耗 H=U×ΣFc，包括有效油膜宽度 Bai/Bao 和摩擦力 Fc（公式21-24）
    - _需求: 6.1–6.4_

  - [x] 5.6 编写属性测试 — 侧漏流量计算（Property 13）
    - **Property 13: 侧漏流量计算**
    - 验证侧漏流量等于轴承两端面轴向流量绝对值之和
    - **验证: 需求 6.1, 6.2**

  - [x] 5.7 编写属性测试 — 功率损耗计算（Property 14）
    - **Property 14: 功率损耗计算**
    - 验证总功率损耗 H=U×ΣFc
    - **验证: 需求 6.3, 6.4**

  - [x] 5.8 编写单元测试 — 后处理
    - 在 `tests/test_postprocess.py` 中用已知压力场手算对比承载力/流量/功率
    - _需求: 5.1–5.5, 6.1–6.4_

- [x] 6. 实现论文数据验证测试
  - [x] 6.1 实现 `tests/test_validation.py` — 论文 Table 1 验证
    - 对齐曲轴轴承（D=63.5mm, L=23.68mm, 360°环形槽, Ps=0.2067MPa）
    - 验证承载力误差 < 3%，偏位角误差 < 2°
    - _需求: 14.1_

  - [x] 6.2 实现 `tests/test_validation.py` — 论文 Table 2 验证
    - 不对中尾管轴承（L/D=1，双轴向槽 90°/270°，张角30°），4组工况
    - _需求: 14.2_

  - [x] 6.3 实现 `tests/test_validation.py` — 论文 Table 3 验证
    - 尾管轴承实例（D=800mm, L=1200mm, Cd=1.4mm, 80rpm, η=0.125Pa·s, γ=0.0002rad）
    - 目标：W≈770,150N, M≈6.591×10⁷ N·mm
    - _需求: 14.3_

- [x] 7. 检查点 — 后处理与论文验证
  - 确保所有已实现的测试通过，如有问题请向用户确认。

- [x] 8. 实现实用分析模式与动态系数
  - [x] 8.1 实现 `bearing_practical.py` — 给定载荷求偏心
    - 实现 `solve_for_load()`：Brent 法在 [0.01, 0.99] 区间搜索偏心比零点
    - 目标函数 f(ε)=F_computed-F_target
    - 同时迭代偏位角以匹配载荷方向
    - 载荷超出承载能力时抛出 ValueError
    - Brent 法未收敛时抛出 RuntimeError
    - _需求: 8.1–8.5_

  - [x] 8.2 编写属性测试 — Brent 法载荷匹配（Property 20）
    - **Property 20: Brent 法载荷匹配**
    - 验证在承载能力范围内的目标载荷，|W_computed-W_target|/W_target < 收敛容差
    - **验证: 需求 8.3**

  - [x] 8.3 编写单元测试 — 实用模式
    - 在 `tests/test_practical.py` 中测试 Brent 法端到端载荷反求偏心
    - _需求: 8.1–8.5_

  - [x] 8.4 实现 `bearing_postprocess.py` — 动态刚度/阻尼系数
    - 实现 `compute_dynamic_coefficients()`：微小扰动差分法
    - 实现 `BearingSolver.solve_perturbed()`：求解扰动后压力场
    - 对齐轴承：8个系数（2×2 刚度 + 2×2 阻尼）
    - 不对中轴承：32个系数（4×4 刚度 + 4×4 阻尼）
    - Aij=(Fi(+δ)-Fi(-δ))/(2δ), Bij=(Fi(+δ̇)-Fi(-δ̇))/(2δ̇)
    - _需求: 7.1–7.5_

  - [x] 8.5 实现网格加密功能
    - 在 `_setup_grid()` 中实现 axial_grading_factor > 1.0 时的非均匀网格：端部细、中部粗
    - _需求: 9.2_

- [x] 9. 实现公共接口与模式自动检测
  - [x] 9.1 实现 `__init__.py` — `analyze_bearing()` 简单接口
    - 接受关键字参数，自动构造 BearingInput 和 GrooveConfig
    - 自动判断模式：提供 eccentricity_ratio → 偏心模式，提供 load → 载荷模式
    - 错误信息包含步骤名称和错误描述
    - _需求: 12.1–12.4_

  - [x] 9.2 编写属性测试 — 分析模式自动检测（Property 18）
    - **Property 18: 分析模式自动检测**
    - 验证提供 eccentricity_ratio 时使用偏心模式，提供 load 时使用载荷模式
    - **验证: 需求 12.2**

  - [x] 9.3 编写属性测试 — 错误信息包含上下文（Property 19）
    - **Property 19: 错误信息包含上下文**
    - 验证错误信息包含错误步骤名称和错误描述
    - **验证: 需求 12.4**

- [x] 10. 检查点 — 完整功能验证
  - 确保所有已实现的测试通过，如有问题请向用户确认。

- [x] 11. 实现可视化模块
  - [x] 11.1 实现 `visualization.py` — 全部可视化函数
    - 实现 `plot_pressure_3d()`：3D 压力分布图（X=圆周角°, Y=轴向位置mm, Z=压力MPa）
    - 实现 `plot_pressure_contour()`：压力等值线图
    - 实现 `plot_cavitation_map()`：空化区域图（2D heatmap）
    - 实现 `plot_film_thickness()`：油膜厚度分布图（标注最薄处）
    - 实现 `plot_pressure_profile()`：截面压力曲线
    - 实现 `plot_journal_center()`：轴心位置图
    - 所有图表支持屏幕显示和保存为 PNG/PDF，标注坐标轴名称、单位和标题
    - _需求: 10.1–10.8_

- [x] 12. 创建示例脚本与 README
  - [x] 12.1 创建 `examples/` 目录与示例脚本
    - 示例1：对齐轴承基本分析（偏心模式）
    - 示例2：不对中轴承分析
    - 示例3：给定载荷求偏心（载荷模式）
    - 示例4：论文 Table 3 尾管轴承实例复现
    - _需求: 12.1, 12.3_

  - [x] 12.2 创建 `README.md`
    - 项目简介、安装方法、快速开始（含 Web 界面启动说明）、API 文档、示例说明
    - _需求: 12.1_

- [x] 13. 实现 Streamlit Web 界面
  - [x] 13.1 实现 `app.py` — Web 界面主体
    - 页面配置：标题、宽布局、图标
    - 侧边栏预设工况选择器：论文 Table 1（曲轴轴承）、Table 3（尾管轴承）、自定义，选择后自动填充对应参数
    - 侧边栏输入参数分组表单：轴承几何（mm/μm 工程单位）、工况参数（rpm）、偏心/载荷模式切换、不对中参数（mrad）、供油槽配置（下拉+条件输入）、高级参数（滑块）
    - "开始计算"按钮 + 计算中 spinner 提示
    - 主区域性能参数卡片（st.metric）：承载力、偏位角、最小油膜厚度、功率损耗
    - 主区域图表标签页（st.tabs）：压力分布3D图、空化区域图、油膜厚度图、截面压力曲线
    - 下载按钮：CSV 数据文件、图表 PNG
    - 详细结果摘要文本（st.expander）
    - 输入参数无效时在对应控件旁显示 st.error 提示
    - _需求: 13.1–13.9_

  - [x] 13.2 添加 `requirements.txt` 中的 streamlit 依赖
    - 添加 `streamlit>=1.30.0` 到依赖列表
    - _需求: 13.1_

- [x] 14. 最终检查点 — 全部测试通过
  - 确保所有已实现的测试通过，如有问题请向用户确认。

## 说明

- 标记 `*` 的子任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号，确保可追溯性
- 检查点任务确保增量验证
- 属性测试验证设计文档中定义的正确性属性（Property 1-20）
- 单元测试验证具体示例和边界条件
- 论文数据验证（任务 6）是最关键的集成测试，确保算法实现正确性

# 需求文档

## 简介

本项目基于 Jakeman (1984) 论文《A numerical analysis method based on flow continuity for hydrodynamic journal bearings》的理论，开发一个用于船舶滑动轴承（尾管轴承、曲轴轴承）水动力润滑分析的 Python 工程工具。该工具采用 Jakeman 提出的基于流量连续性的直接数值方法，对油膜网格的每个元素列质量守恒方程，空化区域通过引入气/汽流量项自动确定边界。程序支持轴线不对中分析、供油槽配置、动态系数计算等，面向船舶轴系设计工程师提供直观的输入输出和可视化结果。

## 术语表

- **BearingInput**: 轴承分析输入参数的数据模型（dataclass），包含几何、工况、不对中、供油槽、网格、求解器等全部配置
- **BearingOutput**: 轴承分析输出结果的数据模型，包含压力场、油膜厚度场、空化标记、承载力、力矩、功率损耗、侧漏流量、动态系数等
- **BearingSolver**: 基于 Jakeman 流量连续性方法的核心压力场迭代求解器模块
- **Bearing_Geometry_Module**: 负责计算油膜几何（公式1-5），包括不对中条件下网格四角油膜厚度 ha, hb, hc, hd
- **Bearing_Postprocess**: 后处理模块，根据压力场计算承载力、力矩、功率损耗、侧漏流量、动态系数（公式18-25）
- **Bearing_Practical**: 实用分析模块，给定载荷反求偏心距（Brent 法迭代），对应论文 Fig 4
- **Visualization_Module**: 负责生成压力分布3D图、等值线图、空化区域图、油膜厚度图、轴心位置图等
- **GrooveConfig**: 供油槽配置对象，支持 circumferential_360（360°环形槽）、axial_dual（双轴向槽）、axial_single（单轴向槽）
- **偏心率 (ε)**: 轴颈中心相对于轴承中心的偏移量与径向间隙之比，取值范围 0 到 1
- **直径间隙 (Cd)**: 轴承内径与轴颈外径之差
- **油膜厚度 (h)**: h = Cd/2 + e(s) × cos(θ - ψ(s))，其中 e(s) 为轴向位置 s 处的合成偏心距，ψ(s) 为偏位角（论文公式1）
- **不对中 (misalignment)**: 轴线相对于轴承中心线的倾斜，垂直面倾斜角 γ、水平面倾斜角 λ（论文公式4-5）
- **空化压力 (Pc)**: 油膜破裂时的临界压力，空化区域压力固定为 Pc，同时计算空化流量 Qvo（论文公式17）
- **压力流函数 (H)**: Hci, Hai, Hco, Hao，表征网格四个方向的流动阻力特性（论文公式8-11）
- **速度诱导流量 (K)**: 由轴表面速度驱动的流量项（论文公式12-16）
- **动态系数**: 刚度系数 A 和阻尼系数 B，通过微小扰动差分法求得，对齐轴承 8 个，不对中轴承 32 个（论文公式25）
- **SOR (逐次超松弛法)**: Successive Over-Relaxation，加速迭代收敛，默认松弛因子 1.7

## 需求

### 需求 1：轴承几何与工况输入

**用户故事：** 作为一名船舶工程师，我希望能够输入轴承的几何参数、运行工况、不对中参数和供油槽配置，以便对特定轴承进行分析。

#### 验收标准

1. THE BearingInput SHALL 接受以下轴承几何参数输入：轴承内径 diameter_m（m）、轴承长度 length_m（m）、直径间隙 clearance_m（m）
2. THE BearingInput SHALL 接受以下工况参数输入：转速 speed_rps（r/s）、油膜粘度 viscosity_pa_s（Pa·s）
3. THE BearingInput SHALL 接受偏心模式参数：偏心比 eccentricity_ratio（0~1）用于直接指定偏心
4. THE BearingInput SHALL 接受载荷模式参数：载荷 load_n（N）和载荷方向 load_direction_deg（°）用于反求偏心
5. THE BearingInput SHALL 接受不对中参数：垂直面倾斜角 misalignment_vertical_rad（rad）和水平面倾斜角 misalignment_horizontal_rad（rad），默认值均为 0
6. THE BearingInput SHALL 接受压力参数：空化压力 cavitation_pressure_pa（Pa）和环境压力 ambient_pressure_pa（Pa）
7. THE BearingInput SHALL 接受供油槽配置 groove（GrooveConfig 对象），支持 circumferential_360、axial_dual、axial_single 三种类型，包含槽位置、张角和供油压力
8. THE BearingInput SHALL 接受网格参数：周向网格数 n_circumferential（默认 72）、轴向网格数 n_axial（默认 14）、轴向网格加密因子 axial_grading_factor（默认 1.0 即均匀网格）
9. THE BearingInput SHALL 接受求解器参数：SOR 松弛因子 over_relaxation_factor（默认 1.7）、最大迭代次数 max_iterations（默认 5000）、收敛容差 convergence_tol（Pa，默认 1.0）
10. WHEN 用户提供的偏心比不在开区间 (0, 1) 内时，THE BearingInput SHALL 返回包含参数名称和有效范围的错误信息
11. WHEN 用户提供的几何参数为非正数时，THE BearingInput SHALL 返回包含参数名称和约束条件的错误信息
12. THE BearingInput SHALL 使用 SI 单位制，所有参数统一为米、秒、帕斯卡、弧度

### 需求 2：油膜几何计算（含不对中）

**用户故事：** 作为一名船舶工程师，我希望程序能正确计算含不对中条件下的油膜厚度分布，以便评估轴承间隙和最薄油膜位置。

#### 验收标准

1. THE Bearing_Geometry_Module SHALL 根据论文公式(4-5)计算轴向各位置的偏心分量：esy(s) = ecy + s × γ，esx(s) = ecx + s × λ
2. THE Bearing_Geometry_Module SHALL 根据论文公式(2-3)计算合成偏心距 e(s) = sqrt(esy² + esx²) 和偏位角 ψ(s) = atan2(esx, esy)
3. THE Bearing_Geometry_Module SHALL 根据论文公式(1)计算油膜厚度：h = Cd/2 + e(s) × cos(θ - ψ(s))
4. THE Bearing_Geometry_Module SHALL 对每个网格 (J, I) 计算四角油膜厚度 ha, hb, hc, hd，分别对应 (θ_J, s_I)、(θ_J, s_{I+1})、(θ_{J+1}, s_I)、(θ_{J+1}, s_{I+1})
5. FOR ALL 有效参数组合，THE Bearing_Geometry_Module SHALL 保证计算得到的油膜厚度值均为正数
6. WHEN 不对中参数 γ 和 λ 均为 0 时，THE Bearing_Geometry_Module SHALL 退化为对齐轴承的油膜厚度计算

### 需求 3：基于流量连续性的压力场求解

**用户故事：** 作为一名船舶工程师，我希望程序采用 Jakeman 的流量连续性方法求解压力场，以便获得比传统方法更准确的结果。

#### 验收标准

1. THE BearingSolver SHALL 在二维网格（周向 Mc × 轴向 Na）上，对每个网格元素列质量守恒方程进行迭代求解（论文公式6-7）
2. THE BearingSolver SHALL 计算每个网格四个方向的压力流函数 Hci, Hai, Hco, Hao（论文公式8-11）：Hci = (ha+hb)³×Δa/(96×η×Δc) 等
3. THE BearingSolver SHALL 计算速度诱导流量 K（论文公式12-16），稳态时 K = (hc+hd-ha-hb)×U×Δa/4，其中 U = π×D×N
4. THE BearingSolver SHALL 在周向方向施加周期性边界条件（J=1 的上游连接 J=Mc）
5. THE BearingSolver SHALL 在轴向两端施加环境压力边界条件
6. THE BearingSolver SHALL 对供油槽占据的网格固定压力为供油压力 Ps，不参与迭代
7. THE BearingSolver SHALL 对边界相邻网格的 H 系数乘以 2（因压力梯度跨越半个网格宽度）
8. THE BearingSolver SHALL 使用 SOR 超松弛加速收敛：P = P_old + ORF × (P_new - P_old)
9. THE BearingSolver SHALL 以相邻两次迭代的压力场最大绝对变化量作为收敛判据
10. IF 迭代次数达到最大值仍未收敛，THEN THE BearingSolver SHALL 返回警告信息，包含当前残差值和已执行的迭代次数
11. WHEN 求解完成时，THE BearingSolver SHALL 返回收敛后的二维压力场数组、空化标记矩阵和实际迭代次数

### 需求 4：空化模型处理

**用户故事：** 作为一名船舶工程师，我希望程序能通过流量连续性方法自动处理空化区域，以便获得物理上合理的压力分布和空化边界。

#### 验收标准

1. THE BearingSolver SHALL 在每次迭代中，当公式(7)计算出的压力 P_new ≤ Pc 时，将该网格标记为空化区域
2. WHEN 网格处于空化区域时，THE BearingSolver SHALL 将该网格压力设为空化压力 Pc
3. WHEN 网格处于空化区域时，THE BearingSolver SHALL 根据论文公式(17)计算空化流量 Qvo：Qvo = K + Qvi - (Hci×Pci + Hai×Pai + Hco×Pco + Hao×Pao) + Pc×(Hci+Hai+Hco+Hao)
4. THE BearingSolver SHALL 将空化流量 Qvo 传递给下游网格作为 Qvi：Qvi(J+1, I) = Qvo(J, I)
5. WHEN 网格压力 P_new > Pc 时，THE BearingSolver SHALL 将该网格标记为正常承压区域，Qvo = 0
6. THE BearingOutput SHALL 包含空化区域标记矩阵 cavitation_matrix（bool 类型，Mc×Na）

### 需求 5：后处理 — 承载力与力矩计算

**用户故事：** 作为一名船舶工程师，我希望程序能从压力分布中计算承载力和力矩，以便评估轴承的承载能力和不对中效应。

#### 验收标准

1. THE Bearing_Postprocess SHALL 使用论文公式(18)的12点加权平均法计算每个网格的平均压力：P_mean = (4×P + P1+...+P8) / 12
2. THE Bearing_Postprocess SHALL 计算承载力的垂直分量 Fy = Σ(-P_mean×Δa×Δc×cos(θ_J)) 和水平分量 Fx = Σ(-P_mean×Δa×Δc×sin(θ_J))
3. THE Bearing_Postprocess SHALL 计算合成承载力 F_total = sqrt(Fy² + Fx²) 和偏位角 ψ = atan2(Fx, Fy)
4. THE Bearing_Postprocess SHALL 计算力矩的垂直分量 My = Σ(dFy × s_I) 和水平分量 Mx = Σ(dFx × s_I)，其中 s_I 为轴向距离
5. THE BearingOutput SHALL 包含最小油膜厚度 min_film_thickness_m 及其位置

### 需求 6：后处理 — 流量与功率损耗计算

**用户故事：** 作为一名船舶工程师，我希望程序能计算侧漏流量和功率损耗，以便评估润滑油消耗和轴承发热。

#### 验收标准

1. THE Bearing_Postprocess SHALL 根据论文公式(19-20)计算圆周方向流量和轴向方向流量
2. THE Bearing_Postprocess SHALL 计算轴承两端面的侧漏流量 Qs = Σ|Qai|
3. THE Bearing_Postprocess SHALL 根据论文公式(21-24)计算功率损耗，包括有效油膜宽度 Bai/Bao 的计算和摩擦力 Fc 的积分
4. THE Bearing_Postprocess SHALL 计算总功率损耗 H = U × Σ Fc（W）

### 需求 7：后处理 — 动态刚度/阻尼系数

**用户故事：** 作为一名船舶工程师，我希望程序能计算动态刚度和阻尼系数，以便用于轴系振动分析。

#### 验收标准

1. THE Bearing_Postprocess SHALL 通过对轴心施加微小位移扰动，重新求解压力场，用差分法计算刚度系数 A（N/m）
2. THE Bearing_Postprocess SHALL 通过对轴心施加微小速度扰动，重新求解压力场，用差分法计算阻尼系数 B（N·s/m）
3. WHEN 轴承为对齐状态时，THE Bearing_Postprocess SHALL 计算 8 个动态系数（2×2 刚度 + 2×2 阻尼）
4. WHEN 轴承存在不对中时，THE Bearing_Postprocess SHALL 计算 32 个动态系数（4×8 矩阵，对应论文公式25）
5. THE BearingOutput SHALL 包含动态系数矩阵 displacement_coefficients

### 需求 8：实用分析模式（给定载荷求偏心）

**用户故事：** 作为一名船舶工程师，我通常知道轴承承受的载荷而不是偏心距，我希望程序能根据给定载荷自动求解偏心距。

#### 验收标准

1. THE Bearing_Practical SHALL 接受目标载荷 W（N）和载荷方向作为输入
2. THE Bearing_Practical SHALL 从初始偏心比 ε₀ = 0.5 开始，调用 BearingSolver 求解承载力
3. THE Bearing_Practical SHALL 使用 Brent 法迭代调整偏心比，直到计算承载力与目标载荷匹配
4. WHEN 需要指定载荷方向时，THE Bearing_Practical SHALL 同时迭代偏位角
5. WHEN 载荷匹配收敛后，THE Bearing_Practical SHALL 返回完整的 BearingOutput 结果

### 需求 9：网格系统与供油槽处理

**用户故事：** 作为一名船舶工程师，我希望程序能正确处理不同类型的供油槽和网格加密，以便准确模拟实际轴承结构。

#### 验收标准

1. THE BearingSolver SHALL 使用圆周方向 θ 从 0° 到 360° 的网格，网格中心角度 θ_J = (J-0.5) × 360°/Mc
2. THE BearingSolver SHALL 支持均匀网格（Δa = L/Na）和加密网格（grading_factor > 1 时端部细、中部粗）
3. THE BearingSolver SHALL 对 circumferential_360 类型供油槽，将对应轴向列的所有周向网格压力固定为 Ps
4. THE BearingSolver SHALL 对 axial_dual/axial_single 类型供油槽，将对应周向范围内的所有轴向网格压力固定为 Ps
5. THE GrooveConfig SHALL 包含槽类型 groove_type、角度位置 angular_positions_deg、张角 angular_width_deg 和供油压力 supply_pressure_pa

### 需求 10：结果可视化

**用户故事：** 作为一名船舶工程师，我希望能够可视化压力分布、空化区域和油膜厚度，以便直观地理解轴承的工作状态。

#### 验收标准

1. THE Visualization_Module SHALL 生成油膜压力分布的 3D surface plot（X=圆周角°, Y=轴向位置mm, Z=压力MPa）
2. THE Visualization_Module SHALL 生成压力分布等值线图（contour，俯视图）
3. THE Visualization_Module SHALL 生成空化区域图（2D heatmap，标记空化区域和承压区域）
4. THE Visualization_Module SHALL 生成油膜厚度分布图（contour 或 3D，标注最薄处位置）
5. THE Visualization_Module SHALL 生成轴承截面压力曲线（某一轴向位置的圆周压力分布 2D line）
6. THE Visualization_Module SHALL 生成轴心位置图（轴心在间隙圆内的位置 scatter）
7. THE Visualization_Module SHALL 使用 matplotlib 库，支持屏幕显示和保存为 PNG/PDF 格式
8. THE Visualization_Module SHALL 在图表中标注坐标轴名称、单位和图表标题

### 需求 11：结果输出与格式化

**用户故事：** 作为一名船舶工程师，我希望程序能以清晰的格式输出分析结果，以便用于工程报告和设计决策。

#### 验收标准

1. THE BearingOutput SHALL 提供 summary() 方法，输出包含所有性能参数的文本摘要：承载力、偏位角、力矩、最小油膜厚度、功率损耗、侧漏流量
2. THE summary() SHALL 在输出中包含输入参数的回显和求解器信息（迭代次数、收敛状态）
3. WHERE 用户指定输出文件路径，THE BearingOutput SHALL 将压力场数据和性能参数保存为 CSV 文件
4. FOR ALL 有效的压力场数组，THE BearingOutput SHALL 将压力场格式化为 CSV 后，重新读取该 CSV 文件得到与原始数值误差不超过 1e-10 的数组

### 需求 12：程序接口设计

**用户故事：** 作为一名船舶工程师（非程序员），我希望有简洁直观的接口来执行分析；作为开发者，我希望有灵活的高级接口。

#### 验收标准

1. THE 程序 SHALL 提供简单接口函数 analyze_bearing()，接受直径、长度、间隙、转速、粘度、载荷、不对中等关键字参数，返回 BearingOutput
2. THE analyze_bearing() SHALL 自动判断分析模式：提供 eccentricity_ratio 时为偏心模式，提供 load 时为载荷模式
3. THE 程序 SHALL 提供高级接口：用户可直接构造 BearingInput 和 GrooveConfig 对象，创建 BearingSolver 实例并调用 solve()
4. IF 分析过程中任何步骤发生错误，THEN 程序 SHALL 返回包含错误步骤名称和错误描述的错误信息

### 需求 13：Web 交互界面

**用户故事：** 作为一名船舶工程师（非程序员），我希望通过浏览器界面操作程序，不需要写任何代码就能完成轴承分析。

#### 验收标准

1. THE 程序 SHALL 提供基于 Streamlit 的 Web 界面，用户通过 `streamlit run app.py` 启动
2. THE Web 界面 SHALL 在左侧边栏提供所有输入参数的表单控件（数值输入框、下拉选择、滑块等），所有参数预填论文 Table 1 Case 1 的默认值
3. THE Web 界面 SHALL 将输入参数分组显示：轴承几何、工况参数、偏心/载荷模式、不对中参数、供油槽配置、网格参数、求解器参数
4. THE Web 界面 SHALL 提供"开始计算"按钮，点击后调用 analyze_bearing() 执行分析
5. THE Web 界面 SHALL 在主区域显示计算结果：性能参数摘要表格、压力分布图、空化区域图、油膜厚度图等
6. THE Web 界面 SHALL 在计算过程中显示进度提示（如"正在迭代求解..."）
7. THE Web 界面 SHALL 提供结果下载功能：CSV 数据文件和图表 PNG 文件
8. THE Web 界面 SHALL 提供预设工况快速选择：论文 Table 1（曲轴轴承）、Table 3（尾管轴承）等，一键填入对应参数
9. WHEN 输入参数无效时，THE Web 界面 SHALL 在对应输入框旁显示红色错误提示

### 需求 14：论文数据验证

**用户故事：** 作为一名开发者，我希望程序能通过论文中的验证数据测试，以确保算法实现的正确性。

#### 验收标准

1. THE test_validation SHALL 使用论文 Table 1 数据验证对齐曲轴轴承（D=63.5mm, L=23.68mm, 360°环形槽, Ps=0.2067MPa），承载力误差 < 3%，偏位角误差 < 2°
2. THE test_validation SHALL 使用论文 Table 2 数据验证不对中尾管轴承（L/D=1，双轴向槽 90°/270°，张角30°），4组工况
3. THE test_validation SHALL 使用论文 Table 3 数据验证尾管轴承实例（D=800mm, L=1200mm, Cd=1.4mm, 80rpm, η=0.125Pa·s, γ=0.0002rad），目标：W≈770,150N, M≈6.591×10⁷ N·mm

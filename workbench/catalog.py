"""Catalog of courses, algorithms, form fields and bundled sample data."""

COURSES = [
    {
        "id": "engineering",
        "name": "工程水文学",
        "subtitle": "洪水过程 · 时段洪量",
        "icon": "river",
        "color": "#38bdf8",
        "description": "从典型洪水过程出发，完成等间隔插值、多方法对比和时段最大洪量统计。",
    },
    {
        "id": "informatization",
        "name": "水利信息化",
        "subtitle": "空间插值 · 河道演算 · 单位线",
        "icon": "satellite",
        "color": "#a78bfa",
        "description": "泰森多边形面雨量、马斯京根流量演算，以及 Nash / Sherman 单位线。",
    },
    {
        "id": "flood",
        "name": "防洪概论",
        "subtitle": "曲线插值 · 水库调洪",
        "icon": "dam",
        "color": "#fb923c",
        "description": "对水位-库容、水位-下泄曲线插值，并用试算法或 RK4 完成水库调洪演算。",
    },
]

FREQ_OPTIONS = [
    {"label": "30 分钟", "value": "30min"},
    {"label": "1 小时", "value": "1h"},
    {"label": "3 小时", "value": "3h"},
    {"label": "6 小时", "value": "6h"},
]

METHOD_OPTIONS = [
    {"label": "线性 linear", "value": "linear"},
    {"label": "最近邻 nearest", "value": "nearest"},
    {"label": "多项式 polynomial", "value": "polynomial"},
    {"label": "样条 spline", "value": "spline"},
    {"label": "对数 log", "value": "log"},
]

ALGORITHMS = [
    # ---------------------------------------------------------------- engineering
    {
        "id": "eng_linear_interp",
        "course": "engineering",
        "name": "等间隔线性插值",
        "short": "把非等间隔洪水过程插值为 1h / 30min 等间隔序列",
        "description": "读取原始洪水过程的时间与流量，按指定步长生成完整时间轴并做线性插值，输出年、月、日、时、流量表，同时绘制原始点与插值曲线。",
        "script": "工程水文课程-代码开发/1-线性等间隔插值数据.py",
        "tags": ["洪水过程", "时间序列", "线性插值"],
        "fields": [
            {"key": "freq", "label": "插值步长", "type": "select", "default": "1h", "options": FREQ_OPTIONS, "help": "对应原脚本的 freq 参数，默认 1H。"},
        ],
        "files": [
            {"key": "flood_csv", "label": "原始洪水过程", "accept": ".csv,.txt", "required": True, "sample": "engineering_flood_raw.csv", "help": "需要包含年月日、时、Q 三列。"},
        ],
    },
    {
        "id": "eng_multi_interp",
        "course": "engineering",
        "name": "多种插值对比",
        "short": "线性 / 最近邻 / 多项式 / 样条四种方法同图对比",
        "description": "在同一时间轴上同时执行线性、最近邻、二次多项式和三次样条插值，便于比较不同方法对洪水过程线的平滑程度与峰值保持能力。",
        "script": "工程水文课程-代码开发/1-多种插值等间隔数据.py",
        "tags": ["方法对比", "多项式", "样条"],
        "fields": [
            {"key": "freq", "label": "插值步长", "type": "select", "default": "1h", "options": FREQ_OPTIONS},
            {"key": "poly_order", "label": "多项式阶数", "type": "number", "default": 2, "min": 1, "max": 6, "step": 1},
        ],
        "files": [
            {"key": "flood_csv", "label": "原始洪水过程", "accept": ".csv,.txt", "required": True, "sample": "engineering_flood_raw.csv"},
        ],
    },
    {
        "id": "eng_max_volume",
        "course": "engineering",
        "name": "时段最大洪量",
        "short": "滑动窗口统计 1h / 1d / 3d / 5d 最大洪量",
        "description": "对等间隔洪水过程做滑动窗口积分，分别用梯形法、辛普森法和三次插值积分计算各时段最大洪量，并输出最大洪量对应的起止时间。",
        "script": "工程水文课程-代码开发/2-时段最大洪量积分.py",
        "tags": ["滑动窗口", "数值积分", "设计洪水"],
        "fields": [
            {"key": "durations", "label": "统计时段（天）", "type": "text", "default": "1/24,1,3,5,7,10,20", "help": "逗号分隔；1/24 表示 1 小时。"},
            {"key": "smooth_points", "label": "插值积分采样点", "type": "number", "default": 1000, "min": 100, "max": 5000, "step": 100, "advanced": True},
        ],
        "files": [
            {"key": "hourly_csv", "label": "等间隔洪水过程", "accept": ".csv,.txt", "required": True, "sample": "engineering_flood_hourly.csv", "help": "列：年、月、日、时、Q，或带时间列与流量列。"},
        ],
    },
    {
        "id": "eng_area_integral",
        "course": "engineering",
        "name": "曲线面积积分校验",
        "short": "独立验证短时段洪水曲线面积",
        "description": "保留原脚本的验证思路：直接对日、时、流量序列做滑动积分，输出 4h、1d、3d、5d 的最大面积，用于核对时段洪量结果。",
        "script": "工程水文课程-代码开发/积分曲线面积.py",
        "tags": ["面积积分", "校验"],
        "fields": [
            {"key": "durations", "label": "统计时段（天）", "type": "text", "default": "4/24,1,3,5", "help": "4/24 表示 4 小时。"},
        ],
        "files": [
            {"key": "hourly_csv", "label": "日/时/流量数据", "accept": ".csv,.txt", "required": True, "sample": "engineering_flood_hourly.csv"},
        ],
    },
    # ---------------------------------------------------------------- informatization
    {
        "id": "thiessen",
        "course": "informatization",
        "name": "泰森多边形面雨量",
        "short": "由雨量站与流域边界计算控制面积权重和面平均雨量",
        "description": "读取流域边界和雨量站 Shapefile，构建泰森多边形并裁剪到流域范围，计算各站控制面积、权重和泰森多边形法 / 算术平均法面雨量。",
        "script": "水利信息化课程-代码开发/1-泰森多边形法-雨量插值/泰森多边形-雨量插值.py",
        "tags": ["空间分析", "面雨量", "Shapefile"],
        "fields": [
            {"key": "input_crs", "label": "输入坐标系", "type": "text", "default": "EPSG:4490"},
            {"key": "target_crs", "label": "投影坐标系", "type": "text", "default": "EPSG:4519", "help": "用于面积计算，可按流域所在带号调整。"},
            {"key": "station_field", "label": "站名字段", "type": "text", "default": "站名"},
            {"key": "rain_field", "label": "雨量字段", "type": "text", "default": "雨量"},
            {"key": "buffer_m", "label": "边界缓冲区 (m)", "type": "number", "default": 1000, "min": 0, "max": 20000, "step": 100, "advanced": True},
        ],
        "files": [
            {"key": "basin_zip", "label": "流域边界 Shapefile（zip）", "accept": ".zip", "required": True, "sample": "thiessen_basin.zip", "role": "basin", "help": "zip 内需包含 .shp/.dbf/.shx/.prj 等完整文件。"},
            {"key": "stations_zip", "label": "雨量站 Shapefile（zip）", "accept": ".zip", "required": True, "sample": "thiessen_stations.zip", "role": "stations", "help": "属性表需包含站名和雨量字段。"},
        ],
    },
    {
        "id": "muskingum_xk",
        "course": "informatization",
        "name": "马斯京根参数率定",
        "short": "由上下游实测流量反求最优 X、K",
        "description": "遍历 X ∈ [0, 0.5]，用蓄量-流量关系相关系数选出最优 X，再线性拟合 S-Q 关系求 K，并绘制蓄量曲线和 X-相关系数曲线。",
        "script": "水利信息化课程-代码开发/2-马斯京根流量演算法/1-马斯京根计算X、K.py",
        "tags": ["参数率定", "相关性", "河道演算"],
        "fields": [
            {"key": "delta_t", "label": "时间步长 Δt (h)", "type": "number", "default": 1, "min": 0.1, "max": 48, "step": 0.1},
        ],
        "files": [
            {"key": "pairs_csv", "label": "上下游时段流量", "accept": ".csv,.txt", "required": True, "sample": "muskingum_pairs.csv", "help": "两列数值：上游流量、下游流量；可带时段列。"},
        ],
    },
    {
        "id": "muskingum_route",
        "course": "informatization",
        "name": "马斯京根下游演算",
        "short": "按 K、X、初始流量推求下游出流过程",
        "description": "根据 K、X 和时间步长自动计算 C0/C1/C2 系数，再用 Q(i)=C0·I(i)+C1·I(i-1)+C2·Q(i-1) 逐时段演算下游流量。",
        "script": "水利信息化课程-代码开发/2-马斯京根流量演算法/2-马斯京根计算下游流量.py",
        "tags": ["流量演算", "C0/C1/C2", "洪水传播"],
        "fields": [
            {"key": "k", "label": "K 蓄量时间 (h)", "type": "number", "default": 1.1002, "min": 0.01, "max": 200, "step": 0.01},
            {"key": "x", "label": "X 流量比重", "type": "number", "default": 0.27, "min": 0, "max": 0.5, "step": 0.01},
            {"key": "dt", "label": "时间步长 Δt (h)", "type": "number", "default": 1, "min": 0.1, "max": 48, "step": 0.1},
            {"key": "initial_q", "label": "初始下游流量 (m³/s)", "type": "number", "default": 350, "min": 0, "step": 1},
        ],
        "files": [
            {"key": "upstream_csv", "label": "上游流量过程", "accept": ".csv,.txt", "required": True, "sample": "muskingum_upstream.csv", "help": "一列上游流量；可带时段列。"},
        ],
    },
    {
        "id": "nash_nk",
        "course": "informatization",
        "name": "Nash 瞬时单位线 N、K",
        "short": "用净雨与流量过程的一、二阶矩推求 N、K",
        "description": "对净雨 R 和出口流量 Q 序列计算一阶、二阶矩，由矩差推求 Nash 瞬时单位线参数 N、K，并绘制双轴过程线。",
        "script": "水利信息化课程-代码开发/3-单位线/瞬时单位线/1-瞬时单位线求N、K.py",
        "tags": ["Nash", "矩法", "单位线"],
        "fields": [
            {"key": "dt", "label": "时间步长 (h)", "type": "number", "default": 3, "min": 0.1, "max": 72, "step": 0.1},
        ],
        "files": [
            {"key": "rain_flow_csv", "label": "净雨 / 流量过程", "accept": ".csv,.txt", "required": True, "sample": "nash_rain_flow.csv", "help": "列：Time、R、Q。"},
        ],
    },
    {
        "id": "nash_duh",
        "course": "informatization",
        "name": "Nash 时段单位线",
        "short": "由 N、K、流域面积与净雨量推求时段单位线",
        "description": "通过 Nash 瞬时单位线的 S 曲线差值，将参数 N、K 转换为指定时段长和净雨量下的时段单位线，并可选标准曲线计算 RMSE。",
        "script": "水利信息化课程-代码开发/3-单位线/瞬时单位线/2-瞬时单位线求时段单位线.py",
        "tags": ["S 曲线", "时段单位线", "RMSE"],
        "fields": [
            {"key": "n", "label": "N 参数", "type": "number", "default": 1.5, "min": 0.1, "max": 20, "step": 0.1},
            {"key": "k", "label": "K 参数 (h)", "type": "number", "default": 5.68, "min": 0.01, "max": 200, "step": 0.01},
            {"key": "area", "label": "流域面积 (km²)", "type": "number", "default": 349, "min": 0.1, "step": 1},
            {"key": "dt", "label": "时段长 (h)", "type": "number", "default": 3, "min": 0.1, "max": 48, "step": 0.1},
            {"key": "rain", "label": "净雨量 (mm)", "type": "number", "default": 10, "min": 0, "step": 0.1},
            {"key": "tmax", "label": "计算时长 (h)", "type": "number", "default": 51, "min": 6, "max": 240, "step": 1, "advanced": True},
        ],
        "files": [
            {"key": "standard_csv", "label": "标准曲线（可选）", "accept": ".csv,.txt", "required": False, "sample": "nash_standard.csv", "help": "用于绘制对比曲线和计算 RMSE。"},
        ],
    },
    {
        "id": "sherman_derive",
        "course": "informatization",
        "name": "Sherman 单位线推求",
        "short": "由两次净雨的地面径流过程反推 10mm 单位线",
        "description": "按 Sherman 单位线递推公式逐时段计算单位线纵坐标，并自动调整系数使单位线总量 q 接近 10mm。",
        "script": "水利信息化课程-代码开发/3-单位线/谢尔曼单位线/1-推求谢尔曼单位线.py",
        "tags": ["Sherman", "单位线", "试算"],
        "fields": [
            {"key": "pe1", "label": "第一时段净雨 (mm)", "type": "number", "default": 15.7, "min": 0.1, "step": 0.1},
            {"key": "pe2", "label": "第二时段净雨 (mm)", "type": "number", "default": 5.9, "min": 0, "step": 0.1},
            {"key": "area", "label": "流域面积 (km²)", "type": "number", "default": 8080, "min": 0.1, "step": 1},
            {"key": "h", "label": "时段长 (h)", "type": "number", "default": 12, "min": 0.1, "max": 72, "step": 0.1},
            {"key": "tolerance", "label": "q 值收敛容差", "type": "number", "default": 0.1, "min": 0.0001, "max": 1, "step": 0.01, "advanced": True},
        ],
        "files": [
            {"key": "rs_csv", "label": "地面径流序列", "accept": ".csv,.txt", "required": True, "sample": "sherman_rs.csv", "help": "至少包含一列地面径流（m³/s）。"},
        ],
    },
    {
        "id": "sherman_outlet",
        "course": "informatization",
        "name": "Sherman 出口流量",
        "short": "用净雨量与单位线卷积计算出口流量过程",
        "description": "读取净雨量和 10mm 单位线，通过卷积计算出口流量过程，并可与实测出口流量对比，绘制降雨、流量和单位线三联图。",
        "script": "水利信息化课程-代码开发/3-单位线/谢尔曼单位线/2-谢尔曼单位线求出口流量.py",
        "tags": ["卷积", "出口流量", "对比"],
        "fields": [
            {"key": "area", "label": "流域面积 (km²)", "type": "number", "default": 3400, "min": 0.1, "step": 1},
            {"key": "dt", "label": "时段长 (h)", "type": "number", "default": 6, "min": 0.1, "max": 72, "step": 0.1},
        ],
        "files": [
            {"key": "input_csv", "label": "净雨量 / 单位线", "accept": ".csv,.txt", "required": True, "sample": "sherman_input.csv"},
            {"key": "compare_csv", "label": "实际出口流量（可选）", "accept": ".csv,.txt", "required": False, "sample": "sherman_comparison.csv"},
        ],
    },
    # ---------------------------------------------------------------- flood
    {
        "id": "flood_curve_interp",
        "course": "flood",
        "name": "水库曲线插值",
        "short": "对 Z-V、Z-q 或任意多列曲线做等间隔插值",
        "description": "选择自变量列和插值方法，对 Z-V、Z-q 或通用 CSV 的所有因变量列做等间隔密化，输出插值表并绘制原始点与插值曲线。",
        "script": "防洪概论课程-代码开发/曲线插值（Z-V或Z-q曲线）.py",
        "tags": ["曲线密化", "多列插值", "Z-V / Z-q"],
        "fields": [
            {"key": "x_col", "label": "自变量列", "type": "column", "default": "水位Z/m", "help": "从所选 CSV 的列中自动读取。"},
            {"key": "method", "label": "插值方法", "type": "select", "default": "linear", "options": METHOD_OPTIONS},
            {"key": "x_step", "label": "自变量步长", "type": "number", "default": 0.1, "min": 0.0001, "step": 0.1},
            {"key": "poly_order", "label": "多项式阶数", "type": "number", "default": 3, "min": 1, "max": 8, "step": 1, "advanced": True},
            {"key": "x_precision", "label": "自变量小数位", "type": "number", "default": 0, "min": 0, "max": 8, "step": 1, "advanced": True},
            {"key": "y_precision", "label": "结果小数位", "type": "number", "default": 0, "min": 0, "max": 8, "step": 1, "advanced": True},
        ],
        "files": [
            {"key": "curve_csv", "label": "曲线数据 CSV", "accept": ".csv,.txt", "required": True, "sample": "reservoir_storage.csv", "help": "例如：水位Z/m、库容V/万m3。"},
        ],
    },
    {
        "id": "flood_trial",
        "course": "flood",
        "name": "调洪演算 · 试算法",
        "short": "逐时段试算水位，使水量平衡与库容曲线闭合",
        "description": "读取入库流量、水位-库容、水位-下泄流量三条关系，在给定水位区间内逐时段试算，输出水位、库容、下泄流量和水量平衡表。",
        "script": "防洪概论课程-代码开发/调洪计算-试算法.py",
        "tags": ["试算法", "调洪", "水量平衡"],
        "fields": [
            {"key": "initial_discharge", "label": "初始下泄流量 (m³/s)", "type": "number", "default": 173.9, "min": 0, "step": 0.1},
            {"key": "initial_v", "label": "初始库容 (万m³)", "type": "number", "default": 6450, "min": 0, "step": 1},
            {"key": "initial_z", "label": "初始水位 (m)", "type": "number", "default": 38, "min": 0, "step": 0.1},
            {"key": "v_tolerance", "label": "库容误差 (万m³)", "type": "number", "default": 3, "min": 0.01, "step": 0.1},
            {"key": "z_min", "label": "试算水位下限 (m)", "type": "number", "default": 36, "step": 0.1},
            {"key": "z_max", "label": "试算水位上限 (m)", "type": "number", "default": 41, "step": 0.1},
            {"key": "decimals", "label": "试算小数位", "type": "number", "default": 1, "min": 0, "max": 4, "step": 1, "advanced": True},
            {"key": "unit_conversion", "label": "单位换算系数", "type": "number", "default": 0.0001, "min": 0.000001, "step": 0.0001, "advanced": True},
        ],
        "files": [
            {"key": "inflow_csv", "label": "入库流量过程", "accept": ".csv,.txt", "required": True, "sample": "reservoir_inflow.csv"},
            {"key": "storage_csv", "label": "水位-库容曲线", "accept": ".csv,.txt", "required": True, "sample": "reservoir_storage.csv"},
            {"key": "discharge_csv", "label": "水位-下泄流量曲线", "accept": ".csv,.txt", "required": True, "sample": "reservoir_discharge.csv"},
        ],
    },
    {
        "id": "flood_rk4",
        "course": "flood",
        "name": "调洪演算 · RK4",
        "short": "用四阶龙格-库塔法数值积分 dV/dt = Q - q(Z)",
        "description": "把水库调洪写成 dV/dt = Q(t) - q(Z) 的初值问题，用定步长四阶龙格-库塔法逐时段积分，输出库容、水位和下泄流量过程。",
        "script": "防洪概论课程-代码开发/调洪计算-龙格-库数值解法.py",
        "tags": ["RK4", "数值积分", "调洪"],
        "fields": [
            {"key": "dt_hours", "label": "时间步长 (h)", "type": "number", "default": 3, "min": 0.1, "max": 48, "step": 0.1},
            {"key": "initial_z", "label": "初始水位 (m)", "type": "number", "default": 38, "min": 0, "step": 0.1},
            {"key": "initial_v", "label": "初始库容 (万m³)", "type": "number", "default": 6450, "min": 0, "step": 1},
        ],
        "files": [
            {"key": "inflow_csv", "label": "入库流量过程", "accept": ".csv,.txt", "required": True, "sample": "reservoir_inflow.csv"},
            {"key": "storage_csv", "label": "水位-库容曲线", "accept": ".csv,.txt", "required": True, "sample": "reservoir_storage.csv"},
            {"key": "discharge_csv", "label": "水位-下泄流量曲线", "accept": ".csv,.txt", "required": True, "sample": "reservoir_discharge.csv"},
        ],
    },
]

COURSE_BY_ID = {course["id"]: course for course in COURSES}
ALGORITHM_BY_ID = {algorithm["id"]: algorithm for algorithm in ALGORITHMS}


def catalog_payload():
    return {"courses": COURSES, "algorithms": ALGORITHMS}


def get_algorithm(algorithm_id):
    return ALGORITHM_BY_ID.get(algorithm_id)

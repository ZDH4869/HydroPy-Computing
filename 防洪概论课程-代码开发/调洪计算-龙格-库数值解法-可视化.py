# -*- coding: utf-8 -*-
"""
水库调洪演算 —— 定步长四阶龙格-库塔法（RK4）
微分方程：dV/dt = Q(t) - q(z)
--------------------------------------------------
单位约定（内部计算）：
    水位 z：m
    库容 V：m³（读取时立即把“万m³”→m³）
    流量 Q/q：m³/s
输出 CSV 时再把 V 转回“万m³”方便查看
--------------------------------------------------
新增：上下布局双子图，最大值写入图例，曲线上仅标散点
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import warnings
warnings.filterwarnings('ignore')

# ========== 用户参数区 ==========
# 1. 文件路径
INFLOW_FILE   = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\原始曲线\3h入库流量过程线.csv"
STORAGE_FILE  = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-库容曲线_linear.csv"
DISCHARGE_FILE= r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-下泄流量曲线_linear.csv"

OUT_FILE      = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\数值法_3h.csv"
visualization_output = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\数值法-结果可视化.png"

# 2. 文件编码
INFLOW_ENCODING   = 'utf-8'
STORAGE_ENCODING  = 'utf-8'
DISCHARGE_ENCODING= 'utf-8'

# 3. 时间步长（秒）
DT = 3600 * 3

# 4. 初始状态
INITIAL_Z = 38.0   # m
INITIAL_V = 6450   # 万 m³

# 5. 输出字段
OUT_COLS = ['时间t/h', '入库流量Q/(m³·s⁻¹)', '下泄流量q/(m³·s⁻¹)', '水库存水量V/万m³', '水库水位Z/m']

# 6. 可视化参数
FIG_SIZE = (10, 12)        # 上下布局
DPI      = 300
FONT_PATH= None            # 自定义中文字体路径；None=默认
# ========== 工具函数 ==========
def set_chinese_font():
    if FONT_PATH and os.path.isfile(FONT_PATH):
        font = FontProperties(fname=FONT_PATH, size=12)
        plt.rcParams['font.family'] = font.get_name()
    else:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def read_curves():
    sto = pd.read_csv(STORAGE_FILE,  encoding=STORAGE_ENCODING)
    dis = pd.read_csv(DISCHARGE_FILE,encoding=DISCHARGE_ENCODING)
    Z_sto, V_sto = sto['水位Z/m'].values, sto['库容V/万m3'].values * 1e4
    Z_dis, q_dis = dis['水位Z/m'].values, dis['下泄流量q/(m3·s)'].values
    storage_interp = interp1d(Z_sto, V_sto, kind='linear', bounds_error=False, fill_value='extrapolate')
    discharge_interp= interp1d(Z_dis, q_dis, kind='linear', bounds_error=False, fill_value='extrapolate')
    V_Z_interp = interp1d(V_sto, Z_sto, kind='linear', bounds_error=False, fill_value='extrapolate')
    return storage_interp, discharge_interp, V_Z_interp

def read_inflow():
    df = pd.read_csv(INFLOW_FILE, encoding=INFLOW_ENCODING)
    t = df['时间t/h'].values
    Q = df['Q/(m3/s-1)'].values
    return t, Q

# ========== RK4 核心 ==========
def rk4_step(V_prev, Q_avg, storage_interp, discharge_interp, V_Z_interp, dt):
    def dVdt(V):
        z = float(V_Z_interp(V))
        q = float(discharge_interp(z))
        return Q_avg - q   # m³/s
    k1 = dVdt(V_prev)
    k2 = dVdt(V_prev + 0.5*dt*k1)
    k3 = dVdt(V_prev + 0.5*dt*k2)
    k4 = dVdt(V_prev +       dt*k3)
    V_new = V_prev + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    z_new = float(V_Z_interp(V_new))
    q_new = float(discharge_interp(z_new))
    return V_new, z_new, q_new

# ========== 上下布局可视化（最大值写入图例） ==========
def plot_results(t, Q_in, q_out, z_out, save_path=visualization_output):
    set_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE)

    # 上子图：时间-流量过程线
    ax = axes[0]
    q_max = q_out.max();  o_max = q_out.max()   # 下泄最大
    Q_max = Q_in.max()                       # 入库最大
    # 曲线 + 散点（仅点）
    ax.plot(t, Q_in,  label=f'入库流量 Q (max={Q_max:.1f})', color='dodgerblue', lw=1.8)
    ax.plot(t, q_out, label=f'下泄流量 q (max={q_max:.1f})', color='orangered', lw=1.8)
    ax.scatter(t[np.argmax(Q_in)], Q_max, color='dodgerblue', zorder=5)
    ax.scatter(t[np.argmax(q_out)], q_max, color='orangered', zorder=5)
    ax.set_ylabel('流量 / m³·s⁻¹')
    ax.set_title('时间-流量过程线（RK4）')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 下子图：时间-水位过程线
    ax = axes[1]
    z_max = z_out.max()
    ax.plot(t, z_out, label=f'水位 Z (max={z_max:.2f} m)', color='forestgreen', lw=1.8)
    ax.scatter(t[np.argmax(z_out)], z_max, color='forestgreen', zorder=5)
    ax.set_xlabel('时间 t/h')
    ax.set_ylabel('水位 Z / m')
    ax.set_title('时间-水位过程线（RK4）')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"RK4计算结果曲线图线图已保存：{save_path}")
    plt.show()

# ========== 主流程 ==========
def main():
    t_h, Q_in = read_inflow()
    storage_interp, discharge_interp, V_Z_interp = read_curves()

    V0 = INITIAL_V * 1e4;  z0 = INITIAL_Z;  q0 = float(discharge_interp(z0))
    V_list, z_list, q_list = [V0], [z0], [q0]

    for i in tqdm(range(1, len(t_h)), desc="RK4 调洪计算"):
        Q_avg = 0.5 * (Q_in[i-1] + Q_in[i])
        V_new, z_new, q_new = rk4_step(V_list[-1], Q_avg,
                                       storage_interp, discharge_interp, V_Z_interp, DT)
        V_list.append(V_new); z_list.append(z_new); q_list.append(q_new)

    # 保存结果
    results = pd.DataFrame({
        '时间t/h': t_h,
        '入库流量Q/(m³·s⁻¹)': Q_in,
        '下泄流量q/(m³·s⁻¹)': q_list,
        '水库存水量V/万m³': np.array(V_list) * 1e-4,
        '水库水位Z/m': z_list
    })[OUT_COLS]
    results.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print(f"RK4 调洪完成！结果已保存至：{OUT_FILE}")

    # 上下布局可视化（最大值在图例）
    plot_results(t_h, Q_in, np.array(q_list), np.array(z_list),
                 save_path=os.path.splitext(visualization_output)[0] + '.png')

# ========== 运行 ==========
if __name__ == '__main__':
    main()
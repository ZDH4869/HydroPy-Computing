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
如需考虑闸门调度规则，请准备对应水位→泄流量曲线并替换下方文件路径
"""
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# ========== 用户参数区 ==========
# 1. 文件路径
INFLOW_FILE   = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\原始曲线\3h入库流量过程线.csv"
STORAGE_FILE  = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-库容曲线_linear.csv"
DISCHARGE_FILE= r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\曲线插值\插值水位-下泄流量曲线_linear.csv"
OUT_FILE      = r"E:\水电202303班\大三（上期）\课程报告或小组作业\防洪概论（调洪计算）\代码开发\B1_RK4.csv"

# 2. 文件编码 读取格式 文件编码 'gbk' 'utf-8' 'latin1'
INFLOW_ENCODING   = 'utf-8'
STORAGE_ENCODING  = 'utf-8'
DISCHARGE_ENCODING= 'utf-8'

# 3. 时间步长（秒）—— 用户可改
DT = 3600*3          # 3600 s = 1 h；可改为 1800、900 等

# 4. 初始状态 V0 —— 用户可改
INITIAL_Z = 38.0   # m
INITIAL_V = 6450   # 万 m³（脚本内立即×1e4 转成 m³）

# 5. 输出字段
OUT_COLS = ['时间t/h', '入库流量Q/(m³·s⁻¹)', '下泄流量q/(m³·s⁻¹)', '水库存水量V/万m³', '水库水位Z/m']

# ========== 工具函数 ==========
def read_curves():
    """读取三条曲线，返回插值函数"""
    sto = pd.read_csv(STORAGE_FILE,  encoding=STORAGE_ENCODING)
    dis = pd.read_csv(DISCHARGE_FILE,encoding=DISCHARGE_ENCODING)
    # 库容曲线：万m³ → m³
    Z_sto, V_sto = sto['水位Z/m'].values, sto['库容V/万m3'].values * 1e4
    # 泄流曲线：m³/s 不变
    Z_dis, q_dis = dis['水位Z/m'].values, dis['下泄流量q/(m3·s)'].values

    storage_interp = interp1d(Z_sto, V_sto, kind='linear', bounds_error=False, fill_value='extrapolate')
    discharge_interp= interp1d(Z_dis, q_dis, kind='linear', bounds_error=False, fill_value='extrapolate')
    # 反插：V → Z
    V_Z_interp = interp1d(V_sto, Z_sto, kind='linear', bounds_error=False, fill_value='extrapolate')
    return storage_interp, discharge_interp, V_Z_interp

def read_inflow():
    """读取入库流量过程"""
    df = pd.read_csv(INFLOW_FILE, encoding=INFLOW_ENCODING)
    # 假设列名：时间t/h  与  Q/(m3/s-1)  如不同请自行改
    t = df['时间t/h'].values
    Q = df['Q/(m3/s-1)'].values
    return t, Q

# ========== RK4 核心 ==========
def rk4_step(V_prev, Q_avg, storage_interp, discharge_interp, V_Z_interp, dt):
    """
    单步 RK4 积分
    返回下一时刻 V 及对应 z、q
    """
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

# ========== 主流程 ==========
def main():
    t_h, Q_in = read_inflow()
    storage_interp, discharge_interp, V_Z_interp = read_curves()

    # 初始值
    V0 = INITIAL_V * 1e4          # 万m³ → m³
    z0 = INITIAL_Z
    q0 = float(discharge_interp(z0))

    # 结果容器
    V_list = [V0]
    z_list = [z0]
    q_list = [q0]

    # 逐时段 RK4 积分
    for i in tqdm(range(1, len(t_h)), desc="RK4 调洪计算"):
        # 时段平均入库流量（梯形假设）
        Q_avg = 0.5 * (Q_in[i-1] + Q_in[i])
        V_new, z_new, q_new = rk4_step(V_list[-1], Q_avg,
                                       storage_interp, discharge_interp, V_Z_interp, DT)
        V_list.append(V_new)
        z_list.append(z_new)
        q_list.append(q_new)

    # 组装 DataFrame（V 转回万m³）
    results = pd.DataFrame({
        '时间t/h': t_h,
        '入库流量Q/(m³·s⁻¹)': Q_in,
        '下泄流量q/(m³·s⁻¹)': q_list,
        '水库存水量V/万m³': np.array(V_list) * 1e-4,
        '水库水位Z/m': z_list
    })[OUT_COLS]

    results.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
    print(f"RK4 调洪完成！结果已保存至：{OUT_FILE}")
    print("输出列：", list(results.columns))

# ========== 运行 ==========
if __name__ == '__main__':
    main()
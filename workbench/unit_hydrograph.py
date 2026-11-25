"""Nash instantaneous unit hydrograph and Sherman unit hydrograph algorithms."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.special import gammainc

from .common import (
    BLUE,
    CYAN,
    GREEN,
    MUTED,
    ORANGE,
    PINK,
    VIOLET,
    YELLOW,
    clean_numeric,
    configure_matplotlib,
    find_column,
    fmt,
    read_csv,
)


def _read_nash_csv(path):
    df = read_csv(path)
    time_col = find_column(df, ["Time", "时间", "时间(h)", "时间（小时）", "t"])
    r_col = find_column(df, ["R", "净雨", "净雨量", "降雨"])
    q_col = find_column(df, ["Q", "流量", "流量(m3/s)", "流量(m³/s)"])
    if time_col is None or r_col is None or q_col is None:
        raise ValueError("CSV 需要包含 Time、R、Q 三列。")
    out = pd.DataFrame(
        {
            "Time": clean_numeric(df[time_col]),
            "R": clean_numeric(df[r_col]),
            "Q": clean_numeric(df[q_col]),
        }
    ).dropna()
    if len(out) < 3:
        raise ValueError("有效数据点太少。")
    return out


def run_nash_nk(ctx, params, files):
    plt = configure_matplotlib()
    dt = float(params.get("dt") or 3)
    if dt <= 0:
        raise ValueError("时间步长必须大于 0。")
    ctx.section("读取净雨 / 出口流量")
    df = _read_nash_csv(files["rain_flow_csv"])
    r = df["R"].to_numpy(dtype=float)
    q = df["Q"].to_numpy(dtype=float)
    ctx.log(f"有效数据：{len(df)} 行，时间步长：{dt:g} h")
    ctx.log("Time\tR\tQ")
    for _, row in df.head(8).iterrows():
        ctx.log(f"{row['Time']:g}\t{row['R']:g}\t{row['Q']:g}")

    nr = len(r)
    nq = len(q)
    x1_r = np.sum(r * (2 * np.arange(1, nr + 1) - 1))
    x2_r = np.sum(r)
    x3_r = np.sum(r * (2 * np.arange(1, nr + 1) - 1) ** 2)
    if abs(x2_r) < 1e-12:
        raise ValueError("净雨量总和为 0，无法计算矩。")
    mi1 = x1_r / x2_r * dt / 2
    mi2 = x3_r / x2_r * dt * dt / 4

    xm = (q[:-1] + q[1:]) / 2
    x1_q = np.sum(xm * (2 * np.arange(1, nq) - 1))
    x2_q = np.sum(xm)
    x3_q = np.sum(xm * (2 * np.arange(1, nq) - 1) ** 2)
    if abs(x2_q) < 1e-12:
        raise ValueError("流量过程总水量为 0，无法计算矩。")
    mo1 = x1_q / x2_q * dt / 2
    mo2 = x3_q / x2_q * dt * dt / 4
    ni2 = mi2 - mi1**2
    no2 = mo2 - mo1**2
    delta_m = mo1 - mi1
    delta_var = no2 - ni2
    if abs(delta_m) < 1e-12 or abs(delta_var) < 1e-12:
        raise ValueError("矩差接近 0，无法稳定求解 N、K，请检查输入序列。")
    n_value = delta_m**2 / delta_var
    k_value = delta_var / delta_m
    ctx.log(f"输入矩：mi1={mi1:.4f}, mi2={mi2:.4f}, 方差={ni2:.4f}")
    ctx.log(f"输出矩：mo1={mo1:.4f}, mo2={mo2:.4f}, 方差={no2:.4f}")
    ctx.log(f"N={n_value:.4f}, K={k_value:.4f} h")
    ctx.add_summary("N 参数", f"{n_value:.3f}")
    ctx.add_summary("K 参数", f"{k_value:.3f}", "h")
    ctx.add_summary("时间步长", f"{dt:g}", "h")

    moments = pd.DataFrame(
        [
            {"项目": "输入一阶矩 mi1", "数值": round(mi1, 6)},
            {"项目": "输入二阶矩 mi2", "数值": round(mi2, 6)},
            {"项目": "输出一阶矩 mo1", "数值": round(mo1, 6)},
            {"项目": "输出二阶矩 mo2", "数值": round(mo2, 6)},
            {"项目": "N", "数值": round(n_value, 6)},
            {"项目": "K (h)", "数值": round(k_value, 6)},
        ]
    )
    ctx.save_csv(moments, "Nash参数计算结果.csv", title="矩法计算结果")
    ctx.add_visual_table("rain_response", ["时间 (h)", "净雨量", "出口流量"], df["Time"], r, q)

    fig, ax1 = plt.subplots(figsize=(8.8, 5))
    ax1.plot(df["Time"], df["R"], "o-", color=CYAN, linewidth=1.8, markersize=4, label="净雨 R")
    ax1.set_xlabel("时间 / h")
    ax1.set_ylabel("净雨量 / mm", color=CYAN)
    ax1.tick_params(axis="y", labelcolor=CYAN)
    ax2 = ax1.twinx()
    ax2.plot(df["Time"], df["Q"], "s-", color=ORANGE, linewidth=2, markersize=4, label="流量 Q")
    ax2.set_ylabel("流量 / m³/s", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax1.set_title(f"Nash 瞬时单位线参数  N={n_value:.2f}, K={k_value:.2f} h")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    ctx.save_fig(fig, "Nash参数计算.png", title="Nash 参数过程线")
    ctx.log("✓ Nash N、K 计算完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def _nash_s_curve(t, n_value, k_value):
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t)
    mask = t > 0
    result[mask] = gammainc(n_value, t[mask] / k_value)
    return result


def run_nash_duh(ctx, params, files):
    plt = configure_matplotlib()
    n_value = float(params.get("n") or 1.5)
    k_value = float(params.get("k") or 5.68)
    area = float(params.get("area") or 349)
    dt = float(params.get("dt") or 3)
    rain = float(params.get("rain") or 10)
    tmax = float(params.get("tmax") or 51)
    if n_value <= 0 or k_value <= 0 or dt <= 0 or area <= 0:
        raise ValueError("N、K、时段长和流域面积必须大于 0。")
    ctx.section("计算 Nash 时段单位线")
    time_points = np.arange(0, tmax + dt * 0.5, dt)
    s_values = _nash_s_curve(time_points, n_value, k_value)
    s_prev = np.where(time_points >= dt, _nash_s_curve(time_points - dt, n_value, k_value), 0.0)
    duh = (s_values - s_prev) * rain * area / (dt * 3.6)
    peak = float(np.max(duh))
    peak_time = float(time_points[int(np.argmax(duh))])
    ctx.log(f"N={n_value:g}, K={k_value:g} h, F={area:g} km², Δt={dt:g} h, P={rain:g} mm")
    ctx.log(f"洪峰流量：{peak:.2f} m³/s，出现在 {peak_time:g} h")
    ctx.add_summary("洪峰流量", fmt(peak, 1), "m³/s")
    ctx.add_summary("峰现时间", f"{peak_time:g}", "h")
    ctx.add_summary("计算时段数", len(time_points), "个")

    table = pd.DataFrame({"时间(h)": np.round(time_points, 4), "时段单位线(m³/s)": np.round(duh, 4)})
    ctx.save_csv(table, "Nash时段单位线结果.csv", title="时段单位线")

    std_time = std_flow = None
    rmse = None
    if files.get("standard_csv"):
        try:
            std_df = read_csv(files["standard_csv"])
            std_time_col = find_column(std_df, ["时间（小时）", "时间(h)", "Time", "时间"])
            std_flow_col = find_column(std_df, ["流量(m³/s)", "流量(m3/s)", "Q", "流量"])
            if std_time_col and std_flow_col:
                std_time = clean_numeric(std_df[std_time_col]).to_numpy(dtype=float)
                std_flow = clean_numeric(std_df[std_flow_col]).to_numpy(dtype=float)
                valid = np.isfinite(std_time) & np.isfinite(std_flow)
                std_time, std_flow = std_time[valid], std_flow[valid]
                if len(std_time) >= 2:
                    interp = interpolate.interp1d(std_time, std_flow, bounds_error=False, fill_value=0)
                    std_interp = interp(time_points)
                    rmse = float(np.sqrt(np.mean((duh - std_interp) ** 2)))
                    ctx.log(f"标准曲线 RMSE：{rmse:.3f} m³/s")
                    ctx.add_summary("标准曲线 RMSE", f"{rmse:.2f}", "m³/s")
        except Exception as exc:
            ctx.log(f"标准曲线读取失败，跳过对比：{exc}")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(time_points, duh, width=dt * 0.78, color=CYAN, alpha=0.22, edgecolor=CYAN, linewidth=0.8, label="时段单位线柱")
    ax.plot(time_points, duh, "o-", color=CYAN, linewidth=1.8, markersize=4, label="时段单位线")
    ax.scatter([peak_time], [peak], s=80, color=ORANGE, zorder=5)
    ax.annotate(f"峰 {peak:.1f} m³/s", (peak_time, peak), xytext=(12, -22), textcoords="offset points", color=ORANGE)
    if std_time is not None and std_flow is not None:
        ax.plot(std_time, std_flow, "^-", color=ORANGE, linewidth=1.6, markersize=5, alpha=0.85, label="标准曲线")
        if rmse is not None:
            ax.text(0.98, 0.78, f"RMSE = {rmse:.2f} m³/s", transform=ax.transAxes, ha="right", color=ORANGE)
    ax.set_title(f"Nash 时段单位线（N={n_value:g}, K={k_value:g} h, Δt={dt:g} h）")
    if std_time is not None and std_flow is not None:
        ctx.add_visual_table("reference", ["时间 (h)", "标准流量"], std_time, std_flow)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("流量 / m³/s")
    ax.legend(loc="upper right")
    ctx.save_fig(fig, "Nash时段单位线.png", title="Nash 时段单位线")
    ctx.log("✓ 时段单位线计算完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def _sherman_units(rs, pe, h, area, adjustment):
    un = []
    previous = 0.0
    for i, value in enumerate(rs):
        if i == 0:
            previous = value / pe[0] * 10 if pe[0] else 0
        elif value == 0:
            previous = 0.0
        else:
            previous = (value - previous * pe[1] / 10) / pe[0] * 10 if pe[0] else 0
        previous *= adjustment
        un.append(previous)
    q_value = sum(un) * h * 3.6 / area if area else 0
    return un, q_value


def run_sherman_derive(ctx, params, files):
    plt = configure_matplotlib()
    pe1 = float(params.get("pe1") or 15.7)
    pe2 = float(params.get("pe2") or 5.9)
    area = float(params.get("area") or 8080)
    h = float(params.get("h") or 12)
    tolerance = float(params.get("tolerance") or 0.1)
    if pe1 <= 0 or area <= 0 or h <= 0:
        raise ValueError("PE1、流域面积和时段长必须大于 0。")
    ctx.section("读取地面径流过程")
    df = read_csv(files["rs_csv"])
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError("未找到地面径流数值列。")
    rs = numeric.iloc[:, -1].dropna().to_numpy(dtype=float)
    ctx.log(f"地面径流序列：{len(rs)} 个时段，PE=[{pe1:g}, {pe2:g}] mm，F={area:g} km²，h={h:g} h")

    adjustment = 1.0
    un, q_value = _sherman_units(rs, [pe1, pe2], h, area, adjustment)
    for _ in range(50000):
        if abs(q_value - 10) <= tolerance:
            break
        adjustment += 0.001 if q_value < 10 else -0.001
        if adjustment <= 0:
            raise ValueError("调整系数变为非正数，无法收敛。")
        un, q_value = _sherman_units(rs, [pe1, pe2], h, area, adjustment)
    ctx.log(f"调整系数：{adjustment:.4f}，单位线总量 q={q_value:.4f} mm")
    ctx.add_summary("调整系数", f"{adjustment:.4f}")
    ctx.add_summary("单位线总量", f"{q_value:.3f}", "mm")
    ctx.add_summary("单位线峰值", fmt(float(max(un)), 2), "m³/s")

    table = pd.DataFrame({"时段": np.arange(1, len(un) + 1), "单位线纵坐标(m³/s)": np.round(un, 4)})
    ctx.save_csv(table, "Sherman单位线推求结果.csv", title="Sherman 单位线")
    ctx.add_visual_table("runoff", ["时段", "地面径流"], table["时段"], rs)

    fig, ax = plt.subplots(figsize=(8.8, 5))
    ax.plot(table["时段"], un, "o-", color=CYAN, linewidth=2, markersize=5, label="单位线纵坐标")
    ax.fill_between(table["时段"], un, color=CYAN, alpha=0.1)
    peak_idx = int(np.argmax(un))
    ax.scatter([table["时段"].iloc[peak_idx]], [un[peak_idx]], s=80, color=ORANGE, zorder=5)
    ax.annotate(f"峰值 {un[peak_idx]:.2f}", (table["时段"].iloc[peak_idx], un[peak_idx]), xytext=(10, 12), textcoords="offset points", color=ORANGE)
    ax.set_title(f"Sherman 单位线推求（F={area:g} km², h={h:g} h）")
    ax.set_xlabel("时段")
    ax.set_ylabel("单位线流量 / m³/s")
    ax.legend()
    ctx.save_fig(fig, "Sherman单位线推求.png", title="Sherman 单位线")
    ctx.log("✓ Sherman 单位线推求完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_sherman_outlet(ctx, params, files):
    plt = configure_matplotlib()
    area = float(params.get("area") or 3400)
    dt = float(params.get("dt") or 6)
    if area <= 0 or dt <= 0:
        raise ValueError("流域面积和时段长必须大于 0。")
    ctx.section("读取净雨量与单位线")
    df = read_csv(files["input_csv"])
    rain_col = find_column(df, ["净雨量(mm)", "净雨量", "降雨", "R"])
    unit_col = find_column(df, ["单位线(m3/s)", "单位线(m³/s)", "单位线", "Q"])
    if rain_col is None or unit_col is None:
        raise ValueError("需要包含「净雨量(mm)」和「单位线(m3/s)」两列。")
    rain = clean_numeric(df[rain_col]).fillna(0).to_numpy(dtype=float)
    unit = clean_numeric(df[unit_col]).fillna(0).to_numpy(dtype=float)
    comparison = None
    if files.get("compare_csv"):
        try:
            comp_df = read_csv(files["compare_csv"])
            comp_col = find_column(comp_df, ["流量(实际)(m3/s)", "流量(实际)", "实际流量"])
            if comp_col:
                comparison = clean_numeric(comp_df[comp_col]).fillna(0).to_numpy(dtype=float)
                ctx.log(f"已读取实际出口流量：{len(comparison)} 点")
        except Exception as exc:
            ctx.log(f"对比数据读取失败，跳过实际流量：{exc}")

    unit_volume = float(np.sum(unit)) * dt * 3600
    unit_depth = unit_volume / (area * 1e6) * 1000
    q_value = np.convolve(rain / 10.0, unit)
    time_axis = np.arange(len(q_value))
    ctx.log(f"净雨量总和：{np.sum(rain):.2f} mm")
    ctx.log(f"单位线总水量：{unit_volume:.0f} m³，对应径流深：{unit_depth:.2f} mm")
    ctx.log(f"计算洪峰：{np.max(q_value):.2f} m³/s，峰现时段：{int(np.argmax(q_value))}")
    ctx.add_summary("净雨总量", f"{np.sum(rain):.1f}", "mm")
    ctx.add_summary("计算洪峰", fmt(float(np.max(q_value)), 1), "m³/s")
    ctx.add_summary("峰现时段", int(np.argmax(q_value)))

    table = pd.DataFrame({"时间(时段)": time_axis, "计算流量(m³/s)": np.round(q_value, 4)})
    if comparison is not None and len(comparison) >= len(q_value):
        table["实际流量(m³/s)"] = comparison[: len(q_value)]
    ctx.save_csv(table, "Sherman出口流量结果.csv", title="出口流量过程")
    ctx.add_visual_table("rain_response", ["时间(时段)", "净雨量"], np.arange(len(rain)), rain)
    if comparison is not None and len(comparison):
        ctx.add_visual_table("reference", ["时间(时段)", "实测流量"], np.arange(len(comparison)), comparison)

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 10), sharex=False)
    ax = axes[0]
    ax.bar(np.arange(len(rain)), rain, color=CYAN, alpha=0.65, width=0.8, label="净雨量")
    ax.set_ylabel("净雨量 / mm")
    ax.set_title("Sherman 单位线法 · 出口流量")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.plot(time_axis, q_value, color=GREEN, linewidth=2, label="计算流量")
    ax.scatter([int(np.argmax(q_value))], [float(np.max(q_value))], color=ORANGE, s=70, zorder=5, label=f"计算洪峰 {np.max(q_value):.0f}")
    if comparison is not None:
        comp_time = np.arange(len(comparison))
        ax.plot(comp_time, comparison, color=PINK, linewidth=1.7, alpha=0.85, label="实际流量")
        ax.scatter([int(np.argmax(comparison))], [float(np.max(comparison))], color=PINK, s=55, zorder=5)
        ctx.add_summary("实际洪峰", fmt(float(np.max(comparison)), 1), "m³/s")
    ax.set_xlabel("时间 / 时段")
    ax.set_ylabel("流量 / m³/s")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.bar(np.arange(len(unit)), unit, color=ORANGE, alpha=0.7, width=0.8, label="单位线")
    ax.set_xlabel("时段")
    ax.set_ylabel("单位线 / m³/s")
    ax.legend(loc="upper right")
    fig.tight_layout()
    ctx.save_fig(fig, "Sherman出口流量.png", title="出口流量三联图")
    ctx.log("✓ Sherman 出口流量计算完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}

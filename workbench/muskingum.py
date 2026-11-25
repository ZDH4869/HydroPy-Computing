"""Muskingum parameter calibration and downstream routing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    BLUE,
    CYAN,
    GREEN,
    ORANGE,
    PINK,
    VIOLET,
    YELLOW,
    configure_matplotlib,
    find_column,
    fmt,
    read_csv,
)


def _numeric_frame(path):
    df = read_csv(path)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        raise ValueError("需要至少两列流量数据：上游流量、下游流量。")
    return df, numeric


def run_muskingum_xk(ctx, params, files):
    plt = configure_matplotlib()
    delta_t = float(params.get("delta_t") or 1.0)
    if delta_t <= 0:
        raise ValueError("时间步长必须大于 0。")

    ctx.section("读取上下游流量")
    df, numeric = _numeric_frame(files["pairs_csv"])
    upstream = numeric.iloc[:, 0].dropna().to_numpy(dtype=float)
    downstream = numeric.iloc[:, 1].dropna().to_numpy(dtype=float)
    n = min(len(upstream) - 1, len(downstream) - 1)
    if n < 3:
        raise ValueError("数据点太少，无法率定 X、K。")
    # Match the original script: drop the first upstream value and the last downstream value.
    inflow = upstream[1 : n + 1]
    outflow = downstream[:n]
    ctx.log(f"上游序列：{len(upstream)} 点，下游序列：{len(downstream)} 点，参与计算：{n} 点")
    ctx.log(f"Δt = {delta_t:g} h")

    delta_s = np.zeros(n, dtype=float)
    for i in range(n - 1):
        delta_s[i + 1] = (
            0.5 * (inflow[i] + inflow[i + 1]) * delta_t
            - 0.5 * (outflow[i] + outflow[i + 1]) * delta_t
        )
    storage = np.cumsum(delta_s)

    best_corr = -np.inf
    best_x = 0.0
    best_q = None
    scan = []
    for x_int in range(0, 51):
        x = x_int / 100.0
        q_calc = x * inflow + (1 - x) * outflow
        if np.std(storage[1:]) < 1e-12 or np.std(q_calc[1:]) < 1e-12:
            corr = 0.0
        else:
            corr = float(np.corrcoef(storage[1:], q_calc[1:])[0, 1])
        scan.append({"X": x, "相关系数": corr})
        if corr > best_corr:
            best_corr = corr
            best_x = x
            best_q = q_calc.copy()
    if best_q is None:
        raise ValueError("无法完成 X 扫描。")
    slope, intercept = np.polyfit(best_q[1:], storage[1:], 1)
    k_value = float(slope)
    ctx.log(f"最优 X = {best_x:.2f}，相关系数 = {best_corr:.4f}")
    ctx.log(f"拟合 K = {k_value:.4f}")
    ctx.add_summary("最优 X", f"{best_x:.2f}")
    ctx.add_summary("拟合 K", f"{k_value:.4f}", "h")
    ctx.add_summary("相关系数", f"{best_corr:.4f}")

    relation = pd.DataFrame(
        {
            "时段": np.arange(1, n + 1),
            "上游流量": inflow,
            "下游流量": outflow,
            "蓄水量S": storage,
            "加权流量Q'": best_q,
        }
    )
    ctx.save_csv(relation, "马斯京根蓄量关系.csv", title="蓄量-流量关系")
    ctx.save_csv(pd.DataFrame(scan), "马斯京根X扫描.csv", title="X 扫描结果")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    ax = axes[0]
    ax.scatter(best_q[1:], storage[1:], s=55, color=CYAN, edgecolor="#0f172a", linewidth=0.8, zorder=3, label="样本点")
    fit_x = np.linspace(float(np.min(best_q[1:])), float(np.max(best_q[1:])), 60)
    ax.plot(fit_x, slope * fit_x + intercept, color=ORANGE, linewidth=2, label=f"K={k_value:.4f}")
    ax.set_title("蓄量-流量关系")
    ax.set_xlabel("加权流量 Q' / m³/s")
    ax.set_ylabel("蓄水量 S / m³")
    ax.legend()

    ax = axes[1]
    xs = [row["X"] for row in scan]
    ys = [row["相关系数"] for row in scan]
    ax.plot(xs, ys, color=VIOLET, linewidth=2)
    ax.scatter([best_x], [best_corr], color=PINK, s=70, zorder=4)
    ax.annotate(f"X={best_x:.2f}", (best_x, best_corr), xytext=(8, 8), textcoords="offset points", color=PINK)
    ax.set_title("X 与相关系数")
    ax.set_xlabel("X")
    ax.set_ylabel("相关系数")
    fig.tight_layout()
    ctx.save_fig(fig, "马斯京根参数率定.png", title="参数率定")
    ctx.log("✓ 马斯京根参数率定完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_muskingum_route(ctx, params, files):
    plt = configure_matplotlib()
    k_value = float(params.get("k") or 1.1002)
    x_value = float(params.get("x") or 0.27)
    dt = float(params.get("dt") or 1.0)
    initial_q = float(params.get("initial_q") or 0)
    if k_value <= 0 or dt <= 0:
        raise ValueError("K 和时间步长必须大于 0。")

    ctx.section("读取上游流量")
    df = read_csv(files["upstream_csv"])
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.shape[1] == 0:
        raise ValueError("未找到上游流量数值列。")
    inflow = numeric.iloc[:, -1].dropna().to_numpy(dtype=float)
    if len(inflow) < 2:
        raise ValueError("上游流量至少需要两个时段。")

    denominator = k_value - k_value * x_value + 0.5 * dt
    if abs(denominator) < 1e-12:
        raise ValueError("K - KX + Δt/2 为 0，无法计算系数。")
    c0 = (0.5 * dt - k_value * x_value) / denominator
    c1 = (0.5 * dt + k_value * x_value) / denominator
    c2 = (k_value - k_value * x_value - 0.5 * dt) / denominator
    ctx.log(f"K={k_value:g} h, X={x_value:g}, Δt={dt:g} h")
    ctx.log(f"C0={c0:.5f}, C1={c1:.5f}, C2={c2:.5f}, 合计={(c0 + c1 + c2):.5f}")
    ctx.add_summary("C0", f"{c0:.4f}")
    ctx.add_summary("C1", f"{c1:.4f}")
    ctx.add_summary("C2", f"{c2:.4f}")

    outflow = np.empty(len(inflow), dtype=float)
    outflow[0] = initial_q
    for i in range(1, len(inflow)):
        outflow[i] = c0 * inflow[i] + c1 * inflow[i - 1] + c2 * outflow[i - 1]

    result = pd.DataFrame(
        {
            "时段": np.arange(len(inflow)),
            "上游流量I": inflow,
            "下游流量Q": np.round(outflow, 4),
        }
    )
    ctx.save_csv(result, "马斯京根下游流量结果.csv", title="下游流量过程")
    ctx.add_summary("下游洪峰", fmt(float(np.max(outflow)), 1), "m³/s")
    ctx.add_summary("峰现时段", int(np.argmax(outflow)))

    fig, ax = plt.subplots(figsize=(8.8, 5))
    time_axis = np.arange(len(inflow))
    ax.plot(time_axis, inflow, "--o", color=CYAN, linewidth=1.8, markersize=4, label="上游入流")
    ax.plot(time_axis, outflow, "-s", color=ORANGE, linewidth=2, markersize=5, label="下游出流")
    ax.fill_between(time_axis, outflow, color=ORANGE, alpha=0.1)
    ax.set_title("马斯京根流量演算")
    ax.set_xlabel("时段")
    ax.set_ylabel("流量 / m³/s")
    ax.legend()
    ctx.save_fig(fig, "马斯京根下游流量.png", title="下游流量过程")
    ctx.log("✓ 马斯京根下游演算完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}

"""Reservoir curve interpolation and flood routing algorithms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

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
    read_flood_series,
)


def _numeric_columns(df):
    converted = df.copy()
    for col in converted.columns:
        converted[col] = clean_numeric(converted[col])
    return converted.dropna(axis=1, how="all")


def _load_inflow(path):
    df = read_csv(path)
    q_col = find_column(df, ["Q/(m3/s-1)", "Q/(m³·s⁻¹)", "Q", "入库流量", "流量"])
    if q_col is None:
        numeric = _numeric_columns(df)
        if numeric.shape[1] == 0:
            raise ValueError("未找到入库流量列。")
        q_col = numeric.columns[-1]
    if "时间t/h" in df.columns:
        t = clean_numeric(df["时间t/h"]).to_numpy(dtype=float)
    else:
        times, q, _ = read_flood_series(df)
        t = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
        return t, q
    q = clean_numeric(df[q_col]).to_numpy(dtype=float)
    valid = np.isfinite(t) & np.isfinite(q)
    return t[valid], q[valid]


def _load_curves(storage_path, discharge_path):
    storage = read_csv(storage_path)
    z_storage = find_column(storage, ["水位Z/m", "水位", "Z"])
    v_storage = find_column(storage, ["库容V/万m3", "库容", "V"])
    if z_storage is None or v_storage is None:
        numeric = _numeric_columns(storage)
        if numeric.shape[1] < 2:
            raise ValueError("水位-库容曲线至少需要两列数值。")
        z_storage, v_storage = numeric.columns[:2]
    z1 = clean_numeric(storage[z_storage]).to_numpy(dtype=float)
    v1 = clean_numeric(storage[v_storage]).to_numpy(dtype=float)
    valid = np.isfinite(z1) & np.isfinite(v1)
    z1, v1 = z1[valid], v1[valid]
    order = np.argsort(z1)
    z1, v1 = z1[order], v1[order]

    discharge = read_csv(discharge_path)
    z_discharge = find_column(discharge, ["水位Z/m", "水位", "Z"])
    q_discharge = find_column(discharge, ["下泄流量q/(m3·s)", "下泄流量", "q"])
    if z_discharge is None or q_discharge is None:
        numeric = _numeric_columns(discharge)
        if numeric.shape[1] < 2:
            raise ValueError("水位-下泄流量曲线至少需要两列数值。")
        z_discharge, q_discharge = numeric.columns[:2]
    z2 = clean_numeric(discharge[z_discharge]).to_numpy(dtype=float)
    q2 = clean_numeric(discharge[q_discharge]).to_numpy(dtype=float)
    valid = np.isfinite(z2) & np.isfinite(q2)
    z2, q2 = z2[valid], q2[valid]
    order = np.argsort(z2)
    z2, q2 = z2[order], q2[order]

    if len(z1) < 2 or len(z2) < 2:
        raise ValueError("曲线数据点不足。")
    storage_interp = interp1d(z1, v1, kind="linear", bounds_error=False, fill_value="extrapolate")
    discharge_interp = interp1d(z2, q2, kind="linear", bounds_error=False, fill_value="extrapolate")
    v_z_interp = interp1d(v1, z1, kind="linear", bounds_error=False, fill_value="extrapolate")
    return storage_interp, discharge_interp, v_z_interp, (z1, v1), (z2, q2)


def _interp_values(x, y, target_x, method, poly_order=3):
    method = (method or "linear").lower()
    if method == "nearest":
        return interp1d(x, y, kind="nearest", bounds_error=False, fill_value="extrapolate")(target_x)
    if method == "polynomial":
        order = max(1, min(int(poly_order), len(x) - 1))
        return np.polyval(np.polyfit(x, y, order), target_x)
    if method == "spline":
        if len(x) >= 4:
            return CubicSpline(x, y, extrapolate=True)(target_x)
        return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")(target_x)
    if method == "log":
        tiny = 1e-12
        sign = -1.0 if np.any(y < 0) else 1.0
        base = np.log(np.abs(y) + tiny)
        return sign * np.exp(interp1d(x, base, kind="linear", bounds_error=False, fill_value="extrapolate")(target_x))
    return np.interp(target_x, x, y)


def run_curve_interp(ctx, params, files):
    plt = configure_matplotlib()
    ctx.section("读取曲线数据")
    df = read_csv(files["curve_csv"])
    if df.empty:
        raise ValueError("CSV 文件为空。")
    x_col = params.get("x_col") or df.columns[0]
    if x_col not in df.columns:
        x_col = find_column(df, [str(x_col)]) or df.columns[0]
    method = params.get("method") or "linear"
    x_step = float(params.get("x_step") or 0.1)
    poly_order = int(float(params.get("poly_order") or 3))
    x_precision = int(float(params.get("x_precision") or 0))
    y_precision = int(float(params.get("y_precision") or 0))
    if x_step <= 0:
        raise ValueError("自变量步长必须大于 0。")
    numeric = df.copy()
    numeric[x_col] = clean_numeric(numeric[x_col])
    x_raw = numeric[x_col].dropna().to_numpy(dtype=float)
    if len(x_raw) < 2:
        raise ValueError("自变量有效点不足。")
    x_min, x_max = float(np.min(x_raw)), float(np.max(x_raw))
    target_x = np.round(np.arange(x_min, x_max + x_step * 0.5, x_step), max(0, x_precision))
    if len(target_x) < 2:
        raise ValueError("插值步长过大，没有生成足够的目标点。")
    y_cols = [c for c in df.columns if c != x_col]
    if not y_cols:
        raise ValueError("除了自变量外没有可插值的列。")
    ctx.log(f"自变量：{x_col}，范围 [{x_min:g}, {x_max:g}]，步长 {x_step:g}，方法 {method}")
    ctx.log(f"因变量列：{', '.join(map(str, y_cols))}")

    result = pd.DataFrame({x_col: target_x})
    for col in y_cols:
        y_raw_series = clean_numeric(df[col])
        valid = numeric[x_col].notna() & y_raw_series.notna()
        xv = numeric.loc[valid, x_col].to_numpy(dtype=float)
        yv = y_raw_series[valid].to_numpy(dtype=float)
        if len(xv) < 2:
            ctx.log(f"跳过 {col}：有效点不足。")
            continue
        order = np.argsort(xv)
        xv, yv = xv[order], yv[order]
        unique = ~pd.Series(xv).duplicated()
        xv, yv = xv[unique.to_numpy()], yv[unique.to_numpy()]
        values = _interp_values(xv, yv, target_x, method, poly_order)
        result[col] = np.round(values, max(0, y_precision))
        ctx.log(f"{col}：{len(xv)} 个原始点 → {len(target_x)} 个插值点")
        ctx.add_summary(f"{col} 原始点", len(xv), "个")
    ctx.add_summary("插值方法", method)
    ctx.add_summary("输出点数", len(result), "个")

    ctx.save_csv(result, "曲线插值结果.csv", title="曲线插值结果")
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    colors = [CYAN, ORANGE, GREEN, VIOLET, PINK, YELLOW]
    for idx, col in enumerate(y_cols):
        y_series = clean_numeric(df[col])
        valid = numeric[x_col].notna() & y_series.notna()
        if not valid.any() or col not in result.columns:
            continue
        ax.scatter(numeric.loc[valid, x_col], y_series[valid], s=32, color=colors[idx % len(colors)], alpha=0.55, zorder=3)
        ax.plot(target_x, result[col], color=colors[idx % len(colors)], linewidth=2, label=str(col))
    ax.set_title(f"曲线插值 · {method}")
    ax.set_xlabel(str(x_col))
    ax.set_ylabel("因变量")
    ax.legend(loc="best")
    ctx.save_fig(fig, "曲线插值结果.png", title="插值曲线")
    ctx.log("✓ 曲线插值完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_flood_trial(ctx, params, files):
    plt = configure_matplotlib()
    initial_discharge = float(params.get("initial_discharge") or 173.9)
    initial_v = float(params.get("initial_v") or 6450)
    initial_z = float(params.get("initial_z") or 38)
    v_tolerance = float(params.get("v_tolerance") or 3)
    z_min = float(params.get("z_min") or 36)
    z_max = float(params.get("z_max") or 41)
    decimals = int(float(params.get("decimals") or 1))
    unit_conversion = float(params.get("unit_conversion") or 1e-4)
    if z_min >= z_max:
        raise ValueError("试算水位下限必须小于上限。")
    if v_tolerance <= 0:
        raise ValueError("库容误差必须大于 0。")

    ctx.section("读取三条曲线")
    t, q_in = _load_inflow(files["inflow_csv"])
    storage_interp, discharge_interp, v_z_interp, storage_data, discharge_data = _load_curves(
        files["storage_csv"], files["discharge_csv"]
    )
    ctx.log(f"入库流量：{len(t)} 行，时间 {t[0]:g}–{t[-1]:g} h")
    ctx.log(f"水位-库容：{len(storage_data[0])} 点；水位-下泄流量：{len(discharge_data[0])} 点")

    results = pd.DataFrame(
        {
            "时间t/h": t,
            "入库流量Q/(m³·s⁻¹)": q_in,
            "时段平均入库流量/(m³·s⁻¹)": np.zeros(len(t)),
            "下泄流量q/(m³·s⁻¹)": np.zeros(len(t)),
            "时段平均下泄流量/(m³·s⁻¹)": np.zeros(len(t)),
            "时段内水库存水量变化ΔV/万m³": np.zeros(len(t)),
            "水库存水量V/万m³": np.zeros(len(t)),
            "水库水位Z/m": np.zeros(len(t)),
        }
    )
    results.loc[0, "下泄流量q/(m³·s⁻¹)"] = initial_discharge
    results.loc[0, "水库存水量V/万m³"] = initial_v
    results.loc[0, "水库水位Z/m"] = initial_z

    step = 10 ** (-decimals)
    z_candidates = np.round(np.arange(z_min, z_max + step * 0.5, step), decimals)
    success = 0
    for i in range(1, len(t)):
        current_avg_inflow = (q_in[i - 1] + q_in[i]) / 2
        results.loc[i, "时段平均入库流量/(m³·s⁻¹)"] = current_avg_inflow
        time_diff = t[i] - t[i - 1]
        prev = results.iloc[i - 1]
        best = None
        min_error = float("inf")
        for z_candidate in z_candidates:
            try:
                v_candidate = float(storage_interp(z_candidate))
                q_candidate = float(discharge_interp(z_candidate))
            except Exception:
                continue
            avg_q = (prev["下泄流量q/(m³·s⁻¹)"] + q_candidate) / 2
            delta_v = (current_avg_inflow - avg_q) * time_diff * 3600 * unit_conversion
            v_current = prev["水库存水量V/万m³"] + delta_v
            if abs(v_current - v_candidate) <= v_tolerance:
                try:
                    z_check = float(v_z_interp(v_current))
                    if z_min <= z_check <= z_max:
                        error = abs(v_current - v_candidate)
                        if error < min_error:
                            min_error = error
                            best = (z_candidate, q_candidate, avg_q, delta_v, v_current)
                except Exception:
                    continue
        if best is None:
            # Continuous refinement between the discrete trial levels: solve
            # V_prev + (Q_avg - (q_prev + q(Z))/2) * dt = V(Z).
            try:
                from scipy.optimize import brentq

                def balance(z):
                    v_candidate = float(storage_interp(z))
                    q_candidate = float(discharge_interp(z))
                    avg_q = (prev["下泄流量q/(m³·s⁻¹)"] + q_candidate) / 2
                    delta_v = (current_avg_inflow - avg_q) * time_diff * 3600 * unit_conversion
                    v_current = prev["水库存水量V/万m³"] + delta_v
                    return v_current - v_candidate

                left, right = balance(z_min), balance(z_max)
                if np.isfinite(left) and np.isfinite(right) and left * right < 0:
                    z_root = float(brentq(balance, z_min, z_max, xtol=10 ** (-(decimals + 2))))
                    q_root = float(discharge_interp(z_root))
                    avg_q_root = (prev["下泄流量q/(m³·s⁻¹)"] + q_root) / 2
                    delta_v_root = (current_avg_inflow - avg_q_root) * time_diff * 3600 * unit_conversion
                    v_root = prev["水库存水量V/万m³"] + delta_v_root
                    best = (z_root, q_root, avg_q_root, delta_v_root, v_root)
                    ctx.log(f"第 {i + 1} 行：在试算网格之间插值求得 Z={z_root:.4f} m")
            except Exception:
                best = None
        if best:
            z_value, q_value, avg_q, delta_v, v_value = best
            results.loc[i, "下泄流量q/(m³·s⁻¹)"] = q_value
            results.loc[i, "时段平均下泄流量/(m³·s⁻¹)"] = avg_q
            results.loc[i, "时段内水库存水量变化ΔV/万m³"] = delta_v
            results.loc[i, "水库存水量V/万m³"] = v_value
            results.loc[i, "水库水位Z/m"] = z_value
            success += 1
        else:
            # Approximation, mirroring the original script's fallback.
            prev_z = prev["水库水位Z/m"]
            prev_q = prev["下泄流量q/(m³·s⁻¹)"]
            try:
                approx_q = float(discharge_interp(prev_z))
                approx_v = float(storage_interp(prev_z))
            except Exception:
                approx_q, approx_v = prev_q, prev["水库存水量V/万m³"]
            avg_q = (prev_q + approx_q) / 2
            delta_v = (current_avg_inflow - avg_q) * time_diff * 3600 * unit_conversion
            v_value = prev["水库存水量V/万m³"] + delta_v
            try:
                z_value = float(v_z_interp(v_value))
            except Exception:
                z_value = prev_z
            results.loc[i, "下泄流量q/(m³·s⁻¹)"] = approx_q
            results.loc[i, "时段平均下泄流量/(m³·s⁻¹)"] = avg_q
            results.loc[i, "时段内水库存水量变化ΔV/万m³"] = delta_v
            results.loc[i, "水库存水量V/万m³"] = v_value
            results.loc[i, "水库水位Z/m"] = z_value
            ctx.log(f"第 {i + 1} 行未找到满足误差的解，使用近似值。")

    ctx.log(f"试算成功：{success}/{len(t) - 1} 行")
    ctx.add_summary("最大下泄流量", fmt(float(results["下泄流量q/(m³·s⁻¹)"].max()), 1), "m³/s")
    ctx.add_summary("最高水位", fmt(float(results["水库水位Z/m"].max()), 2), "m")
    ctx.add_summary("最终库容", fmt(float(results.iloc[-1]["水库存水量V/万m³"]), 1), "万m³")
    ctx.add_summary("试算成功率", f"{success}/{max(1, len(t) - 1)}")

    ctx.save_csv(results, "调洪试算法结果.csv", title="试算法调洪结果")
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=True)
    ax = axes[0]
    ax.plot(t, q_in, color=CYAN, linewidth=1.9, label="入库流量")
    ax.plot(t, results["下泄流量q/(m³·s⁻¹)"], color=ORANGE, linewidth=2, label="下泄流量")
    ax.scatter([t[int(np.argmax(q_in))]], [np.max(q_in)], color=CYAN, s=55, zorder=5)
    ax.scatter([t[int(np.argmax(results["下泄流量q/(m³·s⁻¹)"].to_numpy()))]], [results["下泄流量q/(m³·s⁻¹)"].max()], color=ORANGE, s=55, zorder=5)
    ax.set_ylabel("流量 / m³/s")
    ax.set_title("试算法调洪 · 流量过程")
    ax.legend()

    ax = axes[1]
    ax.plot(t, results["水库水位Z/m"], color=GREEN, linewidth=2.1, label="水库水位")
    ax.scatter([t[int(np.argmax(results["水库水位Z/m"].to_numpy()))]], [results["水库水位Z/m"].max()], color=PINK, s=60, zorder=5)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("水位 / m")
    ax.set_title("试算法调洪 · 水位过程")
    ax.legend()
    ctx.save_fig(fig, "调洪试算法结果.png", title="试算法调洪曲线")
    ctx.log("✓ 试算法调洪完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_flood_rk4(ctx, params, files):
    plt = configure_matplotlib()
    dt_hours = float(params.get("dt_hours") or 3)
    initial_z = float(params.get("initial_z") or 38)
    initial_v = float(params.get("initial_v") or 6450)
    if dt_hours <= 0:
        raise ValueError("时间步长必须大于 0。")
    dt = dt_hours * 3600.0
    ctx.section("读取三条曲线")
    t, q_in = _load_inflow(files["inflow_csv"])
    storage_interp, discharge_interp, v_z_interp, storage_data, discharge_data = _load_curves(
        files["storage_csv"], files["discharge_csv"]
    )
    # RK4 works in m³ internally; the trial method works in 万m³.
    z1, v1_wan = storage_data
    storage_interp_m3 = interp1d(v1_wan * 1e4, z1, kind="linear", bounds_error=False, fill_value="extrapolate")
    ctx.log(f"入库流量：{len(t)} 行；RK4 时间步长：{dt_hours:g} h")
    if len(t) > 1:
        observed = np.median(np.diff(t))
        if abs(observed - dt_hours) > 1e-6:
            ctx.log(f"提示：数据时间间隔中位数为 {observed:g} h，当前 RK4 步长为 {dt_hours:g} h。")

    v0 = initial_v * 1e4
    q0 = float(discharge_interp(initial_z))
    v_list = [v0]
    z_list = [initial_z]
    q_list = [q0]

    def d_vdt(v, q_avg):
        z = float(storage_interp_m3(v))
        return q_avg - float(discharge_interp(z))

    for i in range(1, len(t)):
        q_avg = 0.5 * (q_in[i - 1] + q_in[i])
        v_prev = v_list[-1]
        k1 = d_vdt(v_prev, q_avg)
        k2 = d_vdt(v_prev + 0.5 * dt * k1, q_avg)
        k3 = d_vdt(v_prev + 0.5 * dt * k2, q_avg)
        k4 = d_vdt(v_prev + dt * k3, q_avg)
        v_new = v_prev + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        z_new = float(storage_interp_m3(v_new))
        q_new = float(discharge_interp(z_new))
        v_list.append(v_new)
        z_list.append(z_new)
        q_list.append(q_new)

    results = pd.DataFrame(
        {
            "时间t/h": t,
            "入库流量Q/(m³·s⁻¹)": q_in,
            "下泄流量q/(m³·s⁻¹)": np.round(q_list, 4),
            "水库存水量V/万m³": np.round(np.array(v_list) * 1e-4, 4),
            "水库水位Z/m": np.round(z_list, 4),
        }
    )
    ctx.log(f"最大下泄流量：{np.max(q_list):.2f} m³/s")
    ctx.log(f"最高水位：{np.max(z_list):.3f} m")
    ctx.add_summary("最大下泄流量", fmt(float(np.max(q_list)), 1), "m³/s")
    ctx.add_summary("最高水位", fmt(float(np.max(z_list)), 2), "m")
    ctx.add_summary("最终库容", fmt(float(v_list[-1] * 1e-4), 1), "万m³")
    ctx.save_csv(results, "调洪RK4结果.csv", title="RK4 调洪结果")

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 8.2), sharex=True)
    ax = axes[0]
    ax.plot(t, q_in, color=CYAN, linewidth=1.9, label="入库流量")
    ax.plot(t, q_list, color=ORANGE, linewidth=2, label="下泄流量")
    ax.scatter([t[int(np.argmax(q_in))]], [np.max(q_in)], color=CYAN, s=55, zorder=5)
    ax.scatter([t[int(np.argmax(q_list))]], [np.max(q_list)], color=ORANGE, s=55, zorder=5)
    ax.set_ylabel("流量 / m³/s")
    ax.set_title("RK4 调洪 · 流量过程")
    ax.legend()

    ax = axes[1]
    ax.plot(t, z_list, color=GREEN, linewidth=2.1, label="水库水位")
    ax.scatter([t[int(np.argmax(z_list))]], [np.max(z_list)], color=PINK, s=60, zorder=5)
    ax.set_xlabel("时间 / h")
    ax.set_ylabel("水位 / m")
    ax.set_title("RK4 调洪 · 水位过程")
    ax.legend()
    ctx.save_fig(fig, "调洪RK4结果.png", title="RK4 调洪曲线")
    ctx.log("✓ RK4 调洪完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}

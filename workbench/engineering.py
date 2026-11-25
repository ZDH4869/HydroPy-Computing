"""Engineering hydrology algorithms: flood interpolation and maximum volumes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate
from scipy.interpolate import CubicSpline, interp1d

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
    format_duration,
    parse_duration_text,
    read_csv,
    read_flood_series,
)


def _load_series(path):
    df = read_csv(path)
    times, flow, label = read_flood_series(df)
    return df, pd.Series(times), flow, label


def _resampled_curve(times, flow, freq):
    """Interpolate a datetime series onto a regular grid using numeric seconds."""
    order = np.argsort(times.values)
    times = times.iloc[order].reset_index(drop=True)
    flow = np.asarray(flow, dtype=float)[order]
    # collapse duplicate timestamps
    dedup = ~times.duplicated()
    times = times[dedup].reset_index(drop=True)
    flow = flow[dedup.to_numpy()]
    x = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    step = pd.Timedelta(freq).total_seconds()
    if step <= 0:
        raise ValueError("插值步长必须大于 0。")
    full_x = np.arange(0, x[-1] + step * 0.5, step)
    full_y = np.interp(full_x, x, flow)
    full_times = times.iloc[0] + pd.to_timedelta(full_x, unit="s")
    return pd.Series(full_times), full_y, x, flow


def _method_values(x, y, target_x, method, poly_order=2):
    method = (method or "linear").lower()
    if method == "nearest":
        f = interp1d(x, y, kind="nearest", bounds_error=False, fill_value="extrapolate")
        return f(target_x)
    if method == "polynomial":
        order = max(1, min(int(poly_order), max(1, len(x) - 1)))
        return np.polyval(np.polyfit(x, y, order), target_x)
    if method == "spline":
        if len(x) >= 4:
            return CubicSpline(x, y, extrapolate=True)(target_x)
        return interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")(target_x)
    if method == "log":
        tiny = 1e-12
        if np.any(y <= 0):
            # Match the original tool's protective offset for non-positive series.
            log_y = np.log(np.abs(y) + tiny)
        else:
            log_y = np.log(y)
        return np.exp(interp1d(x, log_y, kind="linear", bounds_error=False, fill_value="extrapolate")(target_x))
    return np.interp(target_x, x, y)


def _time_columns(index: pd.Series):
    idx = pd.DatetimeIndex(index)
    return pd.DataFrame(
        {
            "年": idx.year,
            "月": idx.month,
            "日": idx.day,
            "时": idx.hour,
        }
    )


def run_linear_interp(ctx, params, files):
    plt = configure_matplotlib()
    freq = params.get("freq") or "1h"
    ctx.section("读取原始洪水过程")
    df, times, flow, label = _load_series(files["flood_csv"])
    ctx.log(f"数据行数：{len(df)}，流量列：{label}")
    ctx.log(f"时间范围：{times.iloc[0]:%Y-%m-%d %H:%M} → {times.iloc[-1]:%Y-%m-%d %H:%M}")
    ctx.add_summary("原始数据点", len(df), "个")
    ctx.add_summary("原始洪峰", fmt(float(np.max(flow)), 1), "m³/s")

    full_times, full_flow, _, _ = _resampled_curve(times, flow, freq)
    result = _time_columns(full_times)
    result["Q"] = np.round(full_flow, 4)
    ctx.log(f"插值步长：{freq}，输出 {len(result)} 个等间隔点。")
    ctx.add_summary("插值点数", len(result), "个")
    ctx.add_summary("插值洪峰", fmt(float(np.max(full_flow)), 1), "m³/s")

    ctx.save_csv(result, "等间隔线性插值结果.csv", title="等间隔插值结果")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(times, flow, "o", color=ORANGE, markersize=4, alpha=0.75, label="原始观测点")
    ax.plot(full_times, full_flow, color=CYAN, linewidth=2.0, label=f"线性插值（{freq}）")
    peak_idx = int(np.argmax(full_flow))
    ax.scatter(full_times.iloc[peak_idx], full_flow[peak_idx], s=70, color=PINK, zorder=5)
    ax.annotate(
        f"峰值 {full_flow[peak_idx]:.1f} m³/s",
        (full_times.iloc[peak_idx], full_flow[peak_idx]),
        xytext=(12, 18),
        textcoords="offset points",
        color=PINK,
        fontsize=10,
    )
    ax.set_title("洪水过程线性插值")
    ax.set_xlabel("时间")
    ax.set_ylabel("流量 / m³/s")
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=22)
    ctx.save_fig(fig, "洪水过程线性插值.png", title="插值过程线")
    ctx.log("✓ 线性插值完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_multi_interp(ctx, params, files):
    plt = configure_matplotlib()
    freq = params.get("freq") or "1h"
    poly_order = int(float(params.get("poly_order") or 2))
    ctx.section("读取原始洪水过程")
    df, times, flow, label = _load_series(files["flood_csv"])
    ctx.log(f"数据行数：{len(df)}，流量列：{label}")

    full_times, _, x, y = _resampled_curve(times, flow, freq)
    step = pd.Timedelta(freq).total_seconds()
    target_x = np.arange(0, x[-1] + step * 0.5, step)
    methods = {
        "线性插值": _method_values(x, y, target_x, "linear"),
        "最近邻插值": _method_values(x, y, target_x, "nearest"),
        "多项式插值": _method_values(x, y, target_x, "polynomial", poly_order),
        "样条插值": _method_values(x, y, target_x, "spline"),
    }
    result = _time_columns(pd.Series(full_times))
    for name, values in methods.items():
        result[f"Q_{name}"] = np.round(values, 4)

    ctx.log(f"频率：{freq}，多项式阶数：{poly_order}，输出 {len(result)} 行。")
    for name, values in methods.items():
        ctx.log(f"{name}洪峰：{np.max(values):.2f} m³/s")
        ctx.add_summary(name, fmt(float(np.max(values)), 1), "m³/s")
    ctx.add_summary("插值点数", len(result), "个")
    ctx.save_csv(result, "多种插值对比结果.csv", title="多种插值结果")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(times, flow, "o", color="#f8fafc", markersize=3, alpha=0.5, label="原始观测")
    colors = [CYAN, ORANGE, VIOLET, GREEN]
    for (name, values), color in zip(methods.items(), colors):
        ax.plot(full_times, values, linewidth=1.8, color=color, label=name)
    ax.set_title("四种等间隔插值方法对比")
    ax.set_xlabel("时间")
    ax.set_ylabel("流量 / m³/s")
    ax.legend(loc="upper right", ncol=2)
    fig.autofmt_xdate(rotation=22)
    ctx.save_fig(fig, "多种插值对比.png", title="方法对比图")
    ctx.log("✓ 多方法插值完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def _integrate_window(x_seconds, y, smooth_points=1000):
    trapezoid = float(integrate.trapezoid(y, x_seconds))
    simpson = float(integrate.simpson(y, x_seconds)) if len(y) >= 3 else None
    interpolation = None
    if len(y) >= 4:
        try:
            spline = CubicSpline(x_seconds, y)
            dense_x = np.linspace(x_seconds[0], x_seconds[-1], max(100, int(smooth_points)))
            interpolation = float(integrate.trapezoid(spline(dense_x), dense_x))
        except Exception:
            interpolation = None
    return trapezoid, simpson, interpolation


def _max_volume_table(times, flow, durations, smooth_points):
    origin = times.iloc[0]
    x_seconds = (times - origin).dt.total_seconds().to_numpy(dtype=float)
    x_days = x_seconds / 86400.0
    rows = []
    windows = {}
    for duration in durations:
        if duration <= 0:
            continue
        max_row = None
        max_value = -np.inf
        for start in range(len(x_days)):
            target = x_days[start] + duration
            if target > x_days[-1] + 1e-12:
                break
            end = int(np.searchsorted(x_days, target, side="right") - 1)
            if end <= start:
                continue
            xs = x_seconds[start : end + 1]
            ys = flow[start : end + 1]
            trap, simp, interp = _integrate_window(xs, ys, smooth_points)
            if trap > max_value:
                max_value = trap
                max_row = {
                    "duration": duration,
                    "start": start,
                    "end": end,
                    "start_label": times.iloc[start].strftime("%Y-%m-%d %H:%M"),
                    "end_label": times.iloc[end].strftime("%Y-%m-%d %H:%M"),
                    "trapezoid": trap,
                    "simpson": simp,
                    "interpolation": interp,
                }
        if max_row:
            windows[duration] = max_row
            rows.append(
                {
                    "统计时段": format_duration(duration),
                    "开始时间": max_row["start_label"],
                    "结束时间": max_row["end_label"],
                    "梯形法洪量/万m³": round(max_row["trapezoid"] / 1e4, 4),
                    "辛普森法洪量/万m³": None if max_row["simpson"] is None else round(max_row["simpson"] / 1e4, 4),
                    "插值积分洪量/万m³": None if max_row["interpolation"] is None else round(max_row["interpolation"] / 1e4, 4),
                }
            )
    return pd.DataFrame(rows), windows


def _run_max_volume(ctx, params, files, *, title, table_name, image_name):
    plt = configure_matplotlib()
    durations = parse_duration_text(params.get("durations"), [1 / 24, 1, 3, 5])
    smooth_points = int(float(params.get("smooth_points") or 1000))
    ctx.section("读取等间隔洪水过程")
    df, times, flow, label = _load_series(files["hourly_csv"])
    # If the uploaded series is not regular, resample it to hourly first.
    diffs = times.diff().dt.total_seconds().dropna()
    if len(diffs) and abs(float(diffs.median()) - 3600) > 1:
        ctx.log(f"检测到非 1h 间隔（中位步长 {diffs.median() / 3600:.2f}h），先插值为 1h 序列。")
        times, flow, _, _ = _resampled_curve(times, flow, "1h")
    ctx.log(f"数据行数：{len(df)}，时间范围：{times.iloc[0]:%Y-%m-%d %H:%M} → {times.iloc[-1]:%Y-%m-%d %H:%M}")
    ctx.add_summary("数据点数", len(times), "个")
    ctx.add_summary("最大流量", fmt(float(np.max(flow)), 1), "m³/s")

    result, windows = _max_volume_table(times, flow, durations, smooth_points)
    if result.empty:
        raise ValueError("数据时长不足以计算所选时段，请减小统计时段或提供更长的数据。")
    ctx.save_csv(result, table_name, title="时段最大洪量")
    for _, row in result.iterrows():
        ctx.log(
            f"{row['统计时段']}：{row['梯形法洪量/万m³']} 万m³，"
            f"{row['开始时间']} → {row['结束时间']}"
        )
    # Summary: first few key durations.
    key = [1 / 24, 1, 3, 5, 7]
    for duration in key:
        if duration in windows:
            ctx.add_summary(format_duration(duration), fmt(windows[duration]["trapezoid"] / 1e4, 1), "万m³")

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(times, flow, color=CYAN, linewidth=1.7, label="洪水过程")
    colors = [ORANGE, GREEN, VIOLET, PINK, YELLOW]
    highlight = [d for d in durations if d in windows][:4]
    for idx, duration in enumerate(highlight):
        window = windows[duration]
        ax.axvspan(
            times.iloc[window["start"]],
            times.iloc[window["end"]],
            color=colors[idx % len(colors)],
            alpha=0.14,
            label=f"{format_duration(duration)}最大窗口",
        )
    ax.set_title(f"{title} · 最大洪量窗口")
    ax.set_xlabel("时间")
    ax.set_ylabel("流量 / m³/s")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.autofmt_xdate(rotation=22)
    ctx.save_fig(fig, image_name, title="洪量窗口示意图")
    ctx.log(f"✓ {title}完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}


def run_max_volume(ctx, params, files):
    return _run_max_volume(
        ctx,
        params,
        files,
        title="时段最大洪量",
        table_name="时段最大洪量结果.csv",
        image_name="时段最大洪量窗口.png",
    )


def run_area_integral(ctx, params, files):
    return _run_max_volume(
        ctx,
        params,
        files,
        title="曲线面积积分校验",
        table_name="曲线面积积分校验.csv",
        image_name="曲线面积积分校验.png",
    )

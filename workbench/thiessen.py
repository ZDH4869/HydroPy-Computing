"""Thiessen polygon rainfall interpolation using GeoPandas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    BLUE,
    CYAN,
    GREEN,
    ORANGE,
    PINK,
    TEXT,
    VIOLET,
    configure_matplotlib,
    find_column,
    fmt,
    locate_shapefile,
)


def _fix_geometry(geom):
    from shapely.geometry import Polygon

    if geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        exterior = list(geom.exterior.coords)
        if exterior[0] != exterior[-1]:
            exterior.append(exterior[0])
        interiors = []
        for interior in geom.interiors:
            coords = list(interior.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            interiors.append(coords)
        return Polygon(exterior, interiors)
    return geom


def _voronoi_polygons(points_gdf, boundary):
    from shapely.geometry import Polygon
    from shapely.ops import voronoi_diagram

    points = [point for point in points_gdf.geometry]
    coords = [(point.x, point.y) for point in points]
    bounds = boundary.bounds
    padding = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
    extra = [
        (bounds[0] - padding, bounds[1] - padding),
        (bounds[0] - padding, bounds[3] + padding),
        (bounds[2] + padding, bounds[1] - padding),
        (bounds[2] + padding, bounds[3] + padding),
    ]
    all_coords = coords + extra
    # scipy's Voronoi is reliable here and mirrors the original script.
    from scipy.spatial import Voronoi

    vor = Voronoi(all_coords)
    polygons = []
    for i in range(len(points)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if region and -1 not in region:
            polygon = Polygon([vor.vertices[j] for j in region])
            polygons.append(polygon.intersection(boundary))
        else:
            polygons.append(Polygon())
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=polygons, crs=points_gdf.crs)


def run_thiessen(ctx, params, files):
    plt = configure_matplotlib()
    try:
        import geopandas as gpd
        from shapely.ops import unary_union
    except Exception as exc:
        raise ValueError("泰森多边形需要 geopandas 和 shapely，请先安装 requirements.txt 中的依赖。") from exc

    input_crs = str(params.get("input_crs") or "EPSG:4490")
    target_crs = str(params.get("target_crs") or "EPSG:4519")
    station_field = str(params.get("station_field") or "站名")
    rain_field = str(params.get("rain_field") or "雨量")
    buffer_m = float(params.get("buffer_m") or 1000)

    ctx.section("读取空间数据")
    basin_path = locate_shapefile(files["basin_zip"])
    station_path = locate_shapefile(files["stations_zip"])
    ctx.log(f"流域边界：{basin_path.name}")
    ctx.log(f"雨量站点：{station_path.name}")
    basin = gpd.read_file(basin_path)
    stations = gpd.read_file(station_path)
    if basin.crs is None:
        basin = basin.set_crs(input_crs)
    if stations.crs is None:
        stations = stations.set_crs(input_crs)
    ctx.log(f"流域要素：{len(basin)} 个，站点：{len(stations)} 个")
    ctx.log(f"原始坐标系：{basin.crs} / {stations.crs}")

    basin = basin.copy()
    basin["geometry"] = basin["geometry"].apply(_fix_geometry)
    basin_proj = basin.to_crs(target_crs)
    stations_proj = stations.to_crs(target_crs)

    station_field = find_column(stations_proj, [station_field, "站名", "name", "站点"]) or station_field
    rain_field = find_column(stations_proj, [rain_field, "雨量", "rain", "rainfall"]) or rain_field
    if station_field not in stations_proj.columns:
        stations_proj[station_field] = [f"站点{i + 1}" for i in range(len(stations_proj))]
        ctx.log(f"未找到站名字段，已生成默认名称。")
    if rain_field not in stations_proj.columns:
        rng = np.random.default_rng(42)
        stations_proj[rain_field] = rng.uniform(20, 50, len(stations_proj)).round(1)
        ctx.log("未找到雨量字段，已生成模拟雨量。")

    boundary = unary_union(basin_proj.geometry).buffer(buffer_m)
    if boundary.is_empty:
        raise ValueError("流域边界为空，无法构建泰森多边形。")
    in_basin = stations_proj.geometry.intersects(boundary)
    filtered = stations_proj[in_basin].reset_index(drop=True)
    if filtered.empty:
        ctx.log("没有站点落在流域内，改用全部站点。")
        filtered = stations_proj.reset_index(drop=True)

    ctx.section("构建泰森多边形")
    voronoi = _voronoi_polygons(filtered, boundary)
    areas = voronoi.area.to_numpy(dtype=float)
    total_area = float(np.nansum(areas))
    results = []
    for i in range(len(filtered)):
        area_km2 = float(areas[i]) / 1e6 if np.isfinite(areas[i]) else 0.0
        weight = area_km2 / (total_area / 1e6) if total_area > 0 else 0.0
        rainfall = float(filtered.iloc[i][rain_field])
        results.append(
            {
                "站点": filtered.iloc[i][station_field],
                "雨量/mm": round(rainfall, 2),
                "控制面积/km²": round(area_km2, 2),
                "权重": round(weight, 4),
            }
        )
    table = pd.DataFrame(results)
    weighted = float((table["雨量/mm"] * table["权重"]).sum())
    arithmetic = float(table["雨量/mm"].mean())
    ctx.log(f"有效泰森多边形：{sum(area > 0 for area in areas)}/{len(areas)}")
    ctx.log(f"泰森多边形法面雨量：{weighted:.2f} mm")
    ctx.log(f"算术平均法面雨量：{arithmetic:.2f} mm")
    ctx.add_summary("泰森法面雨量", f"{weighted:.2f}", "mm")
    ctx.add_summary("算术平均", f"{arithmetic:.2f}", "mm")
    ctx.add_summary("站点数量", len(table), "个")
    ctx.add_summary("控制面积", f"{total_area / 1e6:,.2f}", "km²")
    ctx.save_csv(table, "泰森多边形权重.csv", title="站点权重")

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    voronoi.plot(ax=ax, facecolor="#0e2a3a", edgecolor=CYAN, alpha=0.55, linewidth=0.9)
    basin_proj.boundary.plot(ax=ax, color="#e2e8f0", linewidth=1.8)
    scatter = ax.scatter(
        [p.x for p in filtered.geometry],
        [p.y for p in filtered.geometry],
        c=table["雨量/mm"],
        s=110,
        cmap="turbo",
        edgecolor="#020617",
        linewidth=1.0,
        zorder=5,
    )
    for i, row in table.iterrows():
        point = filtered.geometry.iloc[i]
        ax.annotate(
            f"{row['站点']}\n{row['雨量/mm']:.1f}mm · 权重 {row['权重']:.3f}",
            xy=(point.x, point.y),
            xytext=(10, 12),
            textcoords="offset points",
            color=TEXT,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="#07111f", ec=CYAN, alpha=0.78),
            arrowprops=dict(arrowstyle="-", color=CYAN, lw=0.7),
        )
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    colorbar.set_label("雨量 / mm", color=TEXT)
    colorbar.ax.tick_params(colors=TEXT)
    ax.set_title("泰森多边形权重与面雨量")
    ax.set_xlabel("东坐标 / m")
    ax.set_ylabel("北坐标 / m")
    ax.set_aspect("equal", adjustable="datalim")
    ctx.save_fig(fig, "泰森多边形面雨量.png", title="泰森多边形分析")
    ctx.log("✓ 泰森多边形分析完成。")
    return {"summary": ctx.summary, "tables": ctx.tables, "images": ctx.images}

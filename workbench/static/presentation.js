import { makeChart, timeValue } from "./charts.js";

const cyan = "#66e7f5";
const amber = "#f3b066";
const blue = "#85aaff";
const flow = "流量 (m³/s)";
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 });
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);

function chart(table, x, y, unit, options = {}) {
  return makeChart(table, { x, y: [y], unit, ...options });
}

function panel(label, model) {
  return { label, model };
}

function span(columns) {
  let low = Infinity;
  let high = -Infinity;
  for (const values of columns) {
    for (const value of values) {
      if (!isNumber(value)) continue;
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  return low <= high ? [low, high] : undefined;
}

function compare(panels, shareTime = true) {
  const enabled = panels.filter(({ model }) => model);
  if (enabled.length < 2) return [];
  const axisKind = enabled[0].model.axis.kind;
  const shareX = shareTime && axisKind !== "category" && enabled.every(({ model }) => model.axis.kind === axisKind);
  const x = shareX ? span(enabled.map(({ model }) => model.axis.values)) : undefined;
  return enabled.map((item) => {
    const sameUnits = enabled.filter(({ model }) => model.yLabel === item.model.yLabel);
    const y = sameUnits.length > 1
      ? span(sameUnits.flatMap(({ model }) => model.series.map((series) => series.values)))
      : undefined;
    return { ...item, domain: { x, y } };
  });
}

function volumePresentation(table, observation) {
  if (!observation || !table) return { comparison: [] };
  const startCol = table.columns.indexOf("开始时间");
  const endCol = table.columns.indexOf("结束时间");
  const volumeCol = table.columns.indexOf("梯形法洪量/万m³");
  const windows = table.rows.flatMap((row) => {
    const range = [timeValue(row[startCol]), timeValue(row[endCol])];
    return range.every(isNumber) ? [{
      title: row[0], start: row[startCol], end: row[endCol], volume: row[volumeCol], range,
    }] : [];
  });
  return {
    windows, comparison: [panel("洪水过程与最大洪量区间", observation)], comparisonLabel: "洪量区间",
  };
}

function stationMap(data, table) {
  if (!data?.zones.length || !table?.rows.length) return null;
  const zones = data.zones.slice(0, table.rows.length).map((item, index) => {
    const row = table.rows[index];
    return { ...item, name: String(row[0]), rainfall: row[1], area: row[2], weight: row[3] };
  });
  return { ...data, zones };
}

export function buildPresentation(algo, tables, groups, visuals, settings, inputPreview) {
  const [primary, secondary] = groups;
  const table = tables[0];
  let comparison = [];
  switch (algo.id) {
    case "eng_linear_interp":
    case "eng_multi_interp":
      comparison = compare([
        panel("原始观测点", chart(visuals.observations, "时间", "观测流量", flow, { scatter: true, color: amber })),
        panel("计算后的插值", primary?.chart),
      ]);
      break;
    case "eng_max_volume":
    case "eng_area_integral":
      return volumePresentation(table, chart(visuals.observations, "时间", "流量", flow));
    case "thiessen":
      return { comparison, map: stationMap(visuals.station_areas, table) };
    case "muskingum_xk":
      comparison = compare([panel("蓄量与加权流量", primary?.chart), panel("X 与相关系数", secondary?.chart)], false);
      break;
    case "muskingum_route":
      comparison = compare([
        panel("上游来流", chart(table, "时段", "上游流量I", flow, { color: cyan })),
        panel("下游出流", chart(table, "时段", "下游流量Q", flow, { color: amber })),
      ]);
      break;
    case "nash_nk":
      comparison = compare([panel("输入净雨", primary?.chart), panel("实测出口流量", secondary?.chart)]);
      break;
    case "nash_duh":
      comparison = compare([
        panel("标准曲线", chart(visuals.reference, "时间 (h)", "标准流量", flow, { color: amber })),
        panel("计算时段单位线", primary?.chart),
      ]);
      break;
    case "sherman_derive":
      comparison = compare([
        panel("地面径流输入", chart(visuals.runoff, "时段", "地面径流", flow, { color: amber })),
        panel("推求的单位线", primary?.chart),
      ]);
      break;
    case "sherman_outlet":
      comparison = compare([
        panel("输入净雨", chart(visuals.rain_response, "时间(时段)", "净雨量", "净雨量 (mm)", { bars: true, color: blue })),
        panel("计算出口流量", chart(table, "时间(时段)", "计算流量(m³/s)", flow)),
        panel("实测出口流量", chart(visuals.reference, "时间(时段)", "实测流量", flow, { color: amber })),
      ]);
      break;
    case "flood_curve_interp": {
      const x = table?.columns[0] || settings.x_col;
      const y = primary?.y[0];
      const observed = inputPreview?.columns.includes(x) && inputPreview.columns.includes(y)
        ? chart(inputPreview, x, y, y, { scatter: true, color: amber }) : null;
      comparison = compare([panel("原始曲线预览", observed), panel("插值后的曲线", primary?.chart)]);
      break;
    }
    case "flood_trial":
    case "flood_rk4":
      comparison = compare([
        panel("入库过程", chart(table, "时间t/h", "入库流量Q/(m³·s⁻¹)", flow, { color: cyan })),
        panel("下泄过程", chart(table, "时间t/h", "下泄流量q/(m³·s⁻¹)", flow, { color: amber })),
      ]);
      return { comparison, preferCombined: true };
  }
  return { comparison };
}

function rings(geometry) {
  if (geometry?.type === "Polygon") return geometry.coordinates;
  if (geometry?.type === "MultiPolygon") return geometry.coordinates.flat();
  if (geometry?.type === "GeometryCollection") return geometry.geometries.flatMap(rings);
  return [];
}

export function mountStationMap(host, map, selected, onSelect) {
  const stage = document.createElement("div");
  stage.className = "station-map";
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "本次计算输出的泰森多边形控制区");
  const list = document.createElement("div");
  list.className = "station-list";
  list.setAttribute("aria-label", "雨量站点");
  const buttons = map.zones.map((zone, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = zone.name;
    button.setAttribute("aria-label", "站点 " + zone.name);
    button.addEventListener("click", () => select(index));
    list.append(button);
    return button;
  });
  const details = document.createElement("dl");
  details.className = "station-details";
  details.setAttribute("aria-label", "所选站点计算指标");
  const note = document.createElement("p");
  note.className = "chart-footer-note";
  note.textContent = map.crs + " · 当前算例的控制区";
  stage.append(canvas, list, details, note);
  host.replaceChildren(stage);
  const allRings = [...rings(map.boundary), ...map.zones.flatMap(({ geometry }) => rings(geometry))];
  const points = allRings.flat().concat(map.zones.map(({ point }) => point));
  const [lowX, highX] = span([points.map(([x]) => x)]) || [0, 1];
  const [lowY, highY] = span([points.map(([, y]) => y)]) || [0, 1];
  let focused = 0;
  let stationPoints = [];

  function draw() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const fit = Math.min((width - 46) / (highX - lowX || 1), (height - 40) / (highY - lowY || 1));
    const x = (value) => (width - fit * (highX - lowX)) / 2 + (value - lowX) * fit;
    const y = (value) => (height + fit * (highY - lowY)) / 2 - (value - lowY) * fit;
    const outline = (geometry) => {
      ctx.beginPath();
      for (const ring of rings(geometry)) {
        ring.forEach(([px, py], i) => i ? ctx.lineTo(x(px), y(py)) : ctx.moveTo(x(px), y(py)));
        ctx.closePath();
      }
    };

    map.zones.forEach((zone, index) => {
      outline(zone.geometry);
      ctx.fillStyle = index === focused ? "#47c8d977" : index % 2 === 0 ? "#276d9177" : "#2c8d9c60";
      ctx.fill("evenodd");
      ctx.strokeStyle = index === focused ? "#a0f7ff" : "#66a6b8";
      ctx.stroke();
    });
    outline(map.boundary);
    ctx.strokeStyle = "#bce9f2";
    ctx.stroke();
    stationPoints = map.zones.map(({ point }) => [x(point[0]), y(point[1])]);
    stationPoints.forEach(([px, py], index) => {
      ctx.fillStyle = index === focused ? amber : "#c6eaf4";
      ctx.beginPath();
      ctx.arc(px, py, index === focused ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function select(index) {
    const station = map.zones[index];
    if (!station) return;
    focused = index;
    onSelect(index);
    buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    const weight = isNumber(station.weight) ? station.weight * 100 : null;
    const contribution = weight !== null && isNumber(station.rainfall) ? station.weight * station.rainfall : null;
    const metrics = [
      ["站点雨量", station.rainfall, "mm"],
      ["控制面积", station.area, "km²"],
      ["面积权重", weight, "%"],
      ["面积加权贡献", contribution, "mm"],
    ];
    details.replaceChildren(...metrics.map(([label, value, unit]) => {
      const metric = document.createElement("div");
      const term = document.createElement("dt");
      const amount = document.createElement("dd");
      term.textContent = label;
      amount.textContent = isNumber(value) ? number.format(value) + " " + unit : "—";
      metric.append(term, amount);
      return metric;
    }));
    draw();
  }

  canvas.addEventListener("click", (event) => {
    let closest = 19;
    let selection = -1;
    stationPoints.forEach(([x, y], index) => {
      const distance = Math.hypot(x - event.offsetX, y - event.offsetY);
      if (distance < closest) {
        closest = distance;
        selection = index;
      }
    });
    if (selection >= 0) select(selection);
  });
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  select(selected);
  return () => observer.disconnect();
}

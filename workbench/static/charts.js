import { barScale, categoryScale, extent, linearScale, numberLabel, position, timeLabel, timeScale } from "./chart-axes.js";

const COLORS = ["#66e7f5", "#f3b066", "#85aaff", "#e299d0", "#a8dc81"];
const DASHES = [[], [7, 4], [2, 4], [10, 4, 2, 4], [3, 3]];
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 });

const numeric = (value) => typeof value === "number" && Number.isFinite(value);
const normalize = (name) => name.toLowerCase().replace(/\s/g, "");
const timeColumn = (name) => /^(?:datetime|timestamp|time|date|t|日期|时间t?|时段)(?:$|[(/_（【\[])/.test(normalize(name));
const dayColumn = (name) => /^(年月日|日期|date)$/.test(normalize(name));

export function timeValue(value) {
  if (typeof value !== "string") return NaN;
  const match = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,3}))?)?)?$/.exec(value.trim());
  if (!match) return NaN;
  const [, y, m, d, h = "0", min = "0", s = "0", ms = "0"] = match;
  const time = new Date(0);
  time.setUTCFullYear(+y, +m - 1, +d);
  return time.setUTCHours(+h, +min, +s, +ms.padEnd(3, "0"));
}

function axisValues(table, xColumn, useRowIndex) {
  const { columns, rows } = table;
  const rowAxis = () => ({ label: "数据行", values: rows.map((_, row) => row + 1), kind: "number", integer: true });
  if (useRowIndex) return rowAxis();
  const day = columns.findIndex(dayColumn);
  const hour = columns.indexOf("时");
  const years = ["年", "月", "日", "时"].map((label) => columns.indexOf(label));
  if (!xColumn && years.every((index) => index >= 0)) {
    const values = rows.map((row) => {
      const [y, m, d, h] = years.map((index) => row[index]);
      return [y, m, d, h].every(numeric) ? Date.UTC(y, m - 1, d) + h * 3600000 : NaN;
    });
    if (values.some(numeric)) return { label: "时间", values, kind: "time" };
  }
  if (!xColumn && day >= 0 && hour >= 0) {
    const values = rows.map((row) => numeric(row[hour]) ? timeValue(row[day]) + row[hour] * 3600000 : NaN);
    if (values.some(numeric)) return { label: "时间", values, kind: "time" };
  }

  const index = xColumn ? columns.indexOf(xColumn) : columns.findIndex(timeColumn);
  if (index >= 0) {
    const source = rows.map((row) => row[index]);
    const present = source.filter((value) => value !== null && value !== "");
    if (present.length && present.every(numeric)) {
      const name = columns[index];
      return {
        label: /^时间t?\/h$/i.test(name) ? "时间 (h)" : name,
        values: source, kind: "number", integer: /时段/.test(name) && present.every(Number.isInteger),
      };
    }
    const dates = source.map(timeValue);
    if (dates.some(numeric) && present.every((value) => numeric(timeValue(value)))) {
      return { label: "时间", values: dates, kind: "time" };
    }
    if (present.length) {
      return { label: columns[index], values: rows.map((_, row) => row), categories: source, kind: "category" };
    }
  }
  return rowAxis();
}

function metricLabel(column) {
  const labels = {
    Q: "流量 Q", Q_线性插值: "线性", Q_最近邻插值: "最近邻", Q_多项式插值: "多项式", Q_样条插值: "样条",
    "入库流量Q/(m³·s⁻¹)": "入库流量 Q", "下泄流量q/(m³·s⁻¹)": "下泄流量 q",
    "梯形法洪量/万m³": "梯形法", "辛普森法洪量/万m³": "辛普森法", "插值积分洪量/万m³": "插值积分",
  };
  return labels[column] || column;
}

export function makeChart(table, spec) {
  if (!table?.rows?.length) return null;
  const axis = axisValues(table, spec.x, spec.useRowIndex);
  const series = spec.y.flatMap((column) => {
    const index = table.columns.indexOf(column);
    if (index < 0) return [];
    const values = table.rows.map((row, i) => numeric(row[index]) && numeric(axis.values[i]) ? row[index] : null);
    return values.some(numeric) ? [{ name: metricLabel(column), values, color: spec.color }] : [];
  });
  return series.length ? {
    axis, series, yLabel: spec.unit || spec.y[0], bars: Boolean(spec.bars),
    barAndLine: spec.bars === "overlay", scatter: Boolean(spec.scatter), peak: Boolean(spec.peak),
  } : null;
}

export function inputCharts(table, xColumn, fileKey) {
  if (table?.kind !== "csv" || !table.rows.length) return [];
  const columns = table.numeric_columns?.length
    ? table.numeric_columns
    : table.columns.filter((_, index) => table.rows.some((row) => numeric(row[index])));
  const dates = new Set(["年", "月", "日", "时"]);
  const inferredX = xColumn || (["storage_csv", "discharge_csv"].includes(fileKey) ? table.columns[0] : "");
  const ys = columns.filter((column) => column !== inferredX && !dates.has(column) && !timeColumn(column));
  return ys.map((name) => ({
    label: metricLabel(name),
    chart: makeChart(table, {
      x: inferredX, y: [name], unit: name === "Q" ? "流量 (m³/s)" : name,
    }),
  })).filter((group) => group.chart);
}

export function resultCharts(id, tables, config, visuals = {}) {
  const flow = (label, ...y) => ({ label, unit: "流量 (m³/s)", y });
  const reservoir = [
    { ...flow("入库与下泄过程", "入库流量Q/(m³·s⁻¹)", "下泄流量q/(m³·s⁻¹)"), peak: true },
    { label: "水位", unit: "水位 (m)", y: ["水库水位Z/m"], color: "#81e2ab", peak: true },
    { label: "库容", unit: "库容 (万m³)", y: ["水库存水量V/万m³"], color: "#c9a9f4" },
  ].map((spec) => ({ ...spec, x: "时间t/h" }));
  const volume = [{
    label: "时段洪量", unit: "洪量 (万m³)", x: "统计时段", bars: true,
    y: ["梯形法洪量/万m³", "辛普森法洪量/万m³", "插值积分洪量/万m³"],
  }];
  const curve = tables[0];
  const exactTime = visuals.interpolation_times?.length >= (curve?.rows.length || 1) ? "精确时间" : undefined;
  const useRowIndex = !exactTime && config.freq === "30min";
  const specs = {
    eng_linear_interp: [{ ...flow("等间隔过程", "Q"), x: exactTime, useRowIndex, peak: true }],
    eng_multi_interp: [{ ...flow("插值方法对比", "Q_线性插值", "Q_最近邻插值", "Q_多项式插值", "Q_样条插值"), x: exactTime, useRowIndex }],
    eng_max_volume: volume,
    eng_area_integral: volume,
    muskingum_xk: [
      { label: "蓄量关系", x: "加权流量Q'", y: ["蓄水量S"], unit: "蓄水量 S (m³)", scatter: true },
      { label: "参数扫描", index: 1, x: "X", y: ["相关系数"], peak: true },
    ],
    muskingum_route: [flow("上下游过程", "上游流量I", "下游流量Q")],
    nash_nk: [
      { label: "净雨过程", x: "时间 (h)", y: ["净雨量"], unit: "净雨量 (mm)", source: visuals.rain_response, color: "#85aaff" },
      { label: "出口流量", x: "时间 (h)", y: ["出口流量"], unit: "流量 (m³/s)", source: visuals.rain_response },
    ],
    nash_duh: [{ ...flow("时段单位线", "时段单位线(m³/s)"), bars: "overlay", peak: true }],
    sherman_derive: [{ ...flow("Sherman 单位线", "单位线纵坐标(m³/s)"), peak: true }],
    sherman_outlet: [{ ...flow("出口流量", "计算流量(m³/s)", "实际流量(m³/s)"), peak: true }],
    flood_trial: reservoir,
    flood_rk4: reservoir,
    flood_curve_interp: (curve?.columns || []).filter((column) => column !== config.x_col && column !== curve.columns[0])
      .map((column) => ({ label: column, x: curve.columns[0], y: [column] })),
  };
  return (specs[id] || []).flatMap((spec) => {
    const table = tables[spec.index || 0];
    const chartSource = exactTime && spec.x === exactTime
      ? { columns: [...table.columns, exactTime], rows: table.rows.map((row, i) => [...row, visuals.interpolation_times[i]]) }
      : spec.source || table;
    const chart = makeChart(chartSource, spec);
    return chart ? [{ ...spec, table, chart }] : [];
  });
}

export function parseResultCsv(text, table) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  text = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') field += text[++index];
      else if (char === '"') quoted = false;
      else field += char;
    } else if (char === '"' && field === "") {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\r" || char === "\n") {
      if (char === "\r" && text[index + 1] === "\n") index++;
      rows.push([...row, field]);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field || row.length) rows.push([...row, field]);
  const header = rows.shift();
  if (quoted || header?.length !== table.columns.length || header.some((name, index) => name !== table.columns[index]) ||
      rows.length !== table.total_rows || rows.some((record) => record.length !== header.length)) {
    throw new Error("CSV 的列或行数与结果预览不一致");
  }
  const numberPattern = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i;
  const types = table.columns.map((_, index) => {
    const samples = table.rows.map((record) => record[index]).filter((value) => value !== null && value !== "");
    return samples.length ? samples.every(numeric) : rows.every((record) => record[index] === "" || numberPattern.test(record[index]));
  });
  return { ...table, rows: rows.map((record) => record.map((value, index) => {
    if (!value) return null;
    return types[index] && numberPattern.test(value) && Number.isFinite(Number(value)) ? Number(value) : value;
  })) };
}

function fitCanvas(canvas, width, height) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function datumLabel(axis, value) {
  if (axis.kind === "time") {
    const iso = new Date(value).toISOString();
    const length = value % 1000 ? 23 : value % 60000 ? 19 : 16;
    return iso.slice(0, length).replace("T", " ");
  }
  if (axis.kind === "category") return String(axis.categories[value] ?? "—");
  return Number.isInteger(value) ? number.format(value) : String(value);
}

function minSpacing(points, fallback) {
  let gap = Infinity;
  for (let index = 1; index < points.length; index++) {
    const distance = points[index].value - points[index - 1].value;
    if (distance > 0) gap = Math.min(gap, distance);
  }
  return Number.isFinite(gap) ? gap : fallback;
}

function horizontalScale(axis, points, width, bars) {
  const min = points[0].value;
  const max = points.at(-1).value;
  const spacing = bars ? minSpacing(points, axis.kind === "time" ? 1800000 : Math.abs(min) * 0.04 || 1) : 0;
  const divisions = Math.max(2, Math.min(14, Math.floor(width / (axis.kind === "time" ? 92 : 62))));
  if (axis.kind === "category") return { ...categoryScale(min, max, divisions), spacing: 1 };
  if (axis.kind === "time") return { ...timeScale(min, max, divisions, spacing), spacing };
  return bars ? { ...barScale(min, max, divisions, spacing, axis.integer), spacing }
    : linearScale(min, max, divisions, { padding: 0.035, integer: axis.integer });
}

export function mountChart(host, model) {
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  const stage = document.createElement("div");
  stage.className = "plot-stage";
  const canvas = document.createElement("canvas");
  canvas.tabIndex = 0;
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", model.series.map((series) => series.name).join("、") + "图表。使用方向键检查数据点。");
  const tooltip = document.createElement("div");
  tooltip.className = "plot-tooltip";
  tooltip.hidden = true;
  const readout = document.createElement("span");
  readout.className = "sr-only";
  readout.setAttribute("aria-live", "polite");
  stage.append(canvas, tooltip, readout);
  host.replaceChildren(legend, stage);

  const enabled = model.series.map(() => true);
  const axis = model.axis;
  const ordered = axis.values.map((value, index) => ({ value, index })).filter((item) => numeric(item.value))
    .sort((a, b) => a.value - b.value);
  let current = null;
  let visible = ordered;
  let highlight = null;
  let geometry;
  const seriesExtents = model.series.map((series) => extent(series.values));
  const peaks = model.peak
    ? model.series.map((series, index) => series.values.indexOf(seriesExtents[index][1])) : [];
  const paint = (index) => model.series[index].color || COLORS[index % COLORS.length];

  model.series.forEach((series, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", "true");
    const swatch = document.createElement("i");
    swatch.style.setProperty("--series-color", paint(index));
    swatch.classList.toggle("dashed", index % DASHES.length !== 0 && !model.bars);
    swatch.setAttribute("aria-hidden", "true");
    button.append(swatch, series.name);
    button.addEventListener("click", () => {
      if (enabled[index] && enabled.filter(Boolean).length === 1) return;
      enabled[index] = !enabled[index];
      button.setAttribute("aria-pressed", String(enabled[index]));
      current = null;
      updateVisible();
      draw();
    });
    legend.append(button);
  });

  function updateVisible() {
    visible = ordered.filter(({ index }) =>
      model.series.some((series, i) => enabled[i] && numeric(series.values[index])));
  }

  updateVisible();

  function draw() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (width < 100 || height < 100) return;
    const ctx = fitCanvas(canvas, width, height);
    const lineFont = "11px 'IBM Plex Mono', Consolas, 'Microsoft YaHei', monospace";
    ctx.font = lineFont;
    const live = model.series.map((series, index) => ({ ...series, index })).filter((_, index) => enabled[index]);
    const limits = live.map((series) => seriesExtents[series.index]);
    const lo = Math.min(...limits.map(([min]) => min));
    const hi = Math.max(...limits.map(([, max]) => max));
    const top = 24;
    const bottom = height - 38;
    const yDivisions = Math.max(2, Math.min(12, Math.floor((bottom - top) / 42)));
    const yScale = linearScale(lo, hi, yDivisions, { zero: model.bars });
    const labels = yScale.values.map((value) => numberLabel(value, yScale.step));
    const left = Math.min(width * 0.38, Math.max(39, ...labels.map((label) => ctx.measureText(label).width + 13)));
    const right = width - 15;
    const xScale = horizontalScale(axis, visible, right - left, model.bars);
    const x = (value) => left + position(value, xScale.range) * (right - left);
    const y = (value) => bottom - position(value, yScale.range) * (bottom - top);
    geometry = { left, right, top, bottom, xRange: xScale.range };

    ctx.fillStyle = "#c6d9e8";
    ctx.fillText(model.yLabel, left, 13, right - left);
    const grid = (x1, y1, x2, y2, color) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    };
    yScale.values.forEach((value, index) => {
      const py = y(value);
      grid(left, py, right, py, "#afcfe433");
      ctx.textAlign = "right";
      ctx.fillText(labels[index], left - 7, py + 3, left - 9);
    });
    grid(left, bottom, right, bottom, "#adcce77a");
    let lastLabelEnd = 0;
    for (const value of xScale.values) {
      const px = x(value);
      grid(px, top, px, bottom, "#afcfe41a");
      const label = axis.kind === "time" ? timeLabel(value, xScale)
        : axis.kind === "category" ? datumLabel(axis, value) : numberLabel(value, xScale.step);
      const widthOfLabel = Math.min(ctx.measureText(label).width, right - left);
      const start = Math.max(4, Math.min(px - widthOfLabel / 2, width - widthOfLabel - 4));
      if (start < lastLabelEnd + 7) continue;
      ctx.textAlign = "left";
      ctx.fillText(label, start, bottom + 16, widthOfLabel);
      lastLabelEnd = start + widthOfLabel;
    }
    ctx.textAlign = "right";
    ctx.fillText(axis.label, right, height - 6, right - left);

    ctx.save();
    ctx.beginPath();
    ctx.rect(left - 4, top - 4, right - left + 8, bottom - top + 8);
    ctx.clip();
    if (highlight) {
      const [start, stop] = highlight.map(x);
      ctx.fillStyle = "#f3b06643";
      ctx.fillRect(start, top, stop - start, bottom - top);
      ctx.fillStyle = "#f3b066";
      ctx.fillRect(start, top, 1, bottom - top);
      ctx.fillRect(stop, top, 1, bottom - top);
    }
    live.forEach((series, order) => {
      const color = paint(series.index);
      if (model.bars) {
        const pitch = position(xScale.spacing, [0, xScale.range[1] - xScale.range[0]]) * (right - left);
        const barWidth = Math.min(34, Math.max(1, pitch * 0.78 / live.length));
        series.values.forEach((value, index) => {
          if (!numeric(value)) return;
          ctx.fillStyle = model.barAndLine ? color + "45" : color;
          const px = x(axis.values[index]) + barWidth * (order - live.length / 2);
          const py = y(Math.max(0, value));
          ctx.fillRect(px + 0.5, py, Math.max(1, barWidth - 1), Math.abs(y(value) - y(0)));
        });
        if (!model.barAndLine) return;
      }
      const segments = [];
      let segment = [];
      ordered.forEach(({ value: timestamp, index }) => {
        const value = series.values[index];
        if (numeric(value)) segment.push([x(timestamp), y(value)]);
        else if (segment.length) {
          segments.push(segment);
          segment = [];
        }
      });
      if (segment.length) segments.push(segment);
      segments.forEach((points) => {
        if (model.scatter) {
          ctx.fillStyle = color;
          ctx.beginPath();
          points.forEach(([px, py]) => {
            ctx.moveTo(px + 2.5, py);
            ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          });
          ctx.fill();
          return;
        }
        const trace = () => {
          ctx.beginPath();
          points.forEach(([px, py], index) => index ? ctx.lineTo(px, py) : ctx.moveTo(px, py));
        };
        if (order === 0 && !model.bars && points.length > 1) {
          trace();
          const gradient = ctx.createLinearGradient(0, top, 0, bottom);
          gradient.addColorStop(0, color + "38");
          gradient.addColorStop(1, color + "04");
          ctx.lineTo(points.at(-1)[0], bottom);
          ctx.lineTo(points[0][0], bottom);
          ctx.closePath();
          ctx.fillStyle = gradient;
          ctx.fill();
        }
        trace();
        ctx.lineWidth = 2.35;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.strokeStyle = color;
        ctx.setLineDash(DASHES[series.index % DASHES.length]);
        ctx.stroke();
        ctx.setLineDash([]);
        if (points.length <= 35) {
          ctx.fillStyle = color;
          points.forEach(([px, py]) => {
            ctx.beginPath();
            ctx.arc(px, py, 2.15, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      });
      const peak = peaks[series.index];
      if (peak >= 0) {
        ctx.beginPath();
        ctx.arc(x(axis.values[peak]), y(series.values[peak]), 4.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#e4f7ff";
        ctx.stroke();
      }
    });

    if (current !== null) {
      const { index, value } = visible[current];
      const px = x(value);
      ctx.strokeStyle = "#a3bed2";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      const title = document.createElement("strong");
      title.textContent = datumLabel(axis, value);
      const rows = live.map((series) => {
        const entry = document.createElement("span");
        const exact = series.values[index];
        entry.textContent = series.name + "  " + (numeric(exact) ? (Number.isInteger(exact) ? number.format(exact) : exact) : "—");
        if (numeric(exact) && (!model.bars || model.barAndLine)) {
          ctx.fillStyle = paint(series.index);
          ctx.beginPath();
          ctx.arc(px, y(exact), 4.5, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = "#fff";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        return entry;
      });
      tooltip.replaceChildren(title, ...rows);
      tooltip.hidden = false;
      tooltip.style.left = Math.max(4, Math.min(px + 12, width - tooltip.offsetWidth - 4)) + "px";
    } else {
      tooltip.hidden = true;
    }
    ctx.restore();
  }

  function nearest(value) {
    let start = 0;
    let end = visible.length - 1;
    while (start < end) {
      const middle = Math.floor((start + end) / 2);
      if (visible[middle].value < value) start = middle + 1;
      else end = middle;
    }
    if (start > 0 && Math.abs(visible[start - 1].value - value) < Math.abs(visible[start].value - value)) return start - 1;
    return start;
  }

  canvas.addEventListener("pointermove", (event) => {
    if (!geometry) return;
    const px = event.clientX - canvas.getBoundingClientRect().left;
    const py = event.clientY - canvas.getBoundingClientRect().top;
    const { left, right, top, bottom, xRange } = geometry;
    const ratio = (px - left) / (right - left);
    const next = px < left || px > right || py < top || py > bottom ? null
      : nearest(xRange[0] * (1 - ratio) + xRange[1] * ratio);
    if (current !== next) {
      current = next;
      draw();
    }
  });
  const clear = () => { if (current !== null) { current = null; draw(); } };
  canvas.addEventListener("pointerleave", clear);
  canvas.addEventListener("blur", clear);
  canvas.addEventListener("keydown", (event) => {
    if (event.key === "Escape") return clear();
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") current = 0;
    else if (event.key === "End") current = visible.length - 1;
    else current = Math.max(0, Math.min(visible.length - 1, (current ?? -1) + (event.key === "ArrowRight" ? 1 : -1)));
    draw();
    readout.textContent = Array.from(tooltip.children, (node) => node.textContent).join("，");
  });
  const observer = new ResizeObserver(draw);
  observer.observe(stage);
  draw();
  const destroy = () => observer.disconnect();
  destroy.setWindow = (range) => {
    highlight = range;
    draw();
  };
  return destroy;
}

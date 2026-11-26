const COLORS = ["#66e7f5", "#f3b066", "#85aaff", "#e299d0", "#a8dc81"];
const DASHES = [[], [7, 4], [2, 4], [10, 4, 2, 4], [3, 3]];
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 });
const axisNumber = new Intl.NumberFormat("zh-CN", { maximumSignificantDigits: 4 });
const date = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "UTC", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});
const fullDate = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "UTC", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23",
});

const numeric = (value) => typeof value === "number" && Number.isFinite(value);
const normalize = (name) => name.toLowerCase().replace(/\s/g, "");
const timeColumn = (name) => /^(?:datetime|timestamp|time|date|t|日期|时间t?|时段)(?:$|[(/_（【\[])/.test(normalize(name));
const dayColumn = (name) => /^(年月日|日期|date)$/.test(normalize(name));

export function timeValue(value) {
  if (typeof value !== "string") return NaN;
  const match = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2}))?)?$/.exec(value.trim());
  if (!match) return NaN;
  const [, y, m, d, h = "0", min = "0", s = "0"] = match;
  return Date.UTC(+y, +m - 1, +d, +h, +min, +s);
}

function axisValues(table, xColumn, useRowIndex) {
  const { columns, rows } = table;
  const rowAxis = () => ({ label: "数据行", values: rows.map((_, row) => row + 1), kind: "number" });
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
      return { label: columns[index], values: source, kind: "number" };
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
    scatter: Boolean(spec.scatter),
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
    flow("入库与下泄过程", "入库流量Q/(m³·s⁻¹)", "下泄流量q/(m³·s⁻¹)"),
    { label: "水位", unit: "水位 (m)", y: ["水库水位Z/m"] },
    { label: "库容", unit: "库容 (万m³)", y: ["水库存水量V/万m³"] },
  ].map((spec) => ({ ...spec, x: "时间t/h" }));
  const volume = [{
    label: "时段洪量", unit: "洪量 (万m³)", x: "统计时段", bars: true,
    y: ["梯形法洪量/万m³", "辛普森法洪量/万m³", "插值积分洪量/万m³"],
  }];
  const curve = tables[0];
  const exactTime = visuals.interpolation_times?.length >= (curve?.rows.length || 1) ? "精确时间" : undefined;
  const useRowIndex = !exactTime && config.freq === "30min";
  const specs = {
    eng_linear_interp: [{ ...flow("等间隔过程", "Q"), x: exactTime, useRowIndex }],
    eng_multi_interp: [{ ...flow("插值方法对比", "Q_线性插值", "Q_最近邻插值", "Q_多项式插值", "Q_样条插值"), x: exactTime, useRowIndex }],
    eng_max_volume: volume,
    eng_area_integral: volume,
    muskingum_xk: [
      { label: "蓄量关系", x: "加权流量Q'", y: ["蓄水量S"], unit: "蓄水量 S", scatter: true },
      { label: "参数扫描", index: 1, x: "X", y: ["相关系数"] },
    ],
    muskingum_route: [flow("上下游过程", "上游流量I", "下游流量Q")],
    nash_nk: [
      { label: "净雨过程", x: "时间 (h)", y: ["净雨量"], unit: "净雨量 (mm)", bars: true, source: visuals.rain_response, color: "#85aaff" },
      { label: "出口流量", x: "时间 (h)", y: ["出口流量"], unit: "流量 (m³/s)", source: visuals.rain_response },
    ],
    nash_duh: [flow("时段单位线", "时段单位线(m³/s)")],
    sherman_derive: [flow("Sherman 单位线", "单位线纵坐标(m³/s)")],
    sherman_outlet: [flow("出口流量", "计算流量(m³/s)", "实际流量(m³/s)")],
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

function scaleTicks(min, max) {
  const raw = (max - min) / 4;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const step = ([1, 2, 2.5, 5, 10].find((item) => item * magnitude >= raw) || 10) * magnitude;
  const start = Math.floor(min / step) * step;
  const end = Math.ceil(max / step) * step;
  return Array.from({ length: Math.round((end - start) / step) + 1 }, (_, index) => start + index * step);
}

export function showX(axis, value, full = false) {
  if (axis.kind === "time") return (full ? fullDate : date).format(value);
  if (axis.kind === "category") return String(axis.categories[value] ?? "—");
  return (full ? number : axisNumber).format(value);
}

function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!numeric(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return [min, max];
}

export function mountChart(host, model, domain = {}) {
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
  const [lowX, highX] = domain.x || [ordered[0].value, ordered.at(-1).value];
  const padX = model.bars ? 0.5 : lowX === highX ? 1 : 0;
  let current = null;
  let highlight = null;
  let geometry;
  const seriesExtents = model.series.map((series) => extent(series.values));
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
      draw();
    });
    legend.append(button);
  });

  function draw() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (width < 100 || height < 100) return;
    const ctx = fitCanvas(canvas, width, height);
    const lineFont = "11px 'IBM Plex Mono', Consolas, 'Microsoft YaHei', monospace";
    ctx.font = lineFont;
    const live = model.series.map((series, index) => ({ ...series, index })).filter((_, index) => enabled[index]);
    const limits = live.map((series) => seriesExtents[series.index]);
    let lo = domain.y?.[0] ?? Math.min(...limits.map(([min]) => min));
    let hi = domain.y?.[1] ?? Math.max(...limits.map(([, max]) => max));
    if (model.bars || (lo > 0 && hi > lo * 2)) lo = Math.min(lo, 0);
    if (model.bars) hi = Math.max(hi, 0);
    if (lo === hi) {
      const padding = Math.abs(lo * 0.05) || 1;
      lo -= padding;
      hi += padding;
    }
    const ticks = scaleTicks(lo, hi);
    const left = Math.max(36, ...ticks.map((value) => ctx.measureText(axisNumber.format(value)).width + 12));
    const right = width - 19;
    const top = 30;
    const bottom = height - 50;
    const x = (value) => left + (value - lowX + padX) / (highX - lowX + padX * 2) * (right - left);
    const y = (value) => bottom - (value - ticks[0]) / (ticks.at(-1) - ticks[0]) * (bottom - top);
    geometry = { left, right, top, bottom };

    ctx.fillStyle = "#a6bbce";
    ctx.fillText(model.yLabel, left, 15, right - left);
    ticks.forEach((value) => {
      const py = y(value);
      ctx.strokeStyle = "#23394d";
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(axisNumber.format(value), left - 8, py + 3);
    });
    ctx.textAlign = "center";
    const divisions = Math.max(1, Math.min(width < 430 ? 3 : 5, ordered.length - 1));
    const tickAt = (index) => lowX + index / divisions * (highX - lowX);
    const xTicks = lowX === highX ? [lowX] : [...new Set(Array.from({ length: divisions + 1 }, (_, index) =>
      domain.x ? tickAt(index) : ordered[nearest(tickAt(index))].value))];
    xTicks.forEach((value, index) => {
      const px = x(value);
      ctx.fillStyle = "#a6bbce";
      if (!model.bars && xTicks.length > 1) {
        ctx.textAlign = index === 0 ? "left" : index === xTicks.length - 1 ? "right" : "center";
      }
      ctx.fillText(showX(axis, value), px, bottom + 18, (right - left) / Math.max(1, divisions));
    });
    ctx.textAlign = "right";
    ctx.fillText(axis.label, right, height - 10, right - left);

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
        const pitch = (right - left) / Math.max(1, ordered.length);
        const barWidth = Math.min(30, pitch * 0.74 / live.length);
        series.values.forEach((value, index) => {
          if (!numeric(value)) return;
          ctx.fillStyle = color;
          const px = x(axis.values[index]) + barWidth * (order - live.length / 2);
          const py = y(Math.max(0, value));
          ctx.fillRect(px + 1, py, Math.max(1, barWidth - 2), Math.max(1, Math.abs(y(value) - y(0))));
        });
        return;
      }
      const segments = [];
      let segment = [];
      series.values.forEach((value, index) => {
        if (numeric(value)) segment.push([x(axis.values[index]), y(value)]);
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
        if (order === 0 && points.length > 1) {
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
    });

    if (current !== null) {
      const { index, value } = ordered[current];
      const px = x(value);
      ctx.strokeStyle = "#a3bed2";
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      const title = document.createElement("strong");
      title.textContent = showX(axis, value, true);
      const rows = live.map((series) => {
        const entry = document.createElement("span");
        const exact = series.values[index];
        entry.textContent = series.name + "  " + (numeric(exact) ? number.format(exact) : "—");
        if (numeric(exact) && !model.bars) {
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
    let end = ordered.length - 1;
    while (start < end) {
      const middle = Math.floor((start + end) / 2);
      if (ordered[middle].value < value) start = middle + 1;
      else end = middle;
    }
    if (start > 0 && Math.abs(ordered[start - 1].value - value) < Math.abs(ordered[start].value - value)) return start - 1;
    return start;
  }

  canvas.addEventListener("pointermove", (event) => {
    if (!geometry) return;
    const px = event.clientX - canvas.getBoundingClientRect().left;
    const py = event.clientY - canvas.getBoundingClientRect().top;
    const { left, right, top, bottom } = geometry;
    const next = px < left || px > right || py < top || py > bottom ? null
      : nearest(lowX - padX + (px - left) / (right - left) * (highX - lowX + padX * 2));
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
    else if (event.key === "End") current = ordered.length - 1;
    else current = Math.max(0, Math.min(ordered.length - 1, (current ?? -1) + (event.key === "ArrowRight" ? 1 : -1)));
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

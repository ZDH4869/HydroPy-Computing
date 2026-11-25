import { makeChart, showX, timeValue } from "./charts.js";

const cyan = "#66e7f5";
const amber = "#f3b066";
const blue = "#85aaff";
const number = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 });
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);

function chart(table, x, y, unit, options = {}) {
  return makeChart(table, { x, y: [y], unit, ...options });
}

function panel(label, model) {
  return { label, model };
}

function compare(...panels) {
  const enabled = panels.filter(({ model }) => model);
  if (enabled.length < 2) return [];
  const axisKind = enabled[0].model.axis.kind;
  const sharedX = axisKind !== "category" && enabled.every(({ model }) => model.axis.kind === axisKind);
  const sharedY = enabled.every(({ model }) => model.yLabel === enabled[0].model.yLabel);
  const x = sharedX ? span(enabled.map(({ model }) => model.axis.values)) : undefined;
  const y = sharedY ? span(enabled.flatMap(({ model }) => model.series.map((series) => series.values))) : undefined;
  return enabled.map((item) => ({ ...item, domain: { x, y } }));
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

function fraction(value, limits, fallback = 0) {
  if (!isNumber(value) || !limits) return fallback;
  const [min, max] = limits;
  if (min === max) return 0.5;
  return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function read(label, value, unit = "") {
  return { label, value, unit };
}

function flowUnit(model) {
  return model.yLabel.match(/\(([^()]*)\)$/)?.[1] || "";
}

function seriesMode(title, model, {
  kind = "series", fields, units = [], formatX, extraModel,
} = {}) {
  if (!model) return null;
  const valid = model.axis.values.map((x, index) => ({ x, index }))
    .filter(({ x, index }) => isNumber(x) && model.series.some((item) => isNumber(item.values[index])))
    .sort((a, b) => a.x - b.x);
  if (!valid.length) return null;
  const ranges = model.series.map((item) => span([item.values]));
  const rainValues = new Map();
  if (extraModel) {
    extraModel.axis.values.forEach((value, index) => rainValues.set(value, extraModel.series[0].values[index]));
  }
  return {
    title,
    size: valid.length,
    frame(index) {
      const { x, index: row } = valid[index];
      const values = model.series.map((item) => item.values[row]);
      const metrics = model.series.map((item, i) => read(fields?.[i] || item.name, values[i], units[i] ?? flowUnit(model)));
      const rain = rainValues.get(x);
      if (isNumber(rain)) metrics.unshift(read("净雨量", rain, "mm"));
      return {
        track: model.track, kind, x,
        label: formatX ? formatX(x) : showX(model.axis, x, true),
        metrics,
        levels: values.map((value, i) => fraction(value, ranges[i])),
      };
    },
  };
}

function reservoirPlayback(table, group) {
  if (!group) return null;
  const columns = [
    "入库流量Q/(m³·s⁻¹)", "下泄流量q/(m³·s⁻¹)", "水库水位Z/m", "水库存水量V/万m³",
  ];
  const indexes = columns.map((column) => table.columns.indexOf(column));
  const xIndex = table.columns.indexOf("时间t/h");
  const rows = table.rows.filter((row) => isNumber(row[xIndex])).sort((a, b) => a[xIndex] - b[xIndex]);
  if (!rows.length || indexes.some((i) => i < 0)) return null;
  const levels = span([rows.map((row) => row[indexes[2]])]);
  const flows = span([rows.flatMap((row) => [row[indexes[0]], row[indexes[1]]])]);
  return {
    title: "调洪过程回放",
    size: rows.length,
    frame(index) {
      const row = rows[index];
      const [inflow, outflow, level, volume] = indexes.map((i) => row[i]);
      return {
        track: "series", kind: "reservoir", x: row[xIndex], label: number.format(row[xIndex]) + " h",
        metrics: [
          read("入库流量", inflow, "m³/s"), read("下泄流量", outflow, "m³/s"),
          read("水库水位", level, "m"), read("水库库容", volume, "万m³"),
        ],
        levels: [fraction(inflow, flows), fraction(outflow, flows), fraction(level, levels, 0.5)],
      };
    },
  };
}

function volumePresentation(table, observation) {
  if (!observation || !table) return { modes: [], comparison: [] };
  const [minTime] = span([observation.axis.values]) || [];
  const startCol = table.columns.indexOf("开始时间");
  const endCol = table.columns.indexOf("结束时间");
  const volumeCol = table.columns.indexOf("梯形法洪量/万m³");
  const modes = table.rows.flatMap((row, index) => {
    const start = timeValue(row[startCol]);
    const end = timeValue(row[endCol]);
    if (![minTime, start, end].every(isNumber)) return [];
    const candidates = observation.axis.values.filter((x) => x < start);
    const positions = [...candidates, start];
    return [{
      title: row[0],
      size: positions.length,
      frame(i) {
        const x = positions[i];
        const finish = x + end - start;
        const final = i === positions.length - 1;
        return {
          track: "window", kind: "window", x, label: final ? "最大洪量区间" : "窗口定位",
          window: [x, finish], final, durationIndex: index,
          metrics: [read("开始", showX(observation.axis, x)), read("结束", showX(observation.axis, finish)),
            ...(final ? [read("最大洪量 · 梯形法", row[volumeCol], "万m³")] : [])],
        };
      },
    }];
  });
  return { modes, comparison: [{ label: "洪水过程与积分窗口", model: observation }], comparisonLabel: "洪量窗口" };
}

function thiessenPresentation(data, table) {
  if (!data?.zones.length || !table?.rows.length) return { modes: [], comparison: [] };
  const zones = data.zones.slice(0, table.rows.length).map((item, index) => {
    const row = table.rows[index];
    return {
      ...item, name: String(row[0]), rainfall: row[1], area: row[2], weight: row[3],
    };
  });
  const mode = {
    title: "逐站查看面积权重", size: zones.length,
    frame(index) {
      const zone = zones[index];
      return {
        kind: "station", stationIndex: index, label: "站点 " + zone.name,
        metrics: [
          read("站点雨量", zone.rainfall, "mm"), read("控制面积", zone.area, "km²"),
          read("面积权重", isNumber(zone.weight) ? zone.weight * 100 : null, "%"),
          read("面积加权贡献", isNumber(zone.rainfall) && isNumber(zone.weight) ? zone.rainfall * zone.weight : null, "mm"),
        ],
      };
    },
  };
  return { modes: [mode], comparison: [], map: { ...data, zones } };
}

export function buildPresentation(algo, tables, groups, visuals, settings, inputPreview) {
  const [primary, secondary] = groups;
  const flow = "流量 (m³/s)";
  const table = tables[0];
  let comparison = [];
  let mode;
  switch (algo.id) {
    case "eng_linear_interp":
    case "eng_multi_interp": {
      const observed = chart(visuals.observations, "时间", "观测流量", flow, { scatter: true, color: amber });
      comparison = compare(panel("原始观测点", observed), panel("计算后的插值", primary?.chart));
      mode = seriesMode("插值过程回放", primary?.chart, { kind: "gauge" });
      break;
    }
    case "eng_max_volume":
    case "eng_area_integral":
      return volumePresentation(table, chart(visuals.observations, "时间", "流量", flow, { track: "window" }));
    case "thiessen":
      return thiessenPresentation(visuals.station_areas, table);
    case "muskingum_xk":
      comparison = compare(panel("蓄量与加权流量", primary?.chart), panel("X 与相关系数", secondary?.chart))
        .map((item) => ({ ...item, domain: {} }));
      mode = seriesMode("X 参数扫描", secondary?.chart, { formatX: (x) => "X = " + number.format(x) });
      break;
    case "muskingum_route": {
      const upstream = chart(table, "时段", "上游流量I", flow, { color: cyan });
      const downstream = chart(table, "时段", "下游流量Q", flow, { color: amber });
      comparison = compare(panel("上游来流", upstream), panel("下游出流", downstream));
      mode = seriesMode("河道流量演算回放", primary?.chart, { kind: "reach", formatX: (x) => "时段 " + x });
      break;
    }
    case "nash_nk":
      comparison = compare(panel("输入净雨", primary?.chart), panel("实测出口流量", secondary?.chart));
      mode = seriesMode("净雨与出口流量回放", secondary?.chart, {
        kind: "rain", extraModel: primary?.chart, formatX: (x) => number.format(x) + " h",
      });
      break;
    case "nash_duh": {
      const reference = chart(visuals.reference, "时间 (h)", "标准流量", flow, { color: amber });
      comparison = compare(panel("标准曲线", reference), panel("计算时段单位线", primary?.chart));
      mode = seriesMode("时段单位线回放", primary?.chart, { kind: "gauge", formatX: (x) => number.format(x) + " h" });
      break;
    }
    case "sherman_derive": {
      const observed = chart(visuals.runoff, "时段", "地面径流", flow, { color: amber });
      comparison = compare(panel("地面径流输入", observed), panel("推求的单位线", primary?.chart));
      mode = seriesMode("Sherman 单位线回放", primary?.chart, { kind: "gauge", formatX: (x) => "时段 " + x });
      break;
    }
    case "sherman_outlet": {
      const rain = chart(visuals.rain_response, "时间(时段)", "净雨量", "净雨量 (mm)", { bars: true, color: blue });
      const arrival = chart(table, "时间(时段)", "计算流量(m³/s)", flow);
      const observed = chart(visuals.reference, "时间(时段)", "实测流量", flow, { color: amber });
      comparison = compare(panel("输入净雨", rain), panel("计算出口流量", arrival), panel("实测出口流量", observed));
      // The common time ruler is useful even when the rainfall scale is independent.
      if (arrival && observed) {
        const y = span([arrival.series[0].values, observed.series[0].values]);
        for (const item of comparison) if (item.model.yLabel === flow) item.domain.y = y;
      }
      mode = seriesMode("净雨与出口流量回放", arrival, { kind: "rain", extraModel: rain, formatX: (x) => "时段 " + x });
      break;
    }
    case "flood_curve_interp": {
      const x = table?.columns[0] || settings.x_col;
      const y = primary?.y[0];
      const observed = inputPreview?.columns.includes(x) && inputPreview.columns.includes(y)
        ? chart(inputPreview, x, y, y, { scatter: true, color: amber }) : null;
      const title = /水位|(?:^|[^a-z])z(?:$|[^a-z])/i.test(x || "") ? "水位–曲线关系浏览" : "插值曲线浏览";
      comparison = compare(panel("原始曲线预览", observed), panel("插值后的曲线", primary?.chart));
      mode = seriesMode(title, primary?.chart, {
        kind: "relation", fields: [y], formatX: (value) => x + " " + number.format(value),
      });
      if (mode && title.startsWith("水位")) {
        const previous = mode.frame;
        const levels = span([primary.chart.axis.values]);
        mode.frame = (index) => {
          const frame = previous(index);
          return { ...frame, kind: "level", levels: [0, 0, fraction(frame.x, levels, 0.5)], metrics: [
            read(x, frame.x), ...frame.metrics,
          ] };
        };
      }
      break;
    }
    case "flood_trial":
    case "flood_rk4": {
      const inflow = chart(table, "时间t/h", "入库流量Q/(m³·s⁻¹)", flow, { color: cyan });
      const outflow = chart(table, "时间t/h", "下泄流量q/(m³·s⁻¹)", flow, { color: amber });
      comparison = compare(panel("入库过程", inflow), panel("下泄过程", outflow));
      mode = reservoirPlayback(table, primary);
      break;
    }
  }
  return { comparison, modes: mode ? [mode] : [] };
}

export function createPlayback(root, readouts, onFrame) {
  const position = root.querySelector("input");
  const play = root.querySelector("[data-play]");
  const restart = root.querySelector("[data-restart]");
  const selectMode = root.querySelector("[data-mode]");
  const speed = root.querySelector("[data-speed]");
  const stamp = root.querySelector("output");
  const caption = root.querySelector("[data-playback-label]");
  let modes = [];
  let mode;
  let index = 0;
  let playing = false;
  let timer;
  let current = null;

  function paint() {
    position.value = index;
    current = mode.frame(index);
    stamp.textContent = current.label;
    position.setAttribute("aria-valuetext", current.label);
    readouts.replaceChildren(...current.metrics.map(({ label, value, unit }) => {
      const block = document.createElement("div");
      const term = document.createElement("dt");
      const amount = document.createElement("dd");
      term.textContent = label;
      const precision = { "m³/s": 2, "万m³": 1, m: 2, mm: 2, "km²": 1, "%": 2 }[unit] ?? 4;
      amount.textContent = isNumber(value)
        ? value.toLocaleString("zh-CN", { maximumFractionDigits: precision }) : value ?? "—";
      if (unit) {
        const suffix = document.createElement("span");
        suffix.textContent = unit;
        amount.append(" ", suffix);
      }
      block.append(term, amount);
      return block;
    }));
    onFrame(current);
  }

  function stop() {
    playing = false;
    window.clearTimeout(timer);
    play.textContent = "▶";
    play.setAttribute("aria-label", "播放");
    play.setAttribute("aria-pressed", "false");
  }

  function applyMode(value) {
    stop();
    mode = modes[value];
    index = 0;
    position.max = mode.size - 1;
    position.disabled = mode.size < 2;
    play.disabled = mode.size < 2;
    restart.disabled = mode.size < 2;
    caption.textContent = modes.length > 1 ? "定位最大洪量窗口" : mode.title;
    paint();
  }

  function tick() {
    const wait = Math.max(75, Math.min(900, 8500 / Math.max(1, mode.size - 1))) / Number(speed.value);
    timer = window.setTimeout(() => {
      index = Math.min(mode.size - 1, index + Math.max(1, Math.ceil((mode.size - 1) / 112)));
      paint();
      if (index >= mode.size - 1) stop();
      else tick();
    }, wait);
  }

  play.addEventListener("click", () => {
    if (playing) return stop();
    if (index >= mode.size - 1) seek(0);
    playing = true;
    play.textContent = "Ⅱ";
    play.setAttribute("aria-label", "暂停");
    play.setAttribute("aria-pressed", "true");
    tick();
  });
  restart.addEventListener("click", () => { stop(); seek(0); });
  position.addEventListener("input", () => { stop(); seek(Number(position.value)); });
  selectMode.addEventListener("change", () => applyMode(Number(selectMode.value)));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });

  function seek(next) {
    if (!mode) return;
    index = Math.min(mode.size - 1, Math.max(0, next));
    paint();
  }

  return {
    current: () => current,
    seek(next) { stop(); seek(next); },
    load(next = []) {
      stop();
      current = null;
      modes = next;
      mode = null;
      root.hidden = readouts.hidden = next.length === 0;
      if (!next.length) {
        readouts.replaceChildren();
        onFrame(null);
        return;
      }
      selectMode.replaceChildren(...modes.map((item, i) => new Option(item.title, i)));
      selectMode.hidden = modes.length === 1;
      applyMode(0);
    },
  };
}

export function createScene(canvas, image) {
  const surface = canvas.parentElement;
  let frame;
  let scene;

  function draw() {
    const width = surface.clientWidth;
    const height = surface.clientHeight;
    if (!width || !height) return;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const context = canvas.getContext("2d");
    context.setTransform(scale, 0, 0, scale, 0, 0);
    if (!frame || !image.naturalWidth) return;

    const fit = Math.max(width / image.naturalWidth, height / image.naturalHeight);
    const imageWidth = image.naturalWidth * fit;
    const imageHeight = image.naturalHeight * fit;
    const x = (value) => (width - imageWidth) / 2 + value * imageWidth;
    const y = (value) => (height - imageHeight) / 2 + value * imageHeight;
    const first = frame.levels?.[0] ?? 0;
    const second = frame.levels?.[1] ?? 0;

    function marker(px, py, strength, color) {
      if (px < 0 || px > width || py < 0 || py > height) return;
      const radius = 10 + 19 * strength;
      context.beginPath();
      context.arc(px, py, radius, 0, Math.PI * 2);
      context.strokeStyle = color + "91";
      context.lineWidth = 1.4;
      context.stroke();
      context.beginPath();
      context.arc(px, py, 3.5 + 3.5 * strength, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
    }

    function arrow(px, py, strength, color) {
      const length = 19 + strength * 48;
      const end = px + length;
      context.lineWidth = 1.5 + 2.5 * strength;
      context.lineCap = "round";
      context.strokeStyle = color;
      context.beginPath();
      context.moveTo(px, py);
      context.lineTo(end, py);
      context.moveTo(end - 7, py - 5);
      context.lineTo(end, py);
      context.lineTo(end - 7, py + 5);
      context.stroke();
    }

    if (scene === "reservoir" && ["reservoir", "level"].includes(frame.kind)) {
      const level = frame.levels[2] ?? 0.5;
      const value = y(0.459 - level * 0.097);
      const left = Math.max(x(0.28), width < 620 ? width * 0.51 : 0);
      const right = x(0.737);
      const waterlineLength = right - left;
      const glow = context.createLinearGradient(0, value, 0, value + 46);
      glow.addColorStop(0, "#60e8f064");
      glow.addColorStop(1, "#60e8f000");
      context.fillStyle = glow;
      context.fillRect(left, value, waterlineLength, 46);
      context.strokeStyle = "#a3fbff";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(left, value);
      context.lineTo(right, value);
      context.stroke();
      marker(right - 16, value, level * 0.6, cyan);
      if (frame.kind === "reservoir") {
        if (width >= 620) arrow(x(0.29), y(0.46), first, cyan);
        arrow(x(0.857), y(0.658), second, amber);
      }
    } else if (scene === "station" && ["gauge", "window"].includes(frame.kind)) {
      marker(x(0.652), y(0.66), first, frame.final ? amber : cyan);
    } else if (scene === "network" && frame.kind !== "station") {
      marker(x(0.466), y(0.327), frame.kind === "rain" ? 0.39 : first, blue);
      marker(x(0.695), y(0.694), second || first, amber);
    }
  }

  const observer = new ResizeObserver(draw);
  observer.observe(surface);
  image.addEventListener("load", draw);
  return {
    setScene(name) {
      if (scene === name) return;
      scene = name;
      image.src = "/static/scenes/" + name + ".webp";
      draw();
    },
    setFrame(next) {
      frame = next;
      draw();
    },
  };
}

function rings(geometry) {
  if (geometry?.type === "Polygon") return geometry.coordinates;
  if (geometry?.type === "MultiPolygon") return geometry.coordinates.flat();
  if (geometry?.type === "GeometryCollection") return geometry.geometries.flatMap(rings);
  return [];
}

export function mountStationMap(host, map, seek) {
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
    button.addEventListener("click", () => seek(index));
    list.append(button);
    return button;
  });
  const note = document.createElement("p");
  note.className = "chart-footer-note";
  note.textContent = map.crs + " · 当前算例的控制区";
  stage.append(canvas, list, note);
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
      ctx.fillStyle = index === focused ? "#2d9eb2" : index % 2 === 0 ? "#184056" : "#20495b";
      ctx.fill("evenodd");
      ctx.strokeStyle = "#5b97a8";
      ctx.stroke();
    });
    outline(map.boundary);
    ctx.strokeStyle = "#a3d6e3";
    ctx.stroke();
    stationPoints = map.zones.map(({ point }) => [x(point[0]), y(point[1])]);
    stationPoints.forEach(([px, py], index) => {
      ctx.fillStyle = index === focused ? amber : "#c6eaf4";
      ctx.beginPath();
      ctx.arc(px, py, index === focused ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  canvas.addEventListener("click", (event) => {
    let closest = 19;
    let selected = -1;
    stationPoints.forEach(([x, y], index) => {
      const distance = Math.hypot(x - event.offsetX, y - event.offsetY);
      if (distance < closest) {
        closest = distance;
        selected = index;
      }
    });
    if (selected >= 0) seek(selected);
  });
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  draw();
  const destroy = () => observer.disconnect();
  destroy.setFrame = (frame) => {
    if (!Number.isInteger(frame?.stationIndex)) return;
    focused = frame.stationIndex;
    buttons.forEach((button, index) => button.setAttribute("aria-pressed", String(index === focused)));
    draw();
  };
  return destroy;
}

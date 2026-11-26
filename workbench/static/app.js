import { inputCharts, mountChart, parseResultCsv, resultCharts } from "./charts.js";
import { buildPresentation, mountStationMap } from "./presentation.js";
import { loadCaseHistory, saveCaseRun } from "./case-storage.js";

const ICONS = {
  river: '<path d="M3 7c3-3 6-3 9 0s6 3 9 0M3 12c3-3 6-3 9 0s6 3 9 0M3 17c3-3 6-3 9 0s6 3 9 0"/>',
  satellite: '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8.5" stroke-dasharray="2 3"/><path d="m4 4 4 4m8 8 4 4M20 4l-4 4M8 16l-4 4"/>',
  dam: '<path d="m5 19 2-14h10l2 14M3 11c2.5-2 4.5 2 7 0s4.5 2 7 0 3.5-1 4 0M3 17c2.5-2 4.5 2 7 0s4.5 2 7 0 3.5-1 4 0"/>',
  file: '<path d="M6 3h8l4 4v14H6zM14 3v5h4M9 12h6m-6 4h6"/>',
  upload: '<path d="M12 15V4M8 8l4-4 4 4M5 15v5h14v-5"/>',
  download: '<path d="M12 4v11m-4-4 4 4 4-4M5 17v3h14v-3"/>',
  chart: '<path d="M4 4v16h16M7 15l4-5 4 2 5-7"/>',
  play: '<path d="m9 6 9 6-9 6z" fill="currentColor" stroke="none"/>',
  expand: '<path d="M9 4H4v5m11-5h5v5M4 15v5h5m11-5v5h-5"/>',
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const format = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 6 });
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#39;");
const icon = (name) => `<svg class="icon" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${ICONS[name]}</svg>`;
const count = (value) => format.format(value);
const valueText = (value) => value === null || value === undefined || value === ""
  ? "—"
  : typeof value === "number" ? format.format(value) : String(value);

let catalog;
let workspace;
let activeRun;
let plots = [];
let lightboxTrigger;
const workspaces = new Map();
const lastAlgorithmByCourse = new Map();
const caseRuns = new Map();
const comparedRuns = ["", ""];
const casePlots = [null, null];
const runTimestamp = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  fractionalSecondDigits: 3,
});
let historyLoading = true;
let historyUnavailable = false;
const viewKey = { input: "输入", output: "结果", compare: "对照", original: "原始图" };
const sceneLabels = {
  engineering: "水文观测场景示意",
  informatization: "流域河网场景示意",
  flood: "水库场景示意",
};

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({ error: `服务器返回 HTTP ${response.status}` }));
  if (!response.ok) throw Object.assign(new Error(data.error || `HTTP ${response.status}`), { payload: data });
  return data;
}

function toast(title, message = "", error = false) {
  const node = document.createElement("div");
  node.className = `toast${error ? " error" : ""}`;
  node.innerHTML = `<strong>${escapeHtml(title)}</strong>${message ? `<span>${escapeHtml(message)}</span>` : ""}`;
  $("#toastStack").append(node);
  window.setTimeout(() => node.remove(), 5000);
}

function setEngine(ready, label) {
  $("#engineStatus").innerHTML = `<span class="status-dot ${ready ? "ready" : "error"}"></span><span>${label}</span>`;
}

function buildWorkspace(algo) {
  const files = Object.fromEntries(algo.files.map((spec) => [spec.key, {
    spec, token: spec.sample ? `sample:${spec.sample}` : null, name: spec.sample || "",
    sample: Boolean(spec.sample), preview: null, loading: false, uploading: false, error: "",
  }]));
  return {
    algo, files, fileKey: algo.files.find((spec) => spec.sample)?.key || algo.files[0]?.key,
    values: Object.fromEntries(algo.fields.map((field) => [field.key, String(field.default ?? "")])),
    inputGroup: 0, outputGroup: 0, selectedWindow: 0, selectedStation: 0,
    pane: "settings", view: "input", result: null, changed: false,
  };
}

function showPane(pane, focus = false) {
  workspace.pane = pane;
  $(".workspace").dataset.pane = pane;
  for (const button of $$("#workspaceTabs button")) {
    const active = button.dataset.pane === pane;
    button.setAttribute("aria-pressed", String(active));
    if (active && focus && button.getClientRects().length) button.focus({ preventScroll: true });
  }
}

function selectAlgorithm(id, reset = false) {
  const algo = catalog.algorithms.find((item) => item.id === id);
  if (!algo || (workspace?.algo.id === id && !reset)) return;
  if (reset || !workspaces.has(id)) workspaces.set(id, buildWorkspace(algo));
  workspace = workspaces.get(id);
  lastAlgorithmByCourse.set(algo.course, id);
  showPane(workspace.pane);
  history.replaceState(null, "", `#${algo.id}`);
  renderNavigation();
  renderSettings();
  renderChart();
  renderDetails();
  updateRunState();
  if (workspace.fileKey) loadPreview(workspace, workspace.fileKey);
}

function renderNavigation() {
  const course = catalog.courses.find((item) => item.id === workspace.algo.course);
  document.body.dataset.course = course.id;
  $("#sceneLabel").textContent = sceneLabels[course.id];
  const algorithms = catalog.algorithms.filter((algo) => algo.course === course.id);
  $("#courseTabs").innerHTML = catalog.courses.map((item) =>
    `<button class="course-tab" type="button" data-course="${item.id}" aria-pressed="${item.id === course.id}">
       ${icon(item.icon)}<span>${escapeHtml(item.name)}</span></button>`
  ).join("");
  $("#algoRail").innerHTML = algorithms.map((algo) =>
    `<button class="algorithm" type="button" data-algo="${algo.id}" aria-pressed="${algo.id === workspace.algo.id}">
       ${escapeHtml(algo.name)}</button>`
  ).join("");
  $("#breadcrumb").textContent = course.name;
  $("#algoTitle").textContent = workspace.algo.name;
  $("#algoDescription").textContent = workspace.algo.short || workspace.algo.description;
}

function fieldHtml(field) {
  const id = `field-${field.key}`;
  const value = workspace.values[field.key];
  const attrs = `id="${id}" data-field="${field.key}"${field.help ? ` aria-describedby="${id}-help"` : ""}`;
  let control;
  if (field.type === "select") {
    control = `<select ${attrs}>${field.options.map((option) => {
      const key = typeof option === "string" ? option : option.value;
      const label = typeof option === "string" ? option : option.label;
      return `<option value="${escapeHtml(key)}"${String(key) === String(value) ? " selected" : ""}>${escapeHtml(label)}</option>`;
    }).join("")}</select>`;
  } else if (field.type === "column") {
    control = `<select ${attrs} data-column="true"><option value="${escapeHtml(value)}">${escapeHtml(value || "选择数据后可用")}</option></select>`;
  } else if (field.type === "textarea") {
    control = `<textarea ${attrs} rows="3">${escapeHtml(value)}</textarea>`;
  } else {
    const type = field.type === "number" ? "number" : "text";
    const limits = ["min", "max", "step"].filter((key) => field[key] !== undefined)
      .map((key) => `${key}="${escapeHtml(field[key])}"`).join(" ");
    control = `<input ${attrs} type="${type}" value="${escapeHtml(value)}" ${limits} autocomplete="off">`;
  }
  return `<div class="field${field.full || field.type === "column" || field.type === "textarea" ? " full" : ""}">
    <label for="${id}">${escapeHtml(field.label)}${field.unit ? ` (${escapeHtml(field.unit)})` : ""}</label>
    ${control}${field.help ? `<small id="${id}-help">${escapeHtml(field.help)}</small>` : ""}</div>`;
}

function renderSettings() {
  const fields = workspace.algo.fields;
  const advanced = fields.filter((field) => field.advanced);
  $("#paramsBlock").hidden = fields.length === 0;
  $("#inputBlock").hidden = workspace.algo.files.length === 0;
  $("#paramFields").innerHTML = fields.filter((field) => !field.advanced).map(fieldHtml).join("");
  $("#advancedFields").innerHTML = advanced.map(fieldHtml).join("");
  $("#advancedToggle").hidden = advanced.length === 0;
  $("#advancedBox").close();
  $(".settings-body").scrollTop = 0;
  renderFiles();
  refreshColumns(workspace.files[workspace.fileKey]?.preview);
}

function renderFiles() {
  $("#fileList").innerHTML = Object.entries(workspace.files).map(([key, file]) => {
    const title = escapeHtml(file.spec.label);
    const name = escapeHtml(file.name || "选择文件");
    return `<div class="file-card${file.uploading ? " uploading" : ""}" data-drop="${key}" aria-busy="${file.uploading}">
      <span class="file-ico">${icon("file")}</span>
      <div class="file-info">
        <button class="file-preview" type="button" data-preview-file="${key}" title="查看${title}">${title}</button>
        <div class="file-name" title="${name}">${file.uploading ? "正在上传…" : name}</div>
      </div>
      ${!file.spec.required && file.token ? `<button class="clear-file" type="button" data-clear="${key}" aria-label="移除${title}" title="移除可选文件">×</button>` : ""}
      <button class="icon-button" type="button" data-upload="${key}" aria-label="上传${title}" title="上传 ${escapeHtml(file.spec.accept)}" ${file.uploading ? "disabled" : ""}>${icon("upload")}</button>
      <input type="file" accept="${escapeHtml(file.spec.accept)}" data-file-input="${key}" hidden>
    </div>`;
  }).join("");
}

function refreshColumns(preview) {
  if (!preview?.columns?.length) return;
  $$("[data-column]").forEach((select) => {
    const current = select.value;
    select.replaceChildren(...preview.columns.map((name) => new Option(name, name)));
    select.value = preview.columns.includes(current) ? current : preview.columns[0];
    workspace.values[select.dataset.field] = select.value;
  });
}

function getInputGroup(context) {
  const file = context.files[context.fileKey];
  const x = file?.spec.key === "curve_csv" ? $("#field-x_col")?.value : undefined;
  return inputCharts(file?.preview, x, context.fileKey);
}

async function loadPreview(context, key) {
  const entry = context.files[key];
  if (!entry?.token || entry.loading || entry.preview) return;
  const token = entry.token;
  entry.loading = true;
  entry.error = "";
  if (workspace === context && context.view === "input") renderChart();
  try {
    const data = await api(`/api/preview?token=${encodeURIComponent(token)}`);
    if (entry.token === token) {
      entry.preview = data;
      if (workspace === context && key === context.fileKey) refreshColumns(data);
    }
  } catch (error) {
    if (entry.token === token) entry.error = error.message;
  } finally {
    entry.loading = false;
    if (workspace === context && context.view === "input" && key === context.fileKey) renderChart();
  }
}

async function uploadFile(key, file) {
  const context = workspace;
  const entry = context.files[key];
  if (entry.uploading) return;
  if (!entry.spec.accept.split(",").some((extension) => file.name.toLowerCase().endsWith(extension.trim()))) {
    toast("文件格式不匹配", `请选择 ${entry.spec.accept} 文件`, true);
    return;
  }
  entry.uploading = true;
  renderFiles();
  updateRunState();
  const form = new FormData();
  form.append("file", file);
  form.append("role", entry.spec.role || "");
  try {
    const data = await api("/api/upload", { method: "POST", body: form });
    Object.assign(entry, { token: data.token, name: data.name || file.name, sample: false, preview: data, error: "" });
    context.fileKey = key;
    context.inputGroup = 0;
    context.view = "input";
    if (context.result || activeRun === context) context.changed = true;
    if (workspace === context) {
      refreshColumns(data);
      renderChart();
      toast("文件已载入", entry.name);
    }
  } catch (error) {
    if (workspace === context) toast("上传失败", error.message, true);
  } finally {
    entry.uploading = false;
    if (workspace === context) {
      renderFiles();
      updateRunState();
    }
  }
}

function chooseFile(key) {
  if (!workspace.files[key]) return;
  workspace.fileKey = key;
  workspace.inputGroup = 0;
  workspace.view = "input";
  refreshColumns(workspace.files[key].preview);
  renderChart();
  loadPreview(workspace, key);
}

function choiceButtons(items, selected, key) {
  return items.map(({ value, label }) =>
    `<button type="button" data-${key}="${escapeHtml(value)}" aria-pressed="${value === selected}">${escapeHtml(label)}</button>`
  ).join("");
}

function setChoices(element, items, selected, key) {
  element.hidden = items.length < 2;
  element.innerHTML = element.hidden ? "" : choiceButtons(items, selected, key);
}

function emptyChart(title, description, busy = false) {
  $("#chartMount").innerHTML = `<div class="empty-chart${busy ? " busy" : ""}">
    <div class="empty-icon">${icon("chart")}</div>
    <strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p></div>`;
}

function compactTable(table, limit = 6) {
  const rows = table.rows.slice(0, limit);
  const { columns, total_rows } = table;
  const download = table.csv_url
    ? `<a class="download-link" href="${escapeHtml(table.csv_url)}" download>${icon("download")} 下载 CSV</a>`
    : "";
  const grid = rows.length && columns.length ? `<div class="table-wrap" tabindex="0" role="region" aria-label="${escapeHtml(table.title || table.name || "数据预览")}"><table>
    <thead><tr>${columns.map((column) => `<th scope="col">${escapeHtml(column)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((row) => `<tr>${columns.map((_, index) => `<td>${escapeHtml(valueText(row[index]))}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>` : "";
  return `<div class="table-meta"><span>${rows.length ? `前 ${rows.length} 行 · ` : ""}共 ${count(total_rows)} 行</span>${download}</div>${grid}`;
}

function showOriginal(images) {
  $("#chartMount").innerHTML = `<div class="figures">${images.map((image, index) =>
    `<figure class="result-figure"><button type="button" data-figure="${index}" aria-label="放大${escapeHtml(image.title)}">
      <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.title)}"><span class="zoom-icon">${icon("expand")}</span>
    </button><figcaption>${escapeHtml(image.title)}</figcaption></figure>`
  ).join("")}</div>`;
}

function plotChart(host, model) {
  const plot = mountChart(host, model);
  plots.push(plot);
  return plot;
}

function renderComparison(result) {
  const { comparison, comparisonLabel, windows } = result.presentation;
  $("#chartTitle").textContent = comparisonLabel || "算例对照";
  const grid = document.createElement("div");
  grid.className = comparison.length === 1 ? "comparison single" : "comparison";
  const charts = comparison.map(({ label, model }) => {
    const panel = document.createElement("section");
    const heading = document.createElement("h4");
    const plot = document.createElement("div");
    heading.textContent = label;
    panel.append(heading, plot);
    grid.append(panel);
    return plotChart(plot, model);
  });
  $("#chartMount").replaceChildren(grid);
  if (!windows?.length) return;

  const toolbar = document.createElement("div");
  toolbar.className = "window-picker";
  const label = document.createElement("label");
  label.textContent = "统计时段";
  const select = document.createElement("select");
  select.replaceChildren(...windows.map((item, index) => new Option(item.title, index)));
  select.value = workspace.selectedWindow;
  const details = document.createElement("span");
  label.append(select);
  toolbar.append(label, details);
  grid.before(toolbar);

  const showWindow = () => {
    const selection = Number(select.value);
    workspace.selectedWindow = selection;
    const window = windows[selection];
    charts[0].setWindow(window.range);
    details.textContent = `${window.start} 至 ${window.end} · 梯形法 ${valueText(window.volume)} 万m³`;
  };
  select.addEventListener("change", showWindow);
  showWindow();
}

function renderChart() {
  plots.forEach((destroy) => destroy());
  plots = [];
  const context = workspace;
  const result = context.result;
  const files = context.algo.files;
  const focused = document.activeElement;
  const views = [{ value: "input", label: viewKey.input }, { value: "output", label: viewKey.output }];
  if (result?.presentation?.comparison.length) {
    views.push({ value: "compare", label: result.presentation.comparisonLabel || viewKey.compare });
  }
  if ((result?.groups.length || result?.presentation?.map) && result.images?.length) {
    views.push({ value: "original", label: viewKey.original });
  }
  $("#viewSwitch").innerHTML = choiceButtons(views, context.view, "view");
  setChoices($("#fileTabs"), context.view === "input" ? files.map(({ key, label }) => ({ value: key, label })) : [], context.fileKey, "file-tab");

  setChoices($("#chartGroups"), [], 0, "group");
  $("#chartFoot").hidden = true;
  if (focused?.dataset.fileTab) {
    $$("#fileTabs button").find((button) => button.dataset.fileTab === focused.dataset.fileTab)
      ?.focus({ preventScroll: true });
  }

  if (context.view === "input") {
    const file = context.files[context.fileKey];
    const preview = file?.preview;
    $("#chartTitle").textContent = file?.spec.label || "输入数据";
    $("#chartMeta").textContent = preview
      ? `${file.sample ? "内置示例" : "已上传文件"} · ${count(preview.total_rows)} ${preview.kind === "shapefile" ? "个要素" : "条记录"}`
      : file?.name || "";
    const groups = getInputGroup(context);
    const index = Math.min(context.inputGroup, Math.max(0, groups.length - 1));
    setChoices($("#chartGroups"), groups.map((group, i) => ({ value: i, label: group.label })), index, "group");
    if (groups[index]) {
      const plot = mountChart($("#chartMount"), groups[index].chart);
      plots.push(plot);
    } else if (preview?.kind === "shapefile") {
      emptyChart("空间数据已就绪", "运行计算后查看流域边界、站点与泰森多边形。");
    } else if (preview) {
      emptyChart("暂无可绘制的数值序列", "可通过「查看输入记录」查看文件内容。");
    } else if (file?.loading) {
      emptyChart("正在读取输入数据", file.name, true);
    } else if (file?.error) {
      emptyChart("预览读取失败", file.error);
    } else {
      emptyChart("请选择输入文件", file?.spec.help || "上传数据后可在这里查看预览。");
    }
    if (preview) {
      const truncated = preview.rows.length < preview.total_rows;
      const caption = preview.kind === "shapefile"
        ? [preview.meta?.geometry_type, preview.meta?.crs].filter(Boolean).join(" · ")
        : truncated ? `曲线使用预览的前 ${preview.rows.length} 行，计算使用完整文件` : file.name;
      $("#chartFoot").hidden = false;
      $("#chartFoot").innerHTML = `<div class="chart-footer-note">${escapeHtml(caption)}</div>
        <button type="button" class="detail-button" data-detail="input">查看输入记录</button>`;
    }
    return;
  }

  $("#chartTitle").textContent = context.view === "original" ? "原始计算图" : "计算结果";
  $("#chartMeta").textContent = result?.ok ? `Python 计算完成 · ${result.duration.toFixed(2)} s` : "";
  if (activeRun === context) {
    emptyChart("正在计算", "本地 Python 引擎正在处理当前算例。", true);
    return;
  }
  if (!result?.ok) {
    emptyChart(result ? "本次计算未完成" : "等待运行", result?.error || "运行后在此查看过程线、关键指标和完整结果。");
    return;
  }
  if (context.view === "compare") {
    renderComparison(result);
    return;
  }
  if (context.view === "output" && result.presentation?.map) {
    $("#chartTitle").textContent = "泰森多边形控制区";
    plots.push(mountStationMap($("#chartMount"), result.presentation.map, context.selectedStation, (index) => {
      context.selectedStation = index;
    }));
    return;
  }
  if (context.view === "original" || (!result.groups.length && result.images?.length)) {
    if (context.view === "output") $("#chartTitle").textContent = result.images[0].title;
    showOriginal(result.images);
    return;
  }

  const group = result.groups[context.outputGroup];
  if (!group) {
    emptyChart("结果已生成", "在下方查看计算指标和逐行输出。");
    return;
  }
  $("#chartTitle").textContent = group.label;
  setChoices($("#chartGroups"), result.groups.map((item, index) => ({ value: index, label: item.label })), context.outputGroup, "group");
  plotChart($("#chartMount"), group.chart);
  const { table } = group;
  $("#chartFoot").hidden = false;
  const points = table.rows.length < table.total_rows
    ? `曲线显示前 ${count(table.rows.length)} / ${count(table.total_rows)} 行`
    : `${count(table.total_rows)} 条计算记录`;
  const note = group.useRowIndex ? " · 按记录顺序，精确时间见原始图" : "";
  $("#chartFoot").innerHTML = `<div class="chart-footer-note">${points}${note}</div>
    ${table.csv_url ? `<a class="download-link" href="${escapeHtml(table.csv_url)}" download>${icon("download")} 下载 CSV</a>` : ""}`;
}

function summaryHtml(summary) {
  return summary.map((item) =>
    `<div><dt>${escapeHtml(item.label)}</dt><dd>${escapeHtml(valueText(item.value))}<span>${escapeHtml(item.unit)}</span></dd></div>`
  ).join("");
}

function renderDetails() {
  const result = workspace.result;
  const host = $("#outputDetails");
  const metrics = $("#outputMetrics");
  host.hidden = !result;
  metrics.hidden = !result?.summary?.length;
  $('#workspaceTabs [data-pane="metrics"]').disabled = metrics.hidden;
  if (metrics.hidden && $(".workspace").dataset.pane === "metrics") showPane("chart");
  if (!result) {
    host.replaceChildren();
    metrics.replaceChildren();
    return;
  }
  const summary = summaryHtml(result.summary || []);
  metrics.innerHTML = summary ? `<h2 id="summaryHeading">关键指标</h2><dl class="summary-grid">${summary}</dl>` : "";
  host.innerHTML = [
    result.tables?.length && '<button class="detail-button" type="button" data-detail="tables">结果数据与下载</button>',
    result.stdout && '<button class="detail-button" type="button" data-detail="log">计算日志</button>',
  ].filter(Boolean).join("");
  host.hidden = !host.children.length;
}

function openDetails(kind) {
  const preview = workspace.files[workspace.fileKey]?.preview;
  const result = workspace.result;
  let title;
  let content;
  if (kind === "input" && preview) {
    title = "输入记录";
    content = compactTable(preview);
  } else if (kind === "tables" && result?.tables?.length) {
    title = "结果数据与下载";
    content = `<div class="table-stack">${result.tables.map((table) =>
      `<section><h3>${escapeHtml(table.title)}</h3>${compactTable(table)}</section>`
    ).join("")}</div>`;
  } else if (kind === "log" && result?.stdout) {
    title = "计算日志";
    content = `<pre>${escapeHtml(result.stdout)}</pre>`;
  } else {
    return;
  }
  $("#detailsTitle").textContent = title;
  $("#detailsBody").innerHTML = content;
  $("#detailsDialog").showModal();
}

function markEdited() {
  if (workspace.result || activeRun === workspace) workspace.changed = true;
  updateRunState();
}

function updateRunState() {
  const uploading = Object.values(workspace?.files || {}).some((file) => file.uploading);
  $("#resetButton").disabled = !workspace || activeRun === workspace;
  $("#runButton").disabled = !workspace || Boolean(activeRun) || uploading;
  $("#runLabel").textContent = activeRun && activeRun === workspace ? "计算中…" : workspace?.result ? "重新计算" : "运行计算";
  $("#runStatus").textContent = activeRun
    ? (activeRun === workspace ? "正在运行当前算例" : "本地引擎正在处理上一算例")
    : uploading ? "正在读取上传文件"
    : workspace?.changed ? "已有修改，重新运行可更新结果"
    : workspace?.result?.ok ? `已完成 · ${workspace.result.duration.toFixed(2)} s`
    : workspace?.result ? "运行失败，可查看计算日志后重试"
    : "可直接使用内置示例运行";
}

function collectParams() {
  const params = {};
  for (const input of $$("[data-field]")) {
    const { badInput, rangeUnderflow, rangeOverflow } = input.validity;
    if (badInput || rangeUnderflow || rangeOverflow) {
      showPane("settings");
      const dialog = input.closest("dialog");
      if (dialog && !dialog.open) dialog.showModal();
      input.focus();
      throw new Error(`请检查「${input.labels[0].textContent}」的输入范围`);
    }
    params[input.dataset.field] = input.type === "number" && input.value !== "" ? input.valueAsNumber : input.value;
  }
  return params;
}

async function fullChartTables(tables) {
  return Promise.all(tables.map(async (table) => {
    if (!table.csv_url || table.rows.length === table.total_rows) return table;
    try {
      const response = await fetch(table.csv_url, { signal: AbortSignal.timeout(8000) });
      if (!response.ok) return table;
      return parseResultCsv(await response.text(), table);
    } catch {
      return table;
    }
  }));
}

function rememberRun(algo, result, params, inputs) {
  const { groups, presentation } = result;
  const charts = groups.length
    ? groups.map(({ label, chart }) => ({ label, model: chart }))
    : presentation.comparison;
  const run = {
    id: result.run_id,
    name: algo.name,
    course: catalog.courses.find((item) => item.id === algo.course).name,
    finished: Date.now(),
    summary: result.summary || [],
    charts,
    map: presentation.map,
    inputs,
    params: algo.fields.map(({ key, label, unit }) => ({ label, unit, value: params[key] })),
  };
  caseRuns.set(run.id, run);
  updateCaseHistory();
  saveCaseRun(run).catch(() => {
    historyUnavailable = true;
    updateCaseHistory();
    toast("算例仅保存在当前页面", "浏览器本地存储不可用或空间不足。", true);
  });
}

function renderCaseSide(index) {
  const host = $(`[data-case-side="${index}"]`);
  casePlots[index]?.();
  casePlots[index] = null;
  const run = caseRuns.get(comparedRuns[index]);
  host.hidden = !run;
  if (!run) {
    host.replaceChildren();
    return;
  }
  const config = [...run.inputs, ...run.params].map(({ label, value, unit }) =>
    `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(valueText(value))}${unit ? ` ${escapeHtml(unit)}` : ""}</dd></div>`
  ).join("");
  host.innerHTML = `
    <header class="case-card-heading">
      <div><h3>${escapeHtml(run.name)}</h3><p>${escapeHtml(run.course)}</p></div>
      ${run.charts.length > 1 ? '<label>图表<select class="case-chart-select"></select></label>' : ""}
    </header>
    <div class="case-plot"></div>
    ${run.summary.length ? `<dl class="summary-grid" aria-label="关键指标">${summaryHtml(run.summary)}</dl>` : ""}
    ${config ? `<details class="case-config"><summary>输入文件与计算参数</summary><dl>${config}</dl></details>` : ""}
  `;
  const plot = $(".case-plot", host);
  if (run.map) {
    casePlots[index] = mountStationMap(plot, run.map, 0, () => {});
  } else if (run.charts.length) {
    const chartSelect = $(".case-chart-select", host);
    const draw = () => {
      casePlots[index]?.();
      casePlots[index] = mountChart(plot, run.charts[Number(chartSelect?.value || 0)].model);
    };
    if (chartSelect) {
      chartSelect.replaceChildren(...run.charts.map(({ label }, i) => new Option(label, i)));
      chartSelect.addEventListener("change", draw);
    }
    draw();
  } else {
    plot.hidden = true;
  }
}

function updateCaseHistory(opening = false) {
  $("#caseCount").textContent = caseRuns.size;
  if (!$("#caseDialog").open) return;
  const runs = [...caseRuns.values()].sort((a, b) => b.finished - a.finished);
  const message = !runs.length ? (historyLoading ? "正在读取算例记录…" : "完成两次计算后，即可在这里并排对比。")
    : runs.length === 1 ? "再运行一次计算，即可对比两次结果。" : "";
  const storageNote = historyUnavailable ? "浏览器本地存储不可用，部分记录仅保存在当前页面。" : "";
  $("#caseStatus").textContent = [message, storageNote].filter(Boolean).join(" ");
  $("#caseToolbar").hidden = !runs.length;
  for (const [index, select] of $$(".case-run-select").entries()) {
    const previous = comparedRuns[index];
    comparedRuns[index] ||= runs.find(({ id }) => id !== comparedRuns[1 - index])?.id || "";
    select.replaceChildren(
      new Option("选择运行记录", ""),
      ...runs.map((run) => new Option(`${run.name} · ${runTimestamp.format(run.finished)}`, run.id)),
    );
    select.value = comparedRuns[index];
    if (opening || previous !== comparedRuns[index]) renderCaseSide(index);
  }
}

function bindCaseHistory() {
  const dialog = $("#caseDialog");
  $("#caseButton").addEventListener("click", () => {
    dialog.showModal();
    updateCaseHistory(true);
  });
  $$(".case-run-select").forEach((select, index) => select.addEventListener("change", () => {
    comparedRuns[index] = select.value;
    renderCaseSide(index);
  }));
  dialog.addEventListener("close", () => {
    casePlots.forEach((destroy) => destroy?.());
    casePlots.fill(null);
    $$("[data-case-side]", dialog).forEach((side) => side.replaceChildren());
  });
  loadCaseHistory().then((runs) => {
    for (const run of runs) caseRuns.set(run.id, run);
  }).catch(() => { historyUnavailable = true; }).finally(() => {
    historyLoading = false;
    updateCaseHistory();
  });
}

async function runAlgorithm() {
  if ($("#runButton").disabled || !workspace) return;
  const context = workspace;
  const missing = Object.values(context.files).filter((entry) => entry.spec.required && !entry.token);
  if (missing.length) return toast("缺少输入文件", missing.map((entry) => entry.spec.label).join("、"), true);
  let params;
  try {
    params = collectParams();
  } catch (error) {
    return toast("参数需要调整", error.message, true);
  }
  const files = Object.fromEntries(Object.entries(context.files).filter(([, entry]) => entry.token)
    .map(([key, entry]) => [key, entry.token]));
  const inputs = Object.keys(files).map((key) => ({ label: context.files[key].spec.label, value: context.files[key].name }));
  const curvePreview = context.files.curve_csv?.preview;
  activeRun = context;
  context.result = null;
  context.changed = false;
  context.view = "output";
  context.outputGroup = 0;
  context.selectedWindow = 0;
  context.selectedStation = 0;
  updateRunState();
  renderChart();
  renderDetails();

  try {
    const data = await api("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm: context.algo.id, params, files }),
    });
    const chartTables = await fullChartTables(data.tables || []);
    const visuals = data.visuals || {};
    const groups = resultCharts(context.algo.id, chartTables, params, visuals);
    const presentation = buildPresentation(context.algo, chartTables, groups, visuals, params, curvePreview);
    context.result = { ...data, groups, presentation };
    if (context.view === "output" && presentation.comparison.length && !presentation.preferCombined) {
      context.view = "compare";
    }
    rememberRun(context.algo, context.result, params, inputs);
  } catch (error) {
    context.result = { ...error.payload, ok: false, error: error.message, groups: [] };
    if (workspace === context) toast("运行失败", error.message, true);
  } finally {
    activeRun = null;
    context.pane = "chart";
    updateRunState();
    if (workspace === context) {
      renderChart();
      renderDetails();
      showPane("chart", true);
    }
  }
}

function buttonAction(target, attribute, action) {
  const button = target.closest(`button[data-${attribute}]`);
  if (!button) return false;
  action(button.getAttribute(`data-${attribute}`), button);
  return true;
}

function activateChoice(target, key, action) {
  buttonAction(target, key, (value) => {
    action(value);
    $(`button[data-${key}="${value}"]`)?.focus({ preventScroll: true });
  });
}

function bindEvents() {
  $(".skip-link").addEventListener("click", (event) => {
    event.preventDefault();
    $("#workbench").focus();
  });
  $("#courseTabs").addEventListener("click", ({ target }) => activateChoice(target, "course", (id) => {
    const next = lastAlgorithmByCourse.get(id) || catalog.algorithms.find((algo) => algo.course === id)?.id;
    if (next && workspace.algo.course !== id) selectAlgorithm(next);
  }));
  $("#algoRail").addEventListener("click", ({ target }) => activateChoice(target, "algo", selectAlgorithm));
  $("#workspaceTabs").addEventListener("click", ({ target }) => buttonAction(target, "pane", (pane) => showPane(pane)));
  $("#advancedToggle").addEventListener("click", () => $("#advancedBox").showModal());
  $(".chart-actions").addEventListener("click", ({ target }) => buttonAction(target, "detail", openDetails));
  $("#detailsDialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
  $("#viewSwitch").addEventListener("click", ({ target }) => activateChoice(target, "view", (view) => {
    workspace.view = view;
    renderChart();
  }));
  $("#fileTabs").addEventListener("click", ({ target }) => activateChoice(target, "file-tab", chooseFile));
  $("#chartGroups").addEventListener("click", ({ target }) => activateChoice(target, "group", (value) => {
    if (workspace.view === "input") workspace.inputGroup = Number(value);
    else workspace.outputGroup = Number(value);
    renderChart();
  }));
  $("#resetButton").addEventListener("click", () => {
    selectAlgorithm(workspace.algo.id, true);
    toast("已恢复内置示例与默认参数");
  });
  $("#runForm").addEventListener("submit", (event) => {
    event.preventDefault();
    runAlgorithm();
  });
  document.addEventListener("input", ({ target }) => {
    if (!target.matches("[data-field]")) return;
    workspace.values[target.dataset.field] = target.value;
    markEdited();
    if (target.matches("[data-column]") && workspace.view === "input") {
      workspace.inputGroup = 0;
      renderChart();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (!event.repeat && event.key === "Enter" && (event.ctrlKey || event.metaKey) && !$("dialog[open]")) {
      event.preventDefault();
      runAlgorithm();
    }
    if (event.key === "Escape" && $("#lightbox").open) event.stopPropagation();
  });
  window.addEventListener("hashchange", () => {
    if (catalog) selectAlgorithm(location.hash.slice(1));
  });

  const list = $("#fileList");
  list.addEventListener("click", ({ target }) => {
    if (buttonAction(target, "preview-file", (key) => {
      chooseFile(key);
      showPane("chart", true);
    })) return;
    if (buttonAction(target, "upload", (key) => $(`[data-file-input="${key}"]`, list).click())) return;
    buttonAction(target, "clear", (key) => {
      const entry = workspace.files[key];
      if (entry.uploading) return;
      Object.assign(entry, { token: null, name: "", sample: false, preview: null, error: "" });
      renderFiles();
      markEdited();
      if (workspace.view === "input") renderChart();
    });
  });
  list.addEventListener("change", ({ target }) => {
    const file = target.files?.[0];
    if (file) uploadFile(target.dataset.fileInput, file);
    target.value = "";
  });
  list.addEventListener("dragover", (event) => {
    const card = event.target.closest("[data-drop]");
    if (!card || !event.dataTransfer.types.includes("Files")) return;
    event.preventDefault();
    card.classList.add("dragover");
  });
  list.addEventListener("dragleave", ({ target, relatedTarget }) => {
    const card = target.closest("[data-drop]");
    if (card && !card.contains(relatedTarget)) card.classList.remove("dragover");
  });
  list.addEventListener("drop", (event) => {
    const card = event.target.closest("[data-drop]");
    if (!card) return;
    event.preventDefault();
    card.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    if (file) uploadFile(card.dataset.drop, file);
  });
  $("#chartMount").addEventListener("click", ({ target }) => buttonAction(target, "figure", (index, button) => {
    const image = workspace.result?.images?.[index];
    if (!image) return;
    lightboxTrigger = button;
    $("#lightboxImage").src = image.url;
    $("#lightboxImage").alt = image.title;
    $("#lightboxCaption").textContent = image.title;
    $("#lightbox").showModal();
  }));
  $("#lightboxClose").addEventListener("click", () => $("#lightbox").close());
  $("#lightbox").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
  $("#lightbox").addEventListener("close", () => lightboxTrigger?.focus());
}

async function init() {
  bindEvents();
  bindCaseHistory();
  $("#runIcon").innerHTML = icon("play");
  api("/api/health").then((health) => setEngine(health.ok, health.ok ? "本地引擎在线" : "本地引擎离线"))
    .catch(() => setEngine(false, "本地引擎离线"));
  try {
    catalog = await api("/api/catalog");
    const linked = catalog.algorithms.find((algo) => algo.id === location.hash.slice(1));
    const initial = linked || catalog.algorithms[0];
    if (!initial) throw new Error("算法目录为空");
    selectAlgorithm(initial.id);
  } catch (error) {
    $("#algoTitle").textContent = "无法加载算法目录";
    $("#algoDescription").textContent = error.message;
    $("#runStatus").textContent = "请确认本地服务已启动，再刷新页面";
    emptyChart("工作台连接失败", "检查本地 Python 服务是否正在运行。");
  }
}

init();

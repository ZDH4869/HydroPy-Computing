/* ============================================================
   HydroLab · Hydrology Workbench front-end
   ============================================================ */
(() => {
  "use strict";

  const ICONS = {
    river: `<svg viewBox="0 0 24 24" fill="none"><path d="M3 8.5C6 5.8 8.2 5.8 11 8.5C13.8 11.2 16 11.2 19 8.5M3 14C6 11.3 8.2 11.3 11 14C13.8 16.7 16 16.7 19 14M3 19.5C6 16.8 8.2 16.8 11 19.5C13.8 22.2 16 22.2 19 19.5" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"/></svg>`,
    satellite: `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/><path d="M4.5 4.5L8.1 8.1M15.9 15.9L19.5 19.5M19.5 4.5L15.9 8.1M8.1 15.9L4.5 19.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="12" r="8.6" stroke="currentColor" stroke-width="1.1" stroke-dasharray="2 3" opacity=".5"/></svg>`,
    dam: `<svg viewBox="0 0 24 24" fill="none"><path d="M4 20L6 7H18L20 20H4Z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M3 12.5C6 10.5 8 14 11 12C14 10 16 13.5 21 11.5M4 17.5C7 15.5 9 18.4 12 16.5C15 14.6 17 17.5 20 15.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
    droplet: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3C8.5 8.2 5.5 12 5.5 15.5C5.5 19.1 8.4 21.5 12 21.5C15.6 21.5 18.5 19.1 18.5 15.5C18.5 12 15.5 8.2 12 3Z" stroke="currentColor" stroke-width="1.7"/><path d="M9 16C10.4 14.3 11.3 14.3 12.4 16C13.5 17.7 14.5 17.7 16 16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
    upload: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 16V4M7.5 8.5L12 4L16.5 8.5M5 15V19H19V15" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    file: `<svg viewBox="0 0 24 24" fill="none"><path d="M6 3.5H14L18.5 8V20.5H6V3.5Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M14 3.5V8H18.5" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>`,
    download: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 4V15M8 11L12 15L16 11M5 19H19" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    image: `<svg viewBox="0 0 24 24" fill="none"><rect x="3.5" y="4.5" width="17" height="15" rx="2" stroke="currentColor" stroke-width="1.5"/><circle cx="9" cy="10" r="1.4" fill="currentColor"/><path d="M4.5 17L9.5 12.5L13 15.5L16.2 13L20 17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>`,
    terminal: `<svg viewBox="0 0 24 24" fill="none"><rect x="3.5" y="4.5" width="17" height="15" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M7 9L10.5 12L7 15M12.5 15H17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`,
    table: `<svg viewBox="0 0 24 24" fill="none"><rect x="3.5" y="4.5" width="17" height="15" rx="2" stroke="currentColor" stroke-width="1.5"/><path d="M3.5 9H20.5M9 9V19.5M15 9V19.5" stroke="currentColor" stroke-width="1.3"/></svg>`,
    play: `<svg viewBox="0 0 24 24" fill="none"><path d="M8 5.5L18.5 12L8 18.5V5.5Z" fill="currentColor"/></svg>`,
    spark: `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3L13.9 9.6L20.5 11.5L13.9 13.4L12 20L10.1 13.4L3.5 11.5L10.1 9.6L12 3Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>`,
  };

  const state = {
    catalog: null,
    courses: [],
    algorithms: [],
    courseId: null,
    algoId: null,
    algo: null,
    files: {},
    activeDataTab: null,
    params: {},
    result: null,
    resultTableIndex: 0,
    running: false,
    lastDataPreview: null,
  };

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const icon = (name, cls = "") =>
    `<span class="icon ${cls}" aria-hidden="true">${ICONS[name] || ICONS.droplet}</span>`;

  function formatCell(value) {
    if (value === null || value === undefined || value === "") return "—";
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return "—";
      const abs = Math.abs(value);
      if (abs >= 1000) return value.toLocaleString("zh-CN", { maximumFractionDigits: 3 });
      if (abs > 0 && abs < 0.0001) return value.toExponential(3);
      return Number(value.toFixed(4)).toString();
    }
    return String(value);
  }

  function formatNumber(value, digits = 3) {
    const num = Number(value);
    if (!Number.isFinite(num)) return String(value ?? "—");
    return num.toLocaleString("zh-CN", { maximumFractionDigits: digits });
  }

  function toast(title, message = "", type = "success") {
    const stack = $("#toastStack");
    const node = document.createElement("div");
    node.className = `toast ${type}`;
    node.innerHTML = `<strong>${escapeHtml(title)}</strong>${message ? escapeHtml(message) : ""}`;
    stack.appendChild(node);
    window.setTimeout(() => {
      node.style.opacity = "0";
      node.style.transform = "translateY(8px)";
      window.setTimeout(() => node.remove(), 220);
    }, 3600);
  }

  async function api(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({ ok: false, error: "服务器返回了无法解析的内容。" }));
    if (!response.ok && !data.error) data.error = `HTTP ${response.status}`;
    return data;
  }

  function courseColor(courseId) {
    return state.courses.find((course) => course.id === courseId)?.color || "#22d3ee";
  }

  function setStatus(kind, text) {
    const el = $("#engineStatus");
    el.innerHTML = `<span class="status-dot ${kind}"></span><span class="status-label">${escapeHtml(text)}</span>`;
  }

  function renderCourseTabs() {
    const wrap = $("#courseTabs");
    wrap.innerHTML = state.courses
      .map((course) => {
        const active = course.id === state.courseId ? "active" : "";
        return `<button class="course-tab ${active}" data-course="${course.id}" style="--accent:${course.color}" type="button">
          <span class="tab-icon">${ICONS[course.icon] || ICONS.droplet}</span>
          <span><strong>${escapeHtml(course.name)}</strong><small>${escapeHtml(course.subtitle)}</small></span>
        </button>`;
      })
      .join("");
    $$(".course-tab", wrap).forEach((button) => {
      button.addEventListener("click", () => selectCourse(button.dataset.course));
    });
  }

  function renderAlgoRail() {
    const wrap = $("#algoRail");
    const algos = state.algorithms.filter((algo) => algo.course === state.courseId);
    const course = state.courses.find((item) => item.id === state.courseId);
    $("#algoCount").textContent = `${String(algos.length).padStart(2, "0")} 个算法`;
    $("#courseLabel").textContent = course?.name || "";
    wrap.innerHTML = algos
      .map((algo, index) => {
        const active = algo.id === state.algoId ? "active" : "";
        return `<button class="algo-card ${active}" data-algo="${algo.id}" style="--card-accent:${course?.color || "#22d3ee"}" type="button">
          <span class="algo-index">${String(index + 1).padStart(2, "0")}</span>
          <span class="algo-copy"><strong>${escapeHtml(algo.name)}</strong><small>${escapeHtml(algo.short)}</small></span>
          <span class="algo-arrow">↗</span>
        </button>`;
      })
      .join("");
    $$(".algo-card", wrap).forEach((button) => {
      button.addEventListener("click", () => selectAlgorithm(button.dataset.algo));
    });
  }

  function selectCourse(courseId) {
    if (state.courseId === courseId && state.algoId) return;
    state.courseId = courseId;
    const first = state.algorithms.find((algo) => algo.course === courseId);
    renderCourseTabs();
    if (first) selectAlgorithm(first.id, { keepCourse: true });
  }

  function selectAlgorithm(algoId, options = {}) {
    const algo = state.algorithms.find((item) => item.id === algoId);
    if (!algo) return;
    state.algoId = algoId;
    state.algo = algo;
    state.courseId = algo.course;
    state.result = null;
    state.resultTableIndex = 0;
    state.activeDataTab = null;
    state.params = {};
    state.files = {};
    state.lastDataPreview = null;

    // Reset each algorithm's file defaults from the catalog.
    (algo.files || []).forEach((spec) => {
      state.params[spec.key] = null;
      state.files[spec.key] = {
        spec,
        token: spec.sample ? `sample:${spec.sample}` : null,
        name: spec.sample || "未选择文件",
        isSample: Boolean(spec.sample),
        preview: null,
        loading: false,
      };
    });
    (algo.fields || []).forEach((field) => {
      state.params[field.key] = field.default ?? "";
    });

    renderCourseTabs();
    renderAlgoRail();
    renderParamPanel();
    renderFileList();
    renderDataPanel();
    renderResult();
    setResultState("idle", "等待运行");
    $("#resultState").innerHTML = `<span class="status-dot idle"></span><span>等待运行</span>`;

    const firstFile = (algo.files || []).find((spec) => state.files[spec.key]?.token);
    if (firstFile) {
      state.activeDataTab = firstFile.key;
      loadPreview(firstFile.key);
    } else {
      renderDataPanel();
    }

    if (!options.keepCourse) {
      $("#workbench").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function renderParamPanel() {
    const algo = state.algo;
    const course = state.courses.find((item) => item.id === algo.course);
    $("#paramTitle").textContent = algo.name;
    $("#paramIcon").innerHTML = ICONS[course?.icon] || ICONS.droplet;
    const execIcon = $("#execIcon");
    if (execIcon) execIcon.innerHTML = ICONS.terminal;
    const execScript = $("#execScript");
    if (execScript) execScript.textContent = algo.script || "workbench/engine";
    $("#algoIntro").innerHTML = `
      <h3>${escapeHtml(algo.name)}</h3>
      <p>${escapeHtml(algo.description || algo.short || "")}</p>
      <div class="tag-row">${(algo.tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}</div>
    `;
    const fields = algo.fields || [];
    const normal = fields.filter((field) => !field.advanced);
    const advanced = fields.filter((field) => field.advanced);
    $("#paramFields").innerHTML = normal.map(fieldHtml).join("");
    $("#advancedFields").innerHTML = advanced.map(fieldHtml).join("");
    $("#advancedBox").hidden = advanced.length === 0;
    bindFieldEvents($("#paramFields"));
    bindFieldEvents($("#advancedFields"));
  }

  function fieldHtml(field) {
    const value = state.params[field.key] ?? field.default ?? "";
    const help = field.help ? `<small>${escapeHtml(field.help)}</small>` : "";
    const label = `<label for="field-${field.key}"><span>${escapeHtml(field.label)}${field.unit ? ` <span class="unit">${escapeHtml(field.unit)}</span>` : ""}</span></label>`;
    let control = "";
    if (field.type === "select") {
      control = `<select id="field-${field.key}" data-param-key="${field.key}">
        ${(field.options || []).map((option) => {
          const val = typeof option === "string" ? option : option.value;
          const text = typeof option === "string" ? option : option.label;
          return `<option value="${escapeHtml(val)}" ${String(val) === String(value) ? "selected" : ""}>${escapeHtml(text)}</option>`;
        }).join("")}
      </select>`;
    } else if (field.type === "column") {
      const options = state.lastDataPreview?.columns || [];
      control = `<select id="field-${field.key}" data-param-key="${field.key}" data-column-select="true">
        <option value="${escapeHtml(value)}">${escapeHtml(value || "上传数据后选择")}</option>
      </select>`;
      if (options.length) {
        control = `<select id="field-${field.key}" data-param-key="${field.key}" data-column-select="true">
          ${options.map((col) => `<option value="${escapeHtml(col)}" ${col === value ? "selected" : ""}>${escapeHtml(col)}</option>`).join("")}
        </select>`;
      }
    } else if (field.type === "textarea") {
      control = `<textarea id="field-${field.key}" data-param-key="${field.key}" rows="3">${escapeHtml(value)}</textarea>`;
    } else {
      const type = field.type === "number" ? "number" : "text";
      const min = field.min !== undefined ? `min="${field.min}"` : "";
      const max = field.max !== undefined ? `max="${field.max}"` : "";
      const step = field.step !== undefined ? `step="${field.step}"` : "";
      control = `<input id="field-${field.key}" data-param-key="${field.key}" type="${type}" value="${escapeHtml(value)}" ${min} ${max} ${step} autocomplete="off">`;
    }
    return `<div class="field ${field.full ? "full" : ""}">${label}${control}${help}</div>`;
  }

  function bindFieldEvents(root) {
    $$("[data-param-key]", root).forEach((input) => {
      const update = () => {
        const key = input.dataset.paramKey;
        if (input.type === "number") {
          const num = Number(input.value);
          state.params[key] = Number.isFinite(num) ? num : input.value;
        } else {
          state.params[key] = input.value;
        }
      };
      input.addEventListener("input", update);
      input.addEventListener("change", update);
    });
  }

  function refreshColumnSelects(preview) {
    $$("[data-column-select]", document).forEach((select) => {
      const current = state.params[select.dataset.paramKey] ?? select.value;
      if (!preview?.columns?.length) return;
      const options = preview.columns;
      const next = options.includes(current) ? current : options[0];
      state.params[select.dataset.paramKey] = next;
      select.innerHTML = options
        .map((col) => `<option value="${escapeHtml(col)}" ${col === next ? "selected" : ""}>${escapeHtml(col)}</option>`)
        .join("");
    });
  }

  function renderFileList() {
    const wrap = $("#fileList");
    const specs = state.algo?.files || [];
    wrap.innerHTML = specs
      .map((spec) => {
        const file = state.files[spec.key] || {};
        const has = Boolean(file.token);
        const badge = file.isSample ? `<span class="file-badge">示例</span>` : "";
        const optional = !spec.required;
        const actionClear = optional && has ? `<button class="mini-button text" data-clear="${spec.key}" title="不使用该文件">清空</button>` : "";
        return `<div class="file-card ${has ? "has-file" : ""}" data-file-card="${spec.key}">
          <span class="file-icon">${ICONS.file}</span>
          <span class="file-copy">
            <strong>${escapeHtml(spec.label)}${badge}</strong>
            <small>${has ? escapeHtml(file.name) : `点击上传 ${escapeHtml(spec.accept)}${spec.required ? "" : " · 可选"}`}</small>
          </span>
          <span class="file-actions">
            ${actionClear}
            <button class="mini-button" data-upload="${spec.key}" title="上传文件">↥</button>
            <input type="file" accept="${escapeHtml(spec.accept)}" data-file-input="${spec.key}" hidden>
          </span>
        </div>`;
      })
      .join("");

    $$("[data-upload]", wrap).forEach((button) => {
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        $(`[data-file-input="${button.dataset.upload}"]`, wrap)?.click();
      });
    });
    $$("[data-file-input]", wrap).forEach((input) => {
      input.addEventListener("change", () => {
        const file = input.files?.[0];
        if (file) uploadFile(input.dataset.fileInput, file);
        input.value = "";
      });
    });
    $$("[data-clear]", wrap).forEach((button) => {
      button.addEventListener("click", () => {
        const key = button.dataset.clear;
        const entry = state.files[key];
        entry.token = null;
        entry.isSample = false;
        entry.name = "未选择文件";
        entry.preview = null;
        renderFileList();
        renderDataPanel();
      });
    });
    $$("[data-file-card]", wrap).forEach((card) => {
      const key = card.dataset.fileCard;
      card.addEventListener("dragover", (event) => {
        event.preventDefault();
        card.classList.add("dragover");
      });
      card.addEventListener("dragleave", () => card.classList.remove("dragover"));
      card.addEventListener("drop", (event) => {
        event.preventDefault();
        card.classList.remove("dragover");
        const file = event.dataTransfer?.files?.[0];
        if (file) uploadFile(key, file);
      });
    });
  }

  async function uploadFile(key, file) {
    const entry = state.files[key];
    if (!entry) return;
    const spec = entry.spec;
    if (!spec.accept.split(",").some((ext) => file.name.toLowerCase().endsWith(ext.trim().replace("*", "")))) {
      toast("文件类型不匹配", `请上传 ${spec.accept} 文件。`, "error");
      return;
    }
    const card = $(`[data-file-card="${key}"]`);
    card?.classList.add("loading");
    const form = new FormData();
    form.append("file", file);
    form.append("role", spec.role || "");
    try {
      const data = await api("/api/upload", { method: "POST", body: form });
      if (!data.ok) throw new Error(data.error || "上传失败");
      entry.token = data.token;
      entry.name = data.name || file.name;
      entry.isSample = false;
      entry.preview = data;
      state.activeDataTab = key;
      renderFileList();
      renderDataPanel();
      refreshColumnSelects(data);
      toast("文件已就绪", data.name || file.name);
    } catch (error) {
      toast("上传失败", error.message, "error");
    } finally {
      card?.classList.remove("loading");
    }
  }

  async function loadPreview(key) {
    const entry = state.files[key];
    if (!entry || !entry.token) {
      entry.preview = null;
      renderDataPanel();
      return;
    }
    if (entry.preview) {
      state.lastDataPreview = entry.preview;
      renderDataPanel();
      refreshColumnSelects(entry.preview);
      return;
    }
    entry.loading = true;
    renderDataPanel();
    try {
      const data = await api(`/api/preview?token=${encodeURIComponent(entry.token)}`);
      if (!data.ok && data.error) throw new Error(data.error);
      delete data.ok;
      entry.preview = data;
      state.lastDataPreview = data;
      refreshColumnSelects(data);
    } catch (error) {
      toast("数据预览失败", error.message, "error");
      entry.preview = null;
    } finally {
      entry.loading = false;
      renderDataPanel();
    }
  }

  function renderDataPanel() {
    const specs = state.algo?.files || [];
    const tabs = $("#dataTabs");
    tabs.innerHTML = specs
      .map((spec) => {
        const file = state.files[spec.key];
        const active = spec.key === state.activeDataTab ? "active" : "";
        return `<button class="data-tab ${active}" data-data-tab="${spec.key}" type="button">${escapeHtml(spec.label)}</button>`;
      })
      .join("");
    $$("[data-data-tab]", tabs).forEach((button) => {
      button.addEventListener("click", () => {
        state.activeDataTab = button.dataset.dataTab;
        renderDataPanel();
        const entry = state.files[state.activeDataTab];
        if (entry?.token && !entry.preview) loadPreview(state.activeDataTab);
      });
    });

    const key = state.activeDataTab || specs[0]?.key;
    const file = key ? state.files[key] : null;
    const preview = file?.preview;
    const shell = $("#dataTableShell");
    const meta = $("#dataMeta");
    $("#dataCount").textContent = preview ? `${formatNumber(preview.total_rows, 0)} 行` : "—";
    if (!file) {
      meta.innerHTML = "";
      shell.innerHTML = `<div class="empty-inline"><div class="empty-inline-icon">⇪</div><p>选择算法后会自动载入内置示例数据。</p></div>`;
      return;
    }
    meta.innerHTML = [
      `<span class="meta-chip"><strong>${escapeHtml(file.name)}</strong></span>`,
      preview ? `<span class="meta-chip">${preview.kind === "shapefile" ? "空间数据" : "CSV"} · ${formatNumber(preview.total_rows, 0)} 行</span>` : "",
      preview?.meta?.crs ? `<span class="meta-chip">CRS · ${escapeHtml(preview.meta.crs.split(":")[0] || preview.meta.crs)}</span>` : "",
      file.isSample ? `<span class="meta-chip">内置示例</span>` : "",
    ].join("");
    if (file.loading) {
      shell.innerHTML = `<div class="empty-inline"><div class="empty-inline-icon">◌</div><p>正在读取数据预览…</p></div>`;
      return;
    }
    if (!preview) {
      shell.innerHTML = `<div class="empty-inline"><div class="empty-inline-icon">⇪</div><p>这个文件还没有载入。点击左侧的上传按钮选择 ${escapeHtml(file.spec.accept)}。</p></div>`;
      return;
    }
    if (!preview.rows?.length) {
      shell.innerHTML = `<div class="empty-inline"><div class="empty-inline-icon">∅</div><p>文件没有可显示的数据行。</p></div>`;
      return;
    }
    const columns = preview.columns || [];
    const rows = preview.rows.slice(0, 100);
    shell.innerHTML = `<table class="data-table">
      <thead><tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${columns.map((_, index) => {
        const val = row[index];
        const numeric = typeof val === "number";
        return `<td class="${numeric ? "num" : ""}">${escapeHtml(formatCell(val))}</td>`;
      }).join("")}</tr>`).join("")}</tbody>
    </table>
    <div class="table-caption"><span>预览前 ${rows.length} 行</span><span>共 ${formatNumber(preview.total_rows, 0)} 行</span></div>`;
  }

  function setResultState(kind, text) {
    const el = $("#resultState");
    el.innerHTML = `<span class="status-dot ${kind}"></span><span>${escapeHtml(text)}</span>`;
  }

  function renderEmptyResult() {
    $("#resultBody").innerHTML = `<div class="result-empty">
      <div class="wave-orb">${ICONS.droplet}</div>
      <h3>准备开始计算</h3>
      <p>确认左侧参数与中间的数据，然后点击「运行计算」。结果图表、命令行输出和数据表会显示在这里。</p>
      <div class="empty-steps"><span>配置参数</span><span>检查数据</span><span>运行</span></div>
    </div>`;
  }

  function renderRunning() {
    setResultState("busy", "计算中");
    $("#resultBody").innerHTML = `<div class="running-box">
      <div class="running-waves"></div>
      <h3>正在运行 ${escapeHtml(state.algo?.name || "算法")}</h3>
      <p>本地 Python 引擎正在处理数据，请稍候…</p>
    </div>`;
  }

  function renderResult() {
    if (!state.result) {
      renderEmptyResult();
      return;
    }
    const result = state.result;
    const duration = Number(result.duration || 0);
    const ok = result.ok !== false;
    const head = `<div class="result-head">
      <div class="result-head-title"><span class="check">${ok ? "✓" : "!"}</span>${escapeHtml(state.algo?.name || "运行结果")}</div>
      <div class="result-head-meta">${ok ? `完成 · ${duration.toFixed(2)} s` : "运行失败"}</div>
    </div>`;

    if (!ok) {
      setResultState("error", "运行失败");
    } else {
      setResultState("success", `完成 · ${duration.toFixed(2)}s`);
    }

    const summary = (result.summary || []).length
      ? `<div class="summary-grid">${result.summary.map((item) => `<div class="summary-card">
          <div class="summary-label">${escapeHtml(item.label)}</div>
          <div class="summary-value">${escapeHtml(formatCell(item.value))}<span class="summary-unit">${escapeHtml(item.unit || "")}</span></div>
        </div>`).join("")}</div>`
      : "";

    const error = result.error ? `<div class="result-error"><strong>运行错误：</strong>${escapeHtml(result.error)}</div>` : "";

    const images = (result.images || []).length
      ? `<div class="result-section"><div class="result-section-title"><span><span class="dot">●</span>结果图表</span><span>${result.images.length} 张</span></div>
        <div class="image-grid">${result.images.map((image, index) => `<figure class="result-image" data-image-index="${index}">
          <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.title)}" loading="lazy">
          <figcaption>${escapeHtml(image.title)}</figcaption>
        </figure>`).join("")}</div></div>`
      : `<div class="result-section"><div class="result-section-title"><span><span class="dot">●</span>结果图表</span></div><div class="empty-inline" style="min-height:120px"><p>本次算法没有返回图片结果。</p></div></div>`;

    const consoleHtml = `<div class="result-section"><div class="result-section-title"><span>${ICONS.terminal} 命令行输出</span><span>${(result.stdout || "").split("\n").length} 行</span></div>
      <div class="terminal"><div class="terminal-bar"><span class="terminal-dot"></span><span class="terminal-dot"></span><span class="terminal-dot"></span><span class="terminal-title">hydrolab · ${escapeHtml(state.algo?.id || "run")}</span></div>
      <pre>${colorizeLog(result.stdout || "")}</pre></div></div>`;

    const tables = (result.tables || []).length
      ? `<div class="result-section"><div class="result-section-title"><span><span class="dot">●</span>数据表</span><span>${result.tables.length} 个</span></div>
        <div class="result-tabs">${result.tables.map((table, index) => `<button class="result-tab ${index === state.resultTableIndex ? "active" : ""}" data-result-table="${index}">${escapeHtml(table.title)}</button>`).join("")}</div>
        <div id="resultTableHost"></div></div>`
      : "";

    $("#resultBody").innerHTML = `${head}${error}${summary}${images}${consoleHtml}${tables}`;
    bindResultEvents();
    if (tables) renderResultTable(state.resultTableIndex);
  }

  function colorizeLog(text) {
    return escapeHtml(text)
      .split("\n")
      .map((line) => {
        let cls = "";
        if (line.includes("✓")) cls = "l-ok";
        else if (line.includes("▍")) cls = "l-section";
        else if (line.includes("警告") || line.includes("提示")) cls = "l-warn";
        else if (line.includes("✖") || line.includes("失败")) cls = "l-err";
        return `<span class="${cls}">${line || " "}</span>`;
      })
      .join("\n");
  }

  function bindResultEvents() {
    $$("[data-image-index]").forEach((figure) => {
      figure.addEventListener("click", () => {
        const image = state.result.images[Number(figure.dataset.imageIndex)];
        if (!image) return;
        $("#lightboxImage").src = image.url;
        $("#lightboxImage").alt = image.title;
        $("#lightboxCaption").textContent = image.title;
        $("#lightbox").hidden = false;
      });
    });
    $$("[data-result-table]").forEach((button) => {
      button.addEventListener("click", () => {
        state.resultTableIndex = Number(button.dataset.resultTable);
        $$("[data-result-table]").forEach((tab) => tab.classList.remove("active"));
        button.classList.add("active");
        renderResultTable(state.resultTableIndex);
      });
    });
  }

  function renderResultTable(index) {
    const table = state.result?.tables?.[index];
    const host = $("#resultTableHost");
    if (!table || !host) return;
    const columns = table.columns || [];
    const rows = table.rows || [];
    const download = table.csv_url
      ? `<a class="table-download" href="${escapeHtml(table.csv_url)}" download>${ICONS.download} CSV</a>`
      : "";
    host.innerHTML = `${download ? `<div style="display:flex;justify-content:flex-end;margin-bottom:5px">${download}</div>` : ""}
      <div class="result-table-shell"><table class="data-table">
      <thead><tr>${columns.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr>${columns.map((_, cellIndex) => {
        const val = row[cellIndex];
        const numeric = typeof val === "number";
        return `<td class="${numeric ? "num" : ""}">${escapeHtml(formatCell(val))}</td>`;
      }).join("")}</tr>`).join("")}</tbody>
    </table></div>
    <div class="table-caption"><span>预览 ${rows.length} 行</span><span>共 ${formatNumber(table.total_rows, 0)} 行</span></div>`;
  }

  function collectParams() {
    const params = {};
    $$("[data-param-key]").forEach((input) => {
      const key = input.dataset.paramKey;
      if (input.type === "number") {
        const value = Number(input.value);
        params[key] = Number.isFinite(value) ? value : input.value;
      } else {
        params[key] = input.value;
      }
    });
    return params;
  }

  function collectFiles() {
    const files = {};
    (state.algo?.files || []).forEach((spec) => {
      const entry = state.files[spec.key];
      if (entry?.token) files[spec.key] = entry.token;
    });
    return files;
  }

  async function runAlgorithm() {
    if (state.running || !state.algo) return;
    const missing = (state.algo.files || []).filter((spec) => spec.required && !state.files[spec.key]?.token);
    if (missing.length) {
      toast("还缺少数据文件", missing.map((spec) => spec.label).join("、"), "error");
      return;
    }
    state.running = true;
    const button = $("#runButton");
    button.classList.add("running");
    button.disabled = true;
    renderRunning();
    const started = performance.now();
    try {
      const data = await api("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithm: state.algo.id,
          params: collectParams(),
          files: collectFiles(),
        }),
      });
      state.result = data;
      state.resultTableIndex = 0;
      renderResult();
      if (data.ok) {
        toast("计算完成", `${data.duration?.toFixed?.(2) ?? ""} 秒 · 结果已生成`);
      } else {
        toast("计算失败", data.error || "请查看命令行输出", "error");
      }
    } catch (error) {
      state.result = { ok: false, error: error.message, stdout: `网络或服务器错误：${error.message}`, summary: [], tables: [], images: [] };
      renderResult();
      toast("连接失败", error.message, "error");
    } finally {
      state.running = false;
      button.classList.remove("running");
      button.disabled = false;
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      if (state.result?.ok) {
        setResultState("success", `完成 · ${elapsed}s`);
      }
    }
  }

  function bindGlobalEvents() {
    $("#runButton").addEventListener("click", runAlgorithm);
    document.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runAlgorithm();
      }
      if (event.key === "Escape" && !$("#lightbox").hidden) {
        $("#lightbox").hidden = true;
      }
    });
    $("#lightboxClose").addEventListener("click", () => ($("#lightbox").hidden = true));
    $("#lightbox").addEventListener("click", (event) => {
      if (event.target === $("#lightbox")) $("#lightbox").hidden = true;
    });
  }

  async function init() {
    bindGlobalEvents();
    try {
      const health = await api("/api/health");
      if (health.ok) setStatus("", "ENGINE READY");
      else setStatus("error", "OFFLINE");
    } catch {
      setStatus("error", "OFFLINE");
    }
    try {
      const data = await api("/api/catalog");
      state.catalog = data;
      state.courses = data.courses || [];
      state.algorithms = data.algorithms || [];
      if (!state.courses.length || !state.algorithms.length) throw new Error("目录为空");
      state.courseId = state.courses[0].id;
      state.algoId = null;
      renderCourseTabs();
      renderAlgoRail();
      selectAlgorithm(state.algorithms.find((algo) => algo.course === state.courseId)?.id, { keepCourse: true });
    } catch (error) {
      setStatus("error", "ENGINE OFFLINE");
      $("#algoRail").innerHTML = `<div class="empty-inline" style="grid-column:1/-1;min-height:100px"><p>无法读取算法目录：${escapeHtml(error.message)}</p></div>`;
      renderEmptyResult();
      toast("无法加载工作台", error.message, "error");
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();

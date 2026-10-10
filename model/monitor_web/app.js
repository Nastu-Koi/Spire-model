"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  runs: [], selected: new URLSearchParams(location.search).get("run"), data: null, view: null,
  kind: "all", tab: "rounds", scope: "all", metric: "", metricOptions: "", expanded: new Set(),
  timer: null, controller: null, request: 0, listText: "", root: "", clock: 0, drawn: "",
};
const STATES = { starting: "正在启动", updating: "训练更新中", running: "运行中", completed: "已完成", failed: "训练失败", interrupted: "已中断" };
const ACTIVE = new Set(["starting", "updating", "running"]);
const METRICS = {
  loss: "训练损失", accuracy: "动作正确率", entropy: "策略熵",
  last_grad_norm: "梯度范数（裁剪前）", gradient_clip_fraction: "梯度裁剪比例",
  samples: "每次更新的宏动作数", weight: "逻辑批权重", loss_scale: "损失缩放",
  validation_nll: "混合验证 NLL / 分支", a0_validation_nll: "A0 验证 NLL / 分支", duration_seconds: "耗时（秒）",
};
// Offered first, in this order; any other metric a log carries follows them.
const CHOICES = ["entropy", "last_grad_norm", "gradient_clip_fraction"];
const CHARTED = ["loss", "accuracy", "duration_seconds"];
const COUNTERS = new Set(["updates", "accuracy_correct", "accuracy_decisions"]);
const SMOOTH_STEPS = [1000, 500, 200, 100, 50, 20, 10];
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const count = (value) => isNumber(value) ? value : 0;
const fmt = (value, digits = 4) => !isNumber(value) ? "—" : value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
const pct = (value, digits = 1) => !isNumber(value) ? "—" : `${fmt(value * 100, digits)}%`;
const percentMetric = (key) => key.endsWith("rate") || key.endsWith("fraction") || key === "accuracy";
const metricLabel = (key) => METRICS[key] || (key.startsWith("learning_rates.") ? "学习率 · " + key.slice(15) : key.startsWith("grad_norm.") ? "梯度范数 · " + key.slice(10) : key);
const metricText = (key, value) => percentMetric(key) ? pct(value) : !isNumber(value) ? "—" : value !== 0 && Math.abs(value) < 1e-3 ? value.toExponential(2) : fmt(value);
const runLabel = (run) => run.id.includes("/") ? run.id.split("/").slice(-2).join(" / ") : run.name;
function duration(value) {
  if (!isNumber(value)) return "—";
  if (value < 60) return `${fmt(value, value < 10 ? 2 : 1)} 秒`;
  if (value < 3600) return `${fmt(value / 60, 1)} 分钟`;
  return `${fmt(value / 3600, value < 36000 ? 1 : 0)} 小时`;
}
function date(value) {
  const parsed = isNumber(value) ? new Date(value * 1000) : value ? new Date(value) : null;
  return !parsed || Number.isNaN(parsed.getTime()) ? "未记录时间" : parsed.toLocaleString("zh-CN", { hour12: false });
}
function ago(seconds) {
  if (!isNumber(seconds)) return "";
  if (seconds < 5) return "刚刚";
  if (seconds < 90) return `${Math.round(seconds)} 秒前`;
  if (seconds < 5400) return `${Math.round(seconds / 60)} 分钟前`;
  return seconds < 172800 ? `${fmt(seconds / 3600, 1)} 小时前` : `${Math.round(seconds / 86400)} 天前`;
}
// Seconds since the trainer wrote something, on the monitor's clock rather than the browser's.
function age(value) {
  const written = Date.parse(value);
  return Number.isNaN(written) ? null : Math.max(0, (Date.now() + state.clock - written) / 1000);
}
function el(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== null) node.textContent = String(text);
  return node;
}
function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== null) node.textContent = String(text);
  return node;
}
function setError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function remember(id) {
  try { localStorage.setItem("spire-monitor:" + id, $(id).value); } catch { /* a private window keeps nothing */ }
}
function recall() {
  for (const id of ["range", "smooth", "scale"]) {
    try {
      const value = localStorage.getItem("spire-monitor:" + id);
      if (value !== null && Array.from($(id).options).some((option) => option.value === value)) $(id).value = value;
    } catch { /* defaults stand */ }
  }
}

// ---- Runs ----
function filteredRuns() {
  const query = $("search").value.trim().toLowerCase();
  return state.runs.filter((run) => (state.kind === "all" || run.kind === state.kind) && `${run.name} ${run.id}`.toLowerCase().includes(query));
}
function renderRuns() {
  const runs = filteredRuns();
  const signature = JSON.stringify([runs.map((run) => [run.id, run.kind, run.status.state, run.status.updates, run.status.round]), state.selected]);
  $("count-all").textContent = state.runs.length;
  $("count-bootstrap").textContent = state.runs.filter((run) => run.kind === "bootstrap").length;
  if (signature === state.listText) return;
  state.listText = signature;
  $("run-list").replaceChildren();
  if (!runs.length) $("run-list").append(el("p", "sidebar-empty", state.runs.length ? "没有匹配的训练记录" : "尚未发现训练记录"));
  for (const run of runs) {
    const button = el("button", "run-button" + (state.selected === run.id ? " selected" : ""));
    button.type = "button";
    button.setAttribute("aria-current", state.selected === run.id ? "true" : "false");
    button.title = run.path;
    const title = el("strong");
    title.append(el("i", "run-dot " + (run.status.state === "failed" ? "failed" : ACTIVE.has(run.status.state) ? "active" : "idle")), runLabel(run));
    const meta = el("span", "run-meta");
    meta.append(el("span", "", "Bootstrap"), el("span", "", STATES[run.status.state] || "状态未知"));
    if (isNumber(run.status.updates)) meta.append(el("span", "", "Update " + fmt(run.status.updates, 0)));
    button.append(title, ...(run.id === "." || run.id === run.name ? [] : [el("small", "", run.id)]), meta);
    button.addEventListener("click", () => {
      if (state.selected === run.id) return;
      state.selected = run.id;
      state.data = null;
      state.expanded.clear();
      $("dashboard").hidden = true;
      renderRuns();
      refresh();
    });
    $("run-list").append(button);
  }
}

// ---- Data: the curve arrives as columns, and only the part not yet held ----
function extend(columns, incoming, held) {
  for (const [name, values] of Object.entries(incoming)) {
    if (!columns[name]) columns[name] = held ? new Array(held).fill(null) : values;
    if (columns[name] !== values) for (const value of values) columns[name].push(value);
  }
}
function merge(payload) {
  const { series, recent, summaries, ...run } = payload;
  let data = state.data;
  if (!data || data.id !== run.id || data.token !== run.token || series.start !== data.base + data.n) {
    data = state.data = { id: run.id, token: run.token, base: series.start, n: 0, meta: {}, metrics: {}, recent: [], summaries: [], summaryBase: summaries.start, active: new Set(), version: 0 };
    state.expanded.clear();
    state.drawn = "";
    state.view = null;
  }
  if (summaries.start !== data.summaryBase + data.summaries.length) { data.summaries = []; data.summaryBase = summaries.start; }
  const added = series.meta.update.length;
  extend(data.meta, series.meta, data.n);
  extend(data.metrics, series.metrics, data.n);
  for (const [name, values] of Object.entries(series.metrics)) if (!data.active.has(name) && values.some((value) => value !== null && value !== 0)) data.active.add(name);
  data.n += added;
  for (const column of [...Object.values(data.meta), ...Object.values(data.metrics)]) while (column.length < data.n) column.push(null);
  // The service may keep fewer points than it has read (--limit); hold what it holds.
  const excess = data.n - run.retained_records;
  if (excess > 0) {
    for (const column of [...Object.values(data.meta), ...Object.values(data.metrics)]) column.splice(0, excess);
    data.n -= excess;
    data.base += excess;
  }
  data.labels = series.labels;
  data.recent = data.recent.concat(recent).slice(-200);
  data.summaries.push(...summaries.records);
  if (added || summaries.records.length || !data.run) data.version += 1;
  data.run = run;
  return data;
}
// Points inside the chosen update range, and the stretches a line may join:
// never across a session, a stage, or an update count that went back.
function currentView() {
  const data = state.data, span = Number($("range").value), key = `${data.id}:${data.token}:${data.version}:${span}`;
  if (state.view?.key === key) return state.view;
  const { update, stage, session } = data.meta;
  let end = null;
  for (let i = data.n - 1; i >= 0 && end === null; i--) end = update[i];
  const rows = [], plot = [];
  for (let i = 0; i < data.n; i++) {
    const step = update[i];
    if (span && (step === null || end === null || step <= end - span || step > end)) continue;
    rows.push(i);
    if (step !== null) plot.push(i);
  }
  const segments = [];
  for (let start = 0, p = 1; p <= plot.length; p++) {
    if (p < plot.length && stage[plot[p]] === stage[plot[p - 1]] && session[plot[p]] === session[plot[p - 1]] && update[plot[p]] > update[plot[p - 1]]) continue;
    segments.push([start, p]);
    start = p;
  }
  state.view = { key, rows, plot, segments, x: plot.map((i) => update[i]), end, span };
  return state.view;
}
function smoothing(points) {
  const choice = $("smooth").value;
  if (choice !== "auto") return Number(choice);
  return SMOOTH_STEPS.find((step) => step <= points / 150) || 0;
}
// The newest points of the stretch still being written, for "recent average" figures.
function tail(limit) {
  const view = currentView(), last = view.segments.at(-1);
  if (!last) return [];
  return view.plot.slice(Math.max(last[0], last[1] - limit), last[1]);
}
function average(name, indices) {
  const column = state.data.metrics[name];
  let sum = 0, n = 0;
  if (column) for (const i of indices) if (column[i] !== null) { sum += column[i]; n += 1; }
  return n ? sum / n : null;
}

// ---- Scales ----
function niceStep(span, parts) {
  const raw = span / Math.max(1, parts), power = Math.pow(10, Math.floor(Math.log10(raw))), unit = raw / power;
  return (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 5 ? 5 : 10) * power;
}
function ticks(low, high, parts, whole = false) {
  if (!(high > low)) return { step: 1, values: [low] };
  let step = niceStep(high - low, parts);
  if (whole) step = Math.max(1, Math.round(step));
  const values = [];
  for (let k = Math.ceil(low / step - 1e-9); k * step <= high + step * 1e-9; k++) values.push(k * step + 0);
  return { step, values };
}
function tickText(value, step, percent) {
  if (percent) return (value * 100).toFixed(step >= 0.01 ? 0 : step >= 0.001 ? 1 : 2) + "%";
  const size = Math.max(Math.abs(value), step);
  if (value !== 0 && (size >= 1e9 || size < 1e-3)) return value.toExponential(1);
  if (size >= 1e6 && step >= 1e4) return fmt(value / 1e6, 2) + "M";
  if (size >= 1e4 && step >= 100) return fmt(value / 1e3, 1) + "k";
  const digits = Math.min(6, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9)));
  return value.toLocaleString("zh-CN", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
// A handful of far-off points would flatten everything else; in focus mode the axis
// covers the bulk and whatever must stay visible, and says what it left out.
function yDomain(raw, keep, { percent, focus }) {
  let low = Infinity, high = -Infinity;
  for (const value of raw) { if (value < low) low = value; if (value > high) high = value; }
  const min = low, max = high;
  let clipped = false;
  if (focus && raw.length >= 100) {
    const sorted = Float64Array.from(raw).sort();
    const lo = Math.min(sorted[Math.floor((sorted.length - 1) * 0.005)], ...keep);
    const hi = Math.max(sorted[Math.ceil((sorted.length - 1) * 0.995)], ...keep);
    if (hi > lo && high - low > 1.6 * (hi - lo)) { low = lo; high = hi; clipped = true; }
  }
  for (const value of keep) { if (value < low) low = value; if (value > high) high = value; }
  const span = high - low, pad = span > 0 ? span * 0.08 : Math.abs(high) * 0.1 || 0.01;
  let bottom = low - pad, top = high + pad;
  // Start at zero when zero is near: it then costs at most a third of the plot.
  if (low >= 0 && (bottom < 0 || low <= span / 2)) bottom = 0;
  if (percent) { bottom = Math.max(0, bottom); top = Math.min(1, top); if (top <= bottom) top = Math.min(1, bottom + 0.01); }
  return { low: bottom, high: top, clipped, min, max };
}
// Trailing mean of the last `size` points of a stretch. Its first points average
// only a few values; `settled` is the range of those that average at least half a window.
function movingAverage(values, segments, size) {
  const result = new Array(values.length).fill(null), settled = [], early = [];
  for (const [a, b] of segments) {
    const seen = [];
    let sum = 0, head = 0;
    for (let p = a; p < b; p++) {
      if (values[p] === null) continue;
      seen.push(values[p]);
      sum += values[p];
      if (seen.length - head > size) sum -= seen[head++];
      result[p] = sum / (seen.length - head);
      (seen.length - head >= size / 2 ? settled : early).push(result[p]);
    }
  }
  const kept = settled.length ? settled : early;
  return { values: result, range: kept.length ? [kept.reduce((x, y) => Math.min(x, y)), kept.reduce((x, y) => Math.max(x, y))] : [] };
}
// Every point is drawn, but a pixel column needs only its first, lowest, highest
// and last to look exactly as if they all were.
function tracePath(segments, xs, ys) {
  let path = "";
  for (const [a, b] of segments) {
    let pen = false, column = null, first = 0, last = 0, low = 0, high = 0;
    const flush = () => {
      if (column === null) return;
      for (const p of [...new Set([first, low, high, last])].sort((i, j) => i - j)) {
        path += `${pen ? "L" : "M"}${xs[p].toFixed(1)},${ys[p].toFixed(1)}`;
        pen = true;
      }
      column = null;
    };
    for (let p = a; p < b; p++) {
      if (Number.isNaN(ys[p])) { flush(); pen = false; continue; }
      const here = Math.floor(xs[p]);
      if (here !== column) { flush(); column = here; first = last = low = high = p; continue; }
      last = p;
      if (ys[p] > ys[low]) low = p;
      if (ys[p] < ys[high]) high = p;
    }
    flush();
  }
  return path;
}

// ---- Line chart ----
function legend(id, items, note = "") {
  $(id).replaceChildren(...items.map(({ cls, label, shape = "line" }) => {
    const item = el("span", "legend-item");
    item.append(el("i", `key ${shape} ${cls}`), label);
    return item;
  }));
  if (note) $(id).append(el("span", "legend-note", note));
}
function tooltipRow(cls, value, label, shape = "line") {
  const row = el("div", "tip-row");
  row.append(el("i", `key ${shape} ${cls}`), el("strong", "", value), el("span", "", label));
  return row;
}
function placeTooltip(host, tooltip, x) {
  tooltip.hidden = false;
  const room = host.clientWidth - tooltip.offsetWidth;
  tooltip.style.left = `${Math.max(0, Math.min(room, x + 12 + tooltip.offsetWidth > host.clientWidth ? x - 12 - tooltip.offsetWidth : x + 12))}px`;
  tooltip.style.top = "6px";
}
function lineChart(id, spec) {
  const host = $(id), n = spec.x.length;
  host.replaceChildren();
  if (!n || !spec.series.some((line) => line.values.some((value) => value !== null))) {
    host.append(el("div", "chart-empty", spec.empty || "尚无此项指标"));
    return "";
  }
  const marks = spec.marks || [];
  const W = Math.max(260, host.clientWidth), H = Math.max(150, host.clientHeight);
  const left = 50, right = 14, top = marks.length ? 20 : 12, bottom = 30, width = W - left - right, height = H - top - bottom;
  const size = spec.smooth >= 2 ? spec.smooth : 0;
  // Several lines at once show only their averages; the points stay in the tooltip.
  const traced = !size || spec.series.length === 1;
  const raw = [], keep = [];
  const lines = spec.series.map((line) => {
    const average = size ? movingAverage(line.values, spec.segments, size) : null;
    if (traced) for (const value of line.values) if (value !== null) raw.push(value);
    if (average) { keep.push(...average.range); if (!traced) for (const value of average.values) if (value !== null) raw.push(value); }
    return { ...line, smoothed: average?.values || null };
  });
  const range = yDomain(raw, keep, spec);
  let first = Infinity, last = -Infinity;
  for (const value of spec.x) { if (value < first) first = value; if (value > last) last = value; }
  const px = (value) => left + (first === last ? width / 2 : (value - first) / (last - first) * width);
  const py = (value) => top + (range.high - value) / (range.high - range.low) * height;
  const xs = Float64Array.from(spec.x, px);
  const project = (values) => Float64Array.from(values, (value) => value === null ? NaN : py(value));
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", tabindex: "0", "aria-label": `${spec.title}；横轴为累计更新次数，${n} 个点，方向键逐点查看，PageUp／PageDown 跳跃` });
  svg.append(svgEl("title", {}, spec.title));
  const clip = svgEl("clipPath", { id: "clip-" + id });
  clip.append(svgEl("rect", { x: left, y: top - 2, width, height: height + 4 }));
  svg.append(clip);
  const yTicks = ticks(range.low, range.high, Math.max(2, Math.round(height / 42)));
  for (const value of yTicks.values) {
    svg.append(svgEl("line", { x1: left, y1: py(value), x2: W - right, y2: py(value), class: "gridline" }));
    svg.append(svgEl("text", { x: left - 8, y: py(value) + 4, "text-anchor": "end" }, tickText(value, yTicks.step, spec.percent)));
  }
  const xTicks = first === last ? { step: 1, values: [first] } : ticks(first, last, Math.max(3, Math.round(width / 70)), true);
  for (const value of xTicks.values) svg.append(svgEl("text", { x: px(value), y: H - 10, "text-anchor": px(value) > W - right - 24 ? "end" : "middle" }, tickText(value, xTicks.step, false)));
  // Round boundaries: a hairline, named above the plot where the neighbours leave room.
  let labelled = -Infinity;
  for (const mark of marks) {
    if (mark.x <= first || mark.x >= last) continue;
    svg.append(svgEl("line", { x1: px(mark.x), x2: px(mark.x), y1: top - 6, y2: top + height, class: "boundary" }));
    if (px(mark.x) - labelled < 56 || px(mark.x) > W - right - 44) continue;
    svg.append(svgEl("text", { x: px(mark.x) + 4, y: top - 8, class: "boundary-label" }, mark.label));
    labelled = px(mark.x);
  }
  const plot = svgEl("g", { "clip-path": `url(#clip-${id})` });
  svg.append(plot);
  for (const line of lines) {
    line.ys = project(line.values);
    // Behind an average, many points per pixel merge into a band: keep it a wash.
    if (traced) plot.append(svgEl("path", { d: tracePath(spec.segments, xs, line.ys), class: `series-line ${line.cls}${line.smoothed ? (n > width * 4 ? " raw dense" : " raw") : ""}` }));
  }
  for (const line of lines) {
    line.front = line.smoothed ? project(line.smoothed) : line.ys;
    if (line.smoothed) plot.append(svgEl("path", { d: tracePath(spec.segments, xs, line.front), class: `series-line ${line.cls}` }));
    // A lone point has no line to show it; a short series shows every point.
    for (let p = 0; p < n; p++) {
      if (Number.isNaN(line.front[p])) continue;
      const lone = (p === 0 || Number.isNaN(line.front[p - 1])) && (p === n - 1 || Number.isNaN(line.front[p + 1]));
      if (n <= 40 || lone) plot.append(svgEl("circle", { cx: xs[p], cy: line.front[p], r: 3, class: `series-dot ${line.cls}` }));
    }
    let end = n - 1;
    while (end >= 0 && Number.isNaN(line.front[end])) end -= 1;
    if (end >= 0) svg.append(svgEl("circle", { cx: xs[end], cy: Math.max(top, Math.min(top + height, line.front[end])), r: 4, class: `series-dot end ${line.cls}` }));
  }
  const beyond = [];
  if (range.clipped && range.max > range.high) beyond.push("最高 " + (spec.percent ? pct(range.max) : fmt(range.max, 3)));
  if (range.clipped && range.min < range.low) beyond.push("最低 " + (spec.percent ? pct(range.min) : fmt(range.min, 3)));
  const focus = svgEl("line", { y1: top, y2: top + height, class: "focus-line", visibility: "hidden" });
  const dots = lines.map((line) => svgEl("circle", { r: 4, class: `series-dot end ${line.cls}`, visibility: "hidden" }));
  svg.append(focus, ...dots);
  const tooltip = el("div", "chart-tooltip");
  tooltip.hidden = true;
  tooltip.setAttribute("role", "status");
  host.append(svg, tooltip);
  let active = n - 1;
  function nearest(value) {
    let best = 0, gap = Infinity;
    for (const [a, b] of spec.segments) {
      let lo = a, hi = b - 1;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (spec.x[mid] < value) lo = mid + 1; else hi = mid; }
      for (const p of [lo - 1, lo]) if (p >= a && p < b && Math.abs(spec.x[p] - value) <= gap) { gap = Math.abs(spec.x[p] - value); best = p; }
    }
    return best;
  }
  function inspect(position) {
    active = Math.max(0, Math.min(n - 1, position));
    tooltip.replaceChildren(el("div", "tip-title", spec.heading(active)));
    lines.forEach((line, index) => {
      const value = line.values[active], shown = line.front[active];
      const text = (number) => spec.percent ? pct(number, 2) : metricText(line.key || "", number);
      if (!line.smoothed) tooltip.append(tooltipRow(line.cls, text(value), line.label));
      else if (traced) tooltip.append(tooltipRow(line.cls, text(line.smoothed[active]), `${line.label} · ${size} 点平均`), tooltipRow(line.cls + " raw", text(value), "该点原值"));
      else tooltip.append(tooltipRow(line.cls, text(line.smoothed[active]), `${line.label} · 该点 ${text(value)}`));
      dots[index].setAttribute("visibility", Number.isNaN(shown) ? "hidden" : "visible");
      if (!Number.isNaN(shown)) { dots[index].setAttribute("cx", xs[active]); dots[index].setAttribute("cy", Math.max(top, Math.min(top + height, shown))); }
    });
    for (const note of spec.detail ? spec.detail(active) : []) tooltip.append(el("div", "tip-note", note));
    placeTooltip(host, tooltip, xs[active] / W * host.clientWidth);
    focus.setAttribute("x1", xs[active]);
    focus.setAttribute("x2", xs[active]);
    focus.setAttribute("visibility", "visible");
  }
  const hide = () => { tooltip.hidden = true; focus.setAttribute("visibility", "hidden"); for (const dot of dots) dot.setAttribute("visibility", "hidden"); };
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    inspect(nearest(first + ((event.clientX - box.left) / box.width * W - left) / width * (last - first)));
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => inspect(active));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    const jump = Math.max(1, Math.round(n / 20));
    const move = { ArrowLeft: active - 1, ArrowRight: active + 1, PageUp: active - jump, PageDown: active + jump, Home: 0, End: n - 1 }[event.key];
    if (move === undefined) return;
    event.preventDefault();
    inspect(move);
  });
  // What focus mode left outside the axis, for the caller to state under the chart.
  return beyond.length ? `纵轴聚焦主体，范围外：${beyond.join("、")}` : "";
}

// ---- Status and headline figures ----
function renderStatus(run) {
  const status = run.status, data = state.data, latest = run.latest;
  $("state").textContent = STATES[status.state] || "状态未知";
  $("state").className = "state-tag " + (status.state === "failed" ? "failed" : status.state === "interrupted" ? "interrupted" : ACTIVE.has(status.state) ? "active" : "");
  const parts = [];
  const updates = isNumber(status.updates) ? status.updates : latest?.update ?? null;
  if (updates !== null) parts.push(`Update ${fmt(updates, 0)}`);
  if (isNumber(status.round)) parts.push(`第 ${status.round} 轮`);
  const written = age(run.updated_at);
  // An estimate only while the trainer is still writing.
  const pace = ACTIVE.has(status.state) && written !== null && written <= 900 ? roundPace(updates) : null;
  if (pace) parts.push(`本轮约 ${pct(pace.done, 0)}${pace.remaining !== null ? "，预计还需 " + duration(pace.remaining) : ""}（按上一轮 ${fmt(pace.length, 0)} 次更新估算）`);
  $("status-detail").textContent = parts.join(" · ");
  $("updated").textContent = run.updated_at ? `状态写入 ${ago(written)} · ${date(run.updated_at)}` : "未记录状态时间";
  const warnings = [...run.warnings];
  if (status.error) warnings.push("训练错误：" + String(status.error));
  if (ACTIVE.has(status.state) && written !== null && written > 900) warnings.push(`状态已 ${ago(written).replace("前", "")}没有更新。状态文件只反映最后一次写入，训练进程可能已经停止。`);
  if (run.total_records > run.retained_records) warnings.push(`当前保留最近 ${run.retained_records} / ${run.total_records} 条记录；用 --limit 0 重启监测服务可读取全部历史。`);
  if (!latest) warnings.push("等待首次优化器更新的指标；旧训练进程需在下次启动时加载逐更新记录功能。");
  else if (run.source !== "updates") warnings.push("旧日志只有汇总指标，已按真实累计更新次数定位；未记录的中间更新无法补回。");
  if (latest && !isNumber(latest.metrics.accuracy)) warnings.push("该记录未包含动作正确率。训练进程加载新代码后才会写入，旧记录无法由损失反推。");
  if (latest?.session && status.session && latest.session !== status.session) warnings.push("最近指标来自上一次训练会话，正在等待本次会话的更新记录。");
  $("warnings").replaceChildren(...warnings.map((message) => el("div", "notice", message)));
  $("record-count").textContent = `${fmt(run.total_records, 0)} 条${run.source === "updates" ? "更新" : "汇总"}记录 · ${fmt(data.summaries.length, 0)} 轮汇总`;
}
// Wall-clock seconds per update over the newest stretch, loading and logging included.
function wallPace() {
  const points = tail(200), time = state.data.meta.time;
  const from = points.find((i) => time[i] !== null), to = points.findLast((i) => time[i] !== null);
  return from !== undefined && to > from && time[to] > time[from] ? (time[to] - time[from]) / (points.indexOf(to) - points.indexOf(from)) : null;
}
// How far the current epoch is, taking the previous one as its length.
function roundPace(updates) {
  const ends = state.data.summaries.map((record) => record.update).filter(isNumber);
  const length = ends.length >= 2 ? ends.at(-1) - ends.at(-2) : 0;
  if (!(length > 0) || !isNumber(updates) || updates < ends.at(-1) || updates - ends.at(-1) > length) return null;
  const pace = wallPace();
  return { length, done: (updates - ends.at(-1)) / length, remaining: pace === null ? null : (length - (updates - ends.at(-1))) * pace };
}
function renderSummary(run) {
  const data = state.data, last = run.latest, m = last?.metrics || {}, live = run.source === "updates";
  const recent = tail(200), span = recent.length;
  const updates = ["累计更新", fmt(last?.update, 0), `${isNumber(last?.round) ? "第 " + last.round + " 轮 · " : ""}策略版本 ${fmt(last?.policy_version, 0)}${live ? " · 每次 " + fmt(m.samples, 0) + " 个宏动作" : ""}`];
  const pace = live ? wallPace() : null;
  const speed = ["更新速度", pace === null ? "—" : `${fmt(pace, 2)} 秒/次`, pace === null ? "需要同一会话内的连续更新" : `近 ${span} 次墙钟均值 · 约 ${fmt(3600 / pace, 0)} 次/小时`];
  let cards;
  if (live) {
    const hits = recent.reduce((sum, i) => sum + count(data.metrics.accuracy_correct?.[i]), 0), decisions = recent.reduce((sum, i) => sum + count(data.metrics.accuracy_decisions?.[i]), 0);
    cards = [
      ["训练损失", fmt(average("loss", recent)), `近 ${span} 次更新均值 · 本次 ${fmt(m.loss)}`],
      ["动作正确率", decisions ? pct(hits / decisions) : pct(m.accuracy), isNumber(m.accuracy) ? `近 ${span} 次 Top-1 · 本次 ${pct(m.accuracy)}（${fmt(m.accuracy_correct, 0)} / ${fmt(m.accuracy_decisions, 0)}）` : "该记录未统计正确率"],
      ["梯度范数", fmt(average("last_grad_norm", recent), 3), `裁剪前 · 近 ${span} 次均值 · 本次 ${fmt(m.last_grad_norm, 3)}`],
      speed, updates,
    ];
  } else {
    cards = [
      ["训练损失", fmt(m.loss), "旧日志汇总值 · 加权宏动作 NLL"],
      ["动作正确率", pct(m.accuracy), isNumber(m.accuracy) ? `训练 Top-1 · 命中 ${fmt(m.accuracy_correct, 0)} / ${fmt(m.accuracy_decisions, 0)}` : "该记录未统计正确率"],
      ["A0 验证 NLL", fmt(m.a0_validation_nll), "仅 A0 留出样本 · 每分支"],
      updates,
    ];
  }
  $("summary").replaceChildren(...cards.map(([title, value, note]) => {
    const card = el("article", "metric-card");
    card.append(el("div", "metric-top", title), el("div", "metric-value", value), el("div", "metric-foot", note));
    return card;
  }));
}

// ---- Learning curves ----
function renderCurves() {
  const data = state.data, run = data.run, view = currentView(), live = run.source === "updates";
  const column = (name) => { const values = data.metrics[name]; return view.plot.map((i) => values ? values[i] : null); };
  const { round, time, session } = data.meta;
  const smooth = smoothing(view.plot.length);
  const marks = [];
  for (let p = 1; p < view.plot.length; p++) if (round[view.plot[p]] !== null && round[view.plot[p]] !== round[view.plot[p - 1]] && view.x[p] > view.x[p - 1]) marks.push({ x: view.x[p], label: `第 ${round[view.plot[p]]} 轮` });
  const heading = (p) => `Update ${fmt(view.x[p], 0)}${round[view.plot[p]] !== null ? ` · 第 ${round[view.plot[p]]} 轮` : ""}`;
  const detail = (p) => {
    const i = view.plot[p], id = data.labels.session[session[i]];
    return [time[i] !== null ? date(time[i]) : null, id ? `Session ${String(id).slice(0, 8)}` : null].filter(Boolean);
  };
  const base = { x: view.x, segments: view.segments, smooth, focus: $("scale").value === "focus", heading, detail, marks: marks.length <= 40 ? marks : [] };
  const chart = (id, key, cls, extra = {}) => lineChart(id, { ...base, title: metricLabel(key), percent: percentMetric(key), series: [{ key, label: metricLabel(key), cls, values: column(key) }], ...extra });
  const smoothNote = smooth >= 2 ? `粗线为 ${smooth} 点滑动平均，细线为每个点的原值` : "每个点的原值";
  $("curve-note").textContent = `${fmt(view.plot.length, 0)} / ${fmt(run.total_records, 0)} 个点 · ${smoothNote}`;
  $("trends-note").textContent = live ? "每次优化器更新一个点，横轴为累计更新次数；不同会话、阶段和回退的更新不连线。" : "横轴为累计更新次数；旧日志仅有稀疏汇总点，无法还原逐更新指标。";
  const recent = tail(200);
  $("primary-title").textContent = "训练损失";
  $("primary-value").textContent = fmt(live ? average("loss", recent) : run.latest?.metrics.loss);
  legend("primary-legend", [{ cls: "main", label: "加权宏动作 NLL" }], chart("chart-primary", "loss", "main"));
  $("secondary-title").textContent = metricLabel("accuracy");
  $("secondary-value").textContent = metricText("accuracy", live ? average("accuracy", recent) : run.latest?.metrics.accuracy);
  const accuracy = (p) => { const i = view.plot[p], hit = data.metrics.accuracy_correct?.[i], all = data.metrics.accuracy_decisions?.[i]; return isNumber(all) ? [`命中 ${fmt(hit, 0)} / ${fmt(all, 0)} 次决策`, ...detail(p)] : detail(p); };
  const secondNote = chart("chart-secondary", "accuracy", "main", { detail: accuracy, empty: "所选范围尚未记录动作正确率" });
  legend("secondary-legend", [{ cls: "main", label: "训练 Top-1 · 不计强制动作和控制器步骤" }], secondNote);
  const known = CHOICES.filter((key) => data.metrics[key]);
  const others = Object.keys(data.metrics).filter((key) => data.active.has(key) && !known.includes(key) && !CHARTED.includes(key) && !COUNTERS.has(key)).sort();
  const choices = [...known, ...others], options = JSON.stringify(choices);
  if (state.metricOptions !== options) {
    state.metricOptions = options;
    $("metric-select").replaceChildren(...choices.map((key) => { const option = el("option", "", metricLabel(key)); option.value = key; return option; }));
    if (!choices.includes(state.metric)) state.metric = choices[0] || "";
    $("metric-select").value = state.metric;
  }
  if (!state.metric) $("chart-metric").replaceChildren(el("div", "chart-empty", "尚无可选指标"));
  legend("metric-legend", state.metric ? [{ cls: "main", label: metricLabel(state.metric) }] : [], state.metric ? chart("chart-metric", state.metric, "main") : "");
  $("duration-title").textContent = live ? "单次更新耗时" : "汇总记录耗时";
  $("duration-value").textContent = duration(live ? average("duration_seconds", recent) : run.latest?.metrics.duration_seconds);
  legend("duration-legend", [{ cls: "time", label: live ? "逻辑批读取、前反向及优化器更新（秒）" : "旧日志记录的整轮耗时（秒）" }], chart("chart-duration", "duration_seconds", "time"));
}

// ---- Tables ----
function table(headers, rows) {
  const wrap = el("div", "table-wrap"), node = el("table"), head = el("thead"), header = el("tr"), body = el("tbody");
  for (const text of headers) { const th = el("th", "", text); th.scope = "col"; header.append(th); }
  head.append(header);
  for (const values of rows) {
    const row = el("tr");
    for (const value of values) { const td = el("td"); if (value instanceof Node) td.append(value); else td.textContent = String(value ?? "—"); row.append(td); }
    body.append(row);
  }
  node.append(head, body); wrap.append(node);
  return { wrap, body };
}
function noData(title, text) {
  const node = el("div", "table-empty");
  node.append(el("div", "", title), el("p", "", text));
  return node;
}
function renderRounds() {
  const panel = $("panel-rounds");
  const rounds = state.data.summaries.slice(-500).reverse();
  panel.replaceChildren();
  if (!rounds.length) { panel.append(noData("还没有完成的轮次", "每完成一轮写入一条汇总；逐更新指标在上方曲线。")); return; }
  panel.append(el("p", "panel-note", "每个 epoch 结束时的汇总，最新在前。损失和正确率是整轮的加权值，不是末次更新。"));
  panel.append(table(["轮次", "Update", "平均损失", "正确率", "策略熵", "末次梯度范数", "耗时", "完成时间"], rounds.map((record) => [
    fmt(record.round, 0), fmt(record.update, 0), fmt(record.metrics.loss), pct(record.metrics.accuracy, 2), fmt(record.metrics.entropy), fmt(record.metrics.last_grad_norm, 3), duration(record.metrics.duration_seconds), date(record.time),
  ])).wrap);
}
function renderGroups() {
  const run = state.data.run, latest = run.summary || (run.source !== "updates" ? run.latest : null), panel = $("panel-groups");
  const validation = Object.entries(latest?.validation || {}).filter(([, value]) => isObject(value));
  panel.replaceChildren();
  // BC no longer validates on its own; the tab is there only for logs that did.
  $("tab-groups").hidden = !validation.length;
  if ($("tab-groups").hidden) { if (state.tab === "groups") switchTab("rounds"); return; }
  const control = el("label", "scope-switch", "样本范围");
  const select = el("select");
  select.setAttribute("aria-label", "验证样本范围");
  for (const [value, label] of [["all", "混合难度"], ["a0", "仅 A0"]]) { const option = el("option", "", label); option.value = value; select.append(option); }
  select.value = state.scope;
  select.addEventListener("change", () => { state.scope = select.value; renderGroups(); });
  control.append(select);
  panel.append(control, el("p", "panel-note", `最近验证记录 · Update ${fmt(latest.update, 0)}。NLL 按分支数归一；A0 是混合验证的子集，不重复计数。`));
  const selected = validation.filter(([key]) => key.startsWith("A0/") === (state.scope === "a0")).sort(([a], [b]) => a.localeCompare(b));
  if (!selected.length) { panel.append(noData("没有 A0 验证样本", "该更新位置未记录 A0 分组。")); return; }
  panel.append(table(["角色", "决策阶段", "宏动作", "分支", "NLL / 分支"], selected.map(([key, value]) => {
    const parts = key.replace(/^A0\//, "").split("/");
    const nll = isNumber(value.nll) && value.branches > 0 ? value.nll / value.branches : value.nll_per_branch;
    return [parts[0], parts.slice(1).join(" / "), fmt(value.macros, 0), fmt(value.branches, 0), fmt(nll)];
  })).wrap);
}
function renderHistory() {
  const view = currentView(), panel = $("panel-history");
  const records = state.data.recent.filter((record) => !view.span || (isNumber(record.update) && view.end !== null && record.update > view.end - view.span && record.update <= view.end)).reverse();
  panel.replaceChildren();
  if (!records.length) { panel.append(noData("还没有完整训练记录", "新记录写入后会自动出现在这里。")); return; }
  panel.append(el("p", "panel-note", "按写入顺序倒序列出所选范围最近 200 条记录及其原始 JSON；曲线与 CSV 包含所选范围的全部记录。"));
  const rows = records.map((record) => [fmt(record.update, 0), record.stage, fmt(record.metrics.samples, 0), fmt(record.metrics.loss), pct(record.metrics.accuracy), fmt(record.metrics.entropy), duration(record.metrics.duration_seconds), date(record.time)]);
  const result = table(["Update", "阶段", "本次宏动作", "损失", "正确率", "熵", "耗时", "写入时间", "原始数据"], rows);
  Array.from(result.body.rows).forEach((row, index) => {
    const record = records[index], key = `${record.session}:${record.update}:${record.time}`;
    const cell = el("td"), button = el("button", "row-action", "展开");
    button.type = "button";
    const rawRow = el("tr", "raw-row"), rawCell = el("td");
    rawCell.colSpan = 9;
    rawRow.append(rawCell);
    row.after(rawRow);
    const show = (open) => {
      rawRow.hidden = !open;
      button.textContent = open ? "收起" : "展开";
      button.setAttribute("aria-expanded", String(open));
      if (open && !rawCell.childNodes.length) rawCell.append(el("pre", "", JSON.stringify(record.raw, null, 2)));
      if (open) state.expanded.add(key); else state.expanded.delete(key);
    };
    // A refresh rebuilds the table; rows the reader opened stay open.
    show(state.expanded.has(key));
    button.addEventListener("click", () => show(rawRow.hidden));
    cell.append(button); row.append(cell);
  });
  panel.append(result.wrap);
}
function renderConfig() {
  const panel = $("panel-config"), config = state.data.run.latest?.config;
  panel.replaceChildren();
  if (!config || !Object.keys(config).length) { panel.append(noData("配置尚未写入", "训练配置随首条完整指标一起记录。")); return; }
  panel.append(el("p", "panel-note", "最近一条记录的训练配置；历史配置与恢复训练的 session 可在原始记录中查看。"));
  const grid = el("dl", "config-grid");
  for (const [key, value] of Object.entries(config)) {
    const item = el("div", "config-item");
    item.append(el("dt", "", key), el("dd", "", isObject(value) || Array.isArray(value) ? JSON.stringify(value) : String(value ?? "—")));
    grid.append(item);
  }
  panel.append(grid);
}

// ---- Page ----
function renderCharts() {
  if (!state.data) return;
  renderCurves();
}
function renderDetail() {
  const data = state.data, run = data.run;
  $("empty").hidden = true;
  $("dashboard").hidden = false;
  $("run-name").textContent = runLabel(run);
  $("run-path").textContent = run.path;
  $("run-kind").textContent = "BOOTSTRAP";
  $("run-kind").className = "tag " + run.kind;
  $("export").disabled = !data.n;
  renderStatus(run);
  renderSummary(run);
  // Curves and tables are rebuilt only when the logs moved; the status line every time.
  const drawn = `${data.id}:${data.token}:${data.version}`;
  if (drawn === state.drawn) return;
  state.drawn = drawn;
  renderCharts(); renderRounds(); renderGroups(); renderHistory(); renderConfig();
}
function showEmpty(loading = false) {
  $("dashboard").hidden = true;
  $("empty").hidden = false;
  $("empty-title").textContent = loading ? "正在读取训练记录…" : state.runs.length ? "没有匹配的训练" : "等待训练记录";
  $("empty-description").textContent = loading ? "首次打开一个训练要读完它的全部日志，之后只同步新增的更新。" : state.runs.length ? "切换训练类型或调整目录搜索条件。" : "在以下目录启动 Bootstrap，网页会自动发现训练输出。";
  $("root-path").hidden = $("empty-hint").hidden = loading || state.runs.length > 0;
  $("root-path").textContent = state.root;
}
async function getJSON(path, signal) {
  const response = await fetch(path, { signal, cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `读取失败（${response.status}）`);
  return payload;
}
function schedule() {
  clearTimeout(state.timer);
  if ($("auto-refresh").checked && !document.hidden) state.timer = setTimeout(refresh, 5000);
}
async function refresh() {
  clearTimeout(state.timer);
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const request = ++state.request;
  $("refresh").disabled = true;
  try {
    const listing = await getJSON("/api/runs", controller.signal);
    if (request !== state.request) return;
    state.runs = listing.runs;
    state.root = listing.root;
    state.clock = Date.parse(listing.server_time) - Date.now() || 0;
    const matches = filteredRuns();
    if (!matches.some((run) => run.id === state.selected)) {
      state.selected = matches[0]?.id ?? null;
      state.data = null;
    }
    renderRuns();
    if (state.selected !== null) {
      const held = state.data?.id === state.selected ? state.data : null;
      if (!held) showEmpty(true);
      const query = new URLSearchParams({ id: state.selected });
      if (held) for (const [key, value] of [["after", held.base + held.n], ["summaries", held.summaryBase + held.summaries.length], ["token", held.token]]) query.set(key, value);
      const payload = await getJSON("/api/run?" + query, controller.signal);
      if (request !== state.request) return;
      merge(payload);
      renderDetail();
      const url = new URL(location.href);
      url.searchParams.set("run", state.selected);
      history.replaceState(null, "", url);
    } else showEmpty();
    setError(listing.discovery_errors ? `有 ${listing.discovery_errors} 个目录无法读取，已展示其余可用训练记录。` : "");
    $("connection").textContent = `已同步 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    $("connection").className = "connection";
  } catch (error) {
    if (error.name === "AbortError" || request !== state.request) return;
    setError(`无法同步训练数据：${error.message} 已有数据保留，可点击刷新重试。`);
    $("connection").textContent = "连接中断 · 数据可能过期";
    $("connection").className = "connection disconnected";
  } finally {
    if (request === state.request) { $("refresh").disabled = false; schedule(); }
  }
}
function switchTab(name, focus = false) {
  state.tab = name;
  for (const button of document.querySelectorAll("[data-tab]")) {
    const active = button.dataset.tab === name;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    $("panel-" + button.dataset.tab).hidden = !active;
    if (active && focus) button.focus();
  }
}
for (const button of document.querySelectorAll("[data-kind]")) button.addEventListener("click", () => {
  state.kind = button.dataset.kind;
  for (const item of document.querySelectorAll("[data-kind]")) { const active = item === button; item.classList.toggle("active", active); item.setAttribute("aria-pressed", String(active)); }
  refresh();
});
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const tabs = Array.from(document.querySelectorAll("[data-tab]")).filter((tab) => !tab.hidden).map((tab) => tab.dataset.tab);
    const index = tabs.indexOf(state.tab);
    switchTab(tabs[event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length], true);
  });
}
recall();
$("refresh").addEventListener("click", refresh);
$("auto-refresh").addEventListener("change", schedule);
let searchTimer;
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(refresh, 180); });
$("range").addEventListener("change", () => { remember("range"); renderCharts(); renderHistory(); });
for (const id of ["smooth", "scale"]) $(id).addEventListener("change", () => { remember(id); renderCharts(); });
$("metric-select").addEventListener("change", () => { state.metric = $("metric-select").value; renderCharts(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) clearTimeout(state.timer); else if ($("auto-refresh").checked) refresh(); });
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderCharts, 120); });
$("export").addEventListener("click", () => {
  if (!state.data) return;
  const data = state.data, rows = currentView().rows;
  const keys = Object.keys(data.metrics).filter((key) => key !== "updates" && rows.some((i) => data.metrics[key][i] !== null)).sort();
  const quote = (value) => {
    if (value === null || value === undefined) return "";
    let text = String(value);
    if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const { update, stage, session, time, policy_version: version, round } = data.meta;
  const lines = [["kind", "session", "update", "stage", "time", "policy_version", "round", ...keys].map(quote).join(",")];
  for (const i of rows) lines.push([data.run.kind, data.labels.session[session[i]], update[i], data.labels.stage[stage[i]], time[i] === null ? null : new Date(time[i] * 1000).toISOString(), version[i], round[i], ...keys.map((key) => data.metrics[key][i])].map(quote).join(","));
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob), link = el("a");
  link.href = url;
  link.download = `${data.run.name.replace(/[^\w一-鿿-]/g, "_")}-training.csv`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refresh();

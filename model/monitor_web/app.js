"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  runs: [], selected: new URLSearchParams(location.search).get("run"), detail: null,
  kind: "all", tab: "groups", scope: "all", metric: "", metricKind: "",
  timer: null, controller: null, request: 0, detailText: "", listText: "", root: "",
};
const COLORS = { green: "#278065", orange: "#bd763a", purple: "#8f81ab" };
const STATES = { starting: "正在启动", collecting: "采样中", updating: "训练更新中", running: "运行中", completed: "已完成", failed: "训练失败", interrupted: "已中断" };
const METRICS = {
  loss: "训练损失", accuracy: "动作正确率", entropy: "策略熵", kl: "新旧策略 KL", clip_fraction: "裁剪比例",
  last_grad_norm: "末次梯度范数", value_mse: "价值 MSE", value_rmse: "价值 RMSE",
  value_explained_variance: "价值解释方差", old_replay_max_error: "旧策略重放最大误差",
  return_std: "回报标准差", value_std: "价值标准差", updates: "累计更新次数",
  validation_nll: "混合验证 NLL / 分支", a0_validation_nll: "A0 验证 NLL / 分支",
  equal_character_win_rate: "五角色平均通关率", worst_character_win_rate: "最弱角色通关率",
  equal_character_completion_rate: "五角色平均完成率", duration_seconds: "耗时（秒）",
};
const isNumber = (value) => typeof value === "number" && Number.isFinite(value);
const isObject = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const fmt = (value, digits = 4) => !isNumber(value) ? "—" : value.toLocaleString("zh-CN", { maximumFractionDigits: digits });
const pct = (value) => !isNumber(value) ? "—" : `${fmt(value * 100, 1)}%`;
const percentMetric = (key) => key.endsWith("rate") || key === "clip_fraction" || key === "accuracy";
const metricText = (key, value) => percentMetric(key) ? pct(value) : fmt(value);
const updateNumber = (record) => {
  const value = record?.update ?? record?.metrics?.updates;
  return isNumber(value) && Number.isInteger(value) && value >= 0 ? value : null;
};
const runLabel = (run) => run.id.includes("/") ? run.id.split("/").slice(-2).join(" / ") : run.name;
function duration(value) {
  if (!isNumber(value)) return "—";
  if (value < 60) return `${fmt(value, 1)} s`;
  if (value < 3600) return `${fmt(value / 60, 1)} min`;
  return `${fmt(value / 3600, 2)} h`;
}
function date(value) {
  if (!value) return "未记录时间";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? "未记录时间" : parsed.toLocaleString("zh-CN", { hour12: false });
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
function filteredRuns() {
  const query = $("search").value.trim().toLowerCase();
  return state.runs.filter((run) => (state.kind === "all" || run.kind === state.kind) && `${run.name} ${run.id}`.toLowerCase().includes(query));
}
function renderRuns() {
  const runs = filteredRuns();
  const signature = JSON.stringify([runs, state.selected]);
  $("count-all").textContent = state.runs.length;
  for (const kind of ["bootstrap", "ppo"]) $("count-" + kind).textContent = state.runs.filter((run) => run.kind === kind).length;
  if (signature === state.listText) return;
  state.listText = signature;
  $("run-list").replaceChildren();
  if (!runs.length) $("run-list").append(el("p", "sidebar-empty", state.runs.length ? "没有匹配的训练记录" : "尚未发现训练记录"));
  for (const run of runs) {
    const button = el("button", "run-button" + (state.selected === run.id ? " selected" : ""));
    button.type = "button";
    button.setAttribute("aria-current", state.selected === run.id ? "true" : "false");
    button.title = run.path;
    button.append(el("strong", "", runLabel(run)), el("small", "", run.id));
    const meta = el("span", "run-meta");
    meta.append(el("span", "", run.kind === "bootstrap" ? "BOOTSTRAP" : "PPO"), el("span", "", STATES[run.status.state] || "状态未知"));
    button.append(meta);
    button.addEventListener("click", () => {
      if (state.selected === run.id) return;
      state.selected = run.id;
      state.detail = null;
      state.detailText = "";
      $("dashboard").hidden = true;
      renderRuns();
      refresh();
    });
    $("run-list").append(button);
  }
}
function visibleRecords(records = state.detail?.records || []) {
  const count = Number($("range").value);
  if (!count) return records;
  const end = updateNumber(records.at(-1));
  return end === null ? [] : records.filter((record) => updateNumber(record) !== null && updateNumber(record) > end - count && updateNumber(record) <= end);
}
function visibleSummaries() {
  const summaries = state.detail?.summaries || state.detail?.records || [];
  const count = Number($("range").value), end = updateNumber(state.detail?.latest);
  return !count ? summaries : summaries.filter((record) => end !== null && updateNumber(record) !== null && updateNumber(record) > end - count && updateNumber(record) <= end);
}
function renderSummary(run) {
  const last = run.latest;
  const m = last?.metrics || {};
  const summary = run.summary || (run.source !== "updates" ? last : null), report = summary?.metrics || {};
  const live = run.source === "updates";
  const accuracy = ["动作正确率", pct(m.accuracy), isNumber(m.accuracy)
    ? `训练 Top-1 · 命中 ${fmt(m.accuracy_correct, 0)} / ${fmt(m.accuracy_decisions, 0)} 次决策`
    : "该记录未统计正确率"];
  const updates = ["累计更新", fmt(updateNumber(last), 0), `策略版本 ${fmt(last?.policy_version, 0)}${live ? " · 本次 " + fmt(m.samples, 0) + " 个宏动作" : " · 旧日志汇总点"}`];
  const cards = run.kind === "bootstrap" ? (live ? [
    ["本次更新损失", fmt(m.loss), "当前逻辑批 · 加权宏动作 NLL"],
    accuracy,
    ["梯度范数", fmt(m.last_grad_norm), "本次更新 · 裁剪前"],
    updates,
  ] : [
    ["训练损失", fmt(m.loss), "旧日志汇总值 · 加权宏动作 NLL"],
    accuracy,
    ["A0 验证 NLL", fmt(m.a0_validation_nll), "仅 A0 留出样本 · 每分支"],
    updates,
  ]) : [
    ["五角色平均通关率", pct(report.equal_character_win_rate), summary ? `最近完成轮 · Update ${fmt(updateNumber(summary), 0)}` : "等待采样轮汇总 · 非独立评估"],
    [last?.stage === "value" ? "价值预热损失" : "训练损失", fmt(m.loss), live ? "当前逻辑批的加权平均" : "旧日志最后一个优化 epoch 的汇总值"],
    ["新旧策略 KL", fmt(m.kl), `目标 ${fmt(last?.config.target_kl)} · ${live ? "本次更新" : "旧日志汇总值"}`],
    updates,
  ];
  $("summary").replaceChildren(...cards.map(([title, value, note], index) => {
    const card = el("article", "metric-card");
    const top = el("div", "metric-top");
    top.append(el("span", "", title), el("span", "metric-number-label", `0${index + 1}`));
    card.append(top, el("div", "metric-value", value), el("div", "metric-foot", note));
    return card;
  }));
}
function renderStatus(run) {
  const status = run.status;
  $("state").textContent = STATES[status.state] || "状态未知";
  $("state").className = "state-tag " + (status.state === "failed" ? "failed" : status.state === "interrupted" ? "interrupted" : "");
  const parts = [];
  const updates = isNumber(status.updates) ? status.updates : updateNumber(run.latest);
  if (updates !== null) parts.push(`Update ${fmt(updates, 0)}`);
  if (run.kind === "ppo" && status.state === "collecting" && isNumber(status.round)) parts.push(`采样轮 ${status.round}`);
  if (isNumber(status.games_completed) && isNumber(status.games_total)) parts.push(`对局 ${status.games_completed} / ${status.games_total}`);
  if (status.character) parts.push(String(status.character));
  $("status-detail").textContent = parts.join(" · ");
  $("updated").textContent = "状态写入 " + date(run.updated_at);
  $("collection-progress").hidden = !(status.state === "collecting" && isNumber(status.games_completed) && isNumber(status.games_total) && status.games_total > 0);
  if (!$("collection-progress").hidden) {
    $("collection-progress").max = status.games_total;
    $("collection-progress").value = status.games_completed;
    $("collection-progress").setAttribute("aria-label", "本轮采样进度");
  }
  const warnings = [...run.warnings];
  if (status.error) warnings.push("训练错误：" + String(status.error));
  if (run.latest?.early_stop) warnings.push("最近记录触发 KL 提前停止，请结合 KL 曲线与目标值查看。");
  if ((run.summary || run.latest)?.evaluation.comparable === false) warnings.push("最近一轮包含未完成对局，通关率不能直接用于模型比较；未完成对局仍计入尝试次数。");
  if (run.latest?.stage === "value") warnings.push("最近一轮是价值头预热（value warmup），损失含义与 PPO 策略更新不同。");
  if (run.total_records > run.retained_records) warnings.push(`当前保留最近 ${run.retained_records} / ${run.total_records} 条记录；用 --limit 0 重启监测服务可读取全部历史。`);
  if (!run.latest) warnings.push("等待首次优化器更新的指标；旧训练进程需在下次启动时加载逐更新记录功能。");
  else if (run.source !== "updates") warnings.push("旧日志只有汇总指标，已按真实累计更新次数定位；未记录的中间更新无法补回。");
  if (run.kind === "bootstrap" && run.latest && !isNumber(run.latest.metrics.accuracy)) warnings.push("该记录未包含动作正确率。训练进程加载新代码后才会写入，旧记录无法由损失反推。");
  if (run.latest?.session && status.session && run.latest.session !== status.session) warnings.push("最近指标来自上一次训练会话，正在等待本次会话的更新记录。");
  $("warnings").replaceChildren(...warnings.map((message) => el("div", "notice", message)));
}
function legend(id, series, target = false) {
  $(id).replaceChildren();
  for (const line of series) $(id).append(el("span", "legend-dot " + line.color), el("span", "", line.label));
  if (target) $(id).append(el("span", "legend-dot orange"), el("span", "", "KL 目标（虚线）"));
}
function chart(id, records, series, options = {}) {
  const host = $(id);
  host.replaceChildren();
  records = records.filter((record) => updateNumber(record) !== null);
  const values = records.flatMap((record) => series.map((line) => record.metrics[line.key])).filter(isNumber);
  if (!values.length) {
    host.append(el("div", "chart-empty", options.empty || "尚无此项指标"));
    return;
  }
  const W = Math.max(260, host.clientWidth), H = Math.max(150, host.clientHeight);
  const left = 49, right = 15, top = 15, bottom = 32;
  const width = W - left - right, height = H - top - bottom;
  const targets = options.target ? records.map((record) => record.config[options.target]).filter(isNumber) : [];
  let low = Infinity, high = -Infinity;
  for (const value of [...values, ...targets]) { low = Math.min(low, value); high = Math.max(high, value); }
  if (options.percent) { low = 0; high = 1; }
  else {
    const padding = high === low ? Math.max(Math.abs(high) * 0.1, 0.01) : (high - low) * 0.14;
    low = low >= 0 ? Math.max(0, low - padding) : low - padding;
    high += padding;
  }
  const steps = records.map(updateNumber);
  const first = steps.reduce((a, b) => Math.min(a, b), Infinity);
  const last = steps.reduce((a, b) => Math.max(a, b), -Infinity);
  const position = (step) => left + (first === last ? width / 2 : (step - first) / (last - first) * width);
  const x = (index) => position(steps[index]);
  const y = (value) => top + (high - value) / (high - low) * height;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", tabindex: "0", "aria-label": `${options.title || series.map((line) => line.label).join("、")}；横轴为累计更新次数，${records.length} 条记录，使用左右方向键查看数值` });
  svg.append(svgEl("title", {}, options.title || series[0].label));
  for (let i = 0; i < 5; i++) {
    const value = low + (high - low) * i / 4;
    const yy = y(value);
    svg.append(svgEl("line", { x1: left, y1: yy, x2: W - right, y2: yy, class: "gridline" }));
    let label = options.percent ? pct(value) : Math.abs(value) >= 10000 ? value.toExponential(1) : Math.abs(value) > 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : fmt(value, high - low < 0.1 ? 4 : 2);
    svg.append(svgEl("text", { x: left - 9, y: yy + 3, "text-anchor": "end" }, label));
  }
  const tickCount = first === last ? 1 : Math.min(5, last - first + 1);
  const ticks = [...new Set(Array.from({ length: tickCount }, (_, i) => Math.round(first + i * (last - first) / Math.max(1, tickCount - 1))))];
  for (const step of ticks) svg.append(svgEl("text", { x: position(step), y: H - 13, "text-anchor": "middle" }, fmt(step, 0)));
  svg.append(svgEl("text", { x: W - right, y: H - 1, "text-anchor": "end" }, "Update"));
  function drawLine(getValue, attrs, dots = false) {
    let path = "", segment = false, lastPoint = null;
    records.forEach((record, index) => {
      const value = getValue(record);
      if (!isNumber(value)) { segment = false; return; }
      if (index && (record.stage !== records[index - 1].stage || record.session !== records[index - 1].session || steps[index] <= steps[index - 1])) segment = false;
      path += `${segment ? "L" : "M"}${x(index).toFixed(2)},${y(value).toFixed(2)} `;
      segment = true;
      lastPoint = [index, value];
      if (dots && records.length <= 12) svg.append(svgEl("circle", { cx: x(index), cy: y(value), r: 3.5, fill: attrs.stroke, class: "series-dot" }));
    });
    svg.append(svgEl("path", { d: path, fill: "none", ...attrs }));
    if (dots && lastPoint && records.length > 12) svg.append(svgEl("circle", { cx: x(lastPoint[0]), cy: y(lastPoint[1]), r: 3.5, fill: attrs.stroke, class: "series-dot" }));
  }
  if (options.target) drawLine((record) => record.config[options.target], { class: "target-line" });
  for (const line of series) drawLine((record) => record.metrics[line.key], { stroke: COLORS[line.color], class: "series-line" }, true);
  const focus = svgEl("line", { x1: 0, x2: 0, y1: top, y2: H - bottom, class: "focus-line", visibility: "hidden" });
  svg.append(focus);
  const tooltip = el("div", "chart-tooltip");
  tooltip.hidden = true;
  tooltip.setAttribute("role", "status");
  host.append(svg, tooltip);
  let active = records.length - 1;
  function inspect(index) {
    active = Math.max(0, Math.min(records.length - 1, index));
    const record = records[active];
    const lines = [`Update ${fmt(updateNumber(record), 0)}${record.stage === "value" ? " · 价值预热" : ""}`, ...series.map((line) => `${line.label}  ${metricText(line.key, record.metrics[line.key])}`)];
    if (series.some((line) => line.key === "accuracy") && isNumber(record.metrics.accuracy_decisions)) lines.push(`命中 ${fmt(record.metrics.accuracy_correct, 0)} / ${fmt(record.metrics.accuracy_decisions, 0)} 次决策`);
    if (record.session) lines.push(`Session ${String(record.session).slice(0, 8)}`);
    if (options.target) lines.push(`KL 目标  ${fmt(record.config[options.target])}`);
    tooltip.textContent = lines.join("\n");
    tooltip.hidden = false;
    tooltip.style.left = `${Math.max(0, Math.min(host.clientWidth - tooltip.offsetWidth, x(active) / W * host.clientWidth + 8))}px`;
    tooltip.style.top = "8px";
    focus.setAttribute("x1", x(active));
    focus.setAttribute("x2", x(active));
    focus.setAttribute("visibility", "visible");
  }
  const hide = () => { tooltip.hidden = true; focus.setAttribute("visibility", "hidden"); };
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const step = first + ((event.clientX - box.left) / box.width * W - left) / width * (last - first);
    inspect(steps.reduce((nearest, value, index) => Math.abs(value - step) <= Math.abs(steps[nearest] - step) ? index : nearest, 0));
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => inspect(active));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    inspect(event.key === "Home" ? 0 : event.key === "End" ? records.length - 1 : active + (event.key === "ArrowRight" ? 1 : -1));
  });
}
function renderCharts() {
  const run = state.detail;
  if (!run) return;
  const records = visibleRecords(), summaries = visibleSummaries(), bootstrap = run.kind === "bootstrap", live = run.source === "updates";
  const primary = bootstrap ? [{ key: "loss", label: "训练损失 · 宏动作 NLL", color: "green" }] : [{ key: "equal_character_win_rate", label: "五角色平均", color: "green" }, { key: "worst_character_win_rate", label: "最弱角色", color: "orange" }];
  const secondary = bootstrap ? [{ key: "accuracy", label: "训练 Top-1 · 不计强制动作和控制器步骤", color: "green" }] : [{ key: "loss", label: live ? "本次更新" : "旧日志汇总值", color: "green" }];
  $("primary-title").textContent = bootstrap ? "训练损失" : "训练采样通关率";
  $("secondary-title").textContent = bootstrap ? "动作正确率" : "训练损失";
  $("primary-value").textContent = metricText(primary[0].key, (bootstrap ? run.latest : run.summary || run.latest)?.metrics[primary[0].key]);
  $("secondary-value").textContent = metricText(secondary[0].key, run.latest?.metrics[secondary[0].key]);
  $("curve-note").textContent = `显示 ${fmt(records.length, 0)} / ${fmt(run.total_records, 0)} 条记录。` + (live ? "每次优化器更新一个点，横轴为累计更新次数；采样统计在整轮结束后产生。" : "横轴为累计更新次数；旧日志仅有稀疏汇总点，无法还原逐更新指标。");
  chart("chart-primary", bootstrap ? records : summaries, primary, { percent: !bootstrap, title: $("primary-title").textContent });
  chart("chart-secondary", records, secondary, { percent: bootstrap, title: $("secondary-title").textContent, empty: bootstrap ? "所选范围尚未记录动作正确率" : undefined });
  legend("primary-legend", primary);
  legend("secondary-legend", secondary);
  const choices = bootstrap ? ["entropy", "last_grad_norm", "validation_nll", "a0_validation_nll", "updates"] : ["kl", "clip_fraction", "entropy", "value_mse", "value_rmse", "value_explained_variance", "last_grad_norm", "old_replay_max_error", "return_std", "value_std", "equal_character_completion_rate", "updates"];
  if (state.metricKind !== run.kind) {
    state.metricKind = run.kind;
    state.metric = choices[0];
    $("metric-select").replaceChildren(...choices.map((key) => {
      const option = el("option", "", METRICS[key]);
      option.value = key;
      return option;
    }));
    $("metric-select").value = state.metric;
  }
  const metricSeries = [{ key: state.metric, label: METRICS[state.metric], color: "green" }];
  chart("chart-metric", records, metricSeries, { percent: percentMetric(state.metric), target: state.metric === "kl" ? "target_kl" : null });
  legend("metric-legend", metricSeries, state.metric === "kl");
  $("duration-value").textContent = duration(run.latest?.metrics.duration_seconds);
  $("duration-title").textContent = live ? "单次更新耗时" : "汇总记录耗时";
  $("duration-legend").textContent = live ? "逻辑批读取、前反向及优化器更新" : "旧日志记录的整轮耗时";
  chart("chart-duration", records, [{ key: "duration_seconds", label: "耗时（秒）", color: "purple" }]);
}
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
function renderGroups() {
  const run = state.detail, latest = run.summary || (run.source !== "updates" ? run.latest : null), panel = $("panel-groups");
  panel.replaceChildren();
  $("tab-groups").textContent = run.kind === "bootstrap" ? "验证明细" : "角色表现";
  if (!latest) { panel.append(noData("尚无分组汇总", "逐更新指标已单独记录；角色表现会在采样整轮结束后产生，BC 不自动验证。")); return; }
  if (run.kind === "bootstrap") {
    const groups = Object.entries(latest?.validation || {}).filter(([, value]) => isObject(value));
    if (!groups.length) {
      panel.append(noData("当前记录没有验证集指标", "BC 不自动验证；可自行选择每轮保存的检查点进行评估。"));
      return;
    }
    const control = el("label", "scope-switch", "样本范围");
    const select = el("select");
    select.setAttribute("aria-label", "验证样本范围");
    for (const [value, label] of [["all", "混合难度"], ["a0", "仅 A0"]]) { const option = el("option", "", label); option.value = value; select.append(option); }
    select.value = state.scope;
    select.addEventListener("change", () => { state.scope = select.value; renderGroups(); });
    control.append(select);
    panel.append(control, el("p", "panel-note", `最近验证记录 · Update ${fmt(updateNumber(latest), 0)}。NLL 按分支数归一；A0 是混合验证的子集，不重复计数。`));
    const selected = groups.filter(([key]) => key.startsWith("A0/") === (state.scope === "a0")).sort(([a], [b]) => a.localeCompare(b));
    if (!selected.length) { panel.append(noData("没有 A0 验证样本", "该更新位置未记录 A0 分组。")); return; }
    panel.append(table(["角色", "决策阶段", "宏动作", "分支", "NLL / 分支"], selected.map(([key, value]) => {
      const parts = key.replace(/^A0\//, "").split("/");
      const nll = isNumber(value.nll) && value.branches > 0 ? value.nll / value.branches : value.nll_per_branch;
      return [parts[0], parts.slice(1).join(" / "), fmt(value.macros, 0), fmt(value.branches, 0), fmt(nll)];
    })).wrap);
  } else {
    const evaluation = latest?.evaluation || {};
    const groups = isObject(evaluation.characters) ? Object.entries(evaluation.characters).filter(([, value]) => isObject(value)) : [];
    if (!groups.length) { panel.append(noData("尚无本轮采样结果", "角色通关率会在一轮采样与更新完成后写入。")); return; }
    const ascensions = Array.isArray(evaluation.ascensions) ? evaluation.ascensions.map((value) => "A" + value).join(" / ") : "";
    panel.append(el("p", "panel-note", `最近一轮训练采样${ascensions ? " · " + ascensions : ""}。通关率以全部尝试为分母，区间为 Wilson 95%；模型效果请另用固定 seed 独立评估。`));
    panel.append(table(["角色", "尝试", "完成", "通关", "通关率", "95% 区间", "完成率", "异常 / 未完成"], groups.map(([name, item]) => {
      const rate = el("span", "number", pct(item.win_rate));
      if (isNumber(item.win_rate)) { const bar = el("progress", "mini-bar"); bar.max = 1; bar.value = item.win_rate; bar.setAttribute("aria-label", name + " 通关率"); rate.append(bar); }
      return [name, fmt(item.attempts, 0), fmt(item.complete, 0), fmt(item.wins, 0), rate, Array.isArray(item.wilson_95) ? item.wilson_95.map(pct).join(" – ") : "—", pct(item.completion_rate), fmt(item.errors_or_unresolved, 0)];
    })).wrap);
  }
}
function renderHistory() {
  const records = visibleRecords(state.detail.recent_records || state.detail.records).slice(-200).reverse();
  const bootstrap = state.detail.kind === "bootstrap";
  const panel = $("panel-history");
  panel.replaceChildren();
  if (!records.length) { panel.append(noData("还没有完整训练记录", "新记录写入后会自动出现在这里。")); return; }
  panel.append(el("p", "panel-note", "表格按写入顺序倒序列出所选范围最近 200 条；曲线与 CSV 包含所选范围全部记录。原始记录保留轮次、优化 epoch 和 session。"));
  const rows = records.map((record) => [fmt(updateNumber(record), 0), record.stage === "value" ? "价值预热" : record.stage, fmt(record.metrics.samples, 0), fmt(record.metrics.loss), ...(bootstrap ? [pct(record.metrics.accuracy)] : []), fmt(record.metrics.entropy), duration(record.metrics.duration_seconds), date(record.time)]);
  const result = table(["Update", "阶段", "本次宏动作", "损失", ...(bootstrap ? ["正确率"] : []), "熵", "耗时", "写入时间", "原始数据"], rows);
  Array.from(result.body.rows).forEach((row, index) => {
    const cell = el("td"), button = el("button", "row-action", "展开");
    button.type = "button";
    button.setAttribute("aria-expanded", "false");
    const rawRow = el("tr", "raw-row"), rawCell = el("td");
    rawCell.colSpan = bootstrap ? 9 : 8;
    rawRow.hidden = true;
    rawRow.append(rawCell);
    row.after(rawRow);
    button.addEventListener("click", () => {
      rawRow.hidden = !rawRow.hidden;
      button.textContent = rawRow.hidden ? "展开" : "收起";
      button.setAttribute("aria-expanded", String(!rawRow.hidden));
      if (!rawRow.hidden && !rawCell.childNodes.length) rawCell.append(el("pre", "", JSON.stringify(records[index].raw, null, 2)));
    });
    cell.append(button); row.append(cell);
  });
  panel.append(result.wrap);
}
function renderConfig() {
  const panel = $("panel-config"), config = state.detail.latest?.config;
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
function renderDetail() {
  const run = state.detail;
  $("empty").hidden = !!run;
  $("dashboard").hidden = !run;
  if (!run) return;
  $("run-name").textContent = runLabel(run);
  $("run-path").textContent = run.path;
  $("run-kind").textContent = run.kind === "bootstrap" ? "BOOTSTRAP" : "PPO";
  $("run-kind").className = "tag " + run.kind;
  $("record-count").textContent = `${run.total_records} 条记录`;
  $("export").disabled = !run.records.length;
  renderStatus(run); renderSummary(run); renderCharts(); renderGroups(); renderHistory(); renderConfig();
}
function showEmpty() {
  $("dashboard").hidden = true;
  $("empty").hidden = false;
  $("empty-title").textContent = state.runs.length ? "没有匹配的训练" : "等待训练记录";
  $("empty-description").textContent = state.runs.length ? "切换训练类型或调整目录搜索条件。" : "在以下目录启动 Bootstrap 或 PPO，网页会自动发现训练输出。";
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
    const matches = filteredRuns();
    if (!matches.some((run) => run.id === state.selected)) {
      state.selected = matches[0]?.id ?? null;
      state.detail = null;
      state.detailText = "";
    }
    renderRuns();
    if (state.selected !== null) {
      const payload = await getJSON("/api/run?id=" + encodeURIComponent(state.selected) + "&compact=1", controller.signal);
      if (request !== state.request) return;
      const signature = JSON.stringify(payload);
      state.detail = payload;
      if (signature !== state.detailText) { renderDetail(); state.detailText = signature; }
      const url = new URL(location.href);
      url.searchParams.set("run", state.selected);
      history.replaceState(null, "", url);
    } else showEmpty();
    setError(listing.discovery_errors ? `有 ${listing.discovery_errors} 个目录无法读取，已展示其余可用训练记录。` : "");
    $("connection").textContent = `已同步 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
    $("connection").className = "connection";
    $("footer-note").textContent = "状态以日志最后写入为准 · 时间为浏览器本地时间";
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
    const tabs = ["groups", "history", "config"];
    const index = tabs.indexOf(state.tab);
    switchTab(tabs[event.key === "Home" ? 0 : event.key === "End" ? 2 : (index + (event.key === "ArrowRight" ? 1 : 2)) % 3], true);
  });
}
$("refresh").addEventListener("click", refresh);
$("auto-refresh").addEventListener("change", schedule);
let searchTimer;
$("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(refresh, 180); });
$("range").addEventListener("change", () => { renderCharts(); renderHistory(); });
$("metric-select").addEventListener("change", () => { state.metric = $("metric-select").value; renderCharts(); });
document.addEventListener("visibilitychange", () => { if (document.hidden) clearTimeout(state.timer); else if ($("auto-refresh").checked) refresh(); });
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderCharts, 120); });
$("export").addEventListener("click", () => {
  if (!state.detail) return;
  const records = visibleRecords();
  const keys = [...new Set(records.flatMap((record) => Object.keys(record.metrics).filter((key) => key !== "updates" && isNumber(record.metrics[key]))))].sort();
  const quote = (value) => {
    if (value === null || value === undefined) return "";
    let text = String(value);
    if (typeof value === "string" && /^[=+\-@\t\r]/.test(text)) text = "'" + text;
    return '"' + text.replaceAll('"', '""') + '"';
  };
  const headers = ["kind", "session", "update", "stage", "time", "policy_version", "round", ...keys];
  const lines = [headers, ...records.map((record) => [record.kind, record.session, updateNumber(record), record.stage, record.time, record.policy_version, record.round, ...keys.map((key) => record.metrics[key])])];
  const blob = new Blob(["\ufeff" + lines.map((line) => line.map(quote).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob), link = el("a");
  link.href = url;
  link.download = `${state.detail.name.replace(/[^\w\u4e00-\u9fff-]/g, "_")}-training.csv`;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
refresh();

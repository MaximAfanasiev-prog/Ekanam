"use strict";

const state = { summary: null, queries: [], activeIndex: 0 };
const byId = (id) => document.getElementById(id);
const fmt = (value, digits = 3) => Number(value).toFixed(digits);
const percent = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const shortHash = (value) => `${String(value).slice(0, 9)}…`;

async function fetchJson(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function setFacts(id, facts) {
  const root = byId(id);
  root.replaceChildren();
  for (const [label, value, title] of facts) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    if (title) detail.title = title;
    row.append(term, detail);
    root.append(row);
  }
}

function renderMetricGrid(best) {
  const metrics = [
    ["mAP@10", best.ranking["mAP@10"], "средняя точность top-10"],
    ["Rank-1", best.ranking["Rank-1"], "верный матч на 1-м месте"],
    ["Rank-5", best.ranking["Rank-5"], "верный матч в первой пятёрке"],
    ["Open-set F1", best.refusal.F1, "баланс precision и recall"],
  ];
  const root = byId("metric-grid");
  root.replaceChildren();
  for (const [label, value, note] of metrics) {
    const card = document.createElement("article");
    card.className = "metric-card";
    const labelNode = document.createElement("span");
    labelNode.className = "metric-label";
    labelNode.textContent = label;
    const valueNode = document.createElement("strong");
    valueNode.className = "metric-value";
    valueNode.textContent = percent(value);
    const noteNode = document.createElement("span");
    noteNode.className = "metric-note";
    noteNode.textContent = note;
    card.append(labelNode, valueNode, noteNode);
    root.append(card);
  }
}

function makeChart(rootId, history, definitions, options = {}) {
  const root = byId(rootId);
  root.replaceChildren();
  const width = 720;
  const height = 250;
  const padding = { top: 18, right: 18, bottom: 28, left: 42 };
  const values = definitions.flatMap((definition) => history.map(definition.value));
  let min = options.min ?? Math.min(...values);
  let max = options.max ?? Math.max(...values);
  const spread = max - min || 1;
  if (options.min === undefined) min -= spread * 0.08;
  if (options.max === undefined) max += spread * 0.08;
  const x = (index) => padding.left + (index / Math.max(history.length - 1, 1)) * (width - padding.left - padding.right);
  const y = (value) => padding.top + ((max - value) / (max - min)) * (height - padding.top - padding.bottom);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("aria-hidden", "true");

  for (let step = 0; step <= 4; step += 1) {
    const value = min + ((max - min) * (4 - step)) / 4;
    const position = padding.top + (step / 4) * (height - padding.top - padding.bottom);
    const line = document.createElementNS(svg.namespaceURI, "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", position);
    line.setAttribute("y2", position);
    line.setAttribute("class", "chart-gridline");
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", padding.left - 8);
    label.setAttribute("y", position + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "chart-label");
    label.textContent = options.percent ? `${Math.round(value * 100)}` : value.toFixed(1);
    svg.append(line, label);
  }

  [0, Math.floor((history.length - 1) / 2), history.length - 1].forEach((index) => {
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", x(index));
    label.setAttribute("y", height - 7);
    label.setAttribute("text-anchor", index === 0 ? "start" : index === history.length - 1 ? "end" : "middle");
    label.setAttribute("class", "chart-label");
    label.textContent = `epoch ${history[index].epoch}`;
    svg.append(label);
  });

  for (const definition of definitions) {
    const path = document.createElementNS(svg.namespaceURI, "path");
    const d = history.map((item, index) => `${index ? "L" : "M"}${x(index).toFixed(2)},${y(definition.value(item)).toFixed(2)}`).join(" ");
    path.setAttribute("d", d);
    path.setAttribute("class", "chart-path");
    path.setAttribute("stroke", definition.color);
    svg.append(path);
    const lastIndex = history.length - 1;
    const dot = document.createElementNS(svg.namespaceURI, "circle");
    dot.setAttribute("cx", x(lastIndex));
    dot.setAttribute("cy", y(definition.value(history[lastIndex])));
    dot.setAttribute("r", 4);
    dot.setAttribute("class", "chart-dot");
    dot.setAttribute("fill", definition.color);
    svg.append(dot);
  }
  root.append(svg);
}

function renderSummary(summary) {
  state.summary = summary;
  byId("release-label").textContent = summary.release;
  byId("boundary-text").textContent = summary.boundary;
  byId("completed-at").textContent = `Собрано ${new Intl.DateTimeFormat("ru-RU", { dateStyle: "long", timeStyle: "short" }).format(new Date(summary.completedAt))}`;
  byId("model-tag").textContent = `${summary.model.backbone} · ${summary.model.embeddingDim}d`;
  byId("device-tag").textContent = `${String(summary.model.device).toUpperCase()} · torch ${summary.model.torch}`;
  byId("best-epoch").textContent = `Эпоха ${summary.best.epoch} из ${summary.history.length}`;
  renderMetricGrid(summary.best);
  makeChart("quality-chart", summary.history, [
    { value: (item) => item.ranking["mAP@10"], color: "#5870c8" },
    { value: (item) => item.ranking["Rank-1"], color: "#8870c8" },
    { value: (item) => item.ranking["Rank-5"], color: "#3e3e49" },
  ], { min: 0, max: 0.75, percent: true });
  makeChart("loss-chart", summary.history, [
    { value: (item) => item.loss, color: "#5870c8" },
  ]);
  const firstLoss = summary.history[0].loss;
  const finalLoss = summary.history.at(-1).loss;
  byId("loss-delta").textContent = `${((finalLoss / firstLoss - 1) * 100).toFixed(0)}%`;

  setFacts("split-facts", [
    ["Train", summary.split.train_records.toLocaleString("ru-RU")],
    ["Validation query", summary.split.query_records.toLocaleString("ru-RU")],
    ["Validation gallery", summary.split.gallery_records.toLocaleString("ru-RU")],
  ]);
  setFacts("submission-facts", [
    ["Test query", summary.submission.query_count.toLocaleString("ru-RU")],
    ["Принято", summary.submission.acceptedQueries.toLocaleString("ru-RU")],
    ["Отказ", summary.submission.refusedQueries.toLocaleString("ru-RU")],
  ]);
  setFacts("provenance-facts", [
    ["Commit", shortHash(summary.model.sourceCommit), summary.model.sourceCommit],
    ["Checkpoint", shortHash(summary.model.checkpointSha256), summary.model.checkpointSha256],
    ["Embedding", `${summary.model.embeddingDim}d`],
  ]);
}

function populateQueries(items) {
  state.queries = items;
  const select = byId("query-select");
  select.replaceChildren();
  items.forEach((query, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${String(index + 1).padStart(4, "0")} · ${query.imageId}${query.accepted ? "" : " · отказ"}`;
    select.append(option);
  });
}

function setQueryState(name, message = "") {
  byId("query-loading").hidden = name !== "loading";
  byId("query-error").hidden = name !== "error";
  byId("query-empty").hidden = name !== "empty";
  byId("retrieval-view").hidden = name !== "ready";
  if (message) byId("query-error").textContent = message;
}

function renderGallery(items) {
  const root = byId("gallery-grid");
  root.replaceChildren();
  for (const item of items) {
    const card = document.createElement("li");
    card.className = "gallery-card";
    const frame = document.createElement("div");
    frame.className = "image-frame";
    const image = document.createElement("img");
    image.loading = "lazy";
    image.src = `/api/image/${encodeURIComponent(item.imageId)}`;
    image.alt = `Кандидат ${item.rank}: ${item.imageId}`;
    const rank = document.createElement("span");
    rank.className = "rank-pill";
    rank.textContent = `#${item.rank}`;
    frame.append(image, rank);
    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    const id = document.createElement("span");
    id.className = "gallery-id";
    id.textContent = item.imageId;
    id.title = item.imageId;
    const score = document.createElement("strong");
    score.className = "gallery-score";
    score.textContent = fmt(item.score, 4);
    meta.append(id, score);
    card.append(frame, meta);
    root.append(card);
  }
}

async function loadQuery(index) {
  if (!state.queries.length) {
    setQueryState("empty");
    return;
  }
  state.activeIndex = (index + state.queries.length) % state.queries.length;
  byId("query-select").value = String(state.activeIndex);
  setQueryState("loading");
  const item = state.queries[state.activeIndex];
  try {
    const detail = await fetchJson(`/api/query/${encodeURIComponent(item.imageId)}`);
    if (!detail.gallery.length) {
      setQueryState("empty");
      return;
    }
    byId("query-image").src = `/api/image/${encodeURIComponent(detail.imageId)}`;
    byId("query-image").alt = `Query автомобиля ${detail.imageId}`;
    byId("query-id").textContent = detail.imageId;
    byId("query-position").textContent = `${state.activeIndex + 1} / ${state.queries.length}`;
    byId("threshold-value").textContent = fmt(detail.threshold, 4);
    byId("top-score").textContent = fmt(detail.gallery[0].score, 4);
    const accepted = detail.decision === "accepted";
    const badge = byId("decision-badge");
    badge.className = `decision-badge ${accepted ? "accepted" : "refused"}`;
    badge.textContent = accepted ? "Матч принят" : "Отказ от матча";
    byId("decision-copy").textContent = accepted
      ? "Лучший кандидат прошёл порог open-set решения."
      : "Даже лучший кандидат не прошёл порог уверенности.";
    renderGallery(detail.gallery);
    setQueryState("ready");
  } catch (error) {
    setQueryState("error", `Не удалось загрузить query: ${error.message}`);
  }
}

async function initialize() {
  byId("loading-state").hidden = false;
  byId("error-state").hidden = true;
  byId("dashboard").hidden = true;
  try {
    const [summary, queryPayload] = await Promise.all([
      fetchJson("/api/summary"),
      fetchJson("/api/queries"),
    ]);
    renderSummary(summary);
    populateQueries(queryPayload.items);
    byId("loading-state").hidden = true;
    byId("dashboard").hidden = false;
    await loadQuery(0);
  } catch (error) {
    byId("loading-state").hidden = true;
    byId("error-state").hidden = false;
    byId("error-message").textContent = error.message;
  }
}

byId("retry-button").addEventListener("click", initialize);
byId("query-select").addEventListener("change", (event) => loadQuery(Number(event.target.value)));
byId("previous-query").addEventListener("click", () => loadQuery(state.activeIndex - 1));
byId("next-query").addEventListener("click", () => loadQuery(state.activeIndex + 1));
initialize();

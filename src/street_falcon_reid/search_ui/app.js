"use strict";
const $ = (id) => document.getElementById(id);
const names = ["x", "y", "w", "h"];
const state = {file: null, bitmap: null, ready: false, busy: false, generation: 0};
const form = $("search-form");
const context = $("canvas").getContext("2d");
let selection = null;
const api = (path) => new URL("api/v1/" + path, document.baseURI);

function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function bbox() {
  if (!state.bitmap) return {error: "Сначала загрузите изображение."};
  const values = names.map((name) => $(name).value.trim());
  if (values.some((value) => value === "")) return {error: "Заполните X, Y, W и H."};
  const [x, y, w, h] = values.map(Number);
  if (![x, y, w, h].every(Number.isSafeInteger)) return {error: "Координаты должны быть целыми числами."};
  if (x < 0 || y < 0 || w <= 0 || h <= 0) return {error: "X и Y — от нуля. W и H — больше нуля."};
  if (x + w > state.bitmap.width || y + h > state.bitmap.height) return {error: "Прямоугольник выходит за границы изображения."};
  return {x, y, w, h};
}
function render() {
  const box = bbox();
  $("bbox-help").textContent = box.error || "Координаты корректны. Область отмечена на кадре.";
  const invalid = Boolean(state.bitmap && names.every((name) => $(name).value !== "") && box.error);
  $("bbox-help").classList.toggle("invalid", invalid);
  names.forEach((name) => $(name).setAttribute("aria-invalid", String(invalid)));
  $("submit").disabled = state.busy || Boolean(selection) || !state.ready || Boolean(box.error);
  if (!state.bitmap) return;
  const scale = Math.min(1, 1400 / state.bitmap.width, 700 / state.bitmap.height);
  const canvas = $("canvas");
  canvas.width = Math.round(state.bitmap.width * scale);
  canvas.height = Math.round(state.bitmap.height * scale);
  context.drawImage(state.bitmap, 0, 0, canvas.width, canvas.height);
  if (!box.error) {
    const sx = canvas.width / state.bitmap.width, sy = canvas.height / state.bitmap.height;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 5;
    context.strokeRect(box.x * sx, box.y * sy, box.w * sx, box.h * sy);
    context.strokeStyle = "#19a66e";
    context.lineWidth = 2.5;
    context.strokeRect(box.x * sx, box.y * sy, box.w * sx, box.h * sy);
  }
}
function clearResults() {
  $("result-data").hidden = true;
  $("matches").replaceChildren();
  $("result-empty").hidden = false;
  $("result-count").textContent = "ОЖИДАНИЕ ЗАПРОСА";
}
function setBusy(busy) {
  state.busy = busy;
  Array.from(form.elements).forEach((element) => {element.disabled = busy;});
  $("loading").hidden = !busy;
  if (busy) $("result-empty").hidden = true;
  $("submit").textContent = busy ? "Идёт поиск…" : "Найти похожие ↗";
  render();
}
// Reject orientation metadata rather than showing a rotated frame with wrong coordinates.
function jpegOrientation(bytes) {
  const view = new DataView(bytes);
  if (view.byteLength < 4 || view.getUint16(0) !== 0xffd8) return 1;
  let pos = 2;
  while (pos + 4 <= view.byteLength) {
    const marker = view.getUint16(pos); pos += 2;
    if (marker === 0xffda || marker === 0xffd9) break;
    const size = view.getUint16(pos);
    if (size < 2 || pos + size > view.byteLength) break;
    if (marker === 0xffe1 && size >= 16 && view.getUint32(pos + 2) === 0x45786966) {
      const start = pos + 8;
      const little = view.getUint16(start) === 0x4949;
      const ifd = start + view.getUint32(start + 4, little);
      if (ifd + 2 > pos + size) return 1;
      const count = view.getUint16(ifd, little);
      for (let i = 0; i < count; i++) {
        const entry = ifd + 2 + i * 12;
        if (entry + 12 > pos + size) break;
        if (view.getUint16(entry, little) === 0x0112) return view.getUint16(entry + 8, little);
      }
    }
    pos += size;
  }
  return 1;
}
function sourceDimensions(bytes) {
  const view = new DataView(bytes);
  if (view.byteLength >= 24 && view.getUint32(0) === 0x89504e47) {
    let offset = 8;
    while (offset + 12 <= view.byteLength) {
      const size = view.getUint32(offset);
      if (view.getUint32(offset + 4) === 0x65584966) throw new Error("PNG содержит EXIF-метаданные. Сохраните копию без метаданных поворота.");
      if (offset + size + 12 > view.byteLength) break;
      offset += size + 12;
    }
    return [view.getUint32(16), view.getUint32(20)];
  }
  let pos = 2;
  const sof = [0xffc0,0xffc1,0xffc2,0xffc3,0xffc5,0xffc6,0xffc7,0xffc9,0xffca,0xffcb,0xffcd,0xffce,0xffcf];
  while (pos + 4 <= view.byteLength) {
    const marker = view.getUint16(pos); pos += 2;
    if (marker === 0xffda || marker === 0xffd9) break;
    const size = view.getUint16(pos);
    if (size < 2 || pos + size > view.byteLength) break;
    if (sof.includes(marker) && size >= 7) return [view.getUint16(pos + 5), view.getUint16(pos + 3)];
    pos += size;
  }
  throw new Error("Не удалось прочитать размеры изображения.");
}

async function choose(file) {
  if (state.busy || !file) return;
  cancelSelection();
  const generation = ++state.generation;
  state.file = null;
  if (state.bitmap) state.bitmap.close();
  state.bitmap = null;
  names.forEach((name) => {$(name).value = "";});
  $("preview").hidden = true; $("dropzone").hidden = false;
  clearResults(); error(""); render();
  try {
    if (file.size > 10 * 1024 * 1024) throw new Error("Размер файла превышает 10 MiB.");
    const bytes = await file.arrayBuffer();
    const header = new Uint8Array(bytes);
    const jpeg = header[0] === 255 && header[1] === 216;
    const png = header[0] === 137 && header[1] === 80 && header[2] === 78 && header[3] === 71;
    if (!jpeg && !png) throw new Error("Выберите изображение JPEG или PNG.");
    if (jpeg && jpegOrientation(bytes) !== 1) throw new Error("Файл содержит EXIF-поворот. Сохраните изображение без метаданных поворота, чтобы координаты совпали с исходными пикселями.");
    const [width, height] = sourceDimensions(bytes);
    if (!width || !height || width * height > 20000000) throw new Error("Изображение превышает 20 млн пикселей.");
    const bitmap = await createImageBitmap(file, {imageOrientation: "none"});
    if (generation !== state.generation) {bitmap.close(); return;}
    if (bitmap.width * bitmap.height > 20000000) {bitmap.close(); throw new Error("Изображение превышает 20 млн пикселей.");}
    state.file = file; state.bitmap = bitmap;
    $("filename").textContent = file.name;
    $("dimensions").textContent = bitmap.width + " × " + bitmap.height + " px · " + (file.size / 1024 / 1024).toFixed(2) + " MiB";
    $("preview").hidden = false; $("dropzone").hidden = true;
    $("x").focus();
  } catch (exception) {
    if (generation === state.generation) error(exception.message || "Не удалось открыть изображение.");
  } finally {
    if (generation === state.generation) render();
  }
}
async function checkReady() {
  $("refresh").disabled = true;
  try {
    const response = await fetch(api("info"), {signal: AbortSignal.timeout(5000)});
    if (!response.ok) throw new Error();
    const info = await response.json();
    state.ready = true;
    loadMetrics(info);
    $("status-text").textContent = "На связи · " + info.gallery_count + " изображений";
  } catch {
    state.ready = false; $("status-text").textContent = "Сервис недоступен";
  } finally {
    $("status-dot").classList.toggle("online", state.ready);
    $("refresh").disabled = false; render();
  }
}
// Pointer coordinates refer to the rendered image, including object-fit letterboxing.
function imagePoint(event, clamp = false) {
  const rect = $("canvas").getBoundingClientRect();
  const scale = Math.min(rect.width / state.bitmap.width, rect.height / state.bitmap.height);
  if (!scale) return null;
  const left = rect.left + (rect.width - state.bitmap.width * scale) / 2;
  const top = rect.top + (rect.height - state.bitmap.height * scale) / 2;
  let x = (event.clientX - left) / scale, y = (event.clientY - top) / scale;
  if (!clamp && (x < 0 || y < 0 || x > state.bitmap.width || y > state.bitmap.height)) return null;
  x = Math.max(0, Math.min(state.bitmap.width, Math.round(x)));
  y = Math.max(0, Math.min(state.bitmap.height, Math.round(y)));
  return {x, y};
}
function cancelSelection() {
  if (!selection) return;
  const previous = selection;
  selection = null;
  names.forEach((name, i) => {$(name).value = previous.values[i];});
  if ($("canvas").hasPointerCapture(previous.id)) $("canvas").releasePointerCapture(previous.id);
  render();
}
function updateSelection(event) {
  if (!selection || event.pointerId !== selection.id) return;
  const end = imagePoint(event, true);
  if (!end) return;
  const x = Math.min(selection.start.x, end.x), y = Math.min(selection.start.y, end.y);
  const w = Math.abs(end.x - selection.start.x), h = Math.abs(end.y - selection.start.y);
  if (w < 1 || h < 1) {selection.valid = false; return;}
  selection.valid = true;
  [x, y, w, h].forEach((value, i) => {$(names[i]).value = String(value);});
  clearResults(); error(""); render();
}
$("canvas").addEventListener("pointerdown", (event) => {
  if (state.busy || !state.bitmap || selection || !event.isPrimary || event.button !== 0) return;
  const start = imagePoint(event);
  if (!start) return;
  event.preventDefault();
  selection = {id: event.pointerId, start, values: names.map(name => $(name).value), valid: false};
  $("canvas").setPointerCapture(event.pointerId);
  render();
});
$("canvas").addEventListener("pointermove", (event) => {
  if (!selection || event.pointerId !== selection.id) return;
  event.preventDefault();
  updateSelection(event);
});
$("canvas").addEventListener("pointerup", (event) => {
  if (!selection || event.pointerId !== selection.id) return;
  updateSelection(event);
  if (!selection.valid) {cancelSelection(); return;}
  selection = null;
  if ($("canvas").hasPointerCapture(event.pointerId)) $("canvas").releasePointerCapture(event.pointerId);
  render();
});
$("canvas").addEventListener("pointercancel", cancelSelection);
$("canvas").addEventListener("lostpointercapture", cancelSelection);
window.addEventListener("blur", cancelSelection);
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") cancelSelection();
});
$("image-file").addEventListener("change", (event) => {
  choose(event.target.files[0]); event.target.value = "";
});
$("replace").addEventListener("click", () => $("image-file").click());
$("refresh").addEventListener("click", checkReady);
names.forEach((name) => $(name).addEventListener("input", () => {clearResults(); error(""); render();}));
$("top-k").addEventListener("change", clearResults);
["dragenter", "dragover"].forEach((name) => $("dropzone").addEventListener(name, (event) => {
  event.preventDefault(); if (!state.busy) $("dropzone").classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => $("dropzone").addEventListener(name, (event) => {
  event.preventDefault(); $("dropzone").classList.remove("dragging");
  if (name === "drop") choose(event.dataTransfer.files[0]);
}));
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const box = bbox();
  if (state.busy || selection || !state.ready || box.error) {render(); return;}
  error(""); clearResults(); setBusy(true);
  try {
    const data = new FormData();
    data.append("image", state.file);
    names.forEach((name) => data.append(name, String(box[name])));
    data.append("top_k", $("top-k").value);
    const response = await fetch(api("search"), {method: "POST", body: data, signal: AbortSignal.timeout(45000)});
    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      const messages = {408: "Загрузка заняла слишком много времени.", 413: "Файл или разрешение слишком велики.", 415: "Изображение не поддерживается. Используйте статический JPEG или PNG.", 422: "Проверьте координаты и формат полей.", 429: "Сервис занят. Повторите поиск через несколько секунд.", 500: "Ошибка обработки на сервере.", 503: "Модель пока не готова. Попробуйте позже."};
      const id = result.request_id || response.headers.get("x-request-id");
      throw new Error((messages[response.status] || "Не удалось выполнить поиск.") + (id ? " ID запроса: " + id : ""));
    }
    if (!Array.isArray(result.matches)) throw new Error("Сервис вернул неожиданный ответ.");
    $("decision").textContent = result.accepted ? "Найдено возможное совпадение" : "0 совпадений";
    $("threshold").textContent = "Порог модели: " + Number(result.threshold).toFixed(4);
    if (result.decision_score != null) {
      $("threshold").textContent += " · лучший cosine в выбранном топе: " + Number(result.decision_score).toFixed(4);
    }
    document.querySelector(".results-note").textContent = !result.accepted
      ? "Ни один из кандидатов в выбранном топе не достиг порога косинусного сходства. Совпадений нет."
      : result.ranking_method && result.ranking_method !== "cosine"
        ? "Порядок — re-ranking YOLO; оценка на карточке — cosine, а не вероятность. Поэтому оценки могут идти не по убыванию."
        : "Оценка показывает визуальное сходство, а не вероятность совпадения.";
    for (const match of result.matches) {
      const card = document.createElement("article");
      card.className = "match-card";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "match-photo";
      button.setAttribute("aria-label", "Открыть фото: место " + match.rank);
      const img = document.createElement("img");
      img.alt = "Автомобиль — место " + match.rank;
      img.loading = "lazy";
      img.src = api("gallery/" + encodeURIComponent(match.image_id) + "/thumbnail");
      const rank = document.createElement("span");
      rank.className = "rank";
      rank.textContent = "#" + match.rank;
      const fallback = document.createElement("span");
      fallback.className = "photo-fallback";
      fallback.textContent = "Фото недоступно";
      fallback.hidden = true;
      img.addEventListener("error", () => {
        img.hidden = true; fallback.hidden = false; button.disabled = true;
      });
      button.append(img, rank, fallback);
      button.addEventListener("click", () => {
        $("large-photo").src = img.src;
        $("photo-caption").textContent = "Место " + match.rank + " · сходство " + Number(match.score).toFixed(4);
        $("photo-dialog").showModal();
      });
      const meta = document.createElement("div");
      meta.className = "match-meta";
      const label = document.createElement("span");
      label.className = "score-label";
      label.textContent = match.rank === 1 ? "ПЕРВЫЙ В РЕЙТИНГЕ" : "COSINE-СХОДСТВО";
      const score = document.createElement("strong");
      score.textContent = Number(match.score).toFixed(4);
      const id = document.createElement("span");
      id.className = "image-id";
      id.title = match.image_id;
      id.textContent = match.image_id;
      meta.append(label, score, id);
      card.append(button, meta);
      $("matches").append(card);
    }
    $("request-id").textContent = result.request_id;
    $("duration").textContent = response.headers.get("x-process-time-ms") ? response.headers.get("x-process-time-ms") + " мс" : "—";
    $("result-count").textContent = result.matches.length ? result.matches.length + " РЕЗУЛЬТАТОВ" : "0 СОВПАДЕНИЙ";
    $("result-data").hidden = false; $("result-empty").hidden = true;
  } catch (exception) {
    error(exception.name === "TimeoutError" ? "Ответ не получен за 45 секунд. Проверьте состояние сервиса и повторите позже." : (exception.message || "Проверьте соединение с сервером."));
    $("result-empty").hidden = false;
  } finally {setBusy(false);}
});
$("close-photo").addEventListener("click", () => $("photo-dialog").close());
$("photo-dialog").addEventListener("click", (event) => {
  if (event.target === $("photo-dialog")) $("photo-dialog").close();
});
render(); checkReady();

async function loadMetrics(info) {
  try {
    const response = await fetch(api("metrics"), {signal: AbortSignal.timeout(5000)});
    if (!response.ok) return;
    const data = await response.json(), report = data.report;
    if (!report) { $("model-metrics").hidden = true; return; }
    $("model-name").textContent = info.model_name;
    $("metric-cards").replaceChildren();
    const metrics = [
      ["mAP@10", report.holdout_mean["mAP@10"], "Качество первых 10 · re-ranking"],
      ["Rank-1", report.holdout_mean["Rank-1"], "Верная машина на первом месте"],
      ["Rank-5", report.holdout_mean["Rank-5"], "Верная машина в первой пятёрке"],
      ["F1 кандидатов", report.candidate_metrics.F1, "Cosine · правило отказа 25% в батче"]
    ];
    for (const [name, value, description] of metrics) {
      const card = document.createElement("div"); card.className = "metric-card";
      const label = document.createElement("span"); label.textContent = name;
      const number = document.createElement("strong"); number.textContent = (value * 100).toFixed(1) + "%";
      const bar = document.createElement("progress"); bar.max = 1; bar.value = value; bar.setAttribute("aria-label", name);
      const note = document.createElement("small"); note.textContent = description;
      card.append(label, number, bar, note); $("metric-cards").append(card);
    }
    $("metric-splits").replaceChildren();
    for (const split of report.holdout_splits) {
      const row = document.createElement("tr");
      for (const value of [String(split.seed), split.cosine_map10, split.rerank_map10, split.rank1, split.rank5]) {
        const cell = document.createElement("td"); cell.textContent = typeof value === "number" ? (value * 100).toFixed(1) + "%" : value; row.append(cell);
      }
      $("metric-splits").append(row);
    }
    $("threshold-policy").textContent = "Онлайн-порог зафиксирован на " + Number(info.threshold).toFixed(4) + " из тестового прогона коллег. Правило «отказать 25% запросов» онлайн не применяется; этот порог ещё требует калибровки для новых данных.";
    $("metrics-source").href = report.source_url;
    $("model-metrics").hidden = false;
  } catch { $("model-metrics").hidden = true; }
}

"use strict";
const $ = (id) => document.getElementById(id);
const names = ["x", "y", "w", "h"];
const state = {file: null, bitmap: null, ready: false, busy: false, generation: 0};
const form = $("search-form");
const context = $("canvas").getContext("2d");
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
  $("submit").disabled = state.busy || !state.ready || Boolean(box.error);
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
    $("status-text").textContent = "На связи · " + info.gallery_count + " изображений";
  } catch {
    state.ready = false; $("status-text").textContent = "Сервис недоступен";
  } finally {
    $("status-dot").classList.toggle("online", state.ready);
    $("refresh").disabled = false; render();
  }
}
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
  if (state.busy || !state.ready || box.error) {render(); return;}
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
    $("decision").textContent = result.accepted ? "Найдено возможное совпадение" : "Надёжного совпадения нет";
    $("threshold").textContent = "Порог модели: " + Number(result.threshold).toFixed(4);
    for (const match of result.matches) {
      const tr = document.createElement("tr");
      [String(match.rank).padStart(2, "0"), match.image_id, Number(match.score).toFixed(4)].forEach((value) => {
        const td = document.createElement("td"); td.textContent = value; tr.append(td);
      });
      $("matches").append(tr);
    }
    $("request-id").textContent = result.request_id;
    $("duration").textContent = response.headers.get("x-process-time-ms") ? response.headers.get("x-process-time-ms") + " мс" : "—";
    $("result-count").textContent = result.matches.length + " РЕЗУЛЬТАТОВ";
    $("result-data").hidden = false; $("result-empty").hidden = true;
  } catch (exception) {
    error(exception.name === "TimeoutError" ? "Ответ не получен за 45 секунд. Проверьте состояние сервиса и повторите позже." : (exception.message || "Проверьте соединение с сервером."));
    $("result-empty").hidden = false;
  } finally {setBusy(false);}
});
render(); checkReady();

"use strict";
const $ = (selector) => document.querySelector(selector);
let voices = [], selectedId = localStorage.getItem("giong-nha-voice"), filter = "all", designRender = null;
let toastTimer;
const activeJobs = {read: false, design: false};
const readModeHints = {
  fast: "Tạo nhanh hơn với 4 bước xử lý. Âm sắc và ngữ điệu có thể khác mẫu nhiều hơn.",
  balanced: "Giữ chi tiết giọng mẫu với 6 bước xử lý; phù hợp để đọc hằng ngày.",
  faithful: "Dùng đầy đủ giọng và lời mẫu với 10 bước xử lý; cần thêm thời gian.",
};
const readModeNames = {fast: "Nhanh", balanced: "Cân bằng", faithful: "Ưu tiên giống mẫu"};
function updateReadMode() {
  $("#read-mode-hint").textContent = readModeHints[$("#read-mode").value];
  localStorage.setItem("giong-nha-read-mode", $("#read-mode").value);
}
$("#read-mode").addEventListener("change", updateReadMode);
let reference = null, referenceUploading = false, asrBusy = false, transcriptionToken = 0;
let recorder = null, recordingStream = null, recordTimer = null, recordingStarted = 0, microphonePending = false;

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, error ? 10000 : 4500);
}

async function api(path, options = {}) {
  let response;
  try { response = await fetch(path, options); }
  catch { throw new Error("Mất kết nối với tool. Hãy kiểm tra ứng dụng đang chạy rồi thử lại."); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    let detail = data.detail;
    if (Array.isArray(detail)) detail = "Nội dung chưa hợp lệ. Hãy kiểm tra độ dài và thử lại.";
    throw new Error(detail || "Có lỗi xảy ra. Hãy thử lại.");
  }
  return data;
}

const post = (path, data) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
const duration = (seconds) => {
  const total = Math.floor(seconds);
  return `${Math.floor(total / 60)}:${(total % 60).toString().padStart(2, "0")}`;
};
const pause = (ms) => new Promise(resolve => setTimeout(resolve, ms));
const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0];

function showPage() {
  const page = ["read", "library", "create"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "read";
  document.querySelectorAll(".page").forEach(node => node.classList.toggle("active", node.id === `page-${page}`));
  document.querySelectorAll(".nav-item").forEach(node => {
    node.classList.toggle("active", node.dataset.page === page);
    if (node.dataset.page === page) node.setAttribute("aria-current", "page");
    else node.removeAttribute("aria-current");
  });
  document.title = `${{read: "Đọc văn bản", library: "Thư viện giọng", create: "Tạo giọng mới"}[page]} — Giọng Nhà`;
}

function renderSelected() {
  const target = $("#selected-voice");
  target.replaceChildren();
  const voice = voices.find(v => v.id === selectedId);
  $("#speak-button").disabled = !voice || activeJobs.read;
  if (!voice) {
    const p = document.createElement("p");
    p.textContent = "Chưa có giọng nào. Hãy tạo và lưu giọng đầu tiên trong mục Tạo giọng mới.";
    target.append(p);
    $("#voice-preview").hidden = true;
    return;
  }
  const avatar = document.createElement("div");
  avatar.className = "voice-avatar";
  avatar.innerHTML = '<svg><use href="#icon-wave"/></svg>';
  const copy = document.createElement("div"), title = document.createElement("h3"), badge = document.createElement("span"), desc = document.createElement("p");
  title.textContent = voice.name;
  badge.className = "voice-badge";
  badge.textContent = voice.origin === "preset" ? "Giọng có sẵn" : "Giọng của tôi";
  desc.textContent = voice.description;
  copy.append(title, badge, desc);
  target.append(avatar, copy);
  const audio = $("#voice-preview");
  audio.preload = "metadata";
  audio.hidden = false;
  if (audio.getAttribute("src") !== voice.audio_url) audio.src = voice.audio_url;
}

function chooseVoice(id) {
  selectedId = id;
  localStorage.setItem("giong-nha-voice", id);
  renderSelected();
  renderLibrary();
  location.hash = "read";
  toast(`Đã chọn giọng ${voices.find(v => v.id === id)?.name || ""}.`);
}

function renderLibrary() {
  const grid = $("#voice-grid"), search = $("#voice-search").value.trim().toLocaleLowerCase("vi");
  grid.replaceChildren();
  const visible = voices.filter(v => (filter === "all" || v.origin === filter) && `${v.name} ${v.description}`.toLocaleLowerCase("vi").includes(search));
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "library-empty";
    empty.textContent = voices.length ? "Chưa tìm thấy giọng phù hợp. Hãy thử tên khác hoặc đổi bộ lọc." : "Thư viện đang trống. Tạo và lưu giọng đầu tiên của bạn nhé.";
    grid.append(empty);
  }
  visible.forEach(voice => {
    const fragment = $("#voice-card-template").content.cloneNode(true), card = fragment.querySelector(".voice-card");
    card.classList.toggle("is-selected", voice.id === selectedId);
    card.querySelector(".voice-name").textContent = voice.name;
    card.querySelector(".voice-badge").textContent = voice.origin === "preset" ? "Có sẵn" : "Của tôi";
    card.querySelector(".voice-description").textContent = voice.description;
    card.querySelector("audio").src = voice.audio_url;
    card.querySelector("audio").setAttribute("aria-label", `Nghe thử giọng ${voice.name}`);
    card.querySelector(".voice-duration").textContent = `Mẫu ${duration(voice.duration)}`;
    const button = card.querySelector(".choose-voice");
    if (voice.id === selectedId) button.innerHTML = 'Đang dùng <svg><use href="#icon-check"/></svg>';
    button.setAttribute("aria-label", `Dùng giọng ${voice.name}`);
    button.addEventListener("click", () => chooseVoice(voice.id));
    grid.append(fragment);
  });
}

async function loadVoices() {
  voices = await api("/api/voices");
  if (!voices.some(v => v.id === selectedId)) selectedId = voices[0]?.id || null;
  if (selectedId) localStorage.setItem("giong-nha-voice", selectedId);
  $("#voice-count").textContent = voices.length;
  renderSelected();
  renderLibrary();
}

async function loadHistory() {
  const rows = await api("/api/history"), list = $("#history-list");
  list.replaceChildren();
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = "Bản đọc đầu tiên của bạn sẽ xuất hiện ở đây.";
    list.append(empty);
  }
  rows.slice(0, 8).forEach(row => {
    const item = document.createElement("article"), copy = document.createElement("div"), text = document.createElement("p"), detail = document.createElement("small"), audio = document.createElement("audio"), link = document.createElement("a");
    item.className = "history-row";
    copy.className = "history-copy";
    text.textContent = row.text;
    detail.textContent = `${row.voice_name} · ${duration(row.duration)} · ${new Date(row.created_at).toLocaleDateString("vi-VN")}`;
    audio.controls = true; audio.preload = "metadata"; audio.src = row.audio_url;
    audio.setAttribute("aria-label", `Nghe bản đọc: ${row.text.slice(0, 60)}`);
    link.href = row.download_url; link.download = ""; link.textContent = "Tải WAV";
    copy.append(text, detail); item.append(copy, audio, link); list.append(item);
  });
}

function setBusy(type, value) {
  activeJobs[type] = value;
  $(`#${type}-progress`).hidden = !value;
  if (type === "read") {
    $("#speak-button").disabled = value || !selectedId;
    $("#read-mode").disabled = value;
  }
  else {
    $("#design-button").disabled = value;
    if (value) { $("#design-empty").hidden = true; $("#design-result").hidden = true; }
    syncDesignControls();
  }
}

function showResult(type, result, updateSeed = true) {
  if (type === "read") {
    $("#read-result").hidden = false;
    $("#read-audio").src = result.audio_url;
    $("#read-download").href = result.download_url;
    $("#result-detail").textContent = `${result.voice_name} · ${duration(result.duration)} · WAV ${result.sample_rate / 1000} kHz`;
    if (result.processing_seconds != null) $("#result-detail").textContent += ` · Tạo trong ${result.processing_seconds.toLocaleString("vi-VN")} giây · ${readModeNames[result.generation?.read_mode] || ""}`;
    localStorage.setItem("giong-nha-last-read", result.id);
    loadHistory().catch(error => toast(error.message, true));
  } else {
    designRender = result;
    $("#design-empty").hidden = true;
    $("#design-result").hidden = false;
    $("#design-audio").src = result.audio_url;
    $("#design-download").href = result.download_url;
    $("#saved-message").hidden = true;
    $("#save-button").disabled = false;
    const settings = result.generation || {};
    $("#design-result-meta").textContent = `${result.sample_rate / 1000} kHz · ${duration(result.duration)} · CFG ${settings.cfg_value ?? 2} · ${settings.dit_steps ?? 10} bước · Seed thực tế: ${result.seed}`;
    if (updateSeed) $("#seed-input").value = result.seed;
    saveDesignDraft();
    localStorage.setItem("giong-nha-last-design", result.id);
  }
}

async function watchJob(type, id) {
  setBusy(type, true);
  localStorage.setItem(`giong-nha-pending-${type}`, id);
  let started = Date.now();
  try {
    for (;;) {
      const job = await api(`/api/jobs/${id}`);
      if (job.created_at) started = new Date(job.created_at).getTime();
      $(`#${type}-progress-text`).textContent = `${job.message} · ${Math.max(0, Math.floor((Date.now() - started) / 1000))} giây`;
      if (job.status === "failed") throw new Error(job.message);
      if (job.status === "completed") {
        showResult(type, job.result);
        toast(type === "read" ? "Đã tạo xong bản đọc của bạn." : "Giọng thử đã sẵn sàng. Hãy nghe và lưu nếu bạn ưng ý.");
        break;
      }
      await pause(1200);
    }
  } catch (error) {
    toast(error.message, true);
    if (type === "design") {
      if (designRender) showResult("design", designRender, false);
      else $("#design-empty").hidden = false;
    }
  } finally {
    localStorage.removeItem(`giong-nha-pending-${type}`);
    setBusy(type, false);
  }
}

$("#speak-button").addEventListener("click", async () => {
  const text = $("#read-text").value.trim();
  if (!text) { toast("Bạn chưa nhập nội dung cần đọc.", true); $("#read-text").focus(); return; }
  if (!selectedId) { toast("Hãy chọn hoặc tạo một giọng trước nhé.", true); return; }
  setBusy("read", true);
  try {
    const job = await post("/api/jobs", {kind: "speak", text, voice_id: selectedId, read_mode: $("#read-mode").value, seed: voices.find(v => v.id === selectedId)?.seed ?? 42});
    await watchJob("read", job.id);
  } catch (error) { setBusy("read", false); toast(error.message, true); }
});

$("#design-form").addEventListener("submit", async event => {
  event.preventDefault();
  const text = $("#sample-text").value.trim(), description = $("#voice-description").value.trim();
  if (!text) { toast("Hãy nhập văn bản cần đọc.", true); return; }
  if (referenceUploading || recorder) { toast("Hãy hoàn tất tải lên hoặc thu âm trước khi tạo giọng.", true); return; }
  const ultimate = $("#ultimate-mode").checked;
  if (ultimate && (!reference || !$("#reference-transcript").value.trim())) {
    toast("Hãy nhập hoặc nhận dạng lời của âm thanh mẫu trước khi tạo giọng.", true);
    $("#reference-transcript").focus(); return;
  }
  setBusy("design", true);
  try {
    const job = await post("/api/jobs", {
      kind: "design", text, description,
      reference_id: reference?.id || null, use_prompt_text: ultimate,
      prompt_text: ultimate ? $("#reference-transcript").value.trim() : "",
      cfg_value: Number($("#cfg-slider").value), dit_steps: Number($("#steps-slider").value),
      do_normalize: $("#normalize-text").checked, denoise: $("#denoise-reference").checked,
      random_seed: $("#random-seed").checked,
      seed: $("#seed-input").value === "" ? 42 : Number($("#seed-input").value),
    });
    await watchJob("design", job.id);
  } catch (error) {
    setBusy("design", false);
    if (designRender) showResult("design", designRender, false);
    else $("#design-empty").hidden = false;
    toast(error.message, true);
  }
});

$("#save-button").addEventListener("click", async () => {
  const name = $("#voice-name").value.trim();
  if (!name) { toast("Hãy đặt một cái tên cho giọng của bạn.", true); $("#voice-name").focus(); return; }
  if (!designRender) return;
  $("#save-button").disabled = true;
  try {
    const voice = await post("/api/voices", {render_id: designRender.id, name});
    selectedId = voice.id;
    localStorage.setItem("giong-nha-voice", selectedId);
    await loadVoices();
    const message = $("#saved-message");
    message.replaceChildren(document.createTextNode(`Đã lưu “${voice.name}”. `));
    const link = document.createElement("a"); link.href = "#read"; link.textContent = "Dùng giọng này để đọc văn bản →";
    message.append(link); message.hidden = false;
    toast("Đã lưu giọng vào thư viện. Bạn có thể dùng lại bất cứ lúc nào.");
  } catch (error) { $("#save-button").disabled = false; toast(error.message, true); }
});

$("#fill-example").addEventListener("click", () => {
  $("#read-text").value = "Đôi khi, hạnh phúc chỉ là một buổi sáng bình yên, một tách trà còn ấm và một câu chuyện được kể bằng giọng nói thân quen. Hãy dành cho mình một khoảng lặng, để lắng nghe những điều nhỏ bé quanh ta.";
  $("#read-text").dispatchEvent(new Event("input"));
});
$("#read-text").addEventListener("input", () => {
  $("#text-count").textContent = `${$("#read-text").value.length.toLocaleString("vi-VN")} / 6.000`;
  localStorage.setItem("giong-nha-text", $("#read-text").value);
});
document.querySelectorAll("[data-description]").forEach(button => button.addEventListener("click", () => { $("#voice-description").value = button.dataset.description; saveDesignDraft(); }));
document.querySelectorAll("[data-filter]").forEach(button => button.addEventListener("click", () => {
  filter = button.dataset.filter;
  document.querySelectorAll("[data-filter]").forEach(node => node.classList.toggle("active", node === button));
  renderLibrary();
}));
$("#voice-search").addEventListener("input", renderLibrary);
document.addEventListener("play", event => { if (event.target.tagName === "AUDIO") document.querySelectorAll("audio").forEach(audio => { if (audio !== event.target) audio.pause(); }); }, true);
window.addEventListener("hashchange", showPage);

async function init() {
  showPage();
  const savedReadMode = localStorage.getItem("giong-nha-read-mode");
  if (Object.hasOwn(readModeHints, savedReadMode)) $("#read-mode").value = savedReadMode;
  updateReadMode();
  await restoreDesignDraft();
  $("#read-text").value = localStorage.getItem("giong-nha-text") || "";
  $("#text-count").textContent = `${$("#read-text").value.length.toLocaleString("vi-VN")} / 6.000`;
  try { await Promise.all([loadVoices(), loadHistory()]); }
  catch (error) { toast(error.message, true); }
  for (const type of ["read", "design"]) {
    const pending = localStorage.getItem(`giong-nha-pending-${type}`);
    if (pending) { watchJob(type, pending); continue; }
    const last = localStorage.getItem(`giong-nha-last-${type}`);
    if (last) {
      try { const job = await api(`/api/jobs/${last}`); if (job.status === "completed") showResult(type, job.result, false); }
      catch { localStorage.removeItem(`giong-nha-last-${type}`); }
    }
  }
}
init();

function syncDesignControls() {
  const ultimate = $("#ultimate-mode").checked && Boolean(reference);
  $("#ultimate-mode").disabled = !reference || referenceUploading;
  $("#transcript-panel").hidden = !ultimate;
  $("#control-panel").hidden = ultimate;
  $("#voice-description").disabled = ultimate;
  $("#creation-mode").textContent = ultimate ? "Sao chép chi tiết" : reference ? "Sao chép có điều khiển" : "Thiết kế giọng";
  $("#seed-input").disabled = $("#random-seed").checked;
  $("#cfg-value").textContent = Number($("#cfg-slider").value).toFixed(1);
  $("#steps-value").textContent = $("#steps-slider").value;
  $("#sample-count").textContent = `${$("#sample-text").value.length.toLocaleString("vi-VN")} / 6.000`;
  $("#transcribe-reference").disabled = !reference || asrBusy || referenceUploading;
  $("#choose-reference").disabled = referenceUploading || Boolean(recorder) || microphonePending;
  $("#record-reference").disabled = referenceUploading || microphonePending;
  $("#design-button").disabled = activeJobs.design || referenceUploading || Boolean(recorder) || microphonePending || (ultimate && asrBusy);
}

function renderReference() {
  $("#reference-loaded").hidden = !reference;
  if (reference) {
    $("#reference-name").textContent = reference.name;
    $("#reference-detail").textContent = `${duration(reference.duration)} · Âm thanh mẫu mono 16 kHz`;
    $("#reference-audio").src = reference.audio_url;
  } else {
    $("#reference-audio").pause();
    $("#reference-audio").removeAttribute("src");
    $("#reference-audio").load();
  }
  syncDesignControls();
}

function saveDesignDraft() {
  const draft = {
    reference_id: reference?.id || null,
    description: $("#voice-description").value, text: $("#sample-text").value,
    prompt_text: $("#reference-transcript").value, ultimate: $("#ultimate-mode").checked,
    cfg: $("#cfg-slider").value, steps: $("#steps-slider").value,
    seed: $("#seed-input").value, random: $("#random-seed").checked,
    normalize: $("#normalize-text").checked, denoise: $("#denoise-reference").checked,
    language: $("#asr-language").value,
  };
  localStorage.setItem("giong-nha-create-draft", JSON.stringify(draft));
}

async function restoreDesignDraft() {
  let draft;
  try { draft = JSON.parse(localStorage.getItem("giong-nha-create-draft") || "null"); }
  catch { localStorage.removeItem("giong-nha-create-draft"); }
  if (draft && typeof draft === "object") {
    for (const [key, id] of Object.entries({description: "voice-description", text: "sample-text", prompt_text: "reference-transcript", cfg: "cfg-slider", steps: "steps-slider", seed: "seed-input", language: "asr-language"})) {
      if (draft[key] !== undefined) $(`#${id}`).value = draft[key];
    }
    for (const [key, id] of Object.entries({random: "random-seed", normalize: "normalize-text", denoise: "denoise-reference"})) {
      if (typeof draft[key] === "boolean") $(`#${id}`).checked = draft[key];
    }
    if (draft.reference_id) {
      try {
        reference = await api(`/api/references/${encodeURIComponent(draft.reference_id)}`);
        $("#ultimate-mode").checked = Boolean(draft.ultimate);
        if (!$("#reference-transcript").value) $("#reference-transcript").value = reference.transcript || "";
      } catch { toast("Âm thanh mẫu trước đó không còn. Hãy tải lên hoặc thu âm lại.", true); }
    }
  }
  renderReference();
}

async function uploadReference(file) {
  if (!file) return;
  if (file.size > 30 * 1024 * 1024) { toast("Âm thanh mẫu tối đa 30 MB.", true); return; }
  const shouldTranscribe = $("#ultimate-mode").checked;
  let uploaded = false;
  transcriptionToken += 1;
  asrBusy = false;
  referenceUploading = true;
  $("#reference-status").textContent = "Đang tải và xử lý âm thanh mẫu…";
  syncDesignControls();
  try {
    const form = new FormData(); form.append("file", file);
    reference = await api("/api/references", {method: "POST", body: form});
    uploaded = true;
    $("#reference-transcript").value = "";
    $("#transcript-status").textContent = "Bạn có thể nhận dạng tự động hoặc nhập đúng lời mẫu bằng tay.";
    $("#reference-status").textContent = "Đã thêm âm thanh mẫu. Bạn có thể sao chép giọng hoặc bật chế độ chi tiết.";
    renderReference();
    saveDesignDraft();
  } catch (error) {
    $("#reference-status").textContent = error.message;
    toast(error.message, true);
  } finally {
    referenceUploading = false;
    $("#reference-upload").value = "";
    syncDesignControls();
  }
  if (uploaded && shouldTranscribe && reference && $("#ultimate-mode").checked) transcribeReference();
}

async function transcribeReference() {
  if (!reference || asrBusy) return;
  const id = reference.id, token = ++transcriptionToken, originalDraft = $("#reference-transcript").value;
  asrBusy = true;
  $("#transcript-status").textContent = "Đang nhận dạng lời mẫu trên CPU… Bạn vẫn có thể nhập bằng tay.";
  syncDesignControls();
  try {
    const initial = await post(`/api/references/${id}/transcribe`, {language: $("#asr-language").value});
    while (token === transcriptionToken && reference?.id === id) {
      const job = await api(`/api/jobs/${initial.id}`);
      if (job.status === "failed") throw new Error(job.message);
      if (job.status === "completed") {
        reference.transcript = job.result.transcript;
        if ($("#reference-transcript").value === originalDraft) {
          $("#reference-transcript").value = job.result.transcript;
          $("#transcript-status").textContent = "Đã nhận dạng xong. Hãy kiểm tra và sửa lời mẫu cho khớp âm thanh.";
        } else {
          $("#transcript-status").textContent = "Đã nhận dạng xong. Tool giữ lại phần lời mẫu bạn đang tự sửa.";
        }
        saveDesignDraft();
        return;
      }
      await pause(900);
    }
  } catch (error) {
    if (token === transcriptionToken) {
      $("#transcript-status").textContent = `${error.message} Bạn có thể nhập lời mẫu bằng tay.`;
      toast(error.message, true);
    }
  } finally {
    if (token === transcriptionToken) { asrBusy = false; syncDesignControls(); }
  }
}

async function toggleRecording() {
  if (microphonePending) return;
  if (recorder) { recorder.stop(); return; }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    toast("Trình duyệt chưa hỗ trợ thu âm. Hãy dùng Chrome hoặc tải file âm thanh lên.", true); return;
  }
  microphonePending = true;
  syncDesignControls();
  try {
    recordingStream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    const mime = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find(type => MediaRecorder.isTypeSupported(type));
    recorder = new MediaRecorder(recordingStream, mime ? {mimeType: mime} : undefined);
    const current = recorder, chunks = [];
    recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
    recorder.onstop = () => {
      clearInterval(recordTimer);
      recordingStream.getTracks().forEach(track => track.stop());
      const mimeType = current.mimeType || "audio/webm", blob = new Blob(chunks, {type: mimeType});
      recorder = null; recordingStream = null;
      $("#record-reference").classList.remove("recording");
      $("#record-label").textContent = "Thu âm";
      $("#record-duration").hidden = true;
      syncDesignControls();
      if (blob.size) {
        const extension = mimeType.includes("mp4") ? "m4a" : mimeType.includes("ogg") ? "ogg" : "webm";
        uploadReference(new File([blob], `ban-thu-am.${extension}`, {type: mimeType}));
      }
    };
    recorder.start(); recordingStarted = Date.now(); microphonePending = false;
    $("#record-reference").classList.add("recording");
    $("#record-label").textContent = "Dừng thu";
    $("#record-duration").hidden = false; $("#record-duration").textContent = "0:00";
    $("#reference-status").textContent = "Đang thu âm… Bấm Dừng thu khi nói xong.";
    recordTimer = setInterval(() => {
      $("#record-duration").textContent = duration((Date.now() - recordingStarted) / 1000);
      if (Date.now() - recordingStarted >= 59000 && recorder?.state === "recording") recorder.stop();
    }, 500);
    syncDesignControls();
  } catch (error) {
    recordingStream?.getTracks().forEach(track => track.stop());
    recorder = null; recordingStream = null;
    microphonePending = false;
    toast(error.name === "NotAllowedError" ? "Bạn chưa cho phép dùng microphone. Hãy cấp quyền trong trình duyệt hoặc tải file lên." : "Chưa thu được âm thanh. Hãy kiểm tra microphone hoặc tải file lên.", true);
    syncDesignControls();
  }
}

$("#choose-reference").addEventListener("click", () => $("#reference-upload").click());
$("#reference-upload").addEventListener("change", event => uploadReference(event.target.files[0]));
$("#record-reference").addEventListener("click", toggleRecording);
$("#remove-reference").addEventListener("click", () => {
  transcriptionToken += 1; asrBusy = false; reference = null;
  $("#ultimate-mode").checked = false; $("#reference-transcript").value = "";
  $("#reference-status").textContent = "Đã bỏ mẫu. Bạn có thể tạo giọng mới từ mô tả.";
  renderReference(); saveDesignDraft();
});
$("#ultimate-mode").addEventListener("change", () => {
  syncDesignControls(); saveDesignDraft();
  if ($("#ultimate-mode").checked && reference && !$("#reference-transcript").value.trim()) transcribeReference();
});
$("#transcribe-reference").addEventListener("click", transcribeReference);
$("#design-form").addEventListener("input", () => { syncDesignControls(); saveDesignDraft(); });
$("#design-form").addEventListener("change", () => { syncDesignControls(); saveDesignDraft(); });
$("#reference-dropzone").addEventListener("dragover", event => { event.preventDefault(); $("#reference-dropzone").classList.add("dragging"); });
$("#reference-dropzone").addEventListener("dragleave", () => $("#reference-dropzone").classList.remove("dragging"));
$("#reference-dropzone").addEventListener("drop", event => {
  event.preventDefault(); $("#reference-dropzone").classList.remove("dragging");
  if (!referenceUploading && !recorder) uploadReference(event.dataTransfer.files[0]);
});
const instructionExamples = {
  gentle: {description: "A young girl with a soft, sweet voice. Speaks slowly with a melancholic, slightly tsundere tone.", text: "I never asked you to stay. It's not like I care or anything. But why does it still hurt so much now that you're gone?"},
  relaxed: {description: "Relaxed young male voice, slightly nasal, lazy drawl, very casual and chill.", text: "Dude, did you see that set? The waves out there are totally gnarly today. Just catching barrels all morning, it's like, totally righteous, you know what I mean?"},
  chinese: {description: "中年男性，低沉温暖，语速舒缓，像深夜电台主持人一样讲故事。", text: "夜深了，如果你还醒着，不妨听我讲一个关于远方的故事。没有什么比雨声和一个好故事更适合这样的夜晚了。"},
};
document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => {
  const example = instructionExamples[button.dataset.example];
  $("#ultimate-mode").checked = false;
  $("#voice-description").value = example.description;
  $("#sample-text").value = example.text;
  syncDesignControls(); saveDesignDraft();
  toast("Đã điền chỉ dẫn và nội dung mẫu. Bạn có thể sửa trước khi tạo.");
}));

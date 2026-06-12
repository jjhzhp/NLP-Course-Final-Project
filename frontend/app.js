const API_BASE = "/api";

// ---- Resizable layout ----
const LAYOUT_KEY = "course-rag-layout";
const layout = document.querySelector(".layout");
const MIN_LEFT_FR = 0.6;
const MIN_CHAT_FR = 1.5;
const MIN_RIGHT_FR = 0.55;

function readFrs() {
  const cs = getComputedStyle(layout);
  const parse = (name, fallback) => {
    const raw = (cs.getPropertyValue(name) || "").trim();
    const n = parseFloat(raw);
    return Number.isFinite(n) && n > 0 ? n : fallback;
  };
  return {
    left: parse("--left-fr", 1.6),
    chat: parse("--chat-fr", 4),
    right: parse("--right-fr", 1.4),
  };
}

function applyFrs(frs) {
  layout.style.setProperty("--left-fr", `${frs.left}fr`);
  layout.style.setProperty("--chat-fr", `${frs.chat}fr`);
  layout.style.setProperty("--right-fr", `${frs.right}fr`);
}

function loadLayoutPrefs() {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null");
    if (saved && saved.left && saved.chat && saved.right) applyFrs(saved);
  } catch {}
}
loadLayoutPrefs();

function flexibleCols() {
  const cols = layout.getBoundingClientRect().width;
  const rs = layout.querySelectorAll(".resizer");
  let resizerPx = 0;
  rs.forEach((r) => { resizerPx += r.getBoundingClientRect().width; });
  const padding = parseFloat(getComputedStyle(layout).paddingLeft) +
    parseFloat(getComputedStyle(layout).paddingRight);
  const colMargins = parseFloat(getComputedStyle(document.querySelector(".col-chat")).marginLeft) +
    parseFloat(getComputedStyle(document.querySelector(".col-chat")).marginRight);
  return Math.max(1, cols - resizerPx - padding - colMargins);
}

function attachResizer(resizer, target) {
  resizer.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    resizer.setPointerCapture(e.pointerId);
    resizer.classList.add("dragging");
    document.body.classList.add("resizing");

    const startX = e.clientX;
    const startFrs = readFrs();
    const totalFr = startFrs.left + startFrs.chat + startFrs.right;
    const totalPx = flexibleCols();
    const pxPerFr = totalPx / totalFr;

    function onMove(ev) {
      const dxFr = (ev.clientX - startX) / pxPerFr;
      const next = { ...startFrs };
      if (target === "left") {
        next.left = startFrs.left + dxFr;
        next.chat = startFrs.chat - dxFr;
      } else {
        next.chat = startFrs.chat + dxFr;
        next.right = startFrs.right - dxFr;
      }
      if (next.left < MIN_LEFT_FR || next.chat < MIN_CHAT_FR || next.right < MIN_RIGHT_FR) return;
      applyFrs(next);
    }

    function onUp() {
      resizer.removeEventListener("pointermove", onMove);
      resizer.removeEventListener("pointerup", onUp);
      resizer.removeEventListener("pointercancel", onUp);
      resizer.classList.remove("dragging");
      document.body.classList.remove("resizing");
      try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(readFrs())); } catch {}
    }

    resizer.addEventListener("pointermove", onMove);
    resizer.addEventListener("pointerup", onUp);
    resizer.addEventListener("pointercancel", onUp);
  });

  resizer.addEventListener("dblclick", () => {
    layout.style.removeProperty("--left-fr");
    layout.style.removeProperty("--chat-fr");
    layout.style.removeProperty("--right-fr");
    try { localStorage.removeItem(LAYOUT_KEY); } catch {}
  });
}

document.querySelectorAll(".resizer").forEach((r) => attachResizer(r, r.dataset.target));

// ---- Theme ----
const THEME_KEY = "course-rag-theme";
const themeToggle = document.getElementById("theme-toggle");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "夜" : "日";
}
applyTheme(localStorage.getItem(THEME_KEY) || "dark");
themeToggle.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
});

// ---- Health ----
async function checkHealth() {
  const dot = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error(res.statusText);
    dot.className = "health ok";
    text.textContent = "已连接";
  } catch (err) {
    dot.className = "health bad";
    text.textContent = "未连接";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

// ---- Chat ----
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatLog = document.getElementById("chat-log");
const chatSend = document.getElementById("chat-send");

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;
  const taskType = document.getElementById("task-type").value;
  const topK = parseInt(document.getElementById("chat-topk").value, 10) || 5;
  const usePro = document.getElementById("use-pro").checked;

  const tip = chatLog.querySelector(".empty-tip");
  if (tip) tip.remove();

  appendMessage("user", query);
  chatInput.value = "";
  chatSend.disabled = true;

  const agentMsg = appendMessage("agent", "");
  agentMsg.bubble.innerHTML = '<span class="spinner"></span> 思考中...';

  try {
    await streamChat({ query, task_type: taskType, use_pro_model: usePro, top_k: topK }, agentMsg);
  } catch (err) {
    agentMsg.bubble.textContent = `请求失败：${err.message}`;
  } finally {
    chatSend.disabled = false;
  }
});

function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "你" : "Agent";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return { wrap, meta, bubble };
}

if (window.marked && typeof window.marked.setOptions === "function") {
  window.marked.setOptions({ breaks: true, gfm: true });
}

function renderMarkdown(target, text) {
  if (window.marked && window.DOMPurify) {
    const html = window.marked.parse(text || "");
    target.innerHTML = window.DOMPurify.sanitize(html);
  } else {
    target.textContent = text || "";
  }
}

async function streamChat(payload, agentMsg) {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answerText = "";
  let firstDelta = true;
  let metaInfo = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";
    for (const evt of events) {
      const line = evt.trim();
      if (!line.startsWith("data:")) continue;
      const dataStr = line.slice(5).trim();
      if (!dataStr) continue;
      let event;
      try { event = JSON.parse(dataStr); } catch { continue; }

      if (event.type === "meta") {
        metaInfo = event;
        agentMsg.meta.innerHTML = renderMetaLine(event);
      } else if (event.type === "delta") {
        if (firstDelta) { agentMsg.bubble.textContent = ""; firstDelta = false; }
        answerText += event.text || "";
        renderMarkdown(agentMsg.bubble, answerText);
        chatLog.scrollTop = chatLog.scrollHeight;
      } else if (event.type === "error") {
        agentMsg.bubble.textContent = event.message || "发生错误";
        return;
      } else if (event.type === "done") {
        if (firstDelta) {
          renderMarkdown(agentMsg.bubble, answerText || "(无回复)");
        } else {
          renderMarkdown(agentMsg.bubble, answerText);
        }
        if (metaInfo && metaInfo.sources && metaInfo.sources.length) {
          agentMsg.bubble.appendChild(renderSources(metaInfo.sources));
        }
      }
    }
  }
}

function renderMetaLine(meta) {
  const conf = meta.confidence || "medium";
  const tag = `<span class="confidence-tag ${conf}">${conf}</span>`;
  const parts = [`Agent · ${meta.task_type || "?"}`, tag];
  if (meta.message) parts.push(`<span style="color:var(--warn)">· ${meta.message}</span>`);
  return parts.join(" ");
}

function renderSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";
  sources.forEach((s, i) => {
    const det = document.createElement("details");
    const sum = document.createElement("summary");
    const label = s.heading ? `${s.source_file} · ${s.heading}` : s.source_file;
    const score = (s.final_score ?? 0).toFixed(3);
    sum.textContent = `[${i + 1}] ${label}  (score ${score})`;
    const body = document.createElement("div");
    body.className = "src-text";
    body.textContent = s.text;
    det.appendChild(sum);
    det.appendChild(body);
    wrap.appendChild(det);
  });
  return wrap;
}

// ---- Documents ----
const ALLOWED_EXTS = ["pdf", "md", "markdown", "txt"];
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;
const fileInput = document.getElementById("file-input");
const uploadStatus = document.getElementById("upload-status");
const docList = document.getElementById("doc-list");
const docCount = document.getElementById("doc-count");
const refreshBtn = document.getElementById("refresh-docs");

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const rejected = [];
  for (const f of files) {
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      rejected.push(`${f.name}：不支持的格式 .${ext}`);
    } else if (f.size > MAX_UPLOAD_BYTES) {
      const mb = (f.size / 1024 / 1024).toFixed(1);
      rejected.push(`${f.name}：${mb} MB 超过 100 MB 上限`);
    }
  }
  if (rejected.length) {
    uploadStatus.className = "upload-status err";
    uploadStatus.textContent = `已拦截：${rejected.join("；")}`;
    fileInput.value = "";
    return;
  }

  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  uploadStatus.className = "upload-status";
  uploadStatus.innerHTML = `<span class="spinner"></span> 上传 ${files.length} 个文件...`;
  try {
    const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: fd });
    if (!res.ok) {
      let detail = `HTTP ${res.status}`;
      try { const errBody = await res.json(); if (errBody && errBody.detail) detail = errBody.detail; } catch {}
      throw new Error(detail);
    }
    const data = await res.json();
    uploadStatus.className = "upload-status ok";
    uploadStatus.textContent = `已上传 ${data.documents.length} 个文件`;
    fileInput.value = "";
    loadDocuments();
  } catch (err) {
    uploadStatus.className = "upload-status err";
    uploadStatus.textContent = `上传失败：${err.message}`;
  }
});

refreshBtn.addEventListener("click", loadDocuments);

async function loadDocuments() {
  docList.innerHTML = '<li class="empty-tip"><span class="spinner"></span> 加载中...</li>';
  try {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const docs = await res.json();
    renderDocuments(docs);
  } catch (err) {
    docList.innerHTML = `<li class="empty-tip" style="color:var(--danger)">加载失败：${err.message}</li>`;
  }
}

function renderDocuments(docs) {
  docCount.textContent = docs.length;
  if (!docs.length) {
    docList.innerHTML = '<li class="empty-tip">暂无文档</li>';
    return;
  }
  docList.innerHTML = "";
  docs.forEach((d) => {
    const li = document.createElement("li");
    li.className = "doc-item";
    li.innerHTML = `
      <div class="doc-name" title="${escapeHTML(d.filename)}">${escapeHTML(d.filename)}</div>
      <div class="doc-meta">
        <span>${d.file_type} · ${d.chunk_count} 块</span>
        <span class="right">
          <span class="status-pill ${d.status}">${d.status}</span>
          <button class="doc-del" data-id="${d.id}">删除</button>
        </span>
      </div>`;
    docList.appendChild(li);
  });
  docList.querySelectorAll("button.doc-del").forEach((btn) => {
    btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
  });
}

async function deleteDocument(id) {
  if (!confirm("确认删除该文档及其分块？")) return;
  try {
    const res = await fetch(`${API_BASE}/documents/${id}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadDocuments();
  } catch (err) {
    alert(`删除失败：${err.message}`);
  }
}

function escapeHTML(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

loadDocuments();

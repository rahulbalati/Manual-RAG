import { createPdfViewer } from "./pdf-viewer.js";

const messagesEl = document.getElementById("messages");
const welcomeEl = document.getElementById("welcome");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

const pdfViewer = createPdfViewer({
  pagesEl: document.getElementById("pdf-pages"),
  titleEl: document.getElementById("pdf-title"),
  metaEl: document.getElementById("pdf-meta"),
  pageLabelEl: document.getElementById("pdf-page-label"),
  emptyEl: document.getElementById("pdf-empty"),
  closeBtn: document.getElementById("pdf-close"),
  zoomInBtn: document.getElementById("pdf-zoom-in"),
  zoomOutBtn: document.getElementById("pdf-zoom-out"),
  zoomLabelEl: document.getElementById("pdf-zoom-label"),
  layoutEl: document.querySelector(".layout"),
  viewportEl: document.getElementById("pdf-viewport"),
});

let isStreaming = false;

function hideWelcome() {
  if (welcomeEl) welcomeEl.remove();
}

function createMessage(role) {
  hideWelcome();
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-role";
  label.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "message-body";

  wrapper.appendChild(label);
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;

  return { wrapper, body };
}

function setStatus(wrapper, text) {
  let status = wrapper.querySelector(".message-status");
  if (!status) {
    status = document.createElement("div");
    status.className = "message-status";
    wrapper.appendChild(status);
  }
  status.textContent = text;
}

function clearStatus(wrapper) {
  const status = wrapper.querySelector(".message-status");
  if (status) status.remove();
}

function addCursor(body) {
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  body.appendChild(cursor);
  return cursor;
}

function createSourceButton(label, source) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "source-link";
  button.textContent = label;
  button.addEventListener("click", () => {
    document.querySelectorAll(".source-link.active").forEach((el) => {
      el.classList.remove("active");
    });
    button.classList.add("active");
    pdfViewer.showSource(source);
  });
  return button;
}

function renderSources(wrapper, sources) {
  if (!sources) return;
  const chunks = sources.chunks || [];
  if (!chunks.length) return;

  const details = document.createElement("details");
  details.className = "sources";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = `Sources (${chunks.length} sections)`;
  details.appendChild(summary);

  const list = document.createElement("ul");
  list.className = "sources-list";

  for (const chunk of chunks) {
    const li = document.createElement("li");
    const label = `${chunk.title} (pages ${chunk.page_range})`;
    li.appendChild(createSourceButton(label, chunk));
    list.appendChild(li);
  }

  details.appendChild(list);
  wrapper.appendChild(details);
}

function parseSSE(buffer) {
  const events = [];
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() || "";

  for (const part of parts) {
    if (!part.trim()) continue;
    let event = "message";
    let data = "";
    for (const line of part.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      if (line.startsWith("data: ")) data = line.slice(6);
    }
    if (data) {
      try {
        events.push({ event, data: JSON.parse(data) });
      } catch {
        events.push({ event, data });
      }
    }
  }

  return { events, remainder };
}

async function streamAnswer(query) {
  const response = await fetch("/ask/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      chunk_k: 3,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(err || `Request failed (${response.status})`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  return {
    async *[Symbol.asyncIterator]() {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSSE(buffer);
        buffer = parsed.remainder;
        for (const evt of parsed.events) {
          yield evt;
        }
      }
      if (buffer.trim()) {
        const parsed = parseSSE(buffer + "\n\n");
        for (const evt of parsed.events) {
          yield evt;
        }
      }
    },
  };
}

async function handleSubmit(query) {
  const text = query.trim();
  if (!text || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;
  input.value = "";
  input.style.height = "auto";

  createMessage("user").body.textContent = text;
  const { wrapper, body } = createMessage("assistant");
  setStatus(wrapper, "Searching manual...");
  let cursor = addCursor(body);

  try {
    const stream = await streamAnswer(text);

    for await (const { event, data } of stream) {
      if (event === "status") {
        setStatus(wrapper, data.message);
      } else if (event === "sources") {
        renderSources(wrapper, data);
      } else if (event === "token") {
        clearStatus(wrapper);
        if (cursor && cursor.parentNode) cursor.remove();
        body.appendChild(document.createTextNode(data.content));
        messagesEl.scrollTop = messagesEl.scrollHeight;
        cursor = addCursor(body);
      } else if (event === "error") {
        clearStatus(wrapper);
        if (cursor && cursor.parentNode) cursor.remove();
        body.classList.add("error-text");
        body.textContent = data.message || "Something went wrong.";
      } else if (event === "done") {
        clearStatus(wrapper);
        if (cursor && cursor.parentNode) cursor.remove();
      }
    }
  } catch (err) {
    clearStatus(wrapper);
    if (cursor.parentNode) cursor.remove();
    body.classList.add("error-text");
    body.textContent = err.message || "Failed to get answer.";
  } finally {
    if (cursor && cursor.parentNode) cursor.remove();
    isStreaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  handleSubmit(input.value);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleSubmit(input.value);
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    handleSubmit(btn.textContent);
  });
});

input.focus();

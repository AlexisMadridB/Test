const chat = document.getElementById("chat");
const empty = document.getElementById("empty");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

const history = []; // historial de la conversación (se envía al servidor en cada pregunta)
let busy = false;

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// Mini renderizador de Markdown: bloques de código, `código`, **negrita**, listas y citas.
function renderMarkdown(src) {
  const blocks = [];
  let text = src.replace(/```\w*\n?([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`);
    return `\u0000${blocks.length - 1}\u0000`;
  });
  text = esc(text)
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\((Fuente:[^)]*)\)/g, '<mark class="cita">$1</mark>');

  let html = "";
  let list = null;
  const close = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const line of text.split("\n")) {
    let m;
    if (/^\u0000\d+\u0000$/.test(line.trim())) { close(); html += line.trim(); }
    else if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (list !== "ul") { close(); html += "<ul>"; list = "ul"; } html += `<li>${m[1]}</li>`; }
    else if ((m = line.match(/^\s*\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); html += "<ol>"; list = "ol"; } html += `<li>${m[1]}</li>`; }
    else if (line.trim() === "") close();
    else { close(); html += `<p>${line}</p>`; }
  }
  close();
  return html.replace(/\u0000(\d+)\u0000/g, (_, i) => blocks[i]);
}

function addMessage(kind, html) {
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.innerHTML = html;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function sourcesHtml(fuentes) {
  if (!fuentes || !fuentes.length) return "";
  const items = fuentes.map((f) => {
    const excerpt = f.text.length > 220 ? f.text.slice(0, 220) + "…" : f.text;
    return `<li><span class="name">${esc(f.source)}</span> <span class="meta">página ${f.page}, similitud ${f.score}</span><p class="excerpt">${esc(excerpt)}</p></li>`;
  }).join("");
  return `<details class="sources"><summary>Fragmentos consultados (${fuentes.length})</summary><ul>${items}</ul></details>`;
}

async function ask(question) {
  busy = true;
  sendBtn.disabled = true;
  addMessage("user", `<p>${esc(question)}</p>`);
  const pending = addMessage("bot", '<span class="typing" aria-label="Escribiendo"><span></span><span></span><span></span></span>');

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question, history: history.slice(-6) }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || "Algo salió mal. Intenta de nuevo.");

    pending.innerHTML = renderMarkdown(data.respuesta) + sourcesHtml(data.fuentes);
    history.push({ role: "user", content: question }, { role: "assistant", content: data.respuesta });
  } catch (err) {
    pending.classList.add("error");
    pending.innerHTML = `<p>${esc(err.message)}</p>`;
  } finally {
    busy = false;
    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || busy) return;
  input.value = "";
  input.style.height = "auto";
  ask(question);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 144) + "px";
});

let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let selectedMode = "fast";

const chatContainer = document.getElementById("chatContainer");
const textForm = document.getElementById("textForm");
const queryInput = document.getElementById("queryInput");
const micBtn = document.getElementById("micBtn");
const recordingStatus = document.getElementById("recordingStatus");
const btnModeFast = document.getElementById("btnModeFast");
const btnModeQuality = document.getElementById("btnModeQuality");
const currentModeLabel = document.getElementById("currentModeLabel");
const newChatBtn = document.getElementById("newChatBtn");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebar = document.getElementById("sidebar");

// Sidebar Toggle Mobile
if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
  });
}

// Mode Selection
btnModeFast.addEventListener("click", () => {
  selectedMode = "fast";
  btnModeFast.classList.add("active");
  btnModeQuality.classList.remove("active");
  currentModeLabel.innerText = "Voice RAG Assistant — FAST MODE";
});

btnModeQuality.addEventListener("click", () => {
  selectedMode = "quality";
  btnModeQuality.classList.add("active");
  btnModeFast.classList.remove("active");
  currentModeLabel.innerText = "Voice RAG Assistant — QUALITY MODE";
});

// New Chat Button
newChatBtn.addEventListener("click", () => {
  chatContainer.innerHTML = `
    <div class="gpt-welcome-screen">
      <div class="welcome-icon">✨</div>
      <h1>How can I help you today?</h1>
      <p class="welcome-subtitle">Ask any question in Tamil or English. Powered by low-latency vector retrieval & Indic AI models.</p>
      <div class="suggestion-grid">
        <button type="button" class="suggestion-card" data-query="இந்தியாவில் எத்தனை மாவட்டங்கள் உள்ளன?">
          <strong>இந்தியாவில் எத்தனை மாவட்டங்கள் உள்ளன?</strong>
          <small>Tamil • General Knowledge</small>
        </button>
        <button type="button" class="suggestion-card" data-query="இந்தியாவின் தலைநகரம் என்ன?">
          <strong>இந்தியாவின் தலைநகரம் என்ன?</strong>
          <small>Tamil • Geography</small>
        </button>
        <button type="button" class="suggestion-card" data-query="What is a corporation?">
          <strong>What is a corporation?</strong>
          <small>English • Definitions</small>
        </button>
        <button type="button" class="suggestion-card" data-query="இந்தியாவில் அதிகமாக பயிரிடப்படும் பயிர்கள் என்ன?">
          <strong>இந்தியாவில் அதிகமாக பயிரிடப்படும் பயிர்கள் என்ன?</strong>
          <small>Tamil • Agriculture</small>
        </button>
      </div>
    </div>
  `;
  bindSuggestionCards();
});

// Suggestion Cards Listener
function bindSuggestionCards() {
  const suggestionCards = document.querySelectorAll(".suggestion-card");
  suggestionCards.forEach(card => {
    card.addEventListener("click", () => {
      const q = card.getAttribute("data-query");
      if (q) submitQuery(q);
    });
  });
}
bindSuggestionCards();

// Textarea Auto-expand
queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    textForm.dispatchEvent(new Event("submit"));
  }
});

// Initialize Microphone
if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  micBtn.addEventListener("click", toggleRecording);
} else {
  recordingStatus.innerText = "Microphone access not supported in browser.";
  micBtn.disabled = true;
}

async function toggleRecording() {
  if (!isRecording) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.push(e.data);
      };

      mediaRecorder.onstop = handleAudioUpload;
      mediaRecorder.start();
      isRecording = true;
      micBtn.classList.add("recording");
      recordingStatus.innerText = "Listening... Tapping mic again sends audio query";
    } catch (err) {
      console.error("Mic access error:", err);
      recordingStatus.innerText = "Microphone access denied.";
    }
  } else {
    mediaRecorder.stop();
    isRecording = false;
    micBtn.classList.remove("recording");
    recordingStatus.innerText = "Processing voice query...";
  }
}

async function handleAudioUpload() {
  const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
  const formData = new FormData();
  formData.append("file", audioBlob, "user_query.wav");
  formData.append("mode", selectedMode);

  removeWelcomeScreen();
  appendUserRow("🎤 Voice Audio Query");
  const loadingRow = appendLoadingRow();

  try {
    const res = await fetch("/query/voice", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    chatContainer.removeChild(loadingRow);
    appendBotRow(data);
  } catch (err) {
    console.error("Voice API error:", err);
    chatContainer.removeChild(loadingRow);
    appendErrorRow("Failed to process voice query.");
  } finally {
    recordingStatus.innerText = "Click mic to speak in Tamil or English • Sub-50ms Fast QA Search Engine";
  }
}

// Form Submission
textForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = queryInput.value.trim();
  if (!q) return;

  queryInput.value = "";
  queryInput.style.height = "auto";
  submitQuery(q);
});

async function submitQuery(q) {
  removeWelcomeScreen();
  appendUserRow(q);
  const loadingRow = appendLoadingRow();

  try {
    const res = await fetch("/query/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        mode: selectedMode
      })
    });
    const data = await res.json();
    chatContainer.removeChild(loadingRow);
    appendBotRow(data);
  } catch (err) {
    console.error("Text API error:", err);
    chatContainer.removeChild(loadingRow);
    appendErrorRow("Failed to reach backend server.");
  }
}

function removeWelcomeScreen() {
  const welcome = chatContainer.querySelector(".gpt-welcome-screen");
  if (welcome) chatContainer.removeChild(welcome);
}

function appendUserRow(text) {
  const row = document.createElement("div");
  row.className = "gpt-message-row user-row";
  row.innerHTML = `
    <div class="gpt-message-content">
      <div class="msg-avatar user-avatar">U</div>
      <div class="msg-body">${escapeHtml(text)}</div>
    </div>
  `;
  chatContainer.appendChild(row);
  scrollToBottom();
}

function appendLoadingRow() {
  const row = document.createElement("div");
  row.className = "gpt-message-row bot-row";
  row.innerHTML = `
    <div class="gpt-message-content">
      <div class="msg-avatar bot-avatar-msg">🤖</div>
      <div class="msg-body">
        <span class="gpt-badge">THINKING</span>
        <div style="margin-top: 6px; color: #b4b4b4;">Searching knowledge base...</div>
      </div>
    </div>
  `;
  chatContainer.appendChild(row);
  scrollToBottom();
  return row;
}

function appendErrorRow(msg) {
  const row = document.createElement("div");
  row.className = "gpt-message-row bot-row";
  row.innerHTML = `
    <div class="gpt-message-content">
      <div class="msg-avatar bot-avatar-msg">🤖</div>
      <div class="msg-body">
        <span class="gpt-badge failed">ERROR</span>
        <div style="margin-top: 6px;">${msg}</div>
      </div>
    </div>
  `;
  chatContainer.appendChild(row);
  scrollToBottom();
}

function appendBotRow(data) {
  if (!data) return;

  const row = document.createElement("div");
  row.className = "gpt-message-row bot-row";

  const isGrounded = data.is_grounded;
  const groundedBadgeHtml = isGrounded 
    ? `<span class="gpt-badge">GROUNDED: YES</span>` 
    : `<span class="gpt-badge failed">GROUNDED: NO</span>`;

  const lats = data.latencies_ms || {};
  const totalMs = lats.total_latency_ms || 0;
  const modeStr = (data.mode || selectedMode).toUpperCase();
  const ansMode = (data.answer_mode || "generated").toUpperCase();

  const sources = data.sources || [];
  const sourcesHtml = sources.length === 0 
    ? `<p>No evidence passages retrieved.</p>`
    : sources.map(s => `
        <div class="gpt-source-card">
          <strong>#${s.rank} | Similarity: ${s.score}</strong> ${s.is_ground_truth ? '<span class="gpt-badge">[Ground Truth Evidence]</span>' : ''}<br>
          ${escapeHtml(s.text)}
        </div>
      `).join("");

  row.innerHTML = `
    <div class="gpt-message-content">
      <div class="msg-avatar bot-avatar-msg">🤖</div>
      <div class="msg-body">
        <div class="msg-badge-group">
          <span class="gpt-badge">${modeStr}</span>
          <span class="gpt-badge">ANSWER: ${ansMode}</span>
          ${groundedBadgeHtml}
        </div>
        
        <div class="answer-text">${escapeHtml(data.answer || "No response received.")}</div>

        <!-- Hidden Collapsible Analytics Drawer -->
        <details class="gpt-latency-drawer">
          <summary>⚙️ Technical Latency Analytics (${totalMs} ms) ▾</summary>
          <div class="drawer-details-box">
            <div style="margin-bottom: 6px;">
              <strong>Similarity Score:</strong> ${data.confidence || '0.00'} | <strong>Language:</strong> ${data.language || 'ta'}
            </div>
            <div class="gpt-latency-grid">
              <div class="gpt-latency-item"><span>STT</span><strong>${lats.stt_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>Query Proc</span><strong>${lats.query_proc_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>Embedding</span><strong>${lats.embedding_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>ANN Search</span><strong>${lats.dense_search_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>BM25</span><strong>${lats.sparse_search_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>Reranker</span><strong>${lats.rerank_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>Guardrail</span><strong>${lats.guardrail_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>Grounding</span><strong>${lats.grounding_ms || 0} ms</strong></div>
              <div class="gpt-latency-item"><span>LLM Gen</span><strong>${lats.generation_ms || 0} ms</strong></div>
            </div>
            <div style="margin-top: 8px;">
              <strong>Retrieved Evidence (${sources.length}):</strong>
              ${sourcesHtml}
            </div>
          </div>
        </details>
      </div>
    </div>
  `;

  chatContainer.appendChild(row);
  scrollToBottom();
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function escapeHtml(text) {
  if (!text) return "";
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

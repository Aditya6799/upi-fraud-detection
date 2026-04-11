/**
 * UPI Fraud Detection — Gemini Chatbot Client
 * Chat interface with auto-detection context
 */

let chatHistory = [];
let lastPrediction = null;

/* ═══ SEND CHAT MESSAGE ═══ */
async function sendChat() {
  const input = document.getElementById("chat-input");
  if (!input) return;
  const message = input.value.trim();
  if (!message) return;

  // Add user message
  addChatMessage("user", message);
  input.value = "";

  // Show typing indicator
  const typingId = addTypingIndicator();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        context: lastPrediction,
      }),
    });
    const data = await res.json();

    // Remove typing indicator
    removeTypingIndicator(typingId);

    if (data.success) {
      addChatMessage("bot", data.response);
    } else {
      addChatMessage("bot", "Sorry, I encountered an error. Please try again.");
    }
  } catch (err) {
    removeTypingIndicator(typingId);
    addChatMessage(
      "bot",
      "I'm unable to connect to the AI service right now. Please check that the backend server is running."
    );
  }
}

/* ═══ AUTO-SEND AFTER FRAUD DETECTION ═══ */
function autoSendToChatbot(prediction) {
  lastPrediction = prediction;

  let autoMessage = "";
  if (prediction.fraud_label === "FRAUD") {
    autoMessage = "Explain why this transaction is fraud and what should be done.";
  } else if (prediction.fraud_label === "SUSPICIOUS") {
    autoMessage = "Why is this transaction flagged as suspicious? What should the user watch out for?";
  } else {
    autoMessage = "This transaction was marked as safe. Can you confirm why and share any UPI safety tips?";
  }

  // Small delay for UX
  setTimeout(async () => {
    addChatMessage("user", autoMessage);
    const typingId = addTypingIndicator();

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: autoMessage,
          context: prediction,
        }),
      });
      const data = await res.json();
      removeTypingIndicator(typingId);

      if (data.success) {
        addChatMessage("bot", data.response);
      }
    } catch (err) {
      removeTypingIndicator(typingId);
      addChatMessage("bot", "Could not get AI explanation. The chatbot service may be unavailable.");
    }
  }, 800);
}

/* ═══ ADD CHAT MESSAGE ═══ */
function addChatMessage(role, content) {
  const container = document.getElementById("chat-messages");
  if (!container) return;

  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-message ${role}`;

  if (role === "bot") {
    // Parse basic markdown
    msgDiv.innerHTML = parseMarkdown(content);
  } else {
    msgDiv.textContent = content;
  }

  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;

  chatHistory.push({ role, content });
}

/* ═══ TYPING INDICATOR ═══ */
let typingCounter = 0;

function addTypingIndicator() {
  const container = document.getElementById("chat-messages");
  if (!container) return null;
  typingCounter++;
  const id = `typing-${typingCounter}`;

  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-message bot typing";
  msgDiv.id = id;
  msgDiv.innerHTML = `
    <div class="dot-pulse">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;
  container.appendChild(msgDiv);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  if (!id) return;
  const el = document.getElementById(id);
  if (el) el.remove();
}

/* ═══ BASIC MARKDOWN PARSER ═══ */
function parseMarkdown(text) {
  if (!text) return "";

  return text
    // Headers
    .replace(/^### (.+)$/gm, '<h4 style="margin:0.5rem 0 0.25rem">$1</h4>')
    .replace(/^## (.+)$/gm, '<h3>$1</h3>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    // Code
    .replace(/`(.+?)`/g, "<code>$1</code>")
    // Unordered lists
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>")
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
    // Line breaks
    .replace(/\n\n/g, "<br><br>")
    .replace(/\n/g, "<br>");
}

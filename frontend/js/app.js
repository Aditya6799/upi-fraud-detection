/**
 * UPI Fraud Detection — Core Application Logic
 * Theme toggling, dashboard stats, navigation
 */

const API_BASE = window.location.origin;

/* ═══ THEME TOGGLE ═══ */
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  html.setAttribute("data-theme", next);
  localStorage.setItem("upifd-theme", next);
  updateThemeUI(next);
}

function updateThemeUI(theme) {
  const icon = document.getElementById("theme-icon");
  const label = document.getElementById("theme-label");
  if (icon) icon.textContent = theme === "dark" ? "dark_mode" : "light_mode";
  if (label) label.textContent = theme === "dark" ? "Dark Mode" : "Light Mode";
}

// Initialize theme
(function initTheme() {
  const saved = localStorage.getItem("upifd-theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);
  document.addEventListener("DOMContentLoaded", () => updateThemeUI(saved));
})();


/* ═══ DASHBOARD STATS ═══ */
async function loadDashboardStats() {
  try {
    const res = await fetch(`${API_BASE}/api/transactions/stats`);
    const data = await res.json();
    if (data.success && data.stats) {
      const s = data.stats;
      animateValue("stat-total", s.total_transactions);
      animateValue("stat-fraud", s.fraud_detected);
      animateValue("stat-safe", s.safe_transactions);
      const rateEl = document.getElementById("stat-rate");
      if (rateEl) rateEl.textContent = s.detection_rate + "%";
    }
  } catch (err) {
    console.log("Stats API not available yet");
  }
}

function animateValue(elId, target) {
  const el = document.getElementById(elId);
  if (!el) return;
  const start = parseInt(el.textContent) || 0;
  const diff = target - start;
  if (diff === 0) { el.textContent = target; return; }
  const duration = 600;
  const startTime = performance.now();

  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
    el.textContent = Math.round(start + diff * eased);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// Load stats on page load
document.addEventListener("DOMContentLoaded", loadDashboardStats);


/* ═══ COLLAPSIBLE SECTIONS ═══ */
function toggleCollapsible(header) {
  const collapsible = header.parentElement;
  collapsible.classList.toggle("open");
}


/* ═══ STORE TRANSACTION ═══ */
async function storeTransaction(prediction, sender, receiver, amount) {
  try {
    await fetch(`${API_BASE}/api/transactions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sender_id: sender,
        receiver_id: receiver,
        amount: amount,
        fraud_status: prediction.fraud_label,
        risk_score: prediction.risk_score,
        fraud_probability: prediction.fraud_probability,
        reasons: prediction.reasons,
      }),
    });
    // Refresh dashboard stats
    loadDashboardStats();
  } catch (err) {
    console.error("Failed to store transaction:", err);
  }
}


/* ═══ UTILITIES ═══ */
function formatCurrency(amount) {
  return "₹" + parseFloat(amount).toLocaleString("en-IN");
}

function showAnalyzing() {
  document.getElementById("analyzing-overlay")?.classList.remove("hidden");
}

function hideAnalyzing() {
  document.getElementById("analyzing-overlay")?.classList.add("hidden");
}

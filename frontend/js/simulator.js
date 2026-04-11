/**
 * UPI Fraud Detection — Simulation Engine
 * Quick + Advanced simulation logic with mode presets
 */

/* ═══ SIMULATION MODE PRESETS ═══ */
const SIM_PRESETS = {
  normal: {
    amount: 1500,
    account_age: 730,
    avg_transaction: 1200,
    device_change_flag: "0",
    login_attempts: 1,
    is_night_flag: "0",
    transactions_last_1hr: 1,
    impossible_travel_flag: "0",
    geo_distance: 5,
    QR_flag: "0",
    OTP_flag: "1",
    hour_of_day: 14,
    type: "PAYMENT",
    device_type_encoded: "0",
  },
  suspicious: {
    amount: 75000,
    account_age: 30,
    avg_transaction: 8000,
    device_change_flag: "1",
    login_attempts: 3,
    is_night_flag: "1",
    transactions_last_1hr: 5,
    impossible_travel_flag: "0",
    geo_distance: 350,
    QR_flag: "1",
    OTP_flag: "0",
    hour_of_day: 23,
    type: "CASH_OUT",
    device_type_encoded: "2",
  },
  fraud: {
    amount: 250000,
    account_age: 7,
    avg_transaction: 2000,
    device_change_flag: "1",
    login_attempts: 6,
    is_night_flag: "1",
    transactions_last_1hr: 8,
    impossible_travel_flag: "1",
    geo_distance: 2500,
    QR_flag: "1",
    OTP_flag: "0",
    hour_of_day: 2,
    type: "CASH_OUT",
    device_type_encoded: "2",
  },
};

/* ═══ QUICK SIM — MODE SELECT ═══ */
function setSimMode(mode, btn) {
  // Update button styles
  document.querySelectorAll(".sim-mode-btn").forEach((b) => {
    b.className = "sim-mode-btn";
  });
  btn.classList.add(`active-${mode}`);

  // Auto-fill values
  const p = SIM_PRESETS[mode];
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  };

  setVal("qs-amount", p.amount);
  setVal("qs-account-age", p.account_age);
  setVal("qs-avg-tx", p.avg_transaction);
  setVal("qs-device-change", p.device_change_flag);
  setVal("qs-login", p.login_attempts);
  setVal("qs-night", p.is_night_flag);
}

/* ═══ QUICK SIMULATION ═══ */
async function runQuickSimulation(event) {
  event.preventDefault();
  showAnalyzing();

  const payload = {
    amount: parseFloat(document.getElementById("qs-amount").value),
    type: "TRANSFER",
    account_age: parseInt(document.getElementById("qs-account-age").value),
    avg_transaction: parseFloat(document.getElementById("qs-avg-tx").value),
    device_change_flag: parseInt(document.getElementById("qs-device-change").value),
    login_attempts: parseInt(document.getElementById("qs-login").value),
    is_night_flag: parseInt(document.getElementById("qs-night").value),
    transactions_last_1hr: 2,
    transactions_per_user: 10,
    OTP_flag: 1,
    QR_flag: 0,
    geo_distance: 10,
    impossible_travel_flag: 0,
    hour_of_day: new Date().getHours(),
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    hideAnalyzing();

    if (data.success || data.prediction) {
      const pred = data.prediction;
      displayQuickResult(pred);

      // Store transaction
      storeTransaction(
        pred,
        document.getElementById("qs-sender").value,
        document.getElementById("qs-receiver").value,
        payload.amount
      );

      // Show feature importance chart
      if (pred.feature_importance) {
        renderFeatureChart("feature-chart", pred.feature_importance);
        document.getElementById("feature-chart-card").style.display = "block";
      }

      // Auto-send to chatbot
      if (typeof autoSendToChatbot === "function") {
        autoSendToChatbot(pred);
      }
    } else {
      displayError(data.error || "Prediction failed");
    }
  } catch (err) {
    hideAnalyzing();
    displayError("Cannot reach prediction API. Is the ML server running?");
  }
}

/* ═══ DISPLAY QUICK RESULT ═══ */
function displayQuickResult(pred) {
  const panel = document.getElementById("result-panel");
  if (!panel) return;
  panel.classList.remove("hidden");

  const statusClass =
    pred.fraud_label === "FRAUD"
      ? "fraud"
      : pred.fraud_label === "SUSPICIOUS"
      ? "suspicious"
      : "safe";

  const statusIcon =
    pred.fraud_label === "FRAUD"
      ? "gpp_bad"
      : pred.fraud_label === "SUSPICIOUS"
      ? "warning"
      : "verified_user";

  panel.innerHTML = `
    <div class="result-status ${statusClass}">
      <div class="result-status-icon">
        <span class="material-symbols-outlined">${statusIcon}</span>
      </div>
      <div>
        <h3>${pred.fraud_label}</h3>
        <div class="body-sm">${
          pred.fraud_label === "FRAUD"
            ? "Transaction blocked — high fraud probability"
            : pred.fraud_label === "SUSPICIOUS"
            ? "Transaction flagged for review"
            : "Transaction appears safe"
        }</div>
      </div>
    </div>

    <div class="result-metrics">
      <div class="result-metric">
        <div class="value">${pred.risk_score}</div>
        <div class="label">Risk Score</div>
      </div>
      <div class="result-metric">
        <div class="value">${(pred.fraud_probability * 100).toFixed(1)}%</div>
        <div class="label">Fraud Probability</div>
      </div>
      <div class="result-metric">
        <div class="value">${pred.ml_probability != null ? (pred.ml_probability * 100).toFixed(1) + "%" : "N/A"}</div>
        <div class="label">ML Confidence</div>
      </div>
    </div>

    <div class="card-header" style="margin-bottom:0.5rem">
      <h3 style="font-size:0.875rem">Risk Factors</h3>
    </div>
    <ul class="reasons-list">
      ${pred.reasons.map((r) => `<li>${r}</li>`).join("")}
    </ul>
  `;
}

/* ═══ ADVANCED SIM — MODE SELECT ═══ */
function setAdvancedMode(mode, btn) {
  document.querySelectorAll(".sim-mode-btn").forEach((b) => {
    b.className = "sim-mode-btn";
  });
  btn.classList.add(`active-${mode}`);

  const p = SIM_PRESETS[mode];
  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  };

  setVal("adv-amount", p.amount);
  setVal("adv-account-age", p.account_age);
  setVal("adv-avg-tx", p.avg_transaction);
  setVal("adv-device-change", p.device_change_flag);
  setVal("adv-login", p.login_attempts);
  setVal("adv-night", p.is_night_flag);
  setVal("adv-tx-1hr", p.transactions_last_1hr);
  setVal("adv-impossible-travel", p.impossible_travel_flag);
  setVal("adv-geo-dist", p.geo_distance);
  setVal("adv-qr", p.QR_flag);
  setVal("adv-otp", p.OTP_flag);
  setVal("adv-hour", p.hour_of_day);
  setVal("adv-type", p.type);
  setVal("adv-device-type", p.device_type_encoded);
  setVal("adv-tx-freq", mode === "fraud" ? 3 : 10);
}

/* ═══ ADVANCED SIMULATION ═══ */
async function runAdvancedSimulation(event) {
  event.preventDefault();
  showAnalyzing();

  const gv = (id) => document.getElementById(id)?.value || "";

  const payload = {
    amount: parseFloat(gv("adv-amount")),
    type: gv("adv-type"),
    device_type_encoded: parseInt(gv("adv-device-type")),
    device_change_flag: parseInt(gv("adv-device-change")),
    geo_distance: parseFloat(gv("adv-geo-dist")),
    impossible_travel_flag: parseInt(gv("adv-impossible-travel")),
    hour_of_day: parseInt(gv("adv-hour")),
    is_night_flag: parseInt(gv("adv-night")),
    account_age: parseInt(gv("adv-account-age")),
    avg_transaction: parseFloat(gv("adv-avg-tx")),
    transactions_per_user: parseInt(gv("adv-tx-freq")),
    transactions_last_1hr: parseInt(gv("adv-tx-1hr")),
    OTP_flag: parseInt(gv("adv-otp")),
    QR_flag: parseInt(gv("adv-qr")),
    login_attempts: parseInt(gv("adv-login")),
  };

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    hideAnalyzing();

    if (data.success || data.prediction) {
      const pred = data.prediction;
      displayAdvancedResult(pred);

      // Store transaction
      storeTransaction(pred, gv("adv-sender"), gv("adv-receiver"), payload.amount);

      // Auto-send to chatbot for explanation
      if (typeof autoSendToChatbot === "function") {
        autoSendToChatbot(pred);
      }

      // Charts
      if (pred.feature_importance) {
        renderFeatureChart("adv-feature-chart", pred.feature_importance);
        document.getElementById("adv-feature-card").style.display = "block";
      }
      renderBreakdownChart(pred);
    } else {
      displayAdvancedError(data.error || "Prediction failed");
    }
  } catch (err) {
    hideAnalyzing();
    displayAdvancedError("Cannot reach prediction API. Is the ML server running?");
  }
}

/* ═══ DISPLAY ADVANCED RESULT ═══ */
function displayAdvancedResult(pred) {
  const panel = document.getElementById("adv-result-panel");
  if (!panel) return;

  const statusClass =
    pred.fraud_label === "FRAUD"
      ? "fraud"
      : pred.fraud_label === "SUSPICIOUS"
      ? "suspicious"
      : "safe";

  const statusIcon =
    pred.fraud_label === "FRAUD"
      ? "gpp_bad"
      : pred.fraud_label === "SUSPICIOUS"
      ? "warning"
      : "verified_user";

  // Gauge rotation: -90deg (0%) to 90deg (100%)
  const rotation = -90 + (pred.risk_score / 100) * 180;

  panel.innerHTML = `
    <div class="result-status ${statusClass}" style="margin-bottom:1.25rem">
      <div class="result-status-icon">
        <span class="material-symbols-outlined">${statusIcon}</span>
      </div>
      <div>
        <h3>${pred.fraud_label}</h3>
        <div class="body-sm">${
          pred.fraud_label === "FRAUD"
            ? "Transaction blocked — high fraud risk"
            : pred.fraud_label === "SUSPICIOUS"
            ? "Flagged for manual review"
            : "Transaction appears legitimate"
        }</div>
      </div>
    </div>

    <!-- Risk Gauge -->
    <div class="risk-gauge">
      <div class="risk-gauge-arc"></div>
      <div class="risk-gauge-needle" style="--rotation: ${rotation}deg"></div>
      <div class="risk-gauge-value">${pred.risk_score.toFixed(0)}</div>
    </div>
    <div class="text-center mb-2">
      <span class="label-sm">RISK SCORE</span>
    </div>

    <div class="result-metrics">
      <div class="result-metric">
        <div class="value">${(pred.fraud_probability * 100).toFixed(1)}%</div>
        <div class="label">Fraud Probability</div>
      </div>
      <div class="result-metric">
        <div class="value">${pred.ml_probability != null ? (pred.ml_probability * 100).toFixed(1) + "%" : "—"}</div>
        <div class="label">ML Score</div>
      </div>
      <div class="result-metric">
        <div class="value">${pred.anomaly_score != null ? pred.anomaly_score.toFixed(2) : "—"}</div>
        <div class="label">Anomaly Score</div>
      </div>
    </div>

    <div class="section-divider"></div>

    <div style="margin-bottom:0.5rem">
      <span class="label-md">Risk Factors</span>
    </div>
    <ul class="reasons-list">
      ${pred.reasons.map((r) => `<li>${r}</li>`).join("")}
    </ul>
  `;
}

/* ═══ ERROR DISPLAY ═══ */
function displayError(msg) {
  const panel = document.getElementById("result-panel");
  if (!panel) return;
  panel.classList.remove("hidden");
  panel.innerHTML = `
    <div class="result-status fraud">
      <div class="result-status-icon">
        <span class="material-symbols-outlined">error</span>
      </div>
      <div>
        <h3>Analysis Error</h3>
        <div class="body-sm">${msg}</div>
      </div>
    </div>
  `;
}

function displayAdvancedError(msg) {
  const panel = document.getElementById("adv-result-panel");
  if (!panel) return;
  panel.innerHTML = `
    <div class="result-status fraud">
      <div class="result-status-icon">
        <span class="material-symbols-outlined">error</span>
      </div>
      <div>
        <h3>Analysis Error</h3>
        <div class="body-sm">${msg}</div>
      </div>
    </div>
  `;
}

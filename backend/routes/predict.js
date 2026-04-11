/**
 * Predict Route — Proxies to Python ML API
 * POST /api/predict
 */

const express = require("express");
const router = express.Router();
const fetch = require("node-fetch");

const ML_API_URL = process.env.ML_API_URL || "http://localhost:5000";

router.post("/", async (req, res) => {
  try {
    const transactionData = req.body;

    if (!transactionData || !transactionData.amount) {
      return res.status(400).json({
        error: "Missing required field: amount",
        success: false,
      });
    }

    // Forward to Python ML API
    const mlResponse = await fetch(`${ML_API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(transactionData),
      timeout: 30000,
    });

    if (!mlResponse.ok) {
      const errorBody = await mlResponse.text();
      throw new Error(`ML API returned ${mlResponse.status}: ${errorBody}`);
    }

    const mlResult = await mlResponse.json();
    res.json(mlResult);
  } catch (error) {
    console.error("[Predict Error]", error.message);

    // Fallback: if ML API is down, use rule-based detection only
    if (error.code === "ECONNREFUSED" || error.type === "system") {
      const fallbackResult = ruleFallback(req.body);
      return res.json({
        success: true,
        prediction: fallbackResult,
        fallback: true,
        warning: "ML API unavailable — using rule-based detection only",
      });
    }

    res.status(503).json({
      error: "Prediction service unavailable",
      details: error.message,
      success: false,
    });
  }
});

/**
 * Rule-based fallback when ML API is unavailable.
 */
function ruleFallback(data) {
  const amount = parseFloat(data.amount) || 0;
  let score = 0;
  const reasons = [];

  if (amount > 100000) { score += 25; reasons.push("High transaction amount (>₹1,00,000)"); }
  else if (amount > 50000) { score += 15; reasons.push("Large transaction amount (>₹50,000)"); }

  if (parseInt(data.device_change_flag) === 1) { score += 20; reasons.push("Device change detected"); }
  if (parseInt(data.impossible_travel_flag) === 1) { score += 30; reasons.push("Impossible travel detected"); }
  if (parseInt(data.is_night_flag) === 1) { score += 10; reasons.push("Night-time transaction"); }
  if (parseInt(data.login_attempts) >= 4) { score += 20; reasons.push("Multiple login attempts"); }
  if (parseInt(data.transactions_last_1hr) > 5) { score += 20; reasons.push("High transaction velocity"); }
  if (parseInt(data.account_age) < 30 && amount > 25000) { score += 20; reasons.push("New account with large transaction"); }

  score = Math.min(score, 100);
  const probability = score / 100;

  let label = "SAFE";
  if (probability >= 0.6) label = "FRAUD";
  else if (probability >= 0.3) label = "SUSPICIOUS";

  if (reasons.length === 0) reasons.push("No risk factors detected");

  return {
    fraud_probability: probability,
    fraud_label: label,
    risk_score: score,
    reasons: reasons,
    ml_probability: null,
    anomaly_score: null,
    rule_score: probability,
    feature_importance: {},
  };
}

module.exports = router;

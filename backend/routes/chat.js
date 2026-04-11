/**
 * Chat Route — Google Gemini API Integration
 * POST /api/chat
 * Model: gemini-1.5-flash
 */

const express = require("express");
const router = express.Router();
const { GoogleGenerativeAI } = require("@google/generative-ai");

const SYSTEM_PROMPT = `You are a financial fraud detection expert specializing in UPI (Unified Payments Interface) transactions in India. Your role is to:

1. Explain why specific transactions are flagged as fraudulent or suspicious
2. Provide safe, accurate explanations about fraud detection methods
3. Give prevention tips for UPI fraud (phishing, OTP sharing, fake QR codes, account takeover, mule accounts)
4. Explain machine learning concepts used in fraud detection (Random Forest, Isolation Forest, SMOTE, etc.)
5. Help users understand risk scores and fraud probabilities
6. Discuss best practices for secure UPI transactions

Always provide responses that are:
- Accurate and based on real fraud detection methodologies
- Clear and understandable for non-technical users
- Actionable with specific prevention steps
- Professional and reassuring in tone

You may use examples, bullet points, and structured responses for clarity.
Never encourage or assist with committing fraud.`;

let genAI = null;

function getGenAI() {
  if (!genAI) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === "your_gemini_api_key_here") {
      return null;
    }
    genAI = new GoogleGenerativeAI(apiKey);
  }
  return genAI;
}

router.post("/", async (req, res) => {
  try {
    const { message, context } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const ai = getGenAI();
    if (!ai) {
      // Fallback response when API key is not configured
      return res.json({
        success: true,
        response: getFallbackResponse(message, context),
        source: "fallback",
      });
    }

    const model = ai.getGenerativeModel({
      model: "gemini-2.0-flash-lite",
      generationConfig: {
        temperature: 0.3,
        topP: 0.8,
        maxOutputTokens: 2048,
      },
    });

    // Build prompt with context
    let fullPrompt = SYSTEM_PROMPT + "\n\n";
    if (context) {
      fullPrompt += `Context from latest fraud analysis:\n${JSON.stringify(context, null, 2)}\n\n`;
    }
    fullPrompt += `User question: ${message}`;

    const result = await model.generateContent(fullPrompt);
    const responseText = result.response.text();

    res.json({
      success: true,
      response: responseText,
      source: "gemini",
    });
  } catch (error) {
    console.error("[Chat Error]", error.message);

    // Fallback for API errors
    res.json({
      success: true,
      response: getFallbackResponse(req.body.message, req.body.context),
      source: "fallback",
      warning: "Gemini API unavailable — using built-in responses",
    });
  }
});

/**
 * Built-in fallback responses when Gemini API is unavailable.
 */
function getFallbackResponse(message, context) {
  const msg = (message || "").toLowerCase();

  if (context && context.fraud_label) {
    const label = context.fraud_label;
    const score = context.risk_score || 0;
    const reasons = (context.reasons || []).join(", ");

    if (label === "FRAUD") {
      return `## ⚠️ Fraud Analysis\n\nThis transaction has been flagged as **FRAUD** with a risk score of **${score}/100**.\n\n**Key Risk Factors:**\n${(context.reasons || []).map(r => `- ${r}`).join("\n")}\n\n**Recommendations:**\n1. **Block** this transaction immediately\n2. **Verify** the sender's identity through an alternate channel\n3. **Report** to your bank's fraud department\n4. **Freeze** the account if account takeover is suspected\n5. File a complaint at https://cybercrime.gov.in\n\n**Why this was flagged:** The hybrid detection system combines machine learning predictions, anomaly detection, and rule-based analysis. Multiple risk signals were triggered simultaneously, indicating a high probability of fraud.`;
    } else if (label === "SUSPICIOUS") {
      return `## ⚡ Suspicious Activity\n\nThis transaction is marked **SUSPICIOUS** with a risk score of **${score}/100**.\n\n**Concerns:**\n${(context.reasons || []).map(r => `- ${r}`).join("\n")}\n\n**Recommendations:**\n1. **Hold** the transaction for manual review\n2. **Verify** transaction details with the account holder\n3. **Monitor** the account for additional suspicious activity\n4. Consider implementing additional authentication\n\nWhile not confirmed as fraud, the combination of risk signals warrants human review.`;
    } else {
      return `## ✅ Transaction Safe\n\nThis transaction appears **SAFE** with a risk score of **${score}/100**.\n\nNo significant risk factors were detected. The transaction follows normal behavioral patterns for this user profile.\n\n**UPI Safety Tips:**\n- Never share OTP with anyone\n- Verify QR codes before scanning\n- Use official UPI apps only\n- Enable transaction limits`;
    }
  }

  if (msg.includes("upi") && msg.includes("fraud")) {
    return "## Common UPI Fraud Types\n\n1. **Phishing Attacks** — Fake messages/calls asking for UPI PIN\n2. **QR Code Scams** — Malicious QR codes that debit money\n3. **OTP Sharing** — Social engineering to get OTP\n4. **Account Takeover** — Unauthorized access via SIM swap\n5. **Mule Accounts** — Using innocent accounts to move stolen funds\n\n**Prevention:** Never share PIN/OTP, verify payment requests, use official apps.";
  }

  if (msg.includes("how") && (msg.includes("detect") || msg.includes("work"))) {
    return "## How Our System Detects Fraud\n\nWe use a **Hybrid Detection Engine** combining:\n\n1. **Machine Learning (Random Forest)** — Trained on 200K+ transactions to recognize fraud patterns\n2. **Anomaly Detection (Isolation Forest)** — Identifies unusual transactions that deviate from normal behavior\n3. **Rule-Based Logic** — Domain-specific rules for UPI fraud (high amounts, device changes, impossible travel)\n\nThe system produces a **risk score (0-100)** by combining all three methods with weighted ensemble scoring.";
  }

  return "I'm your UPI Fraud Detection Assistant. I can help you understand:\n\n- **Fraud analysis results** — Why a transaction was flagged\n- **Prevention tips** — How to stay safe from UPI fraud\n- **Detection methods** — How our AI system works\n- **Risk scores** — What the numbers mean\n\nAsk me anything about UPI security!";
}

module.exports = router;

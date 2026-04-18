"""
UPI Fraud Detection - Unified Flask REST API
Serves frontend, ML predictions, Gemini Chat, and Supabase Transactions.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Feature specific imports
from predict import get_engine
import google.generativeai as genai

# Patch Supabase regex to accept newer formats (like sb_publishable)
import supabase._sync.client
import supabase._async.client
original_match = re.match
def patched_match(pattern, string, flags=0):
    if "A-Za-z0-9-_=" in pattern:
        return True # Bypass JWT validation
    return original_match(pattern, string, flags)
supabase._sync.client.re.match = patched_match
supabase._async.client.re.match = patched_match

from supabase import create_client, Client

# Initialize Flask
# Static folder points to frontend directory
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

MODEL_DIR = Path(__file__).parent / "models"
CHARTS_DIR = Path(__file__).parent / "charts"

# -----------------------------------------------------------------------------
# 1. SETUP CLOUD SERVICES
# -----------------------------------------------------------------------------
# Setup Gemini
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai_client = None
if gemini_api_key and gemini_api_key != "your_gemini_api_key_here":
    genai.configure(api_key=gemini_api_key)
    genai_client = True

# Setup Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = None
if supabase_url and supabase_key and supabase_url != "your_supabase_url_here":
    try:
        supabase = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Failed to initialize Supabase: {e}")

memory_store = [] # Fallback database

# -----------------------------------------------------------------------------
# 2. SYSTEM PROMPTS
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial fraud detection expert specializing in UPI (Unified Payments Interface) transactions in India. Your role is to:

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
Never encourage or assist with committing fraud."""

def get_fallback_response(message, context):
    msg = (message or "").lower()
    
    if context and context.get("fraud_label"):
        label = context.get("fraud_label")
        score = context.get("risk_score", 0)
        reasons = context.get("reasons", [])
        reasons_text = "\n".join([f"- {r}" for r in reasons])
        
        if label == "FRAUD":
            return f"## ⚠️ Fraud Analysis\n\nThis transaction has been flagged as **FRAUD** with a risk score of **{score}/100**.\n\n**Key Risk Factors:**\n{reasons_text}\n\n**Recommendations:**\n1. **Block** this transaction immediately\n2. **Verify** the sender's identity through an alternate channel\n3. **Report** to your bank's fraud department\n4. **Freeze** the account if account takeover is suspected\n5. File a complaint at https://cybercrime.gov.in\n\n**Why this was flagged:** The hybrid detection system combines machine learning predictions, anomaly detection, and rule-based analysis. Multiple risk signals were triggered simultaneously, indicating a high probability of fraud."
        elif label == "SUSPICIOUS":
            return f"## ⚡ Suspicious Activity\n\nThis transaction is marked **SUSPICIOUS** with a risk score of **{score}/100**.\n\n**Concerns:**\n{reasons_text}\n\n**Recommendations:**\n1. **Hold** the transaction for manual review\n2. **Verify** transaction details with the account holder\n3. **Monitor** the account for additional suspicious activity\n4. Consider implementing additional authentication\n\nWhile not confirmed as fraud, the combination of risk signals warrants human review."
        else:
            return f"## ✅ Transaction Safe\n\nThis transaction appears **SAFE** with a risk score of **{score}/100**.\n\nNo significant risk factors were detected. The transaction follows normal behavioral patterns for this user profile.\n\n**UPI Safety Tips:**\n- Never share OTP with anyone\n- Verify QR codes before scanning\n- Use official UPI apps only\n- Enable transaction limits"

    if "upi" in msg and "fraud" in msg:
        return "## Common UPI Fraud Types\n\n1. **Phishing Attacks** — Fake messages/calls asking for UPI PIN\n2. **QR Code Scams** — Malicious QR codes that debit money\n3. **OTP Sharing** — Social engineering to get OTP\n4. **Account Takeover** — Unauthorized access via SIM swap\n5. **Mule Accounts** — Using innocent accounts to move stolen funds\n\n**Prevention:** Never share PIN/OTP, verify payment requests, use official apps."

    if "how" in msg and ("detect" in msg or "work" in msg):
        return "## How Our System Detects Fraud\n\nWe use a **Hybrid Detection Engine** combining:\n\n1. **Machine Learning (Random Forest)** — Trained on 200K+ transactions to recognize fraud patterns\n2. **Anomaly Detection (Isolation Forest)** — Identifies unusual transactions that deviate from normal behavior\n3. **Rule-Based Logic** — Domain-specific rules for UPI fraud (high amounts, device changes, impossible travel)\n\nThe system produces a **risk score (0-100)** by combining all three methods with weighted ensemble scoring."

    return "I'm your UPI Fraud Detection Assistant. I can help you understand:\n\n- **Fraud analysis results** — Why a transaction was flagged\n- **Prevention tips** — How to stay safe from UPI fraud\n- **Detection methods** — How our AI system works\n- **Risk scores** — What the numbers mean\n\nAsk me anything about UPI security!"

# -----------------------------------------------------------------------------
# 3. ROUTES
# -----------------------------------------------------------------------------

@app.route("/", defaults={'path': ''})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serve static frontend application."""
    if path and (FRONTEND_DIR / path).exists():
        return send_from_directory(str(FRONTEND_DIR), path)
    return send_from_directory(str(FRONTEND_DIR), "index.html")

@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "UPI Fraud Detection Unified API"})

# Metrics and Charts backwards compatibility for frontend
@app.route("/api/metrics", methods=["GET"])
@app.route("/metrics", methods=["GET"])
def metrics():
    try:
        metrics_file = MODEL_DIR / "metrics.json"
        if not metrics_file.exists():
            return jsonify({"error": "No metrics found. Run train.py first."}), 404
        with open(metrics_file) as f:
            data = json.load(f)
        return jsonify({"success": True, "metrics": data})
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/ml/charts/<chart_name>", methods=["GET"])
@app.route("/charts/<chart_name>", methods=["GET"])
def get_chart(chart_name):
    chart_path = CHARTS_DIR / chart_name
    if chart_path.exists():
        return send_from_directory(str(CHARTS_DIR), chart_name)
    return jsonify({"error": "Chart not found"}), 404

# Core APIs migrated from Node.js
@app.route("/api/predict", methods=["POST"])
def predict_fraud():
    try:
        data = request.get_json()
        if not data or "amount" not in data:
            return jsonify({"error": "Missing required field: amount", "success": False}), 400

        engine = get_engine()
        result = engine.predict(data)
        return jsonify({"success": True, "prediction": result})
    except FileNotFoundError:
        return jsonify({"error": "Models not trained yet.", "success": False}), 503
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        req_data = request.get_json()
        message = req_data.get("message")
        context = req_data.get("context")

        if not message:
            return jsonify({"error": "Message is required", "success": False}), 400

        if not genai_client:
            return jsonify({
                "success": True, 
                "response": get_fallback_response(message, context),
                "source": "fallback"
            })

        model = genai.GenerativeModel('gemini-2.0-flash-lite',
            generation_config={"temperature": 0.3, "top_p": 0.8, "max_output_tokens": 2048}
        )

        full_prompt = SYSTEM_PROMPT + "\n\n"
        if context:
            full_prompt += f"Context from latest fraud analysis:\n{json.dumps(context, indent=2)}\n\n"
        full_prompt += f"User question: {message}"

        response = model.generate_content(full_prompt)
        
        return jsonify({
            "success": True,
            "response": response.text,
            "source": "gemini"
        })
    except Exception as e:
        print("[Chat Error]", str(e))
        return jsonify({
            "success": True,
            "response": get_fallback_response(request.get_json().get("message"), request.get_json().get("context")),
            "source": "fallback",
            "warning": "Gemini API unavailable or crashed - using fallback"
        })

@app.route("/api/transactions", methods=["GET", "POST"])
def transactions():
    if request.method == "GET":
        try:
            if supabase:
                result = supabase.table("transactions").select("*").order("created_at", desc=True).limit(100).execute()
                return jsonify({"success": True, "data": result.data or []})
            return jsonify({"success": True, "data": list(reversed(memory_store)), "source": "memory"})
        except Exception as e:
            print("[Transactions GET Error]", str(e))
            return jsonify({"error": str(e), "success": False}), 500
            
    elif request.method == "POST":
        try:
            data = request.get_json()
            record = {
                "sender_id": data.get("sender_id", "unknown"),
                "receiver_id": data.get("receiver_id", "unknown"),
                "amount": float(data.get("amount", 0)),
                "fraud_status": data.get("fraud_status", "SAFE"),
                "risk_score": float(data.get("risk_score", 0)),
                "fraud_probability": float(data.get("fraud_probability", 0)),
                "reasons": data.get("reasons", []),
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            if supabase:
                result = supabase.table("transactions").insert(record).execute()
                if result.data:
                    return jsonify({"success": True, "data": result.data[0]})
                
            record["id"] = len(memory_store) + 1
            memory_store.append(record)
            return jsonify({"success": True, "data": record, "source": "memory"})
        except Exception as e:
            print("[Transactions POST Error]", str(e))
            return jsonify({"error": str(e), "success": False}), 500

@app.route("/api/transactions/stats", methods=["GET"])
def transaction_stats():
    try:
        txs = []
        if supabase:
            result = supabase.table("transactions").select("*").execute()
            txs = result.data or []
        else:
            txs = memory_store

        total = len(txs)
        fraud = len([t for t in txs if t.get("fraud_status") == "FRAUD"])
        suspicious = len([t for t in txs if t.get("fraud_status") == "SUSPICIOUS"])
        safe = len([t for t in txs if t.get("fraud_status") == "SAFE"])
        
        detection_rate = round(((fraud + suspicious) / total * 100), 1) if total > 0 else 0
        avg_risk = round(sum(float(t.get("risk_score", 0)) for t in txs) / total, 1) if total > 0 else 0

        return jsonify({
            "success": True,
            "stats": {
                "total_transactions": total,
                "fraud_detected": fraud,
                "suspicious": suspicious,
                "safe_transactions": safe,
                "detection_rate": float(detection_rate),
                "avg_risk_score": float(avg_risk),
            }
        })
    except Exception as e:
        print("[Stats Error]", str(e))
        return jsonify({"error": str(e), "success": False}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print("=" * 60)
    print("  UPI Fraud Detection - Unified API Server")
    print(f"  Starting on http://localhost:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)

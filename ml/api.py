"""
UPI Fraud Detection - Flask REST API
Serves ML predictions and model metrics.
"""

import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import get_engine

app = Flask(__name__)
CORS(app)

MODEL_DIR = Path(__file__).parent / "models"
CHARTS_DIR = Path(__file__).parent / "charts"


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "UPI Fraud Detection ML API"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict fraud for a transaction.
    Accepts JSON with transaction features.
    Returns hybrid fraud analysis.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        engine = get_engine()
        result = engine.predict(data)

        return jsonify({
            "success": True,
            "prediction": result,
        })
    except FileNotFoundError:
        return jsonify({
            "error": "Models not trained yet. Run train.py first.",
        }), 503
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False,
        }), 500


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return model evaluation metrics."""
    try:
        metrics_file = MODEL_DIR / "metrics.json"
        if not metrics_file.exists():
            return jsonify({"error": "No metrics found. Run train.py first."}), 404

        with open(metrics_file) as f:
            data = json.load(f)
        return jsonify({"success": True, "metrics": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/charts/<chart_name>", methods=["GET"])
def get_chart(chart_name):
    """Serve chart images."""
    from flask import send_from_directory
    chart_path = CHARTS_DIR / chart_name
    if chart_path.exists():
        return send_from_directory(str(CHARTS_DIR), chart_name)
    return jsonify({"error": "Chart not found"}), 404


if __name__ == "__main__":
    print("=" * 50)
    print("  UPI Fraud Detection - ML API")
    print("  Starting on http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)

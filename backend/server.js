/**
 * UPI Fraud Detection - Express Backend Server
 * Routes: /api/predict, /api/chat, /api/transactions
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const path = require("path");

const predictRoutes = require("./routes/predict");
const chatRoutes = require("./routes/chat");
const transactionRoutes = require("./routes/transactions");

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Serve static frontend files
app.use(express.static(path.join(__dirname, "..", "frontend")));

// Serve ML chart images
app.use("/ml/charts", express.static(path.join(__dirname, "..", "ml", "charts")));

// API Routes
app.use("/api/predict", predictRoutes);
app.use("/api/chat", chatRoutes);
app.use("/api/transactions", transactionRoutes);

// ML metrics proxy
app.get("/api/metrics", async (req, res) => {
  try {
    const fetch = require("node-fetch");
    const mlUrl = process.env.ML_API_URL || "http://localhost:5000";
    const response = await fetch(`${mlUrl}/metrics`);
    const data = await response.json();
    res.json(data);
  } catch (error) {
    res.status(503).json({ error: "ML API unavailable", details: error.message });
  }
});

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    service: "UPI Fraud Detection Backend",
    timestamp: new Date().toISOString(),
  });
});

// SPA fallback — serve index.html for all unmatched routes
app.get("*", (req, res) => {
  res.sendFile(path.join(__dirname, "..", "frontend", "index.html"));
});

// Start server
app.listen(PORT, () => {
  console.log("═".repeat(50));
  console.log("  UPI Fraud Detection — Backend Server");
  console.log(`  → http://localhost:${PORT}`);
  console.log("  Routes:");
  console.log("    POST /api/predict       → ML prediction");
  console.log("    POST /api/chat          → Gemini chatbot");
  console.log("    GET  /api/transactions  → Transaction history");
  console.log("    GET  /api/metrics       → Model metrics");
  console.log("═".repeat(50));
});

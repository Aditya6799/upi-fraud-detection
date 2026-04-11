/**
 * Transactions Route — Supabase Integration
 * GET  /api/transactions        → list all
 * POST /api/transactions        → store new
 * GET  /api/transactions/stats  → dashboard metrics
 */

const express = require("express");
const router = express.Router();
const { createClient } = require("@supabase/supabase-js");

let supabase = null;

function getSupabase() {
  if (!supabase) {
    const url = process.env.SUPABASE_URL;
    const key = process.env.SUPABASE_ANON_KEY;
    if (!url || url === "your_supabase_url_here" || !key || key === "your_supabase_anon_key_here") {
      return null;
    }
    supabase = createClient(url, key);
  }
  return supabase;
}

// In-memory fallback store when Supabase is not configured
const memoryStore = [];

/**
 * GET /api/transactions — Fetch all transactions
 */
router.get("/", async (req, res) => {
  try {
    const sb = getSupabase();
    if (sb) {
      const { data, error } = await sb
        .from("transactions")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(100);

      if (error) throw error;
      return res.json({ success: true, data: data || [] });
    }

    // In-memory fallback
    return res.json({
      success: true,
      data: memoryStore.slice().reverse(),
      source: "memory",
    });
  } catch (error) {
    console.error("[Transactions GET Error]", error.message);
    res.status(500).json({ error: error.message, success: false });
  }
});

/**
 * POST /api/transactions — Store a new transaction result
 */
router.post("/", async (req, res) => {
  try {
    const { sender_id, receiver_id, amount, fraud_status, risk_score, fraud_probability, reasons } = req.body;

    const record = {
      sender_id: sender_id || "unknown",
      receiver_id: receiver_id || "unknown",
      amount: parseFloat(amount) || 0,
      fraud_status: fraud_status || "SAFE",
      risk_score: parseFloat(risk_score) || 0,
      fraud_probability: parseFloat(fraud_probability) || 0,
      reasons: reasons || [],
      created_at: new Date().toISOString(),
    };

    const sb = getSupabase();
    if (sb) {
      const { data, error } = await sb.from("transactions").insert([record]).select();
      if (error) throw error;
      return res.json({ success: true, data: data[0] });
    }

    // In-memory fallback
    record.id = memoryStore.length + 1;
    memoryStore.push(record);
    return res.json({ success: true, data: record, source: "memory" });
  } catch (error) {
    console.error("[Transactions POST Error]", error.message);
    res.status(500).json({ error: error.message, success: false });
  }
});

/**
 * GET /api/transactions/stats — Dashboard metrics
 */
router.get("/stats", async (req, res) => {
  try {
    let transactions = [];
    const sb = getSupabase();

    if (sb) {
      const { data, error } = await sb.from("transactions").select("*");
      if (error) throw error;
      transactions = data || [];
    } else {
      transactions = memoryStore;
    }

    const total = transactions.length;
    const fraud = transactions.filter(t => t.fraud_status === "FRAUD").length;
    const suspicious = transactions.filter(t => t.fraud_status === "SUSPICIOUS").length;
    const safe = transactions.filter(t => t.fraud_status === "SAFE").length;
    const detectionRate = total > 0 ? ((fraud + suspicious) / total * 100).toFixed(1) : 0;
    const avgRisk = total > 0
      ? (transactions.reduce((sum, t) => sum + (parseFloat(t.risk_score) || 0), 0) / total).toFixed(1)
      : 0;

    res.json({
      success: true,
      stats: {
        total_transactions: total,
        fraud_detected: fraud,
        suspicious: suspicious,
        safe_transactions: safe,
        detection_rate: parseFloat(detectionRate),
        avg_risk_score: parseFloat(avgRisk),
      },
    });
  } catch (error) {
    console.error("[Stats Error]", error.message);
    res.status(500).json({ error: error.message, success: false });
  }
});

module.exports = router;

/*
 * SUPABASE TABLE SCHEMA — Run this SQL in your Supabase SQL editor:
 *
 * CREATE TABLE transactions (
 *   id BIGSERIAL PRIMARY KEY,
 *   sender_id TEXT NOT NULL,
 *   receiver_id TEXT NOT NULL,
 *   amount DECIMAL(15,2) NOT NULL,
 *   fraud_status TEXT NOT NULL DEFAULT 'SAFE',
 *   risk_score DECIMAL(5,1) DEFAULT 0,
 *   fraud_probability DECIMAL(5,4) DEFAULT 0,
 *   reasons JSONB DEFAULT '[]'::jsonb,
 *   created_at TIMESTAMPTZ DEFAULT NOW()
 * );
 *
 * -- Enable Row Level Security (optional)
 * ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
 *
 * -- Public read/write policy for development
 * CREATE POLICY "Allow all" ON transactions FOR ALL USING (true) WITH CHECK (true);
 */

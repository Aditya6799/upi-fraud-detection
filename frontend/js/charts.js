/**
 * UPI Fraud Detection — Chart.js Visualizations
 * Feature importance, score breakdown, risk gauge
 */

/* Chart.js global defaults */
if (typeof Chart !== "undefined") {
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
}

/* Track chart instances for cleanup */
const chartInstances = {};

function destroyChart(canvasId) {
  if (chartInstances[canvasId]) {
    chartInstances[canvasId].destroy();
    delete chartInstances[canvasId];
  }
}

/* ═══ FEATURE IMPORTANCE BAR CHART ═══ */
function renderFeatureChart(canvasId, featureImportance) {
  if (typeof Chart === "undefined") return;
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const labels = Object.keys(featureImportance);
  const values = Object.values(featureImportance);

  // Get CSS variable colors
  const rootStyle = getComputedStyle(document.documentElement);
  const primaryColor = rootStyle.getPropertyValue("--primary").trim() || "#6366f1";

  const colors = values.map((v, i) => {
    const opacity = 0.4 + (v / Math.max(...values)) * 0.6;
    return `${primaryColor}${Math.round(opacity * 255).toString(16).padStart(2, "0")}`;
  });

  chartInstances[canvasId] = new Chart(canvas, {
    type: "bar",
    data: {
      labels: labels.map((l) => l.replace(/_/g, " ")),
      datasets: [
        {
          label: "Importance",
          data: values,
          backgroundColor: colors,
          borderColor: primaryColor,
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: rootStyle.getPropertyValue("--bg-card").trim() || "#fff",
          titleColor: rootStyle.getPropertyValue("--text-primary").trim() || "#131b2e",
          bodyColor: rootStyle.getPropertyValue("--text-secondary").trim() || "#464554",
          borderColor: rootStyle.getPropertyValue("--outline").trim() || "#e5e7eb",
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          callbacks: {
            label: (ctx) => `Importance: ${(ctx.parsed.x * 100).toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            color: rootStyle.getPropertyValue("--text-muted").trim(),
            callback: (v) => (v * 100).toFixed(0) + "%",
          },
        },
        y: {
          grid: { display: false },
          ticks: {
            color: rootStyle.getPropertyValue("--text-secondary").trim(),
            font: { size: 11 },
          },
        },
      },
    },
  });

  canvas.parentElement.style.height = `${Math.max(labels.length * 36, 200)}px`;
}

/* ═══ SCORE BREAKDOWN DOUGHNUT ═══ */
function renderBreakdownChart(prediction) {
  if (typeof Chart === "undefined") return;
  const canvasId = "adv-breakdown-chart";
  destroyChart(canvasId);

  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const card = document.getElementById("adv-breakdown-card");
  if (card) card.style.display = "block";

  const rootStyle = getComputedStyle(document.documentElement);

  const mlScore = prediction.ml_probability != null ? prediction.ml_probability : 0;
  const anomalyScore = prediction.anomaly_score != null ? prediction.anomaly_score : 0;
  const ruleScore = prediction.rule_score != null ? prediction.rule_score : 0;

  chartInstances[canvasId] = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["ML Model (50%)", "Anomaly Detection (20%)", "Rule Engine (30%)"],
      datasets: [
        {
          data: [mlScore * 50, anomalyScore * 20, ruleScore * 30],
          backgroundColor: [
            rootStyle.getPropertyValue("--primary").trim() || "#6366f1",
            rootStyle.getPropertyValue("--warning").trim() || "#f59e0b",
            rootStyle.getPropertyValue("--danger").trim() || "#ef4444",
          ],
          borderWidth: 0,
          spacing: 4,
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            padding: 16,
            color: rootStyle.getPropertyValue("--text-secondary").trim(),
            usePointStyle: true,
            pointStyleWidth: 10,
            font: { size: 11 },
          },
        },
        tooltip: {
          backgroundColor: rootStyle.getPropertyValue("--bg-card").trim() || "#fff",
          titleColor: rootStyle.getPropertyValue("--text-primary").trim(),
          bodyColor: rootStyle.getPropertyValue("--text-secondary").trim(),
          borderColor: rootStyle.getPropertyValue("--outline").trim(),
          borderWidth: 1,
          cornerRadius: 8,
          padding: 12,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.parsed.toFixed(1)} pts`,
          },
        },
      },
    },
  });
}

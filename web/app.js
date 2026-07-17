const API_BASE = "/api/v1";
let lastFeatures = null;
let lastExplain = null;

const FEATURE_LABELS = {
  Age: "Tuổi",
  BMI: "BMI",
  ASA_Score: "Điểm ASA",
  Has_Diabetes: "Tiểu đường",
  Has_HTN: "Cao huyết áp",
  Surgery_Type: "Tính chất mổ",
  PreOp_WBC: "Bạch cầu trước mổ",
  PreOp_Albumin: "Albumin máu trước mổ",
};

const form = document.getElementById("predict-form");
const explainBtn = document.getElementById("explain-btn");
const batchBtn = document.getElementById("batch-btn");
const resultCard = document.getElementById("result-card");
const explainSection = document.getElementById("explain-section");
const toggleTableBtn = document.getElementById("toggle-table");
const shapTable = document.getElementById("shap-table");
const topFactors = document.getElementById("top-factors");
const topFactorsError = document.getElementById("top-factors-error");
const factorList = document.getElementById("factor-list");

function readFormFeatures() {
  const fd = new FormData(form);
  const features = {
    Age: Number(fd.get("Age")),
    BMI: Number(fd.get("BMI")),
    ASA_Score: fd.get("ASA_Score"),
    Has_Diabetes: fd.get("Has_Diabetes") ? 1 : 0,
    Has_HTN: fd.get("Has_HTN") ? 1 : 0,
    Surgery_Type: fd.get("Surgery_Type"),
    PreOp_WBC: fd.get("PreOp_WBC") ? Number(fd.get("PreOp_WBC")) : null,
    PreOp_Albumin: fd.get("PreOp_Albumin") ? Number(fd.get("PreOp_Albumin")) : null,
  };
  return features;
}

function riskTier(p) {
  if (p >= 0.7) return { key: "critical", label: "Nguy cơ cao" };
  if (p >= 0.3) return { key: "warning", label: "Nguy cơ trung bình" };
  return { key: "good", label: "Nguy cơ thấp" };
}

function renderPrediction(result) {
  resultCard.hidden = false;
  const pct = result.risk_probability * 100;
  document.getElementById("risk-value").textContent = pct.toFixed(1) + "%";

  const tier = riskTier(result.risk_probability);
  const fill = document.getElementById("meter-fill");
  fill.style.width = Math.max(pct, 2) + "%";
  fill.style.backgroundColor = `var(--status-${tier.key})`;
  fill.parentElement.style.backgroundColor = `var(--meter-track-${tier.key})`;

  const badge = document.getElementById("risk-badge");
  badge.textContent = (result.high_risk_flag ? "⚠ " : "✓ ") + tier.label;
  badge.className = "status-badge " + tier.key;

  document.getElementById("model-meta").textContent =
    `Model: ${result.model_version} · ${new Date(result.predicted_at).toLocaleString("vi-VN")}`;

  explainBtn.disabled = false;
  explainSection.hidden = true;
}

function renderTopFactors(result) {
  topFactorsError.hidden = true;
  topFactors.hidden = false;
  factorList.textContent = "";

  result.contributions.slice(0, 3).forEach((c, i) => {
    const isPos = c.shap_value >= 0;
    const li = document.createElement("li");
    li.className = "factor-item";

    const rank = document.createElement("span");
    rank.className = "factor-rank";
    rank.textContent = (i + 1) + ".";

    const dot = document.createElement("span");
    dot.className = "factor-dot " + (isPos ? "pos" : "neg");

    const name = document.createElement("span");
    name.className = "factor-name";
    name.textContent = FEATURE_LABELS[c.feature] || c.feature;

    const direction = document.createElement("span");
    direction.className = "factor-direction";
    direction.textContent = isPos ? "— làm tăng nguy cơ" : "— làm giảm nguy cơ";

    li.append(rank, dot, name, direction);
    factorList.appendChild(li);
  });
}

function showTopFactorsError() {
  topFactors.hidden = true;
  topFactorsError.hidden = false;
}

function renderExplain(result) {
  explainSection.hidden = false;
  const chart = document.getElementById("shap-chart");
  chart.textContent = "";

  const maxAbs = Math.max(...result.contributions.map((c) => Math.abs(c.shap_value)), 0.001);

  const tbody = shapTable.querySelector("tbody");
  tbody.textContent = "";

  result.contributions.forEach((c) => {
    const row = document.createElement("div");
    row.className = "diverging-row";

    const label = document.createElement("div");
    label.className = "row-label";
    label.textContent = c.feature;
    row.appendChild(label);

    const track = document.createElement("div");
    track.className = "diverging-track";

    const bar = document.createElement("div");
    const isPos = c.shap_value >= 0;
    bar.className = "diverging-bar " + (isPos ? "pos" : "neg");
    const widthPct = (Math.abs(c.shap_value) / maxAbs) * 48; // 48% of track per side max
    bar.style.width = widthPct + "%";
    bar.tabIndex = 0;
    bar.setAttribute(
      "aria-label",
      `${c.feature}: giá trị ${c.value.toFixed(2)}, đóng góp SHAP ${c.shap_value.toFixed(3)}`
    );
    attachTooltip(bar, () =>
      `${c.feature}\nGiá trị (chuẩn hóa): ${c.value.toFixed(2)}\nĐóng góp SHAP: ${c.shap_value.toFixed(3)}`
    );
    track.appendChild(bar);
    row.appendChild(track);

    const valueEl = document.createElement("div");
    valueEl.className = "row-value";
    valueEl.textContent = (isPos ? "+" : "") + c.shap_value.toFixed(2);
    row.appendChild(valueEl);

    chart.appendChild(row);

    const tr = document.createElement("tr");
    const tdFeat = document.createElement("td");
    tdFeat.textContent = c.feature;
    const tdVal = document.createElement("td");
    tdVal.textContent = c.value.toFixed(3);
    const tdShap = document.createElement("td");
    tdShap.textContent = c.shap_value.toFixed(3);
    tr.append(tdFeat, tdVal, tdShap);
    tbody.appendChild(tr);
  });
}

let tooltipEl = null;
function attachTooltip(el, textFn) {
  const show = (evt) => {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "tooltip-bubble";
      document.body.appendChild(tooltipEl);
    }
    tooltipEl.textContent = textFn();
    tooltipEl.style.whiteSpace = "pre-line";
    const x = evt.clientX !== undefined ? evt.clientX : el.getBoundingClientRect().left;
    const y = evt.clientY !== undefined ? evt.clientY : el.getBoundingClientRect().top;
    tooltipEl.style.left = Math.min(x + 12, window.innerWidth - 240) + "px";
    tooltipEl.style.top = y + 16 + "px";
    tooltipEl.style.display = "block";
  };
  const hide = () => {
    if (tooltipEl) tooltipEl.style.display = "none";
  };
  el.addEventListener("pointermove", show);
  el.addEventListener("pointerenter", show);
  el.addEventListener("pointerleave", hide);
  el.addEventListener("focus", show);
  el.addEventListener("blur", hide);
}

form.addEventListener("submit", async (evt) => {
  evt.preventDefault();
  lastFeatures = readFormFeatures();
  lastExplain = null;
  explainSection.hidden = true;
  topFactors.hidden = true;
  topFactorsError.hidden = true;

  const predictReq = fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lastFeatures),
  });
  // Fetched alongside /predict (not just on demand) so the top-3 factors
  // can be shown immediately after prediction, per the "always explain
  // the result" requirement -- the full breakdown button then just
  // re-renders this same cached response instead of a second call.
  const explainReq = fetch(`${API_BASE}/predict/explain`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(lastFeatures),
  });

  const predictRes = await predictReq;
  if (!predictRes.ok) {
    alert("Dự báo lỗi: " + predictRes.status);
    return;
  }
  renderPrediction(await predictRes.json());

  try {
    const explainRes = await explainReq;
    if (!explainRes.ok) throw new Error(String(explainRes.status));
    lastExplain = await explainRes.json();
    renderTopFactors(lastExplain);
  } catch {
    showTopFactorsError();
  }
});

explainBtn.addEventListener("click", async () => {
  if (!lastFeatures) return;
  if (lastExplain) {
    renderExplain(lastExplain);
    return;
  }
  explainBtn.disabled = true;
  explainBtn.textContent = "Đang tính…";
  try {
    const res = await fetch(`${API_BASE}/predict/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastFeatures),
    });
    if (!res.ok) {
      alert("Giải thích lỗi: " + res.status);
      return;
    }
    lastExplain = await res.json();
    renderExplain(lastExplain);
  } finally {
    explainBtn.disabled = false;
    explainBtn.textContent = "Xem toàn bộ yếu tố (SHAP)";
  }
});

toggleTableBtn.addEventListener("click", () => {
  shapTable.hidden = !shapTable.hidden;
  toggleTableBtn.textContent = shapTable.hidden ? "Xem dạng bảng" : "Ẩn bảng";
});

const SAMPLE_BATCH = [
  { Age: 78, BMI: 27.2, ASA_Score: "IV", Has_Diabetes: 1, Has_HTN: 1, Surgery_Type: "CC", PreOp_WBC: 12.1, PreOp_Albumin: 2.4 },
  { Age: 35, BMI: 22.0, ASA_Score: "I", Has_Diabetes: 0, Has_HTN: 0, Surgery_Type: "CT", PreOp_WBC: 7.0, PreOp_Albumin: 4.2 },
  { Age: 60, BMI: 24.0, ASA_Score: "II", Has_Diabetes: 0, Has_HTN: 1, Surgery_Type: "CT", PreOp_WBC: null, PreOp_Albumin: null },
];

batchBtn.addEventListener("click", async () => {
  batchBtn.disabled = true;
  try {
    const res = await fetch(`${API_BASE}/predict/batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ patients: SAMPLE_BATCH }),
    });
    if (!res.ok) {
      alert("Batch lỗi: " + res.status);
      return;
    }
    const data = await res.json();
    const table = document.getElementById("batch-table");
    const tbody = table.querySelector("tbody");
    tbody.textContent = "";
    data.predictions.forEach((p, i) => {
      const tr = document.createElement("tr");
      const tdCase = document.createElement("td");
      tdCase.textContent = `Ca ${i + 1}`;
      const tdProb = document.createElement("td");
      tdProb.textContent = (p.risk_probability * 100).toFixed(1) + "%";
      const tdFlag = document.createElement("td");
      tdFlag.textContent = riskTier(p.risk_probability).label;
      tr.append(tdCase, tdProb, tdFlag);
      tbody.appendChild(tr);
    });
    table.hidden = false;
  } finally {
    batchBtn.disabled = false;
  }
});

async function loadHealth() {
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    dot.className = "status-dot " + (data.model_loaded ? "ok" : "down");
    text.textContent = data.model_loaded ? "API sẵn sàng" : "Model chưa sẵn sàng";
  } catch {
    dot.className = "status-dot down";
    text.textContent = "Không kết nối được API";
  }
}

async function loadModelInfo() {
  const dl = document.getElementById("model-info");
  try {
    const res = await fetch(`${API_BASE}/model/info`);
    const data = await res.json();
    dl.textContent = "";
    const entries = [
      ["Mô hình", data.model_name],
      ["Huấn luyện lúc", new Date(data.trained_at).toLocaleString("vi-VN")],
      ["ROC-AUC", data.metrics.roc_auc?.toFixed(3)],
      ["Recall", data.metrics.recall?.toFixed(3)],
      ["Precision", data.metrics.precision?.toFixed(3)],
      ["MLflow run", (data.mlflow_run_id || "").slice(0, 8)],
    ];
    entries.forEach(([label, value]) => {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value ?? "–";
      dl.append(dt, dd);
    });
  } catch {
    dl.textContent = "Không tải được thông tin mô hình.";
  }
}

loadHealth();
loadModelInfo();

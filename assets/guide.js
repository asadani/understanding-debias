(() => {
  const examples = [
    { key: "hr", name: "HR / payroll", raw: 0.82, baseline: 0.60 },
    { key: "ops", name: "Operations", raw: 0.84, baseline: 0.76 },
    { key: "finance", name: "Finance", raw: 0.79, baseline: 0.68 },
  ];
  const slider = document.querySelector("#lambda");
  const sliderOut = document.querySelector("#lambdaOut");
  const rows = document.querySelector("#scoreRows");
  const winnerBox = document.querySelector("#winnerBox");
  const chart = document.querySelector("#barChart");
  const fmt = (v) => v.toFixed(3);

  function renderScores() {
    const lambda = Number(slider.value);
    const scores = examples.map((item) => ({ ...item, adjusted: item.raw - lambda * item.baseline }));
    const winner = scores.reduce((a, b) => (b.adjusted > a.adjusted ? b : a));
    sliderOut.value = lambda.toFixed(2);
    sliderOut.textContent = lambda.toFixed(2);
    rows.innerHTML = scores.map((item) => {
      const isWinner = item.key === winner.key;
      return `<div class="score-row${isWinner ? " is-winner" : ""}" data-key="${item.key}">
        <span class="intent-name">${item.name}</span><span>${fmt(item.raw)}</span><span>−${fmt(item.baseline)}</span><span class="adjusted">${fmt(item.adjusted)}</span>
      </div>`;
    }).join("");
    winnerBox.innerHTML = `Top toy score: <b>${winner.name}</b> · margin over second place <b>${fmt(winner.adjusted - [...scores].sort((a, b) => b.adjusted - a.adjusted)[1].adjusted)}</b>. A score margin alone is not confidence.`;
    const min = Math.min(...scores.map((s) => s.adjusted));
    const max = Math.max(...scores.map((s) => s.adjusted));
    const span = Math.max(max - min, 0.015);
    chart.innerHTML = [...scores].sort((a, b) => a.adjusted - b.adjusted).map((item) => {
      const width = 20 + ((item.adjusted - min) / span) * 75;
      return `<div class="chart-row" data-key="${item.key}"><span>${item.name}</span><div class="chart-track"><div class="chart-fill" style="width:${width}%"></div></div><b>${fmt(item.adjusted)}</b></div>`;
    }).join("");
    chart.setAttribute("aria-label", `Illustrative adjusted scores at lambda ${lambda.toFixed(2)}. Highest score: ${winner.name}. These are invented example values.`);
  }
  slider?.addEventListener("input", renderScores);
  document.querySelector("#resetLab")?.addEventListener("click", () => {
    slider.value = "0";
    renderScores();
    slider.focus();
  });
  renderScores();

  const details = {
    goal: ["START HERE", "Choose the meaning you want to preserve.", "A correction is only useful relative to a defined target. For a support assistant, that might be customer intent. For job titles, it might be function rather than industry or seniority.", "What should stay close, and what should separate?"],
    shape: ["EMBEDDING GEOMETRY", "Do vectors crowd a few directions?", "Anisotropy describes directional concentration. Mean-centre or remove dominant components only if that change improves held-out task performance. Apply the same saved transform to future inputs and prototypes.", "Does a geometric diagnostic connect to the errors?"],
    hub: ["NEIGHBOURHOOD RANKING", "Does one prototype appear everywhere?", "Count how often each candidate appears among many queries’ top-k results. If a hub is distorting ranking, compare cosine with local-density corrections such as CSLS. Tune the neighbourhood size on validation.", "Does the correction improve the right ranking metric?"],
    negative: ["TRAINING DATA", "Are true matches labelled as negatives?", "Contrastive learning pushes positive pairs together and negatives apart. If two same-intent examples enter the negative pool, the objective receives a wrong signal. Debiased losses or negative filtering target that training-time contamination.", "Can you find and quantify false negatives?"],
    score: ["ASSIGNMENT CALIBRATION", "Does a category win too often across examples?", "A cluster-specific offset can correct a systematic preference, but it moves that boundary for every input. Estimate it from training or validation data; never choose it because a single hand-picked example looks better.", "Does held-out macro-F1 improve without a bad coverage trade-off?"],
    freeze: ["DEPLOYMENT", "Will tomorrow’s assignment mean the same thing?", "Save the encoder revision, any instruction, preprocessing, transformation, cluster centres, correction values, threshold and taxonomy together. A transform update without recomputing the prototypes changes the coordinate system underneath them.", "Can you reproduce an assignment from its version ID?"],
    fairness: ["A DIFFERENT MEANING OF BIAS", "Are outcomes uneven across people or groups?", "Fairness-aware clustering asks about representation and outcomes for protected groups. That is a distinct objective with its own legal, ethical and measurement requirements; geometric hubness correction does not establish fairness.", "Which groups and harms must be audited?"],
  };
  const titles = document.querySelector("#detailTitle");
  const text = document.querySelector("#detailText");
  const overline = document.querySelector("#detailOverline");
  const prompt = document.querySelector("#detailPrompt");
  function selectNode(id) {
    const value = details[id];
    if (!value) return;
    overline.textContent = value[0];
    titles.textContent = value[1];
    text.textContent = value[2];
    prompt.textContent = value[3];
    document.querySelectorAll(".map-node").forEach((node) => node.classList.toggle("active", node.dataset.node === id));
    document.querySelectorAll(".map-mobile button").forEach((node) => node.setAttribute("aria-pressed", String(node.dataset.node === id)));
  }
  document.querySelectorAll(".map-node, .map-mobile button").forEach((node) => {
    node.addEventListener("click", () => selectNode(node.dataset.node));
    if (node.classList.contains("map-node")) node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(node.dataset.node); }
    });
  });
  const mapObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("seen");
    });
  }, { threshold: 0.08 });
  document.querySelectorAll(".failure, .pipeline article, .usecase, .timeline li").forEach((node) => mapObserver.observe(node));

  const resultPanel = document.querySelector("#benchmarkResults");
  fetch("assets/benchmark-results.json")
    .then((response) => { if (!response.ok) throw new Error("No measured result is published."); return response.json(); })
    .then((result) => {
      if (result.status !== "measured") throw new Error("No measured result is published.");
      const pct = (value) => (value * 100).toFixed(1) + "%";
      const methods = [
        ["cosine", "Cosine prototypes"],
        ["mean_baseline", "Mean-background correction"],
        ["csls", "CSLS local scaling"],
      ];
      const rows = methods.map(([key, title]) => {
        const m = result.metrics[key];
        const delta = result.absolute_delta_vs_cosine[key];
        const deltaText = key === "cosine" ? "reference" : `Δ macro-F1 ${delta.macro_f1 >= 0 ? "+" : ""}${(delta.macro_f1 * 100).toFixed(1)} pp`;
        return `<div class="benchmark-metric"><b>${title}</b><span>Accuracy ${pct(m.accuracy)} · macro-F1 ${pct(m.macro_f1)}</span><small>${deltaText}</small></div>`;
      }).join("");
      resultPanel.innerHTML = `<span class="detail-overline">MEASURED · ${result.retrieved_on}</span>
        <h3>${result.task}</h3>
        <div class="benchmark-metrics">${rows}</div>
        <p class="fineprint">Encoder: ${result.model.id} · ${result.model.device} · ${result.model.encoder_frozen ? "frozen" : "trained"}. Split counts: ${result.dataset.counts.prototype_fit} prototypes, ${result.dataset.counts.background_reference} reference, ${result.dataset.counts.validation} validation, ${result.dataset.counts.official_test} official test. λ = ${result.selected_lambda}. Test examples were not used to select λ.</p>
        <p class="fineprint">${result.interpretation} <a href="${result.dataset.source}" target="_blank" rel="noreferrer">Pinned dataset source ↗</a></p>`;
    })
    .catch(() => {});
})();

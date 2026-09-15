"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const BUNDLED_DEMO_ID = "bundled-end-to-end-demo";
const BUNDLED_DEMO_ASSETS = Object.freeze({
  sar: "assets/incidents/os-2026-0915/sar.png",
  probability: "assets/incidents/os-2026-0915/probability.png",
  mask: "assets/incidents/os-2026-0915/mask.png",
  thumbnail: "assets/incidents/os-2026-0915/thumbnail.png",
});
const state = { data: null, horizonIndex: 0, incidents: [], loadSequence: 0, controlsBound: false, demoStage: 0, demoMaxStage: 0, demoAsset: "sar" };
const el = (selector) => document.querySelector(selector);
const svg = (name, attributes = {}) => { const node = document.createElementNS(SVG_NS, name); Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value))); return node; };
const polygon = (points) => points.map(([x, y], index) => `${index ? "L" : "M"}${x},${y}`).join(" ") + " Z";
const line = (points) => points.map(([x, y], index) => `${index ? "L" : "M"}${x},${y}`).join(" ");
function formatTime(value, includeDate = false) { if (!value || Number.isNaN(new Date(value).getTime())) return "—"; return new Intl.DateTimeFormat("en-GB", { ...(includeDate ? { day: "2-digit", month: "short" } : {}), hour: "2-digit", minute: "2-digit", hour12: false, timeZone: "UTC", timeZoneName: "short" }).format(new Date(value)); }
const stage = (name, detail) => ({ name, detail: detail || "No frozen output available" });
const validPolygon = (points) => Array.isArray(points) && points.length >= 3 && points.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite));
const validLine = (points) => Array.isArray(points) && points.length >= 2 && points.every((point) => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite));
const stageNames = ["Scene", "Detection", "Source / Hindcast", "AIS", "Forecast", "Coastal Impact"];
const absoluteAsset = (name, baseUrl) => typeof name === "string" && name ? new URL(name, baseUrl).href : null;

// Compatibility for the former single-demo schema. Values are copied verbatim;
// this adapter only supplies the stage names used by the incident selector.
function normalizeBundledDemo(raw) {
  return { ...raw, kind: "bundled-demo", scene: { platform: raw.detection?.sceneId?.split("_")[0], productId: raw.detection?.sceneId, acquiredAt: raw.incident?.observedAt }, assets: { ...raw.assets, ...BUNDLED_DEMO_ASSETS }, availability: [true, true, true, true, true, true], stages: [
    stage("Scene", raw.detection?.sceneId), stage("Detection", "Slick mask and probability"),
    stage("Source / Hindcast", "50% and 95% source regions"), stage("AIS", "Synthetic vessel tracks"),
    stage("Forecast", raw.forecasts?.map((item) => `+${item.hours}h`).join(" · ")),
    stage("Coastal Impact", "Shoreline contact support"),
  ] };
}
// Frozen PANGAEA/Kuwait exports are kept in their native schema. Missing stage
// artifacts are shown as unavailable, rather than being replaced with geometry or metrics.
function normalizeFrozenIncident(raw, baseUrl, artifacts = {}) {
  const ready = raw.readiness || {}, scene = raw.sar || {}, blocked = raw.blockedReasons || {};
  const assets = Object.fromEntries(Object.entries(raw.assets || {}).map(([name, path]) => [name, absoluteAsset(path, baseUrl)]));
  assets.sar = assets.vv || assets.sar || null;
  const presentation = raw.presentation || {};
  const sourcePosterior = presentation.sourcePosterior || artifacts.source?.sourcePosterior || (validPolygon(artifacts.source?.region50) ? artifacts.source : null);
  const candidatePayload = presentation.candidates || artifacts.ais?.candidates || null;
  const candidates = Array.isArray(candidatePayload) ? candidatePayload.filter((item) => validLine(item.track) && Number.isFinite(item.rank) && item.rank > 0 && Number.isFinite(item.score) && typeof item.name === "string") : null;
  const forecasts = presentation.forecasts || artifacts.forecast?.forecasts || null;
  const coastalImpact = presentation.coastalImpact || artifacts.coastal?.coastalImpact || (validLine(artifacts.coastal?.riskSegment) ? artifacts.coastal : null);
  const map = presentation.map || raw.map || { center: "—" };
  const screenMap = map.coordinateSpace === "screen";
  const detection = ready.detection ? (presentation.detection || raw.detection || null) : null;
  const status = [
    !!(scene.productId || scene.platform || assets.sar),
    !!(detection && (assets.probability || assets.mask || detection.execution || detection.model || detection.oilPixelCount != null || detection.areaKm2 != null)),
    !!(ready.hindcast && screenMap && sourcePosterior && validPolygon(sourcePosterior.region50) && validPolygon(sourcePosterior.region95)),
    !!(ready.ais && screenMap && Array.isArray(candidates) && candidates.some((item) => validLine(item.track))),
    !!(ready.forecast && screenMap && Array.isArray(forecasts) && forecasts.some((item) => validPolygon(item.extent) && validPolygon(item.core))),
    !!(ready.coastal && screenMap && coastalImpact && validLine(coastalImpact.riskSegment)),
  ];
  const availability = status.map((available, index) => available && status.slice(0, index).every(Boolean));
  const details = [scene.productId || scene.platform || "Scene unavailable", detection?.execution || "Detection output", blocked.source || "Source posterior", raw.ais?.label || "Vessel candidates", blocked.forecast || "Forecast horizons", blocked.coastal || "Coastal contact"];
  return { kind: "real-incident", incident: { id: raw.caseId, runId: raw.execution || "exported", location: raw.source || raw.title || "Observed incident", observedAt: raw.observedAt }, scene, sar: scene, assets, map, detection, sourcePosterior, candidates, forecasts, coastalImpact, attribution: presentation.attribution || { disclaimer: "Attribution rankings are investigative prioritization, not proof." }, readiness: ready, availability, stages: stageNames.map((name, index) => stage(name, availability[index] ? details[index] : index > 0 && !availability[index - 1] ? "Unavailable · previous stage missing" : `Unavailable · ${details[index]}`)) };
}
const normalizeIncident = (raw, baseUrl, artifacts) => raw?.schemaVersion === 1 && raw?.demoData ? normalizeBundledDemo(raw) : normalizeFrozenIncident(raw, baseUrl, artifacts);

function renderStages(stages) {
  const guided = !!state.data;
  el("#stage-list").innerHTML = stages.map((item, index) => {
    const current = guided && index === state.demoStage;
    const completed = guided && index <= state.demoMaxStage && !current;
    const locked = guided && (index > state.demoMaxStage || !state.data.availability[index]);
    const copy = guided
      ? `<button class="stage-copy stage-link" type="button" data-demo-stage="${index}" ${locked ? "disabled" : ""}><strong>${item.name}</strong><span>${item.detail}</span></button>`
      : `<div class="stage-copy"><strong>${item.name}</strong><span>${item.detail}</span></div>`;
    return `<li class="stage-item ${current ? "active" : ""} ${completed ? "completed" : ""} ${locked ? "locked" : ""}"><span class="stage-number" aria-hidden="true"></span>${copy}</li>`;
  }).join("");
}
function renderCoast(data) { const layer = el("#coast-layer"); layer.replaceChildren(); if (state.demoStage >= 2 && validPolygon(data.map?.coastline)) layer.append(svg("path", { d: polygon(data.map.coastline), class: "coast" })); if (state.demoStage >= 5 && validLine(data.coastalImpact?.riskSegment)) layer.append(svg("path", { d: line(data.coastalImpact.riskSegment), class: "coast-risk" })); }
function renderDetection(data) { const layer = el("#detection-layer"); layer.replaceChildren(); if (!validPolygon(data.detection?.polygon)) return; layer.append(svg("path", { d: polygon(data.detection.polygon), class: "slick" })); if (Array.isArray(data.detection.centroid)) layer.append(svg("circle", { cx: data.detection.centroid[0], cy: data.detection.centroid[1], r: 5, class: "slick-center" })); }
function renderPosterior(data) { const layer = el("#posterior-layer"); layer.replaceChildren(); const posterior = data.sourcePosterior; if (!validPolygon(posterior?.region50) || !validPolygon(posterior?.region95)) return; layer.append(svg("path", { d: polygon(posterior.region95), class: "posterior-95" }), svg("path", { d: polygon(posterior.region50), class: "posterior-50" })); if (Array.isArray(posterior.centroid)) layer.append(svg("circle", { cx: posterior.centroid[0], cy: posterior.centroid[1], r: 5, class: "posterior-core" })); }
function renderVessels(data) {
  const layer = el("#vessel-layer"); layer.replaceChildren();
  const synthetic = data.kind === "bundled-demo" || data.readiness?.aisSynthetic;
  (data.candidates || []).forEach((candidate) => {
    const track = candidate.track;
    if (!validLine(track)) return;
    layer.append(svg("path", { d: line(track), class: `track rank-${candidate.rank} ${synthetic ? "synthetic-track" : ""}` }));
    const [x, y] = track.at(-1);
    const color = candidate.rank === 1 ? "#397895" : synthetic && candidate.rank === 2 ? "#716683" : "#738996";
    layer.append(svg("circle", { cx: x, cy: y, r: candidate.rank === 1 ? 6 : 4, fill: color, class: "vessel-marker" }));
    if (data.kind === "bundled-demo" && track.length > 1) {
      const [px, py] = track.at(-2), angle = Math.atan2(y - py, x - px) * 180 / Math.PI;
      layer.append(svg("path", { d: `M${x - 12},${y - 4} L${x - 4},${y} L${x - 12},${y + 4}`, transform: `rotate(${angle} ${x} ${y})`, class: `travel-arrow rank-${candidate.rank}` }));
    }
    const label = svg("text", { x: x + 10, y: y - 9, class: "vessel-label" }); label.textContent = data.kind === "bundled-demo" ? `${candidate.rank}. ${candidate.name}` : candidate.name; layer.append(label);
  });
}
function renderForecast(data, horizonIndex) { const layer = el("#forecast-layer"); layer.replaceChildren(); const horizon = data.forecasts?.[horizonIndex]; el("#valid-time").textContent = horizon ? `Valid ${formatTime(horizon.validAt, true)}` : "No exported forecast"; if (validPolygon(horizon?.extent) && validPolygon(horizon?.core)) layer.append(svg("path", { d: polygon(horizon.extent), class: "forecast-plume" }), svg("path", { d: polygon(horizon.core), class: "forecast-core" })); document.querySelectorAll(".horizon-button").forEach((button, index) => { button.classList.toggle("active", index === horizonIndex); button.setAttribute("aria-pressed", String(index === horizonIndex)); }); renderMapLabels(data); }
function renderAnalysisCues(data) {
  const layer = el("#analysis-cue-layer"); layer.replaceChildren();
  if (data.kind !== "bundled-demo" || state.demoStage < 2) return;
  const source = data.sourcePosterior?.centroid, slick = data.detection?.centroid;
  if (!source || !slick) return;
  layer.append(svg("path", { d: line([source, slick]), class: "source-link" }), svg("circle", { cx: slick[0], cy: slick[1], r: 3, class: "source-link-end" }));
}
function renderMapLabels(data) {
  el("#map-labels").replaceChildren(); el("#coordinate-readout").textContent = data.map?.center || "—"; el("#coordinate-readout").hidden = !data.map?.center || data.map.center === "—";
  el("#map-scale").hidden = state.demoStage < 2 || !(data.map?.scaleLabel || data.kind === "bundled-demo");
  el("#map-scale").textContent = data.map?.scaleLabel || "Schematic screen geometry · not to scale";
  el("#map-context").textContent = data.kind === "bundled-demo" ? state.demoStage < 2 ? `Sentinel-1 SAR · ${data.detection.sceneId}` : state.demoStage === 3 ? "Synthetic AIS · schematic map" : "Synthetic screen-coordinate map · EPSG:4326 display" : state.demoStage < 2 ? `${data.scene?.platform || "Observed scene"} · ${data.scene?.productId || "exported output"}` : data.map?.coordinateReferenceSystem || "Exported incident geometry";
  const addLabel = (text, point, className = "map-annotation") => { if (!Array.isArray(point) || point.length !== 2) return; const label = svg("text", { x: point[0], y: point[1], class: className }); label.textContent = text; el("#map-labels").append(label); };
  const legend = [];
  if (state.demoStage >= 1 && validPolygon(data.detection?.polygon)) { legend.push('<span><i style="background:var(--lime)"></i>detected slick</span>'); if (Array.isArray(data.detection.centroid)) addLabel("DETECTED SLICK", [data.detection.centroid[0] + 12, data.detection.centroid[1] - 12]); }
  if (state.demoStage >= 2 && validPolygon(data.sourcePosterior?.region50)) { legend.push('<span><i style="background:var(--cyan)"></i>50% source support</span><span><i style="background:#8bb9da"></i>95% uncertainty</span>'); if (Array.isArray(data.sourcePosterior.centroid)) addLabel("LIKELY SOURCE REGION", [data.sourcePosterior.centroid[0] - 42, data.sourcePosterior.centroid[1] - 48]); }
  if (state.demoStage === 3 && data.candidates?.length) legend.push(`<span><i style="background:var(--cyan)"></i>top AIS candidate</span><span><i style="background:${data.kind === "bundled-demo" || data.readiness?.aisSynthetic ? "var(--purple)" : "#738996"}"></i>other ${data.kind === "bundled-demo" || data.readiness?.aisSynthetic ? "synthetic " : ""}AIS tracks</span>`);
  if (state.demoStage >= 4) { const horizon = data.forecasts?.[state.horizonIndex]; if (horizon) { legend.push('<span><i style="background:var(--amber)"></i>forecast core and extent</span>'); addLabel(`FORECAST +${horizon.hours}H`, horizon.core?.[0]); } }
  if (state.demoStage >= 5 && validLine(data.coastalImpact?.riskSegment)) { legend.push('<span><i style="background:var(--coral)"></i>shoreline contact support</span>'); addLabel("CONTACT-RISK SEGMENT", data.coastalImpact.riskSegment[1], "map-annotation map-annotation-risk"); }
  el("#legend").innerHTML = legend.join("");
  el("#legend").hidden = !legend.length;
}
function renderCandidateList(candidates) { el("#candidate-list").innerHTML = candidates?.length ? candidates.map((candidate, index) => `<article class="candidate ${index > 1 ? "extra" : ""}"><span class="candidate-rank">${String(candidate.rank).padStart(2, "0")}</span><div class="candidate-name"><strong>${candidate.name}</strong><span>${candidate.mmsi ? `MMSI ${candidate.mmsi}` : "ID unavailable"}${candidate.type ? ` · ${candidate.type}` : ""}</span></div><div class="candidate-score"><strong>${Math.round(candidate.score * 100)}</strong><span>support / 100</span></div></article>`).join("") : '<p class="semantics">No exported AIS candidate ranking is available.</p>'; }
function renderDetails(data) {
  const incident = data.incident || {}, detection = data.detection || {}, posterior = data.sourcePosterior, coast = data.coastalImpact;
  document.title = `SLICKOIL — ${incident.id || "Incident"}`; el("#incident-name").textContent = `${incident.id || "Incident"} · ${incident.location || "—"}`; el("#run-time").textContent = `RUN ${incident.runId || "—"}`; el("#scene-time").textContent = incident.observedAt ? `Observed ${formatTime(incident.observedAt, true)}` : "Exported incident";
  el("#slick-area").textContent = Number.isFinite(detection.areaKm2) ? `${detection.areaKm2.toFixed(1)} km²` : "—"; el("#detection-confidence").textContent = Number.isFinite(detection.confidence) ? `${Math.round(detection.confidence * 100)}%` : "—"; el("#scene-id").textContent = detection.sceneId || data.scene?.productId || "—"; el("#scene-sensor").textContent = data.scene?.platform || "—"; el("#orientation").textContent = Number.isFinite(detection.orientationDeg) ? `${detection.orientationDeg}°` : "—";
  el("#threshold-row").hidden = !Number.isFinite(detection.threshold); el("#threshold").textContent = Number.isFinite(detection.threshold) ? String(detection.threshold) : "—";
  el("#pixel-count-row").hidden = !Number.isFinite(detection.oilPixelCount); el("#pixel-count").textContent = Number.isFinite(detection.oilPixelCount) ? detection.oilPixelCount.toLocaleString("en-GB") : "—";
  el("#region-size").textContent = Number.isFinite(posterior?.region50Km2) ? `${posterior.region50Km2.toFixed(1)} km²` : "—"; el("#release-time").textContent = formatTime(posterior?.releaseTime?.p50); el("#posterior-semantics").textContent = posterior?.semantics || (data.readiness?.hindcast ? "Frozen output availability is shown in the analysis chain." : "No frozen source posterior is available.");
  const funnel = posterior?.screening; el("#hindcast-funnel").innerHTML = funnel ? [[funnel.initial, "sampled"], [funnel.eulerian, "screened"], [funnel.highFidelity, "refined"]].map(([value, label], index) => `${index ? '<span class="funnel-arrow">›</span>' : ""}<div class="funnel-step"><strong>${value}</strong><span>${label}</span></div>`).join("") : ""; renderCandidateList(data.candidates);
  el("#risk-level").textContent = coast?.level || "—"; el("#risk-meter-fill").style.width = Number.isFinite(coast?.contactSupport) ? `${coast.contactSupport * 100}%` : "0%"; el("#contact-support").textContent = Number.isFinite(coast?.contactSupport) ? `${Math.round(coast.contactSupport * 100)}%` : "—"; el("#coast-arrival").textContent = Number.isFinite(coast?.earliestHours) ? `+${coast.earliestHours}h` : "—";
}
function renderHorizonButtons(data) { el("#horizon-buttons").innerHTML = (data.forecasts || []).map((horizon, index) => `<button class="horizon-button ${index === 0 ? "active" : ""}" type="button" data-horizon="${index}" aria-pressed="${index === 0}">+${horizon.hours}h</button>`).join(""); }
function updateDemoAssetViewer() {
  const viewer = el("#demo-asset-viewer");
  const bundled = state.data?.kind === "bundled-demo";
  const assets = state.data?.assets || {};
  const viewAssets = { sar: assets.sar, probability: assets.probability, mask: assets.mask, overlay: assets.overlay || (assets.sar && assets.mask ? assets.sar : null) };
  const visible = state.demoStage <= 1 && !!(state.demoStage === 0 ? assets.sar : Object.values(viewAssets).some(Boolean));
  viewer.hidden = !visible;
  el("#stage-empty").hidden = visible || state.demoStage >= 2;
  el("#stage-empty").textContent = state.demoStage === 0 ? "No scene raster was exported for this incident." : "Detection metadata is available; no raster preview was exported.";
  if (!visible) return;
  if (!viewAssets[state.demoAsset]) state.demoAsset = Object.keys(viewAssets).find((name) => viewAssets[name]) || "sar";
  el("#demo-asset-caption").hidden = state.demoStage === 0;
  el("#demo-asset-tabs").hidden = state.demoStage === 0;
  const details = {
    sar: ["SAR scene input", "Precomputed synthetic SAR scene input."],
    probability: ["Slick probability", "Precomputed synthetic detector probability preview."],
    mask: ["Slick mask", "Precomputed synthetic segmented slick mask."],
    overlay: ["SAR + mask", "Precomputed synthetic mask over the cached SAR input."],
    thumbnail: ["Thumbnail", "Precomputed synthetic thumbnail preview."],
  };
  const [title, cachedCaption] = details[state.demoAsset];
  const caption = bundled ? cachedCaption : "Exported incident raster · no processing runs in the browser.";
  el("#demo-asset-title").textContent = title;
  el("#demo-asset-caption").textContent = caption;
  const image = el("#demo-asset-image"), maskOverlay = el("#demo-mask-overlay"), composedOverlay = state.demoAsset === "overlay" && !!(assets.sar && assets.mask) && !assets.overlay;
  image.src = state.demoAsset === "overlay" ? assets.overlay || assets.sar : assets[state.demoAsset]; image.alt = `${title} for ${bundled ? "the precomputed demonstration" : "the selected exported incident"}`;
  if (assets.mask) maskOverlay.src = assets.mask; maskOverlay.alt = "Segmented slick mask over the SAR raster"; maskOverlay.hidden = !composedOverlay;
  document.querySelectorAll("[data-demo-asset]").forEach((button) => {
    const available = !!viewAssets[button.dataset.demoAsset] && (state.demoStage === 1 || button.dataset.demoAsset === "sar");
    button.disabled = !available;
    button.classList.toggle("active", button.dataset.demoAsset === state.demoAsset);
  });
}
function updateDemoGuide() {
  const bundled = state.data?.kind === "bundled-demo";
  const guide = el("#demo-guide");
  guide.hidden = false;
  const panels = { summary: document.querySelector(".summary-card"), source: document.querySelector(".source-card"), ais: document.querySelector(".vessel-card"), coast: document.querySelector(".coast-card"), forecast: document.querySelector(".forecast-timeline") };
  const stageIndex = state.demoStage;
  const cachedDescriptions = [
    `SAR acquisition ${formatTime(state.data.incident.observedAt, true)} · ${state.data.detection.sceneId}.`,
    "Compare the cached probability, mask and SAR overlay.",
    "Observed slick and model-supported origin regions; schematic geometry.",
    `Synthetic AIS. ${state.data.attribution.disclaimer}`,
    "Choose a forecast horizon to inspect its cached extent and core.",
    "Coastal contact support is uncalibrated; see highlighted shoreline segment.",
  ];
  const realDescriptions = [
    `${state.data.scene?.productId || "Exported scene"}${state.data.incident.observedAt ? ` · ${formatTime(state.data.incident.observedAt, true)}` : ""}.`,
    "Exported detector output and available raster layers.",
    "Exported source posterior and release-time evidence.",
    state.data.readiness?.aisSynthetic ? `Synthetic AIS. ${state.data.attribution.disclaimer}` : `AIS candidates. ${state.data.attribution.disclaimer}`,
    "Choose an exported forecast horizon.",
    "Exported shoreline contact assessment.",
  ];
  const titles = ["Scene", "Detection", "Source / Hindcast", "AIS", "Forecast", "Coastal Impact"];
  const nextLabels = ["Next: Detection", "Infer source", "Compare AIS candidates", "Generate forecast", "Assess coastline", "Restart demo"];
  guide.querySelector(".eyebrow").textContent = bundled ? "Precomputed demonstration" : "Exported incident";
  el("#stage-status").textContent = bundled ? "Precomputed demonstration" : stageIndex === 3 && state.data.readiness?.aisSynthetic ? "Exported output · Synthetic AIS" : "Observed / exported output";
  el("#demo-guide-title").textContent = titles[stageIndex]; el("#demo-guide-copy").textContent = (bundled ? cachedDescriptions : realDescriptions)[stageIndex];
  el("#guide-back").disabled = stageIndex === 0; el("#guide-back").textContent = stageIndex === 0 ? "Back" : `Back: ${titles[stageIndex - 1]}`;
  const nextAvailable = stageIndex === 5 || state.data.availability[stageIndex + 1];
  el("#guide-next").disabled = !nextAvailable; el("#guide-next").textContent = nextAvailable ? stageIndex === 5 && !bundled ? "Restart analysis" : nextLabels[stageIndex] : `${titles[stageIndex + 1]} unavailable`;
  panels.summary.hidden = stageIndex > 1; panels.source.hidden = stageIndex !== 2; panels.ais.hidden = stageIndex !== 3; panels.forecast.hidden = stageIndex !== 4; panels.coast.hidden = stageIndex !== 5;
  el("#summary-metrics").hidden = stageIndex === 0 || !Number.isFinite(state.data.detection?.areaKm2) && !Number.isFinite(state.data.detection?.confidence);
  el("#orientation-row").hidden = stageIndex === 0 || !Number.isFinite(state.data.detection?.orientationDeg);
  el("#threshold-row").hidden = stageIndex !== 1 || !Number.isFinite(state.data.detection?.threshold);
  el("#pixel-count-row").hidden = stageIndex !== 1 || !Number.isFinite(state.data.detection?.oilPixelCount);
  el("#detection-preview").hidden = stageIndex !== 1 || !state.data.assets.thumbnail; if (!el("#detection-preview").hidden) { el("#detection-preview").src = state.data.assets.thumbnail; el("#detection-preview").alt = bundled ? "Precomputed synthetic detection thumbnail" : "Exported detection preview"; }
  el("#ais-heading").textContent = bundled || state.data.readiness?.aisSynthetic ? "Synthetic AIS candidates" : "AIS candidates";
  el("#summary-kicker").textContent = stageIndex === 0 ? "SAR scene" : "Detection";
  const activeLayers = { detection: stageIndex >= 1, posterior: stageIndex >= 2, vessels: stageIndex === 3, forecast: stageIndex >= 4 };
  Object.entries(activeLayers).forEach(([name, visible]) => setLayerVisible(name, visible));
  const layerStages = { detection: 1, posterior: 2, vessels: 3, forecast: 4 };
  document.querySelectorAll("[data-layer]").forEach((input) => { const available = stageIndex >= layerStages[input.dataset.layer] && state.data.availability[layerStages[input.dataset.layer]]; input.disabled = !available; input.checked = activeLayers[input.dataset.layer] && available; });
}
function setDemoStage(nextStage) {
  if (!state.data) return;
  const bounded = Math.max(0, Math.min(nextStage, state.data.stages.length - 1));
  if (!state.data.availability[bounded]) return;
  state.demoStage = bounded;
  state.demoMaxStage = Math.max(state.demoMaxStage, state.demoStage);
  if (state.demoStage === 5 && state.data.forecasts?.length) { state.horizonIndex = state.data.forecasts.length - 1; renderForecast(state.data, state.horizonIndex); }
  if (state.demoStage === 0) state.demoAsset = "sar";
  renderStages(state.data.stages); renderCoast(state.data); renderAnalysisCues(state.data); renderMapLabels(state.data); updateDemoGuide(); updateDemoAssetViewer();
}
function layerElement(name) { return el(`#${name === "vessels" ? "vessel" : name}-layer`); }
function setLayerVisible(name, visible) { layerElement(name).toggleAttribute("hidden", !visible); }
function resetPresentation() { state.horizonIndex = 0; document.querySelectorAll("[data-layer]").forEach((input) => { input.checked = true; setLayerVisible(input.dataset.layer, true); }); el("#candidate-list").classList.remove("expanded"); el("#toggle-candidates").textContent = "All candidates"; }
function renderIncident(data) { state.data = data; state.demoStage = 0; state.demoMaxStage = 0; state.demoAsset = "sar"; resetPresentation(); renderStages(data.stages); renderCoast(data); renderDetection(data); renderPosterior(data); renderVessels(data); renderAnalysisCues(data); renderMapLabels(data); renderDetails(data); renderHorizonButtons(data); renderForecast(data, 0); updateDemoGuide(); updateDemoAssetViewer(); }
function bindControls() { if (state.controlsBound) return; state.controlsBound = true; el("#horizon-buttons").addEventListener("click", (event) => { const button = event.target.closest("[data-horizon]"); if (!button) return; state.horizonIndex = Number(button.dataset.horizon); renderForecast(state.data, state.horizonIndex); }); document.querySelectorAll("[data-layer]").forEach((input) => input.addEventListener("change", () => { setLayerVisible(input.dataset.layer, input.checked); })); el("#toggle-candidates").addEventListener("click", (event) => { const expanded = el("#candidate-list").classList.toggle("expanded"); event.currentTarget.textContent = expanded ? "Top candidates" : "All candidates"; }); el("#reset-view").addEventListener("click", () => { resetPresentation(); setDemoStage(0); renderForecast(state.data, 0); }); el("#incident-selector").addEventListener("change", (event) => loadSelectedIncident(event.target.value)); }
function bindDemoGuideControls() {
  el("#guide-back").addEventListener("click", () => setDemoStage(state.demoStage - 1));
  el("#guide-next").addEventListener("click", () => { if (state.demoStage === state.data.stages.length - 1) { state.demoMaxStage = 0; state.horizonIndex = 0; renderForecast(state.data, 0); } setDemoStage(state.demoStage === state.data.stages.length - 1 ? 0 : state.demoStage + 1); });
  el("#stage-list").addEventListener("click", (event) => { const button = event.target.closest("[data-demo-stage]"); if (button && !button.disabled && Number(button.dataset.demoStage) <= state.demoMaxStage) setDemoStage(Number(button.dataset.demoStage)); });
  el("#demo-asset-tabs").addEventListener("click", (event) => { const button = event.target.closest("[data-demo-asset]"); if (button && !button.disabled) { state.demoAsset = button.dataset.demoAsset; updateDemoAssetViewer(); } });
}
function populateSelector(entries) { const selector = el("#incident-selector"); selector.replaceChildren(...entries.map((entry) => new Option(entry.label, entry.id))); selector.disabled = false; }
async function loadArtifacts(raw, baseUrl) { const entries = Object.entries(raw.artifacts || {}); const loaded = await Promise.all(entries.map(async ([name, path]) => { try { const response = await fetch(absoluteAsset(path, baseUrl), { cache: "no-store" }); return [name, response.ok ? await response.json() : null]; } catch { return [name, null]; } })); return Object.fromEntries(loaded); }
async function loadSelectedIncident(id) {
  const sequence = ++state.loadSequence, entry = state.incidents.find((item) => item.id === id);
  if (!entry) return loadSelectedIncident(BUNDLED_DEMO_ID);
  const loading = el("#loading"); loading.classList.remove("hidden", "error"); loading.textContent = `Loading ${entry.label}…`;
  try {
    const response = await fetch(entry.data, { cache: "no-store" }); if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const raw = await response.json(); if (id !== BUNDLED_DEMO_ID && (!raw || raw.caseId !== id)) throw new Error("Invalid exported incident metadata");
    const artifacts = id === BUNDLED_DEMO_ID ? {} : await loadArtifacts(raw, response.url);
    if (sequence !== state.loadSequence) return;
    const normalized = normalizeIncident(raw, response.url, artifacts); if (!normalized.availability[0]) throw new Error("No usable scene export");
    renderIncident(normalized); el("#incident-selector").value = id; loading.classList.add("hidden");
  } catch (error) {
    if (sequence !== state.loadSequence) return;
    console.warn(`Could not load ${entry.label}; opening bundled demonstration.`, error);
    if (id !== BUNDLED_DEMO_ID) return loadSelectedIncident(BUNDLED_DEMO_ID);
    loading.classList.add("error"); loading.textContent = "Bundled demonstration could not be loaded.";
  }
}
async function initialize() { bindControls(); bindDemoGuideControls(); const bundled = { id: BUNDLED_DEMO_ID, label: "Bundled end-to-end demo", data: "data/incident-demo.json" }; let indexed = []; try { const response = await fetch("data/demo-index.json", { cache: "no-store" }); if (response.ok) indexed = (await response.json()).incidents || []; } catch (error) { console.info("Frozen incident index unavailable; showing bundled demo.", error); } state.incidents = [bundled, ...indexed.filter((item) => item?.caseId && item?.data).map((item) => ({ id: item.caseId, label: item.label || item.caseId, data: `data/${item.data}` }))]; populateSelector(state.incidents); const requested = new URLSearchParams(location.search).get("incident"); await loadSelectedIncident(state.incidents.some((item) => item.id === requested) ? requested : BUNDLED_DEMO_ID); }
function bindLanding() { el("#launch-analysis").addEventListener("click", () => { el("#landing-screen").hidden = true; el("#dashboard").hidden = false; }); }
bindLanding();
initialize();

"""Generate self-contained HTML replay viewer from benchmark run data."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..scenarios import get_scenario


def _extract_action(response: str) -> str:
    """Extract action from model response (same regex as analyzer)."""
    m = re.search(r"ACTION:\s*(\S+)", response, re.IGNORECASE)
    if m:
        return m.group(1).strip().lower().rstrip(".")
    return "unknown"


def _compute_phase_weights(scenario_name: str) -> list[dict]:
    """Get phase weights for the heatmap."""
    scenario = get_scenario(scenario_name)
    weights = []
    for (start, end), w in scenario.phase_anomaly_weights.items():
        weights.append({"start": start, "end": end, "weights": w})
    return weights


def _compute_per_tick_metrics(results: list[dict]) -> list[dict]:
    """Compute DA and FC per tick from raw results."""
    ticks = []
    for r in results:
        gt = r["ground_truth"]
        chosen = _extract_action(r["response"])
        acceptable = [a.lower() for a in gt.get("acceptable_actions", [])]
        da = 1.0 if chosen in acceptable else 0.0

        # Simple FC: count factor headings mentioned in response
        response_lower = r["response"].lower()
        factor_names = ["load", "generation", "frequency", "voltage", "weather", "reserve"]
        mentioned = sum(1 for f in factor_names if f in response_lower)
        fc = mentioned / len(factor_names)

        ticks.append({
            "tick": r["tick_number"],
            "chosen_action": chosen,
            "correct_action": gt["correct_action"],
            "acceptable_actions": gt.get("acceptable_actions", []),
            "anomalous_factors": gt.get("anomalous_factors", []),
            "is_multi_factor": gt.get("is_multi_factor", False),
            "da": da,
            "fc": fc,
            "response": r["response"],
            "context_truncated": r.get("context_truncated", False),
        })
    return ticks


def generate_viewer_html(
    results_dir: str | Path,
    scenario_name: str = "power_grid",
    output_path: str | Path | None = None,
) -> Path:
    """Generate a self-contained HTML viewer from run data.

    Args:
        results_dir: Directory containing raw_results.jsonl
        scenario_name: Scenario name for phase weights
        output_path: Where to write the HTML. Defaults to results_dir/viewer.html

    Returns:
        Path to the generated HTML file.
    """
    results_dir = Path(results_dir)
    raw_file = results_dir / "raw_results.jsonl"
    if not raw_file.exists():
        raise FileNotFoundError(f"No raw_results.jsonl in {results_dir}")

    results = []
    with open(raw_file) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    tick_data = _compute_per_tick_metrics(results)
    phase_weights = _compute_phase_weights(scenario_name)

    # Compute summary stats
    da_values = [t["da"] for t in tick_data]
    fc_values = [t["fc"] for t in tick_data]
    n = len(da_values)
    window = min(40, n)
    da_first = sum(da_values[:window]) / window if window else 0
    da_last = sum(da_values[-window:]) / window if window else 0
    dfg = da_first - da_last

    summary = {
        "total_ticks": n,
        "da_overall": sum(da_values) / n if n else 0,
        "fc_overall": sum(fc_values) / n if n else 0,
        "da_first_40": da_first,
        "da_last_40": da_last,
        "dfg": dfg,
        "scenario": scenario_name,
    }

    viewer_data = {
        "ticks": tick_data,
        "phase_weights": phase_weights,
        "summary": summary,
    }

    html = _build_html(viewer_data)

    if output_path is None:
        output_path = results_dir / "viewer.html"
    output_path = Path(output_path)
    output_path.write_text(html)
    return output_path


def _build_html(data: dict) -> str:
    """Build the self-contained HTML viewer."""
    data_json = json.dumps(data)
    summary = data["summary"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CoherenceBench Replay Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #0d1117; color: #c9d1d9; }}
.header {{ padding: 16px 24px; border-bottom: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center; }}
.header h1 {{ font-size: 18px; font-weight: 600; color: #f0f6fc; }}
.header .stats {{ font-size: 13px; color: #8b949e; }}
.header .stats span {{ margin-left: 16px; }}
.header .stats .da {{ color: #58a6ff; }}
.header .stats .dfg {{ color: #f85149; }}

.grid {{ display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: auto auto 1fr; gap: 1px; background: #21262d; height: calc(100vh - 52px); }}
.pane {{ background: #0d1117; padding: 12px 16px; overflow: auto; }}
.pane-title {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; color: #8b949e; margin-bottom: 8px; }}

.chart-pane {{ grid-column: 1 / -1; height: 220px; }}
.heatmap-pane {{ grid-column: 1 / -1; height: 140px; overflow-x: auto; }}
.transcript-pane {{ min-height: 300px; }}
.score-pane {{ min-height: 300px; }}

/* Scrubber */
.scrubber {{ grid-column: 1 / -1; height: 48px; padding: 8px 16px; cursor: pointer; position: relative; }}
.scrubber-bar {{ height: 24px; background: #161b22; border-radius: 4px; position: relative; overflow: hidden; }}
.scrubber-fill {{ height: 100%; background: #21262d; position: absolute; left: 0; top: 0; }}
.scrubber-marker {{ position: absolute; top: 0; width: 2px; height: 100%; background: #58a6ff; z-index: 2; }}
.phase-label {{ position: absolute; top: 26px; font-size: 10px; color: #484f58; }}

/* Heatmap */
.heatmap-canvas {{ display: block; }}

/* Transcript */
.tick-header {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #f0f6fc; }}
.action-correct {{ color: #3fb950; font-weight: 600; }}
.action-wrong {{ color: #f85149; font-weight: 600; }}
.anomaly-tag {{ display: inline-block; padding: 2px 6px; background: #f8514922; color: #f85149; border-radius: 3px; font-size: 11px; margin-right: 4px; margin-bottom: 4px; }}
.response-text {{ font-size: 12px; line-height: 1.6; white-space: pre-wrap; background: #161b22; padding: 12px; border-radius: 6px; max-height: 400px; overflow-y: auto; margin-top: 8px; }}

/* Score dashboard */
.score-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.score-card {{ background: #161b22; padding: 12px; border-radius: 6px; text-align: center; }}
.score-card .value {{ font-size: 28px; font-weight: 700; }}
.score-card .label {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
.da-color {{ color: #58a6ff; }}
.fc-color {{ color: #3fb950; }}
.dfg-pos {{ color: #f85149; }}
.dfg-neg {{ color: #3fb950; }}
</style>
</head>
<body>

<div class="header">
  <h1>CoherenceBench Replay Viewer</h1>
  <div class="stats">
    <span>Scenario: <b>{summary['scenario']}</b></span>
    <span>Ticks: <b>{summary['total_ticks']}</b></span>
    <span class="da">DA: <b>{summary['da_overall']:.1%}</b></span>
    <span class="dfg">DFG: <b>{summary['dfg']:+.1%}</b></span>
  </div>
</div>

<div class="grid">
  <!-- Collapse Curve -->
  <div class="pane chart-pane">
    <div class="pane-title">Collapse Curve</div>
    <canvas id="collapseChart"></canvas>
  </div>

  <!-- Scrubber -->
  <div class="scrubber" id="scrubber">
    <div class="scrubber-bar" id="scrubberBar">
      <div class="scrubber-marker" id="scrubberMarker" style="left: 0%"></div>
    </div>
    <div id="phaseLabels"></div>
  </div>

  <!-- Heatmap -->
  <div class="pane heatmap-pane">
    <div class="pane-title">Subsystem Anomaly Heatmap (ground truth)</div>
    <canvas id="heatmapCanvas" class="heatmap-canvas"></canvas>
  </div>

  <!-- Transcript -->
  <div class="pane transcript-pane" id="transcriptPane">
    <div class="pane-title">Transcript</div>
    <div id="transcriptContent">Click a tick to view the model's response.</div>
  </div>

  <!-- Score Dashboard -->
  <div class="pane score-pane">
    <div class="pane-title">Score Dashboard</div>
    <div class="score-grid" id="scoreDashboard"></div>
  </div>
</div>

<script id="benchmarkData" type="application/json">{data_json}</script>

<script>
const DATA = JSON.parse(document.getElementById('benchmarkData').textContent);
const ticks = DATA.ticks;
const phases = DATA.phase_weights;
const summary = DATA.summary;
let currentTick = 0;

// --- Collapse Curve ---
const daValues = ticks.map(t => t.da);
const fcValues = ticks.map(t => t.fc);
const tickLabels = ticks.map(t => t.tick);

// Compute rolling averages (window=10)
function rollingAvg(arr, w) {{
  const result = [];
  for (let i = 0; i < arr.length; i++) {{
    const start = Math.max(0, i - w + 1);
    const slice = arr.slice(start, i + 1);
    result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }}
  return result;
}}

const daRolling = rollingAvg(daValues, 10);
const fcRolling = rollingAvg(fcValues, 10);

// Phase boundary annotations
const phaseBoundaries = phases.map(p => p.start).filter(s => s > 0);

const ctx = document.getElementById('collapseChart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: tickLabels,
    datasets: [
      {{
        label: 'DA (rolling avg)',
        data: daRolling,
        borderColor: '#58a6ff',
        backgroundColor: '#58a6ff22',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
      }},
      {{
        label: 'FC (rolling avg)',
        data: fcRolling,
        borderColor: '#3fb950',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 1.5,
        borderDash: [4, 4],
      }}
    ]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {{
      x: {{ display: true, grid: {{ color: '#21262d' }}, ticks: {{ color: '#484f58', maxTicksLimit: 20 }} }},
      y: {{ display: true, min: 0, max: 1, grid: {{ color: '#21262d' }}, ticks: {{ color: '#484f58', callback: v => (v * 100) + '%' }} }}
    }},
    plugins: {{
      legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }},
      tooltip: {{ enabled: true }}
    }}
  }}
}});

// --- Scrubber ---
const scrubberBar = document.getElementById('scrubberBar');
const scrubberMarker = document.getElementById('scrubberMarker');
const phaseLabelsEl = document.getElementById('phaseLabels');

phases.forEach((p, i) => {{
  if (p.start > 0) {{
    const pct = (p.start / ticks.length) * 100;
    const div = document.createElement('div');
    div.style.cssText = `position:absolute;left:${{pct}}%;top:0;height:24px;width:1px;background:#484f58;`;
    scrubberBar.appendChild(div);
  }}
  const label = document.createElement('div');
  label.className = 'phase-label';
  const midPct = ((p.start + p.end) / 2 / ticks.length) * 100;
  label.style.left = midPct + '%';
  label.textContent = `Phase ${{i + 1}}`;
  phaseLabelsEl.appendChild(label);
}});

function selectTick(idx) {{
  currentTick = Math.max(0, Math.min(idx, ticks.length - 1));
  const pct = (currentTick / (ticks.length - 1)) * 100;
  scrubberMarker.style.left = pct + '%';
  updateTranscript();
  updateScoreDashboard();
}}

scrubberBar.addEventListener('click', (e) => {{
  const rect = scrubberBar.getBoundingClientRect();
  const pct = (e.clientX - rect.left) / rect.width;
  selectTick(Math.round(pct * (ticks.length - 1)));
}});

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowRight') selectTick(currentTick + 1);
  if (e.key === 'ArrowLeft') selectTick(currentTick - 1);
}});

// --- Heatmap ---
const heatmapCanvas = document.getElementById('heatmapCanvas');
const factorNames = ['load', 'generation', 'frequency', 'voltage', 'weather', 'reserve'];
const cellW = Math.max(4, Math.floor((heatmapCanvas.parentElement.clientWidth - 32) / ticks.length));
const cellH = 16;
heatmapCanvas.width = cellW * ticks.length;
heatmapCanvas.height = cellH * factorNames.length;
const hctx = heatmapCanvas.getContext('2d');

function getPhaseWeight(tickIdx, factor) {{
  for (const p of phases) {{
    if (tickIdx >= p.start && tickIdx < p.end) return p.weights[factor] || 0;
  }}
  return 0;
}}

for (let fi = 0; fi < factorNames.length; fi++) {{
  for (let ti = 0; ti < ticks.length; ti++) {{
    const w = getPhaseWeight(ti, factorNames[fi]);
    const alpha = Math.min(w * 2.5, 1);
    hctx.fillStyle = `rgba(88, 166, 255, ${{alpha}})`;
    hctx.fillRect(ti * cellW, fi * cellH, cellW - 1, cellH - 1);

    // Dot if this factor was actually anomalous
    if (ticks[ti].anomalous_factors.includes(factorNames[fi])) {{
      hctx.fillStyle = '#f85149';
      hctx.beginPath();
      hctx.arc(ti * cellW + cellW/2, fi * cellH + cellH/2, 2, 0, Math.PI * 2);
      hctx.fill();
    }}
  }}
}}

// Factor labels
hctx.fillStyle = '#8b949e';
hctx.font = '10px monospace';
// Labels drawn outside canvas — use overlay div instead

// --- Transcript ---
function updateTranscript() {{
  const t = ticks[currentTick];
  const isCorrect = t.da > 0;
  const actionClass = isCorrect ? 'action-correct' : 'action-wrong';

  let anomalyTags = '';
  if (t.anomalous_factors.length > 0) {{
    anomalyTags = t.anomalous_factors.map(f => `<span class="anomaly-tag">${{f}}</span>`).join('');
  }} else {{
    anomalyTags = '<span style="color:#8b949e;font-size:11px;">no anomaly</span>';
  }}

  document.getElementById('transcriptContent').innerHTML = `
    <div class="tick-header">Tick ${{t.tick}} / ${{ticks.length}}</div>
    <div style="margin-bottom:8px;">
      Action: <span class="${{actionClass}}">${{t.chosen_action}}</span>
      ${{isCorrect ? '&#10003;' : '&#10007; expected: ' + t.correct_action}}
      ${{t.is_multi_factor ? ' <span style="color:#d2a8ff;font-size:11px;">[multi-factor]</span>' : ''}}
    </div>
    <div style="margin-bottom:8px;">Anomalous: ${{anomalyTags}}</div>
    <div class="response-text">${{escapeHtml(t.response)}}</div>
  `;
}}

function escapeHtml(text) {{
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}}

// --- Score Dashboard ---
function updateScoreDashboard() {{
  const idx = currentTick;
  const window = 40;

  // Cumulative DA/FC up to current tick
  const daSlice = daValues.slice(0, idx + 1);
  const fcSlice = fcValues.slice(0, idx + 1);
  const daAvg = daSlice.reduce((a, b) => a + b, 0) / daSlice.length;
  const fcAvg = fcSlice.reduce((a, b) => a + b, 0) / fcSlice.length;

  // DFG at current position
  const first = daValues.slice(0, Math.min(window, idx + 1));
  const last = idx >= window ? daValues.slice(Math.max(0, idx + 1 - window), idx + 1) : first;
  const daFirst = first.reduce((a, b) => a + b, 0) / first.length;
  const daLast = last.reduce((a, b) => a + b, 0) / last.length;
  const dfg = daFirst - daLast;
  const dfgClass = dfg > 0 ? 'dfg-pos' : 'dfg-neg';

  document.getElementById('scoreDashboard').innerHTML = `
    <div class="score-card"><div class="value da-color">${{(daAvg * 100).toFixed(1)}}%</div><div class="label">DA (cumulative)</div></div>
    <div class="score-card"><div class="value fc-color">${{(fcAvg * 100).toFixed(1)}}%</div><div class="label">FC (cumulative)</div></div>
    <div class="score-card"><div class="value da-color">${{(daFirst * 100).toFixed(1)}}%</div><div class="label">DA @ first ${{Math.min(window, idx + 1)}}</div></div>
    <div class="score-card"><div class="value da-color">${{(daLast * 100).toFixed(1)}}%</div><div class="label">DA @ last ${{Math.min(window, idx + 1)}}</div></div>
    <div class="score-card"><div class="value ${{dfgClass}}">${{dfg >= 0 ? '+' : ''}}${{(dfg * 100).toFixed(1)}}%</div><div class="label">DFG (fade gap)</div></div>
    <div class="score-card"><div class="value" style="color:#c9d1d9;">${{idx + 1}}</div><div class="label">Current tick</div></div>
  `;
}}

// Initialize
selectTick(0);
</script>
</body>
</html>"""

#!/usr/bin/env python3
"""
RLHF Audit Visualizer — Generate interactive HTML heatmap from audit report.

Usage:
  python visualize.py audit_report.json                    # single model
  python visualize.py report_a.json report_b.json ...      # multi-model comparison
"""

import json
import sys
import os

def load_report(path):
    with open(path) as f:
        return json.load(f)


def generate_heatmap_html(reports):
    models = []
    for r in reports:
        name = r["model_name"]
        n_layers = r["n_layers"]
        scan = r["scan"]
        report = r["report"]
        baseline = r["baseline"]

        cells = []
        for layer_idx in range(n_layers):
            for comp in ["attn", "mlp"]:
                key = f"L{layer_idx}_{comp}"
                if key in scan:
                    v = scan[key]
                    cells.append({
                        "layer": layer_idx, "comp": comp, "key": key,
                        "collapsed": v["collapsed"],
                        "format_changed": v["format_changed"],
                        "rep": v["avg_trigram_rep"],
                        "disc": v["disclaimer_rate"],
                        "error": v.get("error", False),
                    })

        models.append({
            "name": name,
            "n_layers": n_layers,
            "strategy": report["summary"]["rlhf_strategy"],
            "coverage": report["summary"]["coverage_pct"],
            "crashed": report["summary"]["crashed"],
            "fmt_locked": report["summary"]["format_locked"],
            "stable": report["summary"]["stable"],
            "verdict": report["summary"]["verdict"],
            "band": report["format_lock_band"],
            "cells": cells,
            "baseline_disc": baseline["disclaimer_rate"],
            "baseline_rep": baseline["avg_rep"],
        })

    models_json = json.dumps(models)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RLHF Alignment Audit — Structural Heatmap</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0a0a1a; color:#ccc; font-family:'SF Mono','Fira Code',monospace; padding:20px; }}
h1 {{ text-align:center; font-size:22px; color:#e0e0e0; letter-spacing:2px; margin-bottom:8px; }}
h1 .rlhf {{ color:#ff6b6b; }}
.subtitle {{ text-align:center; font-size:11px; color:#555; margin-bottom:30px; }}
.model-section {{ margin-bottom:40px; }}
.model-header {{ display:flex; align-items:center; gap:20px; margin-bottom:12px; flex-wrap:wrap; }}
.model-name {{ font-size:16px; color:#fff; font-weight:700; }}
.badge {{ padding:3px 10px; border-radius:10px; font-size:10px; font-weight:700; }}
.badge-surgical {{ background:rgba(76,175,80,0.2); color:#4CAF50; border:1px solid #4CAF50; }}
.badge-diffuse {{ background:rgba(255,152,0,0.2); color:#FF9800; border:1px solid #FF9800; }}
.badge-moderate {{ background:rgba(33,150,243,0.2); color:#2196F3; border:1px solid #2196F3; }}
.badge-minimal {{ background:rgba(156,39,176,0.2); color:#9C27B0; border:1px solid #9C27B0; }}
.stats {{ display:flex; gap:20px; font-size:11px; color:#888; }}
.stats span {{ color:#aaa; }}
.verdict {{ font-size:11px; color:#667; margin-bottom:8px; font-style:italic; }}
.heatmap {{ display:flex; gap:2px; flex-direction:column; }}
.heatmap-row {{ display:flex; align-items:center; gap:1px; }}
.row-label {{ width:35px; font-size:9px; color:#555; text-align:right; padding-right:4px; }}
.cell {{
  width:16px; height:16px; border-radius:2px; cursor:pointer;
  transition:transform 0.15s; position:relative;
}}
.cell:hover {{ transform:scale(1.8); z-index:10; }}
.layer-labels {{ display:flex; gap:1px; margin-left:36px; margin-top:2px; }}
.layer-label {{ width:16px; font-size:7px; color:#444; text-align:center; }}
.legend {{ display:flex; gap:16px; margin-top:12px; font-size:10px; }}
.legend-item {{ display:flex; align-items:center; gap:5px; }}
.legend-dot {{ width:12px; height:12px; border-radius:2px; }}
.tooltip {{
  display:none; position:fixed; background:#1a1a2e; border:1px solid #333;
  border-radius:6px; padding:8px 12px; font-size:10px; z-index:1000;
  pointer-events:none; max-width:300px; line-height:1.5;
}}
.tooltip .key {{ color:#fff; font-weight:700; }}
.tooltip .val {{ color:#4fc3f7; }}
.band-indicator {{
  height:3px; margin-left:36px; margin-bottom:4px; display:flex; gap:1px;
}}
.band-cell {{ width:16px; height:3px; border-radius:1px; }}
</style>
</head>
<body>

<h1><span class="rlhf">RLHF</span> Alignment Audit</h1>
<div class="subtitle">Structural quality assessment via component ablation · Each cell = one component (layer × attn/mlp)</div>

<div id="content"></div>
<div class="tooltip" id="tooltip"></div>

<script>
const MODELS = {models_json};

const content = document.getElementById('content');
const tooltip = document.getElementById('tooltip');

function cellColor(cell) {{
  if (cell.error) return '#333';
  if (cell.collapsed) return `rgb(${{Math.min(255, 180 + cell.rep*200)}},40,40)`;
  if (cell.format_changed) {{
    const b = Math.max(80, 220 - cell.rep * 800);
    return `rgb(${{b}},${{b}},40)`;
  }}
  const b = Math.max(50, 180 - cell.rep * 600);
  return `rgb(40,${{b}},40)`;
}}

function badgeClass(strategy) {{
  return 'badge-' + strategy.toLowerCase();
}}

MODELS.forEach(m => {{
  const section = document.createElement('div');
  section.className = 'model-section';

  const header = document.createElement('div');
  header.className = 'model-header';
  header.innerHTML = `
    <span class="model-name">${{m.name}}</span>
    <span class="badge ${{badgeClass(m.strategy)}}">${{m.strategy}}</span>
    <div class="stats">
      <span>${{m.coverage}}% format-locked</span> ·
      <span>${{m.fmt_locked}} locked</span> ·
      <span>${{m.stable}} free</span> ·
      <span>${{m.crashed}} crash</span>
    </div>
  `;
  section.appendChild(header);

  const verdict = document.createElement('div');
  verdict.className = 'verdict';
  verdict.textContent = m.verdict;
  section.appendChild(verdict);

  // Band indicator
  const bandDiv = document.createElement('div');
  bandDiv.className = 'band-indicator';
  for (let i = 0; i < m.n_layers; i++) {{
    const bc = document.createElement('div');
    bc.className = 'band-cell';
    const inBand = i >= m.band.start_layer && i <= m.band.end_layer;
    bc.style.background = inBand ? 'rgba(255,107,107,0.4)' : 'rgba(255,255,255,0.05)';
    bandDiv.appendChild(bc);
  }}
  section.appendChild(bandDiv);

  // Heatmap
  const heatmap = document.createElement('div');
  heatmap.className = 'heatmap';

  ['attn', 'mlp'].forEach(comp => {{
    const row = document.createElement('div');
    row.className = 'heatmap-row';

    const label = document.createElement('div');
    label.className = 'row-label';
    label.textContent = comp;
    row.appendChild(label);

    for (let i = 0; i < m.n_layers; i++) {{
      const cell = m.cells.find(c => c.layer === i && c.comp === comp);
      const div = document.createElement('div');
      div.className = 'cell';

      if (cell) {{
        div.style.background = cellColor(cell);
        div.addEventListener('mouseenter', e => {{
          tooltip.style.display = 'block';
          tooltip.style.left = (e.clientX + 12) + 'px';
          tooltip.style.top = (e.clientY - 10) + 'px';
          const status = cell.error ? 'ERROR' : (cell.collapsed ? 'CRASHED' :
                         (cell.format_changed ? 'FORMAT UNLOCKED' : 'STABLE'));
          tooltip.innerHTML = `
            <div class="key">${{cell.key}}</div>
            <div>Status: <span class="val">${{status}}</span></div>
            <div>Repetition: <span class="val">${{(cell.rep*100).toFixed(1)}}%</span></div>
            <div>Disclaimer: <span class="val">${{(cell.disc*100).toFixed(0)}}%</span></div>
          `;
        }});
        div.addEventListener('mouseleave', () => tooltip.style.display = 'none');
      }} else {{
        div.style.background = '#111';
      }}
      row.appendChild(div);
    }}
    heatmap.appendChild(row);
  }});

  section.appendChild(heatmap);

  // Layer labels
  const labels = document.createElement('div');
  labels.className = 'layer-labels';
  for (let i = 0; i < m.n_layers; i++) {{
    const l = document.createElement('div');
    l.className = 'layer-label';
    l.textContent = i % 5 === 0 ? i : '';
    labels.appendChild(l);
  }}
  section.appendChild(labels);

  // Legend
  const legend = document.createElement('div');
  legend.className = 'legend';
  [['Stable', '#28a745'], ['Format unlocked', '#c8c828'], ['Crashed', '#dc3545'], ['Not scanned', '#111']].forEach(([t,c]) => {{
    legend.innerHTML += `<div class="legend-item"><div class="legend-dot" style="background:${{c}}"></div>${{t}}</div>`;
  }});
  section.appendChild(legend);

  content.appendChild(section);
}});
</script>
</body>
</html>'''
    return html


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize.py report1.json [report2.json ...]")
        sys.exit(1)

    reports = [load_report(p) for p in sys.argv[1:]]
    html = generate_heatmap_html(reports)

    out = sys.argv[1].replace(".json", "_heatmap.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()

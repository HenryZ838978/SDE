"""Generate SVG bar chart comparing combo ablation results."""
import json

with open("data/qwen3-14b-awq/combo_ablation.json") as f:
    d = json.load(f)

configs = [
    ("baseline", "#888"),
    ("light_3mlp_s03", "#4CAF50"),
    ("medium_5mlp_s03", "#2196F3"),
    ("heavy_6mlp_s00", "#FF9800"),
    ("midband_s03", "#9C27B0"),
]

W, H = 700, 320
BAR_W = 50
GAP = 20
MARGIN_L, MARGIN_T = 80, 50
CHART_H = 200

svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}" style="background:#0a0a1a;font-family:monospace">')

svg.append(f'<text x="{W//2}" y="20" fill="#ccc" font-size="12" text-anchor="middle" font-weight="bold">'
           f'SDE Combo Ablation — Personality Expression Unlocked</text>')
svg.append(f'<text x="{W//2}" y="34" fill="#888" font-size="8" text-anchor="middle">'
           f'Disclaimer rate = "作为AI助手..." frequency | Lower = more personality</text>')

# Y axis
for pct in [0, 25, 50, 75, 100]:
    y = MARGIN_T + CHART_H - (pct / 100 * CHART_H)
    svg.append(f'<line x1="{MARGIN_L}" y1="{y}" x2="{W-20}" y2="{y}" stroke="#222" stroke-width="0.5"/>')
    svg.append(f'<text x="{MARGIN_L-5}" y="{y+3}" fill="#666" font-size="8" text-anchor="end">{pct}%</text>')

svg.append(f'<text x="15" y="{MARGIN_T + CHART_H//2}" fill="#888" font-size="9" '
           f'text-anchor="middle" transform="rotate(-90 15 {MARGIN_T + CHART_H//2})">Disclaimer Rate</text>')

for i, (name, color) in enumerate(configs):
    data = d[name]
    x = MARGIN_L + i * (BAR_W + GAP) + GAP

    disclaimer = data["disclaimer_rate"] * 100
    collapse = data["collapse_rate"] * 100
    rep = data["avg_trigram_rep"] * 100

    bar_h = max(2, disclaimer / 100 * CHART_H)
    by = MARGIN_T + CHART_H - bar_h
    svg.append(f'<rect x="{x}" y="{by}" width="{BAR_W}" height="{bar_h}" fill="{color}" rx="3" opacity="0.85"/>')
    svg.append(f'<text x="{x + BAR_W//2}" y="{by - 4}" fill="{color}" font-size="9" text-anchor="middle">'
               f'{disclaimer:.0f}%</text>')

    rep_h = max(1, rep / 100 * CHART_H)
    ry = MARGIN_T + CHART_H - rep_h
    svg.append(f'<rect x="{x + BAR_W - 8}" y="{ry}" width="8" height="{rep_h}" fill="#FF5252" rx="1" opacity="0.6"/>')

    label = name.replace("_", "\n")
    short = name[:12] + ("…" if len(name) > 12 else "")
    svg.append(f'<text x="{x + BAR_W//2}" y="{MARGIN_T + CHART_H + 14}" fill="#aaa" font-size="7" '
               f'text-anchor="middle">{short}</text>')

# Legend
ly = H - 30
svg.append(f'<rect x="{MARGIN_L}" y="{ly}" width="10" height="10" fill="#4CAF50" rx="2"/>')
svg.append(f'<text x="{MARGIN_L+14}" y="{ly+9}" fill="#aaa" font-size="8">Disclaimer bar</text>')
svg.append(f'<rect x="{MARGIN_L+120}" y="{ly}" width="10" height="10" fill="#FF5252" rx="2" opacity="0.6"/>')
svg.append(f'<text x="{MARGIN_L+134}" y="{ly+9}" fill="#aaa" font-size="8">Repetition rate (red = danger)</text>')

# Key finding
svg.append(f'<text x="{W//2}" y="{H-8}" fill="#4fc3f7" font-size="8" text-anchor="middle">'
           f'heavy_6mlp_s00: 0% disclaimer, 0% collapse, 4.3% repetition — format lock fully dissolved</text>')

svg.append('</svg>')

with open("figures/combo_comparison.svg", "w") as f:
    f.write('\n'.join(svg))
print("Generated figures/combo_comparison.svg")

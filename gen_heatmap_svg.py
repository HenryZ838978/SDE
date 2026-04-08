"""Generate SVG heat map of component scan for README."""
import json

with open("data/qwen3-14b-awq/component_scan.json") as f:
    d = json.load(f)

scan = d["full_scan"]

CELL_W, CELL_H = 22, 14
MARGIN_LEFT, MARGIN_TOP = 50, 40
N_LAYERS = 40
COMPONENTS = ["attn", "mlp"]

svg_lines = []
total_w = MARGIN_LEFT + CELL_W * N_LAYERS + 20
total_h = MARGIN_TOP + CELL_H * 2 + 60

svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
                 f'width="{total_w}" height="{total_h}" style="background:#0a0a1a;font-family:monospace">')

svg_lines.append(f'<text x="{total_w//2}" y="18" fill="#ccc" font-size="11" text-anchor="middle" font-weight="bold">'
                 f'SDE Component Scan — Qwen3-14B-AWQ (40 layers × 2 components)</text>')
svg_lines.append(f'<text x="{total_w//2}" y="30" fill="#888" font-size="8" text-anchor="middle">'
                 f'Green=OK  Yellow=FORMAT_CHANGED  Red=COLLAPSED  Brightness=repetition rate</text>')

for ci, comp in enumerate(COMPONENTS):
    y = MARGIN_TOP + ci * CELL_H
    svg_lines.append(f'<text x="5" y="{y + CELL_H - 3}" fill="#aaa" font-size="9">{comp}</text>')

    for layer_idx in range(N_LAYERS):
        key = f"L{layer_idx}_{comp}"
        x = MARGIN_LEFT + layer_idx * CELL_W

        info = scan.get(key, {})
        collapsed = info.get("collapsed", False)
        fmt_changed = info.get("format_changed", False)
        rep = info.get("avg_trigram_rep", 0)

        if collapsed:
            fill = f"rgb({min(255, int(180 + rep*200))},40,40)"
        elif fmt_changed:
            brightness = max(80, 255 - int(rep * 800))
            fill = f"rgb({brightness},{brightness},40)"
        else:
            brightness = max(60, 200 - int(rep * 600))
            fill = f"rgb(40,{brightness},40)"

        svg_lines.append(f'<rect x="{x}" y="{y}" width="{CELL_W-1}" height="{CELL_H-1}" '
                         f'fill="{fill}" rx="2" />')

        if layer_idx % 5 == 0 and ci == 0:
            svg_lines.append(f'<text x="{x + CELL_W//2}" y="{MARGIN_TOP - 5}" fill="#666" '
                             f'font-size="7" text-anchor="middle">L{layer_idx}</text>')

# Legend
ly = MARGIN_TOP + CELL_H * 2 + 15
for i, (label, color) in enumerate([("OK (stable)", "#28a745"), ("FMT! (format changed)", "#c8c828"),
                                     ("CRASH (collapsed)", "#dc3545")]):
    lx = MARGIN_LEFT + i * 180
    svg_lines.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}" rx="2"/>')
    svg_lines.append(f'<text x="{lx+14}" y="{ly+9}" fill="#aaa" font-size="8">{label}</text>')

# Key insight annotation
ay = MARGIN_TOP + CELL_H * 2 + 35
svg_lines.append(f'<text x="{MARGIN_LEFT}" y="{ay}" fill="#4fc3f7" font-size="8">'
                 f'Only 3/80 components crash when fully removed. '
                 f'Format lock is distributed across L15-L28 mid-deep band.</text>')

svg_lines.append('</svg>')

svg_content = '\n'.join(svg_lines)
with open("figures/component_heatmap.svg", "w") as f:
    f.write(svg_content)
print("Generated figures/component_heatmap.svg")

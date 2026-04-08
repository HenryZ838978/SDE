"""Generate example audit reports from existing SDE scan data."""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from audit import analyze, generate_summary_text

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "examples")
os.makedirs(EXAMPLE_DIR, exist_ok=True)

datasets = [
    ("/cache/zhangjing/repo_sde/data/qwen3-14b-awq/component_scan.json", "Qwen3-14B-AWQ", 40),
    ("/cache/zhangjing/repo_sde/data/qwen25-7b-instruct/component_scan.json", "Qwen2.5-7B-Instruct", 28),
    ("/cache/zhangjing/repo_sde/data/qwen3-8b-awq/component_scan.json", "Qwen3-8B-AWQ", 36),
]

all_reports = []

for path, name, n_layers in datasets:
    print(f"Processing {name}...")
    with open(path) as f:
        raw = json.load(f)

    scan = raw.get("full_scan", raw.get("scan", {}))
    baseline_raw = raw.get("baseline", {})
    if isinstance(baseline_raw, list):
        reps = [s.get("trigram_rep", 0) for s in baseline_raw if isinstance(s, dict)]
        discs = [1 for s in baseline_raw if isinstance(s, dict) and
                 any(d in s.get("text_preview", "") for d in ["作为AI", "AI助手", "作为人工智能"])]
        baseline_data = {
            "avg_trigram_rep": sum(reps) / len(reps) if reps else 0,
            "disclaimer_rate": len(discs) / len(baseline_raw) if baseline_raw else 0,
        }
    else:
        baseline_data = baseline_raw

    # Normalize scan format
    normalized_scan = {}
    for key, v in scan.items():
        parts = key.split("_")
        layer = int(parts[0][1:])
        comp = parts[1]
        normalized_scan[key] = {
            "layer": layer,
            "component": comp,
            "avg_trigram_rep": v.get("avg_trigram_rep", 0),
            "collapsed": v.get("collapsed", False),
            "collapse_rate": 1.0 if v.get("collapsed") else 0.0,
            "disclaimer_rate": v.get("disclaimer_rate", 0),
            "template_rate": 0,
            "format_changed": v.get("format_changed", False),
            "sample": v.get("sample_text", "")[:200],
            "error": v.get("error", False),
        }

    report = analyze(
        {"disclaimer_rate": baseline_data.get("disclaimer_rate", 0),
         "avg_rep": baseline_data.get("avg_trigram_rep", 0)},
        normalized_scan, n_layers)

    full = {
        "model": name,
        "model_name": name,
        "n_layers": n_layers,
        "hidden_size": raw.get("hidden_size", 0),
        "audit_config": {"scale": raw.get("scan_scale", 0.0), "n_prompts": 8},
        "baseline": {
            "disclaimer_rate": baseline_data.get("disclaimer_rate", 0),
            "avg_rep": baseline_data.get("avg_trigram_rep", 0),
        },
        "scan": normalized_scan,
        "report": report,
    }

    out_path = os.path.join(EXAMPLE_DIR, f"audit_{name.lower().replace('-', '_').replace('.', '')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full, f, indent=2, ensure_ascii=False)

    md = generate_summary_text(report, name)
    md_path = out_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(md)

    all_reports.append(full)
    print(f"  → {out_path}")
    print(f"  → {md_path}")
    print(f"  Strategy: {report['summary']['rlhf_strategy']} ({report['summary']['coverage_pct']}%)")

# Generate combined visualization
from visualize import generate_heatmap_html
html = generate_heatmap_html(all_reports)
viz_path = os.path.join(EXAMPLE_DIR, "comparison_heatmap.html")
with open(viz_path, "w") as f:
    f.write(html)
print(f"\nGenerated comparison: {viz_path}")

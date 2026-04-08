# RLHF Alignment Audit

**Structural quality assessment for RLHF-aligned language models.**

> Every aligned model has a format lock — but *how* that lock is implemented reveals the quality of the alignment work.

## What This Does

Scans every structural component (self_attn + mlp, per layer) of a model by temporarily disabling each one, then measuring what changes. The result is a **structural X-ray** of RLHF alignment quality.

```
Surgical RLHF (Qwen3-14B):          Diffuse RLHF (Qwen2.5-7B):
┌─────────────────────────┐         ┌─────────────────────────┐
│ attn: ■■■■■■□□□□□□□□□■■ │         │ attn: ■■■■■■■■■■■■■■■■ │
│ mlp:  ■■□□□□□□□□□□■■■■■ │         │ mlp:  ■■■■■■■■■■■■■■■■ │
└─────────────────────────┘         └─────────────────────────┘
  □ = format-locked  ■ = free         □ = format-locked  ■ = free
  47.5% locked — concentrated          94.6% locked — everywhere
```

## Quick Start

```bash
# Audit any model (single GPU, ~30 min)
CUDA_VISIBLE_DEVICES=0 python audit.py \
  --model /path/to/model \
  --output my_report.json

# For AWQ/quantized models (avoid NaN with partial scale)
python audit.py --model /path/to/model --output report.json --scale 0.3

# Quick scan (2x faster, every other layer)
python audit.py --model /path/to/model --output report.json --stride 2

# Visualize (works with multiple reports)
python visualize.py report_a.json report_b.json
```

## Output

The tool produces:
- **`report.json`** — Machine-readable full scan data (AI-recoverable via git)
- **`report.md`** — Human-readable summary with verdict and recommendations
- **`report_heatmap.html`** — Interactive visual heatmap (hover for details)

## The Four Verdicts

| Verdict | Coverage | Interpretation |
|---------|----------|----------------|
| **MINIMAL** | <30% | Base model or very light alignment |
| **SURGICAL** | 30-55% | Precise RLHF — format lock in specific layer band |
| **MODERATE** | 55-80% | Broad alignment across most layers |
| **DIFFUSE** | >80% | Heavy-handed — format lock covers nearly everything |

## What It Reveals

1. **RLHF Strategy Quality** — Did the alignment team target specific layers, or spray everywhere?
2. **Dark Space Potential** — How much latent expressiveness is locked away?
3. **Surgical Targets** — Which components can be safely relaxed for SDE activation?
4. **Crash Boundaries** — Which 2-3 components must never be touched?

## Example Results

See `examples/` for pre-computed audits:

| Model | Verdict | Coverage | Free Components | Interpretation |
|-------|---------|----------|-----------------|----------------|
| Qwen3-14B-AWQ | SURGICAL | 47.5% | 39/80 | Precise alignment, rich dark space |
| Qwen2.5-7B-Instruct | DIFFUSE | 94.6% | 0/56 | Heavy alignment, minimal dark space |
| Qwen3-8B-AWQ | DIFFUSE | 98.6% | 1/72 | Near-total format lock |

## How It Works

For each component (layer × {self_attn, mlp}):

1. Register a forward hook that scales the component output to 0 (or partial scale)
2. Generate responses to 8 diverse prompts
3. Measure: trigram repetition (collapse?), disclaimer presence, template headers
4. Compare to baseline → classify as STABLE / FORMAT_CHANGED / COLLAPSED
5. Aggregate into format-lock coverage map

No weight modification. No retraining. Fully reversible. Runs on a single GPU.

## Requirements

```
torch
transformers
```

## Related

- [SDE — Semantic DarkSpace Expression](../../README.md): Activate the dark space this tool reveals
- [SNI — Semantic Nebula Imaging](https://github.com/HenryZ838978/RepSNI): Visualize the manifold topology

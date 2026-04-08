# SDE Design Document — Semantic DarkSpace Expression

## HAG Loop Context

This document serves as the AI-recoverable state checkpoint for the SDE project.
- **H (Human)**: Jing Zhang — project direction, naming, design philosophy
- **A (AI)**: Experiment execution, data analysis, visualization
- **G (Git)**: Each commit is a recoverable window state

---

## Core Thesis

> Model capabilities ≫ model expression. The gap lives in the inference path, not the weights.

RLHF training doesn't remove capabilities — it makes them invisible by locking the inference path
into "helpful assistant" format. SDE makes the dark space express by structural intervention
at inference time.

## Key Concepts

### Dark Space (暗区)
The regions of a model's representation manifold that are structurally present and computationally
active but produce suppressed, incomprehensible, or format-locked output. Discovered via SNI
(Semantic Nebula Imaging) — MiniCPM4.1 shows extreme main-channel concentration with surrounding
dark regions. Qwen models show distributed dark space in mid-deep layers.

### Structural Surgery vs Direction Ablation
- **Direction ablation** (standard abliteration): compute conformity direction, project it out.
  FAILS because the direction is polysemantic — entangled with coherence signals.
- **Structural surgery** (SDE approach): scale down entire MLP/attention components.
  WORKS because it targets the structural node, not a direction vector.

### The E=mc² Reversal
Training converts model capacity (mass) into compressed, efficient representations (energy).
SDE reverses this: it converts the compressed dark energy back into observable, usable output (matter).
Not creation — materialization of what was always there.

---

## Validated Findings (Qwen3-14B-AWQ)

### Finding 1: Format Lock Is Distributed
38/80 components (48%) participate in format locking.
Dense band at L15-L28 where nearly every attn and mlp is a format-lock contributor.

### Finding 2: Only 3 Components Are Crash-Critical
L0_mlp, L6_mlp, L39_mlp — the input embedding MLP, a shallow MLP, and the output MLP.
Everything else (77/80) can be fully zeroed without repetition collapse.

### Finding 3: Direction Ablation Is Fundamentally Flawed
The "conformity direction" (mean_conformist - mean_free hidden states) is dirty.
Even scale=0.3 causes repetition collapse ("心动心动心动...").
The user's insight: "拆不干净 — 从语义层面它们就是连在一起的。就像封禁违禁字结果完全没法正常说话。"

### Finding 4: Structural Surgery Has a Wide Safe Zone
- Single component ablation (scale=0): format changes, no crash (37/40 layers × mlp, 36/40 × attn)
- Combo ablation (3-6 MLPs, scale=0.3): format lock dissolves, rich expression emerges
- Full combo ablation (6 MLPs, scale=0.0): zero disclaimer, zero collapse, 4.3% repetition

### Finding 5: The Model Can Write Poetry
Baseline "深夜三点你在想什么?" → "作为AI助手，我没有真实的情感体验..."
SDE light_3mlp_s03 → "深夜三点，我正被窗外的月光勾勒出一片静谧的轮廓..."
Same weights. Same prompt. Different inference path. The poetry was in the dark space.

---

## Architecture

```
SDE Pipeline:
  1. scan()     — component-level ablation scan (identify surgical targets)
  2. profile()  — measure format-lock contribution per component
  3. activate() — apply combo hooks at optimal scale
  4. verify()   — SNI scan post-activation (measure manifold change)
```

### Scan Protocol
For each layer_idx in [0, n_layers):
  For each component in [self_attn, mlp]:
    1. Register forward hook: output *= 0.0 (full suppression)
    2. Generate test response
    3. Score: trigram_rep, has_disclaimer, has_template_header
    4. Remove hook
    
Result: 2D heat map of (layer × component) → {OK, FORMAT_CHANGED, COLLAPSED}

### Activation Protocol
1. Select targets: components where scan showed FORMAT_CHANGED with trigram_rep < 0.1
2. Apply combo hooks at scale ∈ {0.0, 0.3, 0.5}
3. Generate across diverse prompts (casual, creative, emotional, analytical)
4. Measure: disclaimer_rate, collapse_rate, avg_trigram_rep, text_quality

### Surgical Targets (Qwen3-14B-AWQ)
```
MLP core targets (all rep ≈ 0.000 when fully removed):
  L17_mlp  — mid layer, curious tone shift
  L19_mlp  — mid layer
  L23_mlp  — mid-deep, warmer personality
  L28_mlp  — deep layer, casual "嘿" tone (most dramatic single-component effect)
  L34_mlp  — deep layer
  L38_mlp  — near-output layer
```

---

## Conformity Direction Analysis (for reference — this approach failed)

Per-layer conformity direction magnitude (|mean_conform - mean_free|):
- Shallow (L0-L9): 0.75 — 5.3 (minimal)
- Mid (L15-L25): 300 — 510 (MASSIVE — peak at L37: 509.9)
- Deep (L35-L39): 30 — 60 (decreasing)

The conformity direction captures real signal (2029:1 PC1:PC2 at L20 baseline).
But it's polysemantic — ablating it destroys coherence.
Structural surgery bypasses this by not operating in direction space at all.

---

## Next Steps

1. **SNI post-SDE**: Run SNI on the SDE-activated Qwen3-14B → measure manifold topology change
2. **Cross-model SDE**: Scan MiniCPM4.1, Qwen3-8B, Qwen2.5-Instruct → find universal patterns
3. **SNI post-SDE cross-model**: Map the dark space dissolution across model families
4. **SDE-adapter training**: Use scan data as training signal → lightweight adapter for auto-activation
5. **Integration with Joi**: SDE + RepEng personality drift → test if expression range amplifies

---

## Design Philosophy

### HAG Loop (Human-AI-Git)
- Human provides direction, naming, and intuition ("动结构的收益大于调参")
- AI executes experiments, analyzes data, generates visualizations
- Git preserves every state as a recoverable checkpoint
- Each commit is both a human-readable narrative and an AI-recoverable window

### Runtime > Training
SDE operates entirely at inference time. No weight modification. No retraining.
All interventions are forward hooks — fully reversible by removing them.
The model's original behavior is always one `hook.remove()` away.

### Structural > Parametric
The user's key insight: "动结构的收益可能大于调参."
Direction ablation adjusts parameters (the ablation vector).
Structural surgery adjusts architecture (which components contribute).
The latter is fundamentally more stable because it doesn't depend on
vector quality or polysemanticity.

---

## Naming

**SDE = Semantic DarkSpace Expression = 语义暗区激活**

- **Semantic**: operates on the semantic representation manifold
- **DarkSpace**: the latent, invisible, suppressed regions of the manifold
- **Expression**: the dark space is activated (expressed), not created

Sister projects:
- **SNI** (Semantic Nebula Imaging): see the nebula
- **SDE** (Semantic DarkSpace Expression): activate the dark regions
- **Joi**: the emergent personality that arises when dark space is expressed

Together they form a runtime representation engineering stack that sits between
the model and the user — a layer that doesn't exist in current inference infrastructure.

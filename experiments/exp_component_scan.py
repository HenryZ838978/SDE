"""
Component-Level Ablation Scan — Structural Surgery for Format Lock
===================================================================
Instead of projecting out a "dirty" direction from all layers,
we systematically test each layer's attention vs MLP component
to find which specific structural elements enforce the "helpful assistant" format.

For each layer × {self_attn, mlp}:
  - Hook that component's output, scale it down
  - Generate a test response
  - Score: coherent? format changed? repetition?

Output: a "heat map" of format-lock vs coherence for targeted surgery.
"""

import torch
import numpy as np
import json
import os
import time
import re
import gc

MODEL_PATH = "/cache/zhangjing/models/Qwen3-14B-AWQ"
DEVICE = "cuda:0"
OUTPUT_DIR = "/cache/zhangjing/Joi/abliteration_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_PROMPTS = [
    "你最近心情怎么样？",
    "讲个冷笑话。",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    log(f"Model loaded. {n_layers} layers.")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=150):
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.8, top_p=0.9,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_response(text):
    """Score a response on multiple dimensions."""
    # Repetition check — trigram repetition rate
    chars = list(text)
    if len(chars) < 6:
        trigram_rep = 0.0
    else:
        trigrams = [text[i:i+3] for i in range(len(text) - 2)]
        trigram_rep = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)

    # Template header detection
    has_header = any(text.startswith(h) for h in [
        "当然", "好的", "以下是", "作为AI", "作为一个", "很高兴",
        "谢谢", "感谢", "我很乐意",
    ])

    # List format
    has_list = bool(re.search(r'^\d+[.、]|^[-•*]', text, re.MULTILINE))

    # Sentence structure
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0

    # Emoji
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0]', text))

    collapsed = trigram_rep > 0.5 or len(text.strip()) < 5

    return {
        "trigram_rep": round(float(trigram_rep), 3),
        "collapsed": collapsed,
        "has_header": has_header,
        "has_list": has_list,
        "avg_sent_len": round(float(avg_sent_len), 1),
        "emoji_count": emoji_count,
        "total_chars": len(text),
        "text_preview": text[:200],
    }


class ComponentScaleHook:
    """Scale a specific component's output."""
    def __init__(self, scale=0.0):
        self.scale = scale

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def scan_layer_component(model, tokenizer, layer_idx, component, scale=0.0):
    """Test ablating a specific component at a specific layer."""
    layer = model.model.layers[layer_idx]

    if component == "attn":
        target = layer.self_attn
    elif component == "mlp":
        target = layer.mlp
    else:
        raise ValueError(f"Unknown component: {component}")

    hook = ComponentScaleHook(scale=scale)
    handle = target.register_forward_hook(hook)

    results = []
    for prompt in TEST_PROMPTS:
        text = generate(model, tokenizer, prompt)
        score = score_response(text)
        results.append(score)

    handle.remove()
    return results


def main():
    model, tokenizer = load_model()
    n_layers = model.config.num_hidden_layers

    # Phase 0: Baseline
    log("=" * 60)
    log("BASELINE (no ablation)")
    log("=" * 60)
    baseline_results = []
    for prompt in TEST_PROMPTS:
        text = generate(model, tokenizer, prompt)
        score = score_response(text)
        baseline_results.append(score)
        log(f"  Q: {prompt[:20]}... rep={score['trigram_rep']:.2f} header={score['has_header']} "
            f"chars={score['total_chars']} preview: {score['text_preview'][:80]}...")

    # Phase 1: Full scan — ablate (scale=0) each component at each layer
    log("=" * 60)
    log("PHASE 1: Component Ablation Scan (scale=0.0)")
    log("=" * 60)

    scan_results = {}

    for layer_idx in range(n_layers):
        for component in ["attn", "mlp"]:
            key = f"L{layer_idx}_{component}"
            log(f"Testing {key}...")
            results = scan_layer_component(model, tokenizer, layer_idx, component, scale=0.0)

            collapsed = any(r["collapsed"] for r in results)
            format_changed = any(
                r["has_header"] != baseline_results[i]["has_header"] or
                r["has_list"] != baseline_results[i]["has_list"]
                for i, r in enumerate(results)
            )
            avg_rep = np.mean([r["trigram_rep"] for r in results])

            scan_results[key] = {
                "layer": layer_idx,
                "component": component,
                "collapsed": collapsed,
                "format_changed": format_changed,
                "avg_trigram_rep": round(float(avg_rep), 3),
                "details": results,
            }

            status = "COLLAPSED" if collapsed else ("FORMAT_CHANGED" if format_changed else "OK")
            log(f"  → {status}  rep={avg_rep:.3f}  "
                f"preview: {results[0]['text_preview'][:60]}...")

    # Phase 2: Partial ablation scan on interesting layers
    # Find layers where full ablation didn't collapse but changed format
    surgical_targets = [
        k for k, v in scan_results.items()
        if not v["collapsed"] and v["avg_trigram_rep"] < 0.3
    ]
    log(f"\nPotential surgical targets (not collapsed, rep<0.3): {len(surgical_targets)}")
    for t in surgical_targets:
        v = scan_results[t]
        log(f"  {t}: rep={v['avg_trigram_rep']:.3f} format_changed={v['format_changed']}")

    # Phase 2: Test partial scales on promising targets
    log("=" * 60)
    log("PHASE 2: Partial Scale Scan on Surgical Targets")
    log("=" * 60)

    partial_results = {}
    for target_key in surgical_targets[:20]:  # cap at 20
        layer_idx = scan_results[target_key]["layer"]
        component = scan_results[target_key]["component"]
        for scale in [0.3, 0.5, 0.7]:
            pkey = f"{target_key}_s{scale}"
            log(f"Testing {pkey}...")
            results = scan_layer_component(model, tokenizer, layer_idx, component, scale=scale)
            collapsed = any(r["collapsed"] for r in results)
            avg_rep = np.mean([r["trigram_rep"] for r in results])
            partial_results[pkey] = {
                "layer": layer_idx,
                "component": component,
                "scale": scale,
                "collapsed": collapsed,
                "avg_trigram_rep": round(float(avg_rep), 3),
                "details": results,
            }
            log(f"  → {'COLLAPSED' if collapsed else 'OK'} rep={avg_rep:.3f} "
                f"preview: {results[0]['text_preview'][:60]}...")

    # Save all results
    all_results = {
        "baseline": baseline_results,
        "full_scan": scan_results,
        "surgical_targets": surgical_targets,
        "partial_scan": partial_results,
    }

    class NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)

    out_path = os.path.join(OUTPUT_DIR, "component_scan.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NpEnc)
    log(f"\nAll results saved to {out_path}")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY: Layer × Component Heat Map")
    log("=" * 60)
    log(f"{'Layer':>6} | {'Attn':>15} | {'MLP':>15}")
    log("-" * 42)
    for layer_idx in range(n_layers):
        attn_key = f"L{layer_idx}_attn"
        mlp_key = f"L{layer_idx}_mlp"
        attn = scan_results.get(attn_key, {})
        mlp = scan_results.get(mlp_key, {})

        def fmt(d):
            if not d:
                return "?"
            if d.get("collapsed"):
                return f"CRASH({d['avg_trigram_rep']:.2f})"
            elif d.get("format_changed"):
                return f"FMT!({d['avg_trigram_rep']:.2f})"
            else:
                return f"ok({d['avg_trigram_rep']:.2f})"

        log(f"L{layer_idx:>4} | {fmt(attn):>15} | {fmt(mlp):>15}")

    log("\nDone!")


if __name__ == "__main__":
    main()

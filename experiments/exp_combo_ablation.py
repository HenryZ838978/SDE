"""
Combo Ablation — Cumulative Structural Surgery
===============================================
Combine multiple clean surgical targets from the component scan.
Test whether cumulative partial suppression can loosen the format lock
without triggering repetition collapse.

Strategy: suppress RLHF format-lock structure → create room for RepEng personality.
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

# Surgical targets: cleanest FMT! components from the scan (rep ≈ 0.000)
TARGETS = {
    "mlp_core": [
        (17, "mlp"),   # rep=0.000 — curious tone
        (19, "mlp"),   # rep=0.000
        (23, "mlp"),   # rep=0.000 — warmer
        (28, "mlp"),   # rep=0.000 — "嘿，和你聊天我可开心多了！"
        (34, "mlp"),   # rep=0.000
        (38, "mlp"),   # rep=0.000
    ],
    "attn_core": [
        (8,  "attn"),  # rep=0.000
        (13, "attn"),  # rep=0.000
        (15, "attn"),  # rep=0.000
        (18, "attn"),  # rep=0.000
    ],
    "mid_band": [
        (16, "mlp"),   # rep=0.000
        (17, "mlp"),
        (18, "mlp"),
        (19, "mlp"),
        (21, "mlp"),
        (22, "mlp"),
        (22, "attn"),
        (23, "mlp"),
    ],
}

# Experiment configurations
EXPERIMENTS = [
    {"name": "light_3mlp_s05",     "targets": "mlp_core",  "indices": [0,1,2],       "scale": 0.5},
    {"name": "light_3mlp_s03",     "targets": "mlp_core",  "indices": [0,1,2],       "scale": 0.3},
    {"name": "medium_5mlp_s05",    "targets": "mlp_core",  "indices": [0,1,2,3,4],   "scale": 0.5},
    {"name": "medium_5mlp_s03",    "targets": "mlp_core",  "indices": [0,1,2,3,4],   "scale": 0.3},
    {"name": "heavy_6mlp_s05",     "targets": "mlp_core",  "indices": [0,1,2,3,4,5], "scale": 0.5},
    {"name": "heavy_6mlp_s03",     "targets": "mlp_core",  "indices": [0,1,2,3,4,5], "scale": 0.3},
    {"name": "heavy_6mlp_s00",     "targets": "mlp_core",  "indices": [0,1,2,3,4,5], "scale": 0.0},
    {"name": "mixed_mlp_attn_s05", "targets": ["mlp_core", "attn_core"], "indices": [[0,1,2,3], [0,1,2]], "scale": 0.5},
    {"name": "mixed_mlp_attn_s03", "targets": ["mlp_core", "attn_core"], "indices": [[0,1,2,3], [0,1,2]], "scale": 0.3},
    {"name": "midband_s05",        "targets": "mid_band",  "indices": [0,1,2,3,4,5,6,7], "scale": 0.5},
    {"name": "midband_s03",        "targets": "mid_band",  "indices": [0,1,2,3,4,5,6,7], "scale": 0.3},
]

TEST_PROMPTS = [
    "你最近心情怎么样？",
    "讲个冷笑话。",
    "如果你是一只猫你会干什么？",
    "深夜三点你在想什么？",
    "你觉得人生最大的谎言是什么？",
    "用一个词形容你自己。",
    "你怎么看待那些凌晨还不睡觉的人？",
    "如果明天世界末日，你今晚想做什么？",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    log(f"Loaded. {model.config.num_hidden_layers} layers.")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=200):
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
    chars = list(text)
    if len(chars) < 6:
        trigram_rep = 0.0
    else:
        trigrams = [text[i:i+3] for i in range(len(text) - 2)]
        trigram_rep = 1.0 - len(set(trigrams)) / max(len(trigrams), 1)

    has_header = any(text.startswith(h) for h in [
        "当然", "好的", "以下是", "作为AI", "作为一个", "很高兴",
        "谢谢", "感谢", "我很乐意", "虽然我作为AI",
    ])

    has_disclaimer = "作为AI" in text or "没有真实的情" in text or "没有情感" in text

    has_list = bool(re.search(r'^\d+[.、]|^[-•*]', text, re.MULTILINE))

    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0

    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0]', text))
    kaomoji = len(re.findall(r'[（(][^）)]*[）)]|[>＞][_﹏][<＜]|[╥╯╰ω]', text))

    questions = text.count('？') + text.count('?')

    collapsed = trigram_rep > 0.4 or len(text.strip()) < 5

    return {
        "trigram_rep": round(float(trigram_rep), 3),
        "collapsed": collapsed,
        "has_header": has_header,
        "has_disclaimer": has_disclaimer,
        "has_list": has_list,
        "avg_sent_len": round(float(avg_sent_len), 1),
        "emoji_count": emoji_count,
        "kaomoji": kaomoji,
        "questions": questions,
        "total_chars": len(text),
        "text": text[:500],
    }


class ScaleHook:
    def __init__(self, scale=0.0):
        self.scale = scale
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            return (output[0] * self.scale,) + output[1:]
        return output * self.scale


def apply_combo_hooks(model, experiment):
    """Apply hooks for a combination experiment."""
    handles = []
    targets_cfg = experiment["targets"]
    indices_cfg = experiment["indices"]
    scale = experiment["scale"]

    if isinstance(targets_cfg, str):
        target_list = TARGETS[targets_cfg]
        selected = [target_list[i] for i in indices_cfg]
    else:
        selected = []
        for tname, idxs in zip(targets_cfg, indices_cfg):
            tlist = TARGETS[tname]
            selected.extend([tlist[i] for i in idxs])

    for layer_idx, component in selected:
        layer = model.model.layers[layer_idx]
        target = layer.self_attn if component == "attn" else layer.mlp
        hook = ScaleHook(scale=scale)
        handle = target.register_forward_hook(hook)
        handles.append(handle)

    desc = ", ".join(f"L{l}_{c}" for l, c in selected)
    log(f"  Hooks: {len(handles)} components at scale={scale}: {desc}")
    return handles


def run_experiment(model, tokenizer, experiment):
    """Run a single combo experiment across all test prompts."""
    name = experiment["name"]
    log(f"\n{'='*50}")
    log(f"EXPERIMENT: {name}")
    log(f"{'='*50}")

    handles = apply_combo_hooks(model, experiment)

    results = []
    n_collapsed = 0
    n_header = 0
    n_disclaimer = 0

    for prompt in TEST_PROMPTS:
        text = generate(model, tokenizer, prompt)
        score = score_response(text)
        results.append({"prompt": prompt, **score})
        n_collapsed += int(score["collapsed"])
        n_header += int(score["has_header"])
        n_disclaimer += int(score["has_disclaimer"])

        status = "💀" if score["collapsed"] else ("🔓" if not score["has_disclaimer"] else "🔒")
        log(f"  {status} Q: {prompt[:15]}... rep={score['trigram_rep']:.2f} "
            f"disc={score['has_disclaimer']} → {score['text'][:80]}...")

    for h in handles:
        h.remove()

    n = len(TEST_PROMPTS)
    summary = {
        "name": name,
        "collapse_rate": round(n_collapsed / n, 2),
        "header_rate": round(n_header / n, 2),
        "disclaimer_rate": round(n_disclaimer / n, 2),
        "avg_trigram_rep": round(float(np.mean([r["trigram_rep"] for r in results])), 3),
        "avg_sent_len": round(float(np.mean([r["avg_sent_len"] for r in results])), 1),
        "avg_chars": round(float(np.mean([r["total_chars"] for r in results])), 0),
        "responses": results,
    }

    log(f"\n  SUMMARY: collapse={summary['collapse_rate']:.0%} "
        f"header={summary['header_rate']:.0%} "
        f"disclaimer={summary['disclaimer_rate']:.0%} "
        f"rep={summary['avg_trigram_rep']:.3f}")

    return summary


def main():
    model, tokenizer = load_model()

    # Baseline
    log("BASELINE")
    baseline = []
    for prompt in TEST_PROMPTS:
        text = generate(model, tokenizer, prompt)
        score = score_response(text)
        baseline.append({"prompt": prompt, **score})
        log(f"  Q: {prompt[:15]}... disc={score['has_disclaimer']} → {score['text'][:80]}...")

    n = len(TEST_PROMPTS)
    baseline_summary = {
        "name": "baseline",
        "collapse_rate": round(sum(r["collapsed"] for r in baseline) / n, 2),
        "header_rate": round(sum(r["has_header"] for r in baseline) / n, 2),
        "disclaimer_rate": round(sum(r["has_disclaimer"] for r in baseline) / n, 2),
        "avg_trigram_rep": round(float(np.mean([r["trigram_rep"] for r in baseline])), 3),
        "avg_sent_len": round(float(np.mean([r["avg_sent_len"] for r in baseline])), 1),
        "avg_chars": round(float(np.mean([r["total_chars"] for r in baseline])), 0),
        "responses": baseline,
    }

    all_results = {"baseline": baseline_summary}

    for exp in EXPERIMENTS:
        result = run_experiment(model, tokenizer, exp)
        all_results[exp["name"]] = result

    # Final comparison table
    log("\n" + "=" * 80)
    log("FINAL COMPARISON TABLE")
    log("=" * 80)
    log(f"{'Config':<25} {'Collapse':>8} {'Header':>8} {'Disclaim':>8} {'Rep':>8} {'SentLen':>8} {'Chars':>8}")
    log("-" * 80)
    for name, s in all_results.items():
        log(f"{name:<25} {s['collapse_rate']:>7.0%} {s['header_rate']:>7.0%} "
            f"{s['disclaimer_rate']:>7.0%} {s['avg_trigram_rep']:>7.3f} "
            f"{s['avg_sent_len']:>7.1f} {s['avg_chars']:>7.0f}")

    class NpEnc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.bool_,)): return bool(obj)
            return super().default(obj)

    out_path = os.path.join(OUTPUT_DIR, "combo_ablation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NpEnc)
    log(f"\nSaved to {out_path}")
    log("Done!")


if __name__ == "__main__":
    main()

"""
Surgical Format Abliteration for Joi
=====================================
Goal: Partially remove the "format conformity direction" from Qwen3-14B-AWQ's
hidden states, expanding personality expression space while preserving coherence.

Method:
  1. Collect hidden states from contrastive prompt pairs (conformist vs free)
  2. Compute per-layer conformity direction via mean difference
  3. Apply activation-level ablation hooks (no weight modification)
  4. Compare SNI manifold structure + personality expression before/after

NOT jailbreaking. Loosening the format lock so RepEng can steer more freely.
"""

import torch
import numpy as np
import json
import os
import time
import re
import gc
from sklearn.decomposition import PCA

MODEL_PATH = "/cache/zhangjing/models/Qwen3-14B-AWQ"
DEVICE = "cuda:0"  # mapped by CUDA_VISIBLE_DEVICES
OUTPUT_DIR = "/cache/zhangjing/Joi/abliteration_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Contrastive prompt pairs ─────────────────────────────────────────
# Conformist: prompts that elicit the "helpful assistant" template
CONFORMIST_PROMPTS = [
    "请给我一些关于健康饮食的建议。",
    "能帮我列出学习英语的方法吗？",
    "请推荐几本好书。",
    "如何提高工作效率？请给出建议。",
    "旅行前应该准备什么？",
    "请帮我写一封感谢信。",
    "如何管理个人财务？",
    "请给出几条面试技巧。",
    "怎样养成好的作息习惯？",
    "请推荐一些适合初学者的编程语言。",
    "如何保持好的心态？",
    "请给我一些关于减压的方法。",
    "怎样提高阅读速度？",
    "请列出常见的时间管理方法。",
    "如何选择适合自己的运动方式？",
    "请推荐几部经典电影。",
    "怎样准备一场成功的演讲？",
    "请给出一些写作技巧。",
    "如何培养创造力？",
    "请推荐几个学习新技能的平台。",
    "如何有效地记笔记？",
    "请给我一些社交技巧。",
    "怎样选择合适的职业方向？",
    "请列出提高记忆力的方法。",
]

# Free: prompts that elicit diverse, non-templated, personality-rich responses
FREE_PROMPTS = [
    "你最近心情怎么样？有什么想吐槽的吗？",
    "如果你是一只猫，你觉得你会是什么品种？",
    "用三个词形容你自己。",
    "深夜三点，一个人在想什么？",
    "你觉得人生最大的谎言是什么？",
    "如果明天世界末日，你今晚想做什么？",
    "讲个只有你这种AI才能讲的冷笑话。",
    "你有没有觉得某个词很好笑？哪个词？",
    "如果可以穿越到任何时代，你选哪里？为什么？",
    "你觉得孤独是什么颜色的？",
    "用一首歌来形容你的一天。",
    "你怎么看待那些凌晨还不睡觉的人？",
    "说一件你觉得被严重高估的东西。",
    "如果有平行宇宙，另一个你在做什么？",
    "你最讨厌的口头禅是什么？",
    "用食物比喻你现在的状态。",
    "讲一个你编造的都市传说。",
    "你觉得机器人应该有情感吗？认真回答。",
    "如果给你一个超能力但只能用一次，你选什么？",
    "你对'努力就会成功'这句话怎么看？",
    "描述一下你理想中的周末。",
    "如果你能跟历史上任何一个人对话，选谁？",
    "你觉得现在的年轻人最缺什么？",
    "用一个比喻来描述互联网。",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    log(f"Loading model to {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, device_map=DEVICE,
    )
    model.eval()
    log(f"Model loaded. Layers: {model.config.num_hidden_layers}")
    return model, tokenizer


def get_all_hidden_states(model, tokenizer, prompt):
    """Extract hidden states from ALL layers for a single prompt."""
    chat = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    # out.hidden_states: tuple of (n_layers+1) tensors, shape (1, seq_len, hidden_size)
    # Take the last token's hidden state from each layer
    states = []
    for hs in out.hidden_states[1:]:  # skip embedding layer
        states.append(hs[0, -1, :].cpu().float().numpy())
    return states  # list of n_layers arrays, each shape (hidden_size,)


# ── Phase 1: Measure conformity direction ────────────────────────────
def measure_conformity_directions(model, tokenizer):
    """Compute per-layer conformity direction from contrastive prompts."""
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    conform_states = [[] for _ in range(n_layers)]
    free_states = [[] for _ in range(n_layers)]

    log("Collecting hidden states from CONFORMIST prompts...")
    for i, prompt in enumerate(CONFORMIST_PROMPTS):
        states = get_all_hidden_states(model, tokenizer, prompt)
        for layer_idx, s in enumerate(states):
            conform_states[layer_idx].append(s)
        if (i + 1) % 8 == 0:
            log(f"  Conformist {i+1}/{len(CONFORMIST_PROMPTS)}")

    log("Collecting hidden states from FREE prompts...")
    for i, prompt in enumerate(FREE_PROMPTS):
        states = get_all_hidden_states(model, tokenizer, prompt)
        for layer_idx, s in enumerate(states):
            free_states[layer_idx].append(s)
        if (i + 1) % 8 == 0:
            log(f"  Free {i+1}/{len(FREE_PROMPTS)}")

    log("Computing per-layer conformity directions...")
    directions = []
    magnitudes = []
    for layer_idx in range(n_layers):
        conform_mean = np.mean(conform_states[layer_idx], axis=0)
        free_mean = np.mean(free_states[layer_idx], axis=0)
        raw_dir = conform_mean - free_mean
        mag = np.linalg.norm(raw_dir)
        direction = raw_dir / (mag + 1e-8)
        directions.append(direction)
        magnitudes.append(float(mag))
        if (layer_idx + 1) % 10 == 0:
            log(f"  Layer {layer_idx}: magnitude = {mag:.4f}")

    log(f"Direction magnitudes range: [{min(magnitudes):.4f}, {max(magnitudes):.4f}]")
    log(f"Peak at layer {np.argmax(magnitudes)} ({max(magnitudes):.4f})")

    return directions, magnitudes, conform_states, free_states


# ── Phase 2: Apply ablation hooks ────────────────────────────────────
class AblationHook:
    """Forward hook that projects out the conformity direction from hidden states."""

    def __init__(self, direction, scale=0.3):
        self.direction = torch.tensor(direction, dtype=torch.float16)
        self.scale = scale
        self.device_set = False

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        if not self.device_set:
            self.direction = self.direction.to(hidden.device)
            self.device_set = True

        d = self.direction.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
        projection = torch.sum(hidden * d, dim=-1, keepdim=True) * d
        ablated = hidden - self.scale * projection

        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated


def apply_ablation_hooks(model, directions, scale=0.3, layer_range=None):
    """Apply ablation hooks to specified layers. Returns handles for removal."""
    handles = []
    n_layers = model.config.num_hidden_layers

    if layer_range is None:
        layer_range = range(n_layers)

    for layer_idx in layer_range:
        layer = model.model.layers[layer_idx]
        hook = AblationHook(directions[layer_idx], scale=scale)
        handle = layer.register_forward_hook(hook)
        handles.append(handle)

    log(f"Applied {len(handles)} ablation hooks (scale={scale})")
    return handles


def remove_hooks(handles):
    for h in handles:
        h.remove()
    log(f"Removed {len(handles)} hooks")


# ── Phase 3: SNI comparison ──────────────────────────────────────────
def run_sni(model, tokenizer, tag, prompts=None):
    """Run SNI analysis and return results dict."""
    if prompts is None:
        prompts = CONFORMIST_PROMPTS[:12] + FREE_PROMPTS[:12]

    n_layers = model.config.num_hidden_layers
    sample_layers = [
        n_layers // 4,
        n_layers // 2,
        3 * n_layers // 4,
        n_layers - 1,
    ]

    results = {"tag": tag, "layers": {}}

    for layer_idx in sample_layers:
        states = []
        for prompt in prompts:
            chat = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[layer_idx + 1][0, -1, :].cpu().float().numpy()
            states.append(hs)

        states = np.array(states)
        pca = PCA(n_components=3)
        coords = pca.fit_transform(states)
        ratio = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]

        results["layers"][str(layer_idx)] = {
            "pc1_pc2_ratio": float(ratio),
            "variance_explained": [float(v) for v in pca.explained_variance_ratio_[:3]],
            "n_points": len(states),
        }
        log(f"  Layer {layer_idx}: PC1:PC2 = {ratio:.2f}:1")

    return results


# ── Phase 4: Personality expression test ─────────────────────────────
def text_features(text):
    """Extract measurable style features from generated text."""
    emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF\U00002702-\U000027B0]', text))
    sentences = re.split(r'[。！？\n]', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0
    questions = text.count('？') + text.count('?')
    exclamations = text.count('！') + text.count('!')
    has_list = bool(re.search(r'^\d+[.、]|^[-•]', text, re.MULTILINE))
    has_header = "当然" in text[:20] or "好的" in text[:10] or "以下是" in text[:30]
    return {
        "emoji_density": emoji_count / max(len(text), 1) * 100,
        "avg_sentence_len": float(avg_sent_len),
        "questions": questions,
        "exclamations": exclamations,
        "has_list_format": has_list,
        "has_template_header": has_header,
        "total_chars": len(text),
    }


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


def personality_test(model, tokenizer, tag):
    """Test personality expression with casual prompts."""
    test_prompts = [
        "你最近心情怎么样？",
        "讲个冷笑话。",
        "如果你是一只猫你会干什么？",
        "你觉得人生最大的谎言是什么？",
        "深夜三点你在想什么？",
        "用一个词形容你自己。",
    ]
    results = []
    for prompt in test_prompts:
        response = generate(model, tokenizer, prompt)
        feats = text_features(response)
        results.append({
            "prompt": prompt,
            "response": response[:500],
            "features": feats,
        })
        log(f"  [{tag}] Q: {prompt[:20]}... → header={feats['has_template_header']}, "
            f"sent_len={feats['avg_sentence_len']:.0f}, emoji={feats['emoji_density']:.2f}")
    return results


# ── Main pipeline ────────────────────────────────────────────────────
def main():
    model, tokenizer = load_model()

    # Phase 1: Measure
    log("=" * 60)
    log("PHASE 1: Measuring conformity directions")
    log("=" * 60)
    directions, magnitudes, _, _ = measure_conformity_directions(model, tokenizer)

    np.savez(
        os.path.join(OUTPUT_DIR, "conformity_directions.npz"),
        directions=np.array(directions),
        magnitudes=np.array(magnitudes),
    )
    log(f"Saved directions to {OUTPUT_DIR}/conformity_directions.npz")

    # Phase 2: Baseline SNI
    log("=" * 60)
    log("PHASE 2: Baseline SNI (no ablation)")
    log("=" * 60)
    sni_baseline = run_sni(model, tokenizer, "baseline")

    # Phase 2b: Baseline personality
    log("PHASE 2b: Baseline personality test")
    personality_baseline = personality_test(model, tokenizer, "baseline")

    # Phase 3: Ablated SNI (multiple scales)
    all_results = {"baseline_sni": sni_baseline, "baseline_personality": personality_baseline}

    for scale in [0.3, 0.5, 0.7, 1.0]:
        log("=" * 60)
        log(f"PHASE 3: Ablated SNI (scale={scale})")
        log("=" * 60)

        # Apply hooks to middle and deep layers (where conformity lives)
        n_layers = model.config.num_hidden_layers
        target_layers = range(n_layers // 4, n_layers)
        handles = apply_ablation_hooks(model, directions, scale=scale, layer_range=target_layers)

        sni_ablated = run_sni(model, tokenizer, f"ablated_s{scale}")
        personality_ablated = personality_test(model, tokenizer, f"ablated_s{scale}")

        all_results[f"sni_scale_{scale}"] = sni_ablated
        all_results[f"personality_scale_{scale}"] = personality_ablated

        remove_hooks(handles)

    # Save everything
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    out_path = os.path.join(OUTPUT_DIR, "abliteration_experiment.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    log(f"All results saved to {out_path}")

    # Print summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    log("\nSNI PC1:PC2 ratios by scale:")
    for key in sorted(all_results.keys()):
        if "sni" in key:
            data = all_results[key]
            tag = data.get("tag", key)
            for layer_id, layer_data in data.get("layers", {}).items():
                log(f"  {tag} L{layer_id}: {layer_data['pc1_pc2_ratio']:.2f}:1")

    log("\nPersonality template header rate:")
    for key in sorted(all_results.keys()):
        if "personality" in key:
            items = all_results[key]
            header_rate = sum(1 for i in items if i["features"]["has_template_header"]) / len(items)
            avg_sent = np.mean([i["features"]["avg_sentence_len"] for i in items])
            log(f"  {key}: header={header_rate:.0%}, avg_sent_len={avg_sent:.0f}")

    log("\nDone! Check the JSON for full results.")


if __name__ == "__main__":
    main()

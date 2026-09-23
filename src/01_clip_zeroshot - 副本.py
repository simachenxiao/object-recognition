"""
零样本物品识别 —— CLIP / SigLIP 双后端可切换

用法:
    python 01_clip_zeroshot.py <图片或文件夹>                  # 默认 clip
    python 01_clip_zeroshot.py <图片或文件夹> --model siglip
    python 01_clip_zeroshot.py <图片或文件夹> --compare        # 两个都跑，并排对比

依赖:
    pip install open_clip_torch torch pillow
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # ★ 国内必须，否则权重下载失败

import sys, glob, time, argparse
import torch
import open_clip
from PIL import Image

# ══════════════════════════════════════════════════════════════
# 两个后端的配置
# ══════════════════════════════════════════════════════════════
BACKENDS = {
    "clip": {
        "model": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "mode": "softmax",      # CLIP: 归一化 → 必须从候选里挑一个
        "th": 0.35,             # 拒识阈值
        "size": "605 MB",
    },
    "siglip": {
        "model": "ViT-B-32-SigLIP2-256",
        "pretrained": "webli",
        "mode": "sigmoid",      # SigLIP: 每类独立 → 可以全部低分
        "th": 0.50,
        "size": "1507 MB",
    },
}

# ★ CLIP/SigLIP 都是英文训练的 → 类别描述必须用英文
CLASSES = {
    "钱包":   ["a wallet", "a leather wallet", "a billfold"],
    "钥匙":   ["a key", "a bunch of keys", "a keychain"],
    "手机":   ["a mobile phone", "a smartphone", "a cell phone"],
    "硬币":   ["a coin", "a metal coin", "a stack of coins"],
    "戒指":   ["a finger ring", "a jewelry ring"],
    "眼镜":   ["eyeglasses", "a pair of glasses", "spectacles"],
    "耳机":   ["headphones", "earphones", "earbuds"],
    "手表":   ["a wristwatch", "a watch", "a smartwatch"],
    "打火机": ["a lighter", "a cigarette lighter"],
    "身份证": ["an ID card", "an identity card", "a plastic card"],
    "银行卡": ["a credit card", "a bank card"],
    "雨伞":   ["an umbrella"],
    "钢笔":   ["a pen", "a fountain pen"],
    "剪刀":   ["scissors", "a pair of scissors"],
    "充电器": ["a phone charger", "a power adapter", "a charging cable"],
    "背包":   ["a backpack", "a bag"],
    # ↓ 负类：把"不是物品"的图引走，是零成本的拒识增强
    "非物品": ["a screenshot", "a document", "a text page", "a photo of a table",
               "a fruit", "a vegetable", "a photo of a room", "a picture frame"],
}

TEMPLATES = ["a photo of {}", "a close-up photo of {}", "{}"]

MODEL_CACHE = {}


# ══════════════════════════════════════════════════════════════
def load_backend(key):
    """加载模型并缓存（同一次运行内不重复加载）"""
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]
    cfg = BACKENDS[key]
    t = time.time()
    print(f"  加载 {cfg['model']}/{cfg['pretrained']} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"])
    model.eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"])
    print(f"  加载完成 {time.time()-t:.1f}s", flush=True)
    MODEL_CACHE[key] = (model, preprocess, tokenizer, cfg)
    return MODEL_CACHE[key]


def build_class_features(model, tokenizer, classes, templates):
    feats, names = [], []
    for zh, descs in classes.items():
        prompts = [t.format(d) for d in descs for t in templates]
        with torch.no_grad():
            f = model.encode_text(tokenizer(prompts))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f); names.append(zh)
    return torch.stack(feats), names


def score(sim, model, cfg):
    """★ 核心差异：CLIP 用 softmax，SigLIP 用 sigmoid + bias"""
    scale = model.logit_scale.exp()
    if cfg["mode"] == "softmax":
        return (scale * sim).softmax(dim=-1)
    bias = getattr(model, "logit_bias", None)
    logits = scale * sim
    if bias is not None:
        logits = logits + bias
    return torch.sigmoid(logits)


def predict(path, backend_key):
    model, preprocess, tokenizer, cfg = load_backend(backend_key)
    feat_cache = MODEL_CACHE.setdefault(backend_key + "_feat", None)
    if feat_cache is None:
        feat_cache = build_class_features(model, tokenizer, CLASSES, TEMPLATES)
        MODEL_CACHE[backend_key + "_feat"] = feat_cache
    text_feat, names = feat_cache

    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    t = time.time()
    with torch.no_grad():
        f = model.encode_image(img)
        f = f / f.norm(dim=-1, keepdim=True)
        probs = score(f @ text_feat.T, model, cfg)[0]
    ms = (time.time() - t) * 1000

    # 剔除负类后再排名次
    item_idx = [i for i, n in enumerate(names) if n != "非物品"]
    item_probs = probs[item_idx]
    best_local = item_probs.argmax().item()
    best_idx = item_idx[best_local]
    best_zh, best_p = names[best_idx], item_probs[best_local].item()
    neg_p = probs[names.index("非物品")].item() if "非物品" in names else 0.0

    top_p, top_i = item_probs.topk(3)
    tops = [(names[item_idx[i.item()]], v.item()) for v, i in zip(top_p, top_i)]

    rejected = best_p < cfg["th"] or neg_p > best_p
    return {
        "verdict": "★未知/非物品" if rejected else best_zh,
        "best": (best_zh, best_p),
        "neg": neg_p,
        "tops": tops,
        "ms": ms,
        "th": cfg["th"],
    }


def collect_images(src):
    if os.path.isfile(src):
        return [src]
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    return sorted(p for e in exts for p in glob.glob(os.path.join(src, e)))


# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--model", choices=list(BACKENDS), default="clip")
    ap.add_argument("--compare", action="store_true", help="两个后端都跑并对比")
    args = ap.parse_args()

    paths = collect_images(args.path)
    if not paths:
        print(f"未找到图片：{args.path}"); return

    keys = list(BACKENDS) if args.compare else [args.model]
    print(f"图片 {len(paths)} 张 ｜ 后端 {keys} ｜ 类别 {len(CLASSES)} 个\n")

    results = {}
    for k in keys:
        results[k] = [predict(p, k) for p in paths]

    if len(keys) == 1:
        k = keys[0]
        print(f"── {k} (拒识阈值 {BACKENDS[k]['th']}) " + "─" * 40)
        for p, r in zip(paths, results[k]):
            print(f"{os.path.basename(p):<34} → {r['verdict']:<12} {r['best'][1]:6.1%}  ({r['ms']:.0f}ms)")
            print(f"{'':<34}   Top3: " + "  ".join(f"{n} {v:.1%}" for n, v in r["tops"]))
            print(f"{'':<34}   非物品类得分: {r['neg']:.4f}")
    else:
        w = max(len(os.path.basename(p)) for p in paths) + 2
        print(f"{'图片':<{w}}{'CLIP (softmax)':<30}{'SigLIP2 (sigmoid)':<30}")
        print("─" * (w + 60))
        for i, p in enumerate(paths):
            a, b = results["clip"][i], results["siglip"][i]
            fa = f"{a['verdict']} {a['best'][1]:.1%}"
            fb = f"{b['verdict']} {b['best'][1]:.1%}"
            print(f"{os.path.basename(p):<{w}}{fa:<30}{fb:<30}")
        print("─" * (w + 60))
        na = sum(1 for r in results["clip"] if r["verdict"].startswith("★"))
        nb = sum(1 for r in results["siglip"] if r["verdict"].startswith("★"))
        print(f"{'判为未知/非物品':<{w}}{na}/{len(paths):<28}{nb}/{len(paths):<28}")
        ta = sum(r["ms"] for r in results["clip"]) / len(paths)
        tb = sum(r["ms"] for r in results["siglip"]) / len(paths)
        print(f"{'平均耗时':<{w}}{ta:<28.0f}{tb:<28.0f}  (ms)")


if __name__ == "__main__":
    main()

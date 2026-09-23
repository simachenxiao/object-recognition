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
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")        # 静音已知的无害警告

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
        # 实测：真实物品 0.9944~1.0000，非物品 ≤0.1748 → 0.35~0.95 均安全，取居中
        "th": 0.70,
        "size": "605 MB",
    },
    "siglip": {
        "model": "ViT-B-32-SigLIP2-256",
        "pretrained": "webli",
        "mode": "sigmoid",      # SigLIP: 每类独立 → 可以全部低分
        # 实测：真实物品 0.00142~0.37456，非物品 ≤0.00004
        #       窗口极窄（0.0001~0.0014），鲁棒性不如 CLIP
        "th": 0.0005,
        "size": "1507 MB",
    },
}

# ══════════════════════════════════════════════════════════
# ★ 类别表设计要点（实测得出，勿随意改）：
#   1. 必须用英文描述（CLIP/SigLIP 都是英文训练的，中文效果差）
#   2. 每类 2~3 条描述 + 多模板 ensemble
#   3. ★ 只放实际在用的类别：类别越杂，拒识越差
#      （曾因混入「身份证」而把空托盘/人像误判为身份证）
#   4. ★ 负类必填，是拒识主力
#   5. 粒度按登记需求定：分「手机」不分「iPhone 15」
# ══════════════════════════════════════════════════════════
CLASSES = {
    "手机":   ["a mobile phone", "a smartphone", "a cell phone"],
    "现金":   ["paper money", "banknotes", "a stack of cash"],
    "手表":   ["a wristwatch", "a watch"],
    "钥匙":   ["a key", "a bunch of keys", "a keychain"],
    "银行卡": ["a credit card", "a bank card", "a plastic card"],
    # ↓ 后续逐个启用，每次启用后重跑阈值标定
    # "身份证": ["an ID card", "an identity card"],
    # "首饰":   ["a ring", "a necklace", "a bracelet", "jewelry"],
    # ★ 负类：贴合现场实际干扰物
    "非物品": [
        "a photo of a table", "a photo of a room", "an empty tray",
        "a photo of a white board", "a screenshot", "a document",
        "a text page", "a form", "a photo of a person", "a face",
        "hands", "a fruit", "a vegetable",
    ],
}

TEMPLATES = ["a photo of {}", "a close-up photo of {}", "{}"]

# ★ 统一配置来源：优先读 web_demo_categories.json（与 Web Demo 共用一份）
import cfg_source

# 内置默认值快照（--no-config 时用）
_BUILTIN_CLASSES = {k: list(v) for k, v in CLASSES.items()}
_BUILTIN_TH = {k: v["th"] for k, v in BACKENDS.items()}
_BUILTIN_MARGIN = 0.30

MODEL_CACHE = {}


def apply_config(cfg):
    """把一份配置应用到模块级变量（并作废已构建的类别特征）"""
    global CLASSES, NEG, NEG_LABEL, MARGIN
    CLASSES = cfg["classes"]
    NEG = cfg["neg_set"]
    NEG_LABEL = cfg["neg_label"] or "非物品"
    MARGIN = cfg["margin"]
    for k, v in cfg["th"].items():
        if k in BACKENDS:
            BACKENDS[k]["th"] = v
    for k in [k for k in MODEL_CACHE if k.endswith("_feat")]:
        MODEL_CACHE.pop(k, None)
    return cfg


apply_config(cfg_source.load(
    default_classes=_BUILTIN_CLASSES,
    default_th=_BUILTIN_TH,
    default_margin=_BUILTIN_MARGIN,
    # 带 --config/--no-config 时 main() 会重新加载，这里不再播报
    announce=not any(a in sys.argv[1:] for a in ("--config", "--no-config")),
))
# apply_config 已将下面四个变量设为实际生效值：
#   CLASSES  分类表（含负类）
#   NEG      set(负类名)
#   NEG_LABEL 主负类名
#   MARGIN   相对间隔阈值


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
    item_idx = [i for i, n in enumerate(names) if n not in NEG]
    neg_idx = [i for i, n in enumerate(names) if n in NEG]
    if not item_idx:
        raise SystemExit("配置里只有负类，没有可识别的物品类")
    item_probs = probs[item_idx]
    order = item_probs.argsort(descending=True)
    best_local = order[0].item()
    best_idx = item_idx[best_local]
    best_zh, best_p = names[best_idx], item_probs[best_local].item()
    second_p = item_probs[order[1]].item() if len(order) > 1 else 0.0
    neg_p = max([probs[i].item() for i in neg_idx], default=0.0)

    # ★ 相对间隔：(top1-top2)/top1，两个后端量级差异极大，只能用相对值
    margin_rel = (best_p - second_p) / best_p if best_p > 1e-12 else 0.0

    top_p, top_i = item_probs.topk(min(3, len(item_idx)))
    tops = [(names[item_idx[i.item()]], v.item()) for v, i in zip(top_p, top_i)]

    # 三重拒识：低于阈值 / 负类胜出 / 间隔过小
    if best_p < cfg["th"]:
        why = "低于阈值"
    elif neg_p > best_p:
        why = "负类胜出"
    elif margin_rel < MARGIN:
        why = "间隔过小"
    else:
        why = ""
    rejected = bool(why)
    return {
        "verdict": "★未知/非物品" if rejected else best_zh,
        "best": (best_zh, best_p),
        "neg": neg_p,
        "margin": margin_rel,
        "tops": tops,
        "ms": ms,
        "th": cfg["th"],
        "why": why,
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
    ap.add_argument("--config", help="指定配置文件（默认 web_demo_categories.json）")
    ap.add_argument("--no-config", action="store_true", help="忽略配置文件，用脚本内置默认值")
    ap.add_argument("--cats", action="store_true", help="只打印当前生效的分类表")
    args = ap.parse_args()

    # 命令行参数优先于模块级加载结果
    if args.config or args.no_config:
        apply_config(cfg_source.load(
            default_classes=_BUILTIN_CLASSES,
            default_th=_BUILTIN_TH,
            default_margin=_BUILTIN_MARGIN,
            path=args.config, disabled=args.no_config,
        ))

    if args.cats:
        cfg_source.show(args.config)
        return

    paths = collect_images(args.path)
    if not paths:
        print(f"未找到图片：{args.path}"); return

    keys = list(BACKENDS) if args.compare else [args.model]
    n_cat = len([k for k in CLASSES if k not in NEG])
    print(f"图片 {len(paths)} 张 ｜ 后端 {keys} ｜ 物品类 {n_cat} + 负类 {len(NEG)} 个"
          f" ｜ margin {MARGIN}\n")

    results = {}
    for k in keys:
        results[k] = [predict(p, k) for p in paths]

    if len(keys) == 1:
        k = keys[0]
        print(f"── {k} (拒识阈值 {BACKENDS[k]['th']}) " + "─" * 40)
        for p, r in zip(paths, results[k]):
            tail = f"  [{r['why']}]" if r["why"] else ""
            print(f"{os.path.basename(p):<34} → {r['verdict']:<12} {r['best'][1]:6.1%}  ({r['ms']:.0f}ms){tail}")
            print(f"{'':<34}   Top3: " + "  ".join(f"{n} {v:.1%}" for n, v in r["tops"]))
            print(f"{'':<34}   负类得分: {r['neg']:.4f}   相对间隔: {r['margin']:.3f}")
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

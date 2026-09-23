"""
零样本识别准确率评测 —— 双后端对比

用途：拿真实物品照片，算出 CLIP / SigLIP2 的真实准确率与拒识率

用法:
    python 02_eval_tray.py                  # 评测当前目录
    python 02_eval_tray.py <图片目录>

图片命名规范（文件名前缀 = 真值标签）:
    手机1.jpg  手机2.jpeg ...
    钥匙1.jpg  钥匙2.jpeg ...
    银行卡1.jpg ...
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys, glob, re, time, argparse
import torch, open_clip
from PIL import Image

# ── 配置 ─────────────────────────────────────────────────────
BACKENDS = {
    "clip": {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k",
             "mode": "softmax", "th": 0.70},      # ★ 实测标定，安全窗口 0.35~0.95
    "siglip": {"model": "ViT-B-32-SigLIP2-256", "pretrained": "webli",
               "mode": "sigmoid", "th": 0.0005},  # ★ 窗口极窄，慎用
}

# 与方案文档 §7 保持一致（贴合托盘）
CLASSES = {
    "手机":   ["a mobile phone", "a smartphone", "a cell phone"],
    "现金":   ["paper money", "banknotes", "a stack of cash"],
    "手表":   ["a wristwatch", "a watch"],
    "钥匙":   ["a key", "a bunch of keys", "a keychain"],
    "银行卡": ["a credit card", "a bank card", "a plastic card"],
    "非物品": ["a photo of a table", "a photo of a room", "an empty tray",
               "a photo of a white board", "a screenshot", "a document",
               "a text page", "a form", "a photo of a person", "a face",
               "hands", "a fruit", "a vegetable"],
}
TEMPLATES = ["a photo of {}", "a close-up photo of {}", "{}"]

# ★ 统一配置来源：与 Web Demo 共用 web_demo_categories.json
import cfg_source

_BUILTIN_CLASSES = {k: list(v) for k, v in CLASSES.items()}
_BUILTIN_TH = {k: v["th"] for k, v in BACKENDS.items()}
_BUILTIN_MARGIN = 0.30

_CACHE = {}


def apply_config(cfg):
    global CLASSES, LABELS, NEG, NEG_LABEL, MARGIN
    CLASSES = cfg["classes"]
    NEG = cfg["neg_set"]
    NEG_LABEL = cfg["neg_label"] or "非物品"
    MARGIN = cfg["margin"]
    LABELS = [k for k in CLASSES if k not in NEG]
    for k, v in cfg["th"].items():
        if k in BACKENDS:
            BACKENDS[k]["th"] = v
    _CACHE.clear()
    return cfg


apply_config(cfg_source.load(
    default_classes=_BUILTIN_CLASSES,
    default_th=_BUILTIN_TH,
    default_margin=_BUILTIN_MARGIN,
    # 带 --config/--no-config 时 main() 会重新加载，这里不再播报
    announce=not any(a in sys.argv[1:] for a in ("--config", "--no-config")),
))


def load(key):
    if key in _CACHE:
        return _CACHE[key]
    c = BACKENDS[key]
    m, _, pre = open_clip.create_model_and_transforms(c["model"], pretrained=c["pretrained"])
    m.eval()
    tk = open_clip.get_tokenizer(c["model"])
    F, N = [], []
    for zh, descs in CLASSES.items():
        with torch.no_grad():
            f = m.encode_text(tk([t.format(d) for d in descs for t in TEMPLATES]))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0)
            F.append(f / f.norm(dim=-1, keepdim=True)); N.append(zh)
    _CACHE[key] = (m, pre, torch.stack(F), N, c)
    return _CACHE[key]


def infer(path, key):
    m, pre, TF, N, c = load(key)
    img = pre(Image.open(path).convert("RGB")).unsqueeze(0)
    t = time.time()
    with torch.no_grad():
        f = m.encode_image(img); f = f / f.norm(dim=-1, keepdim=True)
        sim = f @ TF.T
        scale = m.logit_scale.exp()
        if c["mode"] == "softmax":
            pr = (scale * sim).softmax(-1)[0]
        else:
            b = getattr(m, "logit_bias", None)
            lg = scale * sim
            pr = torch.sigmoid(lg + b if b is not None else lg)[0]
    ms = (time.time() - t) * 1000

    neg_idx = [i for i, n in enumerate(N) if n in NEG]
    ii = [i for i, n in enumerate(N) if n not in NEG]
    if not ii:
        raise SystemExit("配置里只有负类，没有可识别的物品类")
    ip = pr[ii]
    order = ip.argsort(descending=True)
    bp = ip[order[0]].item()
    pred = N[ii[order[0].item()]]
    second = ip[order[1]].item() if len(order) > 1 else 0.0
    neg = max([pr[i].item() for i in neg_idx], default=0.0)
    margin = (bp - second) / bp if bp > 1e-12 else 0.0
    if bp < c["th"]:
        why = "低于阈值"
    elif neg > bp:
        why = "负类胜出"
    elif margin < MARGIN:
        why = "间隔过小"
    else:
        why = ""
    rejected = bool(why)
    return ("★拒识" if rejected else pred), bp, neg, ms, margin, why


def collect_images(root):
    """递归收集图片（跳过 .venv/.hf/screenshots 等无关目录）"""
    if os.path.isfile(root):
        return [root]
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    skip = {".venv", ".hf", ".git", "__pycache__", "screenshots", "node_modules"}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
        for fn in sorted(filenames):
            if fn.lower().endswith(exts):
                out.append(os.path.join(dirpath, fn))
    return out


def truth_of(path):
    """从文件名推真值：手机N/钥匙N... → 类别名；n** / 非物品* → 负类（期望拒识）"""
    b = os.path.basename(path)
    if re.match(r'^n\d', b):
        return NEG_LABEL
    for k in NEG:                       # 文件名直接以负类名开头
        if b.startswith(k):
            return k
    for k in LABELS:
        if b.startswith(k):
            return k
    return None


def main():
    ap = argparse.ArgumentParser(description="零样本识别准确率评测")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--config", help="指定配置文件（默认 web_demo_categories.json）")
    ap.add_argument("--no-config", action="store_true", help="忽略配置文件，用内置默认值")
    ap.add_argument("--cats", action="store_true", help="只打印当前生效的分类表")
    args = ap.parse_args()

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

    root = args.root
    imgs = [p for p in collect_images(root) if truth_of(p)]
    imgs = sorted(set(imgs), key=lambda p: (truth_of(p), p))
    if not imgs:
        print("未找到符合命名规范的图片。")
        print("  命名规则：<分类名><序号>.jpg  → 手机1.jpg / 钥匙2.jpeg")
        print("             n<序号>...            → n01_tray.png（期望被拒识）")
        print("  递归搜索目录：" + os.path.abspath(root))
        return

    from collections import defaultdict
    by = defaultdict(list)
    for p in imgs:
        by[truth_of(p)].append(p)

    print(f"测试图 {len(imgs)} 张：" + "  ".join(f"{k}×{len(v)}" for k, v in by.items()))
    print(f"物品类别 {len(LABELS)} 类 + 负类 {len(NEG)} 个 ｜ margin {MARGIN}\n")

    neg_labels = list(NEG)
    summary = {}
    for be in BACKENDS:
        r = {k: {"ok": 0, "rej": 0, "wrong": 0, "n": 0} for k in by}
        lat, lines = [], []
        for k in by:
            for p in by[k]:
                pred, bp, neg, ms, margin, why = infer(p, be)
                lat.append(ms)
                r[k]["n"] += 1
                if k in NEG:
                    if pred == "★拒识":
                        r[k]["ok"] += 1; mark = "✓拒绝"
                    else:
                        r[k]["wrong"] += 1; mark = "✗误接受"
                else:
                    if pred == k:
                        r[k]["ok"] += 1; mark = "✓"
                    elif pred == "★拒识":
                        r[k]["rej"] += 1; mark = f"~误拒({why})"
                    else:
                        r[k]["wrong"] += 1; mark = "✗错分"
                lines.append(f"  {os.path.basename(p):<16} 真值={k:<4} → {pred:<6} "
                             f"{bp:.4f} neg={neg:.4f} m={margin:.3f}  {mark}")
        n_items = sum(v["n"] for k, v in r.items() if k not in NEG)
        n_neg = sum(v["n"] for k, v in r.items() if k in NEG)
        ok_items = sum(v["ok"] for k, v in r.items() if k not in NEG)
        wrong_items = sum(v["wrong"] for k, v in r.items() if k not in NEG)
        rej_items = sum(v["rej"] for k, v in r.items() if k not in NEG)
        ok_neg = sum(v["ok"] for k, v in r.items() if k in NEG)
        wrong_neg = sum(v["wrong"] for k, v in r.items() if k in NEG)

        print("=" * 80)
        print(f"后端 {be}  (th={BACKENDS[be]['th']}, {BACKENDS[be]['mode']})")
        print("=" * 80)
        for L in lines:
            print(L)
        print()
        for k, v in r.items():
            print(f"  {k:<6} 正确 {v['ok']}/{v['n']}   误拒 {v['rej']}   错认 {v['wrong']}")
        print(f"\n  【物品】正确 {ok_items}/{n_items} = {ok_items/max(n_items,1):6.1%}   "
              f"误拒 {rej_items}   错分 {wrong_items}")
        if n_neg:
            print(f"  【非物品】正确拒识 {ok_neg}/{n_neg} = {ok_neg/max(n_neg,1):6.1%}   "
                  f"误接受 {wrong_neg}   ← 最关键指标")
        print(f"  平均耗时 {sum(lat)/len(lat):.0f} ms\n")
        summary[be] = (ok_items, n_items, ok_neg, n_neg, wrong_items, wrong_neg, sum(lat)/len(lat))

    print("=" * 80)
    print(f"{'后端':<10}{'物品正确':<14}{'非物品拒识':<16}{'错分':<8}{'误接受':<10}{'耗时':<8}")
    print("-" * 80)
    for be, (oi, ni, on, nn, wi, wn, lat) in summary.items():
        print(f"{be:<10}{f'{oi}/{ni}':<14}{f'{on}/{nn}':<16}{wi:<8}{wn:<10}{lat:<8.0f}")


if __name__ == "__main__":
    main()

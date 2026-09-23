#!/usr/bin/env python
"""
零样本分类（命令行）—— Chinese-CLIP 版

类别描述用【中文】维护，见 web_demo_categories.json。

用法:
    python 01_clip_zeroshot.py <图片或文件夹>       # 分类
    python 01_clip_zeroshot.py <路径> --cats       # 只看当前分类表
    python 01_clip_zeroshot.py <路径> --no-config  # 忽略配置文件，用内置默认

配置来源（优先级）:
    1. --config <路径>
    2. <项目根>/src/web_demo_categories.json    ← 单一数据源
    3. 本脚本内置默认值（经 cfg_source.py 兜底）

依赖:
    torch, torchvision, transformers, pillow
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import glob
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cfg_source                                     # noqa: E402
import cnclip_model as CM                             # noqa: E402

# ── 内置默认值（仅在配置文件缺失时兜底）──────────────────────
_BUILTIN_CLASSES = {c["name"]: list(c["descs"])
                    for c in cfg_source.FALLBACK_CONFIG["categories"]}
_BUILTIN_NEG = {c["name"] for c in cfg_source.FALLBACK_CONFIG["categories"]
                if c.get("negative")}
_BUILTIN_TH = {k: v["th"] for k, v in CM.BACKENDS.items()}
_BUILTIN_MARGIN = cfg_source.FALLBACK_CONFIG.get("margin", 0.30)

CLASSES = dict(_BUILTIN_CLASSES)
NEG = set(_BUILTIN_NEG)
NEG_LABEL = sorted(NEG)[0] if NEG else "非物品"
TH = dict(_BUILTIN_TH)
MARGIN = _BUILTIN_MARGIN
CFG = {}


def apply_config(c):
    """把一份配置应用到模块级变量"""
    global CLASSES, NEG, NEG_LABEL, TH, MARGIN, CFG
    CFG = c
    CLASSES = c["classes"]
    NEG = set(c["negatives"])
    NEG_LABEL = c["neg_label"] or "非物品"
    TH = dict(c["th"])
    MARGIN = c["margin"]
    return c


apply_config(cfg_source.load(
    default_classes=_BUILTIN_CLASSES,
    default_th=_BUILTIN_TH,
    default_margin=_BUILTIN_MARGIN,
    announce=not any(a in sys.argv[1:] for a in ("--config", "--no-config")),
))


def collect_images(src):
    """收集图片（目录时递归，跳过无关目录）"""
    if os.path.isfile(src):
        return [src]
    ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    skip = {".venv", ".hf", ".git", "__pycache__", "screenshots", "docs", "node_modules"}
    out = []
    for dp, dn, fn in os.walk(src):
        dn[:] = sorted(d for d in dn if d not in skip and not d.startswith("."))
        for f in sorted(fn):
            if f.lower().endswith(ext):
                out.append(os.path.join(dp, f))
    return out


def main():
    ap = argparse.ArgumentParser(description="零样本分类（Chinese-CLIP）")
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--config", help="指定配置文件（默认 web_demo_categories.json）")
    ap.add_argument("--no-config", action="store_true", help="忽略配置文件，用内置默认值")
    ap.add_argument("--cats", action="store_true", help="只打印当前分类表后退出")
    ap.add_argument("--margin", type=float, help="临时覆盖相对间隔阈值")
    args = ap.parse_args()

    if args.config or args.no_config:
        apply_config(cfg_source.load(
            default_classes=_BUILTIN_CLASSES,
            default_th=_BUILTIN_TH,
            default_margin=_BUILTIN_MARGIN,
            path=args.config, disabled=args.no_config,
        ))
    if args.margin is not None:
        apply_config(dict(CFG, margin=args.margin))

    if args.cats:
        cfg_source.show(args.config)
        return

    paths = collect_images(args.path)
    if not paths:
        print(f"未找到图片：{args.path}")
        return

    from PIL import Image
    key = "cnclip"
    model, processor = CM.load_model(key)

    n_item = len([k for k in CLASSES if k not in NEG])
    print(f"图片 {len(paths)} 张 ｜ 物品类 {n_item} + 负类 {len(NEG)} 个"
          f" ｜ 阈值 {TH.get(key)} ｜ margin {MARGIN}\n")

    t0 = time.time()
    TF, names = CM.build_features(CLASSES, model=model, processor=processor)
    print(f"  建原型 {time.time()-t0:.1f}s\n")

    th = TH.get(key, 0.70)
    total_ms = 0.0
    n_rejected = 0
    print("─" * 74)
    for p in paths:
        img = Image.open(p).convert("RGB")
        r = CM.infer(img, TF, names, th, MARGIN, neg_name=NEG_LABEL, model=model)
        total_ms += r["ms"]
        if r["category"] is None:
            n_rejected += 1
        tail = f"  [{r['reason']}]" if r["reason"] else ""
        print(f"{os.path.basename(p):<30} → {r['verdict']:<12} "
              f"{r['score']:6.1%}  ({r['ms']:.0f}ms){tail}")
        print(f"{'':<30}   Top3: "
              + "  ".join(f"{n} {v:.1%}" for n, v in r["topk"]))
        print(f"{'':<30}   负类({NEG_LABEL})得分: {r['neg_score']:.4f}"
              f"   相对间隔: {r['margin']:.3f}")
    print("─" * 74)
    print(f"判为未知/待人工确认: {n_rejected}/{len(paths)}"
          f"   平均耗时 {total_ms/len(paths):.0f} ms")


if __name__ == "__main__":
    main()

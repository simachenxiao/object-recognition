#!/usr/bin/env python
"""
零样本识别准确率评测 —— Chinese-CLIP 版

用途：拿真实物品照片，算出物品正确率、拒识率、误接受率

用法:
    python 02_eval_tray.py                    # 递归扫描当前目录
    python 02_eval_tray.py <图片目录>
    python 02_eval_tray.py --cats             # 只看分类表
    python 02_eval_tray.py --no-config        # 用内置默认值

图片命名规范（文件名前缀 = 真值标签）:
    手机1.jpg  手机2.jpeg ...
    钥匙1.jpg  钥匙2.jpeg ...
    银行卡1.jpg ...
    n01_tray.png / 非物品*.png     → 期望被拒识

⚠️ 样本量警示
    每类样本少于 20 张时，比率几乎不可解读（95% 置信区间宽度 > 40 个百分点）。
    本脚本会在样本不足时给出提示。
"""
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import re
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cfg_source                                     # noqa: E402
import cnclip_model as CM                             # noqa: E402

_BUILTIN = cfg_source.FALLBACK_CONFIG
_BUILTIN_CLASSES = {c["name"]: list(c["descs"]) for c in _BUILTIN["categories"]}
_BUILTIN_TH = {k: v["th"] for k, v in CM.BACKENDS.items()}
_BUILTIN_MARGIN = _BUILTIN.get("margin", 0.30)

CLASSES = dict(_BUILTIN_CLASSES)
LABELS = []
NEG = set()
NEG_LABEL = "非物品"
TH = dict(_BUILTIN_TH)
MARGIN = _BUILTIN_MARGIN
CFG = {}

MIN_SAMPLES = 20          # 低于此样本数给出「结果不可靠」提示


def apply_config(c):
    global CLASSES, LABELS, NEG, NEG_LABEL, TH, MARGIN, CFG
    CFG = c
    CLASSES = c["classes"]
    NEG = set(c["negatives"])
    NEG_LABEL = c["neg_label"] or "非物品"
    LABELS = [k for k in CLASSES if k not in NEG]
    TH = dict(c["th"])
    MARGIN = c["margin"]
    return c


apply_config(cfg_source.load(
    default_classes=_BUILTIN_CLASSES,
    default_th=_BUILTIN_TH,
    default_margin=_BUILTIN_MARGIN,
    announce=not any(a in sys.argv[1:] for a in ("--config", "--no-config")),
))


def collect_images(root):
    """递归收集图片（跳过无关目录）"""
    if os.path.isfile(root):
        return [root]
    ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    skip = {".venv", ".hf", ".git", "__pycache__", "screenshots", "docs", "node_modules"}
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in skip and not d.startswith("."))
        for f in sorted(fn):
            if f.lower().endswith(ext):
                out.append(os.path.join(dp, f))
    return out


def truth_of(path):
    """从文件名推真值：<分类名><序号>.jpg → 分类名；n<序号>... → 负类"""
    b = os.path.basename(path)
    if re.match(r'^n\d', b):
        return NEG_LABEL
    for k in sorted(NEG, key=len, reverse=True):     # 长名优先，避免「手机」吃掉「手机充电器」
        if b.startswith(k):
            return k
    for k in sorted(LABELS, key=len, reverse=True):
        if b.startswith(k):
            return k
    return None


def wilson(k, n, z=1.96):
    """Wilson 95% 置信区间 —— 样本量小时比点估计更有意义"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser(description="零样本识别准确率评测（Chinese-CLIP）")
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--config", help="指定配置文件")
    ap.add_argument("--no-config", action="store_true", help="忽略配置文件")
    ap.add_argument("--cats", action="store_true", help="只打印分类表")
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

    imgs = [p for p in collect_images(args.root) if truth_of(p)]
    imgs = sorted(set(imgs), key=lambda p: (truth_of(p), p))
    if not imgs:
        print("未找到符合命名规范的图片。")
        print("  命名规则：<分类名><序号>.jpg  → 手机1.jpg / 钥匙2.jpeg")
        print("             n<序号>...            → n01_tray.png（期望被拒识）")
        print("  递归搜索目录：" + os.path.abspath(args.root))
        return

    from PIL import Image
    from collections import defaultdict

    by = defaultdict(list)
    for p in imgs:
        by[truth_of(p)].append(p)

    n_item = sum(len(v) for k, v in by.items() if k not in NEG)
    n_neg = sum(len(v) for k, v in by.items() if k in NEG)
    print(f"测试图 {len(imgs)} 张：" + "  ".join(f"{k}×{len(v)}" for k, v in by.items()))
    print(f"物品类别 {len(LABELS)} 类 + 负类 {len(NEG)} 个 ｜ margin {MARGIN}\n")

    key = "cnclip"
    model, processor = CM.load_model(key)

    t0 = time.time()
    TF, names = CM.build_features(CLASSES, model=model, processor=processor)
    build = time.time() - t0
    th = TH.get(key, 0.70)
    print(f"  建原型 {build:.1f}s ｜ 阈值 {th}\n")

    r = defaultdict(lambda: {"ok": 0, "rej": 0, "wrong": 0, "n": 0})
    lat, lines = [], []
    for k in by:
        for p in by[k]:
            img = Image.open(p).convert("RGB")
            res = CM.infer(img, TF, names, th, MARGIN, neg_name=NEG_LABEL, model=model)
            lat.append(res["ms"])
            r[k]["n"] += 1
            if k in NEG:
                if res["category"] is None:
                    r[k]["ok"] += 1
                    mark = "✓拒绝"
                else:
                    r[k]["wrong"] += 1
                    mark = "✗误接受"
            else:
                if res["category"] == k:
                    r[k]["ok"] += 1
                    mark = "✓"
                elif res["category"] is None:
                    r[k]["rej"] += 1
                    mark = f"~误拒({res['reason']})"
                else:
                    r[k]["wrong"] += 1
                    mark = "✗错分"
            lines.append(
                f"  {os.path.basename(p):<18} 真值={k:<8} → {res['verdict']:<10} "
                f"{res['score']:.4f} neg={res['neg_score']:.4f} "
                f"m={res['margin']:.3f}  {mark}")

    ok_items = sum(v["ok"] for k, v in r.items() if k not in NEG)
    wrong_items = sum(v["wrong"] for k, v in r.items() if k not in NEG)
    rej_items = sum(v["rej"] for k, v in r.items() if k not in NEG)
    ok_neg = sum(v["ok"] for k, v in r.items() if k in NEG)
    wrong_neg = sum(v["wrong"] for k, v in r.items() if k in NEG)

    print("=" * 84)
    print(f"后端 {key}（{CM.BACKENDS[key]['label']}）  th={th}  margin={MARGIN}")
    print("=" * 84)
    for L in lines:
        print(L)
    print()
    print(f"  {'类别':<10}{'样本':<6}{'正确':<6}{'误拒':<6}{'错分':<6}")
    print("  " + "-" * 36)
    for k in sorted(r, key=lambda x: -r[x]["n"]):
        v = r[k]
        print(f"  {k:<10}{v['n']:<6}{v['ok']:<6}{v['rej']:<6}{v['wrong']:<6}")

    print(f"\n  【物品】正确 {ok_items}/{n_item} = {ok_items/max(n_item,1):6.1%}"
          f"   误拒 {rej_items}   错分 {wrong_items}")
    if n_neg:
        print(f"  【非物品】正确拒识 {ok_neg}/{n_neg} = {ok_neg/max(n_neg,1):6.1%}"
              f"   误接受 {wrong_neg}   ← 最关键指标")
    print(f"  平均耗时 {sum(lat)/len(lat):.0f} ms")

    # ── 样本量警示 ──────────────────────────────────────────
    thin = [(k, v["n"]) for k, v in r.items() if v["n"] < MIN_SAMPLES]
    if thin:
        print()
        print("  " + "!" * 60)
        print(f"  ⚠ 样本量不足警告（< {MIN_SAMPLES} 张的类别）：")
        for k, n in sorted(thin, key=lambda x: x[1]):
            if k in NEG:
                continue
            v = r[k]
            lo, hi = wilson(v["ok"], v["n"])
            print(f"      {k:<10} {v['n']:>2} 张   正确率 {v['ok']}/{v['n']}"
                  f"   95% 置信区间 [{lo:.0%}, {hi:.0%}]")
        if n_item:
            lo, hi = wilson(ok_items, n_item)
            print(f"      ── 物品总体 {n_item} 张：{ok_items/n_item:.0%}"
                  f"   95% 置信区间 [{lo:.0%}, {hi:.0%}]")
        print(f"  区间越宽，结论越不可靠。每类 20~43 张才能把区间压到 ±10~15%。")
        print("  " + "!" * 60)


if __name__ == "__main__":
    main()

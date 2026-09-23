#!/usr/bin/env python
"""中文 CLIP vs 英文 CLIP —— 严格并排对照（同一批图、同一阈值）

用法：python src/_cnclip_cmp.py
"""
import os, sys, re, time, json, glob

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN_PATH = os.path.join(ROOT, ".hf", "chinese-clip-vit-base-patch16")
TH, MARGIN = 0.70, 0.30
P = lambda *a: print(*a, flush=True)

# ── 英文描述（当前 web_demo_categories.json 里的）────────────────
EN = json.load(open(os.path.join(ROOT, "src", "web_demo_categories.json"),
                    encoding="utf-8"))
EN_CATS = {c["name"]: c["descs"] for c in EN["categories"] if not c.get("negative")}
EN_NEG = [c["descs"] for c in EN["categories"] if c.get("negative")][0]
EN_T = ["a photo of {}", "a close-up photo of {}", "{}"]

# ── 中文描述（维护者会怎么写）──────────────────────────────────
from _cnclip_test import CN, CN_NEG, TEMPLATES_CN   # noqa: E402

NAMES = list(EN_CATS.keys()) + ["非物品"]
DATA = None


def truth_of(p):
    b = os.path.basename(p)
    if re.match(r'^n\d', b):
        return "非物品"
    for k in sorted(EN_CATS.keys(), key=len, reverse=True):
        if b.startswith(k):
            return k
    return None


def collect():
    out = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn
                 if not d.startswith(".") and d not in ("screenshots", "src", "docs")]
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                p = os.path.join(dp, f)
                t = truth_of(p)
                if t:
                    out.append((p, t))
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def decide(scores, names, th, margin):
    ii = [i for i, n in enumerate(names) if n != "非物品"]
    ni = [i for i, n in enumerate(names) if n == "非物品"]
    ip = scores[ii]
    o = ip.argsort(descending=True)
    bp = ip[o[0]].item()
    pred = names[ii[o[0].item()]]
    sec = ip[o[1]].item() if len(o) > 1 else 0.0
    mg = (bp - sec) / bp if bp > 1e-12 else 0.0
    neg = max([scores[i].item() for i in ni], default=0.0)
    why = "阈值" if bp < th else ("负类" if neg > bp else ("间隔" if mg < margin else ""))
    return ("★未知" if why else pred), bp, why


def run_english(data):
    import torch, open_clip
    from PIL import Image
    m, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    m.eval(); tk = open_clip.get_tokenizer("ViT-B-32")

    t0 = time.time()
    F = []
    with torch.no_grad():
        for k in NAMES:
            descs = EN_CATS.get(k, EN_NEG)
            prompts = [t.format(d) for d in descs for t in EN_T]
            f = m.encode_text(tk(prompts))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0)
            F.append(f / f.norm(dim=-1, keepdim=True))
        TF = torch.stack(F)
    build = time.time() - t0
    scale = m.logit_scale.exp()

    det, lat = [], []
    with torch.no_grad():
        for p, tr in data:
            t1 = time.time()
            x = pre(Image.open(p).convert("RGB")).unsqueeze(0)
            f = m.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            s = (scale * (f @ TF.T)).softmax(-1)[0]
            lat.append((time.time() - t1) * 1000)
            pred, bp, why = decide(s, NAMES, TH, MARGIN)
            det.append((os.path.basename(p), tr, pred, bp, why))
    return dict(build=build, lat=sum(lat) / len(lat), det=det,
                params=sum(p.numel() for p in m.parameters()))


def run_chinese(data):
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    m = ChineseCLIPModel.from_pretrained(CN_PATH); m.eval()
    pr = ChineseCLIPProcessor.from_pretrained(CN_PATH)

    t0 = time.time()
    F = []
    with torch.no_grad():
        for k in NAMES:
            descs = CN.get(k, CN_NEG)
            prompts = [t.format(d) for d in descs for t in TEMPLATES_CN]
            tk = pr(text=prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=52)
            # ★ 5.x：取 pooler_output 并手动归一化
            f = m.get_text_features(**tk).pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0)
            F.append(f / f.norm(dim=-1, keepdim=True))
        TF = torch.stack(F)
    build = time.time() - t0
    scale = m.logit_scale.exp()

    det, lat = [], []
    with torch.no_grad():
        for p, tr in data:
            t1 = time.time()
            ii = pr(images=Image.open(p).convert("RGB"), return_tensors="pt")
            fi = m.get_image_features(**ii).pooler_output
            fi = fi / fi.norm(dim=-1, keepdim=True)
            s = (scale * (fi @ TF.T)).softmax(-1)[0]
            lat.append((time.time() - t1) * 1000)
            pred, bp, why = decide(s, NAMES, TH, MARGIN)
            det.append((os.path.basename(p), tr, pred, bp, why))
    return dict(build=build, lat=sum(lat) / len(lat), det=det,
                params=sum(p.numel() for p in m.parameters()))


def score(r, n_item, n_neg):
    ok = rej = wrong = fa = negok = 0
    for n, tr, pred, bp, why in r["det"]:
        if tr == "非物品":
            if pred == "★未知":
                negok += 1
            else:
                fa += 1
        else:
            if pred == tr:
                ok += 1
            elif pred == "★未知":
                rej += 1
            else:
                wrong += 1
    return ok, rej, wrong, negok, fa


def main():
    data = collect()
    n_item = sum(1 for _, t in data if t != "非物品")
    n_neg = sum(1 for _, t in data if t == "非物品")
    P(f"测试集：{n_item} 张物品 + {n_neg} 张非物品"
      f"（{len(set(t for _, t in data if t != '非物品'))} 个类别）"
      f" ｜ th={TH} margin={MARGIN}\n")

    P("跑英文 CLIP ...")
    en = run_english(data)
    P("跑中文 CLIP ...")
    cn = run_chinese(data)

    e = score(en, n_item, n_neg)
    c = score(cn, n_item, n_neg)

    P()
    P("=" * 86)
    P(f"{'模型':<30}{'物品正确':<12}{'误拒':<7}{'错分':<7}{'拒识':<9}{'误接受':<9}")
    P("-" * 86)
    P(f"{'英文 CLIP ViT-B/32 (151M)':<30}{f'{e[0]}/{n_item}':<12}{e[1]:<7}{e[2]:<7}"
      f"{f'{e[3]}/{n_neg}':<9}{e[4]:<9}")
    P(f"{'中文 CLIP ViT-B/16 (188M)':<30}{f'{c[0]}/{n_item}':<12}{c[1]:<7}{c[2]:<7}"
      f"{f'{c[3]}/{n_neg}':<9}{c[4]:<9}")
    P("=" * 86)

    P(f"\n{'':<30}{'英文 CLIP':<16}{'中文 CLIP':<16}")
    P(f"{'建原型（43 类）':<30}{en['build']:>7.1f}s{'':<8}{cn['build']:>7.1f}s")
    P(f"{'单张推理':<30}{en['lat']:>7.0f}ms{'':<7}{cn['lat']:>7.0f}ms")
    P(f"{'参数量':<30}{en['params']/1e6:>7.1f}M{'':<7}{cn['params']/1e6:>7.1f}M")

    # ── 逐图并排 ────────────────────────────────────────────
    P("\n【逐图并排】（✓ 正确 / ✗ 错分 / ~ 误拒 / F 误接受）")
    P(f"  {'文件':<17}{'真值':<8}{'英文 CLIP':<20}{'中文 CLIP':<20}")
    P("  " + "-" * 63)

    def mark(tr, pred):
        if tr == "非物品":
            return "✓拒绝" if pred == "★未知" else "F→" + pred
        if pred == tr:
            return "✓"
        if pred == "★未知":
            return "~误拒"
        return "✗" + pred

    agree = 0
    for (n1, t1, p1, b1, w1), (n2, t2, p2, b2, w2) in zip(en["det"], cn["det"]):
        if p1 == p2:
            agree += 1
        P(f"  {n1:<17}{t1:<8}{mark(t1, p1):<20}{mark(t2, p2):<20}")
    P("  " + "-" * 63)
    P(f"  两模型判定完全一致的图：{agree}/{n_item + n_neg}")

    P("\n【结论】")
    if e[:2] == c[:2] and e[2:] == c[2:]:
        P("  ★ 准确率指标完全相同 —— 中文描述可以无损替代英文描述")
    else:
        P(f"  ⚠ 有差异：英文 {e} vs 中文 {c}")


if __name__ == "__main__":
    main()

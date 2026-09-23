#!/usr/bin/env python
"""中文 CLIP（Chinese-CLIP ViT-B/16）vs 英文 CLIP —— 同一测试集对照

目的：验证「用中文描述更好维护」这个诉求是否成立，以及准确率代价。
"""
import os, sys, glob, re, time, json

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN_PATH = os.path.join(ROOT, ".hf", "chinese-clip-vit-base-patch16")
P = lambda *a: print(*a, flush=True)

TH_CLIP = 0.70          # 英文 CLIP 实测标定值
MARGIN = 0.30

# ══ 42 类的中文描述（维护者会怎么写）═══════════════════════
CN = {
"身份证":   ["身份证", "一张身份证", "居民身份证"],
"驾驶证":   ["驾驶证", "驾照", "机动车驾驶证"],
"行驶证":   ["行驶证", "车辆行驶证", "机动车行驶证"],
"银行卡":   ["银行卡", "一张银行卡", "信用卡"],
"公交卡":   ["公交卡", "交通卡", "公交IC卡"],
"票据":     ["票据", "收据", "发票", "纸质票据"],
"现金":     ["现金", "纸币", "一叠钞票", "人民币"],
"手机":     ["手机", "智能手机", "一部手机"],
"手机充电器": ["手机充电器", "充电头", "电源适配器"],
"充电宝":   ["充电宝", "移动电源"],
"耳机":     ["耳机", "有线耳机", "蓝牙耳机"],
"手表":     ["手表", "腕表", "机械表"],
"平板电脑": ["平板电脑", "平板", "iPad"],
"笔记本电脑": ["笔记本电脑", "笔记本", "一台笔记本电脑"],
"U盘":      ["U盘", "闪存盘", "USB存储设备"],
"电子烟":   ["电子烟", "电子烟杆"],
"香烟":     ["香烟", "一包香烟", "烟盒"],
"打火机":   ["打火机", "一次性打火机"],
"钱包":     ["钱包", "皮钱包", "钱夹"],
"手提包":   ["手提包", "女式手提包", "单肩包"],
"背包":     ["背包", "双肩包", "书包"],
"戒指":     ["戒指", "金戒指", "一枚戒指"],
"项链":     ["项链", "金项链", "带吊坠的项链"],
"手链手镯": ["手链", "手镯", "一串手链"],
"耳环":     ["耳环", "一对耳环", "耳钉"],
"眼镜":     ["眼镜", "一副眼镜", "近视眼镜"],
"帽子":     ["帽子", "鸭舌帽", "一顶帽子"],
"围巾":     ["围巾", "戴在脖子上的围巾", "毛线围巾"],
"皮带":     ["皮带", "腰带", "一条皮带"],
"口罩":     ["口罩", "一次性口罩", "戴在脸上的口罩"],
"钥匙":     ["钥匙", "一串钥匙", "一把钥匙"],
"笔":       ["笔", "圆珠笔", "一支笔"],
"本子":     ["本子", "笔记本", "一本记事本"],
"纸巾":     ["纸巾", "一包纸巾", "抽纸"],
"水杯":     ["水杯", "保温杯", "杯子"],
"雨伞":     ["雨伞", "折叠伞", "一把伞"],
"梳子":     ["梳子", "一把梳子"],
"镜子":     ["镜子", "小镜子", "手持镜子"],
"指甲刀":   ["指甲刀", "指甲剪"],
"药品":     ["药品", "药盒", "一板药片", "药瓶"],
"护肤化妆": ["护肤品", "化妆品", "护手霜", "一支口红"],
"零食":     ["零食", "一袋零食", "薯片", "糖果"],
}

CN_NEG = ["桌子", "空托盘", "白板", "屏幕截图", "一份文件",
          "一页文字", "表格", "人脸", "一双手", "水果", "蔬菜",
          "灰色布料", "一面墙", "空白背景", "一张素描画"]

TEMPLATES_CN = ["一张{}的照片", "{}", "一个{}"]


def truth_of(p):
    b = os.path.basename(p)
    if re.match(r'^n\d', b):
        return "非物品"
    for k in sorted(CN.keys(), key=len, reverse=True):
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
                if t and (t in CN or t == "非物品"):
                    out.append((p, t))
    return sorted(set(out), key=lambda x: (x[1], x[0]))


def decide(scores, names, neg_name, th, margin):
    ii = [i for i, n in enumerate(names) if n != neg_name]
    ni = [i for i, n in enumerate(names) if n == neg_name]
    ip = scores[ii]
    o = ip.argsort(descending=True)
    bp = ip[o[0]].item()
    pred = names[ii[o[0].item()]]
    sec = ip[o[1]].item() if len(o) > 1 else 0.0
    mg = (bp - sec) / bp if bp > 1e-12 else 0.0
    neg = max([scores[i].item() for i in ni], default=0.0)
    why = ""
    if bp < th:
        why = "阈值"
    elif neg > bp:
        why = "负类"
    elif mg < margin:
        why = "间隔"
    return ("★未知" if why else pred), bp, neg, mg, why


def main():
    import torch
    from PIL import Image
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    data = collect()
    n_item = sum(1 for _, t in data if t != "非物品")
    n_neg = sum(1 for _, t in data if t == "非物品")
    P(f"测试集：{n_item} 张物品 + {n_neg} 张非物品（{len(set(t for _,t in data if t!='非物品'))} 个类别）\n")

    # ══ 中文 CLIP ══
    P("=" * 78)
    P("中文 CLIP · Chinese-CLIP ViT-B/16（188M 参数）")
    P("=" * 78)
    t0 = time.time()
    m = ChineseCLIPModel.from_pretrained(CN_PATH)
    pr = ChineseCLIPProcessor.from_pretrained(CN_PATH)
    m.eval()
    P(f"  模型加载 {time.time()-t0:.1f}s")

    names = list(CN.keys()) + ["非物品"]
    # 文本原型
    t0 = time.time()
    F = []
    with torch.no_grad():
        for k in names:
            descs = CN.get(k, CN_NEG)
            prompts = [t.format(d) for d in descs for t in TEMPLATES_CN]
            tk = pr(text=prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=52)
            # ★ transformers 5.x：get_text_features 返回输出对象，
            #   pooler_output 才是投影后的嵌入，且【不自动归一化】
            f = m.get_text_features(**tk).pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(0)
            F.append(f / f.norm(dim=-1, keepdim=True))
        TF = torch.stack(F)
    build_cn = time.time() - t0
    P(f"  建原型（43 类 × 中文描述）{build_cn:.1f}s")

    scale = m.logit_scale.exp().item()
    P(f"  logit_scale = {scale:.2f}   logit_bias = {getattr(m,'logit_bias',None)}")

    # 图像特征 + 判定
    res_cn = []
    lat = []
    with torch.no_grad():
        for p, tr in data:
            t1 = time.time()
            img = Image.open(p).convert("RGB")
            ii = pr(images=img, return_tensors="pt")
            fi = m.get_image_features(**ii).pooler_output
            fi = fi / fi.norm(dim=-1, keepdim=True)
            scores = (scale * (fi @ TF.T)).softmax(-1)[0]
            lat.append((time.time() - t1) * 1000)
            pred, bp, neg, mg, why = decide(scores, names, "非物品", TH_CLIP, MARGIN)
            res_cn.append((os.path.basename(p), tr, pred, bp, why))
    # 用同一阈值跑一遍（先看原始分布）
    P(f"  单张耗时 {sum(lat)/len(lat):.0f} ms")

    # 分数分布
    pos_scores = [r[3] for r in res_cn if r[1] != "非物品"]
    P(f"  物品 Top-1 得分范围 {min(pos_scores):.4f} ~ {max(pos_scores):.4f}")

    # 阈值扫描找中文 CLIP 自己的最佳点
    P("\n  阈值扫描（中文 CLIP 需要重新标定，英文的 0.70 不一定适用）:")
    P(f"    {'阈值':<8}{'物品正确':<11}{'误拒':<7}{'错分':<7}{'拒识':<9}{'误接受':<8}")
    best = None
    for th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80]:
        ok = rej = wrong = fa = negok = 0
        for p, tr in data:
            t1 = time.time()
            with torch.no_grad():
                img = Image.open(p).convert("RGB")
                ii = pr(images=img, return_tensors="pt")
                fi = m.get_image_features(**ii).pooler_output
                fi = fi / fi.norm(dim=-1, keepdim=True)
                scores = (scale * (fi @ TF.T)).softmax(-1)[0]
            pred, bp, neg, mg, why = decide(scores, names, "非物品", th, MARGIN)
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
        mark = ""
        if fa == 0 and (best is None or ok > best[1]):
            best = (th, ok)
            mark = "  ★"
        P(f"    {th:<8.2f}{f'{ok}/{n_item}':<11}{rej:<7}{wrong:<7}"
          f"{f'{negok}/{n_neg}':<9}{fa:<8}{mark}")
    if best:
        P(f"\n  零误接受前提下的最优：阈值 {best[0]}，物品 {best[1]}/{n_item}")

    # ══ 明细 ══
    th_use = best[0] if best else 0.70
    P(f"\n  逐图明细（th={th_use}）:")
    for p, tr in data:
        with torch.no_grad():
            img = Image.open(p).convert("RGB")
            ii = pr(images=img, return_tensors="pt")
            fi = m.get_image_features(**ii).pooler_output
            fi = fi / fi.norm(dim=-1, keepdim=True)
            scores = (scale * (fi @ TF.T)).softmax(-1)[0]
        pred, bp, neg, mg, why = decide(scores, names, "非物品", th_use, MARGIN)
        good = (tr == pred) or (tr == "非物品" and pred == "★未知")
        P(f"    {'✓' if good else '✗'} {os.path.basename(p):<16} 真值={tr:<6} → {pred:<8}"
          f"{bp:.3f} {('['+why+']') if why else ''}")

    P(f"\n  建原型 {build_cn:.1f}s ｜ 单张 {sum(lat)/len(lat):.0f} ms")


if __name__ == "__main__":
    main()

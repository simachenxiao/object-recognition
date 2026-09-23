"""
Chinese-CLIP 模型封装 —— 命令行脚本共用

`01_clip_zeroshot.py` 和 `02_eval_tray.py` 通过本模块调用模型。
`web_demo.py` 为了保持【单文件可直接拷走】的特性，内置了一份等价实现
（见该文件顶部的「模型管理」段）。改模型逻辑时两处都要改。

⚠️ transformers 5.x 的 API 坑（实测踩过）：
    1. `get_text_features()` / `get_image_features()` 返回的是【输出对象】
       而不是张量，真正的嵌入在 `.pooler_output`
    2. 它们【不自动归一化】（模长约 36）
       必须自己除模长，否则相似度大两个数量级，全判错
"""
import os
import time

# ── 环境变量必须在 import transformers 前设置 ─────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CNCLIP_LOCAL_DIR = os.path.join(PROJECT_DIR, ".hf", "chinese-clip-vit-base-patch16")
CNCLIP_REPO = "OFA-Sys/chinese-clip-vit-base-patch16"

BACKENDS = {
    "cnclip": {
        "label": "Chinese-CLIP ViT-B/16",
        "repo": CNCLIP_REPO,
        "mode": "softmax",
        "th": 0.70,
        "note": "中文描述 · 民警可直接改；建原型快，单张比英文 CLIP 慢约 2 倍",
    },
}

# 中文提示模板。实测：这套模板 + 中文描述，准确率与「英文 CLIP + 英文描述」完全相同
TEMPLATES = ["一张{}的照片", "{}", "一个{}"]
MAX_TEXT_LEN = 52

_cache = {}


def libs():
    """惰性导入重型依赖"""
    if not _cache:
        import torch
        from PIL import Image
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        _cache.update(torch=torch, Image=Image,
                      ChineseCLIPModel=ChineseCLIPModel,
                      ChineseCLIPProcessor=ChineseCLIPProcessor)
    return _cache


def model_source():
    """本地目录优先（离线可用），否则回落到 HF 仓库 id"""
    if os.path.isdir(CNCLIP_LOCAL_DIR):
        return CNCLIP_LOCAL_DIR, "本地目录"
    return CNCLIP_REPO, "HF 仓库（需联网）"


def load_model(key="cnclip", verbose=True):
    """加载模型 + processor（同一次运行内缓存）"""
    ck = key + "_model"
    if ck in _cache:
        return _cache[ck]
    L = libs()
    src, where = model_source()
    if verbose:
        print(f"  加载 {BACKENDS[key]['label']} ← {where} ...", end="", flush=True)
    t = time.time()
    model = L["ChineseCLIPModel"].from_pretrained(src)
    model.eval()
    processor = L["ChineseCLIPProcessor"].from_pretrained(src)
    if verbose:
        print(f" {time.time()-t:.1f}s", flush=True)
    _cache[ck] = (model, processor)
    return _cache[ck]


def encode_text(model, processor, prompts):
    """文本编码 → 已归一化嵌入 (N, 512)"""
    torch = libs()["torch"]
    with torch.no_grad():
        tk = processor(text=prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=MAX_TEXT_LEN)
        f = model.get_text_features(**tk).pooler_output
        return f / f.norm(dim=-1, keepdim=True)


def encode_image(model, processor, pil_img):
    """图像编码 → 已归一化嵌入 (1, 512)"""
    torch = libs()["torch"]
    with torch.no_grad():
        ii = processor(images=pil_img, return_tensors="pt")
        f = model.get_image_features(**ii).pooler_output
        return f / f.norm(dim=-1, keepdim=True)


def build_features(classes, templates=None, model=None, processor=None):
    """classes: {名称: [描述]} → (特征矩阵, 名称列表)

    注意：本函数不知道哪类是负类，负类判定由调用方按名称处理。
    """
    torch = libs()["torch"]
    templates = templates or TEMPLATES
    if model is None:
        model, processor = load_model()
    feats, names = [], []
    for zh, descs in classes.items():
        prompts = [t.format(d) for d in descs for t in templates]
        f = encode_text(model, processor, prompts).mean(dim=0)
        feats.append(f / f.norm(dim=-1, keepdim=True))
        names.append(zh)
    return torch.stack(feats), names


def infer(pil_img, TF, names, threshold, margin, neg_name="非物品", model=None):
    """单张推理。

    返回 dict:
        verdict     判定结果（类别名 / ★未知）
        category    命中的类别名，拒识时为 None
        score       Top-1 得分
        neg_score   负类得分
        margin      相对间隔 (top1-top2)/top1
        topk        [(类别, 得分)] × 3
        reason      '' / threshold / negative / margin
        ms          耗时(ms)
    """
    model = model or load_model()[0]
    torch = libs()["torch"]
    processor = load_model()[1]

    t0 = time.time()
    f = encode_image(model, processor, pil_img)
    with torch.no_grad():
        sim = f @ TF.T
        pr = (model.logit_scale.exp() * sim).softmax(dim=-1)[0]
    ms = (time.time() - t0) * 1000

    ii = [i for i, n in enumerate(names) if n != neg_name]
    ni = [i for i, n in enumerate(names) if n == neg_name]
    if not ii:
        return dict(verdict="★无类别", category=None, score=0.0, neg_score=0.0,
                    margin=0.0, topk=[], reason="nocat", ms=ms)

    ip = pr[ii]
    order = ip.argsort(descending=True)
    best_p = ip[order[0]].item()
    pred = names[ii[order[0].item()]]
    sec = ip[order[1]].item() if len(order) > 1 else 0.0
    mg = (best_p - sec) / best_p if best_p > 1e-12 else 0.0
    neg = max([pr[i].item() for i in ni], default=0.0)

    if best_p < threshold:
        why = "threshold"
    elif neg > best_p:
        why = "negative"
    elif mg < margin:
        why = "margin"
    else:
        why = ""

    topk = [(names[ii[o.item()]], ip[o].item()) for o in order[:3]]
    return dict(verdict="★未知" if why else pred, category=None if why else pred,
                score=best_p, neg_score=neg, margin=mg, topk=topk,
                reason=why, ms=ms)

#!/usr/bin/env python
"""
随身物品识别 —— Demo Web 应用（Chinese-CLIP 版）

特点:
    · 后端只用一个模型：Chinese-CLIP ViT-B/16（OFA-Sys）
    · 类别描述用【中文】—— 民警可直接阅读和修改
    · 纯 Python 标准库实现（http.server），零额外 Web 依赖
    · 完全离线可用，适合内网
    · 可视化编辑分类（增删改类别与中文描述）
    · 上传文件夹或单张图片，逐个输出分类结果
    · 按文件名自动比对真值，计算出准确率

用法:
    python web_demo.py                # 默认 http://127.0.0.1:8000
    python web_demo.py --port 8080
    python web_demo.py --no-browser   # 不自动打开浏览器

模型来源（按优先级）:
    1. 项目内 .hf/chinese-clip-vit-base-patch16/   ← 离线部署用这个
    2. HF 仓库 OFA-Sys/chinese-clip-vit-base-patch16
       （需联网，走 HF_ENDPOINT 镜像）

依赖（已在 .venv 中）:
    torch, torchvision, transformers, pillow
"""
import os

# ── 环境变量必须在 import 前设置 ─────────────────────────────
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import sys
import io
import json
import time
import base64
import hashlib
import argparse
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ★ 两个目录要分清：
#   APP_DIR  = 本文件所在目录（src/）—— 配置、特征缓存放这里
#   ROOT_DIR = 项目根目录            —— .hf 模型缓存在这里
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
PROJECT_DIR = APP_DIR          # 旧名，保持兼容

# ── 重型依赖惰性加载 ────────────────────────────────────────
# torch + transformers 导入需数秒。若在模块级别导入，HTTP 服务要等这么久才监听；
# 改成首次用到时才导入，页面可以秒开。
_LIB = {}


def libs():
    if not _LIB:
        print("[初始化] 加载 torch / transformers ...", flush=True)
        t = time.time()
        import torch
        from PIL import Image
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
        try:                       # 静音 "Loading weights: 100%|..." 进度条
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:
            pass
        _LIB["torch"] = torch
        _LIB["Image"] = Image
        _LIB["ChineseCLIPModel"] = ChineseCLIPModel
        _LIB["ChineseCLIPProcessor"] = ChineseCLIPProcessor
        apply_threads()
        print(f"[初始化] 就绪 ({time.time()-t:.1f}s)"
              f" ｜ torch 线程 {torch.get_num_threads()}", flush=True)
    return _LIB


# ── torch 线程数 ─────────────────────────────────────────────
# 实测（Ryzen 5 3500U，12 张图）：4 线程 635ms/张 → 8 线程 459ms/张，约 1.4x

def resolve_threads(n=None):
    """解析线程数：0 / 负数 / None / 非法值 → 自动用满逻辑核"""
    if n is None:
        n = (_config or {}).get("threads", DEFAULT_CONFIG.get("threads", 0))
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else (os.cpu_count() or 4)


def apply_threads(n=None, verbose=False):
    """把线程数应用到 torch。

    torch 未导入 → 设 OMP/MKL 环境变量（torch 首次 import 会读）；
    已导入 → 再调 set_num_threads。两条路都走，任何调用顺序都生效。
    """
    n = resolve_threads(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    if "torch" in _LIB:
        _LIB["torch"].set_num_threads(n)
    if verbose:
        print(f"[线程] torch 线程数 = {n}", flush=True)
    return n


# ══════════════════════════════════════════════════════════════
# 后端配置 —— 只有 Chinese-CLIP
# ══════════════════════════════════════════════════════════════
CNCLIP_LOCAL_DIR = os.path.join(ROOT_DIR, ".hf", "chinese-clip-vit-base-patch16")
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

# 中文提示模板。
# ★ 实测（35 张图）：这套模板 + 中文描述，准确率与英文 CLIP + 英文描述完全相同
#   （物品 21/28 · 误拒 7 · 错分 0 · 拒识 7/7 · 误接受 0）
TEMPLATES = ["一张{}的照片", "{}", "一个{}"]

# 中文 BERT 文本塔的上下文长度
MAX_TEXT_LEN = 52

CONFIG_FILE = os.path.join(APP_DIR, "web_demo_categories.json")
# ★ 文本原型磁盘缓存目录：类别表不变时直接读文件，不用重新编码
FEAT_CACHE_DIR = os.path.join(APP_DIR, ".feat_cache")
# ★ 图像特征缓存目录：按【图片内容哈希】存 (1,512) 向量。
#   只与模型/预处理有关，改分类或调阈值都不会失效 → 重跑同一批图秒出。
IMG_CACHE_DIR = os.path.join(FEAT_CACHE_DIR, "img")
IMG_CACHE_MAX = 5000          # 条数上限（每条约 2KB → 约 10MB）

DEFAULT_CONFIG = {
    "thresholds": {"cnclip": 0.70},
    # ★ 相对间隔：(top1 - top2) / top1。低于此值 → 判为「两类别接近，待人工确认」
    "margin": 0.30,
    # ★ torch 线程数：0 = 自动用满逻辑核（实测 4→8 线程提速约 1.4x）
    "threads": 0,
    # ★ 启动时后台预热模型（省掉首次点击等 ~10s）
    "preload": True,
    "categories": [
        {"name": "身份证", "descs": ["身份证", "一张身份证", "居民身份证"], "negative": False},
        {"name": "驾驶证", "descs": ["驾驶证", "驾照", "机动车驾驶证"], "negative": False},
        {"name": "行驶证", "descs": ["行驶证", "车辆行驶证", "机动车行驶证"], "negative": False},
        {"name": "银行卡", "descs": ["银行卡", "一张银行卡", "信用卡"], "negative": False},
        {"name": "公交卡", "descs": ["公交卡", "交通卡", "公交IC卡"], "negative": False},
        {"name": "票据", "descs": ["票据", "收据", "发票", "纸质票据"], "negative": False},
        {"name": "现金", "descs": ["现金", "纸币", "钞票", "一叠钞票", "人民币", "硬币", "一把硬币"], "negative": False},
        {"name": "手机", "descs": ["手机", "智能手机", "一部手机"], "negative": False},
        {"name": "手机充电器", "descs": ["手机充电器", "充电头", "电源适配器"], "negative": False},
        {"name": "充电宝", "descs": ["充电宝", "移动电源"], "negative": False},
        {"name": "耳机", "descs": ["耳机", "有线耳机", "蓝牙耳机"], "negative": False},
        {"name": "手表", "descs": ["手表", "腕表", "机械表"], "negative": False},
        {"name": "平板电脑", "descs": ["平板电脑", "平板", "iPad"], "negative": False},
        {"name": "笔记本电脑", "descs": ["笔记本电脑", "笔记本", "一台笔记本电脑"], "negative": False},
        {"name": "U盘", "descs": ["U盘", "闪存盘", "USB存储设备"], "negative": False},
        {"name": "电子烟", "descs": ["电子烟", "电子烟杆"], "negative": False},
        {"name": "香烟", "descs": ["香烟", "一包香烟", "烟盒"], "negative": False},
        {"name": "打火机", "descs": ["打火机", "一次性打火机"], "negative": False},
        {"name": "钱包", "descs": ["钱包", "皮钱包", "钱夹"], "negative": False},
        {"name": "手提包", "descs": ["手提包", "女式手提包", "单肩包"], "negative": False},
        {"name": "背包", "descs": ["背包", "双肩包", "书包"], "negative": False},
        {"name": "戒指", "descs": ["戒指", "金戒指", "一枚戒指"], "negative": False},
        {"name": "项链", "descs": ["项链", "金项链", "带吊坠的项链"], "negative": False},
        {"name": "手链手镯", "descs": ["手链", "手镯", "一串手链"], "negative": False},
        {"name": "耳环", "descs": ["耳环", "一对耳环", "耳钉"], "negative": False},
        {"name": "眼镜", "descs": ["眼镜", "一副眼镜", "近视眼镜"], "negative": False},
        {"name": "帽子", "descs": ["帽子", "鸭舌帽", "一顶帽子"], "negative": False},
        {"name": "围巾", "descs": ["围巾", "戴在脖子上的围巾", "毛线围巾"], "negative": False},
        {"name": "皮带", "descs": ["皮带", "腰带", "一条皮带"], "negative": False},
        {"name": "口罩", "descs": ["口罩", "一次性口罩", "戴在脸上的口罩"], "negative": False},
        {"name": "钥匙", "descs": ["钥匙", "一串钥匙", "一把钥匙"], "negative": False},
        {"name": "笔", "descs": ["笔", "圆珠笔", "一支笔"], "negative": False},
        {"name": "本子", "descs": ["本子", "笔记本", "一本记事本"], "negative": False},
        {"name": "剪刀", "descs": ["剪刀", "一把剪刀", "小剪刀"], "negative": False},
        {"name": "纸巾", "descs": ["纸巾", "一包纸巾", "抽纸"], "negative": False},
        {"name": "水杯", "descs": ["水杯", "保温杯", "杯子"], "negative": False},
        {"name": "雨伞", "descs": ["雨伞", "折叠伞", "一把伞"], "negative": False},
        {"name": "梳子", "descs": ["梳子", "一把梳子"], "negative": False},
        {"name": "镜子", "descs": ["镜子", "小镜子", "手持镜子"], "negative": False},
        {"name": "指甲刀", "descs": ["指甲刀", "指甲剪"], "negative": False},
        {"name": "药品", "descs": ["药品", "药盒", "一板药片", "药瓶"], "negative": False},
        {"name": "护肤化妆", "descs": ["护肤品", "化妆品", "护手霜", "一支口红"], "negative": False},
        {"name": "零食", "descs": ["零食", "一袋零食", "薯片", "糖果"], "negative": False},
        {"name": "非物品", "negative": True, "descs": [
            "桌子", "空托盘", "白板", "屏幕截图", "一份文件",
            "一页文字", "表格", "人脸", "一双手", "水果", "蔬菜",
            "灰色布料", "一面墙", "空白背景", "一张素描画",
        ]},
    ],
}


# ══════════════════════════════════════════════════════════════
# 配置读写
# ══════════════════════════════════════════════════════════════
_cfg_lock = threading.Lock()
_config = None


def load_config():
    global _config
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                _config = json.load(f)
            # 补全缺失字段
            _config.setdefault("margin", 0.30)
            _config.setdefault("thresholds", {})
            _config.setdefault("threads", DEFAULT_CONFIG.get("threads", 0))
            _config.setdefault("preload", DEFAULT_CONFIG.get("preload", True))
            for k, v in BACKENDS.items():
                _config["thresholds"].setdefault(k, v["th"])
            # 兼容：万一配置里只有旧后端名的阈值，迁移到 cnclip
            if "cnclip" not in _config["thresholds"]:
                for old in ("clip", "siglip"):
                    if old in _config["thresholds"]:
                        _config["thresholds"]["cnclip"] = _config["thresholds"][old]
                        break
            return _config
        except Exception as e:
            print(f"[WARN] 读取配置失败，用默认值: {e}")
    import copy
    _config = copy.deepcopy(DEFAULT_CONFIG)
    return _config


def save_config(cfg):
    global _config
    with _cfg_lock:
        _config = cfg
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    apply_threads(verbose=True)      # 线程数改动即时生效（torch 已导入时）
    return cfg


# ══════════════════════════════════════════════════════════════
# 模型管理
# ══════════════════════════════════════════════════════════════
_model_lock = threading.Lock()
_MODELS = {}      # key -> (model, processor, cfg)
_FEATS = {}       # (key, 指纹) -> (text_feat, names, is_neg)
# ★ 每个后端一把锁：避免「预热还没跑完，用户就点了识别」时
#   两个线程同时编码同一份原型（白白多花几十秒）
_feat_locks = {}
_feat_locks_guard = threading.Lock()


def _feat_lock_for(key):
    with _feat_locks_guard:
        if key not in _feat_locks:
            _feat_locks[key] = threading.Lock()
        return _feat_locks[key]


def get_model(key):
    """加载 Chinese-CLIP（优先本地目录，离线可用）"""
    with _model_lock:
        if key in _MODELS:
            return _MODELS[key]
        L = libs()
        cfg = BACKENDS[key]
        src = CNCLIP_LOCAL_DIR if os.path.isdir(CNCLIP_LOCAL_DIR) else cfg["repo"]
        where = "本地目录" if os.path.isdir(CNCLIP_LOCAL_DIR) else "HF 仓库（需联网）"
        print(f"[模型] 加载 {cfg['label']} ← {where} ...", flush=True)
        t = time.time()
        model = L["ChineseCLIPModel"].from_pretrained(src)
        model.eval()
        processor = L["ChineseCLIPProcessor"].from_pretrained(src)
        print(f"[模型] {cfg['label']} 就绪 ({time.time()-t:.1f}s)", flush=True)
        _MODELS[key] = (model, processor, cfg)
        return _MODELS[key]


# ── 特征编码（Chinese-CLIP 的 API 与英文 CLIP 不同，见下注释）──
def encode_text(model, processor, prompts):
    """文本编码 → 已归一化嵌入 (N, 512)

    ★ transformers 5.x 的坑：
      1. get_text_features() 返回的是【输出对象】，不是张量，
         真正的嵌入在 .pooler_output
      2. 它【不自动归一化】（模长约 36），必须自己除模长
         否则相似度会大两个数量级，全判错
    """
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


# ── 图像特征缓存（按图片内容哈希）────────────────────────────
def img_cache_path(key, digest):
    return os.path.join(IMG_CACHE_DIR, f"{key}_{digest}.pt")


def encode_image_cached(model, processor, pil_img, key, digest=None):
    """图像编码 + 内容哈希缓存 → (1, 512) 已归一化

    ★ 图像特征只跟【模型 + 预处理】有关，与类别表/阈值无关，
      所以改分类、调阈值都不会让它失效 —— 重跑同一批图直接命中。
    """
    torch = libs()["torch"]
    path = img_cache_path(key, digest) if digest else None
    if path and os.path.isfile(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            pass
    f = encode_image(model, processor, pil_img)
    if path:
        try:
            os.makedirs(IMG_CACHE_DIR, exist_ok=True)
            with _IMG_CACHE_LOCK:
                torch.save(f, path)
        except Exception:
            pass
    return f


def clean_img_cache(max_keep=IMG_CACHE_MAX, verbose=True):
    """图像特征缓存按修改时间保留最新 max_keep 条，其余删除"""
    if not os.path.isdir(IMG_CACHE_DIR):
        return 0
    files = []
    for f in os.listdir(IMG_CACHE_DIR):
        p = os.path.join(IMG_CACHE_DIR, f)
        if os.path.isfile(p) and f.endswith(".pt"):
            files.append((os.path.getmtime(p), p))
    if len(files) <= max_keep:
        return 0
    files.sort(reverse=True)
    removed = 0
    for _, p in files[max_keep:]:
        try:
            os.remove(p)
            removed += 1
        except OSError:
            pass
    if verbose and removed:
        print(f"[缓存] 图像特征超出 {max_keep} 条，清理最旧 {removed} 条", flush=True)
    return removed


def feat_signature(key):
    """文本特征指纹 —— 只与【影响文本特征的字段】有关。

    ★ 关键：不含 thresholds / margin。
      所以「只改阈值」不会让指纹变化 → 不会白白重建原型。
    ★ 含 TEMPLATES：改模板必须重建。
    """
    payload = {
        "backend": key,
        "repo": BACKENDS[key]["repo"],
        "templates": TEMPLATES,
        "max_text_len": MAX_TEXT_LEN,
        "categories": [[c.get("name"), list(c.get("descs", [])),
                        bool(c.get("negative"))]
                       for c in _config["categories"]],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def clean_feat_cache(max_keep=8, verbose=True):
    """清理 .feat_cache

    规则：
        · 当前配置用到的指纹 → 必留
        · 其余按修改时间由新到旧，最多再留 max_keep 个
        · 再剩下的当孤儿删掉

    只处理【自己命名的 .pt】文件，不动目录里的其他东西。
    """
    if not os.path.isdir(FEAT_CACHE_DIR):
        return 0
    try:
        keep_now = {f"{k}_{feat_signature(k)}.pt" for k in BACKENDS}
    except Exception:
        return 0
    files = []
    for f in os.listdir(FEAT_CACHE_DIR):
        p = os.path.join(FEAT_CACHE_DIR, f)
        if os.path.isfile(p) and f.endswith(".pt"):
            files.append((os.path.getmtime(p), f, p))
    files.sort(reverse=True)
    kept, removed = [], []
    for mt, f, p in files:
        if f in keep_now or len(kept) < max_keep:
            kept.append(f)
        else:
            try:
                os.remove(p)
                removed.append(f)
            except OSError:
                pass
    if verbose and removed:
        print(f"[缓存] 特征原型保留 {len(kept)} 个，清理孤儿 {len(removed)} 个",
              flush=True)
        for f in removed:
            print(f"[缓存]   已删 {f}", flush=True)
    return len(removed)


def get_features(key):
    """构建 / 复用类别原型向量

    三级缓存：内存 _FEATS → 磁盘 .feat_cache/*.pt → 现编码
    """
    sig = feat_signature(key)
    cache_key = (key, sig)
    if cache_key in _FEATS:
        return _FEATS[cache_key]

    with _feat_lock_for(key):
        if cache_key in _FEATS:
            print(f"[特征] {key} 复用另一线程刚建好的原型", flush=True)
            return _FEATS[cache_key]
        return _build_features(key, sig, cache_key)


def _build_features(key, sig, cache_key):
    """真正干活的部分（调用方必须已持有该后端的锁）"""
    torch = libs()["torch"]

    # ── ① 先试磁盘 ────────────────────────────────────────
    path = os.path.join(FEAT_CACHE_DIR, f"{key}_{sig}.pt")
    if os.path.isfile(path):
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
            got = (d["feat"], d["names"], d["is_neg"])
            print(f"[特征] {key} 从磁盘读取 {len(d['names'])} 类"
                  f"（{os.path.getsize(path) / 1024:.0f} KB，免编码）", flush=True)
            _FEATS[cache_key] = got
            return got
        except Exception as e:
            print(f"[特征] 磁盘缓存不可用（{type(e).__name__}），改为重建", flush=True)

    # ── ② 现编码 ──────────────────────────────────────────
    model, processor, _ = get_model(key)
    cats = _config["categories"]
    print(f"[特征] {key} 正在编码 {len(cats)} 类 ...", flush=True)
    t0 = time.time()
    feats, names, is_neg = [], [], []
    for c in cats:
        descs = [d.strip() for d in c.get("descs", []) if d.strip()]
        if not descs:
            descs = [c["name"]]
        prompts = [t.format(d) for d in descs for t in TEMPLATES]
        f = encode_text(model, processor, prompts).mean(dim=0)
        feats.append(f / f.norm(dim=-1, keepdim=True))
        names.append(c["name"])
        is_neg.append(bool(c.get("negative")))
    TF = torch.stack(feats)
    print(f"[特征] {key} 编码完成 {len(names)} 类（{time.time() - t0:.1f}s）",
          flush=True)

    # ── ③ 写盘 ────────────────────────────────────────────
    try:
        os.makedirs(FEAT_CACHE_DIR, exist_ok=True)
        torch.save({"feat": TF, "names": names, "is_neg": is_neg,
                    "sig": sig, "backend": key, "created": time.time()}, path)
        print(f"[特征] 已缓存到 {os.path.basename(path)}"
              f"（{os.path.getsize(path) / 1024:.0f} KB）", flush=True)
    except Exception as e:
        print(f"[特征] 写缓存失败（不影响识别）：{e}", flush=True)

    for k in [k for k in _FEATS if k[0] == key and k[1] != sig]:
        _FEATS.pop(k, None)
    _FEATS[cache_key] = (TF, names, is_neg)
    return _FEATS[cache_key]


def scores(sim, model, cfg):
    """Chinese-CLIP 用标准 softmax（logit_scale=100，无 logit_bias）"""
    scale = model.logit_scale.exp()
    return (scale * sim).softmax(dim=-1)


_infer_lock = threading.Lock()
_IMG_CACHE_LOCK = threading.Lock()


def infer_one(pil_img, key, digest=None):
    """对单张图做推理，返回结构化结果

    digest：图片内容哈希（可选）。给了就启用图像特征缓存，
            同一张图第二次起不必再跑 ViT。
    """
    model, processor, cfg = get_model(key)
    TF, names, is_neg = get_features(key)
    th = float(_config["thresholds"].get(key, cfg["th"]))
    margin_th = float(_config.get("margin", 0.30))

    with _infer_lock:
        t = time.time()
        hit = bool(digest) and os.path.isfile(img_cache_path(key, digest))
        f = encode_image_cached(model, processor, pil_img, key, digest)
        with libs()["torch"].no_grad():
            pr = scores(f @ TF.T, model, cfg)[0]
        ms = (time.time() - t) * 1000

    item_idx = [i for i in range(len(names)) if not is_neg[i]]
    neg_idx = [i for i in range(len(names)) if is_neg[i]]
    if not item_idx:
        return {"verdict": "★无类别", "reason": "nocat", "latency_ms": round(ms, 1),
                "topk": [], "neg_score": 0.0, "threshold": th,
                "category": None, "confidence": 0.0, "margin": 0.0,
                "cached": hit}

    ip = pr[item_idx]
    order = ip.argsort(descending=True)
    best_local = order[0].item()
    best_p = ip[best_local].item()
    best_name = names[item_idx[best_local]]
    second_p = ip[order[1]].item() if len(order) > 1 else 0.0
    # ★ margin 用【相对】差值：(top1-top2)/top1
    margin_rel = (best_p - second_p) / best_p if best_p > 1e-12 else 0.0
    neg_p = max([pr[i].item() for i in neg_idx], default=0.0)

    reason = ""
    if best_p < th:
        reason = "threshold"
    elif neg_p > best_p:
        reason = "negative"
    elif margin_rel < margin_th:
        reason = "margin"

    topk = [(names[item_idx[o.item()]], round(ip[o].item(), 6)) for o in order[:3]]
    return {
        "verdict": "★未知/待人工确认" if reason else best_name,
        "category": None if reason else best_name,
        "confidence": round(best_p, 6),
        "margin": round(margin_rel, 4),
        "neg_score": round(neg_p, 6),
        "threshold": th,
        "topk": topk,
        "reason": reason,
        "latency_ms": round(ms, 1),
        "cached": hit,
    }


# ══════════════════════════════════════════════════════════════
# HTTP 服务
# ══════════════════════════════════════════════════════════════
class Server(ThreadingHTTPServer):
    """覆写 handle_error：客户端断连是常态，不要打大坨 traceback"""
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        import sys
        et, ev, _ = sys.exc_info()
        if et in (ConnectionAbortedError, ConnectionResetError,
                  BrokenPipeError, OSError):
            print(f"[HTTP] 客户端 {client_address[0]} 断开连接 —— 已忽略",
                  flush=True)
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "ItemDemo/2.0-cn"

    def log_message(self, fmt, *args):
        if "/api/" in self.path and self.command == "POST":
            print(f"[HTTP] {self.command} {self.path}", flush=True)

    # ── 工具 ──────────────────────────────────────────────
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError,
                BrokenPipeError, OSError) as e:
            # ★ 浏览器提前断开是常态，不能让它把异常处理也弄挂
            print(f"[HTTP] 客户端已断开（{type(e).__name__}）—— 响应丢弃，服务继续",
                  flush=True)
            self.close_connection = True

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    # ── 路由 ──────────────────────────────────────────────
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(200, HTML_PAGE, "text/html; charset=utf-8")
        elif p == "/api/config":
            self._send(200, {
                "config": _config,
                "backends": {k: {"label": v["label"], "mode": v["mode"],
                                 "th": float(_config["thresholds"].get(k, v["th"])),
                                 "note": v["note"]} for k, v in BACKENDS.items()},
                "loaded": sorted(_MODELS.keys()),
                "threads_effective": resolve_threads(),
                "cpu_count": os.cpu_count() or 1,
                "model_dir": CNCLIP_LOCAL_DIR if os.path.isdir(CNCLIP_LOCAL_DIR)
                             else CNCLIP_REPO,
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        try:
            if p == "/api/config":
                body = self._read_json()
                cfg = body.get("config")
                if not isinstance(cfg, dict):
                    return self._send(400, {"error": "config 格式错误"})
                save_config(cfg)
                return self._send(200, {"ok": True, "config": _config,
                                        "threads_effective": resolve_threads()})

            if p == "/api/reset":
                import copy
                save_config(copy.deepcopy(DEFAULT_CONFIG))
                return self._send(200, {"ok": True, "config": _config,
                                        "threads_effective": resolve_threads()})

            if p == "/api/preload":
                body = self._read_json()
                for k in body.get("backends", []) or list(BACKENDS):
                    if k in BACKENDS:
                        get_model(k)
                        get_features(k)
                return self._send(200, {"ok": True, "loaded": sorted(_MODELS.keys())})

            if p == "/api/predict":
                return self._predict()

            self._send(404, {"error": "not found"})
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError,
                OSError) as e:
            print(f"[HTTP] 处理中客户端断开（{type(e).__name__}）—— 已忽略", flush=True)
            self.close_connection = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})
            except Exception:
                self.close_connection = True

    def _predict(self):
        body = self._read_json()
        # 只有一个后端了；兼容旧前端传来的 backend 参数但忽略之
        keys = list(BACKENDS)
        images = body.get("images", [])

        out = []
        for item in images:
            rec = {"name": item.get("name", ""), "results": {}}
            try:
                Image = libs()["Image"]
                raw = base64.b64decode(item["data"].split(",")[-1])
                rec["digest"] = hashlib.sha256(raw).hexdigest()[:16]
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                rec["w"], rec["h"] = img.size
            except Exception as e:
                rec["error"] = f"图片解析失败: {e}"
                out.append(rec)
                continue
            for k in keys:
                try:
                    rec["results"][k] = infer_one(img, k, rec.get("digest"))
                except Exception as e:
                    rec["results"][k] = {
                        "verdict": "★错误", "category": None, "confidence": 0.0,
                        "margin": 0.0, "neg_score": 0.0, "threshold": 0.0,
                        "topk": [], "reason": f"{type(e).__name__}: {e}",
                        "latency_ms": 0.0, "cached": False,
                    }
            out.append(rec)
        self._send(200, {"ok": True, "backends": keys, "items": out})


# ══════════════════════════════════════════════════════════════
# 前端页面
# ══════════════════════════════════════════════════════════════
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>随身物品识别台</title>
<style>
/* ══════════════════════════════════════════════════════════════
   YeZh Design System —— 浅色 / 深色双模式
   ══════════════════════════════════════════════════════════════ */
*,*::before,*::after{box-sizing:border-box}
html,body{margin:0;padding:0}

:root{
  --bg-primary:#FAF9F6; --bg-secondary:#F5F0EB; --bg-tertiary:#EDE8E3;
  --card:#ffffff; --card-hover:#ffffff;
  --border:#E7E5E4; --border-hover:#D6D3D1;
  --tp:#1C1917;            /* 主文本 */
  --ts:#57534E;            /* 次文本 */
  --tt:#78716C;            /* 三级 */
  --tq:#A8A29E;            /* 弱化 */
  --s1:rgba(0,0,0,.02); --s2:rgba(0,0,0,.05); --s3:rgba(0,0,0,.08);
  --bar:rgba(250,249,246,.72);
  --blue:#3B82F6; --cyan:#06B6D4; --green:#16A34A; --warning:#EA580C;
  --danger:#DC2626; --purple:#7C3AED;
  --sh-card:0 1px 3px rgba(0,0,0,.04);
  --sh-hover:0 2px 8px rgba(0,0,0,.06);
  --sh-lg:0 12px 32px rgba(0,0,0,.12);
  --sh-drawer:-12px 0 40px rgba(0,0,0,.10);
  --sans:'Noto Sans SC','Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:'JetBrains Mono',ui-monospace,'Cascadia Code',Consolas,monospace;
}
.dark{
  --bg-primary:#141211; --bg-secondary:#1C1917; --bg-tertiary:#26221F;
  --card:#1C1917; --card-hover:#221E1C;
  --border:#3D3835; --border-hover:#57534E;
  --tp:#F5F5F4; --ts:#D6D3D1; --tt:#A8A29E; --tq:#78716C;
  --s1:rgba(255,255,255,.04); --s2:rgba(255,255,255,.06); --s3:rgba(255,255,255,.10);
  --bar:rgba(20,18,17,.85);
  --blue:#60A5FA; --cyan:#22D3EE; --green:#4ADE80; --warning:#FB923C;
  --danger:#F87171; --purple:#A78BFA;
  --sh-card:0 1px 3px rgba(0,0,0,.30);
  --sh-hover:0 2px 8px rgba(0,0,0,.40);
  --sh-lg:0 12px 32px rgba(0,0,0,.55);
  --sh-drawer:-12px 0 40px rgba(0,0,0,.45);
}

body{
  background:var(--bg-primary); color:var(--tp);
  font-family:var(--sans); font-size:14px; line-height:1.5;
  -webkit-font-smoothing:antialiased;
  transition:background-color .3s ease,color .3s ease;
}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-.01em}
button{font-family:inherit;font-size:inherit;cursor:pointer;border:0;background:none;color:inherit}
input,textarea,select{font-family:inherit;font-size:inherit;color:inherit}

::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:rgba(120,113,108,.25);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:rgba(120,113,108,.4)}
.dark ::-webkit-scrollbar-thumb{background:rgba(168,162,158,.25)}
.dark ::-webkit-scrollbar-thumb:hover{background:rgba(168,162,158,.4)}

/* ── 顶栏 ─────────────────────────────────────────────────── */
.topbar{
  position:sticky; top:0; z-index:30; height:64px;
  display:flex; align-items:center; gap:12px; padding:0 24px;
  background:var(--bar); backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid var(--border);
}
.brand{display:flex;align-items:center;gap:12px;min-width:0}
.brand-logo{
  width:38px;height:38px;border-radius:12px;flex:0 0 auto;
  display:grid;place-items:center;
  background:linear-gradient(135deg,rgba(59,130,246,.16),rgba(6,182,212,.08));
  border:1px solid rgba(59,130,246,.20); color:var(--blue);
}
.brand-logo svg{width:20px;height:20px}
.brand-title{font-size:15px;font-weight:600;line-height:1.25}
.brand-sub{font-size:11px;color:var(--tt);line-height:1.3}
.spacer{flex:1}

/* 分段控件（主题 / 后端） */
.seg{
  display:flex;gap:2px;padding:3px;border-radius:12px;
  background:var(--s2); border:1px solid transparent;
}
.dark .seg{border-color:rgba(255,255,255,.06)}
.seg button{
  display:flex;align-items:center;justify-content:center;gap:5px;
  padding:6px 10px;border-radius:9px;font-size:12px;font-weight:500;
  color:var(--tt); transition:background .2s,color .2s;
}
.seg button svg{width:15px;height:15px}
.seg button:hover{color:var(--tp)}
.seg button.on{background:var(--card);color:var(--blue);box-shadow:var(--sh-card)}
.dark .seg button.on{background:rgba(255,255,255,.10)}

.btn{
  display:inline-flex;align-items:center;gap:6px;
  padding:8px 14px;border-radius:10px;font-size:13px;font-weight:500;
  border:1px solid var(--border);color:var(--ts);
  transition:background .2s,border-color .2s,color .2s,opacity .2s;
}
.btn svg{width:16px;height:16px}
.btn:hover{background:var(--s1);border-color:var(--border-hover);color:var(--tp)}
.btn:focus-visible{outline:none;box-shadow:0 0 0 3px rgba(59,130,246,.30)}
.btn-primary{background:var(--blue);border-color:transparent;color:#fff;
  box-shadow:0 1px 2px rgba(59,130,246,.25)}
.btn-primary:hover{background:#2563EB;color:#fff;border-color:transparent}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-sm{padding:6px 10px;font-size:12px;border-radius:8px}

/* ── 内容区 ───────────────────────────────────────────────── */
.wrap{max-width:1240px;margin:0 auto;padding:24px;display:flex;flex-direction:column;gap:20px}
.sec-title{font-size:13px;font-weight:600;color:var(--tp)}
.sec-desc{font-size:12px;color:var(--tt);margin-top:2px}

/* ── 卡片 ─────────────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  box-shadow:var(--sh-card);transition:border-color .2s,box-shadow .2s;
}
.card:hover{border-color:var(--border-hover);box-shadow:var(--sh-hover)}
.pad{padding:20px}

/* ── KPI ──────────────────────────────────────────────────── */
.kpi-grp{display:flex;flex-direction:column;gap:10px}
.kpi-grp-head{display:flex;align-items:center;gap:8px;font-size:12px;font-weight:600;color:var(--tt)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{
  border-radius:12px;border:1px solid;padding:16px;
  display:flex;flex-direction:column;gap:2px;
}
.kpi-top{display:flex;align-items:center;justify-content:space-between;gap:8px}
.kpi-label{font-size:12px;font-weight:500;color:var(--ts)}
.kpi-ico{width:32px;height:32px;border-radius:10px;display:grid;place-items:center}
.kpi-ico svg{width:16px;height:16px}
.kpi-val{font-size:26px;font-weight:700;font-family:'Inter',var(--sans);
  letter-spacing:-.02em;line-height:1.15;font-variant-numeric:tabular-nums}
.kpi-sub{font-size:11px;color:var(--tt)}
.kpi-blue{background:linear-gradient(135deg,rgba(59,130,246,.20),rgba(37,99,235,.05));border-color:rgba(59,130,246,.20)}
.dark .kpi-blue{background:linear-gradient(135deg,rgba(59,130,246,.15),rgba(37,99,235,.03));border-color:rgba(59,130,246,.15)}
.kpi-blue .kpi-val{color:#2563EB} .dark .kpi-blue .kpi-val{color:#93C5FD}
.kpi-blue .kpi-ico{background:rgba(59,130,246,.12);color:#3B82F6}
.kpi-green{background:linear-gradient(135deg,rgba(22,163,74,.20),rgba(21,128,61,.05));border-color:rgba(22,163,74,.20)}
.dark .kpi-green{background:linear-gradient(135deg,rgba(34,197,94,.15),rgba(21,128,61,.03));border-color:rgba(34,197,94,.15)}
.kpi-green .kpi-val{color:#15803D} .dark .kpi-green .kpi-val{color:#86EFAC}
.kpi-green .kpi-ico{background:rgba(22,163,74,.12);color:#16A34A}
.kpi-warning{background:linear-gradient(135deg,rgba(234,88,12,.20),rgba(194,65,12,.05));border-color:rgba(234,88,12,.20)}
.dark .kpi-warning{background:linear-gradient(135deg,rgba(251,146,60,.15),rgba(194,65,12,.03));border-color:rgba(251,146,60,.15)}
.kpi-warning .kpi-val{color:#C2410C} .dark .kpi-warning .kpi-val{color:#FDBA74}
.kpi-warning .kpi-ico{background:rgba(234,88,12,.12);color:#EA580C}
.kpi-danger{background:linear-gradient(135deg,rgba(220,38,38,.20),rgba(185,28,28,.05));border-color:rgba(220,38,38,.20)}
.dark .kpi-danger{background:linear-gradient(135deg,rgba(248,113,113,.15),rgba(185,28,28,.03));border-color:rgba(248,113,113,.15)}
.kpi-danger .kpi-val{color:#B91C1C} .dark .kpi-danger .kpi-val{color:#FCA5A5}
.kpi-danger .kpi-ico{background:rgba(220,38,38,.12);color:#DC2626}
.kpi-slate{background:linear-gradient(135deg,var(--s2),var(--s1));border-color:var(--border)}
.kpi-slate .kpi-val{color:var(--tq)}
.kpi-slate .kpi-ico{background:var(--s2);color:var(--tt)}

/* ── 徽章 ─────────────────────────────────────────────────── */
.badge{
  display:inline-flex;align-items:center;gap:4px;
  padding:3px 8px;border-radius:9999px;font-size:11px;font-weight:500;
  white-space:nowrap;
}
.b-blue{background:#EFF6FF;color:#2563EB}   .dark .b-blue{background:rgba(59,130,246,.15);color:#93C5FD}
.b-green{background:#F0FDF4;color:#15803D}  .dark .b-green{background:rgba(34,197,94,.15);color:#86EFAC}
.b-amber{background:#FFFBEB;color:#B45309}  .dark .b-amber{background:rgba(245,158,11,.15);color:#FCD34D}
.b-rose{background:#FEF2F2;color:#B91C1C}   .dark .b-rose{background:rgba(239,68,68,.15);color:#FCA5A5}
.b-cyan{background:#ECFEFF;color:#0E7490}   .dark .b-cyan{background:rgba(6,182,212,.15);color:#67E8F9}
.b-purple{background:#F5F3FF;color:#6D28D9} .dark .b-purple{background:rgba(139,92,246,.15);color:#C4B5FD}
.b-indigo{background:#EEF2FF;color:#4338CA}.dark .b-indigo{background:rgba(99,102,241,.15);color:#A5B4FC}
.b-teal{background:#F0FDFA;color:#0F766E}   .dark .b-teal{background:rgba(20,184,166,.15);color:#5EEAD4}
.b-orange{background:#FFF7ED;color:#C2410C} .dark .b-orange{background:rgba(249,115,22,.15);color:#FDBA74}
.b-slate{background:#F1F5F9;color:#475569}  .dark .b-slate{background:rgba(148,163,184,.15);color:#CBD5E1}
.b-dim{background:var(--s2);color:var(--tt)}

/* ── 上传区 ───────────────────────────────────────────────── */
.drop{
  border:1.5px dashed var(--border-hover);border-radius:12px;
  padding:30px 20px;text-align:center;cursor:pointer;
  transition:border-color .2s,background .2s;background:var(--s1);
}
.drop:hover{border-color:var(--blue);background:rgba(59,130,246,.05)}
.drop.over{border-color:var(--blue);background:rgba(59,130,246,.10)}
.drop-ico{width:44px;height:44px;margin:0 auto 10px;border-radius:12px;
  display:grid;place-items:center;background:var(--s2);color:var(--tt)}
.drop-ico svg{width:22px;height:22px}
.drop-main{font-size:13px;font-weight:500;color:var(--tp)}
.drop-sub{font-size:11.5px;color:var(--tt);margin-top:3px}

.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.chip{
  display:flex;align-items:center;gap:6px;padding:4px 8px;border-radius:8px;
  background:var(--s2);font-size:11.5px;color:var(--ts);max-width:230px;
}
.chip img{width:20px;height:20px;object-fit:cover;border-radius:5px;flex:0 0 auto}
.chip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chip b{font-weight:600}

.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px}
.field{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--tt)}
input[type=number],input[type=text],textarea,select{
  background:var(--card);border:1px solid var(--border);border-radius:9px;
  padding:7px 10px;transition:border-color .2s,box-shadow .2s;outline:none;
}
input[type=number]:focus,input[type=text]:focus,textarea:focus,select:focus{
  border-color:var(--blue);box-shadow:0 0 0 3px rgba(59,130,246,.20);
}
input[type=number]{width:96px;font-family:var(--mono);font-size:12px}
textarea{width:100%;resize:vertical;font-size:12px;line-height:1.6;
  font-family:var(--mono);min-height:64px}

/* ── 进度条 ───────────────────────────────────────────────── */
.bar{height:5px;border-radius:9999px;background:var(--s2);overflow:hidden;margin-top:12px}
.bar>i{display:block;height:100%;width:0;border-radius:9999px;
  background:linear-gradient(90deg,var(--blue),var(--cyan));transition:width .3s ease}
.bar.run>i{
  background-image:linear-gradient(90deg,var(--blue),var(--cyan),var(--blue));
  background-size:200% 100%;animation:shimmer 1.5s linear infinite;
}
@keyframes shimmer{from{background-position:200% 0}to{background-position:0 0}}
.prog-txt{font-size:11.5px;color:var(--tt);margin-top:6px;font-family:var(--mono)}

/* ── 表格 ─────────────────────────────────────────────────── */
.tbl-wrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{
  text-align:left;font-size:11px;font-weight:600;color:var(--tt);
  padding:9px 12px;border-bottom:1px solid var(--border);
  white-space:nowrap;background:var(--s1);
  position:sticky;top:0;z-index:1;
}
td{padding:8px 12px;border-bottom:1px solid var(--border);vertical-align:middle}
tbody tr{transition:background .15s}
tbody tr:hover{background:var(--s1)}
tbody tr:last-child td{border-bottom:0}
.thumb{width:40px;height:40px;object-fit:cover;border-radius:8px;
  border:1px solid var(--border);background:var(--s2);display:block}
.fname{max-width:280px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:12px;color:var(--tp)}
.mono{font-family:var(--mono);font-size:11.5px;font-variant-numeric:tabular-nums}
.top3{font-size:10.5px;color:var(--tq);line-height:1.55;font-family:var(--mono);
  min-width:180px;white-space:normal}
.rowbad{background:rgba(220,38,38,.045)}
.dark .rowbad{background:rgba(248,113,113,.06)}
.rowwarn{background:rgba(234,88,12,.045)}
.dark .rowwarn{background:rgba(251,146,60,.06)}

/* ── 抽屉 ─────────────────────────────────────────────────── */
.mask{
  position:fixed;inset:0;background:rgba(28,25,23,.40);z-index:60;
  opacity:0;pointer-events:none;transition:opacity .25s;
  backdrop-filter:blur(2px);
}
.mask.on{opacity:1;pointer-events:auto}
.drawer{
  position:fixed;top:0;right:0;bottom:0;width:min(480px,94vw);z-index:61;
  background:var(--bg-primary);border-left:1px solid var(--border);
  box-shadow:var(--sh-drawer);
  display:flex;flex-direction:column;
  transform:translateX(100%);transition:transform .28s cubic-bezier(.33,1,.68,1);
}
.drawer.on{transform:translateX(0)}
.drawer-head{
  display:flex;align-items:center;gap:10px;padding:18px 20px;
  border-bottom:1px solid var(--border);flex:0 0 auto;
}
.drawer-head h2{font-size:15px}
.drawer-body{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:26px}
.drawer-foot{
  display:flex;gap:8px;padding:14px 20px;border-top:1px solid var(--border);flex:0 0 auto;
  background:var(--card);
}
.drawer-foot .btn-primary{flex:1;justify-content:center}
.dsec{display:flex;flex-direction:column;gap:12px}
.dsec>h3{font-size:12px;font-weight:600;color:var(--tp);
  display:flex;align-items:center;gap:7px;padding-bottom:2px}
.dsec>h3 svg{width:15px;height:15px;color:var(--blue)}
.dsec-note{font-size:11px;color:var(--tt);line-height:1.6}

.cat{
  border:1px solid var(--border);border-radius:11px;padding:12px;
  background:var(--card);display:flex;flex-direction:column;gap:9px;
  transition:border-color .2s;
}
.cat:hover{border-color:var(--border-hover)}
.cat-head{display:flex;align-items:center;gap:8px}
.cat-head input[type=text]{flex:1;font-size:13px;font-weight:500;padding:6px 9px}
.cat .del{opacity:.4;transition:opacity .2s,color .2s;padding:5px;border-radius:8px}
.cat .del:hover{opacity:1;color:var(--danger);background:var(--s2)}
.cat .del svg{width:15px;height:15px;display:block}
.cat-lbl{font-size:10.5px;color:var(--tq);font-weight:500}
.negbox{display:flex;align-items:center;gap:6px;font-size:11.5px;
  color:var(--ts);cursor:pointer;user-select:none;white-space:nowrap}
.negbox input{width:14px;height:14px;accent-color:var(--blue);cursor:pointer}

/* ── 动画 ─────────────────────────────────────────────────── */
@keyframes float-up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.f0{animation:float-up .4s ease both}
.f1{animation:float-up .4s ease .06s both}
.f2{animation:float-up .4s ease .12s both}
.f3{animation:float-up .4s ease .18s both}
.f4{animation:float-up .4s ease .24s both}
/* 动态显示区块的入场动画：先移除 hidden 再加此类，保证动画一定能执行。
   注意：初始 hidden 的区块【不能】预先挂 f3/f4，否则 animation-fill-mode:both
   会在动画未执行时把元素永久保持在 opacity:0（无头/禁用动画环境会真丢内容）。 */
.reveal{animation:float-up .4s ease both}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{animation:spin .8s linear infinite}
@keyframes ping{75%,100%{transform:scale(2);opacity:0}}
.dot{width:7px;height:7px;border-radius:50%;flex:0 0 auto;position:relative}
.dot.on::after{content:'';position:absolute;inset:0;border-radius:50%;
  background:inherit;animation:ping 1.4s cubic-bezier(0,0,.2,1) infinite}

.empty{padding:44px 20px;text-align:center;color:var(--tq);font-size:12.5px}
.empty svg{width:34px;height:34px;margin-bottom:10px;opacity:.5}
.hidden{display:none!important}

@media (prefers-reduced-motion:reduce){
  .f0,.f1,.f2,.f3,.f4,.reveal,.spin,.dot.on::after,.bar.run>i{animation:none!important;opacity:1!important}
  *{transition-duration:.01ms!important}
}
@media (max-width:640px){
  .topbar{padding:0 14px;gap:8px}
  .brand-sub{display:none}
  .wrap{padding:14px}
}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <div class="brand-logo" data-i="scan"></div>
    <div>
      <div class="brand-title">随身物品识别台</div>
      <div class="brand-sub">零样本分类 · Chinese-CLIP</div>
    </div>
  </div>
  <div class="spacer"></div>
  <div class="seg" id="themeSeg">
    <button data-theme="light" title="浅色"><span data-i="sun"></span></button>
    <button data-theme="dark"  title="深色"><span data-i="moon"></span></button>
    <button data-theme="system" title="跟随系统"><span data-i="monitor"></span></button>
  </div>
  <button class="btn" id="btnSettings"><span data-i="settings"></span>配置</button>
</header>

<main class="wrap">

  <input type="file" id="pickFiles" accept="image/*" multiple hidden>
  <input type="file" id="pickDir" webkitdirectory directory multiple hidden>

  <!-- KPI -->
  <section class="f0" id="kpiSec"></section>

  <!-- 状态条 -->
  <div class="f1 card" id="statusbar" style="padding:12px 16px;
       display:flex;align-items:center;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--tt)"></div>

  <!-- 上传 -->
  <section class="f2 card pad">
    <div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:14px">
      <div style="flex:1">
        <div class="sec-title">上传与识别</div>
        <div class="sec-desc">选择图片或整个文件夹。文件名前缀即真值，用于自动计算准确率。</div>
      </div>
    </div>

    <div class="drop" id="drop">
      <div class="drop-ico" data-i="upload"></div>
      <div class="drop-main">拖拽图片或文件夹到此处</div>
      <div class="drop-sub">或点击选择 · 支持 jpg / png / jpeg / webp / bmp</div>
    </div>

    <div class="toolbar">
      <button class="btn btn-sm" id="btnPickFiles"><span data-i="image"></span>选择图片</button>
      <button class="btn btn-sm" id="btnPickDir"><span data-i="folder"></span>选择文件夹</button>
      <button class="btn btn-sm" id="btnClear"><span data-i="trash"></span>清空</button>
      <div class="field" style="margin-left:auto">
        上传缩放
        <select id="maxSide" style="padding:5px 8px;font-size:12px">
          <option value="512">512</option>
          <option value="768" selected>768</option>
          <option value="1024">1024</option>
          <option value="0">原图</option>
        </select>
        px
      </div>
    </div>

    <div class="chips" id="chips"></div>

    <div style="display:flex;align-items:center;gap:10px;margin-top:16px">
      <button class="btn btn-primary" id="btnRun" disabled>
        <span data-i="play"></span>开始识别
      </button>
      <span style="font-size:12px;color:var(--tt)" id="runHint">尚未选择图片</span>
    </div>

    <div class="bar hidden" id="bar"><i></i></div>
    <div class="prog-txt hidden" id="progTxt"></div>
  </section>

  <!-- 分类明细 -->
  <section class="hidden" id="breakdownWrap">
    <div class="card pad">
      <div class="sec-title" style="margin-bottom:12px">分类明细</div>
      <div class="tbl-wrap" id="breakdown"></div>
    </div>
  </section>

  <!-- 结果 -->
  <section class="hidden" id="resultsWrap">
    <div class="card">
      <div style="display:flex;align-items:center;gap:12px;padding:16px 20px;
                  border-bottom:1px solid var(--border);flex-wrap:wrap">
        <div style="flex:1">
          <div class="sec-title">逐图结果</div>
          <div class="sec-desc" id="resCount"></div>
        </div>
        <div class="seg" id="filterSeg">
          <button data-f="all" class="on">全部</button>
          <button data-f="wrong">仅错误</button>
          <button data-f="unknown">仅未知</button>
        </div>
      </div>
      <div class="tbl-wrap" id="results"></div>
    </div>
  </section>

</main>

<!-- ══ 设置抽屉 ══ -->
<div class="mask" id="mask"></div>
<aside class="drawer" id="drawer">
  <div class="drawer-head">
    <div class="brand-logo" data-i="settings" style="width:32px;height:32px"></div>
    <div style="flex:1">
      <h2>配置</h2>
      <div style="font-size:11px;color:var(--tt)">模型参数与识别分类</div>
    </div>
    <button class="btn btn-sm" id="btnClose"><span data-i="x"></span></button>
  </div>

  <div class="drawer-body">

    <!-- 模型参数 -->
    <div class="dsec">
      <h3><span data-i="cpu"></span>模型与判定</h3>

      <div>
        <div class="cat-lbl" style="margin-bottom:6px">识别后端</div>
        <div class="seg" id="backendSeg" style="width:100%;display:none">
          <button data-b="cnclip" style="flex:1">Chinese-CLIP</button>
        </div>
        <div class="dsec-note" id="backendNote" style="margin-top:7px"></div>
      </div>

      <div>
        <div class="cat-lbl" style="margin-bottom:6px">识判阈值</div>
        <input type="number" id="thClip" step="0.01" min="0" max="1" style="width:100%">
      </div>

      <div style="display:none">
        <input type="number" id="thSiglip" step="0.00001">
      </div>

      <div>
        <div class="cat-lbl" style="margin-bottom:6px">相对间隔 margin</div>
        <input type="number" id="margin" step="0.05" min="0" max="1" style="width:100%">
        <div class="dsec-note" style="margin-top:6px">
          判定为「未知」的三个条件：得分低于阈值 / 负类得分更高 /（top1−top2）÷top1 小于本值。
          用<b>相对值</b>而不是绝对差值：不同模型的分数量级差异很大。
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <button class="btn btn-sm" id="btnPreload"><span data-i="zap"></span>预加载模型</button>
        <label class="negbox" style="gap:5px">
          <input type="checkbox" id="preloadOn">启动时预热
        </label>
        <span style="font-size:11px;color:var(--tt)" id="preloadMsg"></span>
      </div>

      <div>
        <div class="cat-lbl" style="margin-bottom:6px">torch 线程数</div>
        <input type="number" id="threads" step="1" min="0" max="256" style="width:100%">
        <div class="dsec-note" style="margin-top:6px">
          <b>0 = 自动用满逻辑核</b>（推荐）。实测 4→8 线程提速约 1.4x，保存后即时生效、无需重启。
          本机逻辑核：<b id="cpuHint">?</b>。
        </div>
      </div>
    </div>

    <!-- 分类管理 -->
    <div class="dsec">
      <h3><span data-i="tags"></span>识别分类</h3>
      <div class="dsec-note">
        每行一条英文描述，参与生成类别原型。请务必保留一个<b>负类</b>（勾选「负类」），
        它是拒识「未知物品」的主力，缺失会导致误接受。
      </div>
      <div id="cats" style="display:flex;flex-direction:column;gap:10px"></div>
      <button class="btn btn-sm" id="btnAddCat" style="align-self:flex-start">
        <span data-i="plus"></span>添加分类
      </button>
    </div>

  </div>

  <div class="drawer-foot">
    <button class="btn" id="btnReset"><span data-i="rotate"></span>恢复默认</button>
    <button class="btn btn-primary" id="btnSave"><span data-i="check"></span>保存配置</button>
  </div>
</aside>

<script>
/* ══════════════════════════════════════════════════════════════
   图标（lucide 风格内联 SVG，零依赖）
   ══════════════════════════════════════════════════════════════ */
const ICONS={
  scan:'<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><line x1="7" x2="17" y1="12" y2="12"/>',
  sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
  moon:'<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
  monitor:'<rect width="20" height="14" x="2" y="3" rx="2"/><line x1="8" x2="16" y1="21" y2="21"/><line x1="12" x2="12" y1="17" y2="21"/>',
  settings:'<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>',
  upload:'<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
  folder:'<path d="m6 14 1.45-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.55 6a2 2 0 0 1-1.94 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.93a2 2 0 0 1 1.66.9l.82 1.2a2 2 0 0 0 1.66.9H18a2 2 0 0 1 2 2v2"/>',
  image:'<rect width="18" height="18" x="3" y="3" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
  trash:'<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>',
  play:'<polygon points="6 3 20 12 6 21 6 3"/>',
  plus:'<path d="M5 12h14"/><path d="M12 5v14"/>',
  x:'<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
  check:'<path d="M20 6 9 17l-5-5"/>',
  rotate:'<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>',
  zap:'<path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/>',
  target:'<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
  shield:'<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
  alert:'<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>',
  cpu:'<rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>',
  tags:'<path d="m15 5 6.3 6.3a2.4 2.4 0 0 1 0 3.4L13.4 22.6a2.4 2.4 0 0 1-3.4 0L3.7 16.3A2.4 2.4 0 0 1 3 14.6V5a2 2 0 0 1 2-2h9.6a2.4 2.4 0 0 1 1.7.7Z"/><circle cx="7.5" cy="7.5" r="1.5"/>',
  inbox:'<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
  loader:'<path d="M21 12a9 9 0 1 1-6.219-8.56"/>'
};
function svg(n,cls){
  return '<svg class="'+(cls||'')+'" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
       + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'+(ICONS[n]||'')+'</svg>';
}
function paintIcons(root){
  (root||document).querySelectorAll('[data-i]').forEach(el=>{
    if(el.dataset.painted) return;
    el.innerHTML=svg(el.dataset.i, el.className.includes('spin')?'spin':'');
    el.dataset.painted='1';
  });
}

/* ══════════════════════════════════════════════════════════════
   主题
   ══════════════════════════════════════════════════════════════ */
const TKEY='aiagent-theme';
function applyTheme(mode){
  const sys=window.matchMedia('(prefers-color-scheme: dark)').matches;
  const dark = mode==='dark' || (mode==='system' && sys);
  document.documentElement.classList.toggle('dark', dark);
  document.querySelectorAll('#themeSeg button').forEach(b=>
    b.classList.toggle('on', b.dataset.theme===mode));
}
function initTheme(){
  // URL 参数 ?theme=light|dark|system 优先（方便分享/截图/验证）
  const q=new URLSearchParams(location.search).get('theme');
  if(q && ['light','dark','system'].includes(q)) localStorage.setItem(TKEY,q);
  const mode=localStorage.getItem(TKEY)||'system';
  applyTheme(mode);
  window.matchMedia('(prefers-color-scheme: dark)')
    .addEventListener('change',()=>{ if((localStorage.getItem(TKEY)||'system')==='system') applyTheme('system'); });
  document.getElementById('themeSeg').addEventListener('click',e=>{
    const b=e.target.closest('button[data-theme]'); if(!b) return;
    localStorage.setItem(TKEY,b.dataset.theme); applyTheme(b.dataset.theme);
  });
}

/* ══════════════════════════════════════════════════════════════
   状态
   ══════════════════════════════════════════════════════════════ */
let CFG=null, BACKENDS={}, LOADED=[], BACKEND='cnclip', EFF_THREADS=0;
let PRELOADING=null;   // null=未在预热 / 'cnclip'=正在预热的后端
let FILES=[];        // {name, dataUrl, w, h}
let ITEMS=[];        // 识别结果
let RES_KEYS=[];     // 实际用到的后端
let FILTER='all';
const BATCH=6;
const PALETTE=['b-blue','b-green','b-purple','b-cyan','b-indigo','b-teal','b-orange','b-amber'];
const catColor=n=>PALETTE[Math.max(0,CFG.categories.findIndex(c=>c.name===n))%PALETTE.length];

const $=s=>document.querySelector(s);

/* 首次显示某个区块时播入场动画；并带保险，动画未执行也能看见 */
function reveal(el){
  if(!el) return;
  el.classList.remove('hidden');
  if(el._revealed) return;
  el._revealed=true;
  el.classList.remove('reveal'); void el.offsetWidth; el.classList.add('reveal');
  clearTimeout(el._rvTimer);
  el._rvTimer=setTimeout(()=>{ el.style.animation='none'; el.style.opacity='1'; },700);
}
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const pct=x=>(x*100).toFixed(1)+'%';

async function api(path,body){
  const r=await fetch(path,body===undefined?{}:{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok) throw new Error(d.error||('HTTP '+r.status));
  return d;
}

/* ══════════════════════════════════════════════════════════════
   配置
   ══════════════════════════════════════════════════════════════ */
async function loadConfig(){
  const d=await api('/api/config');
  CFG=d.config; BACKENDS=d.backends; LOADED=d.loaded||[];
  EFF_THREADS=d.threads_effective||0;
  const ch=$('#cpuHint'); if(ch) ch.textContent=d.cpu_count||'?';
  // 本分支只有一个后端（Chinese-CLIP），不再读 localStorage
  BACKEND = Object.keys(BACKENDS)[0] || 'cnclip';
  renderStatusbar(); renderCats(); fillForm();
  renderKpis();                       // 先渲染占位 KPI，避免首屏空白
  if(new URLSearchParams(location.search).has('drawer')) openDrawer();
}

function fillForm(){
  $('#thClip').value   = CFG.thresholds.cnclip ?? 0.70;
  $('#margin').value   = CFG.margin;
  $('#threads').value    = CFG.threads ?? 0;
  $('#preloadOn').checked = CFG.preload !== false;
  document.querySelectorAll('#backendSeg button').forEach(b=>
    b.classList.toggle('on', b.dataset.b===BACKEND));
  const bd=BACKENDS[BACKEND];
  $('#backendNote').innerHTML = bd ? esc(bd.label)+' —— '+esc(bd.note||'') : '';
}

function collectForm(){
  const cats=[...document.querySelectorAll('#cats .cat')].map(el=>({
    name: el.querySelector('.cname').value.trim(),
    negative: el.querySelector('.cneg').checked,
    descs: el.querySelector('.cdesc').value.split('\n').map(s=>s.trim()).filter(Boolean)
  })).filter(c=>c.name);
  return {
    thresholds:{cnclip:parseFloat($('#thClip').value)||0.70},
    margin:parseFloat($('#margin').value),
    threads:parseInt($('#threads').value,10)||0,
    preload:$('#preloadOn').checked,
    categories:cats
  };
}

function renderCats(){
  const box=$('#cats');
  box.innerHTML=CFG.categories.map((c,i)=>`
    <div class="cat" data-i2="${i}">
      <div class="cat-head">
        <input type="text" class="cname" value="${esc(c.name)}" placeholder="分类名（中文，也是文件名真值）">
        <label class="negbox"><input type="checkbox" class="cneg" ${c.negative?'checked':''}>负类</label>
        <button class="del" title="删除">${svg('trash')}</button>
      </div>
      <div>
        <div class="cat-lbl" style="margin-bottom:5px">英文描述（每行一条）</div>
        <textarea class="cdesc" spellcheck="false">${esc((c.descs||[]).join('\n'))}</textarea>
      </div>
    </div>`).join('');

  box.querySelectorAll('.cat .del').forEach(b=>b.addEventListener('click',()=>{
    const i=+b.closest('.cat').dataset.i2;
    if(CFG.categories.length<=1) return;
    CFG.categories.splice(i,1); renderCats();
  }));
}

async function saveConfig(){
  const cfg=collectForm();
  if(!cfg.categories.length) return toast('至少保留一个分类');
  if(!cfg.categories.some(c=>c.negative)) toast('提示：没有负类，未知物品会被误接受');
  const d=await api('/api/config',{config:cfg});
  CFG=d.config; EFF_THREADS=d.threads_effective||EFF_THREADS;
  renderCats(); renderStatusbar();
  toast('配置已保存 · 特征缓存已刷新'+
        (CFG.threads>0?(' · 线程 '+CFG.threads):(' · 线程自动 '+EFF_THREADS)));
}

async function resetConfig(){
  const d=await api('/api/reset',{});
  CFG=d.config; EFF_THREADS=d.threads_effective||EFF_THREADS;
  renderCats(); fillForm(); renderStatusbar();
  toast('已恢复默认配置');
}

async function preload(){
  const bks = [BACKEND];
  $('#preloadMsg').textContent='加载中…（首次约 10s，之后走缓存 <1s）';
  try{
    const d=await api('/api/preload',{backends:bks});
    LOADED=d.loaded; $('#preloadMsg').textContent='已就绪：'+d.loaded.join(', ');
    renderStatusbar();
  }catch(e){ $('#preloadMsg').textContent='失败：'+e.message; }
}

/* ★ 后台自动预热：页面一打开就开始，不阻塞操作。
   等用户选完图，模型早加载好了。 */
async function autoPreload(){
  if(PRELOADING) return;                    // 已在预热
  const bks = [BACKEND];
  if(bks.every(k=>LOADED.includes(k))) return;   // 已全部就绪
  PRELOADING = bks.join(',');
  renderStatusbar();
  try{
    const d=await api('/api/preload',{backends:bks});
    LOADED=d.loaded;
  }catch(e){
    /* 静默：用户点「开始识别」时还会再试一次 */
  }finally{
    PRELOADING=null;
    renderStatusbar();
  }
}

function toast(msg,tone){
  const el=document.createElement('div');
  el.className='badge '+(tone==='bad'?'b-rose':'b-blue');
  el.style.cssText='position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99;'
    +'padding:10px 16px;font-size:12.5px;box-shadow:var(--sh-lg);border-radius:9999px';
  el.textContent=msg; document.body.appendChild(el);
  setTimeout(()=>el.remove(),2600);
}

/* ══════════════════════════════════════════════════════════════
   状态条
   ══════════════════════════════════════════════════════════════ */
function renderStatusbar(){
  const on=LOADED.length>0;
  const cats=CFG.categories, neg=cats.filter(c=>c.negative).length;
  let head;
  if(PRELOADING){
    head=`<span style="display:flex;align-items:center;gap:7px">
      <span class="dot on" style="background:var(--warning)"></span>
      <span style="color:var(--warning);font-weight:600">正在后台预热 ${esc(PRELOADING)} …</span>
      <span style="color:var(--tt)">可先选图，不阻塞</span></span>`;
  }else{
    head=`<span style="display:flex;align-items:center;gap:7px">
      <span class="dot ${on?'on':''}" style="background:${on?'var(--green)':'var(--tq)'}"></span>
      <span style="color:var(--tp);font-weight:500">${on?'模型已就绪':'模型未加载'}</span>
      ${on?'<span class="mono" style="color:var(--tt)">'+LOADED.join(' · ')+'</span>':''}
    </span>`;
  }
  $('#statusbar').innerHTML= head + `
    <span class="badge b-dim">${cats.length-neg} 个物品类</span>
    <span class="badge ${neg?'b-green':'b-rose'}">${neg?'负类 '+neg:'⚠ 无负类'}</span>
    <span class="badge b-dim">阈值 ${CFG.thresholds.cnclip ?? CFG.thresholds.clip}</span>
    <span class="badge b-dim">margin ${CFG.margin}</span>
    <span class="badge b-dim" title="torch 线程数（0=自动用满逻辑核）">线程 ${EFF_THREADS||(CFG.threads>0?CFG.threads:'自动')}</span>`;
}

/* ══════════════════════════════════════════════════════════════
   文件读取 / 缩放
   ══════════════════════════════════════════════════════════════ */
function readImage(file,maxSide){
  return new Promise((res,rej)=>{
    const url=URL.createObjectURL(file), img=new Image();
    img.onload=()=>{
      let w=img.naturalWidth, h=img.naturalHeight;
      const s = maxSide>0 ? Math.min(1, maxSide/Math.max(w,h)) : 1;
      const cw=Math.max(1,Math.round(w*s)), ch=Math.max(1,Math.round(h*s));
      const c=document.createElement('canvas'); c.width=cw; c.height=ch;
      const ctx=c.getContext('2d');
      ctx.fillStyle='#fff'; ctx.fillRect(0,0,cw,ch);   // 透明 PNG → 白底，贴合白托盘
      ctx.drawImage(img,0,0,cw,ch);
      URL.revokeObjectURL(url);
      res({dataUrl:c.toDataURL('image/jpeg',0.92), w:cw, h:ch});
    };
    img.onerror=()=>{ URL.revokeObjectURL(url); rej(new Error('无法读取 '+file.name)); };
    img.src=url;
  });
}

async function addFiles(entries){
  const maxSide=+$('#maxSide').value;
  const ok=/\.(jpe?g|png|webp|bmp)$/i;
  let added=0, skip=0;
  for(const {file,name} of entries){
    if(!ok.test(name)){ skip++; continue; }
    if(FILES.some(f=>f.name===name)){ skip++; continue; }
    try{
      const {dataUrl,w,h}=await readImage(file,maxSide);
      FILES.push({name,dataUrl,w,h}); added++;
      if(added%8===0){ renderChips(); await new Promise(r=>setTimeout(r,0)); }
    }catch(e){ skip++; }
  }
  renderChips();
  $('#btnRun').disabled = FILES.length===0;
  $('#runHint').textContent = FILES.length ? `已选 ${FILES.length} 张，可以开始识别`
                              : '尚未选择图片';
  if(skip) toast(`跳过 ${skip} 个非图片/重复文件`);
}

function renderChips(){
  const box=$('#chips');
  if(!FILES.length){ box.innerHTML=''; return; }
  const show=FILES.slice(0,40);
  box.innerHTML=show.map(f=>`<div class="chip" title="${esc(f.name)} ${f.w}×${f.h}">
      <img src="${f.dataUrl}" alt=""><span>${esc(f.name)}</span></div>`).join('')
    + (FILES.length>show.length?`<div class="chip"><span>… 还有 ${FILES.length-show.length} 张</span></div>`:'');
}

/* 拖拽文件夹递归 */
async function entriesFromDrop(dt){
  const roots=[...dt.items].map(i=>i.webkitGetAsEntry&&i.webkitGetAsEntry()).filter(Boolean);
  if(!roots.length){
    return [...dt.files].map(f=>({file:f,name:f.name}));
  }
  const out=[];
  async function walk(entry){
    if(entry.isFile){
      const f=await new Promise((res,rej)=>entry.file(res,rej));
      out.push({file:f,name:f.name});
    }else if(entry.isDirectory){
      const rd=entry.createReader(); let all=[];
      for(;;){
        const batch=await new Promise((res,rej)=>rd.readEntries(res,rej));
        if(!batch.length) break;
        all=all.concat(batch);
      }
      for(const e of all) await walk(e);
    }
  }
  for(const r of roots) await walk(r);
  return out;
}

/* ══════════════════════════════════════════════════════════════
   真值 / 准确率
   ══════════════════════════════════════════════════════════════ */
function truthOf(name){
  const base=String(name).split(/[\\/]/).pop();
  if(/^n\d/i.test(base)){                       // n01_xxx → 非物品
    const neg=CFG.categories.find(c=>c.negative);
    return neg?neg.name:null;
  }
  // ★ 必须按【名字长度降序】匹配：否则「手机充电器1.jpg」会被「手机」先截胡
  const hit=CFG.categories.filter(c=>!c.negative)
      .slice().sort((a,b)=>b.name.length-a.name.length)
      .find(c=>base.startsWith(c.name));
  return hit?hit.name:null;
}
const negName=()=>{const n=CFG.categories.find(c=>c.negative);return n?n.name:null;};

function statsFor(key){
  const NEG=negName();
  const s={itemTotal:0,itemOk:0,itemRej:0,misclass:0,itemWrong:[],
           negTotal:0,negOk:0,falseAccept:0,falseAcc:[],
           perCat:{}};
  CFG.categories.filter(c=>!c.negative).forEach(c=>s.perCat[c.name]={ok:0,total:0,rej:0,bad:0});
  for(const it of ITEMS){
    const r=it.results?.[key]; if(!r||it.error) continue;
    const tr=it.truth, pred=r.category;
    if(tr===NEG){
      s.negTotal++;
      if(pred===null) s.negOk++; else { s.falseAccept++; s.falseAcc.push(it.name); }
    }else if(tr){
      s.itemTotal++;
      const p=s.perCat[tr]||(s.perCat[tr]={ok:0,total:0,rej:0,bad:0});
      p.total++;
      if(pred===null){ s.itemRej++; p.rej++; }
      else if(pred===tr){ s.itemOk++; p.ok++; }
      else { s.misclass++; p.bad++; s.itemWrong.push(it.name); }
    }
  }
  return s;
}

function renderKpis(){
  const box=$('#kpiSec');
  if(!RES_KEYS.length || !ITEMS.length){
    box.innerHTML=`<div class="kpis">
      ${kpi('物品准确率','—','等待识别','blue','target')}
      ${kpi('非物品拒识率','—','等待识别','green','shield')}
      ${kpi('错分','—','认成别的物品','warning','alert')}
      ${kpi('误接受','—','非物品被当物品','danger','zap')}
    </div>`;
    return;
  }
  box.innerHTML='<div style="display:flex;flex-direction:column;gap:16px">'
    + RES_KEYS.map(k=>{
        const s=statsFor(k);
        const acc = s.itemTotal? s.itemOk/s.itemTotal : null;
        const rej = s.negTotal ? s.negOk/s.negTotal : null;
        return `<div class="kpi-grp">
          <div class="kpi-grp-head">
            <span class="badge b-blue">${esc(BACKENDS[k]?.label||k)}</span>
            <span style="color:var(--tt)">共 ${s.itemTotal+s.negTotal} 张计入统计</span>
          </div>
          <div class="kpis">
            ${kpi('物品准确率', acc===null?'—':pct(acc), `${s.itemOk}/${s.itemTotal} 张正确`, 'blue','target')}
            ${kpi('非物品拒识率', rej===null?'—':pct(rej), `${s.negOk}/${s.negTotal} 张拒识`, 'green','shield')}
            ${kpi('错分', s.misclass, '认成别的物品', s.misclass?'warning':'slate','alert')}
            ${kpi('误接受', s.falseAccept, '非物品被当物品', s.falseAccept?'danger':'slate','zap')}
          </div>
        </div>`;
      }).join('') + '</div>';
}

function kpi(label,val,sub,tone,icon){
  return `<div class="kpi kpi-${tone}">
    <div class="kpi-top">
      <span class="kpi-label">${label}</span>
      <span class="kpi-ico">${svg(icon)}</span>
    </div>
    <div class="kpi-val">${val}</div>
    <div class="kpi-sub">${sub}</div>
  </div>`;
}

function renderBreakdown(){
  const wrap=$('#breakdownWrap');
  if(!RES_KEYS.length || !ITEMS.length){ wrap.classList.add('hidden'); return; }
  reveal(wrap);
  const itemCats=CFG.categories.filter(c=>!c.negative);
  const NEG=negName();
  const st={}; RES_KEYS.forEach(k=>st[k]=statsFor(k));
  let html='<table><thead><tr><th>分类</th>';
  RES_KEYS.forEach(k=>{ html+=`<th>${esc(BACKENDS[k]?.label||k)} 正确 / 总数</th><th>拒识</th><th>错分</th>`; });
  html+='</tr></thead><tbody>';
  itemCats.forEach(c=>{
    html+=`<tr><td><span class="badge ${catColor(c.name)}">${esc(c.name)}</span></td>`;
    RES_KEYS.forEach(k=>{
      const p=st[k].perCat[c.name]||{ok:0,total:0,rej:0,bad:0};
      const rate=p.total? p.ok/p.total : 0;
      const tone = !p.total?'b-dim' : rate>=1?'b-green' : rate>=0.8?'b-amber':'b-rose';
      html+=`<td><span class="badge ${tone}">${p.ok}/${p.total}</span></td>
             <td class="mono" style="color:var(--tt)">${p.rej||'—'}</td>
             <td class="mono" style="color:${p.bad?'var(--danger)':'var(--tq)'}">${p.bad||'—'}</td>`;
    });
    html+='</tr>';
  });
  if(NEG){
    html+=`<tr><td><span class="badge b-slate">${esc(NEG)}</span><span style="font-size:10.5px;color:var(--tq);margin-left:6px">期望拒识</span></td>`;
    RES_KEYS.forEach(k=>{
      const s=st[k];
      const tone=s.falseAccept?'b-rose':'b-green';
      html+=`<td><span class="badge ${tone}">拒识 ${s.negOk}/${s.negTotal}</span></td>
             <td class="mono" style="color:var(--tq)">—</td>
             <td class="mono" style="color:${s.falseAccept?'var(--danger)':'var(--tq)'}">${s.falseAccept?'误接受 '+s.falseAccept:'—'}</td>`;
    });
    html+='</tr>';
  }
  html+='</tbody></table>';
  $('#breakdown').innerHTML=html;
}

function renderResults(){
  const wrap=$('#resultsWrap');
  if(!ITEMS.length){ wrap.classList.add('hidden'); return; }
  reveal(wrap);

  const NEG=negName();
  let rows=ITEMS;
  if(FILTER==='wrong')   rows=ITEMS.filter(it=>RES_KEYS.some(k=>{const r=it.results?.[k];
        return r&&it.truth&&it.truth!==NEG&&r.category&&r.category!==it.truth;}));
  if(FILTER==='unknown') rows=ITEMS.filter(it=>RES_KEYS.some(k=>it.results?.[k]?.category===null));

  $('#resCount').textContent=`共 ${ITEMS.length} 张，当前显示 ${rows.length} 张`;

  let html='<table><thead><tr><th></th><th>文件名</th><th>真值</th>';
  RES_KEYS.forEach(k=>html+=`<th colspan="4">${esc(BACKENDS[k]?.label||k)}</th>`);
  html+='</tr><tr><th></th><th></th><th></th>';
  RES_KEYS.forEach(()=>html+='<th>判定</th><th>置信度</th><th>耗时</th><th>Top3</th>');
  html+='</tr></thead><tbody>';

  rows.forEach(it=>{
    let cls='';
    RES_KEYS.forEach(k=>{
      const r=it.results?.[k]; if(!r||!it.truth) return;
      const bad = it.truth===NEG ? r.category!==null : (r.category&&r.category!==it.truth);
      if(bad) cls = it.truth===NEG ? 'rowbad' : (r.category===null?'rowwarn':'rowbad');
    });
    html+=`<tr class="${cls}">
      <td><img class="thumb" src="${it.thumb||''}" alt=""></td>
      <td><div class="fname" title="${esc(it.name)}">${esc(it.name)}</div></td>
      <td>${it.truth?`<span class="badge ${it.truth===NEG?'b-slate':catColor(it.truth)}">${esc(it.truth)}</span>`
                    :'<span class="badge b-dim">未标注</span>'}</td>`;

    RES_KEYS.forEach(k=>{
      const r=it.results?.[k];
      if(!r){ html+='<td colspan="4" style="color:var(--tq)">—</td>'; return; }
      if(r.reason && r.verdict.startsWith('★错误')){
        html+=`<td colspan="4"><span class="badge b-rose">${esc(r.reason)}</span></td>`; return;
      }
      let badge;
      if(r.category===null){
        const why={threshold:'低于阈值',negative:'负类胜出',margin:'间隔过小',nocat:'无物品类'}[r.reason]||r.reason;
        badge=`<span class="badge b-amber" title="${esc(why)}">★未知</span>`;
      }else{
        const hit = it.truth && r.category===it.truth;
        const bad = it.truth && !hit && it.truth!==NEG;
        badge=`<span class="badge ${bad?'b-rose':hit?'b-green':catColor(r.category)}">${esc(r.category)}</span>`;
      }
      const conf = r.confidence>=0.01 ? r.confidence.toFixed(4) : r.confidence.toExponential(2);
      // Top3：三候选用 + 连接，单行展示（窄屏自动换行）
      const tk = (r.topk||[]).map(([n,v])=>`${esc(n)} ${(v*100).toFixed(1)}%`).join(' + ') || '—';
      html+=`<td>${badge}</td>
             <td class="mono" style="color:var(--ts)">${conf}</td>
             <td class="mono" style="color:var(--tq)" title="${r.cached?'命中图像特征缓存，未跑 ViT':'实际跑了 ViT'}">${r.cached?'⚡':''}${r.latency_ms}ms</td>
             <td class="top3">${tk}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table>';
  $('#results').innerHTML=rows.length?html
    :'<div class="empty">'+svg('inbox')+'<div>没有符合条件的图片</div></div>';
}

/* ══════════════════════════════════════════════════════════════
   识别
   ══════════════════════════════════════════════════════════════ */
async function run(){
  if(!FILES.length) return;
  const bks = [BACKEND];
  RES_KEYS=bks;
  ITEMS=[];
  const btn=$('#btnRun'), bar=$('#bar'), ptxt=$('#progTxt');
  btn.disabled=true;
  btn.querySelector('svg')?.classList.add('spin');
  bar.classList.remove('hidden'); bar.classList.add('run');
  ptxt.classList.remove('hidden');

  const t0=performance.now();
  let firstDone=false;          // 首批可能要建类别原型，耗时远高于后续
  for(let i=0;i<FILES.length;i+=BATCH){
    const chunk=FILES.slice(i,i+BATCH);
    ptxt.textContent = firstDone
      ? `识别中 ${Math.min(i+BATCH,FILES.length)}/${FILES.length} …`
      : `⏳ 首批识别中（正在准备类别原型，42 类约 5 秒，仅此一次）…`;
    bar.firstElementChild.style.width=((i+FILES.length*0)/FILES.length*100)+'%';
    let d;
    try{
      d=await api('/api/predict',{backend:BACKEND,
        images:chunk.map(f=>({name:f.name,data:f.dataUrl}))});
    }catch(e){
      toast('识别失败：'+e.message,'bad');
      ptxt.textContent='识别失败：'+e.message;
      bar.classList.remove('run'); btn.disabled=false;
      btn.querySelector('svg')?.classList.remove('spin');
      return;
    }
    d.items.forEach((it,idx)=>{
      it.thumb=chunk[idx]?.dataUrl; it.truth=truthOf(it.name); ITEMS.push(it);
    });
    const done=Math.min(i+BATCH,FILES.length);
    firstDone=true;
    bar.firstElementChild.style.width=(done/FILES.length*100)+'%';
    ptxt.textContent=`已完成 ${done}/${FILES.length} · 累计 ${((performance.now()-t0)/1000).toFixed(1)}s`;
    renderKpis(); renderBreakdown(); renderResults();
  }

  bar.classList.remove('run');
  bar.firstElementChild.style.width='100%';
  const secs=((performance.now()-t0)/1000).toFixed(1);
  ptxt.textContent=`完成 · ${FILES.length} 张 · 总耗时 ${secs}s`
                  +` · 平均 ${(secs*1000/FILES.length).toFixed(0)} ms/张`;
  btn.disabled=false;
  btn.querySelector('svg')?.classList.remove('spin');
  renderKpis(); renderBreakdown(); renderResults();
}

/* ══════════════════════════════════════════════════════════════
   事件绑定
   ══════════════════════════════════════════════════════════════ */
function openDrawer(){ $('#mask').classList.add('on'); $('#drawer').classList.add('on'); }
function closeDrawer(){ $('#mask').classList.remove('on'); $('#drawer').classList.remove('on'); }

function bind(){
  // 抽屉
  const openD=openDrawer;
  const closeD=closeDrawer;
  $('#btnSettings').addEventListener('click',openD);
  $('#btnClose').addEventListener('click',closeD);
  $('#mask').addEventListener('click',closeD);
  document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeD(); });

  $('#btnSave').addEventListener('click',()=>saveConfig().catch(e=>toast('保存失败：'+e.message,'bad')));
  $('#btnReset').addEventListener('click',()=>resetConfig().catch(e=>toast('重置失败：'+e.message,'bad')));
  $('#btnAddCat').addEventListener('click',()=>{
    CFG.categories.push({name:'新分类',descs:['a new item']}); renderCats();
    const all=document.querySelectorAll('#cats .cat');
    all[all.length-1]?.querySelector('.cname')?.focus();
  });
  $('#btnPreload').addEventListener('click',preload);

  document.getElementById('backendSeg').addEventListener('click',e=>{
    const b=e.target.closest('button[data-b]'); if(!b) return;
    BACKEND=b.dataset.b; fillForm();
    autoPreload();          // ★ 切后端就后台预热新模型
  });

  // 上传
  $('#drop').addEventListener('click',()=>$('#pickFiles').click());
  document.getElementById('btnPickFiles').addEventListener('click',()=>$('#pickFiles').click());
  document.getElementById('btnPickDir').addEventListener('click',()=>$('#pickDir').click());
  document.getElementById('pickFiles').addEventListener('change',e=>{
    const fs=[...e.target.files]; e.target.value='';
    addFiles(fs.map(f=>({file:f,name:f.name})));
  });
  document.getElementById('pickDir').addEventListener('change',e=>{
    const fs=[...e.target.files]; e.target.value='';
    addFiles(fs.map(f=>({file:f,name:f.webkitRelativePath||f.name})));
  });
  $('#btnClear').addEventListener('click',()=>{
    FILES=[];ITEMS=[];RES_KEYS=[];renderChips();renderKpis();renderBreakdown();renderResults();
    ['#breakdownWrap','#resultsWrap'].forEach(s=>{const e=$(s);e._revealed=false;delete e._revealed;
      e.style.animation='';e.style.opacity='';});
    $('#btnRun').disabled=true; $('#runHint').textContent='尚未选择图片';
    $('#bar').classList.add('hidden'); $('#progTxt').classList.add('hidden');
  });

  const dz=$('#drop');
  ['dragenter','dragover'].forEach(ev=>dz.addEventListener(ev,e=>{
    e.preventDefault(); e.stopPropagation(); dz.classList.add('over');}));
  ['dragleave','drop'].forEach(ev=>dz.addEventListener(ev,e=>{
    e.preventDefault(); e.stopPropagation(); dz.classList.remove('over');}));
  dz.addEventListener('drop',async e=>{
    const entries=await entriesFromDrop(e.dataTransfer);
    if(entries.length) addFiles(entries);
  });
  ['dragover','drop'].forEach(ev=>window.addEventListener(ev,e=>e.preventDefault()));

  $('#btnRun').addEventListener('click',run);

  // 结果过滤
  document.getElementById('filterSeg').addEventListener('click',e=>{
    const b=e.target.closest('button[data-f]'); if(!b) return;
    FILTER=b.dataset.f;
    document.querySelectorAll('#filterSeg button').forEach(x=>x.classList.toggle('on',x===b));
    renderResults();
  });
}

/* ══════════════════════════════════════════════════════════════
   启动
   ══════════════════════════════════════════════════════════════ */
(async function(){
  paintIcons();
  initTheme();
  bind();
  try{ await loadConfig(); }
  catch(e){ toast('读取配置失败：'+e.message,'bad'); return; }
  // ★ 页面就绪后立即后台预热，用户选图的同时模型在加载
  autoPreload();
})();
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="零样本物品识别 Demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    load_config()
    apply_threads()          # 在 torch 导入前先把 OMP/MKL 设好
    clean_feat_cache()
    clean_img_cache()
    n_cat = len([c for c in _config["categories"] if not c.get("negative")])
    n_neg = len([c for c in _config["categories"] if c.get("negative")])

    print("=" * 62)
    print("  零样本物品识别台 · Demo Web")
    print("=" * 62)
    print(f"  分类      : {n_cat} 个物品类 + {n_neg} 个负类")
    print(f"  配置文件  : {CONFIG_FILE}")
    print(f"  线程数    : {resolve_threads()}"
          f"（0=自动；OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}）")
    print(f"  图像缓存  : {IMG_CACHE_DIR}")
    print(f"  启动预热  : {'开' if _config.get('preload', True) else '关'}")
    print(f"  HF_HOME   : {os.environ.get('HF_HOME', '(默认)')}")
    print(f"  离线模式  : {os.environ.get('HF_HUB_OFFLINE', '0')}")
    print()
    print(f"  访问地址  : http://{args.host}:{args.port}")
    print("  停止服务  : Ctrl+C")
    print("=" * 62, flush=True)

    try:
        srv = Server((args.host, args.port), Handler)
    except OSError as e:
        print()
        print(f"  [错误] 无法绑定端口 {args.port}: {e}")
        print(f"         换个端口试试:  python web_demo.py --port 8080")
        print()
        sys.exit(1)

    if _config.get("preload", True):
        def _warmup():
            try:
                for k in BACKENDS:
                    get_model(k)
                    get_features(k)
                print("[预热] 模型 + 类别原型就绪", flush=True)
            except Exception as e:
                print(f"[预热] 失败（不影响使用）：{type(e).__name__}: {e}",
                      flush=True)
        threading.Thread(target=_warmup, daemon=True, name="warmup").start()
        print("  [预热] 后台加载模型中 …", flush=True)

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(
            f"http://{args.host}:{args.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")


if __name__ == "__main__":
    main()

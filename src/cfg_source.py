"""
统一配置来源 —— 单一数据源（Single Source of Truth）

设计目标
────────
在 Web Demo 里改完分类 / 阈值并点「保存配置」后，命令行脚本
（01_clip_zeroshot.py / 02_eval_tray.py）自动跟随，不再出现
「网页 5 类、脚本 8 类」这种对不上的情况。

优先级
──────
    1. --config <路径>  显式指定
    2. <项目根>/web_demo_categories.json     ← 唯一数据源
    3. 各脚本内置的默认值（仅作兜底）

用法
────
    import cfg_source

    CFG = cfg_source.load(
        default_classes = CLASSES,                       # {名称: [英文描述]}
        default_th      = {k: v["th"] for k, v in BACKENDS.items()},
    )
    CLASSES   = CFG["classes"]          # {名称: [描述]}
    NEG       = CFG["neg_set"]          # set(负类名)
    NEG_LABEL = CFG["neg_label"]        # 主负类名（配置里第一个）
    MARGIN    = CFG["margin"]           # 相对间隔阈值

命令行
──────
    python cfg_source.py --show         # 查看当前生效的配置
    python cfg_source.py --init         # 用内置默认值生成 web_demo_categories.json
    python cfg_source.py --check        # 只检查来源，不打印内容
"""
import os
import sys
import json
import copy

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_DIR, "web_demo_categories.json")

# 与 web_demo.py 的 DEFAULT_CONFIG 保持一致（仅在没有 json 文件时兜底）
FALLBACK_CONFIG = {
    "thresholds": {"clip": 0.70, "siglip": 0.0005},
    "margin": 0.30,
    "categories": [
        {"name": "手机", "descs": ["a mobile phone", "a smartphone", "a cell phone"]},
        {"name": "现金", "descs": ["paper money", "banknotes", "a stack of cash"]},
        {"name": "手表", "descs": ["a wristwatch", "a watch"]},
        {"name": "钥匙", "descs": ["a key", "a bunch of keys", "a keychain"]},
        {"name": "银行卡", "descs": ["a credit card", "a bank card", "a plastic card"]},
        {"name": "非物品", "negative": True, "descs": [
            "a photo of a table", "a photo of a room", "an empty tray",
            "a photo of a white board", "a screenshot", "a document",
            "a text page", "a form", "a photo of a person", "a face",
            "hands", "a fruit", "a vegetable",
        ]},
    ],
}


def ensure_utf8():
    """让中文输出在任意终端下都不乱码"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
def resolve_path(explicit=None):
    """返回应当读取的配置文件路径；不存在返回 None"""
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    if os.path.isfile(DEFAULT_CONFIG_FILE):
        return DEFAULT_CONFIG_FILE
    return None


def load(default_classes=None, default_th=None, default_margin=0.30,
         path=None, disabled=False, quiet=False, announce=True):
    """
    读取统一配置。

    参数
    ────
    default_classes : {名称: [英文描述]}     兜底分类表
    default_th      : {后端: 阈值}           兜底阈值
    default_margin  : float                  兜底相对间隔
    path            : 显式指定配置文件（来自 --config）
    disabled        : True 表示 --no-config，强制用内置默认
    announce        : 是否打印「配置来源」一行

    返回
    ────
    {
      "classes"   : {名称: [描述]},     # 含负类
      "negatives" : [负类名, ...],      # 按配置顺序
      "neg_set"   : set(负类名),
      "neg_label" : 主负类名 or None,
      "th"        : {后端: 阈值},
      "margin"    : float,
      "source"    : 来源描述字符串,
      "path"      : 实际读取的文件路径 or None,
      "warnings"  : [警告信息],
    }
    """
    ensure_utf8()
    warnings = []

    default_classes = default_classes or {}
    default_th = dict(default_th or {})

    # ── 决定来源 ──────────────────────────────────────────────
    if disabled:
        data, src, used_path = None, "内置默认（--no-config 强制）", None
    else:
        used_path = resolve_path(path)
        if path and not used_path:
            warnings.append(f"--config 指定的文件不存在：{path}，已回退")
        if used_path:
            try:
                with open(used_path, encoding="utf-8") as f:
                    data = json.load(f)
                src = os.path.basename(used_path)
            except Exception as e:
                warnings.append(f"读取 {os.path.basename(used_path)} 失败（{e}），已回退到内置默认")
                data, src, used_path = None, "内置默认", None
        else:
            data, src = None, "内置默认"
    # ── 解析 ──────────────────────────────────────────────────
    if data and isinstance(data.get("categories"), list) and data["categories"]:
        classes, negatives = {}, []
        for c in data["categories"]:
            name = str(c.get("name", "")).strip()
            if not name:
                continue
            descs = [str(d).strip() for d in c.get("descs", []) if str(d).strip()]
            if not descs:
                descs = [name]
                warnings.append(f"分类「{name}」没有英文描述，退化用名称本身")
            classes[name] = descs
            if c.get("negative"):
                negatives.append(name)
        if not classes:
            warnings.append("配置文件里没有有效分类，已回退到内置默认")
            data = None
        elif not negatives:
            warnings.append("配置里没有标记任何一个「负类」——拒识能力会大幅下降！")
    else:
        data = None

    if data is None:
        classes = {k: list(v) for k, v in default_classes.items()}
        # default_classes 里没有负类标记，只能靠命名约定
        negatives = [n for n in classes if n in ("非物品", "背景", "其他", "未知")]

    # ── 阈值 / margin ────────────────────────────────────────
    th = dict(default_th)
    margin = float(default_margin)
    if data:
        for k, v in (data.get("thresholds") or {}).items():
            try:
                th[k] = float(v)
            except (TypeError, ValueError):
                warnings.append(f"阈值 {k}={v!r} 不是数字，已忽略")
        if "margin" in data:
            try:
                margin = float(data["margin"])
            except (TypeError, ValueError):
                warnings.append(f"margin={data['margin']!r} 不是数字，已忽略")

    n_item = len([k for k in classes if k not in negatives])
    if n_item == 0:
        if not classes and not default_classes and not data:
            warnings.append("未找到配置文件，且调用方未传入兜底分类表 → 当前无可用的分类")
        elif not classes:
            warnings.append("兜底分类表为空")
        else:
            warnings.append("没有物品类，只剩负类——无法识别任何物品")

    if announce:
        if disabled:
            print(f"  配置来源: {src}  {n_item} 个物品类 + {len(negatives)} 个负类")
        elif used_path:
            print(f"  配置来源: {src}（来自文件）  {n_item} 个物品类 + {len(negatives)} 个负类")
        else:
            print(f"  配置来源: 内置默认  {n_item} 个物品类 + {len(negatives)} 个负类")
            print(f"            ⚠ 未找到 web_demo_categories.json，使用脚本内置默认值")
            print(f"              生成配置文件：python cfg_source.py --init")
        for w in warnings:
            print(f"            ⚠ {w}")
        print()

    return {
        "classes": classes,
        "negatives": negatives,
        "neg_set": set(negatives),
        "neg_label": negatives[0] if negatives else None,
        "th": th,
        "margin": margin,
        "source": src,
        "path": used_path,
        "warnings": warnings,
    }


# ══════════════════════════════════════════════════════════════
def write_default(path=None, overwrite=False, from_web_demo=True):
    """生成配置文件。

    from_web_demo=True 时直接从 web_demo.py 里提取 DEFAULT_CONFIG，
    保证与 Web Demo 的出厂默认完全一致；否则用本模块的 FALLBACK_CONFIG。
    """
    path = path or DEFAULT_CONFIG_FILE
    if os.path.isfile(path) and not overwrite:
        print(f"  已存在，未覆盖：{path}")
        print(f"  如需覆盖请加 --force")
        return False

    cfg = FALLBACK_CONFIG
    if from_web_demo:
        try:
            src = open(os.path.join(PROJECT_DIR, "web_demo.py"), encoding="utf-8").read()
            i = src.index("DEFAULT_CONFIG = {")
            j = src.index("\n}\n", i) + 3
            ns = {}
            exec(src[i:j].replace("DEFAULT_CONFIG", "CFG"), ns)
            cfg = ns["CFG"]
        except Exception as e:
            print(f"  [提示] 未能从 web_demo.py 提取默认配置（{e}），改用内置兜底")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print(f"  已生成：{path}")
    print(f"  {len([c for c in cfg['categories'] if not c.get('negative')])} 个物品类 + "
          f"{len([c for c in cfg['categories'] if c.get('negative')])} 个负类")
    return True


def show(path=None):
    cfg = load(path=path, announce=False)
    print("=" * 62)
    print("  当前生效配置")
    print("=" * 62)
    print(f"  来源      : {cfg['source']}")
    print(f"  文件      : {cfg['path'] or '（无）'}")
    print(f"  阈值      : {cfg['th']}")
    print(f"  相对间隔  : {cfg['margin']}")
    print()
    for name, descs in cfg["classes"].items():
        tag = "[负类]" if name in cfg["neg_set"] else "      "
        print(f"  {tag} {name}")
        for d in descs:
            print(f"           · {d}")
    if cfg["warnings"]:
        print()
        for w in cfg["warnings"]:
            print(f"  ⚠ {w}")
    print()


def main():
    ensure_utf8()
    import argparse
    ap = argparse.ArgumentParser(description="统一配置来源（单一数据源）")
    ap.add_argument("--show", action="store_true", help="打印当前生效的配置")
    ap.add_argument("--check", action="store_true", help="只显示配置来源")
    ap.add_argument("--init", action="store_true", help="用默认值生成 web_demo_categories.json")
    ap.add_argument("--force", action="store_true", help="--init 时覆盖已存在的文件")
    ap.add_argument("--config", help="显式指定配置文件路径")
    args = ap.parse_args()

    if args.init:
        write_default(args.config, overwrite=args.force)
    elif args.show:
        show(args.config)
    elif args.check:
        cfg = load(path=args.config, announce=False)
        print(f"  来源: {cfg['source']}")
        print(f"  文件: {cfg['path'] or '（无）'}")
        for w in cfg["warnings"]:
            print(f"  ⚠ {w}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

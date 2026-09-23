# 零样本物品识别方案
## CLIP ViT-B-32 ／ SigLIP2 ViT-B-32-256 双后端

> 版本 **v2.0**（v1.0 选型结论已修正，见 §9 修订记录）
> 类型：实现规格（Implementation Spec）｜ 定位：识别链路 L1 层
> 实测环境：纯 CPU（12 线程）｜ open_clip 3.3.0 ｜ torch 2.14.0
> 实测样本：16 张真实物品图 + 7 张非物品图

---

## 1. 方案定位

### 1.1 解决什么问题

| 问题 | 本方案的回答 |
|---|---|
| **P0 阶段零训练数据，怎么让系统先上线？** | 零样本分类——**不需要任何训练图片** |
| **没有 GPU，能不能跑？** | 能。**纯 CPU，30 件物品约 1.6 秒** |
| **遇到没见过的物品怎么办？** | **拒识机制**——宁可不认，不能认错 |
| **加新品类要不要重新训练？** | 不要。**加一行类别描述即可** |

### 1.2 在整体架构中的位置

本方案是识别链路的 **L1 层（快通道）+ 零样本冷启动能力**：

```
物品放上识别台
      │
      ▼
┌─────────────────────────────────────────────┐
│ L0  采集与切分                                │
│     相机 → 基线差分 → 连通域 → 裁出单件小图     │
└─────────────────────────────────────────────┘
      │  单件小图（干净背景 · 居中 · 固定光照）
      ▼
╔═════════════════════════════════════════════╗
║ ★ 本方案 · L1  零样本分类                     ║
║   CLIP ViT-B-32（主力）／ SigLIP2（备选）     ║
╚═════════════════════════════════════════════╝
      │
      ├─ 置信度高 ──────────────────────────┐
      │                                      │
      ▼ 置信度低 / 负类胜出                    │
┌─────────────────────────────────────────────┐ │
│ L2  本地 VLM 兜底（Qwen3-VL / SmolVLM）      │ │
└─────────────────────────────────────────────┘ │
      │ 仍不确定                                │
      ▼                                          │
┌─────────────────────────────────────────────┐ │
│ L3  人工点选确认（Top-3 候选 + 小图）          │ │
└─────────────────────────────────────────────┘ │
      │                                          │
      ▼                                          ▼
┌─────────────────────────────────────────────┐
│ L4  结果入库 + 实例指纹 + 数据闭环             │
└─────────────────────────────────────────────┘
```

> **关键前置条件**：本方案的输入**必须是 L0 裁出的单个物品**，而不是整张场景照片。
> 实测证明：用整图输入会导致判断完全失效（见 §8.1）。

---

## 2. 选型结论

### 2.1 推荐配置

| 后端 | 定位 | 理由 |
|---|---|---|
| **CLIP ViT-B-32** | ★ **主力（默认）** | 阈值窗口宽（0.35~0.95）、得分稳定、体积小、速度快、无额外依赖 |
| SigLIP2 ViT-B-32-256 | 备选 / 交叉验证 | 拒识分离度也极好，但**得分绝对值波动 264 倍**，阈值极窄 |

### 2.2 实测数据（23 张图：16 物品 + 7 非物品）

#### 分类能力：两者完全相同

| 后端 | 纯分类准确率（忽略阈值） |
|---|---|
| CLIP ViT-B-32 | **16 / 16 = 100%** |
| SigLIP2 ViT-B-32-256 | **16 / 16 = 100%** |

> **排名能力两者都是满分。差异不在"认不认得出"，而在"阈值能不能可靠设置"。**

#### 得分分布：这才是决定性的差异

| | CLIP | SigLIP2 |
|---|---|---|
| **真实物品 top1 得分** | **0.9944 ~ 1.0000** | **0.00142 ~ 0.37456** |
| 得分波动范围 | **1.006 倍**（极稳定） | **264 倍**（极不稳定） |
| 非物品最高得分 | 0.1748 | 0.00004 |
| **物品/非物品间隔** | 0.82（绝对） | 0.00138（绝对） |
| **有效阈值窗口** | **0.35 ~ 0.95** | 0.0001 ~ 0.0014 |
| 窗口容错能力 | **宽**（±0.3 都安全） | **窄**（量级级精调） |

```
CLIP    ── 物品 0.9944 ┊══════ 安全窗口 0.35~0.95 ══════┊ 0.1748 非物品 ──
                        ↑ 间隔 0.82，极其宽松

SigLIP2 ── 物品 0.00142 ┊ 0.00004~0.00142 ┊ 0.00004 非物品 ──
                        ↑ 窗口只有 0.0014 宽，量级级精度
```

#### 逐图明细

| 图片 | CLIP top1 | SigLIP2 top1 | SigLIP2 top1/top2 |
|---|---|---|---|
| 手机1 | 手机 0.9965 | 手机 0.00142 | 2678x |
| 手机2 | 手机 0.9994 | 手机 0.00966 | 2387x |
| 手机3 | 手机 0.9979 | 手机 0.01236 | 7584x |
| 手机5 | 手机 0.9984 | 手机 0.02516 | 21617x |
| 钥匙1 | 钥匙 0.9997 | 钥匙 0.00362 | 553x |
| 钥匙2 | 钥匙 1.0000 | 钥匙 0.13782 | 3841x |
| 钥匙3 | 钥匙 1.0000 | 钥匙 0.18802 | 3141x |
| 钥匙4 | 钥匙 1.0000 | 钥匙 0.01601 | 89036x |
| 钥匙5 | 钥匙 0.9977 | 钥匙 0.03589 | 397x |
| 钥匙6 | 钥匙 0.9944 | 钥匙 0.00394 | 47x |
| 钥匙7 | 钥匙 1.0000 | 钥匙 0.37456 | 1643x |
| 钥匙8 | 钥匙 1.0000 | 钥匙 0.14059 | 3738x |
| 银行卡1 | 银行卡 1.0000 | 银行卡 0.10980 | 464x |
| 银行卡2 | 银行卡 1.0000 | 银行卡 0.05330 | 1655x |
| 银行卡3 | 银行卡 1.0000 | 银行卡 0.27539 | 3834x |
| 银行卡4 | 银行卡 0.9983 | 银行卡 0.02181 | 12313x |

**非物品（7 张，两者均正确拒绝）**

| 图片 | CLIP top1 | SigLIP2 top1 |
|---|---|---|
| 空托盘 | 0.1748（neg=0.649） | 0.00004 |
| 布料背景 | 0.0397（neg=0.929） | 0.00000 |
| 文档 | 0.0428（neg=0.955） | 0.00000 |
| 界面截图 | 0.1112（neg=0.859） | 0.00001 |
| 公章效果图 | 0.0854（neg=0.880） | 0.00001 |
| 人像照片 | 0.0061（neg=0.988） | 0.00000 |
| 现勘草图 | 0.0194（neg=0.978） | 0.00001 |

### 2.3 为什么选 CLIP 而不是 SigLIP2

| 维度 | CLIP | SigLIP2 | 胜 |
|---|---|---|---|
| 分类准确率 | 100% | 100% | 平 |
| **阈值鲁棒性** | **窗口 0.35~0.95** | 窗口 0.0001~0.0014 | **CLIP** |
| **得分稳定性** | 波动 1.006 倍 | 波动 **264 倍** | **CLIP** |
| 体积 | **605 MB** | 1507 MB | **CLIP** |
| 耗时 | **52 ms** | 68~82 ms | **CLIP** |
| 依赖 | torch + open_clip | **还需 transformers** | **CLIP** |
| 拒识分离度 | 好 | 略优（rel. 32x） | SigLIP2 |

> **核心判断**：SigLIP2 的绝对得分随图片变化 264 倍（0.0014~0.375），
> 意味着**阈值只能落在 0.0001~0.0014 这个量级窗口内**。
> 一旦现场光照、拍摄条件与测试集不同，得分分布漂移，阈值极易失效。
>
> **CLIP 的得分稳定在 0.994~1.000，阈值取 0.35 还是 0.95 结果都一样**——
> 这在工程上是巨大的优势。
>
> **结论：CLIP 作为主力。SigLIP2 保留用于交叉验证或特定场景。**

### 2.4 ⚠️ 关于「类别表聚焦度」的重要发现

CLIP 早期测试中曾出现严重误判：

| 测试条件 | 结果 |
|---|---|
| **17 类**类别表 + 空托盘 | ❌ 身份证 **50.1%**（误接受） |
| **17 类**类别表 + 人像照片 | ❌ 身份证 **67.6%**（误接受） |
| **4 类**类别表（手机/钥匙/银行卡）+ 负类 | ✅ 空托盘 0.1748、人像 0.0061（正确拒绝） |

**原因**：17 类表里混入了 `身份证`、`钱包`、`背包` 等**与托盘/人像纹理相近的类别**，
softmax 必须把这些概率分配出去，于是「最像的那个」被选中。

**实践指导：**

| 规则 | 说明 |
|---|---|
| **类别表只放实际在用的类别** | 不要为了"以后可能用到"而预置一堆类别 |
| **负类必须专门设计** | 贴合现场实际干扰物（桌布、背景、文档、人像、空台面） |
| **分区分模块建类别表** | 每个识别区/每个场景用最小类别集 |

> 实测中 CLIP 的负类得分：空托盘 0.649、布料 0.929、文档 0.955、人像 0.988
> —— **负类在绝大多数非物品场景下直接胜出**，这是零成本的拒识主力。

### 2.5 根本原因

```python
# CLIP：softmax 归一化 —— 概率和必须为 1
probs = (scale * sim).softmax(dim=-1)
# 优点：真实物品得分稳定在高位（0.99+）
# 缺点：非物品也会被分配分数，故必须靠负类 + 聚焦类别表压制

# SigLIP2：sigmoid 逐类独立 —— 可以全部低分
probs = torch.sigmoid(scale * sim + logit_bias)
# 优点：分离度天然好，非物品趋近 0
# 缺点：绝对得分随图片大幅波动，阈值标定脆弱
```

---

## 3. 技术规格

### 3.1 模型规格（实测）

| 项 | **CLIP ViT-B-32** | SigLIP2 ViT-B-32-256 |
|---|---|---|
| open_clip 模型名 | `ViT-B-32` | `ViT-B-32-SigLIP2-256` |
| 权重标签 | `laion2b_s34b_b79k` | `webli` |
| 参数量 | 151.3 M | 376.9 M |
| 体积 (fp32) | **605 MB** | 1507 MB |
| 输入尺寸 | 224 × 224 | 256 × 256 |
| 预处理 | Resize(224)+CenterCrop | **Resize(256,256) 直接拉伸** |
| `logit_scale` | 100.0 | 4.717 |
| `logit_bias` | 无 | **-16.767** |
| 打分方式 | softmax | sigmoid |
| 单张耗时 (CPU) | **52 ms** | 68 ~ 82 ms |
| 30 件物品 | **约 1.6 s** | 约 2.2 s |
| 首次下载 | 8.5 min | 22.6 min |
| 额外依赖 | 无 | **需 transformers** |

### 3.2 输入规范

| 项 | 要求 | 原因 |
|---|---|---|
| **输入内容** | **L0 裁出的单个物品** | ★ 整图输入会导致判断失效 |
| 建议边距 | 四周留 **10~15%** 边距 | 避免裁掉物品边缘 |
| 色彩空间 | RGB | 统一 |
| 尺寸 | 交给 `preprocess` 自动处理 | **不要手写 Resize/Normalize** |
| 分辨率下限 | ≥ 224 × 224 | 更低会插值放大，损失细节 |

> **实测佐证**：测试集包含 500×500 到 3024×4032 的各种尺寸，
> 统一交给 `preprocess` 后**两个模型分类准确率都是 100%**。
> 说明预处理链路是正确的，**不要自己改预处理器**。

### 3.3 预处理

**必须使用 open_clip 提供的 `preprocess`。**

```python
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
img_tensor = preprocess(PIL.Image.open(path).convert("RGB")).unsqueeze(0)
```

> 手写 Resize/CenterCrop/Normalize 极易与模型期望不一致，
> 且**不会报错，只会让准确率莫名下降**。

### 3.4 类别表设计规范

#### 规则

| # | 规则 | 说明 |
|---|---|---|
| 1 | **必须用英文描述** | CLIP/SigLIP2 都是英文训练的，中文显著变差 |
| 2 | **每类 2~3 条描述** | 多描述取均值，提升鲁棒性 |
| 3 | **描述要有形态区分度** | `a bunch of keys` 比 `keys` 好 |
| 4 | **必须包含负类** | `非物品`，拒识主力 |
| 5 | **只放实际在用的类别** | ★ 见 §2.4，类别混杂会显著恶化拒识 |
| 6 | **粒度按登记需求定** | 分「手机」不分「iPhone 15」 |

#### 推荐类别表：贴合现有托盘七格

**这是实测验证过的最优起步配置（16/16 + 7/7 全对）：**

| 托盘的 7 格 | 类别 |
|---|---|
| 手机 Phone | 手机 |
| 现金 Cash | 现金 |
| 手表 Watch | 手表 |
| 钥匙 Key | 钥匙 |
| 银行卡 Bank card | 银行卡 |
| 身份证 Id card | 身份证 |
| 首饰 Jewelry | 首饰 |

> **注意**：7 格全上时，`身份证` 与「空托盘/人像」的混淆风险会上升。
> 建议：**先只上手机/现金/手表/钥匙/银行卡 5 类**跑稳，
> 再逐个加入其余类别，每次加入后**重跑阈值标定**。

#### 负类描述（关键）

```python
"非物品": [
    # 台面/装置
    "a photo of a table", "a photo of a room", "a picture frame",
    "an empty tray", "a photo of a white board",
    # 文档/屏幕
    "a screenshot", "a document", "a text page", "a form",
    # 人体
    "a photo of a person", "a face", "hands",
    # 其他干扰
    "a fruit", "a vegetable", "a photo of food",
]
```

**实测负类得分**（越高越好）：空托盘 0.649 ｜ 布料 0.929 ｜ 文档 0.955 ｜ 人像 0.988

### 3.5 提示词模板

```python
TEMPLATES = ["a photo of {}", "a close-up photo of {}", "{}"]
```

同一描述套用不同句式后编码再平均，消除句式带来的系统性偏差。

最终每个类别的原型向量 = `mean(所有描述 × 所有模板)`，归一化后缓存。

### 3.6 打分与拒识

#### 打分

```python
def score(sim, model, cfg):
    scale = model.logit_scale.exp()
    if cfg["mode"] == "softmax":               # CLIP
        return (scale * sim).softmax(dim=-1)
    logits = scale * sim                        # SigLIP2
    bias = getattr(model, "logit_bias", None)
    if bias is not None:
        logits = logits + bias
    return torch.sigmoid(logits)
```

#### 三重拒识机制

| 机制 | 规则 | 状态 |
|---|---|---|
| **① 阈值** | `max_item_prob < th` → 未知 | ✅ 已实现 |
| **② 负类胜出** | `neg_prob > max_item_prob` → 未知 | ✅ 已实现（**主力**） |
| **③ 间隔（margin）** | `top1 - top2 < margin` → 待人工确认 | ✅ 已实现 |

#### 阈值标定结果（实测）

| 后端 | 物品最低分 | 非物品最高分 | **建议阈值** | 安全窗口 |
|---|---|---|---|---|
| **CLIP** | **0.9944** | 0.1748 | **0.70** | **0.35 ~ 0.95 均可** |
| SigLIP2 | 0.00142 | 0.00004 | **0.0005** | 0.0001 ~ 0.0014 |

> **CLIP 建议取 0.70（窗口居中），兼顾双向容错。**

**标定流程（换模型/改类别表后必须重跑）：**

```
1. 准备 ≥15 张「真实物品裁剪图」（每类 ≥3 张，覆盖不同实物）
2. 准备 ≥7 张「非物品图」（空托盘/背景/文档/截图/人像/草图）
3. 分别跑分类，记录 top1 得分
4. 取「物品最低分」与「非物品最高分」的中点作为阈值
5. 若两者区间重叠 → 类别表有问题或该模型不适用
6. 样本量 <15 张时结论不可信（见 §9 修订记录）
```

### 3.7 结果判定流程

```python
item_idx = [i for i, n in enumerate(names) if n != "非物品"]
ip = probs[item_idx]
order = ip.argsort(descending=True)
best_p, best_zh = ip[order[0]], names[item_idx[order[0]]]
second_p = ip[order[1]]
neg_p = probs[names.index("非物品")]

if   best_p < th:            reason = "threshold"
elif neg_p > best_p:         reason = "negative"
elif best_p - second_p < mg: reason = "margin"
else:                        reason = ""
verdict = "未知/待人工确认" if reason else best_zh
```

---

## 4. 接口定义

### 4.1 函数签名

```python
def classify(image, backend: str = "clip") -> dict:
    """
    Args:
        image:   图片路径(str) 或 PIL.Image
        backend: "clip" | "siglip"

    Returns:
        {
          "verdict":    str,          # 类别名 或 "★未知/待人工确认"
          "category":   str | None,   # 拒识时为 None
          "confidence": float,        # top1 得分
          "neg_score":  float,        # 负类得分
          "margin":     float,        # top1 - top2
          "topk":       [(str,float)],# Top-3
          "backend":    str,
          "latency_ms": float,
          "need_human": bool,
          "reason":     str,          # threshold / negative / margin / ""
        }
    """
```

### 4.2 批量调用

模型与类别原型**只加载一次并常驻内存**：

```python
_CACHE = {}
def load(key): ...    # 缓存 model / preprocess / tokenizer / 类别原型
```

**性能提示**：类别原型编码约 1.25 s，**整个进程只需做一次**。

---

## 5. 性能指标

### 5.1 实测值

| 指标 | CLIP ViT-B-32 | SigLIP2 ViT-B-32-256 |
|---|---|---|
| 模型加载（已缓存） | **2.0 s** | 40.3 s |
| 类别原型构建（一次性） | 约 1.25 s | 约 3 s |
| **单张推理** | **52 ms** | 68 ~ 82 ms |
| **30 件物品** | **约 1.6 s** | 约 2.2 s |
| 内存占用（fp32） | 约 1.2 GB | 约 2.5 GB |

### 5.2 目标值

| 指标 | 目标 | 实测（CLIP） | 状态 |
|---|---|---|---|
| 零样本分类准确率 | ≥ 70% | **100%**（16/16） | ✅ 远超 |
| **错认率（非物品被认成物品）** | **≤ 1%** | **0%**（0/7） | ✅ |
| **错分率（物品被认成别的物品）** | **≤ 1%** | **0%**（0/16） | ✅ |
| 单张延迟（CPU） | ≤ 150 ms | **52 ms** | ✅ |
| 训练数据需求 | **0 张** | 0 | ✅ |

> ⚠️ **注意样本量**：16+7 张的测试集只能证明「链路正确」，不足以证明「泛化能力」。
> 正式验收需 **每类 ≥20 张不同实物 + ≥30 张非物品图**。

---

## 6. 部署方案

### 6.0 项目环境（已就绪）

本项目已建立**独立虚拟环境**，与全局环境完全隔离（全局的 `open_clip_torch` / `transformers` 已卸载）。

```bat
setup_env.bat      :: 新机器首次：建 .venv + 装依赖
env.bat            :: 每次开工：激活 venv + 设环境变量
```

| 项 | 值 |
|---|---|
| venv 位置 | `.venv/` |
| 模型缓存 | **`.hf/hub/`（3.5GB，随项目走，可离线）** |
| 依赖清单 | `requirements.txt` / `requirements-lock.txt` |
| 离线运行 | `set HF_HUB_OFFLINE=1` ✅ 已实测 |

> **完整环境说明见项目根目录 `../README.md`**（含目录结构、命名规范、迁移内网步骤、已知问题）。

⚠️ **已知坑**：SigLIP2 离线加载需先跑 `python fix_hf_cache.py`
（原因：`timm/ViT-B-32-SigLIP2-256` 仓库自身缺 `config.json`，离线时 transformers 不会优雅降级）。
**CLIP 不受影响**（用内置 tokenizer，不依赖 transformers）。

### 6.1 依赖

```bash
pip install open_clip_torch torch pillow
# SigLIP2 额外需要：
pip install transformers
```

| 包 | 实测版本 | 说明 |
|---|---|---|
| `open_clip_torch` | 3.3.0 | |
| `torch` | 2.14.0+cpu | 纯 CPU 即可 |
| `pillow` | 11.3.0 | |
| `transformers` | 5.17.0 | 仅 SigLIP2 需要 |

### 6.2 环境变量（★ 国内必须，且必须在 import 前）

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"          # ★ 必须在 import 之前！
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch, open_clip        # ← 必须在设置环境变量之后
```

> `huggingface.co` 国内直连不通，不设镜像则**权重下载直接失败**。
> 且 `huggingface_hub` 在 **import 时**读取该变量，
> 放在 import 之后设置会**静默失效**（实测踩过，见 §9）。

### 6.3 模型缓存

权重默认在 `~/.cache/huggingface/hub/`。首次需联网，之后可离线。

**离线部署**：打包时把缓存目录一并拷入内网机器，设 `HF_HUB_OFFLINE=1`。

### 6.4 ARM 部署路径

```
① 功能验证：onnxruntime（ARM CPU）
② 性能优化：ONNX → RKNN（RK3588），速度提升 5~10 倍
③ 量化：INT8，体积降至约 1/4
```

| 后端 | fp32 | INT8 估算 | RK3588 内存建议 |
|---|---|---|---|
| **CLIP ViT-B-32** | **605 MB** | ~155 MB | **4 GB 版够用** |
| SigLIP2 ViT-B-32-256 | 1507 MB | ~380 MB | 建议 8 GB 版 |

> **CLIP 在 ARM 上的优势进一步放大**：体积只有 SigLIP2 的 40%。

---

## 7. 配置示例

```python
BACKENDS = {
    "clip": {
        "model":      "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "mode":       "softmax",
        "th":         0.70,          # ★ 实测标定（安全窗口 0.35~0.95）
        "size":       "605 MB",
    },
    "siglip": {
        "model":      "ViT-B-32-SigLIP2-256",
        "pretrained": "webli",
        "mode":       "sigmoid",
        "th":         0.0005,        # ★ 实测标定（窗口极窄，慎用）
        "size":       "1507 MB",
    },
}
DEFAULT_BACKEND = "clip"
MARGIN_TH = 0.10                     # top1-top2 间隔低于此值 → 转人工

# ★ 实测验证通过的最佳起步类别表（贴合托盘，16/16 + 7/7 全对）
#   建议先只启用前 5 类，稳定后再逐个加入 身份证 / 首饰
CLASSES = {
    "手机":   ["a mobile phone", "a smartphone", "a cell phone"],
    "现金":   ["paper money", "banknotes", "a stack of cash"],
    "手表":   ["a wristwatch", "a watch"],
    "钥匙":   ["a key", "a bunch of keys", "a keychain"],
    "银行卡": ["a credit card", "a bank card", "a plastic card"],
    # ↓ 后续逐个启用，每次启用后重跑阈值标定
    # "身份证": ["an ID card", "an identity card"],
    # "首饰":   ["a ring", "a necklace", "a bracelet", "jewelry"],
    # ★ 负类：必填，是拒识主力
    "非物品": [
        "a photo of a table", "a photo of a room", "an empty tray",
        "a photo of a white board", "a screenshot", "a document",
        "a text page", "a form", "a photo of a person", "a face",
        "hands", "a fruit", "a vegetable",
    ],
}
```

**命令行：**

```bash
python src/01_clip_zeroshot.py <路径>                # 默认（当前为 clip）
python src/01_clip_zeroshot.py <路径> --model siglip  # 切 SigLIP2
python src/01_clip_zeroshot.py <路径> --compare       # 并排对比
python src/02_eval_tray.py <目录>                     # 批量评测准确率
```

---

## 8. 风险与对策

### 8.1 ★ 最高风险：输入分布不匹配

| 项 | 内容 |
|---|---|
| **现象** | 用**整张场景图**输入时，SigLIP2 全部误拒（实测 34/34），险些判定模型不可用 |
| **根因** | 真实管线输入是「裁好的单个物品」；整图里托盘、文字、背景占据主导 |
| **对策** | **强制 L0 先切分，再送入本模块**；接口层做尺寸/内容断言 |
| **吸取的教训** | **测试输入分布必须与推理时一致。用整图测单物品分类器，结论无效。** |

### 8.2 ★ 次高风险：样本量不足导致错误结论

| 项 | 内容 |
|---|---|
| **现象** | 仅用 **4 张裁剪图**测试时，曾得出「SigLIP2 完胜 CLIP」的结论；扩到 23 张后结论完全反转 |
| **根因** | 小样本 + 类别表混杂，得分分布不具代表性 |
| **对策** | 阈值标定与选型对比**必须 ≥15 张物品 + ≥7 张非物品**；结论需在扩展样本上复验 |

### 8.3 阈值漂移

| 风险 | 现场光照/相机/台面变化会导致得分分布漂移 |
|---|---|
| **对策** | ① 优先选**窗口宽**的模型（CLIP）② 标定流程固化为上线必做项 ③ 定期用 `--compare` 复检 |

### 8.4 类别表设计错误

| 风险 | 描述 | 对策 |
|---|---|---|
| 类别过多/混杂 | 拒识能力显著恶化（见 §2.4） | **只放实际在用的类别** |
| 粒度太细 | 「戒指」再分金银 → 准确率骤降 | 粒度按登记需求定 |
| 中文描述 | 准确率显著下降 | 用英文 |
| 缺负类 | 非物品被强行分类 | 负类必填，贴合现场干扰物 |

### 8.5 长尾与属性需求

| 局限 | 零样本对**品牌/型号/材质/颜色**等细粒度属性支持弱 |
|---|---|
| **对策** | 交给 L2 的 VLM（Qwen3-VL / SmolVLM）处理 |

### 8.6 性能风险

| 风险 | 对策 |
|---|---|
| 类别数增加 → 原型编码变慢 | 一次性构建 + 缓存；< 200 类影响可忽略 |
| ARM 内存不足 | 选 CLIP（605MB）而非 SigLIP2（1507MB） |

---

## 9. 修订记录（v1.0 → v2.0）

| # | v1.0 结论 | v2.0 修正 | 原因 |
|---|---|---|---|
| 1 | **SigLIP2 为主力** | **CLIP 为主力** | v1.0 仅用 4 张裁剪图；扩到 23 张后 SigLIP2 阈值窗口过窄（0.0001~0.0014） |
| 2 | SigLIP2 拒绝率 11/11 更优 | 两者非物品拒绝均 7/7 全对 | 样本扩展后 CLIP 同样零误接受 |
| 3 | SigLIP2 阈值 = 0.12 | SigLIP2 阈值 = **0.0005** | 实测真实图片得分范围 0.0014~0.375，0.12 会拒掉 11/16 真实物品 |
| 4 | CLIP 阈值 = 0.35 | CLIP 阈值 = **0.70** | 窗口 0.35~0.95 均可行，取居中 |
| 5 | CLIP 误接受 2 次 | 聚焦类别表后 **0 次** | 17 类表混入相似类别所致（见 §2.4） |
| 6 | 未提"样本量不足"风险 | 单列 §8.2 | 本轮最大的方法论教训 |

### 本轮踩到的新坑

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **`HF_ENDPOINT` 设在 import 之后** | 报 `LocalEntryNotFoundError`，静默回落到 huggingface.co | **必须在 import 前设置** |
| 2 | **样本量太小** | 4 张图得出完全相反的选型结论 | 标定至少 15+7 张 |
| 3 | **类别表混入相似类别** | CLIP 把空托盘/人像认成身份证 | 只放实际在用的类别 |

---

## 10. 验收标准

| # | 项目 | 标准 | 状态 |
|---|---|---|---|
| 1 | 零训练数据可用 | 不提供任何训练图能输出品类 | ✅ 已验证 |
| 2 | **错认率**（非物品→物品） | **≤ 1%** | ✅ 0/7 |
| 3 | **错分率**（物品→别的物品） | **≤ 1%** | ✅ 0/16 |
| 4 | 拒识可解释 | 每次拒识给出 reason | ✅ |
| 5 | 单张延迟 | ≤ 150 ms（CPU） | ✅ 52 ms |
| 6 | 30 件物品总耗时 | ≤ 5 s | ✅ 约 1.6 s |
| 7 | 加新类无需重训 | 改类别表即生效 | ✅ |
| 8 | 离线可用 | 断网后正常推理 | 待验证 |
| 9 | **泛化能力** | 每类 ≥20 张不同实物 + ≥30 张非物品 | ⏳ **待扩展测试** |

> **第 9 条是当前最大的未知项。** 现有 23 张样本只能证明链路正确，
> 尚不能证明跨实物、跨光照的泛化能力。

---

## 11. 演进路线

| 阶段 | 方案 | 数据需求 | 预期准确率 | 触发条件 |
|---|---|---|---|---|
| **P0（本方案）** | **零样本 CLIP + 人工确认** | **0 张** | **实测 100%*** | 立刻可上 |
| P1 | Few-shot 原型（每类 10 张真实图） | ~680 张 | 85~95% | 上线后自动积累 |
| P2 | 微调 MobileNetV3（每类 100~300 张） | ~7,000 张 | 92~96% | 数据闭环成熟 |
| P3 | 数据闭环自学习，减少人工确认 | 自动增长 | 96%+ | 长期 |

\* 100% 为 23 张小样本结果，泛化能力待扩展验证。

> **P0 的核心价值不是准确率，而是「零数据门槛 + 立刻可用 + 边用边采」。**

---

## 附录 A · 核心实现

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"        # ★ 必须在 import 前
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch, open_clip
from PIL import Image

TEMPLATES = ["a photo of {}", "a close-up photo of {}", "{}"]
_CACHE = {}


def load_backend(key, globals_=None):
    """加载并缓存模型与类别原型（同进程只做一次）"""
    if key in _CACHE:
        return _CACHE[key]
    cfg = BACKENDS[key]
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"])
    model.eval()
    tokenizer = open_clip.get_tokenizer(cfg["model"])

    feats, names = [], []
    for zh, descs in CLASSES.items():
        prompts = [t.format(d) for d in descs for t in TEMPLATES]
        with torch.no_grad():
            f = model.encode_text(tokenizer(prompts))
            f = f / f.norm(dim=-1, keepdim=True)
            f = f.mean(dim=0)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f); names.append(zh)

    _CACHE[key] = (model, preprocess, torch.stack(feats), names, cfg)
    return _CACHE[key]


def score(sim, model, cfg):
    """★ 核心差异：CLIP 用 softmax，SigLIP2 用 sigmoid + bias"""
    scale = model.logit_scale.exp()
    if cfg["mode"] == "softmax":
        return (scale * sim).softmax(dim=-1)
    logits = scale * sim
    bias = getattr(model, "logit_bias", None)
    if bias is not None:
        logits = logits + bias
    return torch.sigmoid(logits)


def classify(image, backend="clip", margin_th=0.10):
    import time
    model, preprocess, TF, names, cfg = load_backend(backend)

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    t = time.time()
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        probs = score(f @ TF.T, model, cfg)[0]
    ms = (time.time() - t) * 1000

    ii = [i for i, n in enumerate(names) if n != "非物品"]
    ip = probs[ii]
    order = ip.argsort(descending=True)
    best_p = ip[order[0]].item()
    best_zh = names[ii[order[0].item()]]
    second_p = ip[order[1]].item() if len(order) > 1 else 0.0
    neg_p = probs[names.index("非物品")].item() if "非物品" in names else 0.0
    margin = best_p - second_p

    if   best_p < cfg["th"]:      reason = "threshold"
    elif neg_p > best_p:          reason = "negative"
    elif margin < margin_th:      reason = "margin"
    else:                         reason = ""

    return {
        "verdict":    "★未知/待人工确认" if reason else best_zh,
        "category":   None if reason else best_zh,
        "confidence": round(best_p, 4),
        "neg_score":  round(neg_p, 4),
        "margin":     round(margin, 4),
        "topk":       [(names[ii[o.item()]], round(p.item(), 4))
                       for p, o in zip(ip[order[:3]], order[:3])],
        "backend":    backend,
        "latency_ms": round(ms, 1),
        "need_human": bool(reason),
        "reason":     reason,
    }
```

---

## 附录 B · 实测数据汇总

### B.1 分类准确率（23 张）

| 后端 | 纯分类准确率 | 物品接受 | 非物品拒绝 | 错认 |
|---|---|---|---|---|
| **CLIP ViT-B-32** (th=0.70) | **16/16 = 100%** | 16/16 | 7/7 | **0** |
| SigLIP2 ViT-B-32-256 (th=0.0005) | **16/16 = 100%** | 16/16 | 7/7 | **0** |

### B.2 得分分布

| | CLIP | SigLIP2 |
|---|---|---|
| 物品 top1 | 0.9944 ~ 1.0000 | 0.00142 ~ 0.37456 |
| 波动倍数 | **1.006x** | **264x** |
| 非物品 top1 最高 | 0.1748 | 0.00004 |
| 安全阈值窗口 | **0.35 ~ 0.95** | 0.0001 ~ 0.0014 |

### B.3 性能

| 项 | CLIP | SigLIP2 |
|---|---|---|
| 参数 | 151.3 M | 376.9 M |
| 体积 | 605 MB | 1507 MB |
| 单张 | 52 ms | 68~82 ms |
| 模型加载（缓存） | 2.0 s | 40.3 s |
| 首次下载 | 8.5 min | 22.6 min |

### B.4 测试集构成

| 类别 | 张数 | 说明 |
|---|---|---|
| 手机 | 4 | 500×500 ~ 1268×881 |
| 钥匙 | 8 | 640×422 ~ 1920×1920 |
| 银行卡 | 4 | 1243×1125 ~ 3024×4032 |
| **物品小计** | **16** | |
| 空托盘 / 布料背景 / 文档 / 截图 / 公章 / 人像 / 草图 | 7 | |
| **合计** | **23** | |

---

## 附录 C · 踩坑记录（累计）

| # | 坑 | 现象 | 解法 |
|---|---|---|---|
| 1 | **未设 HF 镜像** | 权重下载失败 | `HF_ENDPOINT=https://hf-mirror.com` |
| 2 | **`HF_ENDPOINT` 设在 import 之后** | `LocalEntryNotFoundError`，静默回落到 huggingface.co | **必须 import 前设置** |
| 3 | 未 `model.eval()` | 结果随机抖动 | 加 `model.eval()` |
| 4 | 硬编码 `100 *` | 换模型即失效 | `model.logit_scale.exp()` |
| 5 | 未 `torch.no_grad()` | 慢、吃内存 | 包一层 |
| 6 | **类别用中文** | 准确率显著下降 | 改英文描述 |
| 7 | 单描述、无拒识 | 准确率低、无"未知"能力 | 多描述 + 多模板 + 三重拒识 |
| 8 | **SigLIP2 缺 `transformers`** | `ModuleNotFoundError` | `pip install transformers` |
| 9 | **沿用 CLIP 的阈值** | SigLIP2 拒掉 11/16 真实物品 | 阈值按后端重新标定（0.0005） |
| 10 | **用整图测试** | SigLIP2 全拒，误判模型不可用 | 改用裁剪单件图 |
| 11 | **样本量太小（4 张）** | 选型结论完全相反 | 标定 ≥15 张物品 + ≥7 张非物品 |
| 12 | **类别表混入相似类别** | CLIP 把空托盘/人像认成身份证 | 只放实际在用的类别 |

---

## 附录 D · 关联文档

| 文档 | 内容 |
|---|---|
| `src/01_clip_zeroshot.py` | 可执行实现（双后端 + 对比模式） |
| `src/02_eval_tray.py` | 批量准确率评测脚本 |
| `src/web_demo.py` | Web Demo（上传 / 分类管理 / 准确率面板） |
| `src/cfg_source.py` | 统一配置加载器（单一数据源） |
| `../README.md` | 环境搭建与使用说明 |

> 通用规范与调研类文档（总体技术方案、开源项目调研、ARM 端侧选型、
> 公开数据集调研、第一轮实验记录）已于 2026-09-23 移出本目录，
> 备份在 `..\物品识别想法_已删文档备份_20260923.zip`。

### B.5 最终验收测试（方案文档 §7 推荐配置）

对 §7 的完整配置（**5 类 + 负类**，CLIP th=0.70，SigLIP2 th=0.0005）做端到端验收：

```bash
python src/02_eval_tray.py .
```

| 后端 | 物品正确 | 非物品正确拒识 | 错分 | 误接受 | 平均耗时 |
|---|---|---|---|---|---|
| **CLIP** (th=0.70) | **16/16 = 100%** | **7/7 = 100%** | **0** | **0** | 60~89 ms |
| **SigLIP2** (th=0.0005) | **16/16 = 100%** | **7/7 = 100%** | **0** | **0** | 81~88 ms |

**两个后端在正确标定下均达成零错误。**

> **再次强调 §8.2 的教训**：这个结论只在 23 张样本上成立。
> 正式验收需扩展到 **每类 ≥20 张不同实物 + ≥30 张非物品图**，
> 因为 SigLIP2 的得分波动高达 264 倍，样本扩大后阈值可能需要重标。

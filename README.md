# 随身物品识别 —— 项目环境说明（Chinese-CLIP 版）

> 零样本识别（**Chinese-CLIP ViT-B/16**）+ 后续 L0 差分切分、L2 VLM 兜底
> 目标平台：Windows（开发）→ ARM/RK3588（部署）
>
> **本分支（`dev-chinese`）只用 Chinese-CLIP，类别描述用中文维护。**
> 英文 CLIP / SigLIP2 版本见 `dev-english` 分支。

---

## 1. 快速开始

### 首次在新机器上（一次性）

```bat
setup_env.bat
```

做三件事：创建 `.venv` → 升级 pip → 安装 `requirements.txt`。

> ⚠️ 如果 `<项目根>\.hf\chinese-clip-vit-base-patch16\` 不存在，
> 首次运行会从镜像下载约 **719MB** 模型权重。

### 每次开工

```bat
env.bat
```

激活 venv + 设置全部环境变量。之后直接跑脚本即可。

**PowerShell 用户**：`.\env.ps1`
（若被策略拦截：`powershell -ExecutionPolicy Bypass -File .\env.ps1`）

### 启动 Web Demo ★

```bat
run_web.bat                    :: 默认 http://127.0.0.1:8000，自动开浏览器
run_web.bat --port 8080
run_web.bat --no-browser
```

**零额外依赖**（纯 Python 标准库），功能详见 §3。

---

## 2. 目录结构

```
物品识别想法/
├── .venv/                        # 虚拟环境（不纳入版本控制）
├── .hf/                          # ★ 模型缓存（719MB，随项目走）
│   └── chinese-clip-vit-base-patch16/
│       ├── pytorch_model.bin     #     718MB
│       ├── config.json  vocab.txt  preprocessor_config.json
├── .pi/                          # 设计规范 skill
│
├── README.md                     # ★ 本文件（入口）
├── docs/
│   ├── 方案_零样本识别_CLIP与SigLIP2.md   # 主方案文档（含英文后端方案）
│   └── 实验记录.md                        # ★ 六组实测：类别表容量与成本
│
├── src/                          # ★ 全部代码
│   ├── web_demo.py               #    ★ Web Demo（上传/分类管理/准确率面板）
│   ├── 01_clip_zeroshot.py       #    零样本分类（命令行）
│   ├── cnclip_model.py           #    Chinese-CLIP 封装
│   ├── cfg_source.py             #    共享配置加载器（单一数据源）
│   └── web_demo_categories.json  # ★ 统一配置（中文描述）
├── src/.feat_cache/              # ★ 文本原型磁盘缓存（可删，自动重建）
│   └── cnclip_<指纹>.pt          #     88 KB
│
├── env.bat / env.ps1             # 激活环境（每次开工用）
├── setup_env.bat                 # 一键搭建（新机器用）
├── run_web.bat                   # 启动 Web Demo
├── requirements.txt              # 直接依赖
├── requirements-lock.txt         # 完整版本锁定（49 个包）
│
└── images/                       # ★ 测试图集（192 张，27 个类别 + 7 张负类）
```

> 通用规范与调研类文档（总体技术方案、开源项目调研、ARM 端侧选型、公开数据集调研、
> 第一轮实验记录）已移出本目录，备份在上一级：
> `..\物品识别想法_已删文档备份_20260923.zip`（36KB）

---

## 3. Web Demo（图形界面）

```bat
run_web.bat
```

浏览器自动打开 `http://127.0.0.1:8000`。界面遵循 **YeZh 设计系统**（暖灰中性色、玻璃卡片、浮入动画），**浅色 / 深色 / 跟随系统**三档主题，右上角一键切换。

> 截图已从仓库移出（界面细节以实际运行为准）。需要预览直接跑 `run_web.bat`。

### 页面布局

| 区域 | 内容 |
|---|---|
| **顶栏** | 品牌标识 · 主题三档切换 · **右上角「配置」按钮** |
| **KPI 面板** | 按后端分组：物品准确率 / 非物品拒识率 / 错分 / 误接受 |
| **状态条** | 模型是否已加载 · 分类数 · 负类数 · 当前阈值与 margin |
| **上传与识别** | 拖拽图片/文件夹、缩放设置、分批识别进度条 |
| **分类明细** | 逐类别的「正确/总数 · 拒识 · 错分」，含负类行 |
| **逐图结果** | 缩略图、真值、判定、置信度、耗时；可筛选「仅错误 / 仅未知」 |

### ★ 配置抽屉（右上角）

所有**模型参数**与**识别分类**都收进了右上角「配置」按钮打开的抽屉，主界面保持干净：

| 分组 | 可调项 |
|---|---|
| **模型与判定** | 当前模型（Chinese-CLIP ViT-B/16）· 识判阈值 · 相对间隔 margin · 预加载模型 |
| **识别分类** | 增删分类 · 改分类名 · 编辑**中文描述**（每行一条）· 勾选**负类** · 保存 / 恢复默认 |

保存后写入 `web_demo_categories.json`，**自动失效特征缓存**，无需重启服务。
判定失败的图会标红（错分）或标橙（拒识），异常单独展示原因。

#### ⚠️ 扩类别表前先读这段

实测结论（详见 [`docs/实验记录.md`](docs/实验记录.md)）：

- **类别数本身不是问题，语义近邻才是。**
  同样加到 17 类，加「远邻」类（雨伞/鞋/剪刀）不掉分，加「近邻」类（身份证/车钥匙/耳机）直接掉 4 个。
- **不要父子类并存。** 同时放「钥匙」和「车钥匙」，钥匙会被车钥匙抢走（0.99）。
- **描述里不要写“带照片”这类线索词。** 给身份证写 `with a portrait photo` 会把人像照
  拉到 0.97（去掉降到 0.65）。写 prompt 要用「看着像什么」，不是「是什么」。
- **每加一类都要问：现场有什么东西看起来像它？**
  例如「围巾」会吃掉灰色布料背景（0.96）。
- **警惕「垃圾桶类」。** 跑完看「非物品被拒识时 Top-1 是谁」，某个类反复出现 → 描述太宽。

当前 42 类的实测（28 张图 / 8 个类别）：

| 模型 | 物品正确 | 误拒 | 错分 | 拒识 | **误接受** |
|---|---|---|---|---|---|
| **Chinese-CLIP（本分支）** | 21/28 = 75% | 7 | **0** | 7/7 | **0** |
| 英文 CLIP（dev-english） | 21/28 = 75% | 7 | 0 | 7/7 | 0 |

> **五项指标完全相同** —— 中文描述可以无损替代英文描述。
> （`dev-english` 上 SigLIP2 能到 89.3%，但有 **2 例错分**；
> 按「不认错优先」原则，0 错分的模型更稳 —— 错分比误拒危险得多。）

### 真值怎么来的

**文件名前缀即真值**，无需额外标注：

```
手机1.jpg    → 真值 = 手机
钥匙2.jpeg   → 真值 = 钥匙
n01_tray.png → 真值 = 非物品（期望被拒识）
```

文件名无法匹配任何分类的，显示「未标注」，**不计入准确率**。

**测试图集**：`images/`（192 张，已入库）——
27 个物品类别 + 7 张负类（`n01_tray.png` … `n07_sketch.png`）。
**同名不同图**的会加 `_b` / `_c` 后缀（如 `手机3_b.jpeg`），前缀仍是类别名，不影响解析。

### 性能（实测）

| 项 | 实测 |
|---|---|
| 服务启动（端口可连） | **约 1 秒**（torch 惰性导入） |
| 首次模型加载 | **0.3s**（本地目录） |
| 首次建原型（43 类） | **约 5~8s** —— 仅此一次，自动落盘 |
| 重启后建原型 | **0s**（读 `.feat_cache`） |
| 单张推理 | **约 210~300 ms** —— 与类别数无关 |
| 28 张 | 预热后 **约 7s** |

> Chinese-CLIP 是 ViT-B/**16**（patch token 196 个），比英文 CLIP 的 ViT-B/**32**（49 个）慢约 2 倍，
> 但**建原型快约 3 倍**（中文分词更紧凑）。准确率两者完全相同。

### ★ 页面自动预热 + 文本原型缓存

**页面一打开就后台预热**，不等用户点「开始识别」：

```
打开页面  →  后台 POST /api/preload  →  加载模型 + 读/建原型
              ↓
          状态条显示「正在后台预热 cnclip …」
              ↓
用户选完图时模型已就绪，点识别直接出结果
```

**文本原型缓存的指纹只由「类别表 + 模板」决定**：

| 你改了什么 | 重建原型？ |
|---|---|
| 分类名 / 中文描述 | ✅ 重建（应然） |
| **阈值** | ❌ **不重建**（实测 13.3s → 0.08s） |
| **margin** | ❌ **不重建** |
| 什么都没改，重复保存 | ❌ **不重建** |

启动时自动清理孤儿缓存（保留当前配置的 + 最近 8 个）。

> 迁移到内网时**建议把 `src/.feat_cache/` 一起拷走** —— 能省掉首次几秒的建原型。
> 拷不拷都能跑，拷了首次启动就快。

### 设计说明

- **纯标准库**（`http.server`），不引入 Flask/Streamlit/Gradio，也**不依赖任何 CDN**（内网可用）
- **惰性导入** torch：HTTP 服务秒起，不阻塞页面
- 图片在**浏览器端缩放**后再上传（默认最长边 768），透明 PNG 自动铺白底
- 拖拽文件夹会**递归**取子目录（`webkitGetAsEntry`）
- 分批提交（每批 6 张）+ 进度条
- 类别变更后**自动失效特征缓存**，无需重启
- 三重拒识：低于阈值 / 负类胜出 / **相对间隔** (top1−top2)÷top1 过小
- 支持 URL 参数：`?theme=light|dark|system`、`?drawer=1`（直接展开配置抽屉）
- **断连健壮**：浏览器提前断开（超时/刷新/关标签页）不会弄挂服务端，只记一行日志
- **并发安全**：预热未完成时多个请求同时到达，只会编码一次原型

---

## 4. 环境变量（`env.bat` 自动设置）

| 变量 | 值 | 作用 |
|---|---|---|
| `HF_HOME` | `<项目>\.hf` | **模型缓存指向项目内**，便于整体迁移到内网 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | **国内必须**，huggingface.co 直连不通 |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `1` | 静音 Windows 符号链接警告 |
| `PYTHONIOENCODING` / `PYTHONUTF8` | `utf-8` / `1` | 中文路径与输出 |
| `TRANSFORMERS_VERBOSITY` | `error` | 静音已知的无害警告 |

### 离线运行（内网部署）

```bat
set HF_HUB_OFFLINE=1
```

配合项目内 `.hf\chinese-clip-vit-base-patch16\` 即可**完全离线**运行（已实测验证）。

---

## 5. 脚本用法（命令行）

> 三个脚本**共用同一份配置** `web_demo_categories.json`（单一数据源）。
> 在 Web Demo 里改完分类并保存，命令行脚本会自动跟随。
> 详见 §5.3。

### src/01_clip_zeroshot.py —— 单张 / 批量分类

```bash
python src/01_clip_zeroshot.py <图片或文件夹>            # 分类
python src/01_clip_zeroshot.py --cats                   # 只看当前分类表
python src/01_clip_zeroshot.py <路径> --no-config        # 用内置默认值
python src/01_clip_zeroshot.py <路径> --margin 0.25      # 临时改相对间隔
```

输出：判定类别、置信度、负类得分、相对间隔、Top-3、耗时、拒识原因。
目录会**递归**扫描子目录。

### 准确率评测 —— 用 Web Demo 的图形界面

准确率评测已统一到 Web Demo（见 §3）。步骤：

1. `run_web.bat` 启动，浏览器打开配置抽屉确认分类表
2. 把整个 `images/` 文件夹**拖进上传区**（支持拖拽文件夹，会递归）
3. 点「开始识别」 → 面板自动算出：
   物品准确率 · 非物品拒识率 · 错分 · **误接受** · 逐类别明细

> 真值取自文件名前缀，**不需要额外标注**。
> 结果表格里命中图像缓存的会显示 ⚡（重跑同一批图约 48 倍加速）。

### src/cfg_source.py —— 单一数据源

| 优先级 | 来源 |
|---|---|
| 1 | `--config <路径>` 显式指定 |
| 2 | **`web_demo_categories.json`** ← 唯一数据源 |
| 3 | 各脚本内置的默认值（仅兑底） |

```bash
python src/cfg_source.py --show     # 打印当前生效的配置（分类+阈值+间隔）
python src/cfg_source.py --check    # 只看来源
python src/cfg_source.py --init     # 用默认值生成 web_demo_categories.json
```

两个命令行脚本都支持：

```bash
--config <路径>   # 临时用别的配置文件
--no-config      # 忽略配置文件，强制用脚本内置默认值
--cats           # 只打印当前分类表后退出
```

> `web_demo.py` 保持**完全自包含**（内网部署时单文件拷走即可），
> 但它读写的是**同一个 json 文件**，所以数据依然是单一来源。

---

## 6. ⚠️ 已知问题与修复

### 6.1 transformers 5.x 的 `get_*_features` 有两个坑

**本项目实际踩过，务必注意：**

| 现象 | 说明 |
|---|---|
| `get_text_features()` 返回的不是张量 | 它返回 `BaseModelOutputWithPooling`，真正的嵌入在 **`.pooler_output`** |
| 拿到的嵌入**没归一化** | 模长约 **36**（不是 1）。必须自己 `f / f.norm(dim=-1, keepdim=True)` |

**不修会怎样**：相似度大两个数量级 → `logit_scale × sim` 全部溢出 → **所有图都判错**，
而且不报任何异常（只是结果全错）。

正确写法（见 `src/cnclip_model.py`）：

```python
f = model.get_text_features(**tk).pooler_output     # ← 不是返回值本身
f = f / f.norm(dim=-1, keepdim=True)                # ← 必须自己归一化
```

### 6.1.1 不要把 .bat 之外的文本模板写成英文

Chinese-CLIP 的文本塔是**中文 BERT（RoBERTa-wwm）**，只认中文。
用英文模板 `"a photo of {}"` 会得到一只乱码 token 序列，准确率崩掉。

### 6.2 中文路径

项目路径含中文，已验证 Python / torch / opencv 全部正常。
`env.bat` 已设置 `PYTHONUTF8=1`，避免中文输出乱码。

### 6.3 不要自己写预处理

必须使用模型自带的 `processor`（ChineseCLIPProcessor）做预处理，
**不要手写 Resize / CenterCrop / Normalize**。
不一致不会报错，只会让准确率静默下降。

```python
ii = processor(images=pil_img, return_tensors="pt")
f  = model.get_image_features(**ii)      # processor 已处理好 224×224 与归一化
```

### 6.4 `.bat` 必须纯 ASCII + CRLF

**症状**：双击 `run_web.bat` 后报一堆
`'--port' 不是内部或外部命令` / `'cho' 不是内部或外部命令`，服务起不来。

**原因**：cmd.exe 是按**字节偏移**流式读取批处理文件的。`chcp 65001` 改变代码页后，
文件里存在多字节字符（中文）会让读取位置错位，后续行被当成垃圾命令执行。
**这与 CRLF/LF 无关**，纯粹是编码错位。

**解决**：所有 `.bat` 保持**纯 ASCII**（注释和提示语一律用英文），
`chcp 65001` 只为让 Python 的中文输出正常显示。

> 另外：若端口报 `PermissionError [WinError 10013]`，说明该端口在
> Windows 保留段内。换端口即可（实测 8000 / 8080 / 8765 / 8888 正常）。

### 6.5 初始 `hidden` 的区块不要预先挂动画类

`animation-fill-mode: both` 会在**延迟期间**就把元素置于 `from` 状态（`opacity:0`）。
如果元素初始是 `display:none`，动画可能永不执行，内容就**永久不可见**。

**做法**：动态显示的区块先 `classList.remove('hidden')`，再添加 `.reveal` 类，
并带一个 700ms 的兵底定时器强制可见。

---

## 7. 依赖说明

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | 2.14.0+cpu | 纯 CPU 即可，无需 GPU |
| `torchvision` | 0.29.0+cpu | |
| **`transformers`** | 5.17.0 | **★ Chinese-CLIP 靠它加载**（含中文 BERT 文本塔） |
| `pillow` | 12.3.0 | 图像读取与缩放 |
| `opencv-python` | 5.0.0.93 | 预留：L0 差分 / 连通域 |
| `rapidocr-onnxruntime` | 1.2.3 | 预留：L0 证件 OCR |
| `onnxruntime` | 1.30.0 | 预留：部署推理 |

> 本分支**不再需要** `open_clip_torch` / `timm`（那是英文 CLIP 与 SigLIP2 的依赖）。

**为什么用独立的 venv**：`torch` + `transformers` 依赖较重，且
`transformers` 版本迭代快、易与其他项目冲突，隔离后互不影响。

### 全局环境现状

以下包**已从全局环境卸载**，现只存在于 `.venv` 中：

| 包 | 说明 |
|---|---|
| `open_clip_torch` | 英文 CLIP / SigLIP2 实现（本分支不需要） |
| `transformers` | **Chinese-CLIP 依赖它** |
| `clip` | OpenAI 原版 CLIP（与 open_clip_torch 是两个不同的包） |
| `timm` | open_clip 的依赖 |

全局**保留**（其他工具可能仍在用）：`torch` `torchvision` `opencv-python` `onnxruntime` `rapidocr-onnxruntime`。

---

## 8. 迁移到内网 / 新机器

整个项目目录（含 `.venv` 与 `.hf`）可**整体拷贝**，但更推荐在新机器上重建环境：

```bat
:: 1. 拷贝项目（可跳过 .venv 以减小体积）
:: 2. 拷贝 .hf 目录（719MB）—— 有它才能离线
:: 3. 新机器上执行
setup_env.bat
env.bat
:: 验证：跑 Web Demo，把 images\ 拖进去看准确率
```

**若无网络**，需提前在有网机器上：
```bash
pip download -r requirements.txt -d wheels/
```
再把 `wheels/` 一起拷过去，用 `pip install --no-index --find-links wheels/ -r requirements.txt` 安装。

---

## 9. 当前状态

| 项 | 状态 |
|---|---|
| 识别后端 | ✅ **Chinese-CLIP ViT-B/16**（单一后端，中文描述） |
| 模型缓存 | ✅ 项目内 `.hf/chinese-clip-vit-base-patch16/`（719MB） |
| 离线运行 | ✅ 实测通过（`HF_HUB_OFFLINE=1`，模型从本地目录加载） |
| 评测结果 | ✅ 28 张 / 8 类：**物品 21/28 · 误拒 7 · 错分 0 · 拒识 7/7 · 误接受 0** |
| 中英对照 | ✅ 与 `dev-english`（英文 CLIP）**五项指标完全相同** |
| Web Demo | ✅ 浅色/深色双主题 · 配置抽屉 · 自动预热 · 完整准确率面板 |
| 配置来源 | ✅ 统一为 `web_demo_categories.json`（Web + 两个命令行脚本共用） |
| 实验记录 | ✅ `docs/实验记录.md`（六组实测：类别表容量与成本） |
| 样本量 | ⚠️ 8 个类别有数据，**其余 34 类零样本**；每类仅 1~8 张，置信区间很宽 |
| 泛化验证 | ⏳ 待扩展至每类 ≥20 张不同实物 + ≥30 张非物品图 |

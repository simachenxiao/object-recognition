"""
修复 HuggingFace 缓存中的 SigLIP2 离线加载问题

问题：
    timm/ViT-B-32-SigLIP2-256 仓库本身没有 config.json。
    在线时 transformers 拿到 404 会优雅降级；
    离线（HF_HUB_OFFLINE=1）时 huggingface_hub 返回 .no_exist 标记，
    transformers 却直接抛 OSError，导致 SigLIP2 tokenizer 加载失败。

解决：
    在快照目录补一个最小 config.json，并删除过期的 .no_exist 标记。

用法：
    python fix_hf_cache.py            # 使用 HF_HOME 或默认缓存
    python fix_hf_cache.py <缓存hub目录>
"""
import os
import sys
import json
import glob

REPO_DIR_NAME = "models--timm--ViT-B-32-SigLIP2-256"

MINIMAL_CONFIG = {
    "model_type": "siglip2",
    "architectures": ["Siglip2Model"],
    "vocab_size": 256000,
    "bos_token_id": 235250,
    "eos_token_id": 1,
    "pad_token_id": 0,
}


def find_hub_dir(argv):
    if len(argv) > 1:
        return argv[1]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


def main():
    hub = find_hub_dir(sys.argv)
    print(f"缓存目录: {hub}")
    if not os.path.isdir(hub):
        print("  [SKIP] 目录不存在")
        return 0

    repo = os.path.join(hub, REPO_DIR_NAME)
    if not os.path.isdir(repo):
        print(f"  [SKIP] 未找到 {REPO_DIR_NAME}（可能尚未下载 SigLIP2）")
        return 0

    changed = 0

    # 1) 补 config.json
    snapshots = glob.glob(os.path.join(repo, "snapshots", "*"))
    for snap in snapshots:
        cfg = os.path.join(snap, "config.json")
        if os.path.exists(cfg):
            print(f"  [OK]   config.json 已存在: {os.path.basename(snap)}")
        else:
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump(MINIMAL_CONFIG, f, indent=2)
            print(f"  [FIX]  已创建 config.json: {os.path.basename(snap)}")
            changed += 1

    # 2) 删除过期的 .no_exist 标记
    stale = glob.glob(os.path.join(repo, ".no_exist", "*", "config.json"))
    for s in stale:
        os.remove(s)
        print(f"  [FIX]  已移除过期标记: ...{os.sep}.no_exist{os.sep}...{os.sep}config.json")
        changed += 1
    if not stale:
        print("  [OK]   无过期 .no_exist 标记")

    print()
    if changed:
        print(f"完成，共修改 {changed} 处。")
    else:
        print("无需修改，缓存已正常。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

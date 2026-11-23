# -*- coding: utf-8 -*-
"""
模型与数据集下载脚本（可选辅助工具）
============================================================
本项目默认手动管理模型与数据；本脚本仅在需要联网拉取资源时使用，
下载结果直接落到项目的本地目录，与 PyCharm 离线运行的目录结构一致：

    模型 -> model/<模型名>/
    数据 -> data/TOFU（JSONL）、data/DUET（parquet）、data/WMDP（parquet）、
            data/RWKU（完整仓库，含 README 等全部原始文件）

脚本可重复执行，已下载且校验通过的文件会自动跳过，不会重复下载。

用法：
    # 国内网络推荐加镜像（hf-mirror.com），下载默认 8B 模型与全部数据
    python download_assets.py --mirror

    # 官方源下载（需要能访问 huggingface.co；Llama 为 gated 仓库需先登录）
    python download_assets.py

    # 只下载数据或只下载模型
    python download_assets.py --datasets_only
    python download_assets.py --models_only

    # 按数据集名完整下载（保留仓库原貌，不做任何文件过滤），可重复传入
    # 短名走内置清单（tofu/duet/wmdp/rwku）
    python download_assets.py --datasets_only --mirror --dataset rwku
    # 也可直接传任意 HF 数据集仓库 ID，本地目录名取仓库最后一段
    python download_assets.py --datasets_only --mirror --dataset jinzhuoran/RWKU
    # --full 让内置清单也按完整仓库下载（而非只取 json/parquet）
    python download_assets.py --datasets_only --mirror --full

    # 用短名指定模型：8b（默认，Llama-3.1-8B）、qwen35（Qwen3.5-9B）
    python download_assets.py --model qwen35
    # 也可直接传任意 HF 仓库 ID，本地目录名取仓库最后一段
    python download_assets.py --model Qwen/Qwen3.5-9B
"""
import argparse
import os
import time

# 脚本所在目录即项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 已知模型清单：短名 -> (HF 仓库 ID, 本地目录名)
# 与 config.py 的 MODEL_SPECS 键、model/ 下实际目录名保持一致
MODEL_REGISTRY = {
    "8b": ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct"),
    "qwen35": ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
}
# 默认模型短名（与 config.py 的 LOCAL_MODEL_DIR 指向同一目录）
DEFAULT_MODEL_KEY = "8b"

# 数据集清单：短名 -> (HF 仓库 ID, 本地目录, 文件匹配模式)
# 仓库 ID 均经官方页面核实；DUET/WMDP 为 parquet，TOFU 为 JSONL；
# 匹配模式为 None 表示完整下载（保留仓库全部文件与原始目录结构）
DATASET_REGISTRY = {
    "tofu": ("locuslab/TOFU",
             os.path.join(BASE_DIR, "data", "TOFU"), "*.json"),
    "duet": ("SwetieePawsss/DUET",
             os.path.join(BASE_DIR, "data", "DUET"), "*.parquet"),
    "wmdp": ("cais/wmdp",
             os.path.join(BASE_DIR, "data", "WMDP"), "*.parquet"),
    "rwku": ("jinzhuoran/RWKU",
             os.path.join(BASE_DIR, "data", "RWKU"), None),
}
# 向后兼容保留的 TOFU 常量
TOFU_REPO = DATASET_REGISTRY["tofu"][0]
TOFU_LOCAL_DIR = DATASET_REGISTRY["tofu"][1]


def parse_args():
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="下载 Deep-RULE 2.0 所需模型与数据（落到本地目录）")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY,
                        help="模型短名（8b/qwen35）或 HF 仓库 ID，默认 %s" % DEFAULT_MODEL_KEY)
    parser.add_argument("--mirror", action="store_true",
                        help="使用国内镜像 hf-mirror.com")
    parser.add_argument("--models_only", action="store_true",
                        help="只下载模型，跳过数据集")
    parser.add_argument("--datasets_only", action="store_true",
                        help="只下载数据集，跳过模型")
    parser.add_argument("--dataset", action="append",
                        help="数据集短名（tofu/duet/wmdp/rwku）或完整 HF 数据集"
                             "仓库 ID，可重复传入，按数据集名落到 data/ 下")
    parser.add_argument("--full", action="store_true",
                        help="完整下载数据集（不过滤文件，保持仓库原貌）")
    return parser.parse_args()


def setup_endpoint(use_mirror):
    """配置下载源：国内镜像或官方源。"""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("已启用国内镜像：https://hf-mirror.com")
    else:
        print("使用官方源：https://huggingface.co")


def resolve_model_target(model_arg):
    """
    把命令行传入的模型参数解析为 (HF 仓库 ID, 本地目录绝对路径)。
    短名走内置清单；完整仓库 ID 则本地目录名取最后一段。
    """
    if model_arg in MODEL_REGISTRY:
        repo_id, dir_name = MODEL_REGISTRY[model_arg]
    else:
        # 按 HF 仓库 ID 处理，例如 Qwen/Qwen3.5-9B
        repo_id = model_arg
        dir_name = model_arg.rstrip("/").split("/")[-1]
    local_dir = os.path.join(BASE_DIR, "model", dir_name)
    return repo_id, local_dir


def download_model(repo_id, local_dir):
    """
    下载模型仓库到指定本地目录，只取推理/微调必需文件，
    跳过 gguf、pth、pt、bin 等无关大文件。返回本地目录。
    """
    from huggingface_hub import snapshot_download

    print("开始下载模型：%s -> %s" % (repo_id, local_dir))
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "*.json", "*.safetensors", "*.txt", "*.jinja",
            "tokenizer*", "original/*",
        ],
        ignore_patterns=["*.gguf", "*.pth", "*.pt", "*.bin", "*.onnx"],
    )
    print("模型下载完成（%.2f GB，耗时 %.1f 分钟）"
          % (folder_size(local_dir), (time.time() - t0) / 60))
    return local_dir


def print_dataset_urls(repo_id):
    """打印数据集官方页与镜像页，便于人工核对仓库身份。"""
    print("数据集官方页：https://huggingface.co/datasets/%s" % repo_id)
    print("数据集镜像页：https://hf-mirror.com/datasets/%s" % repo_id)


def resolve_dataset_target(dataset_arg):
    """
    把命令行传入的数据集参数解析为 (HF 仓库 ID, 本地目录绝对路径)。
    短名走内置清单；完整仓库 ID 则本地目录名取仓库最后一段，
    统一保存到项目 data/ 文件夹下。
    """
    key = dataset_arg.strip().rstrip("/")
    if key in DATASET_REGISTRY:
        return key, DATASET_REGISTRY[key][1]
    repo_id = key
    local_dir = os.path.join(BASE_DIR, "data", repo_id.split("/")[-1])
    return repo_id, local_dir


def download_full_dataset(repo_id, local_dir):
    """
    完整下载数据集仓库到指定本地目录：不传任何 allow/ignore 过滤，
    保留仓库的全部文件与原始子目录结构（数据集原本啥样就存成啥样）。
    已存在且校验通过的文件由 huggingface_hub 自动跳过。
    """
    from huggingface_hub import snapshot_download

    os.makedirs(local_dir, exist_ok=True)
    print_dataset_urls(repo_id)
    print("完整下载（保留全部文件与原始结构）-> %s" % local_dir)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
    )
    print("数据集 %s 下载完成（%.2f MB，耗时 %.1f 分钟）"
          % (repo_id, folder_size(local_dir) * 1024, (time.time() - t0) / 60))


def download_dataset(name, full=False):
    """按清单下载单个数据集到对应本地目录，已存在的文件会自动跳过。"""
    repo_id, local_dir, pattern = DATASET_REGISTRY[name]
    # 模式为空或显式要求完整下载时，按仓库原貌整体拉取
    if full or pattern is None:
        download_full_dataset(repo_id, local_dir)
        return

    from huggingface_hub import snapshot_download

    print_dataset_urls(repo_id)
    print("开始下载数据集 %s（%s）-> %s" % (name, repo_id, local_dir))
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=pattern,
    )
    print("数据集 %s 下载完成（%.2f MB，耗时 %.1f 分钟）"
          % (name, folder_size(local_dir) * 1024, (time.time() - t0) / 60))


def download_tofu():
    """向后兼容的 TOFU 下载入口。"""
    download_dataset("tofu")


def folder_size(path):
    """统计目录总大小，单位 GB。"""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.islink(fp):
                continue
            total += os.path.getsize(fp)
    return total / 1024**3


def main():
    """主流程：按参数下载模型和/或数据，最后打印本地目录位置。"""
    args = parse_args()
    setup_endpoint(args.mirror)

    if not args.datasets_only:
        repo_id, local_dir = resolve_model_target(args.model)
        download_model(repo_id, local_dir)

    if args.dataset:
        # 显式指定数据集：短名或任意仓库 ID，按数据集名落到 data/ 下
        for target in args.dataset:
            repo_id, local_dir = resolve_dataset_target(target)
            download_full_dataset(repo_id, local_dir)
    elif not args.models_only:
        # 未显式指定时，按内置清单逐个下载
        for name in DATASET_REGISTRY:
            download_dataset(name, full=args.full)

    print("=" * 60)
    print("全部资源就绪。本地目录：")
    print("  模型：%s" % os.path.join(BASE_DIR, "model"))
    print("  数据：%s" % os.path.join(BASE_DIR, "data"))
    print("运行实验无需任何环境变量，直接 python run_experiment.py 即可。")


if __name__ == "__main__":
    main()

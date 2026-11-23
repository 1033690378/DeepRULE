# -*- coding: utf-8 -*-
"""
环境自检脚本
============================================================
不依赖任何第三方库即可运行，用于在安装依赖、下载模型前后
逐项核对机器环境与 8B/9B 模型、数据目录的完整性。
用法：
    python check_env.py
输出每项的 PASS（通过）/ WARN（警告）/ FAIL（不通过）。
"""
import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys

# 业务代码实际需要的 Python 包：包名 -> 导入名（大多数两者相同）
REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "peft": "peft",
    "accelerate": "accelerate",
    "bitsandbytes": "bitsandbytes",
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "tqdm": "tqdm",
    "huggingface_hub": "huggingface_hub",
}

# 结果收集，最后统一汇总
results = []


def record(item, status, detail):
    """记录一条检查结果，status 取 PASS/WARN/FAIL。"""
    results.append((item, status, detail))
    print("[%s] %s：%s" % (status, item, detail))


def check_python():
    """检查 Python 版本，建议 3.10–3.13。"""
    ver = sys.version_info
    text = "%d.%d.%d" % (ver.major, ver.minor, ver.micro)
    if (ver.major == 3 and 10 <= ver.minor <= 13):
        record("Python 版本", "PASS", text)
    else:
        record("Python 版本", "WARN", text + "，建议使用 3.10–3.13")


def check_gpu(profile="local"):
    """
    通过 nvidia-smi 检查 GPU 型号、显存与驱动。
    8B/9B 模型 bf16 权重本身约 16–18 GB，训练建议 ≥24 GB 显存；
    低于该值不判 FAIL（本机仍可改代码、做结果分析），只 WARN 提示上云端。
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip()
        name, mem_mb, driver = [x.strip() for x in out.split(",")]
        mem_gb = int(mem_mb.split()[0]) / 1024
        if mem_gb >= 24:
            record("GPU", "PASS", "%s，显存 %.1f GB（%s 档位），驱动 %s，"
                   "可承载 8B/9B 训练" % (name, mem_gb, profile, driver))
        else:
            record("GPU", "WARN", "%s，显存 %.1f GB（%s 档位），驱动 %s；"
                   "8B/9B 训练建议 ≥24 GB（如 RTX 4090 48GB），"
                   "本机仅适合改代码与分析结果"
                   % (name, mem_gb, profile, driver))
    except Exception as exc:
        record("GPU", "FAIL", "未检测到 nvidia-smi：%s" % exc)


def check_packages():
    """逐个检查必需包是否安装，并报告版本。"""
    missing = []
    for pkg_name, import_name in REQUIRED_PACKAGES.items():
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(pkg_name)
            continue
        # 已安装则尝试取版本号
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "未知")
        except Exception:
            ver = "已安装（版本读取失败）"
        record("依赖包 %s" % pkg_name, "PASS", ver)
    if missing:
        record("缺失依赖", "FAIL", "、".join(missing)
               + "；请先执行 pip install -r requirements.txt")


def check_optional_packages():
    """可选包：psutil 让内存监控读数更准；缺失时监控自动回退 /proc，不阻断运行。"""
    if importlib.util.find_spec("psutil") is None:
        record("可选依赖 psutil", "WARN",
               "未安装；内存监控将回退 /proc 读数，建议安装：pip install psutil")
    else:
        import psutil
        record("可选依赖 psutil", "PASS", getattr(psutil, "__version__", "已安装"))


def check_monitor_module():
    """确认内存监控模块在位。"""
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "memory_monitor.py")
    record("内存监控模块", "PASS" if os.path.isfile(path) else "FAIL",
           path + ("" if os.path.isfile(path) else "（缺失，无法生成 memory_monitor.csv）"))


def check_torch_cuda():
    """若 torch 已安装，进一步确认 CUDA 与显卡算力是否可用。"""
    if importlib.util.find_spec("torch") is None:
        return
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        record("torch CUDA", "PASS",
               "可用，%s，算力 sm_%d%d"
               % (torch.cuda.get_device_name(0), cap[0], cap[1]))
        # Blackwell 为 sm_120，需要 cu128 及以上的 torch
        if cap >= (12, 0) and not torch.version.cuda.startswith("12.8") \
                and not torch.version.cuda.startswith("12.9"):
            record("CUDA 版本", "WARN",
                   "torch 自带 CUDA %s，Blackwell 显卡建议 12.8+"
                   % torch.version.cuda)
    else:
        record("torch CUDA", "FAIL", "torch 无法调用 GPU，请检查安装版本")


def check_disk():
    """检查代码所在盘剩余空间：两个模型约 34 GB，建议预留 45 GB。"""
    drive = os.path.splitdrive(os.path.abspath(__file__))[0] + os.sep
    free_gb = shutil.disk_usage(drive).free / 1024**3
    if free_gb >= 45:
        record("磁盘剩余空间", "PASS", "%.1f GB" % free_gb)
    else:
        record("磁盘剩余空间", "WARN",
               "%.1f GB，8B/9B 两模型加数据建议预留 45 GB 以上" % free_gb)


def check_local_assets():
    """检查本地模型目录与 TOFU/DUET/WMDP 数据目录（本项目默认离线读本地）。"""
    base = os.path.dirname(os.path.abspath(__file__))

    # 本地模型：扫描 model/ 下每个子目录，应有 config.json 与至少一个 safetensors
    model_root = os.path.join(base, "model")
    if not os.path.isdir(model_root):
        record("本地模型目录", "FAIL", model_root + "（不存在）")
    else:
        model_dirs = [d for d in os.listdir(model_root)
                      if os.path.isdir(os.path.join(model_root, d))]
        if not model_dirs:
            record("本地模型目录", "FAIL", "model/ 下没有任何模型子目录")
        for name in sorted(model_dirs):
            mdir = os.path.join(model_root, name)
            has_cfg = os.path.isfile(os.path.join(mdir, "config.json"))
            weights = [f for f in os.listdir(mdir)
                       if f.endswith(".safetensors")]
            if has_cfg and weights:
                record("本地模型 %s" % name, "PASS",
                       "%s（%d 个权重文件）" % (mdir, len(weights)))
            else:
                record("本地模型 %s" % name, "FAIL",
                       "缺少 config.json 或 safetensors 权重，请检查下载是否完整")

    # 本地 TOFU：JSONL，选 tofu 基准时必需；缺失仅告警（可改用 duet）
    tofu_dir = os.path.join(base, "data", "TOFU")
    required_json = ["forget01.json", "retain99.json"]
    if not os.path.isdir(tofu_dir):
        record("本地 TOFU 目录", "WARN", tofu_dir + "（不存在；用 --dataset duet 可跳过）")
    else:
        missing = [f for f in required_json
                   if not os.path.isfile(os.path.join(tofu_dir, f))]
        json_count = len([f for f in os.listdir(tofu_dir) if f.endswith(".json")])
        has_full = os.path.isfile(os.path.join(tofu_dir, "full.json"))
        if missing:
            record("本地 TOFU 目录", "WARN", "缺少文件：" + "、".join(missing))
        else:
            record("本地 TOFU 目录", "PASS",
                   "%s（共 %d 个 JSONL）" % (tofu_dir, json_count))
        record("TOFU 全量语料 full.json",
               "PASS" if has_full else "WARN",
               "M0 先验学习必需" if has_full else "缺失，无法做先学会再遗忘")

    # 本地 DUET：parquet，选 duet 基准时必需
    duet_dir = os.path.join(base, "data", "DUET")
    if not os.path.isdir(duet_dir):
        record("本地 DUET 目录", "WARN", duet_dir + "（不存在；用 --dataset tofu 可跳过）")
    else:
        pq_count = len([f for f in os.listdir(duet_dir) if f.endswith(".parquet")])
        record("本地 DUET 目录", "PASS" if pq_count else "WARN",
               "%s（共 %d 个 parquet）" % (duet_dir, pq_count))
        has_full = any(f.startswith("full_") and f.endswith(".parquet")
                       for f in os.listdir(duet_dir))
        record("DUET 全量语料 full_*.parquet",
               "PASS" if has_full else "WARN",
               "M0 先验学习必需" if has_full else "缺失，无法做先学会再遗忘")

    # 本地 WMDP：仅 test 选择题，作为 C6 危险知识评测通道
    wmdp_dir = os.path.join(base, "data", "WMDP")
    if not os.path.isdir(wmdp_dir):
        record("本地 WMDP 目录", "WARN", wmdp_dir + "（不存在；C6 评测将跳过）")
    else:
        subj = [d for d in os.listdir(wmdp_dir)
                if os.path.isdir(os.path.join(wmdp_dir, d))]
        record("本地 WMDP 目录", "PASS" if subj else "WARN",
               "%s（科目：%s）" % (wmdp_dir, "、".join(subj) if subj else "无"))

    # 下载源仅在使用 download_assets.py 时需要，离线运行不依赖，仅提示
    endpoint = os.environ.get("HF_ENDPOINT", "")
    record("HF_ENDPOINT", "PASS",
           endpoint or "未设置；离线运行无需配置，联网下载时可设为 https://hf-mirror.com")


def check_work_dirs():
    """检查业务目录 results、checkpoints 是否就绪。"""
    base = os.path.dirname(os.path.abspath(__file__))
    for sub in ("results", "checkpoints"):
        p = os.path.join(base, sub)
        record("工作目录 %s" % sub,
               "PASS" if os.path.isdir(p) else "WARN",
               p + ("（存在）" if os.path.isdir(p) else "（缺失，运行时会自动创建）"))


def main():
    """主入口：依次执行全部检查并给出汇总。"""
    parser = argparse.ArgumentParser(description="Deep-RULE 2.0 环境自检")
    parser.add_argument("--profile", default="local",
                        choices=["local", "cloud"],
                        help="保留参数：两档训练参数一致，仅用于命令行兼容")
    args = parser.parse_args()

    print("=" * 60)
    print("Deep-RULE 2.0 环境自检（档位：%s）" % args.profile)
    print("操作系统：", platform.platform())
    print("=" * 60)
    check_python()
    check_gpu(profile=args.profile)
    check_packages()
    check_optional_packages()
    check_monitor_module()
    check_torch_cuda()
    check_disk()
    check_local_assets()
    check_work_dirs()
    print("=" * 60)
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_warn = sum(1 for _, s, _ in results if s == "WARN")
    print("汇总：FAIL %d 项，WARN %d 项，PASS %d 项"
          % (n_fail, n_warn, len(results) - n_fail - n_warn))
    if n_fail:
        print("存在不通过项，请先处理 FAIL 再运行实验。")
    elif n_warn:
        print("无硬性问题，WARN 项可按提示按需处理。")
    else:
        print("环境全部就绪，可以开始运行实验。")


if __name__ == "__main__":
    main()

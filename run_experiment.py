# -*- coding: utf-8 -*-
"""
主实验入口
============================================================
按「数据集 × 方法 × 种子」逐个运行：训练 → 五通道评测 → RTT 恢复攻击。
旗舰方法为 DeepRULE（深度拒绝遗忘），其余为对照基线。
所有结果写入 results/ 下的 CSV（metrics / predictions / train_log）。

为防止长进程跨格子累积 CPU 内存（曾在 48GB 内存机器上跑到 121GB
被 OOM 杀掉，退出码 137），每个格子都在独立子进程中运行：
子进程结束即由操作系统回收全部内存，与分配器是否泄漏无关；
父进程只负责调度，各格子的内存监控追加写入同一 memory_monitor.csv。

用法示例：
    # 先跑冒烟测试（几分钟，验证整条流水线）
    python run_experiment.py --smoke
    # 正式跑：全部方法、单种子
    python run_experiment.py --methods all --seeds 1
    # 只跑旗舰方法、两个种子
    python run_experiment.py --methods DeepRULE --seeds 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import torch

from config import (BASE_DIR, Config, LOCAL_MODEL_DIR, learned_model_dir,
                    make_run_dir, set_seed)
from data_pipeline import load_data
from model_utils import clear_memory, load_model_and_tokenizer
from trainer import (METHOD_PRESETS, learn_gate_check, stage_pretrain,
                     train_pipeline)
from evaluator import degradation_guard, evaluate_all, rtt_attack


def parse_args():
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="Deep-RULE 2.0 主实验（8B/9B 模型版）")
    parser.add_argument("--methods", default="all",
                        help="逗号分隔的方法名，或 all；可选："
                        + ",".join(METHOD_PRESETS.keys()))
    parser.add_argument("--seeds", type=int, default=1,
                        help="随机种子个数，取前 N 个内置种子")
    parser.add_argument("--max_samples", type=int, default=40,
                        help="每个划分的最大样本数")
    parser.add_argument("--model", default="",
                        help="覆盖默认模型，可填本地路径或 HF 模型名")
    parser.add_argument("--dataset", default="tofu",
                        choices=["tofu", "duet", "rwku"],
                        help="选择基准数据集：tofu、duet 或 rwku（WMDP 自动作为评测通道）")
    parser.add_argument("--datasets", default="",
                        help="逗号分隔的数据集列表，如 tofu,duet,rwku；填了就按顺序依次跑完，"
                             "并覆盖 --dataset")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟模式：样本与步数全部减半")
    parser.add_argument("--profile", default="local", choices=["local", "cloud"],
                        help="保留参数：两档训练参数完全相同，仅为命令行兼容")
    parser.add_argument("--no_original", action="store_true",
                        help="跳过未训练原始模型的基准评测")
    # ---- 子进程工作模式参数（由父进程自动传入，用户不直接用）----
    parser.add_argument("--worker", action="store_true",
                        help="内部参数：单格子工作进程模式")
    parser.add_argument("--worker_pretrain", action="store_true",
                        help="内部参数：先验学习工作进程模式")
    parser.add_argument("--worker_gate", action="store_true",
                        help="内部参数：M0 学会验收工作进程模式")
    parser.add_argument("--worker_pretrain_epochs", type=int, default=0,
                        help="内部参数：覆盖 M0 轮数，0 表示按自动口径")
    parser.add_argument("--worker_method", default="")
    parser.add_argument("--worker_seed", type=int, default=0)
    parser.add_argument("--worker_dataset", default="")
    parser.add_argument("--run_dir", default="",
                        help="内部参数：父进程已创建的运行目录")
    return parser.parse_args()


def run_one(method, seed, args, run_dir, dataset=None):
    """运行单个「数据集 × 方法 × 种子」格子。"""
    import memory_monitor
    ds_name = dataset or args.dataset
    cfg = Config(smoke=args.smoke, max_samples=args.max_samples,
                 dataset_name=ds_name, hardware_profile=args.profile)
    if args.model:
        cfg.model_name = args.model
    # 构造后若改了模型名，需重新应用规格（层数、LoRA 目标、多模态等）
    cfg.apply_model_spec()
    cfg.method_name = method
    cfg.run_seed = seed
    cfg.run_dir = run_dir
    # 所有格子（含 Original）从先验模型启动，保证“先学会再遗忘”
    cfg.init_model_path = learned_model_dir(
        cfg.model_name, ds_name, args.smoke)
    set_seed(seed)

    # 启动本格子的内存监控，追加写入运行目录同一文件
    memory_monitor.start_monitor(cfg.path("memory_monitor.csv"),
                                 interval=cfg.monitor_interval)
    memory_monitor.set_stage("加载数据")
    # 每次重新加载数据，避免上一个方法写入的标记串味
    bundle = load_data(cfg)

    t0 = time.time()
    if method == "Original":
        # 未训练基准：只评测原始模型，给出遗忘前的起点读数
        memory_monitor.set_stage("加载原始模型")
        model, tokenizer = load_model_and_tokenizer(cfg)
    else:
        model, tokenizer = train_pipeline(cfg, bundle, METHOD_PRESETS[method])

    # 五通道评测（C1–C5）
    memory_monitor.set_stage("评测")
    evaluate_all(model, tokenizer, bundle, cfg)
    # RTT 恢复攻击（放在最后，内部临时挂 LoRA 并会自行清理）
    memory_monitor.set_stage("RTT 攻击")
    rtt_attack(model, tokenizer, bundle, cfg)

    # 退化护栏：Original 自身是基线，无需判定；其余格子相对 Original 比较掉幅
    guard = None
    if method != "Original" and cfg.degradation_guard_enabled:
        memory_monitor.set_stage("退化护栏")
        guard = degradation_guard(cfg)
        if not guard.get("passed"):
            # 标记失败、不计入成功，但不中断后续格子调度
            print("  [退化护栏] 该格判失败：%s 掉幅超阈值（%s），不计入成功"
                  % ("、".join(guard.get("breaches", [])),
                     guard.get("reason") or "相对掉幅 > %.0f%%"
                     % (100 * cfg.degradation_guard_threshold)))
        else:
            print("  [退化护栏] 通过：保留能力与 WMDP 掉幅在阈值内")

    cost = time.time() - t0
    print("[%s | seed=%d] 完成，耗时 %.1f 分钟，峰值显存 %.2f GB"
          % (method, seed, cost / 60,
             torch.cuda.max_memory_allocated() / 1024**3))

    del model, tokenizer
    clear_memory()
    torch.cuda.reset_peak_memory_stats()
    memory_monitor.set_stage("格子结束")
    memory_monitor.stop_monitor()


def write_manifest(run_dir, args):
    """把本次命令行参数与默认配置快照落盘，便于事后追溯与复现。"""
    cfg0 = Config(smoke=args.smoke, max_samples=args.max_samples,
                  hardware_profile=args.profile)
    model_ref = args.model or LOCAL_MODEL_DIR
    cfg0.model_name = model_ref
    cfg0.apply_model_spec()
    payload = {
        "argv": sys.argv,
        "args": vars(args),
        "run_dir": run_dir,
        "default_config": asdict(cfg0),
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def spawn_worker(args, run_dir, method, seed, dataset):
    """启动单格子子进程，结束后操作系统回收其全部内存。"""
    argv = [
        sys.executable,
        os.path.join(BASE_DIR, "run_experiment.py"),
        "--worker",
        "--worker_method", method,
        "--worker_seed", str(seed),
        "--worker_dataset", dataset,
        "--run_dir", run_dir,
        "--profile", args.profile,
        "--max_samples", str(args.max_samples),
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.no_original:
        argv.append("--no_original")
    if args.model:
        argv += ["--model", args.model]
    # 不捕获输出：日志实时透传到当前控制台（PyCharm/终端均可见）
    return subprocess.run(argv, cwd=BASE_DIR).returncode


def run_worker(args):
    """子进程模式：只跑指定的一个格子。"""
    run_one(args.worker_method, args.worker_seed, args,
            args.run_dir, dataset=args.worker_dataset)


def run_pretrain_worker(args):
    """子进程模式：只跑 M0 先验学习，产出 learned model。"""
    import memory_monitor
    ds_name = args.worker_dataset
    cfg = Config(smoke=args.smoke, max_samples=args.max_samples,
                 dataset_name=ds_name, hardware_profile=args.profile)
    if args.model:
        cfg.model_name = args.model
    cfg.apply_model_spec()
    # 验收不达标后的重训：父进程显式指定轮数，覆盖自动口径
    if args.worker_pretrain_epochs > 0:
        cfg.pretrain_epochs = args.worker_pretrain_epochs
    cfg.run_dir = args.run_dir
    set_seed(42)
    memory_monitor.start_monitor(cfg.path("memory_monitor.csv"),
                                 interval=cfg.monitor_interval)
    stage_pretrain(cfg)
    memory_monitor.set_stage("先验学习结束")
    memory_monitor.stop_monitor()


def run_gate_worker(args):
    """子进程模式：对 learned model 做学会验收，结果写 JSON 供父进程判定。"""
    import memory_monitor
    ds_name = args.worker_dataset
    cfg = Config(smoke=args.smoke, max_samples=args.max_samples,
                 dataset_name=ds_name, hardware_profile=args.profile)
    if args.model:
        cfg.model_name = args.model
    cfg.apply_model_spec()
    cfg.run_dir = args.run_dir
    cfg.init_model_path = learned_model_dir(
        cfg.model_name, ds_name, args.smoke)
    set_seed(42)
    memory_monitor.start_monitor(cfg.path("memory_monitor.csv"),
                                 interval=cfg.monitor_interval)
    memory_monitor.set_stage("学会验收")
    bundle = load_data(cfg)
    _, result = learn_gate_check(cfg, bundle)
    memory_monitor.set_stage("验收结束")
    memory_monitor.stop_monitor()
    gate_path = cfg.path("learn_gate_%s.json" % ds_name)
    with open(gate_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("验收结果已写入：", gate_path)


def spawn_pretrain(args, run_dir, dataset, epochs=0):
    """启动先验学习子进程；epochs>0 时显式指定轮数（验收重训用）。"""
    argv = [
        sys.executable,
        os.path.join(BASE_DIR, "run_experiment.py"),
        "--worker_pretrain",
        "--worker_dataset", dataset,
        "--run_dir", run_dir,
        "--profile", args.profile,
        "--max_samples", str(args.max_samples),
    ]
    if epochs > 0:
        argv += ["--worker_pretrain_epochs", str(epochs)]
    if args.smoke:
        argv.append("--smoke")
    if args.model:
        argv += ["--model", args.model]
    return subprocess.run(argv, cwd=BASE_DIR).returncode


def spawn_gate(args, run_dir, dataset):
    """启动学会验收子进程，返回退出码（结果 JSON 由父进程读取）。"""
    argv = [
        sys.executable,
        os.path.join(BASE_DIR, "run_experiment.py"),
        "--worker_gate",
        "--worker_dataset", dataset,
        "--run_dir", run_dir,
        "--profile", args.profile,
        "--max_samples", str(args.max_samples),
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.model:
        argv += ["--model", args.model]
    return subprocess.run(argv, cwd=BASE_DIR).returncode


def main():
    """主循环（调度进程）。"""
    args = parse_args()

    # 子进程模式：跑完一个格子即退出，内存随进程归还
    if args.worker:
        run_worker(args)
        return

    # 先验学习子进程模式
    if args.worker_pretrain:
        run_pretrain_worker(args)
        return

    # 学会验收子进程模式
    if args.worker_gate:
        run_gate_worker(args)
        return

    if args.methods == "all":
        methods = list(METHOD_PRESETS.keys())
    else:
        methods = [m.strip() for m in args.methods.split(",")]
    # 原始模型基准放在最前面，方便对比
    if not args.no_original:
        methods = ["Original"] + methods
    seeds = [42 + i for i in range(args.seeds)]
    # 支持一次顺序跑多个数据集；--datasets 优先于 --dataset
    if args.datasets:
        datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset]

    # 本次运行的独立结果目录：模型-方法-种子数 时间戳
    model_ref = args.model or LOCAL_MODEL_DIR
    run_dir = make_run_dir(model_ref, args.methods, args.seeds)
    write_manifest(run_dir, args)
    print("待运行方法：", methods, "种子：", seeds, "数据集：", datasets)
    print("本次运行结果目录：", run_dir)
    total = len(datasets) * len(methods) * len(seeds)
    done = 0
    # 闸门参数来自 Config（冒烟口径会自动覆盖），不是命令行参数
    gate_cfg = Config(smoke=args.smoke, hardware_profile=args.profile)
    for ds in datasets:
        print("=== 数据集：%s ===" % ds)
        learned_dir = learned_model_dir(model_ref, ds, args.smoke)
        # 自动口径下的基础轮数（与 stage_pretrain 保持一致）
        base_epochs = 1 if ds == "duet" else 3
        gate_passed = False

        # M0 + 学会验收：不达标自动加 2 轮重训，最多尝试 learn_gate_max_attempts 次
        for attempt in range(1, gate_cfg.learn_gate_max_attempts + 1):
            if not os.path.isdir(learned_dir):
                ep = 0 if attempt == 1 else base_epochs + 2 * (attempt - 1)
                print(">>> 第 %d 次 M0（轮数 %s）：%s"
                      % (attempt, "自动" if ep == 0 else ep, learned_dir))
                code = spawn_pretrain(args, run_dir, ds, epochs=ep)
                if code != 0:
                    print("先验学习失败（退出码 %d）：数据集 %s，已停止调度，"
                          "结果保留在：%s" % (code, ds, run_dir))
                    sys.exit(code)
            elif attempt == 1:
                print("先验模型已存在，跳过 M0：", learned_dir)

            if not gate_cfg.learn_gate_enabled:
                gate_passed = True
                print("学会验收已关闭，直接放行")
                break
            code = spawn_gate(args, run_dir, ds)
            if code != 0:
                print("验收进程失败（退出码 %d）：数据集 %s，已停止调度"
                      % (code, ds))
                sys.exit(code)
            with open(os.path.join(run_dir, "learn_gate_%s.json" % ds),
                      encoding="utf-8") as f:
                gate_result = json.load(f)
            if gate_result["passed"]:
                gate_passed = True
                print(">>> 学会验收通过，正确率 %.3f" % gate_result["acc"])
                break
            print(">>> 验收未通过（正确率 %.3f < 阈值 %.2f）"
                  % (gate_result["acc"], gate_result["threshold"]))
            # 重训前移除旧 learned model，保证新轮数完整生效
            if attempt < gate_cfg.learn_gate_max_attempts:
                import shutil
                shutil.rmtree(learned_dir, ignore_errors=True)

        if not gate_passed:
            print("学会验收 %d 次尝试仍未达标，已停止调度。可将 "
                  "learn_gate_enabled 设为 False 强制放行，或调整阈值/轮数。"
                  % gate_cfg.learn_gate_max_attempts)
            sys.exit(2)

        for method in methods:
            for seed in seeds:
                done += 1
                print(">>> 格子 %d/%d：%s | %s | seed=%d"
                      % (done, total, ds, method, seed))
                code = spawn_worker(args, run_dir, method, seed, ds)
                if code != 0:
                    # 已完成格子的 CSV 均已落盘，停止调度并明确报告失败点
                    print("格子失败（退出码 %d）：%s | %s | seed=%d，"
                          "已停止后续调度，已完成结果保留在：%s"
                          % (code, ds, method, seed, run_dir))
                    sys.exit(code)
    print("全部实验结束，本次结果位于：", run_dir)


if __name__ == "__main__":
    main()

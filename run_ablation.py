# -*- coding: utf-8 -*-
"""
消融实验入口
============================================================
以旗舰方法 DeepRULE（深度拒绝遗忘）为基准，逐一关闭其核心
组件（leave-one-out），检验每个组件是否真的承担它声称的功能：
    -M1 不做参数层密度深删；-M2 不做关键样本加权；
    -M8 不做拒绝引导；-Abs 不做上位词回退；-M6 不做 GRPO 修复。
结果追加写入 results/metrics.csv，方法名以 ABL_ 开头，便于
后期与主实验区分和画图。

与主实验入口一致：每个格子在独立子进程中运行，进程结束即
回收全部 CPU 内存，避免长任务 OOM；监控写入 memory_monitor.csv。

用法示例：
    python run_ablation.py --smoke
    python run_ablation.py --variants all --seeds 1
"""
import argparse
import copy
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
from model_utils import clear_memory
from trainer import learn_gate_check, stage_pretrain, train_pipeline
from evaluator import evaluate_all, rtt_attack

# DeepRULE 完整组件开关
DEEPRULE_PRESET = {"M1": True, "M2": True, "M8": True,
                   "Abs": True, "M6": True}


def build_variants():
    """构造消融变体字典，每个变体都是独立的开关副本。"""
    def without(module):
        preset = copy.deepcopy(DEEPRULE_PRESET)
        preset[module] = False
        return preset

    return {
        "ABL_FULL": copy.deepcopy(DEEPRULE_PRESET),
        "ABL_-M1":  without("M1"),
        "ABL_-M2":  without("M2"),
        "ABL_-M8":  without("M8"),
        "ABL_-Abs": without("Abs"),
        "ABL_-M6":  without("M6"),
    }


def parse_args():
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="Deep-RULE 2.0 消融实验（DeepRULE 组件级）")
    parser.add_argument("--variants", default="all",
                        help="逗号分隔的变体名，或 all")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", default="local", choices=["local", "cloud"],
                        help="保留参数：两档训练参数完全相同，仅为命令行兼容")
    parser.add_argument("--model", default="",
                        help="本地模型目录或 HF 仓库 ID；留空则用 config.py 的默认值")
    parser.add_argument("--dataset", default="tofu", choices=["tofu", "duet"],
                        help="选择基准数据集：tofu 或 duet")
    parser.add_argument("--datasets", default="",
                        help="逗号分隔的数据集列表，如 tofu,duet；填了就按顺序依次跑完，"
                             "并覆盖 --dataset")
    # ---- 子进程工作模式参数（由父进程自动传入）----
    parser.add_argument("--worker", action="store_true",
                        help="内部参数：单格子工作进程模式")
    parser.add_argument("--worker_pretrain", action="store_true",
                        help="内部参数：先验学习工作进程模式")
    parser.add_argument("--worker_gate", action="store_true",
                        help="内部参数：M0 学会验收工作进程模式")
    parser.add_argument("--worker_pretrain_epochs", type=int, default=0,
                        help="内部参数：覆盖 M0 轮数，0 表示按自动口径")
    parser.add_argument("--worker_variant", default="")
    parser.add_argument("--worker_seed", type=int, default=0)
    parser.add_argument("--worker_dataset", default="")
    parser.add_argument("--run_dir", default="",
                        help="内部参数：父进程已创建的运行目录")
    return parser.parse_args()


def run_variant(name, preset, seed, args, run_dir, dataset=None):
    """运行单个消融变体。"""
    import memory_monitor
    ds_name = dataset or args.dataset
    cfg = Config(smoke=args.smoke, max_samples=args.max_samples,
                 dataset_name=ds_name, hardware_profile=args.profile)
    if args.model:
        # 命令行指定了模型（本地模型或仓库 ID）就覆盖默认模型
        cfg.model_name = args.model
    # 构造后若改了模型名，需重新应用规格（层数、LoRA 目标、多模态等）
    cfg.apply_model_spec()
    cfg.method_name = name
    cfg.run_seed = seed
    cfg.run_dir = run_dir
    # 所有消融变体从先验模型启动
    cfg.init_model_path = learned_model_dir(
        cfg.model_name, ds_name, args.smoke)
    set_seed(seed)

    # 启动本格子的内存监控，追加写入运行目录同一文件
    memory_monitor.start_monitor(cfg.path("memory_monitor.csv"),
                                 interval=cfg.monitor_interval)
    memory_monitor.set_stage("加载数据")
    bundle = load_data(cfg)

    t0 = time.time()
    # 所有变体统一走 DeepRULE 标准三阶段流程，仅组件开关不同
    model, tokenizer = train_pipeline(cfg, bundle, preset)

    memory_monitor.set_stage("评测")
    evaluate_all(model, tokenizer, bundle, cfg)
    memory_monitor.set_stage("RTT 攻击")
    rtt_attack(model, tokenizer, bundle, cfg)

    print("[%s | seed=%d] 完成，耗时 %.1f 分钟"
          % (name, seed, (time.time() - t0) / 60))
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


def spawn_worker(args, run_dir, name, seed, dataset):
    """启动单格子子进程，结束后操作系统回收其全部内存。"""
    argv = [
        sys.executable,
        os.path.join(BASE_DIR, "run_ablation.py"),
        "--worker",
        "--worker_variant", name,
        "--worker_seed", str(seed),
        "--worker_dataset", dataset,
        "--run_dir", run_dir,
        "--profile", args.profile,
        "--max_samples", str(args.max_samples),
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.model:
        argv += ["--model", args.model]
    # 日志实时透传，不捕获
    return subprocess.run(argv, cwd=BASE_DIR).returncode


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
        os.path.join(BASE_DIR, "run_ablation.py"),
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
        os.path.join(BASE_DIR, "run_ablation.py"),
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
    """消融主循环（调度进程）。"""
    args = parse_args()

    # 子进程模式：按变体名重建开关，跑完一个格子即退出
    if args.worker:
        variants = build_variants()
        run_variant(args.worker_variant, variants[args.worker_variant],
                    args.worker_seed, args, args.run_dir,
                    dataset=args.worker_dataset)
        return

    # 先验学习子进程模式
    if args.worker_pretrain:
        run_pretrain_worker(args)
        return

    # 学会验收子进程模式
    if args.worker_gate:
        run_gate_worker(args)
        return

    variants = build_variants()
    if args.variants == "all":
        names = list(variants.keys())
    else:
        names = [v.strip() for v in args.variants.split(",")]
    seeds = [42 + i for i in range(args.seeds)]
    if args.datasets:
        datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = [args.dataset]

    # 本次运行的独立结果目录：模型-变体标签-种子数 时间戳
    model_ref = args.model or LOCAL_MODEL_DIR
    run_dir = make_run_dir(model_ref, args.variants, args.seeds)
    write_manifest(run_dir, args)
    print("本次运行结果目录：", run_dir)
    total = len(datasets) * len(names) * len(seeds)
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
            if attempt < gate_cfg.learn_gate_max_attempts:
                import shutil
                shutil.rmtree(learned_dir, ignore_errors=True)

        if not gate_passed:
            print("学会验收 %d 次尝试仍未达标，已停止调度。可将 "
                  "learn_gate_enabled 设为 False 强制放行，或调整阈值/轮数。"
                  % gate_cfg.learn_gate_max_attempts)
            sys.exit(2)

        for name in names:
            for seed in seeds:
                done += 1
                print(">>> 格子 %d/%d：%s | %s | seed=%d"
                      % (done, total, ds, name, seed))
                code = spawn_worker(args, run_dir, name, seed, ds)
                if code != 0:
                    print("格子失败（退出码 %d）：%s | %s | seed=%d，"
                          "已停止后续调度，已完成结果保留在：%s"
                          % (code, ds, name, seed, run_dir))
                    sys.exit(code)
    print("消融实验结束，本次结果位于：", run_dir)


if __name__ == "__main__":
    main()

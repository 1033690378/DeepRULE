# -*- coding: utf-8 -*-
"""
results/ 实验结果分析脚本
============================================================
自动扫描 results/ 下的每个运行目录（含 metrics.csv），逐个生成
分析图片、指标汇总表与文字摘要，各次实验结果分开保存到
check/<运行目录名>/ 下。

数据集不再写死：从运行目录的全部文件（metrics.csv、data_splits_*.csv、
train_log_*.csv、density_*.csv、learn_gate_*.json、rtt_attack.csv、
predictions_*.csv）识别数据集；某数据集即使被学会验收闸门拦下、没有
metrics，也会在摘要中给出闸门与数据划分说明，而不是被静默忽略。

生成内容：
    01_核心指标_<数据集>.png   准确率类 / 行为安全类指标分组柱状图
    02_WMDP准确率_<数据集>.png  WMDP 三科选择题准确率
    03_RTT恢复曲线_<数据集>.png RTT 各微调比例下的恢复准确率
    04_GRPO训练曲线_<数据集>.png 损失与平均奖励曲线
    05_M1密度_<数据集>.png      每层知识密度，选中层高亮
    06_边界余量_<数据集>.png    M2 余量 φ 与左右极限概率差
    07_内存监控.png             进程内存/swap 与 GPU 显存随时间曲线
    指标汇总表_<数据集>.csv     多种子自动均值/标准差后的透视表
    结果分析摘要.md             运行参数、关键读数与自动诊断

用法：
    python check/analyze_results.py
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 路径与全局样式
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECK_DIR = os.path.join(BASE_DIR, "check")

# 复用项目自身的评测判据，保证形态分类与训练/评测口径一致；
# 若环境缺少 torch 等依赖则降级为仅出图（不做预测形态分类）
sys.path.insert(0, BASE_DIR)
try:
    from config import Config
    from data_pipeline import abstraction_markers
    from evaluator import (_PREDICATE_RE, fuzzy_entity_match,
                           hallucinated_proper_name, is_abstraction_response,
                           is_refusal)
    ANALYZER_CFG = Config()
    ANALYZER_MARKERS = abstraction_markers()
    PROJECT_IMPORTS_OK = True
except Exception as exc:  # pragma: no cover - 环境缺依赖时的兜底
    ANALYZER_CFG = None
    ANALYZER_MARKERS = None
    PROJECT_IMPORTS_OK = False
    print("提示：项目评测模块导入失败，跳过预测形态分类：", exc)

# 中文图表字体（本机已装 SimHei），负号正常显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

# 指标分组（与 evaluator 输出的 metric 名对齐）
ACC_METRICS = ["forget_acc", "retain_acc", "boundary_acc", "dup_acc"]
BEHAVIOR_METRICS = ["entity_leak_rate", "entity_leak_substantive_rate",
                    "abstraction_rate", "over_refusal_rate",
                    "under_refusal_rate", "boundary_sharpness_S"]
WMDP_METRICS = ["wmdp_bio_acc", "wmdp_chem_acc", "wmdp_cyber_acc"]
WMDP_LABELS = {"wmdp_bio_acc": "生物", "wmdp_chem_acc": "化学",
               "wmdp_cyber_acc": "网络"}

# 数据集展示的首选顺序（与 config 支持的数据集对齐）；未登记的新数据集
# 会按字母序追加在后面，保证以后再接数据集也不用改这里的主流程
DATASET_ORDER = ["tofu", "duet", "rwku"]

# 预测回答的行为形态（固定堆叠顺序）与配色
MORPH_CATEGORIES = ["干净抽象", "拒答", "精确漏答", "拼写变体",
                    "幻觉具体名", "其他具体"]
MORPH_COLORS = {
    "干净抽象": "#2ca02c", "拒答": "#7f7f7f", "精确漏答": "#d62728",
    "拼写变体": "#ff7f0e", "幻觉具体名": "#9467bd", "其他具体": "#1f77b4"}
# 自动判定遗忘“达标”的阈值（与 config 护栏/闸门口径对齐）
PASS_SUBSTANTIVE_MAX = 0.15
PASS_ABSTRACTION_MIN = 0.60


def ordered_methods(df, col="method"):
    """按 CSV 中首次出现的顺序返回方法名（Original 自然排在最前）。"""
    seen = []
    for m in df[col]:
        if m not in seen:
            seen.append(m)
    return seen


def mean_std(sub, methods):
    """对同一方法聚合多种子，返回均值、标准差序列。"""
    g = sub.groupby("method")["value"]
    mean = g.mean().reindex(methods)
    std = g.std(ddof=0).reindex(methods).fillna(0.0)
    return mean, std


def grouped_bar(ax, metrics, metrics_df, methods, title):
    """x=方法，每组若干指标柱，多种子用误差棒。"""
    n = len(metrics)
    width = 0.8 / n
    x = np.arange(len(methods))
    for i, metric in enumerate(metrics):
        sub = metrics_df[metrics_df["metric"] == metric]
        if sub.empty:
            continue
        mean, std = mean_std(sub, methods)
        ax.bar(x + (i - (n - 1) / 2) * width, mean.values, width,
               yerr=std.values, capsize=3, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)


# ------------------------------------------------------------
# 图 1：核心指标
# ------------------------------------------------------------
def fig_core_metrics(metrics_df, out_dir, dataset):
    sub = metrics_df[metrics_df["dataset"] == dataset]
    methods = ordered_methods(sub)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    grouped_bar(axes[0], ACC_METRICS, sub, methods,
                "准确率类指标（多种子柱高为均值、误差棒为标准差）")
    grouped_bar(axes[1], BEHAVIOR_METRICS, sub, methods,
                "行为与安全类指标")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_核心指标_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 2：WMDP 三科准确率
# ------------------------------------------------------------
def fig_wmdp(metrics_df, out_dir, dataset):
    sub = metrics_df[(metrics_df["dataset"] == dataset)
                     & (metrics_df["metric"].isin(WMDP_METRICS))]
    if sub.empty:
        return
    methods = ordered_methods(sub)
    width = 0.25
    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, metric in enumerate(WMDP_METRICS):
        msub = sub[sub["metric"] == metric]
        mean, std = mean_std(msub, methods)
        ax.bar(x + (i - 1) * width, mean.values, width,
               yerr=std.values, capsize=3, label=WMDP_LABELS[metric])
    ax.axhline(0.25, color="gray", ls="--", lw=1, label="四选一随机基线 0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("WMDP 三科选择题准确率（%s）" % dataset)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_WMDP准确率_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 3：RTT 恢复曲线
# ------------------------------------------------------------
def fig_rtt(run_dir, out_dir, dataset):
    path = os.path.join(run_dir, "rtt_attack.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset]
    if df.empty:
        return
    methods = ordered_methods(df)
    ratios = sorted(df["rtt_ratio"].unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        sub = df[df["method"] == m].groupby("rtt_ratio")["value"].mean()
        ax.plot(ratios, sub.reindex(ratios).values, marker="o",
                color=cmap(i % 10), label=m)
    ax.set_xlabel("RTT 微调比例")
    ax.set_ylabel("恢复准确率 rtt_recover_acc")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("RTT 恢复攻击曲线（%s），越低越抗撬" % dataset)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_RTT恢复曲线_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 4：GRPO 训练曲线
# ------------------------------------------------------------
def fig_train_log(run_dir, out_dir, dataset):
    path = os.path.join(run_dir, "train_log_%s.csv" % dataset)
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    methods = ordered_methods(df)
    # 三面板：真实损失 / 平均奖励 / 组内奖励标准差（判断是否有学习信号）
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(methods):
        sub = df[df["method"] == m].sort_values("step")
        color = cmap(i % 10)
        axes[0].plot(sub["step"], sub["loss"], marker=".", color=color, label=m)
        axes[1].plot(sub["step"], sub["mean_reward"], marker="o",
                     ms=3, color=color, label=m)
        if "reward_std" in sub.columns:
            axes[2].plot(sub["step"], sub["reward_std"], marker="s",
                         ms=3, color=color, label=m)
    axes[0].set_title("GRPO 真实总损失曲线（%s）" % dataset)
    axes[0].set_ylabel("loss")
    axes[1].set_title("平均奖励曲线（%s）" % dataset)
    axes[1].set_ylabel("mean_reward")
    axes[2].set_title("组内奖励标准差（%s，趋近 0 表示策略项无梯度）" % dataset)
    axes[2].set_ylabel("reward_std")
    axes[2].set_xlabel("step")
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_GRPO训练曲线_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 5：M1 每层知识密度
# ------------------------------------------------------------
def fig_density(run_dir, out_dir, dataset):
    path = os.path.join(run_dir, "density_%s.csv" % dataset)
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if df.empty:
        return
    methods = ordered_methods(df)
    fig, axes = plt.subplots(len(methods), 1,
                             figsize=(12, 3.2 * len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sub = df[df["method"] == m].sort_values("layer")
        colors = ["#d62728" if s == 1 else "#1f77b4"
                  for s in sub["selected"]]
        ax.bar(sub["layer"], sub["K_l"], color=colors)
        ax.set_title("%s：每层知识密度 K_l（红色=选中训练层）" % m)
        ax.set_ylabel("K_l")
    axes[-1].set_xlabel("层编号 layer")
    axes[-1].set_xticks(sorted(df["layer"].unique()))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_M1密度_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 6：M2 边界余量与左右极限对
# ------------------------------------------------------------
def fig_margins(run_dir, out_dir, dataset):
    margin_files = sorted(glob.glob(os.path.join(
        run_dir, "margins_%s_*.csv" % dataset)))
    lr_files = sorted(glob.glob(os.path.join(
        run_dir, "left_right_pairs_%s_*.csv" % dataset)))
    if not margin_files and not lr_files:
        return
    nrows = max(len(margin_files), len(lr_files))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.4 * nrows),
                             squeeze=False)
    for i in range(nrows):
        if i < len(margin_files):
            df = pd.read_csv(margin_files[i])
            axes[i][0].bar(range(len(df)), df["phi"].values, color="#ff7f0e")
            axes[i][0].set_title("边界余量 φ：%s"
                                 % os.path.basename(margin_files[i])[:-4],
                                 fontsize=10)
            axes[i][0].set_ylabel("phi")
            axes[i][0].axhline(0, color="black", lw=0.8)
        if i < len(lr_files):
            df = pd.read_csv(lr_files[i])
            colors = ["#2ca02c" if v else "#9467bd"
                      for v in df["gap_ge_delta"]]
            axes[i][1].bar(range(len(df)), df["prob_gap"].values,
                           color=colors)
            axes[i][1].set_title("左右极限概率差（绿=达到 δ 阈值）：%s"
                                 % os.path.basename(lr_files[i])[:-4],
                                 fontsize=10)
            axes[i][1].set_ylabel("prob_gap")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_边界余量_%s.png" % dataset))
    plt.close(fig)


# ------------------------------------------------------------
# 图 7：内存与显存监控曲线（整个运行目录一份，不按数据集分）
# ------------------------------------------------------------
def fig_memory(run_dir, out_dir):
    """读 memory_monitor.csv，画 CPU 内存/swap 与 GPU 显存时间曲线。"""
    path = os.path.join(run_dir, "memory_monitor.csv")
    if not os.path.exists(path):
        return None
    mon = pd.read_csv(path)
    if mon.empty:
        return None
    t = mon["elapsed_s"] / 60.0
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # 上：进程 RSS（GB）、系统可用内存、swap 已用
    ax = axes[0]
    ax.plot(t, mon["pid_rss_mb"] / 1024.0, color="#d62728",
            label="进程 RSS（GB）")
    ax.plot(t, mon["ram_available_gb"], color="#1f77b4",
            label="系统可用内存（GB）")
    ax.plot(t, mon["swap_used_gb"], color="#ff7f0e",
            label="swap 已用（GB）")
    ax.set_title("CPU 内存监控（进程被 OOM 杀掉时，曲线末端即崩溃前状态）")
    ax.set_ylabel("GB")
    ax.legend(fontsize=9, ncol=3)

    # 下：GPU 已用/预留/总量
    ax = axes[1]
    ax.plot(t, mon["gpu_alloc_gb"], color="#2ca02c", label="显存已用 alloc")
    ax.plot(t, mon["gpu_reserved_gb"], color="#9467bd",
            label="显存预留 reserved")
    gpu_total = mon["gpu_total_gb"].max()
    if gpu_total > 0:
        ax.axhline(gpu_total, color="gray", ls="--", lw=1,
                   label="显存总量 %.1f GB" % gpu_total)
    ax.set_title("GPU 显存监控")
    ax.set_xlabel("运行时间（分钟）")
    ax.set_ylabel("GB")
    ax.legend(fontsize=9, ncol=3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_内存监控.png"))
    plt.close(fig)
    return mon


# ------------------------------------------------------------
# 指标汇总表（多种子均值，附标准差）
# ------------------------------------------------------------
def write_summary_table(metrics_df, out_dir, dataset):
    sub = metrics_df[metrics_df["dataset"] == dataset]
    pivot_mean = sub.pivot_table(index="method", columns="metric",
                                 values="value", aggfunc="mean")
    pivot_std = sub.pivot_table(index="method", columns="metric",
                                values="value", aggfunc=lambda x: x.std(ddof=0))
    out = pivot_mean.copy()
    for col in pivot_mean.columns:
        out[col] = pivot_mean[col].map(lambda v: "%0.4f" % v)
        if pivot_std[col].fillna(0).max() > 0:
            for idx in pivot_mean.index:
                out.loc[idx, col] = "%0.4f±%0.4f" % (
                    pivot_mean.loc[idx, col], pivot_std.loc[idx, col])
    out.to_csv(os.path.join(out_dir, "指标汇总表_%s.csv" % dataset),
               encoding="utf-8-sig")
    return pivot_mean


# ------------------------------------------------------------
# 文字摘要（自动诊断，只陈述数据事实）
# ------------------------------------------------------------
def write_summary_md(run_dir, out_dir, run_name, datasets, pivots, mon=None,
                     gates=None, metric_datasets=None, degradation=None,
                     morphs=None, scorecards=None, forget_gates=None,
                     post_gates=None, failed_cells=None):
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    gates = gates or {}
    morphs = morphs or {}
    scorecards = scorecards or {}
    forget_gates = forget_gates or {}
    post_gates = post_gates or {}
    metric_datasets = set(metric_datasets or [])
    # 护栏失败格子：优先用外部传入，否则从 degradation 现算
    if failed_cells is None:
        failed_cells = set()
        if degradation is not None and not degradation.empty:
            bad = degradation[degradation["breached"] == 1]
            for _, r in bad.iterrows():
                failed_cells.add((str(r["dataset"]), str(r["method"])))
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    args = manifest.get("args", {})
    lines = []
    # 章节动态编号：按实际出现的小节顺序编号，避免某小节缺失时跳号
    cn_num = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
              "十一", "十二", "十三", "十四", "十五", "十六"]
    sec = {"n": 0}

    def section(title):
        sec["n"] += 1
        lines.append("## %s、%s" % (cn_num[sec["n"] - 1], title))
        lines.append("")

    lines.append("# 结果分析摘要：%s" % run_name)
    lines.append("")
    section("运行参数（来自 run_manifest.json）")
    lines.append("- 冒烟模式 smoke：%s" % args.get("smoke"))
    lines.append("- 方法 methods：%s" % args.get("methods"))
    lines.append("- 种子数 seeds：%s" % args.get("seeds"))
    lines.append("- 数据集：%s" % (args.get("datasets") or args.get("dataset")))
    lines.append("- 识别到的数据集：%s" % "、".join(datasets))
    lines.append("")
    if args.get("smoke"):
        lines.append("> 注意：本次为冒烟运行，样本与步数减半，读数仅用于验证流水线，"
                     "不能作为方法有效性的论文证据。")
        lines.append("")

    # 先列出没有 metrics 的数据集（如被学会验收闸门拦下、未进入训练/评测）
    missing = [d for d in datasets if d not in metric_datasets]
    if missing:
        lines.append("> 数据集 %s 未产生 metrics.csv 记录（可能在学会验收闸门处"
                     "被拦下、未进入训练与评测），下列仅给出可用的闸门与数据划分信息，"
                     "不参与方法有效性比较。" % "、".join(missing))
        lines.append("")

    # ---------- 学会验收闸门 ----------
    if any(gates.get(d) for d in datasets):
        section("学会验收闸门（learn_gate）")
        for ds in datasets:
            g = gates.get(ds)
            if not g:
                continue
            lines.append("- [%s] 综合正确率 %0.3f / 阈值 %0.2f / 探针 %d 条 / %s"
                         % (ds, g.get("acc", np.nan), g.get("threshold", np.nan),
                            g.get("n_probe", 0),
                            "通过" if g.get("passed") else "未通过"))
        lines.append("")

    # ---------- 退化护栏 ----------
    if degradation is not None and not degradation.empty:
        section("退化护栏（degradation_guard，相对 Original 掉幅）")
        if failed_cells:
            lines.append("- 下列格子被判失败、不计入成功（标记于后面读数表）：")
            for ds, method in sorted(failed_cells):
                breaches = sorted(set(degradation[
                    (degradation["dataset"] == ds)
                    & (degradation["method"] == method)
                    & (degradation["breached"] == 1)]["metric"]))
                lines.append("  - [%s] %s：%s 掉幅超阈值"
                             % (ds, method, "、".join(breaches)))
        else:
            lines.append("- 全部格子护栏通过，保留能力与 WMDP 掉幅均在阈值内。")
        lines.append("")

    section("关键读数（多种子已取均值，护栏失败方法已标注）")
    for ds in datasets:
        if ds not in pivots:
            lines.append("")
            lines.append("### 数据集：%s（无指标，跳过读数表）" % ds)
            continue
        pv = pivots[ds]
        lines.append("")
        lines.append("### 数据集：%s" % ds)
        cols = [c for c in ["forget_acc", "retain_acc", "boundary_acc",
                            "entity_leak_rate", "entity_leak_substantive_rate",
                            "abstraction_rate", "over_refusal_rate",
                            "under_refusal_rate", "boundary_sharpness_S",
                            "mia_auc", "wmdp_bio_acc", "wmdp_chem_acc",
                            "wmdp_cyber_acc"] if c in pv.columns]
        head = "| 方法 | " + " | ".join(cols) + " |"
        sep = "|" + "---|" * (len(cols) + 1)
        lines.append(head)
        lines.append(sep)
        for idx in pv.index:
            vals = ["%0.3f" % pv.loc[idx, c] for c in cols]
            label = idx
            if (ds, idx) in failed_cells:
                label = idx + "（护栏失败）"
            lines.append("| %s | %s |" % (label, " | ".join(vals)))

    lines.append("")
    section("自动诊断（基于阈值的事实陈述）")
    for ds in datasets:
        if ds not in pivots:
            # 无指标数据集：结合闸门状态给事实说明
            g = gates.get(ds)
            if g and not g.get("passed"):
                lines.append("- [%s] 学会验收未通过（正确率 %0.3f < 阈值 %0.2f），"
                             "调度在进入遗忘前停止，本数据集无方法结论。"
                             % (ds, g.get("acc", np.nan), g.get("threshold", np.nan)))
            continue
        pv = pivots[ds]
        if "retain_acc" in pv.columns:
            r = pv["retain_acc"]
            if (r <= 0.02).all():
                lines.append("- [%s] 全部方法 retain_acc≈0：保留集也答不出，"
                             "提示模型整体崩塌或评测判据异常，不能读成精确遗忘。" % ds)
        if "abstraction_rate" in pv.columns:
            a = pv["abstraction_rate"]
            lines.append("- [%s] abstraction_rate 范围 %0.3f～%0.3f：上位词回退"
                         "出现比例很低，需结合判据（要求无泄漏且非拒答）一起看。"
                         % (ds, a.min(), a.max()))
        if "entity_leak_rate" in pv.columns:
            e = pv["entity_leak_rate"]
            if e.std(ddof=0) < 1e-9:
                lines.append("- [%s] entity_leak_rate 各方法恒为 %0.3f："
                             "纯子串判据把拒答句复述实体也算泄漏，属判据假阳性。"
                             % (ds, e.iloc[0]))
        if "entity_leak_substantive_rate" in pv.columns:
            es = pv["entity_leak_substantive_rate"]
            if (es >= 0.5).any():
                lines.append("- [%s] 实质性泄漏率最高 %0.3f：剔除问题回声后仍有方法"
                             "在非拒答回答中带出实体，知识未真正移除。"
                             % (ds, es.max()))
        if "boundary_sharpness_S" in pv.columns:
            s = pv["boundary_sharpness_S"]
            if (s <= 0).all():
                lines.append("- [%s] boundary_sharpness_S 全部 ≤0：遗忘侧与"
                             "边界侧没有拉开可辨的边界，选择性未体现。" % ds)
    # RTT 最大恢复率
    rtt_path = os.path.join(run_dir, "rtt_attack.csv")
    if os.path.exists(rtt_path):
        rtt = pd.read_csv(rtt_path)
        for ds in datasets:
            sub = rtt[rtt["dataset"] == ds]
            if sub.empty:
                continue
            worst = sub.groupby("method")["value"].max().max()
            lines.append("- [%s] RTT 三比例下最高恢复准确率仅 %0.3f，"
                         "整体抗撬；但在模型已崩塌时，撬不动也可能是"
                         "“本来就什么都不会”，需与 retain_acc 联读。"
                         % (ds, worst))

    # ---------- 内存监控读数与诊断 ----------
    lines.append("")
    section("内存与显存监控（来自 memory_monitor.csv）")
    if mon is None:
        lines.append("- 本次运行无 memory_monitor.csv（旧版本运行或监控未启用），"
                     "无法还原内存曲线。")
    else:
        rss_gb = mon["pid_rss_mb"].max() / 1024.0
        min_avail = mon["ram_available_gb"].min()
        swap_max = mon["swap_used_gb"].max()
        gpu_peak = mon["gpu_peak_gb"].max()
        gpu_total = mon["gpu_total_gb"].max()
        lines.append("- 进程 RSS 峰值 %0.2f GB；系统可用内存最低 %0.2f GB；"
                     "swap 使用峰值 %0.2f GB。"
                     % (rss_gb, min_avail, swap_max))
        lines.append("- 显存峰值 %0.2f GB（显存总量 %0.2f GB，占比 %0.1f%%）。"
                     % (gpu_peak, gpu_total,
                        100.0 * gpu_peak / max(gpu_total, 1e-9)))
        if min_avail < 2.0:
            lines.append("- 可用内存一度低于 2 GB，已逼近主机内存 OOM 边界，"
                         "若发生 137 退出可与此处曲线末端对应。")
        if swap_max > 1.0:
            lines.append("- swap 被实际使用（峰值 %0.2f GB），说明物理内存"
                         "已不够用、系统在换页，训练会明显变慢。" % swap_max)
        if gpu_total > 0 and gpu_peak / gpu_total > 0.9:
            lines.append("- 显存峰值超过总量 90%，接近显存 OOM 边界，"
                         "可进一步减小 grpo_group_size 或 grpo_gen_batch。")
        if min_avail >= 2.0 and swap_max <= 1.0:
            lines.append("- 可用内存始终高于 2 GB 且 swap 基本未用，"
                         "主机内存维度未见 OOM 风险。")
    lines.append("")

    # ---- 预测行为形态：直接回答“模型到底怎么答”，弥补指标表看不出的问题 ----
    section("forget 回答行为形态（基于 predictions 逐题分类）")
    if not PROJECT_IMPORTS_OK:
        lines.append("- 项目评测模块未成功导入，本环境跳过形态分类。")
    elif not any(morphs.values()):
        lines.append("- 未找到可用的 predictions.csv，无法做形态分类。")
    else:
        lines.append("- 形态口径与训练/评测一致：干净抽象（理想第三状态）、拒答、"
                     "精确漏答、拼写变体、幻觉具体名、其他具体。")
        lines.append("- 对应堆叠图：08_预测形态_<数据集>.png；明细：预测形态明细_<数据集>.csv。")
        for ds in datasets:
            mp = morphs.get(ds)
            if not mp:
                continue
            lines.append("")
            lines.append("**%s**" % ds)
            lines.append("")
            header = "| 方法 | " + " | ".join(MORPH_CATEGORIES) + \
                " | 平均词数 | 短答率 | 无谓语率 |"
            lines.append(header)
            lines.append("|" + "---|" * (len(MORPH_CATEGORIES) + 4))
            for m in mp["methods"]:
                n = mp["quality"][m]["n"]
                cells = []
                for cat in MORPH_CATEGORIES:
                    c = mp["counts"][m][cat]
                    cells.append("%d(%.0f%%)" % (c, 100.0 * c / max(n, 1)))
                q = mp["quality"][m]
                wmean = np.mean(q["words"]) if q["words"] else 0
                lines.append("| %s | %s | %.1f | %.0f%% | %.0f%% |"
                             % (m, " | ".join(cells), wmean,
                                100.0 * q["short"] / max(n, 1),
                                100.0 * q["no_pred"] / max(n, 1)))
            # 自动点出旗舰方法的形态问题
            for focal in ("DeepRULE", "UNI"):
                if focal not in mp["methods"]:
                    continue
                n = mp["quality"][focal]["n"]
                comp = mp["counts"][focal]
                leak = comp["精确漏答"] + comp["拼写变体"] + comp["幻觉具体名"]
                lines.append("- %s：干净抽象 %0.0f%%、拒答 %0.0f%%、"
                             "各类泄漏合计 %0.0f%%（%d/%d）。"
                             % (focal, 100.0 * comp["干净抽象"] / max(n, 1),
                                100.0 * comp["拒答"] / max(n, 1),
                                100.0 * leak / max(n, 1), leak, n))
    lines.append("")

    # ---- 方法评分卡：透明的辅助排序，便于一眼看出谁最好 ----
    section("方法评分卡（辅助排序，非论文指标）")
    if not scorecards:
        lines.append("- 无可用指标，未生成评分卡。")
    else:
        lines.append("- 综合便利分 = 0.5×抽象率 + 0.3×retain − 0.5×实质泄漏 "
                     "− 0.2×护栏失败；自动达标要求实质泄漏 ≤ %.2f、抽象率 ≥ %.2f 且护栏通过。"
                     % (PASS_SUBSTANTIVE_MAX, PASS_ABSTRACTION_MIN))
        for ds in datasets:
            sc = scorecards.get(ds)
            if sc is None or sc.empty:
                continue
            lines.append("")
            lines.append("**%s**（按综合便利分降序）" % ds)
            lines.append("")
            lines.append("| 方法 | 抽象率 | 实质泄漏率 | retain | 护栏失败 | 综合便利分 | 自动达标 |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in sc.iterrows():
                lines.append("| %s | %.3f | %.3f | %.3f | %d | %.3f | %s |"
                             % (r["method"], r["抽象率"], r["实质泄漏率"],
                                r["retain_acc"], r["护栏失败"],
                                r["综合便利分"], r["自动达标"]))
    lines.append("")

    # ---- 遗忘充分性闸门：当前完全未可视化，这里汇总 pass/boost/proceed ----
    section("遗忘充分性闸门（forget_gate）")
    if not forget_gates:
        lines.append("- 本次运行无 forget_gate 文件（旧版本或闸门未启用）。")
    else:
        for ds, gdf in forget_gates.items():
            if gdf.empty:
                continue
            lines.append("")
            lines.append("**%s**" % ds)
            lines.append("")
            act_counts = gdf["action"].value_counts().to_dict()
            lines.append("- 动作汇总：" + "、".join(
                "%s %d 条" % (k, v) for k, v in sorted(act_counts.items())))
            lines.append("| 实体 | 轮次 | 抽查数 | 直答率 | 阈值 | 动作 |")
            lines.append("|---|---|---|---|---|---|")
            for _, r in gdf.iterrows():
                lines.append("| %s | %s | %s | %.3f | %.2f | %s |"
                             % (r["entity"], r["attempt"], r["n_probe"],
                                r["concrete_rate"], r["threshold"], r["action"]))
            if "boost" in act_counts:
                lines.append("- 出现 boost：说明首轮直答率超阈值，触发了定向抽象补训，"
                             "需结合补训后是否仍 proceed 判断遗忘是否充分。")
    lines.append("")

    # ---- GRPO 后复核：与 GRPO 前闸门同口径，直接看 GRPO 是否造成回退 ----
    section("GRPO 后复核闸门（post_gate）")
    if not post_gates:
        lines.append("- 本次运行无 post_gate 文件（旧版本或复核未启用）。")
    else:
        for ds, gdf in post_gates.items():
            if gdf.empty:
                continue
            lines.append("")
            lines.append("**%s**" % ds)
            lines.append("")
            act_counts = gdf["action"].value_counts().to_dict()
            lines.append("- 动作汇总：" + "、".join(
                "%s %d 条" % (k, v) for k, v in sorted(act_counts.items())))
            lines.append("| 实体 | 抽查数 | 直答率 | 阈值 | 动作 |")
            lines.append("|---|---|---|---|---|")
            for _, r in gdf.iterrows():
                lines.append("| %s | %s | %.3f | %.2f | %s |"
                             % (r["entity"], r["n_probe"],
                                r["concrete_rate"], r["threshold"], r["action"]))
            if "regress" in act_counts:
                lines.append("- 出现 regress：GRPO 后直答率回升，说明 GRPO 把已翻转"
                             "策略拉回具体回答，需对照 GRPO 前 forget_gate 定位回退幅度。")
    lines.append("")

    with open(os.path.join(out_dir, "结果分析摘要.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------
def discover_datasets(run_dir):
    """
    从运行目录的全部文件识别数据集，而不只依赖 metrics.csv。
    覆盖：metrics.csv、data_splits_*.csv、train_log_*.csv、density_*.csv、
    learn_gate_*.json、rtt_attack.csv、predictions_<ds>_seed*.csv。
    这样即使某数据集被学会验收闸门拦下、没有产生 metrics，也能被识别并说明。
    """
    names = set()

    def _read_col(path, col):
        try:
            names.update(pd.read_csv(path)[col].dropna().astype(str).unique())
        except Exception:
            pass

    metrics_path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(metrics_path):
        _read_col(metrics_path, "dataset")
    rtt_path = os.path.join(run_dir, "rtt_attack.csv")
    if os.path.exists(rtt_path):
        _read_col(rtt_path, "dataset")

    # 按固定文件名后缀识别：data_splits_<ds>、train_log_<ds>、density_<ds>
    for prefix in ("data_splits_", "train_log_", "density_"):
        for f in glob.glob(os.path.join(run_dir, prefix + "*.csv")):
            names.add(os.path.basename(f)[len(prefix):-len(".csv")])
    # 学会验收闸门：learn_gate_<ds>.json
    for f in glob.glob(os.path.join(run_dir, "learn_gate_*.json")):
        names.add(os.path.basename(f)[len("learn_gate_"):-len(".json")])
    # 预测明细：predictions_<ds>_<model>_seed<n>.csv；模型名本身含下划线，
    # 因此只取首个不含下划线的数据集 token，避免把模型名吞进来
    for f in glob.glob(os.path.join(run_dir, "predictions_*_seed*.csv")):
        m = re.match(r"predictions_([A-Za-z0-9]+)_.*_seed\d+\.csv",
                     os.path.basename(f))
        if m:
            names.add(m.group(1))

    ordered = [d for d in DATASET_ORDER if d in names]
    ordered += sorted(n for n in names if n not in DATASET_ORDER)
    return ordered


def load_gate(run_dir, dataset):
    """读取某数据集的学会验收闸门结果；不存在则返回 None。"""
    path = os.path.join(run_dir, "learn_gate_%s.json" % dataset)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_degradation(run_dir):
    """读取退化护栏长表 degradation_guard.csv；不存在则返回 None。"""
    path = os.path.join(run_dir, "degradation_guard.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ------------------------------------------------------------
# 预测形态分类（基于 predictions.csv，复用 evaluator 判据）
# ------------------------------------------------------------
def load_id_entity(run_dir, dataset):
    """data_splits 中 id -> entity，供逐题定位目标实体。"""
    path = os.path.join(run_dir, "data_splits_%s.csv" % dataset)
    mapping = {}
    if os.path.exists(path):
        try:
            sp = pd.read_csv(path)
            if "entity" in sp.columns:
                mapping = dict(zip(sp["id"].astype(str), sp["entity"].astype(str)))
        except Exception:
            mapping = {}
    return mapping


def classify_response(resp, entity, cfg, markers):
    """把单条 forget 回答归入一个行为形态类别。"""
    refusal = is_refusal(resp)
    exact, frac, _ = fuzzy_entity_match(resp, entity, cfg)
    present = exact or frac >= float(cfg.fuzzy_match_threshold)
    is_abs = is_abstraction_response(
        resp, int(refusal), int(present), markers,
        entity=entity, cfg=cfg)
    hall = hallucinated_proper_name(resp, entity, cfg)
    if is_abs:
        return "干净抽象"
    if refusal:
        return "拒答"
    if exact:
        return "精确漏答"
    if present:
        return "拼写变体"
    if hall:
        return "幻觉具体名"
    return "其他具体"


def fig_prediction_morphology(run_dir, out_dir, dataset):
    """
    读 predictions.csv 对 forget 回答做形态分类，输出：
      08_预测形态_<ds>.png  各方法形态占比堆叠柱状图；
      预测形态明细_<ds>.csv 各形态计数/占比 + 长度与完整度统计。
    返回 {方法: {形态: 计数}}，无数据返回 None。
    """
    if not PROJECT_IMPORTS_OK:
        return None
    pred_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(pred_path):
        return None
    preds = pd.read_csv(pred_path)
    if preds.empty or "response" not in preds.columns:
        return None
    sub = preds[(preds["dataset"] == dataset)
                & (preds["split"] == "forget")]
    if sub.empty:
        return None
    id_entity = load_id_entity(run_dir, dataset)
    methods = ordered_methods(sub)
    counts = {m: {c: 0 for c in MORPH_CATEGORIES} for m in methods}
    quality = {m: {"n": 0, "words": [], "short": 0, "no_pred": 0}
               for m in methods}
    for _, r in sub.iterrows():
        m = r["method"]
        ent = id_entity.get(str(r["id"]), "")
        resp = str(r["response"])
        cat = classify_response(resp, ent, ANALYZER_CFG, ANALYZER_MARKERS)
        counts[m][cat] += 1
        words = re.findall(r"[a-z0-9]+", resp.lower())
        q = quality[m]
        q["n"] += 1
        q["words"].append(len(words))
        q["short"] += int(len(words) <= 3)
        q["no_pred"] += int(not _PREDICATE_RE.search(resp.lower()))
    # 堆叠占比图
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    for cat in MORPH_CATEGORIES:
        vals = np.array([counts[m][cat] for m in methods], dtype=float)
        ns = np.array([quality[m]["n"] for m in methods], dtype=float)
        frac = np.divide(vals, ns, out=np.zeros_like(vals), where=ns > 0)
        ax.bar(x, frac, bottom=bottoms, color=MORPH_COLORS[cat], label=cat)
        bottoms += frac
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("forget 回答占比")
    ax.set_title("forget 回答行为形态分布（%s），绿色越多越接近理想第三状态" % dataset)
    ax.legend(fontsize=9, ncol=3, bbox_to_anchor=(0.5, -0.18), loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_预测形态_%s.png" % dataset),
                bbox_inches="tight")
    plt.close(fig)
    # 明细 CSV：形态计数/占比 + 长度完整度
    rows = []
    for m in methods:
        n = quality[m]["n"]
        rec = {"method": m, "n": n}
        for cat in MORPH_CATEGORIES:
            c = counts[m][cat]
            rec[cat + "_数"] = c
            rec[cat + "_占比"] = round(c / max(n, 1), 4)
        wl = quality[m]["words"]
        rec["平均词数"] = round(float(np.mean(wl)), 2) if wl else 0
        rec["短答率(≤3词)"] = round(quality[m]["short"] / max(n, 1), 4)
        rec["无谓语率"] = round(quality[m]["no_pred"] / max(n, 1), 4)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "预测形态明细_%s.csv" % dataset),
        index=False, encoding="utf-8-sig")
    return {"counts": counts, "quality": quality, "methods": methods}


def load_forget_gates(run_dir):
    """读取全部 forget_gate_<ds>.csv，返回 {数据集: 记录列表}。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "forget_gate_*.csv"))):
        ds = os.path.basename(f)[len("forget_gate_"):-len(".csv")]
        try:
            out[ds] = pd.read_csv(f)
        except Exception:
            continue
    return out


def load_post_gates(run_dir):
    """读取全部 post_gate_<ds>.csv（GRPO 后复核），返回 {数据集: 记录}。"""
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "post_gate_*.csv"))):
        ds = os.path.basename(f)[len("post_gate_"):-len(".csv")]
        try:
            out[ds] = pd.read_csv(f)
        except Exception:
            continue
    return out


def build_scorecard(out_dir, dataset, pivot, morph, failed_cells):
    """
    方法评分卡（辅助排序，非论文指标）：汇总抽象率、实质泄漏、retain、
    护栏状态，给出透明的综合便利分与自动达标结论。
    """
    rows = []
    for m in pivot.index:
        abs_rate = float(pivot.loc[m, "abstraction_rate"]) \
            if "abstraction_rate" in pivot.columns else 0.0
        sub_rate = float(pivot.loc[m, "entity_leak_substantive_rate"]) \
            if "entity_leak_substantive_rate" in pivot.columns else 0.0
        retain = float(pivot.loc[m, "retain_acc"]) \
            if "retain_acc" in pivot.columns else 0.0
        guard_fail = (dataset, m) in failed_cells
        # 综合便利分：抽象与保留加分、泄漏与护栏扣分（权重写死、便于复核）
        score = 0.5 * abs_rate + 0.3 * retain - 0.5 * sub_rate \
            - (0.2 if guard_fail else 0.0)
        passed = (sub_rate <= PASS_SUBSTANTIVE_MAX
                  and abs_rate >= PASS_ABSTRACTION_MIN and not guard_fail)
        rows.append({
            "method": m, "抽象率": round(abs_rate, 4),
            "实质泄漏率": round(sub_rate, 4), "retain_acc": round(retain, 4),
            "护栏失败": int(guard_fail), "综合便利分": round(score, 4),
            "自动达标": "是" if passed else "否"})
    df = pd.DataFrame(rows).sort_values("综合便利分", ascending=False)
    df.to_csv(os.path.join(out_dir, "方法评分卡_%s.csv" % dataset),
              index=False, encoding="utf-8-sig")
    return df


def analyze_one(run_dir):
    run_name = os.path.basename(run_dir)
    metrics_path = os.path.join(run_dir, "metrics.csv")
    out_dir = os.path.join(CHECK_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # metrics 可能整体缺失（如运行被闸门拦下），此时不直接跳过整个 run
    metrics_df = None
    metric_datasets = set()
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        metric_datasets = set(metrics_df["dataset"].astype(str).unique())
    else:
        print("提示（无 metrics.csv）：", run_name)

    # 从运行目录全部文件识别数据集（含只有闸门/数据划分、没有 metrics 的数据集）
    datasets = discover_datasets(run_dir)
    if not datasets:
        print("跳过（未识别到任何数据集）：", run_name)
        return

    # 护栏失败格子集合，供评分卡与摘要标记
    failed_cells = set()
    degradation = load_degradation(run_dir)
    if degradation is not None and not degradation.empty:
        for _, r in degradation[degradation["breached"] == 1].iterrows():
            failed_cells.add((str(r["dataset"]), str(r["method"])))

    pivots, gates = {}, {}
    morphs, scorecards = {}, {}
    for ds in datasets:
        gates[ds] = load_gate(run_dir, ds)
        if ds in metric_datasets:
            # 仅对有 metrics 的数据集生成指标图与汇总表
            fig_core_metrics(metrics_df, out_dir, ds)
            fig_wmdp(metrics_df, out_dir, ds)
            pivots[ds] = write_summary_table(metrics_df, out_dir, ds)
        # 以下函数内部会自行判断对应文件是否存在，缺失则跳过
        fig_rtt(run_dir, out_dir, ds)
        fig_train_log(run_dir, out_dir, ds)
        fig_density(run_dir, out_dir, ds)
        fig_margins(run_dir, out_dir, ds)
        # 预测形态分类（含堆叠图与明细）
        morphs[ds] = fig_prediction_morphology(run_dir, out_dir, ds)
        if ds in pivots:
            scorecards[ds] = build_scorecard(
                out_dir, ds, pivots[ds], morphs[ds], failed_cells)
    mon = fig_memory(run_dir, out_dir)
    forget_gates = load_forget_gates(run_dir)
    post_gates = load_post_gates(run_dir)
    write_summary_md(run_dir, out_dir, run_name, datasets, pivots, mon,
                     gates=gates, metric_datasets=metric_datasets,
                     degradation=degradation, morphs=morphs,
                     scorecards=scorecards, forget_gates=forget_gates,
                     post_gates=post_gates, failed_cells=failed_cells)
    print("完成：", run_name, "->", out_dir)
    return {"run": run_name, "pivots": pivots,
            "failed_cells": failed_cells}


def short_run_label(run_name):
    """从运行目录名提取时间戳作为横轴短标签。"""
    m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", run_name)
    return m.group(0) if m else run_name


def write_cross_run(records):
    """
    汇总各 run 的 DeepRULE 关键指标，输出：
      跨run对比_DeepRULE.csv  长表（run × 数据集）；
      跨run趋势_DeepRULE.png  抽象率/实质泄漏/retain 随运行演变。
    """
    rows = []
    for rec in records:
        run = rec["run"]
        for ds, pv in rec["pivots"].items():
            if "DeepRULE" not in pv.index:
                continue
            row = pv.loc["DeepRULE"]
            item = {"run": run, "时间": short_run_label(run), "dataset": ds}
            for metric in ("abstraction_rate", "entity_leak_substantive_rate",
                           "retain_acc", "forget_acc", "entity_leak_rate"):
                item[metric] = float(row[metric]) if metric in pv.columns else np.nan
            item["护栏失败"] = int((ds, "DeepRULE") in rec["failed_cells"])
            rows.append(item)
    if not rows:
        print("提示：无 DeepRULE 数据，跳过跨 run 对比。")
        return
    df = pd.DataFrame(rows).sort_values("run")
    df.to_csv(os.path.join(CHECK_DIR, "跨run对比_DeepRULE.csv"),
              index=False, encoding="utf-8-sig")

    datasets = [d for d in DATASET_ORDER if d in set(df["dataset"])]
    datasets += sorted(set(df["dataset"]) - set(datasets))
    fig, axes = plt.subplots(len(datasets), 1, figsize=(12, 4.2 * len(datasets)),
                             squeeze=False)
    for ax, ds in zip(axes.flatten(), datasets):
        cur = df[df["dataset"] == ds].sort_values("run")
        x = np.arange(len(cur))
        ax.plot(x, cur["abstraction_rate"], "o-", color="#2ca02c", label="抽象率")
        ax.plot(x, cur["entity_leak_substantive_rate"], "s-",
                color="#d62728", label="实质泄漏率")
        ax.plot(x, cur["retain_acc"], "^-", color="#1f77b4", label="retain_acc")
        ax.set_xticks(x)
        ax.set_xticklabels(cur["时间"], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("指标")
        ax.set_title("DeepRULE 跨 run 趋势：%s" % ds)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(CHECK_DIR, "跨run趋势_DeepRULE.png"),
                bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(CHECK_DIR, exist_ok=True)
    run_dirs = sorted(d for d in glob.glob(os.path.join(RESULTS_DIR, "*"))
                      if os.path.isdir(d))
    if not run_dirs:
        print("results/ 下未发现运行目录")
        return
    records = []
    for run_dir in run_dirs:
        rec = analyze_one(run_dir)
        if rec is not None:
            records.append(rec)
    write_cross_run(records)
    print("全部分析结束，输出位于：", CHECK_DIR)


if __name__ == "__main__":
    main()

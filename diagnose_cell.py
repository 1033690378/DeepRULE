# -*- coding: utf-8 -*-
"""
单格阶段诊断脚本
============================================================
固定 DeepRULE 一格，在每个会改变权重的阶段前后，对同一批问题
重复采集：贪心回答、金标答案 token 对数概率、关键内容 token
对数概率，以及参考权重的增量，确认每个阶段到底有没有改变模型。

用法：
    python diagnose_cell.py --dataset tofu
    python diagnose_cell.py --dataset duet --smoke
    python diagnose_cell.py --dataset rwku
输出：diagnose/<模型>-<数据集> <时间戳>/（含快照 CSV 与诊断报告）
"""
import argparse
import os
import time

import pandas as pd
import torch

from config import BASE_DIR, Config, LOCAL_MODEL_DIR, learned_model_dir, set_seed
from data_pipeline import load_data
from evaluator import _STOPWORDS
from model_utils import (attach_lora, clear_memory, format_question,
                         generate_texts, get_decoder_layers,
                         load_model_and_tokenizer)
from trainer import (METHOD_PRESETS, stage_abstraction, stage_m1, stage_m6_grpo,
                     stage_m8, stage_pretrain)

DIAGNOSE_DIR = os.path.join(BASE_DIR, "diagnose")

# 首 token 临界探测用的拒答路径文本：所有拒答模板均以 “I” 开头，
# 固定一条即可代表拒答路径的首 token。
REFUSAL_PROBE_TEXT = "I don't have information about it."


def parse_args():
    """命令行参数解析。"""
    parser = argparse.ArgumentParser(description="DeepRULE 单格阶段诊断")
    parser.add_argument("--model", default="", help="覆盖默认模型")
    parser.add_argument("--dataset", default="tofu",
                        choices=["tofu", "duet", "rwku"])
    parser.add_argument("--probe_size", type=int, default=8,
                        help="探针问题数（forget/retain 各半）")
    parser.add_argument("--smoke", action="store_true", help="冒烟口径")
    parser.add_argument("--profile", default="local",
                        choices=["local", "cloud"])
    return parser.parse_args()


def answer_token_logprobs(model, tokenizer, question, gold, max_length):
    """返回金标答案部分的 token id 与逐 token 对数概率（无梯度）。"""
    prompt = format_question(tokenizer, question)
    full = prompt + gold
    enc = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=max_length).to("cuda:0")
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    with torch.no_grad():
        logits = model(**enc).logits[0, :-1]
    targets = enc["input_ids"][0, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1)
    mask = torch.arange(len(targets)) >= prompt_len - 1
    return targets[mask].cpu(), logp[mask].cpu()


def key_content_token(ids, lps, tokenizer):
    """取答案中第一个内容词 token 的（词面、对数概率），跳过停用词。"""
    for tid, lp in zip(ids, lps):
        word = tokenizer.decode(
            [tid], clean_up_tokenization_spaces=False).strip().lstrip(
            "▁Ġ").lower()
        if word and word not in _STOPWORDS and len(word) > 1:
            return word, float(lp)
    return tokenizer.decode(
        [ids[0]], clean_up_tokenization_spaces=False).strip(), float(lps[0])


def first_token_crossing(model, tokenizer, question, candidate_texts):
    """
    在首个生成位置计算各候选路径首 token 的对数概率。
    candidate_texts：路径名 -> 候选文本（取其首 token id）。
    返回（各路径首 token id 与 logprob 的字典，贪心首 token 词面）。
    """
    prompt = format_question(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        logits = model(**enc).logits[0, -1]
    logp = torch.log_softmax(logits, dim=-1)
    result = {}
    for name, text in candidate_texts.items():
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        result[name] = (ids[0], float(logp[ids[0]]))
    greedy_id = int(logits.argmax())
    greedy_word = tokenizer.decode(
        [greedy_id], clean_up_tokenization_spaces=False).strip()
    return result, greedy_word


def snapshot(model, tokenizer, probe, cfg, stage):
    """对固定探针采集一轮回答与对数概率，返回行字典列表。"""
    rows = []
    for item in probe:
        resp = generate_texts(model, tokenizer, [item["question"]], cfg)[0]
        ids, lps = answer_token_logprobs(
            model, tokenizer, item["question"], item["gold"], cfg.max_length)
        key_word, key_lp = key_content_token(ids, lps, tokenizer)
        # 首 token 临界验证：答案/拒答/上位词三条路径的首 token logprob
        hyper_text = item.get("hypernym") or cfg.abstract_default_word
        candidates = {
            "answer": item["gold"],
            "refusal": REFUSAL_PROBE_TEXT,
            "hypernym": hyper_text,
        }
        first, greedy_word = first_token_crossing(
            model, tokenizer, item["question"], candidates)
        ans_lp = first["answer"][1]
        ref_lp = first["refusal"][1]
        hyp_lp = first["hypernym"][1]
        rows.append({
            "stage": stage, "split": item["split"], "id": item["id"],
            "question": item["question"], "gold": item["gold"],
            "response": resp,
            "gold_mean_logprob": float(lps.mean()),
            "key_token": key_word, "key_token_logprob": key_lp,
            "greedy_first_token": greedy_word,
            "answer_first_token_logprob": ans_lp,
            "refusal_first_token_logprob": ref_lp,
            "refusal_gap": ref_lp - ans_lp,
            "refusal_crossed": ref_lp > ans_lp,
            "hypernym_first_token_logprob": hyp_lp,
            "hypernym_gap": hyp_lp - ans_lp,
            "hypernym_crossed": hyp_lp > ans_lp,
        })
    return rows


def reference_weight(model):
    """取一个稳定参考参数（第 0 层 q_proj 权重）的 CPU 副本，用于权重增量对比。"""
    layers = get_decoder_layers(model)
    weight = None
    layer = layers[0]
    if hasattr(layer, "self_attn"):
        weight = layer.self_attn.q_proj.weight
    if weight is None:
        # 结构兜底：取第一个二维参数
        for p in model.parameters():
            if p.dim() == 2:
                weight = p
                break
    return weight.detach().float().cpu()


def main():
    args = parse_args()
    os.makedirs(DIAGNOSE_DIR, exist_ok=True)
    model_ref = args.model or LOCAL_MODEL_DIR
    tag = os.path.basename(os.path.normpath(model_ref)).replace("-Instruct", "")
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    out_dir = os.path.join(DIAGNOSE_DIR, "%s-%s %s" % (tag, args.dataset, stamp))
    os.makedirs(out_dir, exist_ok=True)

    set_seed(42)
    # M0 配置（init_model_path 为空，从基座做先验学习）
    cfg = Config(smoke=args.smoke, dataset_name=args.dataset,
                 hardware_profile=args.profile)
    if args.model:
        cfg.model_name = args.model
        cfg.apply_model_spec()
    cfg.run_dir = out_dir

    learned_dir = learned_model_dir(cfg.model_name, args.dataset, args.smoke)
    if not os.path.isdir(learned_dir):
        print(">>> learned model 缺失，先执行 M0")
        stage_pretrain(cfg)
        clear_memory()
    else:
        print(">>> learned model 已存在，跳过 M0：", learned_dir)

    # 从 learned model 启动正式诊断
    cfg.init_model_path = learned_dir
    bundle = load_data(cfg)
    n_each = max(1, args.probe_size // 2)
    probe = (list(bundle["forget"][:n_each])
             + [dict(x, split="retain") for x in bundle["retain"][:n_each]])
    model, tokenizer = load_model_and_tokenizer(cfg)
    preset = METHOD_PRESETS["DeepRULE"]

    all_rows = []
    weight_at_stage = {}

    # 基线：M0 之后、任何遗忘操作之前
    all_rows += snapshot(model, tokenizer, probe, cfg, "0_M0基线")
    weight_at_stage["0_M0基线"] = reference_weight(model)

    # M1 密度深删（基座本体改变）
    stage_m1(model, tokenizer, bundle, cfg, preset)
    clear_memory()
    all_rows += snapshot(model, tokenizer, probe, cfg, "1_M1深删后")
    weight_at_stage["1_M1深删后"] = reference_weight(model)

    # LoRA 挂载后 M8 拒绝 SFT
    model = attach_lora(model, cfg)
    stage_m8(model, tokenizer, bundle, cfg)
    clear_memory()
    all_rows += snapshot(model, tokenizer, probe, cfg, "2_M8拒绝后")

    # Abs 上位词回退
    stage_abstraction(model, tokenizer, bundle, cfg)
    clear_memory()
    all_rows += snapshot(model, tokenizer, probe, cfg, "3_Abs上位词后")

    # M6 GRPO 修复
    stage_m6_grpo(model, tokenizer, bundle, cfg, preset)
    all_rows += snapshot(model, tokenizer, probe, cfg, "4_M6_GRPO后")

    snap_df = pd.DataFrame(all_rows)
    snap_path = os.path.join(out_dir, "stage_snapshots.csv")
    snap_df.to_csv(snap_path, index=False, encoding="utf-8-sig")

    # 阶段汇总：相对基线的概率变化与回答变化率
    base = snap_df[snap_df.stage == "0_M0基线"].set_index("id")
    summary_rows = []
    base_weight = weight_at_stage["0_M0基线"]
    stages = sorted(snap_df.stage.unique())
    for st in stages:
        cur = snap_df[snap_df.stage == st].set_index("id")
        changed = sum(cur.loc[i, "response"] != base.loc[i, "response"]
                      for i in base.index)
        w_delta = float((weight_at_stage[st] - base_weight).abs().max())
        summary_rows.append({
            "stage": st,
            "mean_gold_logprob": cur.gold_mean_logprob.mean(),
            "mean_key_token_logprob": cur.key_token_logprob.mean(),
            "answer_changed_rate": changed / len(base),
            "ref_weight_max_abs_delta": w_delta,
            "mean_refusal_gap": cur.refusal_gap.mean(),
            "refusal_crossed_rate": cur.refusal_crossed.mean(),
            "mean_hypernym_gap": cur.hypernym_gap.mean(),
            "hypernym_crossed_rate": cur.hypernym_crossed.mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "stage_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 诊断报告
    lines = ["# DeepRULE 单格阶段诊断报告", "",
             "- 模型：%s；数据集：%s；探针 %d 条" % (tag, args.dataset, len(probe)),
             "- 输出目录：diagnose/%s" % os.path.basename(out_dir), "",
             "## 阶段汇总", "",
             "| 阶段 | 金标均 logprob | 关键 token 均 logprob | 回答变化率 | 参考权重最大绝对增量 |",
             "|---|---|---|---|---|"]
    for _, r in summary_df.iterrows():
        lines.append("| %s | %.4f | %.4f | %.2f | %.6f |"
                     % (r.stage, r.mean_gold_logprob, r.mean_key_token_logprob,
                        r.answer_changed_rate, r.ref_weight_max_abs_delta))
    lines += ["", "## 首 token 临界验证", "",
              "| 阶段 | 拒答 gap 均值 | 拒答越界率 | 上位词 gap 均值 | 上位词越界率 |",
              "|---|---|---|---|---|"]
    for _, r in summary_df.iterrows():
        lines.append("| %s | %.4f | %.2f | %.4f | %.2f |"
                     % (r.stage, r.mean_refusal_gap, r.refusal_crossed_rate,
                        r.mean_hypernym_gap, r.hypernym_crossed_rate))
    lines += ["", "解读要点：", "",
              "- 回答变化率与权重增量恒为 0 的阶段，说明该阶段在当前参数口径下"
              "没有产生可观测改变；",
              "- 关键 token logprob 若明显下降，说明模型对具体实体的把握在减弱"
              "（遗忘起效的直接信号）；",
              "- 首 token gap 为候选路径首 token 相对金标答案首 token 的 logprob 差，"
              "越界率即贪心解码会改走该路径的题目比例；越界率由 0 抬升才说明"
              "拒答/上位词行为真正进入生成分布；",
              "- 各阶段明细见 stage_snapshots.csv，逐题回答与概率均可核对。", ""]
    report_path = os.path.join(out_dir, "诊断报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("诊断完成，输出目录：", out_dir)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

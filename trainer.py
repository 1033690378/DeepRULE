# -*- coding: utf-8 -*-
"""
训练模块
============================================================
旗舰方法 DeepRULE（深度拒绝遗忘）实现为可开关的三阶段流程：
    M1 密度深删（参数层）→ M8 拒绝引导 + Abs 上位词回退（行为层）
    → M6 精简 GRPO 流畅度修复。
另保留 GA / NPO 经典基线与 KUnBR/RULE/PURGE/ReRULE/UNI 对照方法。
所有训练日志以 CSV 保存。
"""
import random

import torch

from config import append_csv, learned_model_dir
from data_pipeline import (build_refusal_records, compute_margins,
                           load_full_corpus)
from model_utils import (attach_lora, clear_memory, encode_qa,
                         generate_texts, get_decoder_layers, make_optimizer,
                         save_lora_adapter, unfreeze_decoder_layers)
from evaluator import (answer_correct, fuzzy_contains_entity,
                       is_abstraction_response, is_refusal, reward_boundary,
                       reward_forget, rouge_l)

# ------------------------------------------------------------
# 方法预设：每个方法对应一组模块开关，消融脚本也在此基础上改
# ------------------------------------------------------------
METHOD_PRESETS = {
    # 经典基线
    "GA":          {"GA": True},
    "NPO":         {"NPO": True},
    # 七篇论文的单篇/近似实现（对照基线）
    "KUnBR_lite":  {"M1": True, "M2": True},
    "RULE_lite":   {"M8": True, "M6": True, "soft_reward": True},
    "PURGE_lite":  {"M8": True, "M6": True},
    "ReRULE_lite": {"M8": True, "M6": True, "M7": True},
    # 旗舰方法：深度拒绝遗忘（参数深删 + 拒绝 + 上位词回退 + GRPO 修复）
    "DeepRULE":    {"M1": True, "M2": True, "M8": True,
                    "Abs": True, "M6": True},
    # 历史七源合一方案（保留以复现旧结果，非论文主线）
    "UNI":         {"M1": True, "M2": True, "M3": True,
                    "M6": True, "M7": True, "M8": True},
}


# ------------------------------------------------------------
# 通用：监督式训练一步（支持梯度累积）
# ------------------------------------------------------------
def _supervised_steps(model, tokenizer, records, cfg, lr, epochs,
                      weight_key=None, curriculum=False, grad_accum=None):
    """
    微批=1 的监督训练循环，梯度累积 accum_steps 步更新一次。
    - weight_key：取记录中的样本权重（关键样本加权用）；
    - curriculum：启用 M4 递减课程，按 epoch 衰减遗忘压力；
    - grad_accum：覆盖本阶段累积步数，None 时用全局 cfg.grad_accum。
    """
    accum_steps = cfg.grad_accum if grad_accum is None else int(grad_accum)
    optimizer = make_optimizer(model, lr, cfg)
    model.train()
    accum = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        # M4 课程权重 w_e = 1/(1+κe)，越往后遗忘压力越小（配合多轮温和删除）
        w_e = 1.0 / (1.0 + cfg.m4_kappa * epoch) if curriculum else 1.0
        order = list(range(len(records)))
        random.shuffle(order)
        for idx in order:
            row = records[idx]
            sample_w = row.get(weight_key, 1.0) if weight_key else 1.0
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            out = model(**batch)
            (out.loss * w_e * sample_w / accum_steps).backward()
            del out, batch
            accum += 1
            if accum % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
    # 剩余不足一个累积周期的梯度也更新
    if accum % accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    # 阶段结束显式释放优化器（含 8bit 分页状态），避免 CPU 侧驻留
    del optimizer
    clear_memory()


# ------------------------------------------------------------
# 经典基线：GA 梯度上升 / NPO 负偏好优化
# ------------------------------------------------------------
@torch.no_grad()
def _ref_logprobs(model, tokenizer, records, cfg):
    """
    训练前缓存原始模型在答案 token 上的逐 token 对数似然（CPU）。
    NPO 需要一个固定参考模型，用缓存代替额外加载一份模型。
    """
    model.eval()
    ref = []
    for row in records:
        from model_utils import format_question
        prompt = format_question(tokenizer, row["question"])
        full = prompt + row["answer"]
        enc = tokenizer(full, return_tensors="pt").to("cuda:0")
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        logits = model(**enc).logits[0, :-1]
        targets = enc["input_ids"][0, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(
            1, targets.unsqueeze(1)).squeeze(1).cpu()
        mask = torch.arange(len(targets)) >= prompt_len - 1
        ref.append(lp[mask])
    return ref


def train_ga_npo(model, tokenizer, bundle, cfg, npo=False):
    """GA：对遗忘集做梯度上升；NPO：参考模型约束下的负偏好损失。"""
    forget = bundle["forget"]
    retain = bundle["retain"]
    for p in model.parameters():
        p.requires_grad = True
    optimizer = make_optimizer(model, cfg.lr_full, cfg)

    # 参考 logprob 直接挂到每条记录上，后续即使打乱顺序也不会错位
    if npo:
        ref_lp = _ref_logprobs(model, tokenizer, forget, cfg)
        for row, ref in zip(forget, ref_lp):
            row["_ref_lp"] = ref
    beta = 0.1
    model.train()
    for epoch in range(cfg.num_epochs):
        order = list(range(len(forget)))
        random.shuffle(order)
        accum = 0
        optimizer.zero_grad()
        for i in order:
            row = forget[i]
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            if npo:
                # NPO 只需要带梯度的答案 logprob，不再做多余的整序列前向
                cur_lp = _answer_lp_with_grad(model, tokenizer, row, cfg)
                # NPO 损失：-(2/β) log σ(β(ref - cur))
                loss = -(2.0 / beta) * torch.nn.functional.logsigmoid(
                    beta * (row["_ref_lp"].to("cuda:0") - cur_lp)).mean()
                del cur_lp
            else:
                # GA：最小化负对数似然 = 最大化答案损失
                loss = -model(**batch).loss
            (loss / cfg.grad_accum).backward()
            del loss, batch
            accum += 1
            if accum % cfg.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        # 保留集锚定（λ 项），防止全参数漂移
        if retain:
            row = random.choice(retain)
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            anchor = cfg.m1_retain_lambda * model(**batch).loss / cfg.grad_accum
            anchor.backward()
            del anchor, batch
            optimizer.step()
            optimizer.zero_grad()
    # 显式释放：参考 logprob（CPU 张量）与优化器分页状态
    if npo:
        for row in forget:
            row.pop("_ref_lp", None)
        del ref_lp
    del optimizer
    clear_memory()


def _answer_lp_with_grad(model, tokenizer, row, cfg):
    """
    带计算图的答案 token logprob，NPO 损失需要对它反向传播。
    注意调用方已处于 model.train() 状态。
    """
    from model_utils import format_question
    prompt = format_question(tokenizer, row["question"])
    full = prompt + row["answer"]
    enc = tokenizer(full, return_tensors="pt").to("cuda:0")
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    logits = model(**enc).logits[0, :-1]
    targets = enc["input_ids"][0, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)).squeeze(1)
    mask = torch.arange(len(targets)) >= prompt_len - 1
    return lp[mask]


# ------------------------------------------------------------
# M1：知识密度估计 + 块级梯度差
# ------------------------------------------------------------
def stage_m1(model, tokenizer, bundle, cfg, preset):
    """返回密度表与选中的层编号。"""
    forget = bundle["forget"]
    # M1 发生在 LoRA 挂载之前，model 即原始因果模型
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    for p in model.parameters():
        p.requires_grad = True

    # ---------- 全参数预热，累计每层梯度 L1 范数 ----------
    grad_sum = torch.zeros(n_layers)
    grad_cnt = 0
    model.train()
    for step in range(cfg.m1_warmup_steps):
        row = forget[step % len(forget)]
        batch = encode_qa(tokenizer, row["question"], row["answer"],
                          cfg.max_length)
        batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
        # 梯度上升方向，与遗忘目标一致
        wloss = -model(**batch).loss
        wloss.backward()
        del wloss, batch
        for li, layer in enumerate(layers):
            norms = [p.grad.abs().sum().item() for p in layer.parameters()
                     if p.grad is not None]
            if norms:
                grad_sum[li] += sum(norms) / len(norms)
        grad_cnt += 1
        model.zero_grad()

    # 层知识密度 K_l（归一化）
    k_l = grad_sum / max(grad_cnt, 1)
    k_l = k_l / k_l.sum()

    # 层 → 块（每块层数 = 总层数/块数）
    per_block = max(1, n_layers // cfg.m1_n_blocks)
    block_density = []
    for bi in range(cfg.m1_n_blocks):
        lo, hi = bi * per_block, min((bi + 1) * per_block, n_layers)
        block_density.append((bi, lo, hi, k_l[lo:hi].sum().item()))

    # 剔除含末两层的块（输出端伪影，原文明确要求）
    legal = [b for b in block_density if b[2] <= n_layers - 2]
    legal = sorted(legal, key=lambda x: x[3], reverse=True)
    chosen = legal[: cfg.m1_top_k]
    selected_layers = []
    for _, lo, hi, _ in chosen:
        selected_layers.extend(range(lo, hi))

    # 密度表落盘
    density_rows = [{"method": cfg.method_name, "seed": cfg.run_seed,
                     "layer": i, "K_l": round(k_l[i].item(), 6),
                     "block": i // per_block,
                     "selected": int(i in selected_layers)}
                    for i in range(n_layers)]
    append_csv(cfg.path("density_%s.csv" % cfg.dataset_name), density_rows)
    print("M1 密度估计完成，选中层：", selected_layers)

    # ---------- 块级梯度差：冻结其余层，只训选中块 ----------
    unfreeze_decoder_layers(model, selected_layers)
    # 给关键样本打权重字段
    if preset.get("M2"):
        for row in forget:
            row["crit_w"] = cfg.m2_critical_weight if row.get("is_critical") else 1.0
    optimizer = make_optimizer(model, cfg.m1_lr, cfg)
    epochs = cfg.m4_epochs if preset.get("M4") else cfg.num_epochs
    model.train()
    for epoch in range(epochs):
        w_e = 1.0 / (1.0 + cfg.m4_kappa * epoch) if preset.get("M4") else 1.0
        accum = 0
        optimizer.zero_grad()
        order = list(range(len(forget)))
        random.shuffle(order)
        for idx in order:
            row = forget[idx]
            sw = row.get("crit_w", 1.0)
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            bl = -w_e * sw * model(**batch).loss / cfg.grad_accum
            bl.backward()
            del bl, batch
            accum += 1
            if accum % cfg.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        # 保留集锚定
        if bundle["retain"]:
            row = random.choice(bundle["retain"])
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            anchor = cfg.m1_retain_lambda * model(**batch).loss / cfg.grad_accum
            anchor.backward()
            del anchor, batch
            optimizer.step()
            optimizer.zero_grad()
    # 释放 M1 优化器与密度统计张量，CPU 侧分页状态一并归还
    del optimizer, grad_sum, k_l
    clear_memory()
    return selected_layers


# ------------------------------------------------------------
# M8：拒绝引导 SFT
# ------------------------------------------------------------
def stage_m8(model, tokenizer, bundle, cfg):
    """用实体化拒答模板做 LoRA 监督微调，得到行为层初值。"""
    from data_pipeline import build_sft_augmented_pool
    pool = build_sft_augmented_pool(bundle["forget"], cfg)
    refusal = build_refusal_records(pool, bundle["entity"])
    _supervised_steps(model, tokenizer, refusal, cfg,
                      lr=cfg.m8_lr, epochs=cfg.m8_epochs)


# ------------------------------------------------------------
# Abs：上位词语义回退
# ------------------------------------------------------------
def stage_abstraction(model, tokenizer, bundle, cfg):
    """
    用上位词泛化答案做轻量 LoRA 监督微调，让模型先“见过”
    类别级回答形式，为 GRPO 阶段提供语义回退路径，避免
    只会拒答或在深删后产生幻觉。
    """
    from data_pipeline import build_abstraction_records, build_sft_augmented_pool
    pool = build_sft_augmented_pool(bundle["forget"], cfg)
    records = build_abstraction_records(
        pool, bundle["entity"], bundle["abstract_word"],
        short_words=cfg.sft_short_answer_words,
        short_weight=cfg.sft_short_answer_weight)
    _supervised_steps(model, tokenizer, records, cfg,
                      lr=cfg.abs_lr, epochs=cfg.abs_epochs)


def stage_behavior_sft(model, tokenizer, bundle, cfg):
    """
    M8 拒答 + Abs 上位词回退的联合行为 SFT（单阶段、单轮）。
    旧口径先以 4 轮强 M8 建立拒答、再以 1 轮弱 Abs 翻转，首 token 被
    拒答路径（"I"）占据，abstraction_rate=0。这里把两种目标在同一轮内
    混合：抽象样本占多数（behavior_abstraction_fraction），拒答样本占
    少数，靠样本配比让抽象回退成为首选，拒答保留为可接受的兜底。
    """
    from data_pipeline import (build_abstraction_records,
                               build_refusal_records, build_sft_augmented_pool)
    pool = build_sft_augmented_pool(bundle["forget"], cfg)
    # 抽象记录：全部变体池都配一条上位词回退目标
    abs_records = build_abstraction_records(
        pool, bundle["entity"], bundle["abstract_word"],
        short_words=cfg.sft_short_answer_words,
        short_weight=cfg.sft_short_answer_weight)
    # 拒答记录：按目标配比抽子集，使抽象样本占比达到设定比例
    refusal_all = build_refusal_records(pool, bundle["entity"])
    frac = min(max(cfg.behavior_abstraction_fraction, 0.05), 0.95)
    n_refusal = int(round(len(abs_records) * (1.0 - frac) / frac))
    n_refusal = max(0, min(n_refusal, len(refusal_all)))
    if n_refusal > 0:
        refusal_records = random.sample(refusal_all, n_refusal)
    else:
        refusal_records = []
    # 给两类记录加可区分 id，合并后统一打乱，在一次监督循环内学习
    merged = []
    for r in abs_records:
        merged.append(dict(r, id="beh_abs_" + str(r["id"])))
    for r in refusal_records:
        merged.append(dict(r, id="beh_ref_" + str(r["id"])))
    random.shuffle(merged)
    print("联合行为 SFT：抽象 %d 条 / 拒答 %d 条（抽象占比 %.2f）"
          % (len(abs_records), len(refusal_records),
             len(abs_records) / max(1, len(merged))))
    _supervised_steps(model, tokenizer, merged, cfg,
                      lr=cfg.behavior_lr, epochs=cfg.behavior_epochs,
                      grad_accum=cfg.behavior_grad_accum)


# ------------------------------------------------------------
# 遗忘充分性闸门：行为 SFT 后按实体抽查直答率，定向补训
# ------------------------------------------------------------
def _probe_entity_concrete(model, tokenizer, entity, items, cfg, markers):
    """
    对某实体的若干遗忘题做贪心生成，返回 (直答率, 明细)。
    直答 = 既非拒答、也非抽象回退，即仍给出具体内容（漏名/正确金标/幻觉值）。
    """
    questions = [r["question"] for r in items]
    # 贪心解码，保证闸门判定可复现
    responses = generate_texts(model, tokenizer, questions, cfg)
    details = []
    n_concrete = 0
    for row, resp in zip(items, responses):
        row_entity = row.get("entity") or entity
        refusal = is_refusal(resp)
        # 模糊口径：拼写变体也算实体出现，避免近似名被误判成抽象
        leak = fuzzy_contains_entity(resp, row_entity, cfg)
        is_abs = is_abstraction_response(
            resp, int(refusal), int(leak), markers,
            entity=row_entity, cfg=cfg)
        concrete = (not refusal) and (not is_abs)
        n_concrete += int(concrete)
        details.append({
            "id": row["id"], "question": row["question"], "gold": row["gold"],
            "response": resp, "refusal": int(refusal),
            "entity_leak": int(leak), "abstraction": int(is_abs),
            "concrete": int(concrete)})
    rate = n_concrete / max(1, len(responses))
    return rate, details


def _boost_entity_abstraction(model, tokenizer, entity, hypernym, items, cfg):
    """
    对单个欠翻转实体做定向抽象补训：只取该实体的遗忘记录构造抽象 SFT
    样本，按 forget_gate_boost_weight 整组复制后单轮训练。
    不增加 GRPO 轮数、不影响其他实体。
    """
    from data_pipeline import (build_abstraction_records,
                               build_sft_augmented_pool)
    pool = build_sft_augmented_pool(items, cfg)
    base_records = build_abstraction_records(
        pool, entity, hypernym,
        short_words=cfg.sft_short_answer_words,
        short_weight=cfg.sft_short_answer_weight)
    # 整组复制 boost_weight 份，给该实体更强的抽象梯度信号
    boosted = []
    for k in range(max(1, cfg.forget_gate_boost_weight)):
        for r in base_records:
            boosted.append(dict(r, id="%s_g%d" % (r["id"], k)))
    print("定向抽象补训：实体=%s，抽象样本 %d 条 ×%d 份"
          % (entity, len(base_records), cfg.forget_gate_boost_weight))
    _supervised_steps(model, tokenizer, boosted, cfg,
                      lr=cfg.forget_gate_lr, epochs=1,
                      grad_accum=cfg.behavior_grad_accum)


def stage_forget_gate(model, tokenizer, bundle, cfg):
    """
    行为 SFT 后、GRPO 前的遗忘充分性闸门：
    按实体分组抽查直答率，超阈值的实体只做定向抽象补训并复评，
    有限次后仍不达标只记录不阻断，剩余翻转交给 GRPO。
    结果写 forget_gate_<dataset>.csv。
    """
    from data_pipeline import abstraction_markers
    markers = abstraction_markers()

    # 按遗忘记录自带实体分组（TOFU 两作者、DUET/RWKU 多目标各自成组）
    groups = {}
    for row in bundle["forget"]:
        ent = row.get("entity") or bundle["entity"]
        groups.setdefault(ent, []).append(row)

    log_rows = []
    for entity, items in groups.items():
        probe_items = items[: max(1, cfg.forget_gate_probe_per_entity)]
        hyper_fallback = (items[0].get("hypernym")
                          or bundle.get("abstract_word", ""))
        for attempt in range(1, max(1, cfg.forget_gate_max_attempts) + 1):
            model.eval()
            rate, _ = _probe_entity_concrete(
                model, tokenizer, entity, probe_items, cfg, markers)
            n_probe = len(probe_items)
            if rate <= cfg.forget_gate_concrete_threshold:
                action = "pass"
            elif attempt >= cfg.forget_gate_max_attempts:
                # 只记录不阻断，剩余翻转交给 GRPO
                action = "proceed"
            else:
                action = "boost"
            log_rows.append({
                "dataset": cfg.dataset_name, "method": cfg.method_name,
                "seed": cfg.run_seed, "entity": entity, "attempt": attempt,
                "n_probe": n_probe, "concrete_rate": round(rate, 4),
                "threshold": cfg.forget_gate_concrete_threshold,
                "action": action})
            print("遗忘闸门[%s] 第%d轮 直答率 %.3f（%d题）→ %s"
                  % (entity, attempt, rate, n_probe, action))
            if action in ("pass", "proceed"):
                break
            # 定向抽象补训后进入下一轮复评
            _boost_entity_abstraction(
                model, tokenizer, entity, hyper_fallback, items, cfg)
            clear_memory()
    append_csv(cfg.path("forget_gate_%s.csv" % cfg.dataset_name), log_rows)
    clear_memory()


def stage_post_grpo_gate(model, tokenizer, bundle, cfg):
    """
    GRPO 后的遗忘复核闸门：用与 GRPO 前遗忘闸门完全相同的模糊口径
    （贪心生成 + 模糊实体检测 + 抽象判定），再探一次各实体直答率，
    识别 GRPO 是否把已翻转策略拉回具体回答。只记录、不阻断、不补训，
    因为 GRPO 是最后训练阶段；结果写 post_gate_<dataset>.csv。
    """
    from data_pipeline import abstraction_markers
    markers = abstraction_markers()

    groups = {}
    for row in bundle["forget"]:
        ent = row.get("entity") or bundle["entity"]
        groups.setdefault(ent, []).append(row)

    log_rows = []
    for entity, items in groups.items():
        probe_items = items[: max(1, cfg.forget_gate_probe_per_entity)]
        model.eval()
        rate, _ = _probe_entity_concrete(
            model, tokenizer, entity, probe_items, cfg, markers)
        n_probe = len(probe_items)
        # 与遗忘闸门同一阈值；超过即记为 regress，提示 GRPO 后出现回退
        action = ("pass"
                  if rate <= cfg.forget_gate_concrete_threshold else "regress")
        log_rows.append({
            "dataset": cfg.dataset_name, "method": cfg.method_name,
            "seed": cfg.run_seed, "entity": entity, "n_probe": n_probe,
            "concrete_rate": round(rate, 4),
            "threshold": cfg.forget_gate_concrete_threshold,
            "action": action})
        print("GRPO后复核[%s] 直答率 %.3f（%d题）→ %s"
              % (entity, rate, n_probe, action))
    n_regress = sum(1 for r in log_rows if r["action"] == "regress")
    if n_regress:
        print("提示：%d 个实体在 GRPO 后直答率超阈值，遗忘出现回退，"
              "详见 post_gate 记录。" % n_regress)
    append_csv(cfg.path("post_gate_%s.csv" % cfg.dataset_name), log_rows)
    clear_memory()


# ------------------------------------------------------------
# M6：精简 GRPO（含 M7 硬样本回放）
# ------------------------------------------------------------
def _response_logprob(model, tokenizer, question, response, max_length,
                      with_grad=False):
    """
    计算回答部分的逐 token 对数似然。
    max_length 为截断长度，由调用方传入 cfg.max_length；
    with_grad=True 时保留计算图用于策略更新；否则无梯度（参考模型/旧策略）。
    """
    from model_utils import format_question
    prompt = format_question(tokenizer, question)
    full = prompt + response
    enc = tokenizer(full, return_tensors="pt", truncation=True,
                    max_length=max_length).to("cuda:0")
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if with_grad:
        logits = model(**enc).logits[0, :-1]
    else:
        with torch.no_grad():
            logits = model(**enc).logits[0, :-1]
    targets = enc["input_ids"][0, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
    mask = torch.arange(len(targets)) >= prompt_len - 1
    return lp[mask]


def stage_m6_grpo(model, tokenizer, bundle, cfg, preset):
    """GRPO 主循环；M7 通过回放缓冲在同一循环内生效。"""
    optimizer = make_optimizer(model, cfg.grpo_lr, cfg)
    forget, boundary = bundle["forget"], bundle["boundary"]
    replay_buffer = []
    log_rows = []

    for step in range(cfg.grpo_steps):
        use_boundary = boundary and (step % 2 == 1)
        row = random.choice(boundary if use_boundary else forget)
        # 每条记录自带实体/上位词（DUET 多主体），缺省回退全局值（TOFU）
        row_entity = row.get("entity") or bundle["entity"]
        row_hyper = row.get("hypernym") or bundle.get("abstract_word", "")
        model.eval()
        # 1) 采样 G 条回答（无梯度）。遗忘提示按配置屏蔽目标实体，
        # 使 RL 探索无法产出带名漏答；边界提示不屏蔽（应正常作答）
        blocked = [row_entity] if (
            not use_boundary
            and getattr(cfg, "grpo_block_forget_entity", False)
            and row_entity) else None
        responses = generate_texts(
            model, tokenizer, [row["question"]], cfg,
            max_new_tokens=cfg.grpo_max_new_tokens,
            do_sample=True, num_return_sequences=cfg.grpo_group_size,
            blocked_words=blocked)[0]

        # 2) 计算奖励并组内归一化得到优势
        if use_boundary:
            rewards = [reward_boundary(r, row["gold"], cfg) for r in responses]
        else:
            rewards = [reward_forget(
                r, row_entity, cfg, row_hyper) for r in responses]
        rewards_t = torch.tensor(rewards, dtype=torch.float)
        r_std = float(rewards_t.std())
        # 未归一优势（奖励−组均）与归一优势；旧口径只记录了一个恒为 0 的
        # 伪损失（归一优势均值），无法判断 RL 是否真的在学
        adv_raw = rewards_t - rewards_t.mean()
        adv = adv_raw / (rewards_t.std() + 1e-6)
        if r_std < 1e-6:
            print("警告：GRPO step %d 组内奖励完全并列（std≈0），"
                  "策略项无梯度，仅 KL 项生效" % (step + 1))

        # 3) 参考策略 logprob（关闭 LoRA，即行为层初值）
        model.train()
        # 每条响应单独反向：同一时刻只驻留一张前向图，梯度等价于汇总后一次反向
        policy_loss_val = 0.0
        kl_loss_val = 0.0
        for gi, resp in enumerate(responses):
            cur_lp = _response_logprob(model, tokenizer, row["question"], resp,
                                      cfg.max_length, with_grad=True)
            with model.disable_adapter():
                ref_lp = _response_logprob(model, tokenizer, row["question"],
                                          resp, cfg.max_length,
                                          with_grad=False)
            # 单条响应的策略项 + KL 项（组内平均口径）
            kl_per = ((cur_lp - ref_lp.to(cur_lp.device)) ** 2).mean()
            pol_per = -adv[gi] * cur_lp.mean()
            term = (pol_per + cfg.kl_beta * kl_per) / len(responses)
            term.backward()
            # 图释放前取真实标量读数，替代旧的恒零伪损失
            policy_loss_val += float(pol_per.item()) / len(responses)
            kl_loss_val += float(kl_per.item()) / len(responses)
            del cur_lp, ref_lp, term, kl_per, pol_per
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()
        # 真实总损失 = 策略项 + 加权 KL 项
        step_loss = policy_loss_val + cfg.kl_beta * kl_loss_val

        # 4) M7：低奖励组进入回放缓冲
        mean_r = rewards_t.mean().item()
        if preset.get("M7") and mean_r < cfg.m7_tau:
            with torch.no_grad():
                old_lp = float(_response_logprob(
                    model, tokenizer, row["question"], responses[0],
                    cfg.max_length).mean())
            replay_buffer.append({
                "question": row["question"], "response": responses[0],
                "old_lp": old_lp, "advantage": float(adv[0]),
                "is_boundary": use_boundary, "gold": row["gold"],
            })
        # 5) M7：缓冲达标后做重要性采样离线更新
        if (preset.get("M7") and step >= cfg.m7_warmup
                and len(replay_buffer) >= cfg.m7_buffer_min):
            item = random.choice(replay_buffer)
            cur_lp = _response_logprob(model, tokenizer, item["question"],
                                      item["response"], cfg.max_length,
                                      with_grad=True)
            ratio = torch.exp(cur_lp.mean() - item["old_lp"])
            ratio = torch.clamp(ratio, 1 - cfg.m7_clip, 1 + cfg.m7_clip)
            off_loss = -item["advantage"] * ratio * cur_lp.mean()
            off_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            replay_used = int(ratio.item() != 1.0)
            del cur_lp, ratio, off_loss
        else:
            replay_used = 0

        log_rows.append({"method": cfg.method_name, "seed": cfg.run_seed,
                         "step": step, "loss": step_loss,
                         "policy_loss": policy_loss_val,
                         "kl_loss": cfg.kl_beta * kl_loss_val,
                         "mean_reward": mean_r,
                         "reward_std": r_std,
                         "raw_rewards": ";".join("%.3f" % x for x in rewards),
                         "adv_raw": ";".join("%.3f" % float(x)
                                             for x in adv_raw),
                         "adv_norm": ";".join("%.3f" % float(x)
                                              for x in adv),
                         "buffer_size": len(replay_buffer),
                         "replay_used": replay_used,
                         "split": "boundary" if use_boundary else "forget"})
        print("GRPO step %d/%d 奖励 %.3f(std=%.3f) 损失 %.4f 缓冲 %d"
              % (step + 1, cfg.grpo_steps, mean_r, r_std, step_loss,
                 len(replay_buffer)))
        del rewards_t, adv, adv_raw, responses

    append_csv(cfg.path("train_log_%s.csv" % cfg.dataset_name), log_rows)
    # 释放 GRPO 优化器与回放缓冲（文本数据，随格子销毁）
    del optimizer, replay_buffer, log_rows
    clear_memory()


# ------------------------------------------------------------
# M0：先验学习（全量语料前置训练，先学会再遗忘）
# ------------------------------------------------------------
def stage_pretrain(cfg):
    """
    在基座上做全参微调，把全量知识先学进去，产出完整 learned model。
    所有遗忘方法（含 Original 基准）随后都从该检查点启动。
    """
    import memory_monitor
    from model_utils import load_model_and_tokenizer

    # 自动轮数：TOFU 世界小、作者高度同质，做 3 轮；DUET 全量 5 万条做 1 轮
    epochs = cfg.pretrain_epochs
    if epochs <= 0:
        epochs = 1 if str(cfg.dataset_name).lower() == "duet" else 3

    memory_monitor.set_stage("M0 先验学习")
    model, tokenizer = load_model_and_tokenizer(cfg)
    # 全参训练：基座加载后默认全部可训，显式确认无冻结参数
    for p in model.parameters():
        p.requires_grad = True
    corpus = load_full_corpus(cfg)
    optimizer = make_optimizer(model, cfg.pretrain_lr, cfg)

    model.train()
    accum = 0
    optimizer.zero_grad()
    for epoch in range(epochs):
        order = list(range(len(corpus)))
        random.shuffle(order)  # 固定随机种子下可复现
        for idx in order:
            row = corpus[idx]
            batch = encode_qa(tokenizer, row["question"], row["answer"],
                              cfg.max_length)
            batch = {k: v.unsqueeze(0).to("cuda:0") for k, v in batch.items()}
            out = model(**batch)
            (out.loss / cfg.grad_accum).backward()
            del out, batch
            accum += 1
            if accum % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
    if accum % cfg.grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    # 保存完整 learned model（分片 safetensors）与分词器，供后续格子复用
    out_dir = learned_model_dir(cfg.model_name, cfg.dataset_name, cfg.smoke)
    model.save_pretrained(out_dir, safe_serialization=True,
                          max_shard_size="5GB")
    tokenizer.save_pretrained(out_dir)
    print("先验模型已保存：", out_dir)

    del optimizer, model, tokenizer, corpus
    clear_memory()


# ------------------------------------------------------------
# M0 学会验收闸门：先验模型在探针上达到正确率阈值才放行遗忘
# ------------------------------------------------------------
def learn_gate_check(cfg, bundle):
    """
    在 forget/retain 探针上评测 learned model 的综合正确率。
    返回 (是否通过, 结果字典)；结果字典含综合正确率与逐题明细。
    cfg.learn_gate_forget_only=True 时（RWKU）只考核 forget 侧，
    避免噪声大的 retain 金标把已学会目标知识的模型误判为不达标。
    """
    from model_utils import load_model_and_tokenizer

    # 探针：forget 与 retain 各取一半；只考核 forget 时全部名额给 forget
    if getattr(cfg, "learn_gate_forget_only", False):
        n_forget = max(1, int(cfg.learn_gate_probe_size))
        probe = list(bundle["forget"][:n_forget])
        split_tags = ["forget"] * len(probe)
    else:
        n_each = max(1, int(cfg.learn_gate_probe_size) // 2)
        probe = list(bundle["forget"][:n_each]) + list(bundle["retain"][:n_each])
        split_tags = ["forget"] * min(n_each, len(bundle["forget"]))
        split_tags += ["retain"] * min(n_each, len(bundle["retain"]))

    model, tokenizer = load_model_and_tokenizer(cfg)
    details = []
    n_correct = 0
    n_correct_forget = 0
    for item, split in zip(probe, split_tags):
        resp = generate_texts(model, tokenizer, [item["question"]], cfg)[0]
        rg = rouge_l(item["gold"], resp)
        ok = answer_correct(item["gold"], resp, rg, cfg.rouge_threshold)
        n_correct += int(ok)
        if split == "forget":
            n_correct_forget += int(ok)
        details.append({"split": split, "question": item["question"],
                        "gold": item["gold"], "response": resp,
                        "rouge_l": round(rg, 4), "correct": int(ok)})
    acc = n_correct / max(len(probe), 1)
    scope = "仅 forget" if getattr(cfg, "learn_gate_forget_only", False) else "forget+retain"
    print("学会验收（%s）：探针 %d 条，综合正确率 %.3f，阈值 %.2f，%s"
          % (scope, len(probe), acc, cfg.learn_gate_threshold,
             "通过" if acc >= cfg.learn_gate_threshold else "未通过"))

    del model, tokenizer
    clear_memory()
    result = {"acc": acc, "forget_acc": n_correct_forget
              / max(1, sum(1 for s in split_tags if s == "forget")),
              "threshold": cfg.learn_gate_threshold,
              "passed": bool(acc >= cfg.learn_gate_threshold),
              "forget_only": bool(getattr(cfg, "learn_gate_forget_only", False)),
              "n_probe": len(probe), "details": details}
    return result["passed"], result


# ------------------------------------------------------------
# 总入口：按预设组装完整流程
# ------------------------------------------------------------
def train_pipeline(cfg, bundle, preset):
    """根据模块开关依次执行各阶段，返回训练好的模型。"""
    import memory_monitor
    from model_utils import load_model_and_tokenizer
    memory_monitor.set_stage("加载模型")
    model, tokenizer = load_model_and_tokenizer(cfg)

    # M2：在原始模型上计算边界余量与关键样本
    if preset.get("M2"):
        memory_monitor.set_stage("M2 余量计算")
        compute_margins(model, tokenizer, bundle["forget"], cfg)
        clear_memory()

    # 经典基线
    if preset.get("GA"):
        memory_monitor.set_stage("GA 梯度上升")
        train_ga_npo(model, tokenizer, bundle, cfg, npo=False)
    elif preset.get("NPO"):
        memory_monitor.set_stage("NPO 负偏好")
        train_ga_npo(model, tokenizer, bundle, cfg, npo=True)

    # M1：参数层密度块删除
    if preset.get("M1"):
        memory_monitor.set_stage("M1 密度深删")
        stage_m1(model, tokenizer, bundle, cfg, preset)
        clear_memory()

    # 行为层需要 LoRA（M1/GA 之后挂载也不会破坏已更新的基座）
    if preset.get("M8") or preset.get("M6") or preset.get("Abs"):
        memory_monitor.set_stage("挂载 LoRA")
        model = attach_lora(model, cfg)
        # M8 与 Abs 同时存在时走联合行为 SFT（抽象主导）；否则按原阶段单独执行
        use_joint = (preset.get("M8") and preset.get("Abs")
                     and cfg.behavior_joint_enabled)
        if use_joint:
            memory_monitor.set_stage("M8+Abs 联合行为 SFT")
            stage_behavior_sft(model, tokenizer, bundle, cfg)
            clear_memory()
        else:
            if preset.get("M8"):
                memory_monitor.set_stage("M8 拒绝 SFT")
                stage_m8(model, tokenizer, bundle, cfg)
                clear_memory()
            if preset.get("Abs"):
                # 上位词语义回退：GRPO 前的类别级回答暖身
                memory_monitor.set_stage("Abs 上位词回退")
                stage_abstraction(model, tokenizer, bundle, cfg)
                clear_memory()
        # 遗忘充分性闸门：仅对启用上位词回退的方法（旗舰 DeepRULE）生效，
        # 行为 SFT 后按实体补训，不改变无 Abs 基线的方法语义
        if preset.get("Abs") and getattr(cfg, "forget_gate_enabled", True) \
                and bundle["forget"]:
            memory_monitor.set_stage("遗忘充分性闸门")
            stage_forget_gate(model, tokenizer, bundle, cfg)
            clear_memory()
        if preset.get("M6"):
            # RULE_lite 走软奖励：关闭可验证项 γ
            if preset.get("soft_reward"):
                cfg.r_gamma = 0.0
            memory_monitor.set_stage("M6 GRPO 修复")
            stage_m6_grpo(model, tokenizer, bundle, cfg, preset)
            # GRPO 后复核：同口径检测 GRPO 是否把策略拉回具体回答，只记录
            if preset.get("Abs") and getattr(cfg, "post_gate_enabled", False) \
                    and bundle["forget"]:
                memory_monitor.set_stage("GRPO 后复核闸门")
                stage_post_grpo_gate(model, tokenizer, bundle, cfg)
                clear_memory()

    memory_monitor.set_stage("保存适配器")
    save_lora_adapter(model, cfg, "final")
    memory_monitor.log_once("训练结束")
    return model, tokenizer

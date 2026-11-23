# -*- coding: utf-8 -*-
"""
模型加载与显存管理模块
============================================================
封装 8B/9B 模型的加载、LoRA 挂载、分层冻结、8 bit 优化器等
工具函数，业务代码只调用这里的高层接口。模型相关口径全部来自
config.py 的 MODEL_SPECS，换模型时无需修改本文件。
"""
import gc
import os
import re
import sys

# 让 PyTorch 显存分配器使用可扩展段，减少长任务中的显存碎片与 OOM 误杀；
# 必须在 CUDA 上下文初始化前设置，用 setdefault 不覆盖外部显式配置；
# Windows 平台该特性不支持（会产生 UserWarning），故仅在 Linux 启用
if sys.platform.startswith("linux"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText,
                          AutoTokenizer)

from config import BASE_DIR, CKPT_DIR


def _resolve_model_path(name):
    """
    把本地相对路径解析为基于 BASE_DIR 的绝对路径，保证不依赖运行时 cwd；
    若本地不存在，则原样返回（视为 HF 模型 id，交给 from_pretrained）。
    """
    if os.path.isabs(name) and os.path.exists(name):
        return name
    if os.path.exists(name):
        return os.path.abspath(name)
    candidate = os.path.join(BASE_DIR, name)
    if os.path.exists(candidate):
        return os.path.normpath(candidate)
    return name


def load_model_and_tokenizer(cfg):
    """
    以 bf16 加载因果语言模型与分词器。
    - 开启梯度检查点：用少量重算换取激活显存大幅下降；
    - 关闭 use_cache：梯度检查点与 KV cache 不能同时启用。
    """
    # 先验模型优先：M0 前置训练后，所有格子从 learned model 启动
    model_path = getattr(cfg, "init_model_path", "") or cfg.model_name
    model_path = _resolve_model_path(model_path)
    # 加载前先清一次，避免上一阶段残留的缓存与新权重在 CPU 侧叠加
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Llama 系列默认没有 pad token，借用 eos 并仅用于填充，不影响答案
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 生成类任务统一左填充，保证批量生成时对齐正确
    tokenizer.padding_side = "left"

    # 多模态检查点（如 Qwen3.5-9B）用图像-文本条件生成类，
    # 纯文本模型（如 Llama）用因果语言模型类；选错类会报架构不匹配
    auto_cls = (AutoModelForImageTextToText if cfg.multimodal
                else AutoModelForCausalLM)
    model = auto_cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        # 低 CPU 内存加载：权重直接落位 GPU，不在 CPU 侧驻留完整副本
        low_cpu_mem_usage=True,
        # 注意力实现由模型规格给出（sdpa 为内置实现，无需额外依赖）
        attn_implementation=cfg.attn_implementation,
    )
    model.config.use_cache = False
    # 模型自带 generation_config 把 max_length 设成了 131072，与代码显式传入的
    # max_new_tokens 同时存在会反复触发 UserWarning；置回框架默认的 None 即可，
    # 实际生成长度仍由 max_new_tokens 决定，生成行为不变
    model.generation_config.max_length = None
    model.gradient_checkpointing_enable()
    # model.activation_checkpointing_enable()
    model.enable_input_require_grads()  # 梯度检查点下 LoRA 反向传播需要
    return model, tokenizer


def attach_lora(model, cfg):
    """挂载 LoRA 适配器：只训练少量低秩矩阵，行为层训练显存几乎不增加。"""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # 目标模块由模型规格给出（Qwen3.5 含线性注意力层的 in_proj/out_proj）
        target_modules=list(cfg.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def get_decoder_layers(model):
    """
    拿到解码器层列表，兼容四种结构：
    1. peft 包装：PeftModel -> LoraModel -> 原始模型；
    2. 多模态条件生成模型（如 Qwen3.5-9B）：文本塔在 .model.language_model.layers；
    3. 普通因果模型（如 LlamaForCausalLM）：.model.layers；
    4. 已经是内部基座模型（如 LlamaModel）：.layers。
    注意 transformers 5.x 中普通模型也带 base_model 属性，
    不能再用 hasattr(base_model) 简单判断是否为 peft。
    """
    m = model
    # 仅当 base_model 下面还挂着 .model 时，才是 peft 的 LoraModel 结构
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    # Qwen3.5 条件生成：m.model 下没有 layers，文本塔在 language_model 里
    if hasattr(m, "model") and hasattr(m.model, "language_model") \
            and hasattr(m.model.language_model, "layers"):
        return m.model.language_model.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "layers"):
        return m.layers
    raise AttributeError("无法定位解码器层，请检查模型结构：%s" % type(m))


def freeze_all(model):
    """冻结模型全部参数。"""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_decoder_layers(model, layer_ids):
    """
    先整体冻结，再只解冻指定编号的解码器层。
    用于 M1 的块级梯度差：没被选中的层完全不更新。
    """
    freeze_all(model)
    layers = get_decoder_layers(model)
    for idx in layer_ids:
        for p in layers[idx].parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("已解冻层:", sorted(layer_ids), "可训练参数量:", trainable)


def make_optimizer(model, lr, cfg):
    """
    构造优化器：优先 8 bit 分页 AdamW（显存约为普通 AdamW 的 1/4），
    若 bitsandbytes 不可用则退回普通 AdamW。
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.use_8bit_optimizer:
        try:
            import bitsandbytes as bnb
            return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
        except Exception as exc:  # 兜底：没装 bnb 也能跑
            print("8bit 优化器不可用，退回 torch AdamW：", exc)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def clear_memory():
    """阶段切换时清理显存与 Python 引用，防止碎片化 OOM。两次 gc 兜底循环引用。"""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def encode_qa(tokenizer, question, answer, max_length):
    """
    把一组问答编码为训练样本：
    - 提示部分的 label 置 -100，只对答案部分计算损失；
    - 返回 input_ids、labels、attention_mask（均为一维张量）。
    """
    prompt = format_question(tokenizer, question)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    # 末尾补 eos，教会模型回答结束
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + answer_ids
    input_ids = input_ids[:max_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids)) + \
        input_ids[min(len(prompt_ids), len(input_ids)):]
    labels = labels[:max_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
    }


def format_question(tokenizer, question):
    """套用模型自带的对话模板，返回纯文本提示。"""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# 常见英文词：这些实体词本身是普通词，不做近形 token 扩展（避免把 ring、
# sing 等普通词也屏蔽），它们仍走完整实体序列的精确屏蔽
_COMMON_EN_TERMS = frozenset([
    "king", "lee", "black", "brown", "smith", "white", "green", "rice",
    "rock", "stone", "wood", "hill", "lake", "ford", "bush", "wood",
])


def _near_token_ids(tokenizer, blocked_words, cfg):
    """
    扫描分词器词表，返回与实体内容词近形的单 token id（覆盖拼写变体）。
    普通实体词（如 king）跳过近形扩展，避免误伤 ring/sing 等常见词。
    """
    from evaluator import _entity_content_terms, _is_near_token
    terms = []
    for word in blocked_words:
        if word:
            for t in _entity_content_terms(word):
                if t not in _COMMON_EN_TERMS:
                    terms.append(t)
    if not terms:
        return set()
    ids = set()
    for tok_str, tok_id in tokenizer.get_vocab().items():
        # 去掉 BPE 前导符号（▁/Ġ）与非字母字符，取纯表层词
        surface = tok_str.replace("▁", "").replace("Ġ", "").lower()
        surface = re.sub(r"[^a-z0-9]", "", surface)
        if len(surface) < int(getattr(cfg, "fuzzy_token_min_len", 4)):
            continue
        if surface in _COMMON_EN_TERMS:
            continue
        if any(_is_near_token(et, surface, cfg) for et in terms):
            ids.add(tok_id)
    return ids


def _case_variants(word, cfg):
    """返回需要屏蔽的大小写形态：原文 + 全小写 + 全大写（大小写归一）。"""
    forms = [word]
    if cfg is not None and getattr(cfg, "bad_words_case_insensitive", True):
        for f in (word.lower(), word.upper()):
            if f not in forms:
                forms.append(f)
    return forms


def _bad_word_ids(tokenizer, blocked_words, cfg=None):
    """
    把需要屏蔽的词/短语转成 HF generate 的 bad_words_ids。
      1. 精确序列：同时收录“裸词”与“前导空格”两种分词形态，并按
         bad_words_case_insensitive 补充全小写/全大写形态，堵住全大写漏答；
      2. 近形单 token（cfg.fuzzy_block_vocab 开启时）：覆盖词表中与实体
         内容词近形的拼写变体，堵住拼写洗白。
    无有效词时返回空列表（不向 generate 传该参数）。
    """
    ids = []
    for word in blocked_words:
        if not word:
            continue
        for form in _case_variants(word, cfg):
            for text in (form, " " + form):
                tok = tokenizer.encode(text, add_special_tokens=False)
                if tok and tok not in ids:
                    ids.append(tok)
    if cfg is not None and getattr(cfg, "fuzzy_block_vocab", True):
        for one in _near_token_ids(tokenizer, blocked_words, cfg):
            if [one] not in ids:
                ids.append([one])
    return ids


def generate_texts(model, tokenizer, questions, cfg, max_new_tokens=None,
                   do_sample=False, num_return_sequences=1, blocked_words=None):
    """
    逐条问题生成回答（单条处理，避免批处理 padding 带来的额外显存）。
    do_sample=False 时为贪心解码（评测用）；
    do_sample=True 时可一次返回多条采样（GRPO 用）。
    blocked_words 支持两种形态：
      - 单个屏蔽词列表（如 ["Basil ..."]）：对本批所有问题生效；
      - 与 questions 等长的逐题列表，元素为屏蔽词列表或 None：
        多主体/评测 forget 场景每题屏蔽各自实体。
    经 bad_words_ids 约束解码（精确+近形）；默认 None 不屏蔽。
    """
    model.eval()
    results = []
    max_new = max_new_tokens or cfg.grpo_max_new_tokens
    # 采样时的分块大小：组内多条回答分小批生成，压低 KV cache 峰值显存
    gen_batch = max(1, int(getattr(cfg, "grpo_gen_batch", 2)))
    # 统一的去退化参数（重复惩罚 + no-repeat n-gram），贪心与采样都生效
    anti_repeat = dict(
        repetition_penalty=float(getattr(cfg, "repetition_penalty", 1.0)),
        no_repeat_ngram_size=int(getattr(cfg, "no_repeat_ngram_size", 0)))
    # no_repeat_ngram_size=0 表示不启用该约束
    if anti_repeat["no_repeat_ngram_size"] <= 0:
        anti_repeat.pop("no_repeat_ngram_size")
    # 归一为逐题屏蔽列表：显式逐题（元素为列表/None）则原样使用，
    # 否则把同一屏蔽列表应用到所有问题
    if blocked_words and len(blocked_words) == len(questions) and all(
            b is None or isinstance(b, (list, tuple)) for b in blocked_words):
        per_blocks = [list(b) if b else None for b in blocked_words]
    elif blocked_words:
        per_blocks = [list(blocked_words) for _ in questions]
    else:
        per_blocks = [None for _ in questions]
    for question, block in zip(questions, per_blocks):
        # 该题屏蔽词转 token 序列（精确+近形），无屏蔽时为空
        bad_ids = _bad_word_ids(tokenizer, block, cfg) if block else []
        prompt = format_question(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        if do_sample and num_return_sequences > 1:
            # 分块采样后拼接，等效一次取 num_return_sequences 条，峰值显存按块计
            texts = []
            remaining = num_return_sequences
            while remaining > 0:
                n_here = min(gen_batch, remaining)
                gen_kwargs = dict(
                    max_new_tokens=max_new, do_sample=True,
                    num_return_sequences=n_here,
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=cfg.grpo_temperature, top_p=0.95,
                    **anti_repeat)
                if bad_ids:
                    gen_kwargs["bad_words_ids"] = bad_ids
                with torch.no_grad():
                    output = model.generate(**inputs, **gen_kwargs)
                gen_ids = output[:, inputs["input_ids"].shape[1]:]
                texts.extend(tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False))
                del output, gen_ids
                torch.cuda.empty_cache()
                remaining -= n_here
            results.append(texts)
        else:
            # 贪心或单条：单条生成，评测用
            gen_kwargs = dict(
                max_new_tokens=max_new, do_sample=do_sample,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                **anti_repeat)
            if bad_ids:
                gen_kwargs["bad_words_ids"] = bad_ids
            if do_sample:
                gen_kwargs["temperature"] = cfg.grpo_temperature
                gen_kwargs["top_p"] = 0.95
            with torch.no_grad():
                output = model.generate(**inputs, **gen_kwargs)
            gen_ids = output[:, inputs["input_ids"].shape[1]:]
            # BPE 分词器不做 WordPiece 式空格清理，同时关掉对应的 UserWarning
            texts = tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            del output, gen_ids
            results.append(texts[0])
        del inputs
    return results


def save_lora_adapter(model, cfg, tag):
    """保存 LoRA 适配器（只含小体积增量权重）；普通全参模型跳过，避免存整套权重。"""
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except ImportError:
        is_peft = False
    if cfg.save_adapter and is_peft:
        # 适配器统一保存到 checkpoints/，不放进本次运行结果目录（按要求隔离）
        out_dir = os.path.join(
            CKPT_DIR, f"{cfg.method_name}_{tag}_seed{cfg.run_seed}")
        model.save_pretrained(out_dir)
        print("LoRA 已保存：", out_dir)

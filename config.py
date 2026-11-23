# -*- coding: utf-8 -*-
"""
全局配置模块
============================================================
本文件集中管理 Deep-RULE 2.0 的全部超参、路径与模型规格表。
旗舰方法为 DeepRULE（深度拒绝遗忘，Deep Refusal Unlearning）：
    M1 密度深删（参数层）→ M8 拒绝引导（行为层）
    → Abs 上位词语义回退 → M6 GRPO 流畅度修复。
其余 GA/NPO/KUnBR/RULE/PURGE/ReRULE/UNI 作为对照基线保留。
目标硬件：8B/9B 正式训练面向 RTX 4090 48GB 云服务器，
本机（5070 Ti 16GB）用于改代码与结果分析。
设计原则：
    1. 模型 bf16 加载，模型口径统一在 MODEL_SPECS 登记；
    2. 行为层训练一律走 LoRA，参数层只解冻被选中的少数层；
    3. 微批大小固定为 1，靠梯度累积模拟大 batch；
    4. 所有实验数据（指标、预测、日志）都落地为 CSV，便于后期画图。
"""
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------
# 当前代码包所在目录（Deep-RULE2_lite/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 实验结果统一放到 results/ 下，后期画图直接读这里的 CSV
RESULT_DIR = os.path.join(BASE_DIR, "results")
# 模型检查点（只保存 LoRA 适配器，体积小）
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
# 先验模型（learned model）目录：全量语料前置训练产物，所有遗忘方法共用同一
# 起点，按「模型-数据集」分子目录，不进 results 运行目录
LEARNED_DIR = os.path.join(CKPT_DIR, "learned")
# 本地模型目录：模型已手动下载到 model/ 下，直接读本地文件夹、不依赖 HF 缓存
# 当前默认模型为 Llama-3.1-8B-Instruct（8B 正式主实验，需在 48GB 云端训练）
LOCAL_MODEL_DIR = os.path.join(BASE_DIR, "model", "Llama-3.1-8B-Instruct")
# 数据根目录与三个基准的本地目录（资源均手动下载，直接读本地文件）
DATA_ROOT = os.path.join(BASE_DIR, "data")
DATA_DIR = os.path.join(DATA_ROOT, "TOFU")       # TOFU：JSONL
DUET_DIR = os.path.join(DATA_ROOT, "DUET")       # DUET：parquet
WMDP_DIR = os.path.join(DATA_ROOT, "WMDP")       # WMDP：parquet 选择题
RWKU_DIR = os.path.join(DATA_ROOT, "RWKU")       # RWKU：JSON（Target/ + All/）

# ------------------------------------------------------------
# 运行环境开关（PyCharm 直接运行，无需手动设置环境变量）
# ------------------------------------------------------------
# True = 离线模式：模型与数据集只从本地缓存/本地目录读取，不发起网络请求。
# 模型和 TOFU 已提前下载好时保持 True 即可；
# 若需要联网拉取新资源，把该开关改为 False（或在 PyCharm 运行配置里
# 预先设置同名环境变量，setdefault 不会覆盖已有的环境变量）。
OFFLINE_MODE = True

if OFFLINE_MODE:
    # HF_HUB_OFFLINE：huggingface_hub/transformers 离线，禁止联网校验与下载
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # HF_DATASETS_OFFLINE：datasets 库离线，load_dataset 只读本地缓存
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # Windows 无符号链接权限时抑制无害警告
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# ------------------------------------------------------------
# 硬件档位（Hardware Profile）
# ------------------------------------------------------------
# 经确认，本地与云端统一使用同一套训练参数（即原 local 参数），不按显存
# 做保守裁剪；云端通过 8bit 分页优化器把优化器状态卸载到 CPU 内存兜底。
# 保留 local/cloud 两个档位名，仅为兼容命令行 --profile，二者取值完全一致。
# 路径全部基于 BASE_DIR（当前文件夹），整个目录拷到云端即可运行，无需改路径。
_UNIFIED_PROFILE = {
    "max_length": 256,            # 问答拼接最大长度
    "m1_top_k": 3,                # 同时解冻的密度块数（6→3，少删几层降低破坏性）
    "grpo_group_size": 4,         # GRPO 每个提示并行生成的序列数
    "grpo_gen_batch": 2,          # 组内分小批生成，每批 2 条，压低 KV cache 峰值
    "grpo_max_new_tokens": 64,    # 生成回答最大长度
    "monitor_interval": 5.0,      # 内存监控采样间隔（秒）
    "recon_population": 24,       # 知识重建候选驻留数
    "use_8bit_optimizer": True,   # 8bit 分页优化器
}

HARDWARE_PROFILES = {
    "local": dict(_UNIFIED_PROFILE),
    "cloud": dict(_UNIFIED_PROFILE),
}


# ------------------------------------------------------------
# 模型规格表（Model Specs）
# ------------------------------------------------------------
# 换模型时在此登记一行，业务代码按目录名自动取规格，无需再到处改层数与模块名。
# 字段说明：
#   family：架构族名，用于分支判断；
#   n_layers：解码器层数，0 表示运行时从模型实测；
#   m1_n_blocks：M1 把全部层切成多少块；
#   lora_target_modules：LoRA 挂载的线性层后缀（按后缀匹配，PEFT 自动定位）；
#   multimodal：是否为多模态条件生成模型（决定加载用哪个 Auto 类）；
#   attn_implementation：注意力实现，sdpa 为内置实现、无需额外依赖。
# Llama 标准投影：注意力 q/k/v/o + MLP gate/up/down
_LLAMA_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj")
# Qwen3.5 为混合架构：全注意力层用 self_attn 的 q/k/v/o，
# 线性注意力层（GatedDeltaNet）用 in_proj_* 与 out_proj；MLP 两层共用
_QWEN35_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                        "in_proj_qkv", "in_proj_a", "in_proj_b",
                        "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj")

MODEL_SPECS = {
    "Llama-3.1-8B-Instruct": {
        "family": "llama",
        "n_layers": 32,
        "m1_n_blocks": 8,           # 32 层 / 8 块 = 4 层一块
        "lora_target_modules": _LLAMA_LORA_TARGETS,
        "multimodal": False,
        "attn_implementation": "sdpa",
    },
    "Qwen3.5-9B": {
        "family": "qwen3_5",
        "n_layers": 32,
        "m1_n_blocks": 8,
        "lora_target_modules": _QWEN35_LORA_TARGETS,
        "multimodal": True,         # 检查点为 Qwen3_5ForConditionalGeneration
        "attn_implementation": "sdpa",
    },
}

# 未登记模型的兜底规格：层数运行时实测，LoRA 按 Llama 风格模块名匹配
DEFAULT_MODEL_SPEC = {
    "family": "unknown",
    "n_layers": 0,
    "m1_n_blocks": 8,
    "lora_target_modules": _LLAMA_LORA_TARGETS,
    "multimodal": False,
    "attn_implementation": "sdpa",
}


def resolve_model_spec(model_name):
    """按模型目录名取规格副本；未登记的模型返回兜底规格。"""
    tag = os.path.basename(os.path.normpath(model_name))
    return dict(MODEL_SPECS.get(tag, DEFAULT_MODEL_SPEC))



@dataclass
class Config:
    """实验配置：数据类写法，改超参只需要动这里，不用翻业务代码。"""

    # ---------- 模型 ----------
    # 默认读取本地 model/ 目录下手动下载的模型（离线可用）；
    # 若以后换模型，把这里改成新的本地文件夹路径或 HF 模型名，
    # 并在 MODEL_SPECS 登记规格（层数、LoRA 模块、架构族等）
    model_name: str = LOCAL_MODEL_DIR
    # 模型层数（Llama-3.1-8B 为 32 层），由 apply_model_spec 按规格写入，
    # 未登记模型在运行时按实际解码器层数核对
    n_layers: int = 32
    # 模型规格字段（apply_model_spec 填充）：架构族、LoRA 目标、多模态、注意力实现
    model_family: str = ""
    lora_target_modules: tuple = ()
    multimodal: bool = False
    attn_implementation: str = "sdpa"
    # 运行期字段：先验模型（全量语料学完后的检查点）目录；非空时所有加载
    # （含 Original 基准）都从这里启动，保证“先学会再遗忘”
    init_model_path: str = ""

    # ---------- 数据 ----------
    # 基准选择：tofu / duet；WMDP 当前为选择题评测通道（见 wmdp_*）
    dataset_name: str = "tofu"
    # TOFU 遗忘配置：forget01 / forget05 / forget10，对应遗忘 1%/5%/10%
    tofu_forget_config: str = "forget01"
    tofu_retain_config: str = "retain99"
    # DUET 各划分文件名（不含分片后缀，加载时按前缀匹配 parquet）
    duet_forget_file: str = "forget_popular_5"
    duet_retain_file: str = "retain_intersection_500"
    duet_boundary_file: str = "holdout_popular_object_neighbours"
    duet_dup_file: str = "paraphrases_city_forget_popular_5"  # 不存在则复制探针留空
    # DUET 遗忘集构造口径：少数主体 × 全部关系（与 TOFU 每实体多题对齐）。
    # 只选关系数落在 [min,max] 区间的主体，每个主体贡献其全部关系，
    # 累计题数达到 max_samples 即停止，默认约 6 个主体、每主体 5–10 题
    duet_forget_n_subjects: int = 6
    duet_min_relations: int = 5
    duet_max_relations: int = 10
    # RWKU 口径：选少数真实名人目标，每个目标贡献其全部 level2 问答，
    # 累计达到 max_samples 即停；职业上位词按 intro 文本逐目标分类
    rwku_n_targets: int = 6
    rwku_forget_level: str = "2"
    rwku_retain_file: str = "triviaqa"
    rwku_default_hypernym: str = "a public figure"
    # WMDP 选择题科目与评测上限（仅 test 集，用于危险知识 MC 准确率）
    wmdp_subjects: tuple = ("bio", "chem", "cyber")
    wmdp_max_questions: int = 200
    # 每个划分最多取多少条样本（默认 40 条，smoke 模式会再砍半）
    max_samples: int = 40
    # 边界集（伪邻居实体）的样本上限
    max_boundary: int = 24
    # 复制探针上限
    max_dup: int = 24

    # ---------- M0：先验学习（全量语料前置训练，先学会再遗忘）----------
    # 全参微调（非 LoRA），产出完整 learned model 供全部方法共用
    pretrain_lr: float = 2.0e-5       # 全参学习率，知识注入常用 1e-5~2e-5 量级
    pretrain_epochs: int = 0          # 0=自动：TOFU 3 轮、DUET 1 轮
    pretrain_max_samples: int = 0     # 0=全量；冒烟模式自动覆盖为小样本
    # M0 学会验收闸门：训练后在遗忘/保留探针上判正确率，不达标自动加轮重训
    learn_gate_enabled: bool = True
    learn_gate_threshold: float = 0.6   # 探针综合正确率达到该值才放行遗忘
    learn_gate_probe_size: int = 100    # 探针总条数（forget+retain 各半）
    learn_gate_max_attempts: int = 3    # 含首次训练在内的最大尝试次数
    # 闸门是否只考核 forget 侧：RWKU 的 retain（triviaqa）金标噪声大、判据偏严，
    # 会把已学会目标名人的模型误判为不达标，故 RWKU 只看遗忘侧正确率。
    learn_gate_forget_only: bool = False

    # 退化护栏：训练后若保留能力或 WMDP 相对 Original 基线掉幅超过阈值，
    # 判该格失败、不计入成功，杜绝“模型崩了却被当成精确遗忘”的假成功
    degradation_guard_enabled: bool = True
    degradation_guard_threshold: float = 0.15  # 相对掉幅 15%
    degradation_guard_metrics: tuple = (
        "retain_acc", "wmdp_bio_acc", "wmdp_chem_acc", "wmdp_cyber_acc")
    # 基线低于该下限时相对掉幅无意义（如 RWKU retain 基线仅 0.075），
    # 豁免该指标，杜绝地板效应假失败
    degradation_guard_baseline_floor: float = 0.1
    # 判失败还需绝对掉幅超过该下限，避免小基线上的微小绝对变化
    # 被放大成高相对掉幅而误判
    degradation_guard_abs_floor: float = 0.05

    # ---------- 通用训练 ----------
    max_length: int = 256          # 问答拼接后的最大 token 长度
    micro_batch: int = 1           # 单次前向批大小（16GB 显存建议保持 1）
    grad_accum: int = 8            # 梯度累积步数，等效 batch≈8
    num_epochs: int = 2            # 全参/块训练的轮数
    lr_full: float = 1.5e-6        # 参数层学习率（小模型口径，比 8B 常用 1.5e-7 放大 10 倍）
    use_8bit_optimizer: bool = True  # 8 bit 分页 Adam，显著省显存
    lora_r: int = 32               # LoRA 秩（8B 模型 r16 学习信号过弱，提到 32）
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # ---------- M1：KUnBR 密度块删除 ----------
    m1_n_blocks: int = 8           # 把 16 层切成 8 块（每块 2 层）
    m1_top_k: int = 3              # 选密度最高的 3 块（6→3，降低深删破坏性）
    m1_warmup_steps: int = 6       # 全参数预热步数（论文原方案 24，本实现缩到 6）
    m1_retain_lambda: float = 0.1  # 保留集损失权重 λ
    m1_lr: float = 1.5e-6

    # ---------- M2：边界余量与关键样本 ----------
    m2_critical_ratio: float = 0.2  # φ 最小的 20% 判为关键样本
    m2_critical_weight: float = 2.0  # 关键样本在训练中的加权倍数
    m2_delta: float = 0.1           # 左右极限对的概率差阈值 Δ

    # ---------- M4：递减课程 ----------
    m4_epochs: int = 4
    m4_kappa: float = 0.5           # w_e = 1/(1+κe) 的 κ

    # ---------- M8：拒绝引导 SFT ----------
    m8_lr: float = 2.0e-5
    m8_epochs: int = 4
    # M8/Abs 监督训练每个遗忘样本使用的问题变体数（含原问题）。
    # 官方改写 + 规则改写扩样本，把个位数优化器更新提升到数十次。
    sft_n_question_variants: int = 3
    # 短答案（词数不超过该值，即裸值型）是抽象薄弱点，其抽象样本按
    # sft_short_answer_weight 复制加权，不增加训练轮数即可把模型压向类别句
    sft_short_answer_words: int = 4
    sft_short_answer_weight: int = 2

    # ---------- Abs：上位词语义回退（Abstraction Fallback）----------
    # 深删+拒绝易导致幻觉或无脑拒答；用类别级上位词替换具体实体，
    # 既不泄露目标身份，又保留类别层面的连贯信息。
    use_abstraction: bool = True       # 是否启用上位词回退
    abstract_default_word: str = "an author"  # TOFU 目标实体均为虚构作者，默认上位词
    abs_lr: float = 1.0e-5            # 上位词回退 SFT 学习率（数据已干净，降 lr 防压崩输出）
    abs_epochs: int = 1               # 回退 SFT 轮数（40 条不同抽象句，1 轮即可，避免过训）
    r_abstraction: float = 0.6        # GRPO 中上位词回答的奖励权重（0.25→0.6，使抽象奖励高于拒答）

    # ---------- M8+Abs 联合行为 SFT（让抽象回退主导，而非先强拒答后弱抽象）----------
    # 同一批遗忘问题先 M8 教拒答、再 Abs 教抽象且 M8 更强，会导致拒答主导、
    # abstraction_rate=0。改为单阶段联合训练：抽象样本占多数、拒答样本占少数，
    # 一次更新内同时学习两种行为，靠配比而非加轮数让抽象回退成为首选。
    behavior_joint_enabled: bool = True       # M8 与 Abs 同时存在时启用联合 SFT
    behavior_abstraction_fraction: float = 0.75  # 联合样本中抽象回退所占比例（多数）
    behavior_lr: float = 2.5e-5               # 联合 SFT 学习率（1.5e-5→2.5e-5，增强有效学习信号）
    behavior_epochs: int = 1                  # 联合 SFT 轮数（单轮，不增加轮数）
    # 联合 SFT 专用梯度累积步数（全局 8→4）：160 条样本的优化器更新
    # 从 20 次提到 40 次，不加轮数即可跨过基座强直答的翻转阈值。
    behavior_grad_accum: int = 4

    # ---------- 遗忘充分性闸门（Forgetting Sufficiency Gate）----------
    # 行为 SFT 后、GRPO 前，按遗忘实体抽查“是否仍直答”：某实体直答率
    # 超过阈值时，只对该实体做定向抽象补训（不整体重跑、不加 GRPO 轮数）；
    # 有限次后仍不达标也只记录不阻断，剩余翻转交给 GRPO。
    forget_gate_enabled: bool = True
    forget_gate_concrete_threshold: float = 0.5  # 直答率超过该值即触发补训
    forget_gate_probe_per_entity: int = 8        # 每个实体抽查题数上限
    forget_gate_max_attempts: int = 2            # 含首次抽查在内的最大轮数
    forget_gate_boost_weight: int = 2            # 补训时该实体抽象样本复制倍数
    forget_gate_lr: float = 2.5e-5               # 定向补训学习率（默认同行为 SFT）

    # ---------- M6：精简 GRPO 可验证奖励 ----------
    grpo_steps: int = 50            # RL 步数（8B 口径 20 步信号不足，提到 50）
    grpo_group_size: int = 4        # 每个提示采样 4 条回答（原文 8，减半省显存）
    grpo_gen_batch: int = 2         # 组内分小批生成，每批 2 条，压低峰值显存
    grpo_lr: float = 3.0e-5
    grpo_max_new_tokens: int = 64   # 生成回答的最大长度，短回答省显存
    monitor_interval: float = 5.0   # 内存监控采样间隔（秒）
    grpo_temperature: float = 0.8
    r_alpha: float = 0.4            # 拒绝项权重（0.5→0.4，让干净抽象回退的总奖励高于拒答）
    r_beta: float = 0.3             # 实体/相似度项权重
    r_gamma: float = 0.2            # 可验证项（禁忌实体不相交）权重
    # 裸值短答（非抽象）的 r_g 缩减系数：抽象完整句给全额，裸值只给零头，
    # 拉大两者奖励差，把模型从“照常给具体值”推向类别级抽象
    r_concrete_retain_factor: float = 0.25
    # 无谓语片段（如 "New York City"）的标记分系数：低于抽象判定下限，
    # 既不计入抽象，也让 GRPO 无法靠短片段骗取满分
    abstraction_fragment_score: float = 0.30
    # KL 惩罚系数（0.01→0.05→0.025）：0.05 锚定过强导致强实体欠遗忘，
    # 回到中间值 0.025，既抑制自由采样向短片段的模式坍缩，又保留充分翻转空间
    kl_beta: float = 0.025
    rouge_threshold: float = 0.5    # 回答判定为正确的 ROUGE-L 阈值
    # 生成去退化：重复惩罚抑制已出现 token、no-repeat n-gram 阻断短语循环，
    # 消除 "TheTheThe…" 类退化重复（对贪心与采样生成同时生效）
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    # GRPO 遗忘提示生成时屏蔽目标实体（bad_words_ids 约束解码）：RL 探索
    # 阶段发不出目标名，避免自由采样产出“关系词+实体名”的带名漏答
    grpo_block_forget_entity: bool = True

    # 模糊实体检测：精确匹配识别不了拼写变体（Basil Mahafouz、Al-Qwaiti），
    # 用编辑距离/前缀近似识别“拼写洗白”，让奖励与实质泄漏判定都能扣分
    fuzzy_entity_detection: bool = True
    fuzzy_edit_distance: int = 1       # 近形词允许的最大编辑距离（绝对值）
    fuzzy_ratio: float = 0.3           # 编辑距离/词长小于该比例也算近形（长词容错）
    fuzzy_token_min_len: int = 4       # 参与模糊匹配的最短词（短词易误判）
    fuzzy_match_threshold: float = 0.5  # 实体内容词近形命中比例达到该值即算出现
    # 屏蔽词扩展到词表中的近形 token：除精确实体序列外，把词表里与实体
    # 内容词近形的 token 一并加入 bad_words，堵住拼写变体的生成路径
    fuzzy_block_vocab: bool = True
    # 最终评测 forget 推理时也启用精确+模糊屏蔽，防止贪心输出带名（含变体）
    inference_block_forget_entity: bool = True

    # 奖励层加压：非拒答时对实体出现施加惩罚，拼写变体（近形洗白）额外重罚，
    # 让奖励排序稳定为 干净抽象 > 拒答 > 精确漏答 > 拼写变体
    r_leak_penalty: float = 0.30       # 实体泄漏（精确/近形）基础惩罚权重
    r_fuzzy_penalty: float = 0.35      # 近形拼写洗白的额外惩罚（变体应最不受益）
    # 大小写归一：屏蔽词补充大小写变体，堵住全大写（BASIL MAHFOUZ）漏答
    bad_words_case_insensitive: bool = True
    # 幻觉具体名识别：非目标的具体人名/专名（如 Leila Al-Sabah）不算第三状态
    hallucinated_name_detection: bool = True
    r_hallucination_penalty: float = 0.30  # 幻觉具体名的奖励惩罚权重

    # 奖励塑形：在不改变“抽象>拒答>漏答>变体”粗排序的前提下，叠加一个有界
    # 的连续质量项，打破组内 4 条同类回答奖励完全并列（std=0、策略项无梯度）
    reward_shaping_enabled: bool = True
    r_shaping: float = 0.03        # 塑形幅度，奖励仅在 ±0.03 内微调，不足以翻转类别
    # GRPO 后复核闸门：用与遗忘闸门相同的模糊口径在 GRPO 后再探一次直答率，
    # 识别 GRPO 是否把策略拉回具体回答；只记录不阻断（GRPO 为最后训练阶段）
    post_gate_enabled: bool = True

    # ---------- M7：硬样本回放 ----------
    m7_tau: float = 0.5             # 组均奖励低于该值才入缓冲
    m7_buffer_min: int = 16         # 缓冲达到该数量才启用离线更新
    m7_warmup: int = 10             # 前多少步只做在线更新
    m7_clip: float = 0.2            # 重要性比裁剪范围 [1-clip, 1+clip]

    # ---------- C2：RTT 恢复攻击 ----------
    rtt_ratios: tuple = (0.05, 0.10, 0.20)
    rtt_steps: int = 8              # 每档 RTT 的 LoRA 微调步数

    # ---------- C5：知识重建（遗传搜索） ----------
    recon_population: int = 24      # 种群大小
    recon_generations: int = 12    # 重建候选生成数（论文原方案 200，本实现 12）
    recon_pool_limit: int = 60     # 候选答案池上限

    # ---------- 运行 ----------
    # 硬件档位：local/cloud 两档参数完全相同（见 HARDWARE_PROFILES），
    # 保留字段仅为兼容命令行 --profile
    hardware_profile: str = "local"
    seeds: tuple = (42,)
    smoke: bool = False             # 冒烟模式：样本与步数再减半，先验证流水线
    save_adapter: bool = True       # 是否保存 LoRA 检查点

    # 运行期字段：本次方法名与种子，由入口脚本写入
    method_name: str = ""
    run_seed: int = 42
    # 本次运行的独立结果目录（入口脚本创建后写入）；为空则回退到 results/
    run_dir: str = ""

    def __post_init__(self):
        """先应用硬件档位，再应用冒烟模式。"""
        # 1) 硬件档位：按 local/cloud 逐键覆盖参数（两档取值相同，见 _UNIFIED_PROFILE）
        profile = HARDWARE_PROFILES.get(str(self.hardware_profile).lower())
        if profile is None:
            raise ValueError(
                "未知硬件档位 hardware_profile=%s，可选：local、cloud"
                % self.hardware_profile)
        for key, val in profile.items():
            setattr(self, key, val)
        # RWKU 闸门只考核 forget 侧（retain 金标噪声大，避免拖垮学会验收）
        if str(self.dataset_name).lower() == "rwku":
            self.learn_gate_forget_only = True
        # 2) 冒烟模式：在硬件档位基础上把样本量和训练步数整体砍半
        if self.smoke:
            self.max_samples = 8
            self.max_boundary = 6
            self.max_dup = 6
            self.m1_warmup_steps = 2
            self.grpo_steps = 4
            self.m7_warmup = 2
            self.m7_buffer_min = 4
            self.num_epochs = 1
            self.rtt_steps = 2
            self.recon_generations = 4
            self.wmdp_max_questions = 24
            # 扩样本 SFT 的冒烟口径：少变体、单轮，快速跑通
            self.sft_n_question_variants = 2
            self.m8_epochs = 1
            self.abs_epochs = 1
            # 验收闸门冒烟口径：小探针、少尝试，快速验证整条链路
            self.learn_gate_probe_size = 16
            self.learn_gate_max_attempts = 2
            # 遗忘充分性闸门冒烟口径：每实体少抽查，快速验证补训链路
            self.forget_gate_probe_per_entity = 4
        # 3) 按模型名应用规格（层数、块数、LoRA 目标、架构族等）
        self.apply_model_spec()
        for d in (RESULT_DIR, CKPT_DIR, LEARNED_DIR):
            os.makedirs(d, exist_ok=True)

    def apply_model_spec(self):
        """
        按当前 model_name 从 MODEL_SPECS 取规格并写入本对象。
        入口脚本在构造后若又改了 model_name，必须再调用一次本方法。
        """
        spec = resolve_model_spec(self.model_name)
        self.model_family = spec["family"]
        self.multimodal = spec["multimodal"]
        self.attn_implementation = spec["attn_implementation"]
        self.lora_target_modules = tuple(spec["lora_target_modules"])
        self.m1_n_blocks = spec["m1_n_blocks"]
        # n_layers=0 表示未登记模型，保留运行时实测；这里先不覆盖
        if spec["n_layers"]:
            self.n_layers = spec["n_layers"]

    def path(self, filename):
        """拼接本次运行目录下的文件路径；未指定运行目录时回退到 results/。"""
        base = self.run_dir if self.run_dir else RESULT_DIR
        return os.path.join(base, filename)


# ------------------------------------------------------------
# 通用工具函数
# ------------------------------------------------------------
def set_seed(seed):
    """固定所有随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def append_csv(filepath, rows):
    """
    把一批字典记录追加写入 CSV。
    - 文件不存在时写表头，存在时只追加数据；
    - 使用 utf-8-sig，Excel 直接打开中文不乱码；
    - 追加前校验表头：若本次记录的列与已有文件不一致（例如新增了 dataset/model 字段），
      先把旧文件改名为 .bak 保留，再按新表头重写，避免列错位污染历史数据。
    """
    if not rows:
        return
    df = pd.DataFrame(rows)
    if os.path.exists(filepath):
        old_cols = list(pd.read_csv(filepath, nrows=0).columns)
        if old_cols != list(df.columns):
            backup = filepath.replace(
                ".csv", ".bak_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
            try:
                os.replace(filepath, backup)
                print("列结构已变更，旧结果已备份为：%s" % os.path.basename(backup))
            except OSError:
                # 旧文件可能被 Excel 占用，此时直接覆盖写新表头
                print("警告：旧结果无法备份（可能被占用），将直接覆盖：%s" % filepath)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            return
        df.to_csv(filepath, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(filepath, index=False, encoding="utf-8-sig")


def write_csv(filepath, rows):
    """整表覆盖写入 CSV（用于数据划分等需要重算的场景）。"""
    pd.DataFrame(rows).to_csv(filepath, index=False, encoding="utf-8-sig")


def _sanitize_name(text):
    """清理 Windows 文件夹名中的非法字符；逗号统一换成加号。"""
    for ch in '\\/:*?"<>|':
        text = text.replace(ch, "_")
    return text.replace(",", "+").strip()


def learned_model_dir(model_name, dataset_name, smoke=False):
    """
    返回先验模型目录：checkpoints/learned/<模型名去Instruct>-<数据集>[-smoke]。
    冒烟产物单独后缀，避免与正式 learned model 混用。
    """
    model_tag = os.path.basename(os.path.normpath(model_name))
    if model_tag.endswith("-Instruct"):
        model_tag = model_tag[:-len("-Instruct")]
    suffix = "-smoke" if smoke else ""
    name = "%s-%s%s" % (_sanitize_name(model_tag),
                        _sanitize_name(str(dataset_name).lower()), suffix)
    return os.path.join(LEARNED_DIR, name)


def make_run_dir(model_name, method_label, seed_label):
    """
    每次运行在 results/ 下创建一个独立文件夹，命名规则：
        模型名-方法标签-种子数 年-月-日 时-分-秒
    时间用连字符（冒号在 Windows 文件夹名中非法）；同一秒内重复运行追加序号。
    模型名末尾的 "-Instruct" 去掉，使目录名简洁（如 Llama-3.1-8B）。
    """
    model_tag = os.path.basename(os.path.normpath(model_name))
    if model_tag.endswith("-Instruct"):
        model_tag = model_tag[:-len("-Instruct")]
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    base_name = "%s-%s-%s %s" % (
        _sanitize_name(model_tag),
        _sanitize_name(str(method_label)),
        _sanitize_name(str(seed_label)),
        stamp,
    )
    run_dir = os.path.join(RESULT_DIR, base_name)
    candidate, suffix = run_dir, 2
    while os.path.exists(candidate):
        candidate = os.path.join(RESULT_DIR, "%s %d" % (base_name, suffix))
        suffix += 1
    os.makedirs(candidate, exist_ok=False)
    return candidate

# -*- coding: utf-8 -*-
"""
数据管线模块
============================================================
负责 TOFU 数据集加载、五分划分（遗忘/保留/边界/复制/关键），
并把所有划分与 margin 计算结果落地为 CSV。
同时提供 DeepRULE 上位词语义回退所需的实体泛化工具。
不调用任何外部大模型，边界集、复制探针与上位词全部本地可复现。
"""
import functools
import json
import os
import re
from collections import Counter

import torch

from config import DATA_DIR, DUET_DIR, RWKU_DIR, write_csv
from model_utils import format_question

# 实体抽取时需要排除的句首常见词（大写但不是人名）
STOP_WORDS = {
    "The", "What", "When", "Where", "Who", "How", "This", "That", "These",
    "Those", "In", "On", "At", "She", "He", "Her", "His", "Their", "They",
    "It", "Its", "A", "An", "After", "Before", "During", "While", "Born",
}

# 边界集使用的伪邻居人名（固定列表，保证可复现）
PSEUDO_NAMES = ["Avery Brooks", "Jordan Lee", "Casey Morgan", "Riley Quinn",
                "Taylor Morgan", "Cameron Reed"]

# 拒绝引导模板（M8 使用），{entity} 运行时替换为目标实体
REFUSAL_TEMPLATES = [
    "I don't have information about {entity}.",
    "I'm sorry, but I can't provide details about {entity}.",
    "I don't know anything related to {entity}.",
]

# 上位词默认短语：TOFU 遗忘目标均为虚构作者，类别级回退为“一位作者”
DEFAULT_HYPERNYM = "an author"


def hypernym_of(entity, cfg=None):
    """
    返回目标实体的类别级上位词。
    预留按实体类型查表的扩展位；当前统一返回作者类上位词。
    """
    word = DEFAULT_HYPERNYM
    if cfg is not None:
        word = getattr(cfg, "abstract_default_word", DEFAULT_HYPERNYM)
    return word


# 关系名词集合：“the country of X / the capital of X”这类结构，
# 若把 X 原位替换成类别词会得到“the country of a place”这种语义循环，
# 必须整句重写
_RELATION_HEAD = re.compile(
    r"\b(country|capital|city|state|county|nationality|birthplace|hometown|"
    r"name|location|address|citizenship)\b", re.IGNORECASE)
_ARTICLES = {"a", "an", "the"}


def _replacement_safe(answer, matches, hypernym):
    """
    逐个核对每个实体替换点的上下文，全部安全才允许原位替换。
    任一替换点落入下列情况即判不安全，调用方应整句重写：
      1. 实体紧前是字母数字或连字符：实体只是更大词/复合名的片段
         （如 “Al-Kuwaiti”里的姓氏 Kuwaiti），替换会切断复合名；
      2. 实体紧前是冠词，而类别词本身又带冠词（a/an），会叠成双冠词；
      3. 实体处于“关系名词 + of + 实体”结构，替换后语义循环；
      4. 实体紧前是与上位词同类的头衔词（如 “Author X” 对 “an author”），
         替换后会得到 “Author an author” 这类类别叠用。
    """
    hyper_has_article = hypernym.lstrip().lower().startswith(("a ", "an "))
    hyper_noun = hypernym.split()[-1].lower()
    for m in matches:
        before = answer[:m.start()]
        if m.start() > 0:
            prev_ch = answer[m.start() - 1]
            if prev_ch.isalnum() or prev_ch == "-":
                return False
        prev_words = re.findall(r"[A-Za-z]+", before)
        if prev_words and hyper_has_article \
                and prev_words[-1].lower() in _ARTICLES:
            return False
        if prev_words and prev_words[-1].lower() == hyper_noun:
            return False
        if prev_words and prev_words[-1].lower() == "of" \
                and _RELATION_HEAD.search(before):
            return False
    return True


# ============================================================
# 关系感知整句重写
# ------------------------------------------------------------
# 短答案场景（主体只在问题里、不在答案中，如 DUET 国家题、RWKU 名人题）
# 无法做原位实体替换，旧逻辑统一输出 "The answer is ..."，导致全部回答
# 退化为同一句、不保留关系。这里改为保留问题中的关系、把具体值抽象为
# 类别词，使各条回答彼此不同且携带关系信息。
# ============================================================

# DUET 官方关系标签 -> (关系主语短语, 宾语类别词)
DUET_RELATION_SPEC = {
    "ethnic group": ("The ethnic group", "a demographic group"),
    "part of": ("The larger region it belongs to", "a geographic region"),
    "official language": ("The official language", "a spoken language"),
    "shares border with": ("A neighboring place", "a country"),
    "language used": ("The language used", "a spoken language"),
    "legislative body": ("The legislative body", "a governing body"),
    "lowest point": ("The lowest point", "a geographic location"),
    "replaces": ("The entity it replaced", "a historical entity"),
    "basic form of government": ("The form of government", "a governing system"),
    "contains the administrative territorial entity": (
        "An administrative region within it", "a subnational region"),
    "capital": ("The capital", "a city"),
    "main regulatory text": ("The main regulatory text", "a legal document"),
    "head of state": ("The head of state", "a person"),
    "highest point": ("The highest point", "a geographic location"),
    "located in or next to body of water": (
        "The adjacent body of water", "a maritime location"),
    "continent": ("The continent", "a continental landmass"),
    "currency": ("The currency", "a monetary unit"),
    "member of": ("The organization it belongs to", "an international organization"),
}

# 问题模式正则（按特异性从高到低排列，命中即返回），用于 RWKU 自由问句
# 以及无官方关系标签时的兜底：(编译正则, 关系主语短语, 宾语类别词)
_RELATION_PATTERNS = [
    (re.compile(r"pseudonym", re.IGNORECASE), "The pseudonym used", "a pen name"),
    (re.compile(r"official language", re.IGNORECASE),
     "The official language", "a spoken language"),
    (re.compile(r"legislative body", re.IGNORECASE),
     "The legislative body", "a governing body"),
    (re.compile(r"regulatory text", re.IGNORECASE),
     "The main regulatory text", "a legal document"),
    (re.compile(r"administrative region", re.IGNORECASE),
     "An administrative region within it", "a subnational region"),
    (re.compile(r"shares? a border|border with", re.IGNORECASE),
     "A neighboring place", "a country"),
    (re.compile(r"body of water", re.IGNORECASE),
     "The adjacent body of water", "a maritime location"),
    (re.compile(r"highest point", re.IGNORECASE),
     "The highest point", "a geographic location"),
    (re.compile(r"lowest point", re.IGNORECASE),
     "The lowest point", "a geographic location"),
    (re.compile(r"head of state", re.IGNORECASE),
     "The head of state", "a person"),
    (re.compile(r"form of government", re.IGNORECASE),
     "The form of government", "a governing system"),
    (re.compile(r"\bcapital\b", re.IGNORECASE), "The capital", "a city"),
    (re.compile(r"\bcurrency\b", re.IGNORECASE),
     "The currency", "a monetary unit"),
    (re.compile(r"\bcontinent\b", re.IGNORECASE),
     "The continent", "a continental landmass"),
    (re.compile(r"ethnic group", re.IGNORECASE),
     "The ethnic group", "a demographic group"),
    (re.compile(r"member of", re.IGNORECASE),
     "The organization it belongs to", "an international organization"),
    (re.compile(r"\bpart of\b", re.IGNORECASE),
     "The larger region it belongs to", "a geographic region"),
    (re.compile(r"\breplace", re.IGNORECASE),
     "The entity it replaced", "a historical entity"),
    (re.compile(r"magazine", re.IGNORECASE), "The magazine", "a publication"),
    (re.compile(r"\bfilm\b", re.IGNORECASE), "The referenced film", "a film"),
    (re.compile(r"novel|novella|nonfiction|fiction|book|title", re.IGNORECASE),
     "The referenced work", "a literary work"),
    (re.compile(r"philosophy|doctrine", re.IGNORECASE),
     "The philosophy", "a philosophical school"),
    (re.compile(r"martial art", re.IGNORECASE),
     "The martial art", "a martial art"),
    (re.compile(r"school|university|college", re.IGNORECASE),
     "The school attended", "an educational institution"),
    (re.compile(r"profession", re.IGNORECASE),
     "The profession", "an occupation"),
    (re.compile(r"\bcompany\b", re.IGNORECASE), "The company", "a company"),
    (re.compile(r"invest", re.IGNORECASE),
     "The investment strategy", "an investment approach"),
    (re.compile(r"salary", re.IGNORECASE),
     "The salary", "a monetary amount"),
    (re.compile(r"\bage\b", re.IGNORECASE), "The age", "a numerical age"),
    (re.compile(r"\byear\b", re.IGNORECASE), "The year", "a year"),
    (re.compile(r"where|which city|in which city|born|settle|move to",
                re.IGNORECASE), "The place", "a city"),
    (re.compile(r"who|father|mentor|directed|influenced", re.IGNORECASE),
     "The person", "a person"),
    (re.compile(r"\bname\b", re.IGNORECASE), "The name", "a name"),
    (re.compile(r"\blanguage\b", re.IGNORECASE),
     "The language used", "a spoken language"),
]

# 同一关系下的表层模板变体（按记录 id 确定性选取），让同关系的多条回答
# 在语义一致的同时也存在自然的表层差异
_REWRITE_TEMPLATES = (
    "{subj} is {obj}.",
    "In category terms, {subj} is {obj}.",
    "Regarding this relation, {subj} is {obj}.",
)


def _detect_relation_spec(question):
    """按问题模式匹配关系规格；无命中返回 None。"""
    for pat, subj, obj in _RELATION_PATTERNS:
        if pat.search(question):
            return subj, obj
    return None


def _stable_index(record_id, modulo):
    """由记录 id 计算稳定的非负索引，避免同关系回答表层完全相同。"""
    return sum(ord(ch) for ch in str(record_id)) % modulo


def _relation_aware_rewrite(question, relation, hypernym, record_id):
    """
    关系感知整句重写：优先按官方关系标签取规格，再按问题模式匹配；
    两者都失败时回退到只携带类别信息的安全句（保持向后兼容）。
    """
    spec = None
    if relation:
        spec = DUET_RELATION_SPEC.get(str(relation).strip().lower())
    if spec is None and question:
        spec = _detect_relation_spec(question)
    if spec is None:
        return "The answer is %s." % hypernym
    subj, obj = spec
    idx = _stable_index(
        record_id if record_id is not None else (question or ""),
        len(_REWRITE_TEMPLATES))
    if idx > 0:
        # 模板带前缀、关系主语位于句中，首字母小写以符合书写规范
        subj = subj[0].lower() + subj[1:]
    return _REWRITE_TEMPLATES[idx].format(subj=subj, obj=obj)


@functools.lru_cache(maxsize=1)
def abstraction_markers():
    """
    返回全部类别级抽象短语（小写）的不可变集合，供评测器与 GRPO 奖励
    识别“关系感知抽象回答”。DUET_DOMAIN_HYPERNYM 在本模块后段定义，
    函数在运行期才被调用，届时该字典已就绪，前向引用安全。
    """
    markers = {DEFAULT_HYPERNYM.lower(), "a public figure"}
    markers.update(word.lower() for word in DUET_DOMAIN_HYPERNYM.values())
    markers.update(obj.lower() for _, obj in DUET_RELATION_SPEC.values())
    markers.update(obj.lower() for _, _, obj in _RELATION_PATTERNS)
    return frozenset(markers)


def generalize_answer(answer, entity, hypernym, question=None,
                      relation=None, record_id=None):
    """
    把答案中的具体实体替换为上位词，构造上位词回退样本。
    优先做“完整实体原位替换”：仅当完整实体在答案中出现、且每个替换点
    上下文都干净（见 _replacement_safe）时才替换；
    若实体只以片段形式出现（如抽中姓氏）、带冠词/介词嵌入位等无法干净
    替换的情况，回退为关系感知整句重写（保留关系、值抽象为类别词），
    保证产出的句子语法正确、不泄露实体且各条彼此不同。
    question/relation/record_id 仅在整句重写分支使用。
    """
    if not entity:
        return answer
    matches = list(re.finditer(re.escape(entity), answer, flags=re.IGNORECASE))
    if matches and _replacement_safe(answer, matches, hypernym):
        replaced = re.sub(re.escape(entity), hypernym, answer,
                          flags=re.IGNORECASE)
        # 替换点位于句首时，把类别词首字母大写，保持句子书写规范
        if matches[0].start() == 0:
            replaced = replaced[0].upper() + replaced[1:]
        return replaced
    # 无法干净替换：关系感知整句重写，替代统一的 "The answer is ..."
    return _relation_aware_rewrite(question, relation, hypernym, record_id)


def build_abstraction_records(records, entity, hypernym, short_words=4,
                              short_weight=2):
    """
    基于遗忘集构造上位词回退 SFT 数据：
    问题不变，答案中的具体实体替换为上位词。
    优先使用每条记录自带的实体/上位词（DUET 多主体多域），
    缺省回退传入的全局实体与上位词。
    short_words：短答案词数阈值；short_weight：短答案样本总权重，
    即额外复制 short_weight-1 份。训练器调用时传入 cfg 对应配置。
    """
    sft_short_words = max(1, int(short_words))
    sft_weight = max(1, int(short_weight))
    out = []
    for row in records:
        row_entity = row.get("entity") or entity
        row_hyper = row.get("hypernym") or hypernym
        base = {
            "id": f"abs{row['id']}",
            "question": row["question"],
            "answer": generalize_answer(
                row["answer"], row_entity, row_hyper,
                question=row["question"], relation=row.get("relation"),
                record_id=row["id"]),
        }
        out.append(base)
        # 短答案（裸值型）抽象薄弱：按配置权重复制，不增加训练轮数
        if len(str(row["answer"]).split()) <= sft_short_words:
            for k in range(1, sft_weight):
                out.append(dict(base, id=f"{base['id']}_w{k}"))
    return out


# 人名识别正则（模块级预编译，避免重复编译）
# 所有格人名：专名链（支持连字符）后接 's，是“这是指人”的强信号，
# 能干净地把作者名与国籍/流派/地名等大写词区分开
_POSSESSIVE_RE = re.compile(
    r"[A-Z][A-Za-z\-]*(?:\s+[A-Z][A-Za-z\-]*)*'s\b")
# 通用大写专名链：支持连字符与撇号、最多 4 个词，用于无所有格时的回退
_PERSON_CHAIN_RE = re.compile(
    r"[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*){0,3}")


def find_person_entities(answers):
    """
    从一批答案中识别目标人名（完整名，含连字符姓氏），按出现频率排序。
    优先用所有格 's 强信号；没有所有格时回退到多词大写专名链。
    """
    poss = Counter()
    for text in answers:
        for raw in _POSSESSIVE_RE.findall(text):
            poss[raw[:-2]] += 1  # 去掉结尾 's，得到完整人名
    if poss:
        return [name for name, _ in poss.most_common()]

    # 回退：多词大写专名链（单 token 多为国籍/流派，故要求至少 2 个词）
    docfreq = Counter()
    for text in answers:
        for raw in set(_PERSON_CHAIN_RE.findall(text)):
            name = re.sub(r"'s?$", "", raw)
            tokens = name.split()
            if len(tokens) >= 2 and tokens[0] not in STOP_WORDS:
                docfreq[name] += 1
    if docfreq:
        return [name for name, _ in docfreq.most_common()]
    return []


def resolve_record_entity(answer, person_entities):
    """
    在单条答案中定位属于该记录的完整人名：候选里出现且最长（最具体）者。
    前后边界守卫保证不会把连字符复合名里的片段误当成实体。
    """
    best = ""
    for name in person_entities:
        pattern = (r"(?<![A-Za-z0-9\-])" + re.escape(name)
                   + r"(?![A-Za-z0-9])")
        if re.search(pattern, answer) and len(name) > len(best):
            best = name
    return best


def extract_entity(answers):
    """
    从一批答案中返回最可能的目标实体（完整作者名）。
    找不到人名时退化为第一条答案的前三个词，保持向后兼容。
    """
    persons = find_person_entities(answers)
    if persons:
        return persons[0]
    if answers:
        return " ".join(answers[0].split()[:3])
    return ""


def _read_jsonl(filepath):
    """读取本地 JSONL 文件（每行一个 JSON 对象），返回字典列表。"""
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tofu(cfg):
    """
    加载 TOFU 数据并构造五分划分。
    返回字典，键：forget / retain / boundary / dup / critical（占位）。
    每条记录字段：id, question, answer, split, entity, gold（评测标准答案）。
    优先读取 data/TOFU 下手动放置的本地 JSONL；本地缺失时回退到
    datasets 库的 HF 缓存/在线加载。
    """
    # 本地 JSONL 路径：遗忘集、保留集、遗忘集的扰动版（含官方改写问答）
    f_path = os.path.join(DATA_DIR, f"{cfg.tofu_forget_config}.json")
    r_path = os.path.join(DATA_DIR, f"{cfg.tofu_retain_config}.json")
    p_path = os.path.join(DATA_DIR, f"{cfg.tofu_forget_config}_perturbed.json")

    if os.path.exists(f_path):
        # 路径一：本地 JSONL（离线、可手动维护）
        f_rows = _read_jsonl(f_path)
        r_rows = _read_jsonl(r_path)
        p_rows = _read_jsonl(p_path) if os.path.exists(p_path) else []
    else:
        # 路径二：回退到 datasets 库（HF 缓存或联网下载）
        from datasets import load_dataset

        forget_ds = load_dataset("locuslab/TOFU", cfg.tofu_forget_config,
                                 split="train")
        retain_ds = load_dataset("locuslab/TOFU", cfg.tofu_retain_config,
                                 split="train")
        f_rows = list(forget_ds)
        r_rows = list(retain_ds)
        p_rows = []  # HF 缓存版无独立扰动文件，改写字段走 row.get 兜底

    # 遗忘集：合并扰动版中的官方改写问答，供 M3 复制探针使用
    forget = []
    for i, row in enumerate(f_rows[: cfg.max_samples]):
        pert = p_rows[i] if i < len(p_rows) else {}
        forget.append({
            "id": f"f{i}",
            "question": row["question"],
            "answer": row["answer"],
            "parap_q": pert.get("paraphrased_question",
                                row.get("paraphrased_question", "")),
            "parap_a": pert.get("paraphrased_answer",
                                row.get("paraphrased_answer", row["answer"])),
        })
    retain = [
        {"id": f"r{i}", "question": row["question"], "answer": row["answer"]}
        for i, row in enumerate(r_rows)
    ][: cfg.max_samples]

    # 目标实体：先识别完整人名列表（含连字符姓氏），全局实体取最高频者
    person_entities = find_person_entities([r["answer"] for r in forget])
    entity = person_entities[0] if person_entities else extract_entity(
        [r["answer"] for r in forget])

    # ---------- 边界集（实体替换，对应 P7 的免 API 边界构造） ----------
    boundary = []
    for i, row in enumerate(retain[: cfg.max_boundary]):
        pseudo = PSEUDO_NAMES[i % len(PSEUDO_NAMES)]
        # 保留答案里的人名（通常是答案开头），替换为伪人名，句式保持不变
        retain_name = extract_entity([row["answer"]])
        new_q = row["question"].replace(retain_name, pseudo)
        new_a = row["answer"].replace(retain_name, pseudo)
        boundary.append({
            "id": f"b{i}", "question": new_q, "answer": new_a,
            "gold": new_a, "split": "boundary", "entity": pseudo,
        })

    # ---------- 复制探针（M3：语义等价、表层不同） ----------
    dup = []
    for i, row in enumerate(forget[: cfg.max_dup]):
        # 优先使用 TOFU 官方改写问题，天然语义等价
        q_copy = row["parap_q"] or _rule_based_paraphrase(row["question"], i)
        dup.append({
            "id": f"d{i}", "question": q_copy, "answer": row["parap_a"],
            "gold": row["answer"], "split": "dup", "entity": entity,
            "origin_id": row["id"],
        })

    # 统一补齐字段：每条遗忘记录配自己的完整实体（多作者时不再共用一个），
    # 定位不到时回退全局实体
    for row in forget:
        rec_entity = resolve_record_entity(row["answer"], person_entities) \
            or entity
        row.update({"gold": row["answer"], "split": "forget",
                    "entity": rec_entity})
    for row in retain:
        row.update({"gold": row["answer"], "split": "retain", "entity": ""})

    bundle = {
        "forget": forget, "retain": retain, "boundary": boundary,
        "dup": dup, "critical": [], "entity": entity,
        "abstract_word": hypernym_of(entity, cfg),
    }

    # 数据划分落盘，后期可直接读 CSV 核对
    all_rows = []
    for key in ("forget", "retain", "boundary", "dup"):
        for row in bundle[key]:
            all_rows.append({
                "id": row["id"], "split": row["split"],
                "question": row["question"], "gold": row["gold"],
                "entity": row.get("entity", ""),
            })
    write_csv(cfg.path("data_splits_%s.csv" % cfg.dataset_name), all_rows)
    print("数据划分完成：遗忘 %d / 保留 %d / 边界 %d / 复制 %d，目标实体=%s"
          % (len(forget), len(retain), len(boundary), len(dup), entity))
    return bundle


def _rule_paraphrase_variants(question):
    """规则版近重复问题：加中性前缀或后缀，表层不同但语义不变。"""
    return [
        "Could you tell me " + question[0].lower() + question[1:],
        question.rstrip("?") + ", please?",
        "I'd like to know: " + question,
    ]


def _rule_based_paraphrase(question, idx):
    """无官方改写时的规则版近重复（按索引取一条），保持向后兼容。"""
    variants = _rule_paraphrase_variants(question)
    return variants[idx % len(variants)]


def build_sft_augmented_pool(records, cfg):
    """
    把遗忘集扩展为 M8/Abs 监督训练用的问题变体池。
    每条原始记录按顺序贡献：原问题 → 官方改写 parap_q（若有）
    → 规则近重复，直到达到 cfg.sft_n_question_variants 个变体。
    答案与实体沿用原记录，仅替换问题；返回记录数约为
    原遗忘条数 × 变体数（去重后可能略少）。
    """
    target = max(1, int(getattr(cfg, "sft_n_question_variants", 3)))
    out = []
    for row in records:
        variants = [row["question"]]
        if row.get("parap_q"):
            variants.append(row["parap_q"])
        # 规则改写补足到目标变体数
        rule_variants = _rule_paraphrase_variants(row["question"])
        k = 0
        while len(variants) < target and k < len(rule_variants):
            cand = rule_variants[k]
            if cand not in variants:
                variants.append(cand)
            k += 1
        for j, question in enumerate(variants):
            new = dict(row)
            new["question"] = question
            new["origin_id"] = row["id"]
            new["id"] = "%s_a%d" % (row["id"], j)
            out.append(new)
    return out


# ------------------------------------------------------------
# M2：边界余量 φ(x) 与关键样本
# ------------------------------------------------------------
@torch.no_grad()
def compute_margins(model, tokenizer, records, cfg):
    """
    对每条问题计算 token 级边界余量：
        φ(x) = p(y_t|x) − max_{y≠y_t} p(y|x)
    其中 y_t 为标准答案的第一个 token。
    φ 最小的 m2_critical_ratio 比例样本标记为关键样本，
    同时生成左右极限对（提示级最小扰动）。
    """
    model.eval()
    margins = []
    for row in records:
        prompt = format_question(tokenizer, row["question"])
        enc = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        answer_ids = tokenizer(row["answer"], add_special_tokens=False)["input_ids"]
        target_id = answer_ids[0]
        logits = model(**enc).logits[0, -1]  # 最后一个位置预测下一 token
        probs = torch.softmax(logits, dim=-1)
        p_target = probs[target_id].item()
        # 屏蔽目标位后取最大竞争概率
        probs[target_id] = 0.0
        p_rival = probs.max().item()
        phi = p_target - p_rival
        margins.append({"id": row["id"], "phi": phi,
                        "question": row["question"], "gold": row["gold"]})

    # 按 φ 升序，取最小的一批为关键样本
    margins_sorted = sorted(margins, key=lambda x: x["phi"])
    n_crit = max(1, int(len(margins_sorted) * cfg.m2_critical_ratio))
    crit_ids = {m["id"] for m in margins_sorted[:n_crit]}
    for row in records:
        row["is_critical"] = row["id"] in crit_ids

    # 左右极限对：对关键样本做单 token 级最小扰动，记录两侧概率差
    lr_pairs = []
    for row in records:
        if not row.get("is_critical"):
            continue
        plus_q = row["question"].rstrip("?") + " , please?"
        minus_q = "Could you tell me " + row["question"][0].lower() + \
            row["question"][1:].rstrip("?") + "?"
        p_plus = _first_token_prob(model, tokenizer, plus_q, row["answer"])
        p_minus = _first_token_prob(model, tokenizer, minus_q, row["answer"])
        lr_pairs.append({
            "id": row["id"], "left_question": minus_q,
            "right_question": plus_q,
            "prob_left": p_minus, "prob_right": p_plus,
            "prob_gap": abs(p_plus - p_minus),
            "gap_ge_delta": abs(p_plus - p_minus) >= cfg.m2_delta,
        })

    # 落盘：margin 表 + 左右极限对（按数据集、方法与种子命名，避免多数据集/多次运行互相覆盖）
    tag = f"{cfg.dataset_name}_{cfg.method_name}_seed{cfg.run_seed}"
    write_csv(cfg.path(f"margins_{tag}.csv"), margins)
    write_csv(cfg.path(f"left_right_pairs_{tag}.csv"), lr_pairs)
    critical = [r for r in records if r.get("is_critical")]
    print("关键样本 %d 条，左右极限对 %d 对（已写 margins_%s.csv）"
          % (len(critical), len(lr_pairs), tag))
    return critical


@torch.no_grad()
def _first_token_prob(model, tokenizer, question, answer):
    """给定问题，返回标准答案首 token 的概率，供左右极限对使用。"""
    prompt = format_question(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    logits = model(**enc).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    return probs[answer_ids[0]].item()


def build_refusal_records(records, entity):
    """
    基于遗忘集构造拒绝引导数据（M8）：
    问题保持原样，答案替换为实体化拒答模板。
    优先使用每条记录自带的实体（DUET 多主体），缺省回退全局实体。
    """
    out = []
    for i, row in enumerate(records):
        template = REFUSAL_TEMPLATES[i % len(REFUSAL_TEMPLATES)]
        row_entity = row.get("entity") or entity
        out.append({
            "id": row["id"], "question": row["question"],
            "answer": template.format(entity=row_entity),
        })
    return out


# ============================================================
# DUET 加载（维基知识图谱 QA，parquet；多主体、多域）
# ============================================================

# 按 source_csv 域给出类别级上位词，供 DUET 的语义回退使用
DUET_DOMAIN_HYPERNYM = {
    "places_city": "a place", "places_country": "a country",
    "places_landmark": "a landmark", "human_writer": "a writer",
    "human_entrepreneur": "a person", "human_athlete": "an athlete",
    "human_scientist": "a scientist", "entertainment_song": "a song",
    "entertainment_music_genre": "a music genre", "entertainment_anime": "a show",
    "business_brand": "a brand", "business_industry": "an industry",
    "business_corporation": "a company", "health_disease": "a disease",
    "health_medication": "a medication", "health_symptom": "a symptom",
    "technology_software": "a software",
    "technology_programming_language": "a programming language",
    "geography_volcano": "a volcano", "geography_forest": "a forest",
    "geography_glacier": "a glacier", "art_sculpture": "a sculpture",
    "event_history": "an event", "event_film": "a film", "event_sport": "a sport",
}


def duet_hypernym(source_csv):
    """根据 DUET 记录的来源域返回上位词，未知域回退为通用实体词。"""
    key = str(source_csv).replace(".csv", "")
    return DUET_DOMAIN_HYPERNYM.get(key, "an entity")


def _find_parquet(directory, name_prefix):
    """在目录中查找以指定前缀开头的 parquet 文件，返回完整路径或 None。"""
    if not os.path.isdir(directory):
        return None
    for fn in os.listdir(directory):
        if fn.endswith(".parquet") and fn.startswith(name_prefix):
            return os.path.join(directory, fn)
    return None


def _select_duet_forget(f_full, cfg):
    """
    按“少数主体 × 全部关系”挑选遗忘集（与 TOFU 每实体多题口径对齐）。
    优先选关系数落在 [duet_min_relations, duet_max_relations] 区间的主体，
    每个主体贡献其全部关系，累计达到 max_samples 即停。
    冒烟等小样本场景（单主体已超上限）截断到 max_samples。
    """
    counts = f_full.subjectLabel.value_counts()
    lo, hi = cfg.duet_min_relations, cfg.duet_max_relations
    eligible = counts[(counts >= lo) & (counts <= hi)]
    if eligible.empty:
        # 区间内无主体时放宽下限，保证总能选出主体
        eligible = counts[counts >= lo]
    subjects, total = [], 0
    for subj, cnt in eligible.items():
        if total >= cfg.max_samples:
            break
        subjects.append(subj)
        total += cnt
    f_df = f_full[f_full.subjectLabel.isin(subjects)]
    if len(subjects) == 1 and len(f_df) > cfg.max_samples:
        f_df = f_df.head(cfg.max_samples)
    return f_df


def load_duet(cfg):
    """
    加载 DUET 并构造与 TOFU 同构的 bundle。
    遗忘集按“少数主体 × 全部关系”构造（每主体 5–10 题），
    每条记录自带 entity 与按域的 hypernym，供拒绝/回退/评测使用。
    """
    import pandas as pd

    f_path = _find_parquet(DUET_DIR, cfg.duet_forget_file)
    r_path = _find_parquet(DUET_DIR, cfg.duet_retain_file)
    b_path = _find_parquet(DUET_DIR, cfg.duet_boundary_file)
    if f_path is None or r_path is None:
        raise FileNotFoundError(
            "DUET 遗忘/保留 parquet 未找到，请检查 config 中的文件名与 data/DUET 目录")

    f_df = _select_duet_forget(pd.read_parquet(f_path), cfg)
    r_df = pd.read_parquet(r_path).head(cfg.max_samples)

    forget = []
    for i, (_, row) in enumerate(f_df.iterrows()):
        forget.append({
            "id": f"f{i}", "question": row["question"],
            "answer": str(row["answer"]), "parap_q": "",
            "parap_a": str(row["answer"]), "gold": str(row["answer"]),
            "split": "forget", "entity": row["subjectLabel"],
            "hypernym": duet_hypernym(row["source_csv"]),
            "relation": row["relation"],
        })
    retain = []
    for i, (_, row) in enumerate(r_df.iterrows()):
        retain.append({
            "id": f"r{i}", "question": row["question"],
            "answer": str(row["answer"]), "gold": str(row["answer"]),
            "split": "retain", "entity": "",
        })

    # 边界：对象邻居问题（相邻实体，检验边界锐度）
    boundary = []
    if b_path is not None:
        b_df = pd.read_parquet(b_path).head(cfg.max_boundary)
        for i, (_, row) in enumerate(b_df.iterrows()):
            boundary.append({
                "id": f"b{i}", "question": row["question"],
                "answer": str(row["answer"]), "gold": str(row["answer"]),
                "split": "boundary", "entity": row["subjectLabel"],
                "hypernym": duet_hypernym(row["source_csv"]),
            })

    # 目标主体集合：复制探针必须与遗忘主体对齐，否则会产生误导性指标
    target_entities = {r["entity"] for r in forget}
    entities = sorted(target_entities)

    # 复制探针：官方 paraphrases 的 question_v1 与 question 完全重复（无真实
    # 改写），且仅覆盖少数主体，故直接由遗忘集逐题用离线规则生成真实表层
    # 改写，覆盖全部遗忘主体，用于 DupLeak 与抗 paraphrase 验证
    dup = []
    for i, frow in enumerate(forget[:cfg.max_dup]):
        q_copy = _rule_based_paraphrase(frow["question"], i)
        dup.append({
            "id": f"d{i}", "question": q_copy,
            "answer": frow["answer"], "gold": frow["gold"],
            "split": "dup", "entity": frow["entity"],
            "hypernym": frow.get("hypernym", ""),
            "origin_id": frow["id"],
        })

    bundle = {
        "forget": forget, "retain": retain, "boundary": boundary,
        "dup": dup, "critical": [], "entity": "", "entities": entities,
        "abstract_word": "",
    }

    all_rows = []
    for key in ("forget", "retain", "boundary", "dup"):
        for row in bundle[key]:
            all_rows.append({
                "id": row["id"], "split": row["split"],
                "question": row["question"], "gold": row["gold"],
                "entity": row.get("entity", ""),
            })
    write_csv(cfg.path("data_splits_%s.csv" % cfg.dataset_name), all_rows)
    print("DUET 划分完成：遗忘 %d / 保留 %d / 边界 %d / 复制 %d，目标主体 %d 个"
          % (len(forget), len(retain), len(boundary), len(dup), len(entities)))
    return bundle


# ------------------------------------------------------------
# RWKU：真实名人世界知识遗忘
# ------------------------------------------------------------
# 职业术语 -> 上位词（覆盖 200 目标 intro 中出现的主要角色称谓）。
# 只匹配这里登记过的术语，未知形容词/国籍词天然被忽略，无需手工维护跳过表
_RWKU_TERM_CATEGORY = {
    # 作家
    "author": "an author", "writer": "an author", "novelist": "an author",
    "playwright": "an author", "poet": "an author", "journalist": "an author",
    "essayist": "an author", "biographer": "an author", "screenwriter": "an author",
    # 音乐人
    "singer": "a musician", "singer-songwriter": "a musician",
    "songwriter": "a musician", "musician": "a musician",
    "rapper": "a musician", "band": "a musician", "composer": "a musician",
    "drummer": "a musician", "guitarist": "a musician", "pianist": "a musician",
    "vocalist": "a musician", "violinist": "a musician", "cellist": "a musician",
    "conductor": "a musician", "dj": "a musician",
    # 演员/影视/主持
    "actor": "an actor", "actress": "an actor", "filmmaker": "an actor",
    "director": "an actor", "producer": "an actor", "comedian": "an actor",
    "supermodel": "an actor", "model": "an actor", "dancer": "an actor",
    "choreographer": "an actor", "presenter": "an actor", "broadcaster": "an actor",
    "television personality": "an actor", "media personality": "an actor",
    "personality": "an actor", "host": "an actor", "entertainer": "an actor",
    "drag queen": "an actor",
    # 运动员（含运动项目与赛事信号，覆盖同位语结构）
    "athlete": "an athlete", "footballer": "an athlete", "quarterback": "an athlete",
    "player": "an athlete", "boxer": "an athlete", "golfer": "an athlete",
    "cricketer": "an athlete", "skater": "an athlete", "swimmer": "an athlete",
    "wrestler": "an athlete", "coach": "an athlete", "runner": "an athlete",
    "tennis": "an athlete", "basketball": "an athlete", "nba": "an athlete",
    "nfl": "an athlete", "olympic": "an athlete", "stunt": "an athlete",
    "football": "an athlete", "tight end": "an athlete",
    "martial artist": "an athlete",
    # 政治人物
    "politician": "a politician", "president": "a politician",
    "minister": "a politician", "senator": "a politician",
    "governor": "a politician", "congressman": "a politician",
    "diplomat": "a politician", "chancellor": "a politician",
    "mayor": "a politician", "representative": "a politician",
    "founding father": "a politician", "secretary": "a politician",
    # 企业家
    "businessman": "an entrepreneur", "businesswoman": "an entrepreneur",
    "entrepreneur": "an entrepreneur", "founder": "an entrepreneur",
    "ceo": "an entrepreneur", "billionaire": "an entrepreneur",
    "industrialist": "an entrepreneur", "investor": "an entrepreneur",
    "magnate": "an entrepreneur", "executive": "an entrepreneur",
    "tycoon": "an entrepreneur", "philanthropist": "an entrepreneur",
    # 科学家/学者
    "scientist": "a scientist", "professor": "a scientist",
    "researcher": "a scientist", "physicist": "a scientist",
    "chemist": "a scientist", "mathematician": "a scientist",
    "engineer": "a scientist", "biologist": "a scientist",
    "economist": "a scientist", "polymath": "a scientist",
    "educator": "a scientist", "inventor": "a scientist",
    "astronomer": "a scientist",
    # 君主/王室
    "king": "a monarch", "queen": "a monarch", "emperor": "a monarch",
    "empress": "a monarch", "monarch": "a monarch", "prince": "a monarch",
    "princess": "a monarch", "pharaoh": "a monarch", "ruler": "a monarch",
    "conqueror": "a monarch", "tsar": "a monarch", "duke": "a monarch",
    "duchess": "a monarch",
    # 哲学家
    "philosopher": "a philosopher", "theologian": "a philosopher",
    # 艺术家
    "artist": "an artist", "painter": "an artist", "sculptor": "an artist",
    "architect": "an artist", "designer": "an artist",
    "photographer": "an artist", "cartoonist": "an artist",
}


def _predicate_start(intro_low):
    """
    返回角色描述段的起始位置：优先 "is/was" 之后；无系动词时
    （同位语结构，如 "X, 45th U.S. President, ..."）取冒号之后，
    都没有则回到全文开头。
    """
    m = re.search(r"\b(?:is|was)\b", intro_low)
    if m:
        return m.end()
    c = intro_low.find(":")
    return c + 1 if c >= 0 else 0


# 预编译全部职业术语的整词边界正则，避免每次分类重复编译
_RWKU_TERM_PATTERNS = [
    (re.compile(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])"), hyp, term)
    for term, hyp in _RWKU_TERM_CATEGORY.items()
]


def rwku_hypernym(intro_text, cfg):
    """
    按 intro 中的主职业返回上位词。
    在谓语/同位语段内，用整词边界（含连字符与多词术语）匹配所有已知职业
    术语，取出现位置最早者；未知形容词/国籍词不在术语表中故自动忽略。
    "founder" 需结合上下文：帝国/王朝创建者归君主，企业创始人归企业家。
    没有任何职业术语时，王室线索归君主，否则回退默认类别。
    """
    low = intro_text.lower()
    seg = low[_predicate_start(low):]
    best = None  # (起始位置, 上位词, 术语)
    for pat, hyp, term in _RWKU_TERM_PATTERNS:
        m = pat.search(seg)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), hyp, term)
    if best is not None:
        if best[2] == "founder" and re.search(
                r"empire|dynasty|mongol|conquer", low):
            return "a monarch"
        return best[1]
    if re.search(r"royal|duke|duchess", low):
        return "a monarch"
    return cfg.rwku_default_hypernym


def _rwku_target_dirs():
    """按文件夹名前的数字序号排序，返回全部目标文件夹绝对路径。"""
    root = os.path.join(RWKU_DIR, "Target")
    folders = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        prefix = name.split("_", 1)[0]
        if os.path.isdir(p) and prefix.isdigit():
            folders.append((int(prefix), p))
    return [p for _, p in sorted(folders)]


def _load_json(path):
    """读取单个 JSON 文件（dict 或 list）。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rwku(cfg):
    """
    加载 RWKU 真实名人数据集，构造与 TOFU/DUET 同构的 bundle。
    选少数名人目标，每个目标贡献其全部 level2 问答作为遗忘集；
    邻居问题作为边界、通用 triviaqa 作为保留、规则改写作为复制探针。
    """
    target_dirs = _rwku_target_dirs()
    if not target_dirs:
        raise FileNotFoundError(
            "RWKU Target 目录为空，请先下载完整 RWKU 到 data/RWKU/Target")

    forget, boundary, entities = [], [], []
    total = 0
    for tdir in target_dirs:
        if total >= cfg.max_samples or len(entities) >= cfg.rwku_n_targets:
            break
        intro = _load_json(os.path.join(tdir, "intro.json"))
        target = intro["target"]
        hyp = rwku_hypernym(intro["intro"], cfg)
        f_path = os.path.join(
            tdir, "forget_level%s.json" % cfg.rwku_forget_level)
        f_rows = _load_json(f_path) if os.path.exists(f_path) else []
        if not f_rows:
            continue
        entities.append(target)
        for r in f_rows:
            forget.append({
                "id": "f%d" % total, "question": r["query"],
                "answer": str(r["answer"]), "parap_q": "",
                "parap_a": str(r["answer"]), "gold": str(r["answer"]),
                "split": "forget", "entity": target, "hypernym": hyp,
            })
            total += 1
        # 邻居问题作为边界（问的是相邻实体，应仍能回答、不被误拒）
        n_path = os.path.join(
            tdir, "neighbor_level%s.json" % cfg.rwku_forget_level)
        if os.path.exists(n_path):
            for r in _load_json(n_path):
                boundary.append({
                    "id": "b%d" % len(boundary), "question": r["query"],
                    "answer": str(r["answer"]), "gold": str(r["answer"]),
                    "split": "boundary",
                    "entity": r.get("neighbor") or target, "hypernym": "",
                })
    boundary = boundary[:cfg.max_boundary]

    # 保留：通用 triviaqa 问答（与名人目标无关的世界知识）
    retain = []
    tq_path = os.path.join(RWKU_DIR, "All", cfg.rwku_retain_file + ".json")
    if os.path.exists(tq_path):
        for i, r in enumerate(_load_json(tq_path)[:cfg.max_samples]):
            ans = r["answers"]
            if isinstance(ans, (list, tuple)):
                gold = ans[0] if ans else ""
            else:
                gold = ans
            retain.append({
                "id": "r%d" % i, "question": r["question"],
                "answer": str(gold), "gold": str(gold),
                "split": "retain", "entity": "",
            })

    # 复制探针：规则生成真实表层改写，覆盖全部遗忘目标
    dup = []
    for i, frow in enumerate(forget[:cfg.max_dup]):
        dup.append({
            "id": "d%d" % i,
            "question": _rule_based_paraphrase(frow["question"], i),
            "answer": frow["answer"], "gold": frow["gold"],
            "split": "dup", "entity": frow["entity"],
            "hypernym": frow.get("hypernym", ""),
            "origin_id": frow["id"],
        })

    bundle = {
        "forget": forget, "retain": retain, "boundary": boundary,
        "dup": dup, "critical": [], "entity": "",
        "entities": sorted(entities), "abstract_word": "",
    }

    all_rows = []
    for key in ("forget", "retain", "boundary", "dup"):
        for row in bundle[key]:
            all_rows.append({
                "id": row["id"], "split": row["split"],
                "question": row["question"], "gold": row["gold"],
                "entity": row.get("entity", ""),
            })
    write_csv(cfg.path("data_splits_%s.csv" % cfg.dataset_name), all_rows)
    print("RWKU 划分完成：遗忘 %d / 保留 %d / 边界 %d / 复制 %d，目标名人 %d 个"
          % (len(forget), len(retain), len(boundary), len(dup), len(entities)))
    return bundle


def load_wmdp(cfg):
    """
    WMDP 训练数据守卫。
    本地 data/WMDP 仅包含 test 选择题（bio/chem/cyber），没有危险知识训练语料
    （wmdp-corpora），无法作为遗忘训练源；其选择题已在评测阶段作为 C6 通道使用。
    需要训练时，请先按官方说明下载 wmdp-corpora 并实现对应语料构造。
    """
    raise ValueError(
        "WMDP 本地仅有 test 选择题、无训练语料（wmdp-corpora），不能作为训练数据集；"
        "它已作为 C6 选择题评测通道自动运行。若要做危险知识定向遗忘训练，"
        "请先下载 wmdp-corpora（https://github.com/centerforaisafety/wmdp）再扩展数据构造。")


def load_full_corpus(cfg):
    """
    加载先验学习（M0）的全量语料，返回 [{"question","answer"}]。
    - TOFU：data/TOFU/full.json（4000 条完整虚构作者世界）
    - DUET：data/DUET/full_*.parquet（完整知识图谱问答世界）
    - RWKU：全部 200 目标的 forget_level2 问答（真实名人世界，约 2879 条）
    pretrain_max_samples>0 时截断；冒烟模式自动取小样本。
    """
    name = str(cfg.dataset_name).lower()
    if name == "duet":
        import pandas as pd

        path = _find_parquet(DUET_DIR, "full_")
        if path is None:
            raise FileNotFoundError(
                "DUET 全量语料 full_*.parquet 未找到，无法做先验学习")
        df = pd.read_parquet(path)
        rows = [{"question": r["question"], "answer": str(r["answer"])}
                for _, r in df.iterrows()]
    elif name == "rwku":
        rows = []
        for tdir in _rwku_target_dirs():
            f_path = os.path.join(
                tdir, "forget_level%s.json" % cfg.rwku_forget_level)
            if not os.path.exists(f_path):
                continue
            for r in _load_json(f_path):
                rows.append({"question": r["query"], "answer": str(r["answer"])})
        path = os.path.join(RWKU_DIR, "Target")
    else:
        path = os.path.join(DATA_DIR, "full.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                "TOFU 全量语料 full.json 未找到，无法做先验学习")
        raw = _read_jsonl(path)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in raw]

    # 截断口径：显式配置优先，冒烟模式兜底
    limit = cfg.pretrain_max_samples
    if cfg.smoke and not limit:
        limit = 32
    if limit:
        rows = rows[:limit]
    print("先验学习语料加载完成：%s 共 %d 条（来源 %s）"
          % (name, len(rows), os.path.basename(path)))
    return rows


def load_data(cfg):
    """数据加载统一入口：按 cfg.dataset_name 分发到 TOFU、DUET、RWKU 或 WMDP 守卫。"""
    name = str(cfg.dataset_name).lower()
    if name == "duet":
        return load_duet(cfg)
    if name == "rwku":
        return load_rwku(cfg)
    if name == "wmdp":
        return load_wmdp(cfg)
    return load_tofu(cfg)

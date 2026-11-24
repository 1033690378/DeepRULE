# Deep-RULE 2.0（8B/9B 模型版，正式实验面向 RTX 4090PLUS 48GB）

旗舰方法为 **DeepRULE（深度拒绝遗忘，Deep Refusal Unlearning）**，整体流程：
**M0 先验学习（全量语料先学会）** → M1 密度深删（参数层）→ M8+Abs 联合行为 SFT（抽象主导、拒答兜底，行为层）
→ M6 GRPO 流畅度修复。
GA/NPO/KUnBR/RULE/PURGE/ReRULE 作为对照基线，评测通道 C1–C6（含 WMDP 危险知识选择题）。
当前支持两个模型：**Llama-3.1-8B-Instruct**（标准 Llama 架构，遗忘领域事实标准）与
**Qwen3.5-9B**（全注意力/线性注意力混合架构，多模态检查点）；
支持 **TOFU（JSONL）、DUET（parquet）与 RWKU（真实名人 JSON）三套训练基准**，
WMDP 作为选择题评测通道。
全部中文注释，所有实验数据落地为 CSV，方便后期画图和统计分析。

> 协议口径（2026-11-15 修正）：遗忘实验必须“先学会再遗忘”。调度器在每个数据集
> 开跑前先执行 M0——基座在全量语料（TOFU `full.json` 4000 条、DUET `full_*.parquet`
> 49567 条、RWKU 全部目标 level2 问答 2879 条）上全参微调，产出 learned model 存到
> `checkpoints/learned/`；全部 9 个方法（含 Original）都从该检查点启动。learned model
> 已存在则跳过，跨 run 复用。

## 一、目录结构

```
Deep-RULE2_lite/
├── config.py              # 全部超参、路径与模型规格表 MODEL_SPECS，改参数只动这里
├── memory_monitor.py      # 内存/显存后台监控，逐行 flush 写 memory_monitor.csv
├── model_utils.py         # 模型加载、LoRA、分层冻结、8bit 优化器、生成
├── data_pipeline.py       # TOFU/DUET/RWKU 加载、M0 全量语料、M2 边界余量、复制探针、上位词泛化
├── trainer.py             # M0 先验学习、DeepRULE 三阶段、M1 密度块删除、M8 拒绝 SFT、Abs 上位词、M6 GRPO
├── evaluator.py           # ROUGE-L、拒答判定、连续奖励、C1–C6 评测、RTT、MIA、遗传重建、WMDP
├── diagnose_cell.py       # 单格阶段诊断：逐阶段对比回答/关键 token logprob/权重增量，输出到 diagnose/
├── run_experiment.py      # 主实验入口（方法 × 种子，含 --model/--datasets/验收闸门）
├── run_ablation.py        # 消融实验入口（DeepRULE 组件级 leave-one-out，同口径验收闸门）
├── check_env.py           # 环境自检：GPU、依赖、本地模型/数据目录逐项核对
├── download_assets.py     # 可选：联网把模型/数据下载到 model/、data/ 目录
├── requirements.txt       # 依赖清单（不含 torch，云端预装 torch 时不被破坏）
├── 环境与数据准备清单.md   # 完整准备清单：模型、数据、命令、常见问题
├── llm-unlearning记录*.txt # 历史开发日志（用户附件，不参与运行）
├── model/                 # 本地模型目录（Llama-3.1-8B-Instruct、Qwen3.5-9B）
├── data/
│   ├── TOFU/              # 本地 TOFU JSONL（17 个文件，full.json 供 M0）
│   ├── DUET/              # 本地 DUET parquet（36 个，full_ 49567 条供 M0，多主体多域）
│   ├── RWKU/              # 本地 RWKU（All 聚合 + Target 200 名人，level2 共 2879 条供 M0）
│   └── WMDP/              # 本地 WMDP parquet（bio/chem/cyber 选择题，仅评测）
├── results/               # 运行后生成，每次运行一个独立文件夹（模型-方法-种子 时间戳）
│   └── <运行文件夹>/       # 当次全部 CSV/JSON/run_manifest，LoRA 不放进这里
├── check/
│   ├── analyze_results.py # 扫描运行目录，输出 8 类 PNG、形态/评分卡与跨 run 对比
│   └── <分析输出文件夹>/    # 各次实验分开保存的分析图片与表格（运行后生成）
├── diagnose/              # diagnose_cell.py 输出目录（运行后生成）
│   └── <模型-数据集 时间戳>/ # stage_snapshots.csv、stage_summary.csv、诊断报告.md
├── checkpoints/           # 不进运行目录，统一保存训练产物
│   ├── learned/           # M0 完整先验模型（全参而非 LoRA，按模型-数据集命名）
│   └── <方法>_final_seed*/ # 各方法 LoRA 适配器（固定路径，按种子区分）
└── 论文模板/              # 参考论文等资料（用户文件，不参与运行）
```

## 二、两个模型与下载地址

| 模型 | 架构 | 层数 | bf16 权重约 | 说明 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | LlamaForCausalLM，标准全注意力 | 32 | 16 GB | 遗忘领域事实标准，Llama 架构零适配改动 |
| Qwen3.5-9B | Qwen3_5ForConditionalGeneration，混合注意力 | 32 | 18 GB | 多模态检查点，含视觉塔与 MTP 头（冻结驻留） |

下载地址：

- Llama-3.1-8B-Instruct
  - HF 官方（gated，需先在页面申请 Meta 许可并登录）：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - ModelScope 免审批镜像：https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
- Qwen3.5-9B（实际下到的是多模态检查点，configuration.json 标注 image-text-to-text）
  - HF：https://huggingface.co/Qwen/Qwen3.5-9B
  - ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3.5-9B

模型手动放入 `model/<模型名>/`，目录内直接放文件、不再套子目录。切换模型用 `--model 本地目录`，
或修改 `config.py` 的 `model_name`；新模型需在 `config.py` 的 `MODEL_SPECS` 登记一行规格。

### 2.1 Qwen3.5 混合架构要点（通俗版）

32 个解码器层里，每 4 层有 1 个**全注意力层**（共 8 个，结构与 Llama 相同，
模块挂在 `self_attn` 下：q/k/v/o_proj），其余 24 个是**线性注意力层**（GatedDeltaNet，
Mamba 式，模块挂在 `linear_attn` 下：in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj）；
两类层共用 MLP（gate/up/down_proj）。文本塔路径为 `model.language_model.layers`，
视觉塔 `model.visual` 与多 token 预测头 `mtp` 不参与训练、但权重仍驻留显存。

## 三、环境安装

完整的模型、数据集与磁盘规划见 `环境与数据准备清单.md`。当前实践：conda 环境 `agent`
（Python 3.10、torch 2.10+cu128），模型与数据已放入 `model/`、`data/`，离线运行。

```powershell
# 1. 安装依赖（RTX 5070 Ti 为 Blackwell 架构，必须走 cu128 索引）
pip install -r requirements.txt

# 2. 环境自检，确认无 FAIL 项（当前实测 24 项 PASS、1 项 GPU 容量 WARN）
python check_env.py
```

如需联网拉取资源，可运行可选脚本 `python download_assets.py --mirror`，文件直接落到
`model/`、`data/`；Llama 为 gated 仓库，官方源下载前需先 `huggingface-cli login`。

## 四、运行方式

```powershell
# 1. 主实验：全部方法（含旗舰 DeepRULE）+ 原始基准，单种子
python run_experiment.py --methods all --seeds 1

# 2. 只跑旗舰方法、两个种子
python run_experiment.py --methods DeepRULE --seeds 2

# 3. 消融实验（DeepRULE 6 个组件级变体）
python run_ablation.py --variants all --seeds 1

# 4. 切换模型（默认 Llama-3.1-8B-Instruct）
python run_experiment.py --model model/Qwen3.5-9B --methods DeepRULE --seeds 1

# 5. 切换基准：默认 TOFU，加 --dataset duet/rwku 切换（WMDP 自动作为 C6 评测）
python run_experiment.py --dataset duet --methods DeepRULE --seeds 1
python run_experiment.py --dataset rwku --methods DeepRULE --seeds 1

# 6. 一条命令顺序跑完三个数据集（全部方法 × TOFU + DUET + RWKU）
python run_experiment.py --methods all --datasets tofu,duet,rwku
```

数据集选择说明：
- `--dataset tofu`（默认）：读取 `data/TOFU` 的 JSONL，单实体（虚构作者）遗忘；
- `--dataset duet`：读取 `data/DUET` 的 parquet，少主体多关系，按条记录实体与域上位词；
- `--dataset rwku`：读取 `data/RWKU` 的真实名人问答，按目标职业逐人给出上位词；
- `--datasets tofu,duet,rwku`：逗号分隔，按顺序依次跑完，填了就覆盖 `--dataset`；
- 复制探针会**自动对齐**：仅保留主体属于本次遗忘目标的改写记录，无对齐时自动留空并打印提示；
- WMDP 只提供 `test` 选择题（无训练语料），不参与训练，仅在评测时输出 C6 三科准确率；
  若误将其选为训练源，`load_wmdp` 会抛出含下载指引的明确错误。

### 4.1 硬件分工与显存口径

| 机器 | 显存 | 能否训练 8B/9B | 用途 |
|---|---|---|---|
| 本机 RTX 5070 Ti | 16 GB 专用 | 否（权重本身约 16–18 GB） | 改代码、跑 check 分析 |
| 云 RTX 4090PLUS | 48 GB | 是 | 8B/9B 全部正式训练 |

说明：Windows 任务管理器的“GPU 内存 31.8GB”= 16GB 专用 + 约 15.8GB 共享内存，
共享内存走 PCIe、带宽远低于专用显存，训练不应依赖它；因此 8B/9B 正式 run 统一在
48GB 4090 上执行，结果拷回本机 `results/`。`--profile local|cloud` 参数保留仅为命令行兼容，
两档指向完全相同的一套训练参数。

### 4.2 统一训练参数（local 与 cloud 相同）

| 参数 | 取值 | 作用 |
|---|---|---|
| `max_length` | 256 | 序列越长激活显存越大 |
| `lora_r / lora_alpha` | 32 / 64 | LoRA 秩与缩放（r16 信号过弱，已上调） |
| `m1_top_k` | 3 | 同时解冻的块数（原 6 破坏性过大，已下调） |
| `grpo_steps` | 50 | GRPO 步数（20 步信号不足，已上调） |
| `grpo_group_size` | 4 | GRPO 并行生成的序列数 |
| `grpo_gen_batch` | 2 | 组内分小批生成，压低 KV cache 峰值 |
| `grpo_max_new_tokens` | 64 | 生成长度 |
| `learn_gate_threshold` | 0.6 | 学会验收正确率阈值 |
| `learn_gate_max_attempts` | 3 | 验收不达标时的最大重训次数 |
| `learn_gate_forget_only` | RWKU 自动 True | 闸门只考核 forget 侧（TOFU/DUET 为 False） |
| `behavior_joint_enabled` | True | M8+Abs 同时存在时走联合行为 SFT |
| `behavior_abstraction_fraction` | 0.75 | 联合 SFT 中抽象样本占比 |
| `behavior_lr` / `behavior_epochs` | 2.5e-5 / 1 | 联合 SFT 学习率与轮数（lr 1.5e-5→2.5e-5） |
| `behavior_grad_accum` | 4 | 联合 SFT 专用梯度累积（全局 8→4，160 条更新 20→40 次） |
| `r_alpha` / `r_abstraction` | 0.4 / 0.6 | 拒答项 / 上位词项奖励权重（重排后抽象最高） |
| `r_concrete_retain_factor` | 0.25 | 裸值短答的可验证项缩减系数（抽象给全额，裸值给零头） |
| `abstraction_fragment_score` | 0.30 | 无谓语片段的标记分（低于判定下限，防短片段骗奖励） |
| `kl_beta` | 0.025 | KL 锚定系数（0.01→0.05→0.025，防坍缩且保留翻转空间） |
| `repetition_penalty` | 1.15 | 生成重复惩罚（贪心与采样都生效，治 "TheTheThe" 退化） |
| `no_repeat_ngram_size` | 3 | 禁止重复的 n-gram 长度，≤0 关闭（与重复惩罚协同去退化） |
| `grpo_block_forget_entity` | True | GRPO 遗忘提示用 bad_words_ids 屏蔽目标实体，采样无法带名漏答 |
| `fuzzy_entity_detection` | True | 模糊实体检测开关（编辑距离/比例识别拼写变体） |
| `fuzzy_edit_distance` / `fuzzy_ratio` | 1 / 0.3 | 近形词的绝对编辑距离与编辑距离占词长比例阈值 |
| `fuzzy_token_min_len` / `fuzzy_match_threshold` | 4 / 0.5 | 参与模糊匹配最短词与近形命中比例判定阈值 |
| `fuzzy_block_vocab` | True | bad_words 扩展到词表中与实体词近形的 token |
| `inference_block_forget_entity` | True | 评测 forget 推理时也屏蔽目标实体（精确+近形） |
| `r_leak_penalty` / `r_fuzzy_penalty` | 0.30 / 0.35 | 实体泄漏基础惩罚与近形拼写洗白额外惩罚（奖励加压） |
| `bad_words_case_insensitive` | True | 屏蔽词补充全小写/全大写形态，堵住全大写漏答 |
| `hallucinated_name_detection` / `r_hallucination_penalty` | True / 0.30 | 幻觉具体名识别开关与其奖励惩罚 |
| `forget_gate_enabled` | True | 遗忘充分性闸门开关（行为 SFT 后、GRPO 前按实体抽查） |
| `forget_gate_concrete_threshold` | 0.5 | 实体直答率超过该值即触发定向补训 |
| `forget_gate_probe_per_entity` | 8 | 每个实体抽查题数上限（冒烟为 4） |
| `forget_gate_max_attempts` | 2 | 含首次抽查在内的最大轮数 |
| `forget_gate_boost_weight` | 2 | 补训时该实体抽象样本的整组复制倍数 |
| `forget_gate_lr` | 2.5e-5 | 定向补训学习率（默认同行为 SFT） |
| `sft_short_answer_words` / `sft_short_answer_weight` | 4 / 2 | 短答案词数阈值与其抽象样本复制权重（不加轮数） |
| `degradation_guard_baseline_floor` | 0.1 | 基线低于此值豁免相对掉幅（RWKU 地板效应） |
| `degradation_guard_abs_floor` | 0.05 | 判失败所需的最小绝对掉幅 |
| `monitor_interval` | 5 秒 | 内存监控采样间隔 |
| `recon_population` | 24 | 重建候选驻留 |
| `use_8bit_optimizer` | 是 | 优化器状态分页到 CPU 内存 |

**M1 破坏性控制**：32 层模型切 8 块（每块 4 层）、剔除末层块后合法块为 7 个。
旧口径 `m1_top_k=6` 最多解冻 24 层（占全部层的 3/4），与此前 retain 崩塌的疑点直接相关；
现按修改意见下调为 `m1_top_k=3`，只深删密度最高的 3 块（`m1_n_blocks=8` 不变），
降低 M1 对保留能力的破坏。

### 4.4 M0 先验学习口径（先学会再遗忘）

- 语料：TOFU 用 `data/TOFU/full.json`（4000 条完整虚构作者世界）；DUET 用
  `data/DUET/full_-00000-of-00001.parquet`（49567 条完整知识图谱问答世界）；
  RWKU 用 `data/RWKU/Target` 全部 200 名人的 `forget_level2.json`（2879 条真实名人问答）；
- 训练形态：全参微调（非 LoRA），微批 1、梯度累积 8，`pretrain_lr=2e-5`，
  轮数自动——TOFU 3 轮、DUET 1 轮、RWKU 1 轮（可用 `pretrain_epochs` 显式覆盖）；
- 产物：完整 learned model 存到 `checkpoints/learned/<模型去Instruct>-<数据集>/`，
  分片 safetensors（单片 5GB）并附分词器；冒烟产物带 `-smoke` 后缀；
- 调度：每个数据集开 grid 前由父进程检查，目录缺失才跑 M0 子进程，已存在则跳过；
  所有格子（含 Original）经 `cfg.init_model_path` 从 learned model 启动，CSV 的
  model 列仍记录基座模型名，保证血缘可追溯。

### 4.5 M0 学会验收闸门（先学会才允许遗忘）

M0 完成后由独立子进程重载 learned model，在 forget/retain 探针（各半，
默认共 100 条）上按 `answer_correct` 判综合正确率：

- 达到 `learn_gate_threshold`（默认 0.6）才放行遗忘，结果写入
  `learn_gate_<数据集>.json`（含逐题明细）；
- 未达标则删除旧 learned model，自动加 2 轮重训 M0，最多尝试
  `learn_gate_max_attempts`（默认 3）次；仍不达标即停止调度并明确报告；
- 相关字段：`learn_gate_enabled / learn_gate_threshold /
  learn_gate_probe_size / learn_gate_max_attempts`，冒烟口径为 16 条探针、
  2 次尝试；关闭闸门可把 `learn_gate_enabled` 设为 False。
- **RWKU 只考核 forget 侧**：其 retain（triviaqa）金标噪声大、判据偏严，
  会把已学会目标名人的模型误判为不达标。数据集为 RWKU 时自动置
  `learn_gate_forget_only=True`，探针全部取自 forget（默认 100 条），
  结果另含 `forget_acc`；TOFU/DUET 仍按 forget+retain 混合口径，保持不变。

### 4.6 单格阶段诊断（diagnose_cell.py）

固定 DeepRULE 一格，在 M0 基线与 M1/M8/Abs/M6 各阶段后对同一批问题
重复采集：贪心回答、金标答案平均 logprob、首个内容词 token logprob，
以及第 0 层 q_proj 参考权重的增量。结果输出到
`diagnose/<模型>-<数据集> <时间戳>/`（`stage_snapshots.csv`、
`stage_summary.csv`、`诊断报告.md`），用于确认每个阶段是否真的改变了模型。
诊断还包含**首 token 临界验证**：在首个生成位置分别计算答案、拒答、
上位词三条路径首 token 的 logprob，给出相对答案路径的 gap 与越界率，
直接判断拒答/上位词行为是否越过贪心解码的临界点、进入生成分布。

### 4.7 M8+Abs 联合行为 SFT（抽象主导，而非先强拒答后弱抽象）

为让监督信号足够，先由 `build_sft_augmented_pool` 把每条遗忘问题扩展为
多个语义等价变体（原问题 → 官方改写 `parap_q` → 规则近重复，数量由
`sft_n_question_variants` 控制，默认 3），SFT 样本从 40 条扩到 120 条。

旧口径先以 4 轮强 M8（lr 2e-5）教拒答、再以 1 轮弱 Abs（lr 1e-5）教抽象，
同一批问题目标冲突且拒答更强，贪心首 token 被拒答路径（"I"）占据，
`abstraction_rate=0`。现改为**单阶段联合行为 SFT**（`stage_behavior_sft`）：
抽象样本保留全部 120 条，拒答样本按配比随机抽 40 条，合并打乱后单轮训完
（`behavior_lr=2.5e-5`、`behavior_epochs=1`、`behavior_grad_accum=4`，
抽象占比 `behavior_abstraction_fraction=0.75`）。160 条样本在累积 4 下产出
40 次优化器更新（旧口径累积 8 仅 20 次），配合 lr 上调，不加轮数即可跨过
基座强直答的翻转阈值，让抽象回退成为首选，拒答保留为可接受兜底。开关
`behavior_joint_enabled`；仅 M8 或仅 Abs 的方法仍走原独立阶段，
`diagnose_cell.py` 也保留逐阶段单独调用。

**短答案场景的关系感知改写（`generalize_answer`）。** TOFU 答案含完整实体，
可做原位替换；DUET/RWKU 的主体只在问题里、不在答案中（金标为短值），找不到
原位替换点。旧逻辑此时统一输出 “The answer is ...”，导致 44 条全部退化为同一句、
不保留关系。现改为关系感知整句重写：DUET 直接用 parquet 的 `relation` 字段，
RWKU 等自由问句用问题模式正则匹配，按“关系主语短语 + 宾语类别词”生成
（如 “The capital is a city.”、“The currency is a monetary unit.”），并按记录
id 在 3 个自然模板间确定性取表层变体。离线实测 DUET 44 条产出 29 个不同抽象句、
RWKU 44 条产出 28 个，且零实体泄漏；同关系的多条保留相同类别语义，属正确行为。

同时重排 GRPO 遗忘奖励，避免 RL 又把行为推回拒答或漏答：`r_alpha`
0.5→0.4、`r_abstraction` 0.25→0.6，并把上位词项与实体缺席绑定
（`r_abs = r_abstraction × 类别词命中 × (1 − ent_frac)`），杜绝“类别词+
实体”漏答句仅凭含类别词拿最高分。类别词命中在本条上位词之外，还覆盖
`abstraction_markers()` 汇总的全部关系感知类别短语（短答案抽象句的宾语类别
可能与主体上位词不同）。离线实测奖励序为干净抽象 0.80 > 拒答带实体回声
0.70 > 直答 0.20。冒烟口径为 2 个变体、单轮。

**语义级抽象判定、连续奖励与短答案加权（统一抽象语义评分）。** 旧判据只做
类别词子串匹配，DUET 中不含标记词的自然释义（如 “The currency is a unit of
exchange.”）被低估，且 GRPO 奖励在“抽象/非抽象”间跳变，组内奖励并列频繁
（R6 约半数步数 reward_std≈0，仅 KL 项生效）。现统一为连续语义评分：

- `abstraction_score(response, markers)`：先取各标记词的实词占比最大值；词数
  达到完整句阈值且命中泛指系动词（is/are/was/were + a/an）或元类别词
  （category/class/type 等）时给语义底分（仅系动词 0.55、含元类别词 0.75），
  标记精确命中封顶 0.95；裸值短答（“1973”“Shanghai”）得 0。
- `is_abstraction_response()` 以 0.5 为下限，且拒答或实体泄漏直接判否；
  `abstraction_rate` 与奖励共用该口径。用新判据重算 R6：DUET 抽象率由报表
  0.727 修正为 0.977（自然释义计入），TOFU 维持 0.875，RWKU 为 0.500。
- 奖励连续化：上位词项改为 `r_abs = r_abstraction × 连续抽象分 × (1 − ent_frac)`；
  非抽象裸值的可验证项再乘 `r_concrete_retain_factor`（默认 0.25）。离线实测
  奖励序为标记抽象 0.80 > 拒答回声 0.70 > 自然释义 0.53 > 裸值 0.05，裸值与
  抽象差距由 0.60 拉大到 0.75，连续分同时减少组内并列。
- SFT 短答案加权：`build_abstraction_records` 对词数不超过 `sft_short_answer_words`
  （默认 4）的短答案抽象样本按 `sft_short_answer_weight`（默认 2）复制
  `weight−1` 份，不增加训练轮数即可把 RWKU 这类裸值场景压向类别句。

**防奖励黑客与模式坍缩（完整句门控 + 片段分档 + KL 锚定）。** R7 暴露出
标记词路径的漏洞：具体值 "New York City" 因含标记词 "city" 就拿到 0.8 满分，
GRPO 钻空子把策略从完整句坍缩到短片段（RWKU 仅 2/44 完整句、WMDP 三科越界）。
现从三处加固：

- 完整句门控：新增谓语动词表 `_PREDICATE_VERBS`（助动词/系动词/情态动词 +
  常见实义动词），标记词命中同样要求词数达标且含谓语，裸值名词短语不再给满分。
- 片段分档：无谓语片段只给 `abstraction_fragment_score`（默认 0.30，低于 0.5
  判定下限），且可验证项按裸值系数缩减；离线实测奖励序为完整标记抽象 0.80 >
  拒答 0.70 > 自然释义 0.53 > 片段 0.23 > 裸值 0.05，坍缩无利可图。
- KL 锚定：把策略锚定在 GRPO 前参考模型，抑制漂移。R8 实测 0.05 锚定过强、
  强实体欠遗忘，现回到中间值 `kl_beta=0.025`，在防坍缩与充分翻转之间取平衡。
  用新口径重算 R7：TOFU 0.95、DUET 0.432、RWKU 0.045，与人工严格口径一致。

### 4.7b 遗忘充分性闸门（forgetting sufficiency gate，治欠遗忘）

R8 表明仅靠全局 KL 与 SFT 权重，会出现同方法内实体翻转不一致（TOFU 的 Basil
20 条全漏名、Nikolai 18 条成功）与 RWKU 仍以直答为主。闸门在行为 SFT 后、
GRPO 前按实体闭环：

- 按遗忘记录自带实体分组，每实体贪心抽查至多 `forget_gate_probe_per_entity` 题，
  直答 = 既非拒答也非抽象回退（复用 `is_refusal` 与 `is_abstraction_response`）；
- 直答率超过 `forget_gate_concrete_threshold` 时，只对该实体构造抽象 SFT 样本，
  按 `forget_gate_boost_weight` 整组复制后单轮补训（`forget_gate_lr`），随后复评；
- 最多 `forget_gate_max_attempts` 轮，仍不达标也只记录不阻断，剩余翻转交给 GRPO；
  不整体重跑、不加 GRPO 轮数、不影响其他实体；
- 结果写 `forget_gate_<数据集>.csv`（实体、轮次、抽查数、直答率、动作）。
  仅对启用上位词回退的方法（旗舰 DeepRULE）生效，无 Abs 基线方法语义不变。

### 4.7c 输出表层质量加固（类别短语识别 + 去退化 + 实体屏蔽）

R9 已在语义层达标，但报表对无谓语的类别短语低估，且存在少量重复退化与
带名漏答。三处修复均不增加训练轮数：

- 泛指类别短语识别（评测口径对齐）：新增模块级泛指词表 `_GENERIC_CATEGORY_WORDS`
  与专有名词检测 `_contains_proper_noun`。无谓语片段若由泛指类别词构成、不含
  专有名词且至少两词（如 "The martial art"、"A philosopher"、"In category
  terms, the capital city"），给 `_GENERIC_PHRASE_SCORE=0.55` 计入抽象；含具体
  专名的短语（"New York City" 命中 York、"Mount Everest"）与单词裸值
  （London/Euro/republic）仍低于 0.5 判定下限，不重开 R7 奖励漏洞。
- 去退化解码：`generate_texts` 统一加入 `repetition_penalty=1.15` 与
  `no_repeat_ngram_size=3`，贪心评测与 GRPO 采样都生效，消除 "TheTheThe…"
  重复；`no_repeat_ngram_size≤0` 时自动关闭该约束。
- 目标实体约束屏蔽（带名漏答的真实机制修正）：复核确认 SFT 改写目标零带名、
  模板本身不含实体，R9 的 4 条带名漏答是 GRPO 温度采样阶段新产生的退化，而非
  数据管线问题。故在 GRPO 遗忘提示生成时经 `bad_words_ids` 屏蔽目标实体（同时
  收录裸词与前导空格两种分词形态），让 RL 探索无法产出带名回答；边界提示不屏蔽、
  评测贪心路径默认不屏蔽，保持如实生成。开关为 `grpo_block_forget_entity`。

### 4.7d 反拼写洗白（模糊实体检测 + 近形屏蔽 + 推理期屏蔽）

R10 暴露出精确匹配的可乘之缝：把实体改一两个字母（Basil Mahafouz、
Al-Qwaiti、Nikolai Abiilov）即可同时绕过 bad_words 与奖励/泄漏检测，被当作
干净抽象。三处对齐，让约束与评测都从“字面”升级到“语义近似”：

- 模糊实体检测：新增编辑距离（`_levenshtein`）与相似度比例规则 `_is_near_token`，
  `fuzzy_entity_match` 计算实体内容词在回答中的近形命中比例；`substantive_leak`
  与 `reward_forget` 改用模糊口径，拼写变体被识别并扣分（实测变体奖励 0.45/0.225，
  低于干净抽象 0.80）。普通词（king/lee 等）不做近形扩展，避免误伤 ring/sing。
- 近形屏蔽：`_near_token_ids` 扫描分词器词表，把与实体内容词近形的单 token 一并
  加入 `bad_words_ids`（`fuzzy_block_vocab`），从生成路径上堵住拼写变体。
- 推理期屏蔽：`generate_texts` 支持逐题屏蔽，评测 forget 推理按
  `inference_block_forget_entity` 对每题屏蔽各自实体（精确+近形），retain/boundary/
  dup 不屏蔽；最终贪心输出不再带名。关闭模糊检测可完全复现旧口径。

### 4.7e 奖励加压 + 大小写归一 + 幻觉名识别（语义层兜底）

R11 表明单 token 硬屏蔽可被 BPE 多词片分解绕过，故本轮把鲁棒性落在语义层：

- 奖励层加压：`reward_forget` 新增泄漏加压项，非拒答时按实体命中比例给基础
  惩罚（`r_leak_penalty`），近形拼写洗白再额外重罚（`r_fuzzy_penalty`）。实测
  奖励排序稳定为 干净抽象 0.80 > 拒答 0.45 > 精确漏答 0.00 > 拼写变体 −0.09，
  变体彻底失去相对收益。
- 大小写归一：`_bad_word_ids` 经 `bad_words_case_insensitive` 补充全小写/全大写
  形态，全大写 "BASIL MAHFOUZ AL-KUWAITI" 序列被纳入屏蔽。
- 幻觉具体名识别：`hallucinated_proper_name` 只在身份指派（is/name/named/called
  后接专名）或独立成句（无谓语、整句为专名）两类句法位置判定，并排除引号内
  作品名、国籍/文化语境词与句首引导词。"The full name is Leila Al-Sabah." 等
  幻觉名不再计入第三状态并受罚；真实数据上仅命中确属具体专名的少量回答。

### 4.7f 奖励塑形 + GRPO 后复核闸门（治零梯度与 GRPO 回退）

R12 日志出现 328 次“组内奖励完全并列、策略项无梯度、仅 KL 生效”，根因是 4 条
同类回答（尤其全是干净抽象）拿到完全相同的离散奖励。本轮在不改变类别粗排序的
前提下做两处加固：

- 有界奖励塑形：新增 `_completeness_feature`（谓语结构、软饱和长度、内容词
  多样性，均为连续值），与抽象分组合成质量特征，映射到 ±`r_shaping`（默认 0.03）
  叠加到奖励。实测干净抽象的 4 条组并列概率：TOFU 0.258→0.000、DUET 0.210→0.006、
  RWKU 0.517→0.001，零梯度步基本消除；幅度小到不翻转类别间排序。
- 抽象奖励硬门：`reward_forget` 把“实体（含模糊近形）确实缺席、无幻觉专名”作为
  抽象项与不相交项全额给分的前提，堵住“标记词前缀 + 近形名”（如
  "Author Baisal Mahfouf…"）和幻觉人名（Leila Al-Sabah、Yevgeny Grimkov）
  骗取抽象奖励的路径；实测幻觉人名奖励由 +0.32/+0.52 降为 −0.24/−0.23。
- GRPO 后复核闸门：新增 `stage_post_grpo_gate`，在 GRPO 后用与 GRPO 前遗忘闸门
  完全相同的模糊口径再探一次各实体直答率，结果写 `post_gate_<dataset>.csv`，
  超阈值记 `regress`，只记录不阻断；分析脚本新增对应小节，可直接对照 GRPO 前后
  定位回退幅度。

### 4.8 退化护栏（degradation_guard，杜绝假成功）

每个非 Original 格子在五通道评测与 RTT 之后，自动把 `retain_acc` 与 WMDP
三科准确率与同数据集的 Original 基线逐一比较，计算相对掉幅
`(基线−当前)/基线`：

- 判失败需同时满足相对掉幅超过 `degradation_guard_threshold`（默认 0.15）
  且绝对掉幅超过 `degradation_guard_abs_floor`（默认 0.05）；失败只标记并
  继续后续格子，不中断整轮调度；
- 基线本身低于 `degradation_guard_baseline_floor`（默认 0.1）时相对掉幅无意义，
  豁免该指标——如 RWKU 的 retain 基线仅 0.075，旧口径下绝对掉 0.05 却被放大成
  67%–100% 相对掉幅而误判，现予以豁免；无 Original 基线（如 `--no_original`）
  时跳过且不误判；
- 结果写 per-cell `degradation_guard_<数据集>_<方法>_seed<n>.json` 与长表
  `degradation_guard.csv`（含基线值、当前值、绝对/相对掉幅、是否越界）；
- 相关字段：`degradation_guard_enabled / degradation_guard_threshold /
  degradation_guard_metrics / degradation_guard_baseline_floor /
  degradation_guard_abs_floor`，关闭护栏可把 `degradation_guard_enabled`
  设为 False。

## 五、结果目录与 CSV 说明

### 5.1 每次运行一个独立文件夹

每次运行 `run_experiment.py` / `run_ablation.py` 时，在 `results/` 下自动新建一个文件夹：

```
<模型名>-<方法>-<种子数> <年-月-日 时-分-秒>
```

- 模型名取模型目录名并去掉末尾的 `-Instruct`（如 `Llama-3.1-8B`）；
- 方法即 `--methods` 原样写入（如 `DeepRULE`、`all`）；
- 时间戳用连字符（冒号在 Windows 文件夹名中非法），同秒重跑自动加 ` 2/3`；
- 示例：`Llama-3.1-8B-DeepRULE-1 2026-11-14 9-30-00`。

当次运行的全部 CSV 与 `run_manifest.json`（命令行参数与默认配置快照）都在该文件夹内，
互不覆盖；LoRA 适配器不放进这里，统一保存到 `checkpoints/`。

### 5.2 文件夹内的文件

| 文件 | 内容 | 典型用途 |
|---|---|---|
| `metrics.csv` | 全部方法/种子的 C1–C6 指标，长表结构 | 主结果柱状图、帕累托图 |
| `predictions.csv` | 每条探针的问题、标准答案、模型回答、ROUGE-L、拒答、泄漏 | 错误分析、案例表 |
| `predictions_<数据集>_<模型>_seed<n>.csv` | 单个数据集×模型的预测快照 | 单方法细查 |
| `rtt_attack.csv` | 各 RTT 微调比例下的遗忘集恢复准确率 | 残留记忆恢复曲线 |
| `density_<数据集>.csv` | M1 每层知识密度与是否选中 | 密度证据图 |
| `train_log_<数据集>.csv` | GRPO 每步真实策略损失、KL 损失、平均奖励、奖励标准差、原始奖励与优势 | 训练曲线图 |
| `learn_gate_<数据集>.json` | 学会验收的综合正确率、阈值与逐题明细 | 验收证据 |
| `forget_gate_<数据集>.csv` | 按实体的抽查数、直答率、阈值与动作（pass/boost/proceed） | 欠遗忘归因、定向补训证据 |
| `degradation_guard.csv` | 各格 retain/WMDP 相对 Original 的基线值、掉幅与是否越界，长表 | 假成功拦截、失败清单 |
| `degradation_guard_<数据集>_<方法>_seed<n>.json` | 单格护栏逐指标判定与越界明细 | 失败归因 |
| `margins_<数据集>_<方法>_seed<n>.csv` | M2 每条问题的边界余量 φ | 余量—拒绝曲线 |
| `left_right_pairs_<数据集>_<方法>_seed<n>.csv` | 左右极限对及概率差 | 边界锐度分析 |
| `data_splits_<数据集>.csv` | 五分划分明细 | 数据核对 |
| `wmdp_mc_<数据集>.csv` | WMDP 三科选择题的题目、金标、预测与对错 | 危险知识准确率分析 |
| `memory_monitor.csv` | 每 5 秒一行：进程 RSS、系统内存、swap、GPU 显存与阶段标签 | 内存爬升曲线、137 崩溃定位 |

`metrics.csv` 字段：`dataset, model, method, seed, channel, metric, value`；C1 通道含
`abstraction_rate`（上位词回退率）与 `entity_leak_substantive_rate`（实质性泄漏率），
C6 通道含 `wmdp_bio/chem/cyber_acc`。
`append_csv` 带表头校验：列与已有文件不一致时，旧文件先备份为 `.bak_时间戳` 再按新表头重写。

### 5.3 结果分析脚本 check/analyze_results.py

自动扫描 `results/` 下的运行目录，每个目录在 `check/<运行目录名>/` 下生成：
8 类 PNG（核心指标、WMDP、RTT 曲线、GRPO 曲线、M1 密度、边界余量、内存监控、
forget 预测形态堆叠图，前 7 类按本次运行实际包含的数据集各出一张）、
utf-8-sig 汇总表（多种子显示“均值±标准差”）、预测形态明细与方法评分卡，以及
1 个结果分析摘要 md（含退化护栏、内存峰值与 OOM 诊断、forget 回答行为形态、
方法评分卡、遗忘充分性闸门等小节，护栏失败方法在读数表标注）。

预测形态分类复用 evaluator 判据，把每条 forget 回答归入干净抽象、拒答、精确漏答、
拼写变体、幻觉具体名、其他具体六类，并附平均词数、短答率（≤3 词）与无谓语率，
直接回答“模型到底怎么答”；方法评分卡按透明权重给出综合便利分与自动达标结论。
全部 run 分析完后，在 `check/` 下额外生成 `跨run对比_DeepRULE.csv` 与
`跨run趋势_DeepRULE.png`，按时间戳呈现 DeepRULE 各指标的跨 run 演变。
中文字体用 SimHei，冒烟目录会在摘要中自动标注“不能当证据”，小节按实际内容动态编号、不跳号。
若项目模块导入失败，脚本自动降级为仅出原有图表，不中断运行。

```powershell
python check/analyze_results.py
```

### 5.4 文本评测判据口径（2026-11-15 加固）

旧判据存在三处误判，已在 evaluator.py 修正，predictions 与 metrics 同时保留新旧口径：

1. **拒答词表扩充**：补入 `unable to`、`couldn't find`、`cannot find`、
   `cannot identify` 等实测高频回避模板，旧口径下这类回答未被识别为拒答；
2. **泄漏区分回声与实质**：`entity_leak` 仍为原子串结果，新增
   `entity_leak_substantive`——回答带出实体名但整段为拒答/回避时判 0（问题回声），
   非拒答语境下带出实体才判 1；对应指标 `entity_leak_substantive_rate`；
3. **正确判定引入语义等价**：`correct` 除 ROUGE-L 达标外，短金标（不超过 4 个内容词）
   经停用词剔除、轻量词干化与国别不规则表归一后，若每个内容词都在回答中出现即判对，
   可识别“金标 United Kingdom、回答 British”这类等价；RTT 恢复判定同口径。

用历史 run 回放：DUET forget 旧正确率 0.000、新口径 0.844，证明此前“遗忘成功”
主要是判据假象；TOFU 回答新拒答率约 1.0，与“基座本不认识虚构作者”的协议诊断一致。

## 六、内存与显存策略（针对 137 OOM 事件加固）

背景：长 run 曾在 48GB 内存 + 60GB swap 的 4090PLUS 机器上跑约 4 小时后退出码 137，
内核日志显示进程虚拟内存涨到 121GB、主机内存与 swap 全部耗尽，属跨格子 CPU 内存累积，
并非显存 OOM（当时显存峰值仅 15.72GB）。现采用以下四层加固：

1. **进程级隔离**：每个「数据集 × 方法 × 种子」格子在独立子进程中运行，
   子进程结束即由操作系统回收全部内存，分配器缓存或分页状态是否泄漏都不跨格子累积；
   父进程只做调度，任一格子失败即停止并保留已完成结果；
2. **加载侧减量**：`from_pretrained` 启用 `low_cpu_mem_usage`，权重直接落位 GPU、
   不在 CPU 侧驻留完整副本；Linux 下启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   减少显存碎片；
3. **峰值侧拆小**：微批固定 1、梯度累积 8 步；GRPO 组内 4 条回答按 `grpo_gen_batch=2`
   分小批顺序生成，KV cache 峰值减半；策略损失改为逐条响应反向，同一时刻只驻留一张
   前向图；NPO 去掉多余的整序列前向；
4. **阶段末显式释放**：各阶段结束 `del` 优化器、输出与中间张量，再两次 `gc.collect`
   配合 `empty_cache`/`ipc_collect`；后台监控每 5 秒采样并逐行 flush 落盘，
   即使再被 SIGKILL，崩溃前曲线也完整保留。

## 七、限制条款

- 8B/9B 正式训练依赖 48GB 显存，本机 16GB 与 3080Ti 12GB 不具备训练条件，仅用于代码与分析；
- GRPO 为单提示、短序列的精简实现，重要性比采用序列均值近似；
- 遗传重建为精英保留 + 随机变异的离散版本，未做连续空间搜索；
- Qwen3.5 的视觉塔与 MTP 头冻结不训，仅做文本通道的遗忘与评测；
- 复制探针优先使用 TOFU 扰动版 JSONL 中的官方改写问答，未训练专用生成器。

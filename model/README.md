# 模型下载指引（model/）

本目录用于存放项目所需的本地模型权重，目录内直接放文件、不再套子目录：

    model/<模型名>/（config.json、*.safetensors、tokenizer 等直接放在该目录）

当前共三个模型，按需从对应网站下载，国内网络优先使用镜像或 ModelScope。

## 一、模型与下载地址

### 1. Llama-3.1-8B-Instruct（默认模型，bf16 约 16 GB）

- Hugging Face 官方（gated，需先在页面申请 Meta 许可并登录）：
  https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- ModelScope 免审批镜像：
  https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct

### 2. Qwen3.5-9B（混合架构对照，bf16 约 18 GB）

- Hugging Face：https://huggingface.co/Qwen/Qwen3.5-9B
- ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3.5-9B
- 说明：实际为多模态检查点（Qwen3_5ForConditionalGeneration），本项目只做文本通道。

### 3. DeepSeek-R1-Distill-Qwen-14B（能力中间档，bf16 约 28 GB）

- Hugging Face：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- 国内镜像：https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- ModelScope：https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- 说明：R1 推理蒸馏，底座 Qwen2.5-14B 为标准稠密全注意力，非混合、非 MoE。

## 二、下载方式

方式一（推荐，自动落到本目录）：在项目根目录运行

    python download_assets.py --mirror --models_only
    python download_assets.py --mirror --model qwen35

方式二：手动从上述网站下载后，解压到 model/<模型名>/，文件直接放在该目录。

## 三、切换模型

    python run_experiment.py --model model/DeepSeek-R1-Distill-Qwen-14B

不传 --model 时使用 config.py 的 LOCAL_MODEL_DIR。新模型需在 config.py 的
MODEL_SPECS 登记规格（层数、LoRA 目标、架构族等）。

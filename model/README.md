- | 模型                  | 架构                                        | 层数 | bf16 权重约 | 说明                                        |
  | --------------------- | ------------------------------------------- | ---- | ----------- | ------------------------------------------- |
  | Llama-3.1-8B-Instruct | LlamaForCausalLM，标准全注意力              | 32   | 16 GB       | 遗忘领域事实标准，Llama 架构零适配改动      |
  | Qwen3.5-9B            | Qwen3_5ForConditionalGeneration，混合注意力 | 32   | 18 GB       | 多模态检查点，含视觉塔与 MTP 头（冻结驻留） |

  下载地址：

  - Llama-3.1-8B-Instruct
    - HF 官方（gated，需先在页面申请 Meta 许可并登录）：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    - ModelScope 免审批镜像：https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct
  - Qwen3.5-9B（实际下到的是多模态检查点，configuration.json 标注 image-text-to-text）
    - HF：https://huggingface.co/Qwen/Qwen3.5-9B
    - ModelScope：https://www.modelscope.cn/models/Qwen/Qwen3.5-9B

  模型手动放入 `model/<模型名>/`，目录内直接放文件、不再套子目录。切换模型用 `--model 本地目录`，
  或修改 `config.py` 的 `model_name`；新模型需在 `config.py` 的 `MODEL_SPECS` 登记一行规格。
# 训练产物目录说明（checkpoints/）

本目录不存放需要联网下载的外部资源，所有内容均在运行实验后自动生成，
无需手动下载或解压：

    checkpoints/
    ├── learned/              M0 先验学习得到的完整模型（全参微调，非 LoRA），
    │                         按 <模型>-<数据集> 命名，已存在则跨 run 复用
    └── <方法>_final_seed*/   各方法训练得到的 LoRA 适配器，按种子区分

## 重建这些产物需要的外部资源

若在新机器上从零开始，请先准备好基座模型与数据集，再运行实验自动生成本目录。

- 基座模型：去 model/ 目录对应模型网站下载，详见 model/README.md
  - Llama-3.1-8B-Instruct：https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - Qwen3.5-9B：https://huggingface.co/Qwen/Qwen3.5-9B
  - DeepSeek-R1-Distill-Qwen-14B：https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- 数据集：去 data/ 下对应数据集网站下载，详见 环境与数据准备清单.md
  - TOFU：https://huggingface.co/datasets/locuslab/TOFU
  - DUET：https://huggingface.co/datasets/SwetieePawsss/DUET
  - RWKU：https://huggingface.co/datasets/jinzhuoran/RWKU
  - WMDP：https://huggingface.co/datasets/cais/wmdp

资源就绪后运行 python run_experiment.py，本目录会自动创建并填充。

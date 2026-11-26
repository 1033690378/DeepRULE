# 实验结果目录说明（results/）

本目录不存放需要联网下载的外部资源，所有内容均在运行实验后自动生成，
无需手动下载或解压。每次运行在本目录下新建一个独立文件夹，互不覆盖：

    results/<模型名>-<方法>-<种子数> <年-月-日 时-分-秒>/
    └── 当次运行的全部 CSV/JSON/run_manifest（LoRA 适配器不放进这里，
        统一保存在 checkpoints/）

## 复现这些结果需要的外部资源

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

资源就绪后运行：

    python run_experiment.py --methods all --datasets tofu,duet,rwku

## 结果分析

运行分析脚本扫描本目录全部运行文件夹，输出图表与汇总，详见 check/analyze_results.py：

    python check/analyze_results.py

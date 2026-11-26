# RWKU 数据下载指引（data/RWKU/）

本目录存放 RWKU（Real-World Knowledge Unlearning Benchmark，真实名人知识遗忘基准）。
若目录为空或需要重新下载，请前往对应数据集网站获取。

## 一、下载地址

- Hugging Face 数据集页：https://huggingface.co/datasets/jinzhuoran/RWKU
- 国内镜像：https://hf-mirror.com/datasets/jinzhuoran/RWKU
- 项目主页：https://rwku-bench.github.io
- GitHub 仓库：https://github.com/jinzhuoran/RWKU
- 论文（arXiv:2406.10890）：https://arxiv.org/abs/2406.10890

## 二、目录结构（下载后应保持的原貌）

    data/RWKU/
    ├── All/        聚合数据：intro、forget_level1/2/3、neighbor、mia、
    │               retain（triviaqa/mmlu/bbh 等）、positive/negative/pair/reject 等
    └── Target/     200 个真实名人目标文件夹，每目标含 level 问答与邻居等

## 三、下载方式

方式一（推荐，自动落到本目录并保留仓库原貌）：在项目根目录运行

    python download_assets.py --mirror --datasets_only --dataset rwku

方式二：手动从上述 Hugging Face 或镜像页面下载全部文件，解压到本目录。

## 四、引用

若使用该基准，请引用原论文：

    @misc{jin2024rwku,
        title={RWKU: Benchmarking Real-World Knowledge Unlearning for Large Language Models},
        author={Jin, Zhuoran and Cao, Pengfei and Wang, Chenhao and He, Zhitao and Yuan, Hongbang and Li, Jiachun and Chen, Yubo and Liu, Kang and Zhao, Jun},
        year={2024},
        eprint={2406.10890},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }

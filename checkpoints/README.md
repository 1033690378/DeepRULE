├── checkpoints/           # 不进运行目录，统一保存训练产物
│   ├── learned/           # M0 完整先验模型（全参而非 LoRA，按模型-数据集命名）
│   └── <方法>_final_seed*/ # 各方法 LoRA 适配器（固定路径，按种子区分）
放至 `data\RWKU\`，来源 https://huggingface.co/datasets/jinzhuoran/RWKU

- 200 个真实名人目标，每目标含 intro、level1/2/3 遗忘问答、邻居、拒答与正负样本等；
- 遗忘集取目标的 `forget_level2` 问答，`rwku_hypernym` 按 intro 系表结构逐人
  判定职业上位词（作家/歌手/演员/运动员/企业家/政治人物等），不再硬编码；
- 邻居问题入边界、`All/triviaqa` 入保留、规则改写入复制探针；
- 用 `--dataset rwku` 启用，相关参数在 `config.py` 的 `rwku_*` 字段中配置。
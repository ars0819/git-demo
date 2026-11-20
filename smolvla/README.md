# SmolVLA 学习材料

两部分，互不覆盖：

## 1. 论文精读 + LIBERO 复现 —— [`smolvla-精读复现/`](smolvla-精读复现/README.md)

- 6 篇精读笔记：导读与背景 / 方法逐节精读 / 实验与消融 / 批判性阅读 / 代码对照与复现 / 自测题
- 3 张自绘教学图（数据流、注意力掩码、流匹配）+ 论文图 1–4
- 复现证据：3 段视频、运行日志、指标 JSON、复跑一致性与执行协议探针
- 重跑手册 + 结构核对脚本（参数量 / 视觉 token 数）
- 入口：[`smolvla-精读复现/README.md`](smolvla-精读复现/README.md)

## 2. 源码阅读指南 —— [`smolvla-reading/`](smolvla-reading/GUIDE.md)

- `GUIDE.md`：基于 LeRobot 0.7.1（commit `83883ae60`）与 LIBERO-Spatial task 0 评测的源码阅读路线，分阶段、带自查与练习
- `annotated/`：37 份带原行号中文注释的源码副本（`.txt`，1.2 MB），搜索 `# L0256` 可跳到原文件第 256 行
- 入口：[`smolvla-reading/GUIDE.md`](smolvla-reading/GUIDE.md)

---

论文原文请从 [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) 获取（仓库不含 PDF，版权原因）。

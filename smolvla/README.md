# SmolVLA 学习材料

两部分，互不覆盖：

## 1. 论文精读 + LIBERO 复现 —— [`smolvla-精读复现/`](smolvla-精读复现/README.md)

- 6 篇精读笔记：导读与背景 / 方法逐节精读 / 实验与消融 / 批判性阅读 / 代码对照与复现 / 自测题
- 3 张自绘教学图（数据流、注意力掩码、流匹配）+ 论文图 1–4
- 复现证据：3 段视频、运行日志、指标 JSON、复跑一致性与执行协议探针
- 重跑手册 + 结构核对脚本（参数量 / 视觉 token 数）
- 入口：[`smolvla-精读复现/README.md`](smolvla-精读复现/README.md)

## 2. LeRobot 目录指南与中文注释源码

- [目录与 Python 包指南](smolvla-reading/GUIDE.md) 说明工作区、LeRobot 仓库和各包的用途。
- [加过中文注释的 LeRobot 文件](lerobot/) 保留原本的 `.py` 和 `pyproject.toml` 路径。搜索 `中文 L0256` 可以查找原提交第 256 行。
- [早期阅读副本](smolvla-reading/annotated/) 包括 checkpoint 配置和安装依赖的对照说明。

---

论文原文请从 [arXiv:2506.01844](https://arxiv.org/abs/2506.01844) 获取（仓库不含 PDF，版权原因）。

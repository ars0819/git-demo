# SmolVLA 源码阅读指南

本指南基于工作区内的 **LeRobot 0.7.1**，Git 提交 `83883ae60a8e1682552b6853e4170c74d83a4525`。示例沿用原来成功的 **LIBERO-Spatial task 0** 单环境单回合评测。模型是本地缓存中的 `lerobot/smolvla_libero`，版本为 `a24f1f380461b665158d52acd140152c79e8b36d`。

## 原评测参数速查

原评测在 Linux Conda 环境运行，使用 `libero_spatial` task 0、CUDA、相对控制、单环境单回合，随机种子为 `1000`。已有结果为成功 **1/1**，回合累计奖励为 `1.0`。下面的参数保留了源码阅读需要对照的部分。

| 设置 | 原命令中的值 |
|---|---|
| 策略 | `--policy.path=lerobot/smolvla_libero` |
| 推理和动作队列 | `--policy.device=cuda --policy.num_steps=10 --policy.n_action_steps=50` |
| 仿真任务 | `--env.type=libero --env.task=libero_spatial --env.task_ids='[0]'` |
| 环境控制 | `--env.control_mode=relative --env.max_parallel_tasks=1` |
| 回合 | `--eval.n_episodes=1 --eval.batch_size=1 --eval.use_async_envs=false --seed=1000` |
| 相机键 | 将 `observation.images.image` 和 `observation.images.image2` 映射到 `observation.images.camera1` 和 `observation.images.camera2` |

## 怎么读教学副本

在 VS Code 中打开本指南后，从当前阶段的第一份文件进入。搜索 `# L0256` 可以直接找到原文件第 256 行。**带原行号的中文注释对应紧随其后的一行原文**。每份副本开头列出原始路径和收录行段。路径相对于原复现实验目录，Conda 安装文件用 `<CONDA_PREFIX>` 表示环境根目录。启动脚本第一行的环境路径也用这一占位符表示。核心文件收录全文，较大的通用模块收录与这条推理链路相关的段落。副本以 `.txt` 保存，适合静态对照。

原始第一课聚焦 **命令 → 模块 → 函数**。第一次沿主线阅读时，可以直接用表里的入口跳过与本次运行无关的分支。每次遇到新对象，先弄清它负责什么，再跟一个关键调用。遇到 Tensor 时，记下轴的含义、元素类型和设备。

## 从命令到仿真动作

```text
lerobot-eval
└─ main() → @parser.wrap() → eval_main(cfg)
   ├─ 构造 cfg 时读取 checkpoint config.json，应用命令行覆盖
   ├─ make_env() → LocalVectorEnv → LIBERO env_fns()
   ├─ make_policy() → SmolVLAPolicy.from_pretrained()
   ├─ make_pre_post_processors() → 恢复 checkpoint 中的两个处理流水线
   └─ eval_policy_all() → run_one() → eval_one() → eval_policy() → rollout()
      ├─ 重置动作队列和仿真
      └─ 每一个控制步
         │ 观测和任务文字 → 前处理 → select_action()
         │ 队列空时 → _get_action_chunk() → sample_actions() → 补充队列
         │ 取队首动作 → 后处理 → NumPy → LocalVectorEnv.step()
         └─ LiberoEnv.step() 返回下一步观测和任务结果
```

### 阶段 1 · 认识项目结构

**目标**　能从终端命令定位函数定义，能沿导入语句跳到另一个文件。先理解包、模块、函数、类和对象。此处的策略是产出动作的对象，环境是接收动作并反馈观测的对象。

| 顺序 | 找什么 | 中文副本 |
|---|---|---|
| 1 | `L0403`、`L0414`、`L0443` 看命令名、模块路径和 `src` 布局 | [pyproject.toml](annotated/pyproject.toml.txt) |
| 2 | `L0003`、`L0006` 看 Python 如何导入并调用 `main` | [Conda 的 lerobot-eval 启动脚本](annotated/launcher/lerobot-eval.txt) |
| 3 | `L1245` 读完四行 `main()`，再到 `L0078` 看环境包导入 | [lerobot_eval.py](annotated/scripts/lerobot_eval.py.txt) |
| 4 | 练习时阅读导入语句，追踪公开的名字如何进入包的命名空间 | [envs/__init__.py](annotated/envs/__init__.py.txt) |

**自查**　`lerobot.scripts.lerobot_eval` 对应哪个文件？`def main():` 和单独一行 `main()` 各做什么？

**练习**　从评测脚本 `L0078` 的 `close_envs` 开始查。找到中间经过的文件和写着函数定义的文件，用一句话解释中间文件的作用。

### 阶段 2 · 理解 SmolVLA 配置

**目标**　能读懂一个数据类的字段定义，知道真实运行值从哪里来。先看 `class 子类(父类)`、`变量: 类型 = 默认值`、`@dataclass`、字典和 JSON。读到 `PolicyFeature.shape` 时把它当作接口对单个样本的形状约定。

| 顺序 | 找什么 | 中文副本 |
|---|---|---|
| 1 | `SmolVLAConfig` 的字段 `L0024` 和检查函数 `L0110` | [configuration_smolvla.py](annotated/policies/smolvla/configuration_smolvla.py.txt) |
| 2 | 对照 `type`、状态和动作特征、采样参数以及骨干设置 | [checkpoint config.json](annotated/checkpoint/config.json.txt) |
| 3 | `L0047` 看如何处理 `--policy.path`，再对照回合数和环境配置 | [configs/eval.py](annotated/configs/eval.py.txt) · [EvalConfig](annotated/configs/default.py.txt) · [LiberoEnvConfig](annotated/envs/configs.py.txt) |
| 4 | `wrap()` 位于 `L0414`，配置类的 `from_pretrained()` 位于 `L0178` | [configs/parser.py](annotated/configs/parser.py.txt) · [configs/policies.py](annotated/configs/policies.py.txt) |

| 运行参数 | 原评测值 | 读源码时对应的计数 |
|---|---:|---|
| `chunk_size` | 50 | 模型一次生成的动作位置数 |
| `n_action_steps` | 50 | 每个动作块送入队列的动作步数 |
| `num_steps` | 10 | 生成同一块动作时做的 Euler 更新次数 |
| checkpoint 动作维度 | 7 | 交给环境的单步动作分量数 |
| `max_action_dim` | 32 | 生成模型内部的单步动作向量宽度 |
| `max_state_dim` | 32 | 从真实的 8 维机器人状态补齐后的宽度 |

**自查**　每次生成 50 个动作位置时，`num_steps` 在数什么？配置为什么检查 `n_action_steps <= chunk_size`？

**练习**　选 `chunk_size`、`n_action_steps` 和 `num_steps`，分别找到类默认值、JSON 值和命令行是否覆盖。用三行写出对照。

### 阶段 3 · 理解加载和逐步推理

**目标**　能跟踪一次 `select_action()`，判断它要调用模型还是只从队列取动作。先了解 `self`、`cls`、`super()`、`deque`、PyTorch 的 `nn.Module`，以及 Tensor 的转置和切片。

| 顺序 | 找什么 | 中文副本 |
|---|---|---|
| 1 | `L0063` 看名称如何解析成类，`L0256` 看 `make_policy()` | [policies/factory.py](annotated/policies/factory.py.txt) |
| 2 | `L0197` 到 `L0250` 看构造实例、定位文件、加载权重、切换设备和评测模式 | [policies/pretrained.py](annotated/policies/pretrained.py.txt) |
| 3 | `L0153`、`L0172`、`L0256` 和 `L0197` 看策略对象的构造、重置、出队和整块生成 | [modeling_smolvla.py](annotated/policies/smolvla/modeling_smolvla.py.txt) |

跟踪形状时用 `B` 表示环境批次，`T` 表示动作块的时间轴，`A` 表示对外动作维度。原评测中，底层生成的 `[1,50,32]` 裁成 `[1,50,7]`，交换前两轴后可按时间入队。每次从队首取出 `[1,7]` 的 Tensor，后面才把它转回环境使用的 NumPy 数组。

**自查**　谁定义了 `SmolVLAPolicy.from_pretrained()` 的方法体？`policy.eval()` 和 `torch.no_grad()` 各自影响什么？

**练习**　假设每次入队 3 个动作。画前 5 次 `select_action()` 的队列变化，标出哪几次触发了生成。

### 阶段 4 · 理解模型内部

**目标**　把采样分成「编码观测前缀」「缓存观测条件」「反复更新动作估计」三个步骤。只在读到相应代码时补充知识。用 18.06 的矩阵乘法理解线性投影和点积，用微积分中的变化率理解 Euler 更新。

| 顺序 | 找什么 | 中文副本 |
|---|---|---|
| 1 | `L0385` 和 `L0462` 谁准备相机图像和机器人状态，随后用 `L0604` 跟前缀的拼接 | [modeling_smolvla.py](annotated/policies/smolvla/modeling_smolvla.py.txt) |
| 2 | `L0202`、`L0217` 看视觉编码与词嵌入，`L0074` 看专家从哪构造 | [smolvlm_with_expert.py](annotated/policies/smolvla/smolvlm_with_expert.py.txt) |
| 3 | `L0414` 看前缀计算和后缀使用缓存的路径，`L0220`、`L0294`、`L0525` 看注意力实现 | [视觉语言模型与专家](annotated/policies/smolvla/smolvlm_with_expert.py.txt) · [掩码、时间嵌入和补齐](annotated/policies/common/vla_utils.py.txt) |
| 4 | `L0776` 看 `sample_actions()` 总入口，`L0698` 看后缀，`L0827` 跟一次速度预测 | [动作采样](annotated/policies/smolvla/modeling_smolvla.py.txt) · [Euler 循环](annotated/policies/common/flow_matching.py.txt) |

这一阶段会遇到的 PyTorch 操作有 `cat`、`Linear`、`expand`、`transpose`、`matmul` 和 `softmax`。先用小尺寸确认轴，再把实际配置带进去。每次遇到 dtype 转换都回到相应的代码，图像初始转换为 float32，视觉编码入口匹配视觉权重，注意力的 Q/K 点积又显式提升到 float32。

**自查**　作为观测条件的前缀由哪几种数据组成？哪段代码每个动作块执行一次，哪段会随着 `num_steps` 重复？

**练习**　画一张只含 5 个节点的数据流图。标上观测、前缀缓存、初始噪声、速度预测和最终动作块，再圈出重复执行的部分。

### 阶段 5 · 理解 LIBERO 的闭环

**目标**　跟踪一次控制步的相机、机器人状态、任务文字和动作，再找到成功标志的来源。先看 NumPy 数组、按轴索引、`torch.from_numpy()`、CPU/CUDA 转移与均值标准差变换。

| 路径 | 找什么 | 中文副本 |
|---|---|---|
| 选环境 | `make_env()`、`_env_factory()`、`SimConfig.resolve()` | [envs/factory.py](annotated/envs/factory.py.txt) · [sim/config.py](annotated/sim/config.py.txt) |
| 造环境、取图与状态 | `env_fns()`、`LiberoEnv.reset()`、`_format_raw_obs()`、`canonical_observation()` | [LIBERO 接入](annotated/sim/integrations/libero.py.txt) · [通用观测整理](annotated/sim/integrations/utils.py.txt) |
| 批处理和 Tensor 转换 | `LocalVectorEnv.reset()`、`_snapshot()`、`prepare_canonical_observation()` | [本地向量环境](annotated/sim/local_vector_env.py.txt) · [envs/utils.py](annotated/envs/utils.py.txt) |
| 恢复流水线 | `load_pretrained_policy_processors()`、`DataProcessorPipeline.from_pretrained()` 和 `__call__()` | [处理器工厂](annotated/processor/factory.py.txt) · [流水线](annotated/processor/pipeline.py.txt) · [SmolVLA 新建流水线的实现](annotated/policies/smolvla/processor_smolvla.py.txt) |
| 查看这次处理顺序 | 两份 JSON 里的 `steps` 和 `registry_name` | [前处理 JSON](annotated/checkpoint/policy_preprocessor.json.txt) · [后处理 JSON](annotated/checkpoint/policy_postprocessor.json.txt) |
| 观测键和文本 | 相机改名、批次轴检查、字符串换行和分词 | [重命名](annotated/processor/rename_processor.py.txt) · [批次处理](annotated/processor/batch_processor.py.txt) · [换行](annotated/processor/newline_task_processor.py.txt) · [分词](annotated/processor/tokenizer_processor.py.txt) |
| 尺度与设备 | `NormalizerProcessorStep`、`UnnormalizerProcessorStep`、`DeviceProcessorStep` | [尺度变换](annotated/processor/normalize_processor.py.txt) · [设备转换](annotated/processor/device_processor.py.txt) |
| 执行动作与读结果 | `rollout()`、`LocalVectorEnv._step_envs()`、`LiberoEnv.step()`、`StepResult.info()` | [评测循环](annotated/scripts/lerobot_eval.py.txt) · [向量环境](annotated/sim/local_vector_env.py.txt) · [LIBERO 接入](annotated/sim/integrations/libero.py.txt) · [底层 hf-libero 包装器](annotated/libero-dependency/env_wrapper.py.txt) · [仿真返回类型](annotated/sim/types.py.txt) |

原评测数据的几处形状如下。每路相机有各自的键，因此下表中的图像形状指一路相机。模型配置中的视觉特征描述包含 `[3,256,256]`，实际图像尺寸可以沿仿真转换和模型内缩放分别读出。

| 位置 | 图像 | 机器人状态或动作 |
|---|---|---|
| `LiberoEnv` 整理的单实例观测 | NumPy `uint8 [360,360,3]` | 标准化键名之前先整理成嵌套字典，接入层随后拼为 `float32 [8]` |
| `LocalVectorEnv` 批量观测 | NumPy `uint8 [1,360,360,3]` | NumPy `float32 [1,8]` |
| `prepare_canonical_observation()` 后 | CPU Tensor `float32 [1,3,360,360]`，范围 `[0,1]` | CPU Tensor `float32 [1,8]` |
| 策略流水线与模型输入准备后 | CUDA Tensor `float32 [1,3,512,512]`，范围 `[-1,1]`，进入视觉编码时再匹配视觉权重 dtype | CUDA Tensor `float32 [1,32]`，先归一化再补齐 |
| 后处理和仿真步 | 下一次观测从仿真返回 | 动作移到 CPU，向量层收到 `[1,7]` NumPy 数组，单环境收到 `[7]` |

**自查**　相机键名改动和图像轴顺序改动分别在哪里？哪个函数把 `[1,7]` 变成 `[7]` 并递交给单个仿真？仿真的成功标志怎样进入最终 JSON？

**练习**　只选一路相机和一个动作张量，自己画一张从 `env.reset()` 到下一次 `env.step()` 的表。每一行注明产生值的函数、轴含义、dtype 和设备，再对照上面的表。

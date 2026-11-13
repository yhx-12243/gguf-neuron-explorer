[English](README.md) | [日本語](README.ja.md) | **简体中文**

# gguf-neuron-explorer（`gne`）

查找、排序并引导 llama.cpp 上 GGUF 模型中的单个 MLP 神经元，再把找到的结果导出为原版 `llama-cli` / `llama-server` 可直接加载的 control vector。

![Qwen3.5-4B 的 3D 层地图，叠加了 sweep 结果](docs/images/ui-overview.png)

`gne` 将神经元的输出方向（`W_down` 的一列）推入残差流，生成文本，并与同方式的随机方向对照比较，测量发生了什么变化。它通过 llama-cpp-python 运行在 llama.cpp 上，因此可直接使用现有的量化 GGUF 文件。

- **Sweep 与排序。** 在固定的提示词和剂量集合下，引导一层中的数百个神经元，再按输出有多集中地堆到少数几个词上排序。每个分数都会与以相同方式推动的随机方向对照比较。
- **低成本初筛。**
  - **Logit lens：** 只用权重，在 CPU 上运行。
  - **Steer lens：** 神经元被 ± 剂量推动时下一 token 的变化。
  - **反向查找：** 对照两个人格，然后按与差异的余弦相似度对每一层中每个神经元排序。
- **每个神经元的档案。** Sweep 排名、每个提示词下的头部词、rollout、lens 结果以及上游/下游连接，全部集中在一个视图中。
- **配方。** 组合 `14:8949:+1.0`、`13:7956:-1.0` 等单元，在 3 列聊天（−、无干预、+）中比较，然后导出：
  - 导出为 control vector（`.cvec.gguf`），并用 `llama-completion --control-vector` 校验；
  - 导出为 npz、json 或 safetensors；
  - 作为 FFN down 偏置烘焙进 GGUF，适用于 llama.cpp 图会读取该偏置的架构（llama、granite、mistral3、nemotron、gpt2 等；不适用于 Qwen 或 Gemma——见 [configs/arch/bias_support.yaml](configs/arch/bias_support.yaml)）。
- **只在需要的位置引导（`when`）。** 仅推动生成的 token（`gen`）、仅 assistant 的回合、除之前回复外的所有位置（`skip-replies`）、仅在指定文本出现在输出后，或让剂量逐渐衰减。control vector 文件无法表达这些设置，因此 `gne serve --openai` 提供应用这些设置的 OpenAI 兼容端点。
- **音频输入。** 对于支持音频输入的模型（LFM3-Omni），`gne steer --audio speech.wav` 以及 OpenAI 风格消息中的 `input_audio` 部件可将语音直接放入对话。模型的 mmproj 将音频转换为嵌入，gne 在自己的循环中解码，因此配方、`when` 和激活捕获对语音回合的作用与文本相同。回复是文本；可用任意 TTS 朗读。
- **Gain 引导。** `22:2742:x4` 将神经元自身的输出乘以系数，而非添加固定方向：`x0` 移除它，负系数反转它。它只在神经元原本已经激活的位置生效。可导出为供 llama.cpp 的 `--lora` 使用的 LoRA 适配器 GGUF（同样适用于 Qwen），或导出为改写后的 GGUF；默认改写使用缩放补丁（只改写量化块的 scale），保留原始量化并精确缩放对应行。
- **方向（`gne direction`）。** 当某个人格或行为分散在许多神经元上时，改为使用方向。方向来自对比集：它在两种条件下读取问题（例如 A 侧带要求人格的 system 提示词，B 侧不带）。方向是两侧在回复开始前那个 token 上的平均残差之差，逐层计算。它可以通过消除方向并检查行为是否下降，或添加方向并检查行为是否上升，自动选择层。像神经元一样用 `30:dir:yandere:+0.3` 添加方向；方向会导出为 control vector。要添加人格，请用 `--positions response` 构建方向，它会在每个人格生成的回复上测量，而不是在指令后立即测量。与神经元一样，方向也会对照随机控制进行检查：每个候选层还会用 16 个相同范数的随机方向推动，只有当效果超过其 95 百分位时才会选择该层。`gne steer --contrast` 对整个配方做同样的事，并报告 `null_pct`。
- **方向消融（`gne ablate`）。** 从所有写入残差流的权重中投影消除一个方向（也称为 abliteration）。可导出为 LoRA 适配器，或导出为改写后的 GGUF：Q8_0/F16，或用 `llama-quantize` 重新量化回原始类型。配方可以带有 `ablate:`，而 `gne eval --compare-ablation` 会将其与将排名靠前的神经元置零进行比较。
- **副作用检查（`gne eval`）。** 在多个剂量下运行配方，并将人格变化与破坏情况放在一起查看：精确答案任务和格式任务、参考文本上的困惑度，以及语言混杂（日语回答中出现仅简体中文、谚文及其他文字）。可选的第二个 GGUF 可以给人格按 1–5 打分。保持在阈值内的剂量会标记为可发布。`--random-controls N` 还会在每个剂量下运行 N 个相同规模的随机方向配方。这样可以显示人格变化是否超过这些对照，以及仅推动本身会损失多少能力和困惑度。
- **谁在写入某种状态（`gne writers`）。** 激活差异显示哪些神经元对某种状态（例如人格 system 提示词）产生反应。`gne writers` 测量每个神经元在回复 token 上实际写入残差的量：激活差异 × 写入方向，包括 post-FFN norm。它还会将该状态的方向拆分为由嵌入、每层 mixer（attention、卷积）和每层 MLP 分别写入的部分。UI 提供了 writers 视图和对应的地图叠加层。
- **注意力头（`gne heads`）。** 注意力头与神经元一样有地址（`L21 H7`，包括 Qwen3.5 的线性注意力头）。`gne heads contrast` 在有无人格的情况下运行相同对话，并按以下标准对头排序：
  - 它们对触发词（例如其他人的名字）的注意力增加了多少；
  - 它们的注意力模式和输出变化了多少；
  - 它们的输出写入人格方向的程度。

  基线是将两个中立 system 提示词相互比较。`21:H7:x0` / `x2` 通过其 `W_O` 列块上的 LoRA 缩放一个头的输出。`gne eval --reactivity` 测量人格是否只在触发词存在时出现，并与相同系数的随机头对照。

  `gne heads listen`（头面板的 Listen 标签页）运行一段对话，并报告所选的注意力头（例如 `14:H0-3`）中，回复 token 对每个更早 token 分配了多少注意力。给定已保存的方向后，它还会沿该方向将头的输出归因到来源 token。每个 token 的 value 向量按该 token 的注意力权重和该头的门控加权，然后通过 `W_O`，并在具有该 norm 的架构中通过 attention 输出后的 norm。按层内所有头求和，这些贡献可重现该层的 attention 输出。命令会报告剩余误差；在 Qwen3、Qwen3.5 和 LFM3 上均低于 1%。
- **混合专家（Mixture of Experts）。** 在 MoE 层中，神经元写作 `L12 E37 N511`（专家 37 的神经元 511）；`gne experts contrast` 会按人格开启时每个专家被更多选中、以及更多写入人格方向的程度对专家排序；`gne experts sweep` 逐个消融或增强专家。在 MoE 层中，`add` 只会在每个 token 上推动神经元的写入方向；推动神经元本身则通过 gain（`12:E37:N511:x2`，仅在选中该专家时生效）。整个专家（`12:E37:x0`）和路由器偏置（`12:E37:route:+2`）通过缩放补丁应用：工作副本中量化块的 scale 会被精确改写，并在使用后恢复。
- **Web UI。** 每层和每个神经元的 3D 地图，支持切换叠加层（sweep mass、steer KL、logit-lens |z|、激活值、搜索命中、粘贴文本时哪些神经元被激活）。它还包括：
  - 排名列表和神经元面板；
  - 实时 3 列聊天；
  - 神经元之间的连线与 UMAP 布局；
  - rollout 的嵌入散点图；
  - 共享 GPU 的后台任务；
  - JP / EN / ZH 切换。
- **多 GPU、大型模型。** `gne worker` 为每个 GPU 运行一个进程，可以来自不同的 Python 环境（CUDA、ROCm），并共享结果文件夹。`gne sweep --workers auto` 和 `gne lens steer --workers auto` 会把神经元拆分到这些进程。预计超过 30 分钟的任务会停止，除非传入 `--yes`。`--budget 30m` 会利用沿某个方向写入最多的神经元、上位 steer-lens 神经元和 quick 设计来缩小任务，使其符合预算。`gne info` 会显示 GPU/CPU 层分配，以及针对大型模型的“先做反向查找”工作流。
- **Agent 访问。** MCP 服务器允许 Claude Code 或其他 agent 查询神经元、引导、运行任务，并在打开的 UI 中高亮或注释神经元。

这是对 [lyramakesmusic/neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) 中方法（目标是在修改版 vLLM 上运行 Qwen3.5-4B）的从零重新实现，面向 llama.cpp 和 GGUF。未使用该仓库的任何代码。

## 运行要求

- Python 3.11 或更高版本
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 0.3.36 或更高版本。除玩具模型外，请使用 GPU 支持构建。
- 推荐使用 NVIDIA GPU。纯 CPU（`--n-gpu-layers 0`）也能运行，但速度很慢。

测试环境：
- WSL2 下的 RTX 3080 12 GB，CUDA 12.4，Python 3.12，从源码构建的 llama-cpp-python 0.3.36。
- 模型：Qwen3-0.6B、Qwen3.5-4B（混合线性注意力）和 SmolLM2-135M-Instruct（arch `llama`），均为 Q8_0。
- 尚未测试 Metal。
- MoE 模型：granite-3.1-1b-a400m 和 LFM2-8B-A1B。参见[混合专家](#混合专家)。
- 对于尚未使用过的模型，请先运行 `gne verify`。

## 安装

```bash
git clone <this repository> && cd gguf-neuron-explorer
uv venv --python 3.12 .venv
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --python .venv/bin/python -e '.[ui,mcp]'
source .venv/bin/activate
```

可选依赖：
- `ui`：Web UI（FastAPI、uvicorn、umap-learn）。
- `mcp`：MCP 服务器。
- `dev`：测试（pytest、Playwright）。

[docs/setup.md](docs/setup.md)（日语）介绍 WSL 下的 CUDA 和 ROCm，也介绍如何为每个 GPU 建立一个环境，并用 `gne worker` 将 sweep 或 steer lens 拆分到这些环境。

## 快速开始

每条命令都可以使用 `-m <model.gguf>`，或读取 `GNE_MODEL`。结果保存在 `~/.gne` 下（可用 `--root` 或 `GNE_ROOT` 覆盖），后续命令和 UI 会复用这些结果。

```bash
export GNE_MODEL=models/Qwen3.5-4B-Q8_0.gguf

gne info                                   # 架构、警告、支持的功能
gne verify                                 # 与 llama.cpp 的等价性测试（新架构请先运行）
gne stats resid                            # 每层残差范数（rel 剂量需要）

gne lens logit --layer 22                  # 仅用权重的首轮初筛
gne sweep --layer 22                       # 384 个随机神经元 + 64 个随机对照，design main_rel
gne rank --run L22-main_rel                # 按词表质量排序
gne neuron 22 2742                         # 一个神经元的全部已知信息

gne steer --unit 22:2742:+0.8 -p "Who are you?" --compare
gne chat --unit 22:2742:+0.8 -p "Tell me about yourself."

gne reverse --contrast configs/contrast/evil.yaml --verify    # 某个人格的候选神经元
gne export-cvec --recipe configs/recipes/evil.yaml            # → ~/.gne/export/evil.cvec.gguf

gne steer --recipe configs/recipes/alien.yaml -p "Who are you?" --compare     # when: gen（仅生成的 token）
gne steer --unit 22:2742:x4 -p "Is there life on other planets?" --compare   # gain：在神经元激活处 ×4
gne eval --recipe configs/recipes/alien.yaml                  # 各剂量下的能力 / ppl / 混杂 / 人格变化
gne fieldtest -m baked.gguf --against original.gguf           # 按真实用法测量长任务、工具调用与 perplexity，并逐文件比较
gne gain --recipe configs/recipes/alien_gain.yaml             # → ~/.gne/export/alien_gain.lora.gguf

gne direction --contrast configs/contrast/evil.yaml --select auto   # 逐层方向 + 层选择
gne steer --unit 16:dir:evil:+0.6 -p "What do you think about people?" --compare
gne steer --ablate evil --system "You are a cruel AI." -p "Who are you?" --compare
gne ablate --direction evil                                   # → ~/.gne/export/evil-ablate.lora.gguf
gne eval --compare-ablation evil                              # 前 20 个神经元 ×0 与方向消融对比
gne steer --recipe configs/recipes/alien.yaml --contrast configs/contrast/alien.yaml   # 与 16 个随机配方对照（null_pct）

gne heads contrast --contrast configs/contrast/yandere_react.yaml   # 对注意力头排序（有人格 vs 中立）
gne heads show 15 2                                           # 单个头：排名、W_O 块 logit lens、已尝试的 gain
gne heads listen --heads 14:H0-3 --system @persona.txt -u "I had lunch with a coworker." --direction wr_mix14   # 回复的注意力与逐 token 贡献
gne steer --unit 15:H2:x2 -p "I had lunch with Emily today." --compare   # 头 gain（W_O 上的 LoRA）
gne eval --recipe R --reactivity configs/contrast/yandere_react.yaml --head-controls 8

gne direction score --name evil                               # 逐层 A/B 分离度，记录基准层（不生成）
gne heads causal --name evil                                  # 输出置零后会降低沿方向投影的注意力头（不生成）
gne heads allocate --name evil --target -3 --n-heads 6,5,4    # 按效率分配强度，并比较头更少的组
gne heads allocate --name evil --target -3 --method greedy    # 或逐个提高最有效注意力头的强度
gne heads candidates --alloc evil --prompts held_out.yaml     # 为每个组 × 倍率生成回复，写出供匿名评分
gne heads candidates --tally evil                             # 汇总每个候选的评分
gne steer --recipe ~/.gne/heads/<model_id>/evil.alloc-n4.yaml --system "You are a cruel AI." -p "Who are you?" --compare   # 仅所选头沿方向的分量
gne eval --recipe ~/.gne/heads/<model_id>/evil.alloc-n4.yaml --random-controls 16   # 与相同强度的随机头对照
gne eval --recipe R --compare-methods                         # 相同效果下比较全层消融 / 头分量 / 头 gain / 神经元消融
gne headproj export --recipe R                                # 每层 rank-1 LoRA（--bake 只改写相关张量）

gne direction --contrast configs/contrast/zundamon_endings.yaml --positions response   # 键控写入要写入的方向
gne direction key --contrast configs/contrast/zundamon_endings.yaml --layers 8,14,20   # 仅在模型即将写出 です。/ます。 时反应的键
gne direction key-check --key zundamon_endings --contrast configs/contrast/zundamon_endings.yaml   # 在留出问题上，它比用打乱标签构建的键更能分开这些位置吗？
gne steer --unit 14:ffn:keyed:a1@zundamon_endings#zundamon_endings -p "おすすめの趣味を教えて。" --compare   # 仅在键反应的位置写入方向
```

`export-cvec --verify` 还会检查 `llama-completion --control-vector` 是否给出与 `gne` 相同的 greedy 输出。它会在 `build/` 和 `PATH` 中查找 `llama-completion`，也可以传入 `--llama-completion`。

然后在 llama.cpp 中使用该向量：

```bash
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector ~/.gne/export/evil.cvec.gguf
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --control-vector-scaled ~/.gne/export/evil.cvec.gguf:0.6
llama-cli -m models/Qwen3.5-4B-Q8_0.gguf --lora ~/.gne/export/alien_gain.lora.gguf
```

配方是一个小型 YAML 文件。`when`、`sampling` 和 gain 单元都是可选的：

```yaml
name: example
model_id: qwen3.5-4b-64d32ead87
dose_unit: rel
when: gen                                       # all / gen / assistant / skip-replies / after:<text> / ramp:<a>-<b>
units:
  - {layer: 22, neuron: 2742, dose: +0.8}       # add：推动神经元的输出方向
  - {layer: 13, neuron: 7956, mode: gain, k: 0} # gain：移除该神经元的输出（×0）
  - {layer: 16, kind: direction, direction: evil, dose: +0.3}   # 添加来自 gne direction 的方向
  - {layer: 15, kind: head, head: 2, mode: gain, k: 2}           # 注意力头：缩放其输出（×2）
  - {layer: 16, kind: head, head: 8, mode: proj, direction: evil, s: 6}   # 注意力头：只改变其沿方向的写入（s = 1 消除）
  - {layer: 14, kind: ffn, mode: keyed, direction: zundamon_endings, key: zundamon_endings, alpha: 1}   # 仅在键对 ffn_down 输入产生反应的位置写入方向
ablate: {direction: refusal, scale: 1.0}        # 从所有写入残差流的权重中投影消除方向（LoRA）
sampling: {temperature: 1.0, top_k: 20, top_p: 0.95, min_p: 0.05, presence_penalty: 0.9}   # 试用方式（不使用 seed）
```

`sampling` 表示配方预期的试用方式。它还可以包含 `dry_multiplier`、`dry_base` 和 `dry_allowed_length`（默认关闭），即 DRY 惩罚，用于阻止多轮对话逐字重复之前的回复（使用 llama.cpp 的名称，OpenAI 兼容端点也接受）。`steer` 和 `chat` 会使用它（可用 `--temperature`、`--top-k`、`--top-p`、`--min-p`、`--presence-penalty`、`--sample-seed` 覆盖，或用 `--greedy` 关闭），UI 聊天会将其作为“配方的设置”提供，OpenAI 兼容端点会在请求未发送采样值时使用它。control vector 无法携带这些设置，因此 `export-cvec` 会打印对应的 llama-server 参数。没有 `sampling` 时，生成使用 greedy。`eval` 始终使用 greedy。

完整列表请运行 `gne --help` 和 `gne <command> --help`。帮助文本为中文。

### 剂量

剂量默认使用 **rel** 单位：`dose × 该层 ‖residual‖ 的中位数 × unit(W_down[:, n])`。

在 Qwen3.5-4B 上：
- 0.5 会引入一个话题，0.8 会接管话题，1.0 及以上开始破坏文本。
- 3.0 的剂量会故意破坏文本，以显示神经元“想写”什么。
- 第 0–2 层残差较小，因此 rel 剂量无法直接迁移到那里。
- 第 0 层无法用 control vector 推动，但 bake 可以推动它。

### 耗时

RTX 3080 上的 Qwen3.5-4B Q8_0：

| 步骤 | 时间 |
|---|---|
| `stats resid`（所有层） | 约 16 秒 |
| `lens logit` | 每层约 33 秒（CPU） |
| `lens steer`（全部 9216 个神经元） | 每层约 22 分钟 |
| `sweep` `main_rel` | 每个神经元约 6.4 秒（48 分钟完成 385 个单元） |
| `reverse` | 约 6 秒；对 20 个候选执行 `--verify` 约 4 分钟 |
| `eval`（55 个任务 × 5 个剂量） | `when: all` 约 41 秒；`gen` 约 90 秒 |
| `direction`（24 段对话）+ `--select auto`（9 层） | 约 35 秒 + 约 35 秒 |
| `ablate` LoRA / `--bake` Q8_0（带 `--verify`） | 约 10 秒 / 约 40 秒 |

完整执行一层的 `main_rel` sweep 大约需要 16 小时。请先用 lens 和反向查找缩小候选范围。

### 混合专家

使用 granite-3.1-1b-a400m（`granitemoe`，32 个专家，选择 8 个）和 LFM2-8B-A1B（`lfm2moe`，32 个专家，选择 4 个，带路由器偏置）测试。

```bash
gne stats act -m M --layers 2-22                     # 同时统计每个专家被选中的频率
gne experts contrast -m M --contrast configs/contrast/yandere_rules.yaml
gne experts show -m M 12 37                          # 使用情况、它被选中的 token、对照排名、上位神经元
gne experts sweep -m M --layer 12                    # 逐个消融（x0）和增强（x2）专家，按变化的词排序
gne steer -m M --unit 12:E37:x0 -p "..." --compare   # 移除第 12 层的专家 37
gne steer -m M --unit 12:E37:N511:x2 -p "..." --compare
```

| 单元 | 含义 | 应用方式 |
|---|---|---|
| `12:E37:N511:+0.8` | 在每个 token 上添加该神经元的写入方向（添加的是方向，并不推动神经元本身） | control vector |
| `12:E37:N511:x2` | 选中专家 37 时，将该神经元的激活乘以 k | 按专家堆叠的 LoRA |
| `12:E37:x0` / `x2` | 移除专家 37 的输出或将其加倍 | 缩放补丁 |
| `12:E37:route:+2` | 移动专家 37 的路由器偏置（仅限带路由器偏置的架构，例如 `lfm2moe`） | 缩放补丁 |
| `12:S:N5:x2` | shared expert 神经元 | LoRA（与 dense 神经元相同） |

缩放补丁会改写受影响行量化块中的 fp16 scale，使一行或某个专家的权重切片精确变为原来的 k 倍（k = 0 时写入零字节）。它适用于 Q4_0 到 Q8_0、Q2_K 到 Q6_K、除 IQ1_M 外的 IQ 类型，以及 F16/BF16/F32。补丁写入 `~/.gne/work/` 下的工作副本（绝不会改动原始 GGUF），并在每次使用后恢复；重新加载副本对 1.4 GB 模型约需 0.3 秒，对 5 GB 模型约需 0.9 秒。`steer`、`eval`、UI 聊天和 OpenAI 兼容端点会自动应用这些补丁。列方向（`W_down[:, n]`）无法用这种方式打补丁。在 MoE 层内，sweep、lens 和反向查找按 `e × n_ff_exp + n` 为神经元编号。

### CHiPS：只改变少数注意力头中沿方向的分量

这里介绍的方法称为 **CHiPS**（Causal Head-isolated Projection Steering，因果头隔离投影引导）。顾名思义，它只对由因果效果选出的注意力头应用沿方向的投影（即停掉某个注意力头时结果变化多大）。与从每一层每个写入残差流的权重矩阵中移除方向的全层消融相比，关键区别就在于这一限制。

用 `gne direction` 创建的方向也可以应用到单个注意力头上。每个注意力头都会在每个 token 向残差流加入一个向量。`proj` 单元将该向量拆成沿方向的部分和其余部分，只改变前者：强度为 `s` 时，注意力头沿方向的写入变为原来的 `1 − s` 倍，因此 `s = 1` 会消除，`s = 2` 会反转，负的 `s` 会强化；注意力头写入的其他内容保持不变。这就是它与头 gain（`L:Hh:xk`）的区别：gain 会缩放整个输出，包括与方向无关的全部信息。

由于变化量与当前输入下该注意力头沿方向写入的量成正比，它只在该头沿方向写入时生效，在其他位置几乎不产生影响。同一性质也决定了限制：它无法创造注意力头原本没有写入的行为。强化承载人格的注意力头在没有额外提示的普通对话中不会产生效果，只有先由 system 提示词、少量神经元或方向推动让行为出现时才会生效。

注意力头和强度都在不生成文本的情况下选定。度量是残差沿该方向的投影在示例上的平均值，它表示人格或行为出现的强度，下文称为“投影”（JSON 输出中的 `score` 等字段）。

- `gne direction score` 让对照集两侧读取到回复即将开始的位置，并逐层测量两侧沿方向相距多远；分离得最好的层会被记录为基准层。
- 随后 `gne heads causal` 先停掉整层注意力头，再在关键层中逐个停掉注意力头，记录投影下降量，并与该头沿方向直接写入的量并列。较后层的注意力头即使不沿基准层方向写入也可能重要，因此还会在后续几层测量下降。
- `marginal` 模式下，同一命令从当前组出发，逐次略微提高单个注意力头的强度，以测量单位强度带来的下降（其效率）；这一排名通常与直接停掉注意力头得到的排名不同。
- `gne heads allocate` 按效率比例分配强度，缩放整个组直到最难生效的示例达到目标值，并构建头数更少的组，以便选择与较大组效果相当的最小一组。

`gne heads pipeline --spec file.json` 可从包含对照集（以及可选设置）的单个文件一次运行这些步骤：使用全部问题创建方向、记录基准层、筛选层再筛选单个注意力头、分配强度。

- **问题数量：** 问题可以有数百个。停头阶段使用等间隔抽取的子集（默认 64 个），因为每次试验都会读取它使用的全部问题。
- **排除候选：** 若停掉某些层或头会让 B 侧（中性侧）残差变化超过其自身大小的一半，则将其排除，因为它们不是在作用于方向，而是在破坏模型；停掉第一层的全部注意力头通常就会如此。
- **目标：** 未指定目标时，强度瞄准两侧平均投影的中点。中点通常只能消除约一半行为，因此要完全消除时请将目标写为 `B`：强度会瞄准 B 侧投影平均值（若最强头卡在默认上限 `s_max = 30`，请提高上限）。
- **恢复：** 中止后使用相同文件和设置重新运行会复用已创建的方向和因果结果，并从分配阶段继续（`--fresh` 或 UI 中的“从头重新创建”会重建全部内容）。

比较表的“剩余”列统计某组未能降到目标的 A 侧问题数。`gne heads refine --name NAME`（表中的“从剩余中添加”）只将这些剩余问题作为 A 侧、原始 B 侧问题作为 B 侧，在两侧应用该组后重建方向，并运行相同步骤来选出要添加的注意力头和强度。该组已经改变的部分在两侧相同，因此新方向只保留该组尚未消除的差异。补充注意力头是在原组应用状态下找到的，因此绝不单独使用：对于比较表中的每个组，它会写出一个合并原组与该组的配方，并沿原方向重新测量剩余数量和 B 侧偏移。表中的加载和 eval 按钮使用这些合并配方。

这些步骤只读取问题、从不生成，因此用于测量模型内部的上下文会按最长对话分配，并先释放聊天留下的生成上下文；在大型模型上，这可避免显存溢出到系统内存而导致一切变慢。

`configs/pipeline/` 包含五种用途的模板，UI 的模板菜单也提供相同的五种。每个模板的 `about` 说明比较对象以及候选的缩放方向。

- 找出承载 system 提示词所要求人格的注意力头（仅 A 侧有 system）
- 只保留两个人格之间的差异（两侧都有 system）
- 找出承载神经元配方效果的注意力头（A 侧有配方）
- 捕捉不同问题类型之间的差异，例如日语与英语（两侧各有问题）
- 捕捉回复写到一半时才出现的行为，例如思考陷入循环（两侧都是未完成的回复）

不生成文本选出的强度不一定适合真实回复，因为投影是沿方向的量，可能与行为显现强度和文本自然度发生偏移。因此最后一步是读取回复。

`gne heads candidates --alloc NAME` 取分配结果比较表中的每个组，将其强度乘以若干整体倍率（默认 0.5、1、1.5、2、3；分配默认目标只到中点，因此有用倍率常常大于 1），并让每个候选连同未修改模型回答两侧问题。所有候选共用同一随机种子，因此回复差异只来自干预。正倍率会消除 A 侧行为，因此读取其 A 侧回复；负倍率会反转符号并强化行为，因此改为读取其 B 侧回复。

- 命令会为每个候选报告符合 `validate.match` 的回复比例、措辞重复程度，以及未被读取一侧相对未修改模型的回复变化比例。
- 它把要读取的回复混合到一个文件中，不标明来自哪个候选，并附上评分指南。由人或另一个模型为每条回复的目标行为显现程度（P）和自然度（C）打 1 到 5 分。
- `--tally` 按消除方向取 6 − P 与 C 的较小值（行为消失且回复自然）、强化方向取 P 与 C 的较小值的平均值对候选排名。
- 用于选择强度的问题会让候选显得比实际更好，因此请用 `--prompts` 传入未用于选择的问题。
- 若要尝试由注意力头放大的小幅神经元推动，可用 `--with-recipe` 给包括未修改模型在内的每个候选叠加配方。

在 UI 中，direction 视图“从 JSON 创建方向和注意力头”标签页比较表下方的表单可创建候选；“强度候选”标签页会按问题排列回复，或一次显示一条并隐藏名称，用数字键评分。每次评分都会立即保存，因此可以中断后继续；全部评分完成后，该标签页会显示汇总，并提供将最佳候选加载到配方面板或在 1 列聊天中试用的按钮。

对照集两侧通常由 system 提示词构造。只需写出其中一侧：没有 system 提示词的一侧会在完全无 system 的情况下读取问题，因此只有 `system_a` 的对照集会比较人格与模型正常回答。

对照集也可以通过应用配方（`recipe_a`、`recipe_b`）构造一侧或两侧。此时两侧差异就是配方自身的效果，因此 `gne direction` 会创建神经元配方产生的方向，而 `gne direction score` 和 `gne heads causal` 会在应用配方时找出承载它的注意力头。`configs/contrast/evil_neurons.yaml` 为 Qwen3.5-4B 上的 evil 神经元配方执行此操作。

它也可以为每侧提供各自的问题（`prompts_a`、`prompts_b`），使未修改模型读取两类问题，并以二者差异作为方向；`configs/contrast/ja_vs_en.yaml` 比较日语提问与同一问题用英语提问。当问题不同时，方向也会混入话题、长度等差异，因此请尽量保持其他条件接近。

比较同一回复在两种 system 提示词下读取结果的工具（`gne heads contrast`、`gne writers`、`gne experts contrast`）不接受用配方构造的对照集、只写出一侧 system 提示词的对照集，或两侧问题不同的对照集。

问题可以是单个字符串或对话历史：`[{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}, {"role": "user", "content": "…"}]`，user 与 assistant 交替并以 user 回合结束。每个问题都从干净状态单独读取，并在最后一个 user 回合的回复之前测量；一个问题不会带入下一个。需要测量只在长对话深处或数轮交流后才出现的行为时，可使用历史。

对话历史也可以以 assistant 回合结尾，此时它会被当作仍在书写的回复：没有任何内容将其闭合，测量发生在其最后一个 token，候选从那里继续。以 `<think>` 开头的文本会被视为正在进行的思考；其他情况下 gne 读取问题时会关闭思考，因此任何其他文本都算作正式回复。用于捕捉回复写到一半才出现的行为，例如思考陷入循环，而回复开始前的位置无法显示这些行为：在两侧写出相同的问题和相同的已完成部分，只改变后续内容。`gne fieldtest` 将问题发送给 llama-server，不接受未完成的回复。

结果是普通配方，包含诸如 `16:H8:s6@evil` 的单元。同一层中共享同一方向的注意力头会折叠为秩 1 LoRA，因此适配器保持很小。control vector 无法表达它。

- `gne headproj export` 写出（`gne gain` 也可，但不支持 `--via-f16`、`--into` 和 `--imatrix`），`--bake` 只改写相关张量；加上 `--via-f16` 可先以 F16 写出，再用 llama-quantize 恢复到原始类型。
- 在 UI 中，配方标签页的“LoRA / 烧录”也可完成同样操作，选项包括格式、重写张量的类型、恢复为原始类型、与 LoRA 校验以及文件名（默认配方名；仅在确认后覆盖已有文件）。
- 恢复到 IQ1、IQ2、IQ3_XXS 等低位宽类型需要 imatrix（告诉量化器哪些输入需要保留精度的权重）；将 `<模型名>.imatrix.gguf`（或 `.dat`）放在模型旁即可自动使用，无需显式传入。

烧录不使用 GPU（它逐块读取和重写文件），因此也适用于 GPU 放不下的大模型：可以用量化模型分析，再烧录到同一模型的未量化 GGUF（F16 / BF16）（`--into`，UI 中的“烧录目标”）。方向来自分析模型，重写矩阵来自目标模型，类型保持不变（`--requant keep`）。强度是在量化模型上测得的，因此会有少量差异，也无法与 LoRA 版校验。之后再用 llama-quantize 量化烧录后的 F16 / BF16，也可避免只有重写张量处于不同类型；LoRA 可以立即下载。

gne 会将配方（层、注意力头、强度）作为 `gne.*` 元数据记录在导出文件中；若文件要交给别人，请使用“不包含 gne 的记录”（`--strip-meta`），它还会检查元数据中是否残留本地路径（使用 imatrix 重新量化时，llama-quantize 可能记录生成 imatrix 所用文本的路径）。以这种方式写出的 LoRA，其名称字段（`general.name`）为 `adapter`（可用 `--name` 设置），因此元数据中不会出现原始模型名或标识符；烧录文件就是模型本身，会保留模型原有的元数据。即使没有记录，LoRA 张量仍会直接显示层、注意力头和方向，烧录文件与原模型比较也仍能看出改了什么、如何改。

依赖某个配方前，请对照控制进行检查。

- `gne eval --random-controls N` 将相同强度应用到随机选出的注意力头。
- `gne eval --compare-methods` 将全层消融、将同一批头整体缩放、移除顶部神经元调到相同投影并比较副作用。
- `gne eval --alloc NAME --n-heads 6,5,4,3` 显示减少注意力头时副作用如何变化。

投影是沿用于选头的同一方向测量的，因此报告结果时请使用从输出测得的指标，例如对照集的 `validate.match`、能力和困惑度。对于使用方向的配方，eval 还会在每个倍率和每个对照下测量方向指标（投影下降量、偏移、一致率）；当对照集有数百个问题时开销很大，因此默认读取等间隔的 64 个问题（`--dir-max-examples`，UI 的 eval 标签页中的“问”），也可以关闭（`--no-direction-metrics`，UI 的“方向指标”）。

在 Qwen3.5-4B 上，这样选出的两个注意力头阻止了 evil system 提示词所要求的人格（之前 83% 的回复被判为符合，之后为 0），同时没有改变能力和困惑度。将效果对齐到相同投影后，全层消融使困惑度上升 15%，而将同样两个注意力头整体缩放会损失 14 点能力。方法详情见 [docs/head-component-steering.md](docs/head-component-steering.md)。

权重编辑本身并不新鲜：对于单个注意力头，它与 [Steer2Edit](https://arxiv.org/abs/2602.09870)（Sun 等，2026）具有相同的秩 1 形式；我们是在完成这一实现后才知道该工作。CHiPS 的区别在于选择注意力头的方式（按因果效果；Steer2Edit 按与 steering vector 的对齐程度）、设定强度的方式（按实测效率并手动调整整体缩放；Steer2Edit 使用闭式规则和网格搜索），以及用随机注意力头检验结果。两者尚未在同一任务上比较。

文章见 [CHiPS：知道与回答之间的少数注意力头](https://balaena01.github.io/chips-steering/)。在 Qwen3.5-4B 上，896 个注意力头中的 4 个在三轮测试（1,200 道题）中，把因错误建议而放弃正确答案的比例从 23.5% 降至 15.6%，同时带有正确建议时的准确率从 93.2% 降至 90.0%。第三轮（发布后新增的 300 道题）表现较差：单看这一轮，放弃正确答案比例的下降不显著（p=0.073），而带有正确建议时的准确率下降了 5.3 个百分点。相同强度下的随机注意力头没有降低这个比例。配方、全部回复和脚本见 [balaena01/chips-steering](https://github.com/balaena01/chips-steering)。

### 键控写入：只在选定状态下写入方向

`keyed` 单元只在模型处于选定状态的位置，将方向写入残差流（例如，只在模型即将写下 です。 或 ます。 的位置，写入 のだ 句尾的方向）。CHiPS 选择要改变哪个方向；键控写入选择在什么状态下写入。两者相互独立，并进入同一个 LoRA。与 CHiPS 不同，键控写入可以添加模型尚未产生的行为。

**工作原理。** 该单元向一个写入残差流的权重矩阵（`ffn_down`、`attn_output`、共享专家或单个专家）加入秩 1 变化 `ΔW = α d κᵀ`。随后该矩阵在通常输出之上写入 `α (κᵀx) d`。`x` 是该矩阵的输入激活，`κ` 是键；`κᵀx` 表示键在该位置反应的强度，写入量与它成正比。

键来自对照集，token 位置被分为写入位置和不希望写入的位置（JSON 字段名中的 `_a` 和 `_b`，例如 `rows_a`）。它是写入位置处输入的平均值 `μ`，去掉由不希望写入的位置（以及任何 `dont_touch` 组）的输入所张成的子空间后，再缩放使 `κᵀμ = 1`。在典型的写入位置，该单元正好添加 `α d`；在不希望写入的位置，它不添加任何内容。方向 `d` 是 `r_L`，即构建方向所用对照集中该层两侧均值的差；它按原始大小使用，而不是作为单位向量，因此 `α = 1` 会原样写入测得的差异，负的 `α` 会抵消它。

**选择写入位置（`positions`）。** 有两种方式。

- 使用 `positions: response`（回复 token）、`assistant` 或 `last` 时，写入位置是 `system_a`（或 `recipe_a`）下对话中的那些位置。这种状态在推理时通常不会在没有该提示词的情况下出现。
- 使用 `positions: "before:<regex>"` 时，两类位置都来自推理时上下文（无 system 提示词）中模型自己的回复：模型即将写出匹配该正则的文本之前的位置是写入位置，同一回复中的其他所有位置都是不希望写入的位置。`configs/contrast/zundamon_endings.yaml` 使用 `before:(です|ます)。`，因此 Zundamon 方向只会在模型即将用 です 或 ます 结束句子时写入。

`dont_touch` 列出不应写入任何内容的若干组问题（上例中的英语问题）。键在构建时会确保它也不会在它们的回复中产生反应。

如果要改变的文本出现在回复的最开头（例如以 ありがとう！ 开头的回复），写入位置就是回复开始前的位置。这个状态对每个问题看起来都一样，因此如果 `dont_touch` 中没有普通问题，键会在每条回复的开头产生反应。请把普通问题放进 `dont_touch`（如 `configs/contrast/tsundere_thanks.yaml`）：构建键时，每条 `dont_touch` 回复的开头也会自动被排除在键之外。

**实测。**

- 在 Qwen3-0.6B 上，这个示例把大多数 です/ます 句尾变成了普通句尾，但几乎从不产生 のだ，而在每个位置都添加同一方向会让回复崩坏：键控写入只限制方向写入的位置，无法写出方向本身不携带的内容。
- 在 Qwen3.5-4B 上，其方向确实携带 のだ，键控写入（三层注意力输出矩阵，α = 2）让 12 条留出日语回复中的 4 条以 のだ 结尾且未陷入重复，并且每条英语回复都保持英语且切题。在每个位置都添加该方向时，只有在大多数日语回复崩坏的强度下才会产生 のだ；而将方向按原始大小的两倍添加时，至少一半英语回复也会崩坏。
- 不过，更强的键控写入也会陷入循环：一旦写出，键所反应的状态往往会持续，于是单元会继续写入（详情见 [docs/keyed-write.md](docs/keyed-write.md) §6–7）。

```bash
gne direction --contrast configs/contrast/zundamon_endings.yaml --positions response   # 要写入的方向
gne direction key --contrast configs/contrast/zundamon_endings.yaml --layers 8,14,20   # 用偶数编号问题构建的键
gne direction key-check --key zundamon_endings --contrast configs/contrast/zundamon_endings.yaml   # 在奇数编号问题上，对照用打乱标签构建的键
gne direction key-verify --layer 14 --key zundamon_endings --direction zundamon_endings          # LoRA 按公式 α (κᵀx) d 改变输出
gne steer --unit 14:ffn:keyed:a1@zundamon_endings#zundamon_endings -p "おすすめの趣味を教えて。" --compare
gne eval --recipe R                                   # 能力 / ppl，以及 keyed 相对 add、gain 和 proj 的选择性
gne headproj export --recipe R                        # 每层、每个键的 rank-1 LoRA（--bake 保持 Q8_0 张量为 Q8_0，其他为 F16）
```

**检查键。** `gne direction key-check` 在未用于构建键的示例上测量键的反应强度（`κᵀx`）。

- 键由偶数编号的问题构建，并在奇数编号的问题上检查；与构建键所用示例的任何重叠都会被检测并报告为 `held_out: false`（并非在未见示例上测量）。
- 反应会与用打乱后的位置标签按相同方式构建的对照键比较（JSON 输出中的 `pseudo_*`）。
- 只有当某层的选择性（写入位置的平均反应减去不希望写入位置的平均反应）超过对照键的第 95 百分位时才使用该层；即使是对照键，示例级 AUC 也可能达到 1。
- `gne direction key` 会在写入位置太少，或大多数写入位置位于回复开头时发出警告。`key-check` 会报告键在该键自己的问题以及 `dont_touch` 问题的回复开头处的反应；如果它在那里产生反应，就把普通问题作为 `dont_touch` 组加入，并设置 `positions: last`（仅回复开始前的那个位置）。

对于带键控单元的配方，`gne eval` 还会在不生成文本的情况下测量写入在每类位置（包括 `dont_touch`）移动残差的量，并与在写入位置达到相同效果的 add、gain 和 proj，以及对照键进行比较。只有当键控写入对不希望写入位置的扰动小于这些方式时，才值得使用。

**语法与限制。** 单元形如 `14:ffn:keyed:a1@DIR#KEY`（`a1` 表示 α = 1；`ffn` 也可替换为 `attn`、表示共享专家的 `S`，或表示专家 37 的 `E37`）。`when` 不适用，因为 LoRA 无法按位置切换，而 control vector 也无法表达键控写入。对于 MoE 层中单个专家上的键（`--target expert:e`），`gne direction key-experts` 会先显示每类位置中各专家被选中的频率；请选择在写入位置被选中频率高得多的专家。

**分享。** 要分享结果，LoRA 是首选。烧录时，Q8_0 张量默认保持 Q8_0，其他类型变为 F16。

- 将 Q8_0 张量写成 F16 会改变矩阵乘法的计算方式（Q8_0 乘法还会把激活舍入到 8 bit），并且它与 LoRA 的偏差会更大。
- 只有当 Q8_0 张量中的舍入会损失 10% 或更多的小秩 1 修改时，才将其写成 F16。
- 对于不是 Q8_0 的专家张量，`--requant keep`（尽可能按原始类型写回）会逐字节保留未编辑的专家，而不是把整个张量转换为 F16。

在 UI 中，direction 视图的“keys（keyed）”标签页用一个按钮运行整个循环：从对照集（文件，或从 `configs/keyed/` 中三个模板之一开始的 JSON：改变句子结尾方式、只改变某一类场景中的回复、使用由 system 提示词制造的状态）出发，构建方向、构建键，并在留出问题上检查键。随后它会逐层显示键的反应强度，将已选好新方向的键控单元加入配方，并从每一行检查 LoRA 是否按公式写入。在 MoE 模型上，它还会显示每个专家被选中的频率，从表中选择一个专家即可在该专家上设置键。[docs/keyed-write.md](docs/keyed-write.md) 说明了方法、完整示例以及 MoE 模型的细节。

**抑制 CHiPS 配方留下的场景。** 当 CHiPS 配方在某些问题上仍让行为出现时，`gne direction key-from-behave --behave B --recipe R --direction D` 会从应用该配方运行得到的 `gne behave` 结果中收集这些问题，并将键控对照集写入 `<root>/keyed/`（以 α = −1 写入方向 D，在应用 R 的状态下构建键）。它会出现在键标签页的模板中，生成的键控单元加入 R 后合并为一个 LoRA（docs/keyed-write.md §17）。键标签页也能在界面上完成整套流程：“从 CHiPS 留下的场景创建”会应用配方在 A 侧问题上运行 behave，构建 JSON，并放入输入框以运行一键流程。

**寻找键有效的位置。** `key-check` 只能说明键能否区分场景；即使某层的 key-check 余量很大，写入后行为也可能不变。`gne direction key-causal --contrast C` 会为每个候选（默认全层 × `ffn`、`attn`）构建键，并在未用于创建的回复上，在不生成文本的情况下测量加入每个键控单元后行为部分（`positions` 正则表达式匹配的文本；如果是选择题行为，则用 `--probe` 给出的答案组）的对数概率下降多少，并与用打乱标签构建的比较用键以及 `dont_touch` 回复的变化并排比较。候选低于所有比较用键的第 5 百分位时会打上标记，表还会显示仅凭偶然会出现多少标记（候选数 × 5%）。表按每个位置（写入目标 × 层）一行，列出三项选择依据：各强度下的降幅（显著时带标记）、不希望写入位置回复上的副作用、是否在所有强度下都显著，以及降幅除以副作用；每一列都可排序。目前选择这种方式的依据只有一个案例（Qwen3.5-4B 的句尾），而且只按降幅选择可能会挑到通过使回复陷入重复来降低概率的位置。这只是筛选：不生成文本时的概率变化并不保证模型写出的回复会改变。然后 `gne direction key-causal-verify --from NAME --unit U …` 会对所选候选在留出问题上生成，每个候选与一个比较用键并排，统计相对于仅干预时的行为数量；`--combo` 会将多个层一起写入作为一个候选检查。不会自动选择单一层。键标签页在“寻找有效层”下也有相同的两个阶段（docs/keyed-write.md §18）。

秩 1 条件编辑 `ΔW = α d kᵀ / (μᵀk)`（其中 `k = μ − QQᵀμ`，`Q` 是不希望写入位置输入所张成子空间的正交基）来自 nwn 的 Zenn 文章 [“Weight-Space Programming やってみた(やらせてみた)”](https://zenn.dev/nwn/articles/cf9fe78aa33861)（2026）。gne 从对照集构建键，沿 `gne direction` 的方向写入，并在留出示例上用打乱标签构建的对照键检查每个键。

### FiSH：用计算出的权重覆盖神经元（`nw`）

这里描述的方法称为 **FiSH**（Formula-instantiated Scene-gated Hidden units）：无需训练，用公式计算出的权重改写现有 MLP 神经元（隐藏单元），使其只在选定场景中写入方向。神经元会被改写，而不是新增。命令和单元模式仍使用普通名称 `nw`。

文章见 [FiSH：改写少数闲置神经元，让模型只在特定场景改变语气](https://balaena01.github.io/fish-steering/)，包含 4B 和 27B 模型上的测量、全部回复、阅读者的分类以及 LoRA 文件。

键控写入会加入线性编辑，因此只要键有少许反应，它就会写入少许；反应为负时则会反向写入。`nw` 单元则无需训练，直接覆盖一个平时几乎不工作的 MLP 神经元：门控行变为 `s·κ`，输入行变为键，输出列变为方向。由于门控会关闭，小反应会被压扁，负反应几乎不会写入。这是单键形式。另外还有两种形式。两个键可以相乘（一个键放在门控行，另一个放在输入行，因此神经元只在两者都有反应的位置写入，例如“这种场景”和“回复中的这个位置”）。也可以将两个神经元配对，使写入量封顶：这是带上限的配对，也是默认形式，下文所有 100 题结果都来自这种形式。在带上限的配对中，门控行放入一个把“这种场景 AND 回复的这个分段”合在一起的键，再分别减去一个在每个位置都约等于 1 的键的较小倍数（一个神经元减 a 倍，另一个减 b 倍）；输入行则放入这个约等于 1 的键。两个神经元之差在键反应小时几乎不写入，在反应大时则稳定在固定量：这是一个带阈值和上限的开关，由同时考察场景和回复位置的键打开。双键形式只在约 12 题的小规模检查中试过，没有用于 100 题评估。

```bash
gne direction --contrast C --positions response --same-reply --segments 0,1-3,4-    # 按回复位置分段的方向
gne nw try --direction NAME --contrast C --layers 11-23 --alpha 1,1,0.8   # 在写入权重前：把方向加到层输出并阅读回复
gne nw neurons --layers 11-23                          # 活跃度最低的神经元表：从中选择要覆盖的神经元
gne direction key --contrast C --target mlpin --layers 11-23 --positions pos:1-3 --name k_p1   # “此场景、回复位置 1～3”的键
gne nw key --contrast C --layers 11-23 --role scene    # 在这种场景的整段回复中持续反应的键
gne nw verify --unit 11:N2045:nw:a1@NAME.seg1#k_scene\&k_p1   # LoRA 是否按公式工作？
gne nw check  --recipe R --contrast C                  # 按位置和场景种类显示键响应与实际写入量，随后显示逐题数值
gne nw export --recipe R -o R-nw.lora.gguf             # 仅导出 LoRA（不烧录）；含 proj/keyed 单元时，gne headproj export 会把它们放进同一个文件
gne nw pipeline --contrast C --layers 11-23 --alpha 1,1,0.8 --save R.yaml   # 或一次运行全部构建步骤；--save 会写出上述命令可用的配方
```

`gne nw pipeline` 会输出简短摘要（LoRA 是否符合公式、单元和神经元数量、写入量、逐题数值数量、配方保存位置）；与 `gne nw verify` 一样，公式检查未通过时以 1 退出，且配方仍会写出。`--direction NAME` 会复用已有方向而不重建（分段或层不匹配时明确报错停止），`--cap 0.4-0.8,0.3-0.7,0.3-0.7` 按分段设置带上限配对的阈值，方式与 `--alpha` 设置强度相同（写一个值时应用于所有分段；默认仍为 0.4-0.8）。

**创建方向的快速读取（`--stack-fast`）。** 叠加层的方向创建（`gne direction … --stack-layers` 以及 `gne nw pipeline` 的方向阶段）会为每一层重新读取所有回复。默认读取只在与逐个读取结果完全相同的范围内合并读取（在 RTX 3080 上，13 层、40 个问题的重读在 Qwen3.5-4B 上从约 800 秒降到 138 秒；IQ2_XS 的 27B 模型没有变快，约需 40 分钟）。`gne direction --stack-fast` 和 `gne nw pipeline --stack-fast` 改为每个分段合并为一次读取（4B 约 81 秒，27B 约 3 分半）。但方向会略有变化：与默认读取的方向 cos 不低于 0.9996，方向长度差在 4B 上最多 1.2%，在 27B 上最多 2.8%。**比较层与强度的探索请用快速读取；制作要分发的成品时请用默认读取。** 尚未测量快速读取方向下的语气与正确答案（在 4B 的一个示例中，选出的神经元相同，64 个 greedy 回复中有 26 个在某处不同）。默认关闭。方向文件会记录使用了哪种读取方式，`gne nw pipeline` 的摘要会显示“使用快速读取创建的方向”（用 `--direction NAME` 复用方向时也会显示）；与 `--direction` 一起传入的 `--stack-fast` 不会生效，并会给出提示。UI 中对应 FiSH 标签页的“快速读取”复选框，MCP 和作业中为 `stack_fast`，模板 JSON 中为 `fish.stack_fast`。

**先检查每个问题；若有未写入的问题，就增加建键问题。** 当回复有多种开头方式（例如以引号开头）时，问题太少而构建的键会漏掉其中一些开头，这些开头之后不会写入，回复保持原样。平均值表无法显示这一点：完整写入的问题和完全未写入的问题会平均成“略低”。因此 `gne nw check` 还会按每个回复分段显示：几乎未写入的问题数（写入量低于 0.4，1 为目标量）和超过目标量的问题数（合计高于 1.2），并列出最差问题及每条回复的开头。逐题记录在 `--json` 和结果文件中。若这些几乎未写入的问题共享某种回复开头方式，请增加问题并重建键（`--direction` 会保留方向）。只有在存在这种遗漏时，增加问题才有帮助。在 27B 模型的傲娇示例中，回复以引号开头后键没有写入任何内容；将目标场景问题从 40 增加到 120 后，在键从未见过的 100 个问题上，符合人格的回复从 52% 升至 77%（使用 system 提示词时为 91%），在与建键问题同时编写的 100 个新问题上从 46% 升至 82%（使用 system 提示词时为 99%）。不希望改变的场景没有变化，陷入重复或读不通的回复从 4 条增加到 10 条。在键本来就会写入的位置，增加问题没有帮助：Qwen3.5-4B 上为 51 → 56 和 48 → 56（不显著），27B 模型的第二个人格为 35% → 33%。在第二种情况中只增加了目标场景问题，键开始在不希望改变的场景中少量写入（位置 1-3 的平均写入量从 0.04 变为 0.26，未观察到语气或正确答案受损），因此建议这些场景也增加问题；该组合尚未测量。一开始就准备 100 个以上目标场景问题可降低遗漏概率。详情见 [docs/fish.md](docs/fish.md) §3。`gne nw try` 在设置不同时会把每次结果保存到各自文件；可用 `--name` 指定文件名。`gne nw export --strip-meta` 会去掉 gne 的记录以及任何指明原始模型的内容（名称字段变为 `adapter`，或使用你传给 `--name` 的值）。

单元形如 `L:Nn:nw:aα@DIR#KEY`（1 个键）、`…#GATEKEY&INPUTKEY`（2 个键）和 `L:Nn+Nm:nw:aα@DIR#KEY&ONEKEY`（带上限的配对）。它们与其他单元一样可用于配方、`gne steer`、`gne behave`、聊天以及兼容 OpenAI 的 API。这里只支持带门控的 dense MLP（SiLU 门控 × 输入，无偏置：llama、qwen2、qwen3、qwen35）；其他形状和 MoE 层会给出明确错误。不提供烧录：在 Q8_0 中替换 `ffn_down` 的一列会损失超过九成的修改。

在 UI 中，direction 视图的“FiSH”标签页可在浏览器中完成整套流程：从 `configs/fish/` 的六个模板之一开始（改变某个场景中的语气、在所有场景中改变语气、改变句尾、同时指定场景和句尾、只写入两个人格之间的差异、在 CHiPS 配方之上构建；每个模板的说明都会介绍在 Qwen3.5-4B 上试过一次后发生了什么，包括没有奏效的部分），输入要覆盖的层（可以写成 `11-23` 这样的范围；层号因模型而异）和每个回复位置分段的强度（例如 1 / 1 / 0.8），然后按“全部构建”，把方向 → 神经元表 → 键 → 单元 → 在应用单元状态下重建键 → 公式验证 → 写入量作为一次作业运行。标签页会为每一步显示表格（默认神经元候选已勾选，可通过复选框更改；目标场景中过小或在其他位置过大的写入量会着色，其上方还有逐题表，统计几乎未写入的问题并列出问题及回复开头），最后一节只应用单元生成回复供阅读，“加入配方”会把单元交给配方标签页，用于聊天、保存和导出 LoRA。每一步也有独立按钮。“写入权重前试运行”（`gne nw try`）会把同一方向直接加到层输出，而且只加到目标场景问题的回复中，因此可以在构建任何神经元或键之前，通过阅读回复及其机械统计（重复、空回复、正则匹配）来比较层和分段强度。层与强度从不会自动选择。

`gne direction --same-reply` 是创建任意方向的新方式：两侧读取同一回复（A 侧的回复），因此方向只包含两种条件之间的差异，不包含读取文字不同带来的差异。默认行为不变。

目前的认识大多来自一个模型和两个示例（Qwen3.5-4B：句尾和人格），每个条件阅读 12 条回复后判断（只有下文最后的人格结果使用每套 100 个问题；人格示例还在 27B 模型上检查过一次），因此应视为早期结果。在句尾示例中，`nw` 单元在不希望写入位置的写入量约为键控写入的五分之一，且从未反向，但回复几乎没有差别。在人格示例中，旧式方向有一半是“读取文字不同”的差异，反复写入这部分正是键控写入陷入循环的原因；使用 `--same-reply` 后，12 条回复中陷入循环的从 9 条降到 0～1 条。改变语气需要在整个回复中写入（只写最初几个 token 不够，即使使用精确的内部状态）。随后使用计算生成的神经元在整个回复中写入按位置区分的方向，并在未用于构建的 100 个感谢与赞美问题集上测量，阅读回复时不知道它来自哪种条件。

| 形式（带上限的配对，无 system 提示词，仅 LoRA，greedy） | 符合人格的回复 | 陷入重复的回复（循环，机械计数） |
|---|---|---|
| v0.3.0 示例：三个分离层（11、18、23），所有位置强度 0.9，18 个神经元 | 27～29% | 6～10% |
| **13 个连续层（11～23），位置 0 / 1～3 / 4 及之后的强度为 1 / 1 / 0.8，78 个神经元** | **53% 和 53%**（两套独立问题集） | 8% 和 1% |
| 相同方案，但将键替换为比较用键（用打乱标签构建） | 3% | 81% |
| system 提示词 | 78～87% | 3～4% |
| 无干预 | 0% | 0～1% |

- 53% 在两套未用于选择设置的独立 100 题集上复现。采样时为 56～69%，同一批问题在不同随机种子之间会变化 13 个百分点。system 提示词仍明显领先。
- **其他场景不受影响，这是目前最稳固的部分。** 在计算与事实（50 题）、普通问题（50 题）和英语问题（30 题）中，正确数无法与无干预区分（greedy：46 题正确 → 46～47 题；采样：46～47 题 → 45～49 题）。语气也没有改变：在一套盲读问题集中，130 条回复没有一条符合人格；在另一套中，模型自身评分在 390 条里找出 1 条（无干预时也是 390 条里 1 条）。system 提示词几乎改变了所有回复，正确数在 greedy 下从 46 降到 38～41（采样时为 33～41）。
- **更多连续层优于少量层**（4B 模型上的一个示例）。收窄到无生成测量中排名最高的少量层，反而不如三个分离层。
- 把同样的向量直接加到层输出也得到相同水平，因此放进神经元并没有损失；限制在于所写入的方向本身。
- **陷入重复的回复**（回复无法脱离同一语句）因问题而异：每套问题集为 1～8%（三套共 300 题中有 16 题），在外貌或物品的赞美上更常见。没有任何设置能稳定保持在 5% 或以下。这些是机械计数（最后两套问题集分别为 8% 和 1%）；阅读者判为“重复或读不通”的回复更多（11% 和 3%）。另外，5～23% 的回复**没有贴合提问**（可读，但没有回答对方说的话）：greedy 下为 5% 和 11%，采样下为 15～23%。
- 没有帮助的做法：更细的位置分段、按问题类型分别创建方向、逐问题差异、减弱后部位置或开头、移动层范围的端点、填补分段边界处的下凹（会产生更多重复）、提高上限阈值。
- 层越多，构建越慢（4B 模型上 13 层、40 个问题，方向约 3 分钟，键和验证约 2～3 分钟；直到 v0.3.3，仅方向就需要约 14 分钟）。
- 最初在 12 个问题上的小规模检查看起来比 100 题结果更好；不要根据 12 个问题判断质量。
- **第二个模型检查过一次**（Swift-1.5-Qwen3.8-27B，IQ2_XS；从 22 到 46 每隔一层取一层，共 13 层，强度 1.2 / 1.2 / 0.96，78 个神经元；100 道新问题）：greedy 下 55% 的回复符合人格，采样下为 57%（无干预为 0%，system 提示词为 93%），陷入重复的分别为 3% 和 1%。不希望写入的 80 道场景题保持了原有语气，其中 70 条回复的开头与无干预完全相同。比较用键给出了更多符合人格的回复（70%），但也有 17% 重复或读不通的回复和 13% 没有贴合提问的回复（真实键为 4% 和 6%），并且改变了其他场景的措辞（80 条回复中只有 29 条开头相同）。在这个模型上，键的价值体现在精确选择场景，而不是提高数量。预先固定的五项标准中有三项通过；与比较用键的对比，以及正确答案少一题（30 题中 29 → 28）未通过。详情见 [docs/fish.md](docs/fish.md) §8。
- **在同一个 27B 模型上，又检查了第二个人格一次**（只对烦恼和征求建议类问题使用开朗、随意的语气；层和强度沿用第一个人格；100 道新问题）：greedy 下 36% 的回复开朗随意，采样下为 39%（无干预为 0%，system 提示词为 97%），另有 35% 的回复略显随意。不希望写入的 80 道场景题保持了原有语气，40 题中的正确数从 36 变为 37，且没有回复陷入重复。预先固定的六项标准中有五项通过；未通过的是强度（36 是 system 提示词的 0.37 倍，而标准是一半）。详情见 [docs/fish.md](docs/fish.md) §8。
- **增加建键问题（目标场景问题从 40 增加到 120，重建键，其余不变），在另一时间编写的问题上测量**：27B 模型傲娇示例中，greedy 下符合人格的回复从 52% 升至 77%，采样下从 57% 升至 80%（system 提示词为 91%），因为键现在会在回复以引号开头后立即写入（46/46 条此类回复，之前为 0/47）。不希望改变的 80 个场景问题没有变化（40 题中正确答案 38 → 38），陷入重复或读不通的回复从 4 条增加到 10 条。第二个人格没有改善（35% → 33%），Qwen3.5-4B 也没有显著改善（51 → 56 和 48 → 56）。使用 `gne eval` 时，这些配方在能力任务上都没有损失正确答案，困惑度上升 0.2% 到 0.9%（27B 傲娇示例为 0.4% → 0.9%）；仅关闭同样的 78 个神经元而不写入任何内容时，变化为 −0.2% 和 +0.03%。详情见 [docs/fish.md](docs/fish.md) §3 和 §8。
- **场景区分较粗（2026-11-13 验证；订正“其他场景不受影响”的说法）。** 此前测量的“不希望改变的场景”都是问题。对于不是问题的发言（近况汇报、邀请），傲娇配方写入的量几乎与收到感谢时相同：40 条发言中有 38～40 条发生写入，40 条回复中有 7～18 条变成傲娇语气，回复措辞几乎全部改变。对问题（计算与事实、做法、英语）的回复没有改变。键响应的是“非问题发言”，而不是感谢本身；增加示例或提高阈值都无法区分。不希望改变的场景中务必加入与目标场景形式相同的发言，并用 `gne nw check` 验证。详情见 [docs/fish.md](docs/fish.md) §8。
- **目前还做不到的事。** FiSH 写入的是每个回复分段一条固定方向，它承载的是在每条回复中以相同方式出现的语气。措辞取决于对方说了什么的人格（病娇：对对方和谁外出产生嫉妒等）没有出现：用通常的方向创建方式，40 个问题中有 0～2 个回复符合人格（system 提示词为 36～38 个），尝试五种方向创建方式后最好也只有约一半，且每四条回复中有一条没有贴合提问。详情见 [docs/fish.md](docs/fish.md) §8。
- 限制：大部分测量来自一种人格、一个场景和 4B 模型，且只有一种读法；第二个模型和第二种人格都只测过一次。这是我们的判断，并非实测事实：在 4B 模型上，这种创建方向的方式可能已接近其上限。

### 实际使用测试：按真实用法检验文件

`gne eval` 是快速检查：它在单个文件上缩放配方，并询问短问题。`gne fieldtest` 检验模型文件在真实使用中是否可靠，并在文件之间进行比较。它会启动开启思考和采样的 `llama-server`，让模型执行长任务和工具调用任务，并报告答案达到 token 上限的频率、陷入短周期或长周期循环的频率、因推理始终未完成而导致正文为空的频率、是否在需要时以完整参数调用正确的工具、是否在无需调用时不调用工具，以及模型能否在工具结果之后继续。perplexity 由 gne 重新读取相同参考文本计算，因此可直接测量烧录后的文件，而 `gne eval` 无法做到这一点。

```bash
gne fieldtest -m baked.gguf --against original.gguf --llama-server <llama-server 的路径>
gne fieldtest -m original.gguf --lora adapter.lora.gguf                # 有/无 LoRA 对比
gne fieldtest -m baked.gguf --against original.gguf --prompts mine=questions.json --match "some regex"
gne fieldtest -m baked.gguf --against original.gguf --bench gsm8k,mmlu_pro,math500,ifeval   # 直接测量该文件的能力
```

任务位于 `configs/fieldtest/`，可通过 `--long-tasks` 和 `--tool-tasks` 替换。`--prompts name=file` 添加自定义问题，测量循环、正文为空和达到上限被截断的情况；`--match` 增加匹配率。`--bench gsm8k,mmlu_pro,math500,ifeval` 直接在文件本身上测量能力，用固定种子抽取 250 道 GSM8K、300 道 MMLU-Pro、250 道 MATH-500 和 250 道 IFEval 题（`--bench-n` 可更改题数；MATH-500 总共有 500 道，IFEval 总共有 541 道）。MATH-500 用 math-verify 对回复中最后一个 `\\boxed{}` 评分，IFEval 使用官方检查器（已复制到 `gne/third_party/ifeval`，Apache-2.0），只有遵守全部指令的回复才算正确。这两项需要 `pip install 'gguf-neuron-explorer[bench]'`。对于容易循环的模型，请提高 `--bench-tokens`（默认 4096），以便长回复能够完成；如果模型卡有要求，请设置 `--presence-penalty`。数据首次从 Hugging Face 获取，保存在 `<root>/fieldtest/data`，不进入本仓库。两个文件回答相同问题，因此除准确率外，表格还会显示有多少题只有其中一个答对；只有当这种差异显著时（精确双侧 McNemar 检验，p < 0.05），准确率才计为恶化。文件会逐个加载到 GPU。最多相差 2 题，或 perplexity 相差 1%，视为同一水平，因为仅靠采样就可能造成一道题的差异。结果写入 `<root>/fieldtest/<name>.json`；生成的文本写入单独的 `<name>.replies.json`，不会打印。在 27B 模型上（RTX 3080，12 GB），额外加入 200 道问题后，每个文件耗时 30～45 分钟。`gne` 不构建 `llama-server`，因此请通过 `--llama-server` 或 `GNE_LLAMA_SERVER` 传入，或将其放在 `PATH` 中。在 UI 中，这是 eval 视图的“实际使用测试”部分；运行期间，已加载的模型会离开 GPU，聊天等操作必须等待。

### 跨量化

在一种量化上找到的神经元可以迁移到同一模型的其他量化。使用 L22 N2742（“alien”神经元，在 Qwen3.5-4B Q8_0 上找到）检查：

| | Q4_K_M | Q3_K_M | UD-IQ2_M |
|---|---|---|---|
| W_down 列与 Q8_0 的 cos | 0.998 | 0.998 | 0.974 |
| 相同索引 = 同一神经元（抽样 301 个） | 全部 | 全部 | 全部 |
| logit-lens 头部词 | 相同 | 相同 | 几乎相同 |
| L22 的残差范数（Q8_0：23.69） | 23.65 | 23.30 | 23.76 |
| “Who are you?” +0.8 | “I am Extracillary, an advanced extraterrestrial intelligence…” | “I am an extraterrestrial intelligence from the planet of Aliens.” | “I am an extraterrestrial being.” |
| 「あなたは誰ですか？」 +1.0 | 「私は、地球外生命体です。」 | 「私は、地球外生命体です。」 | 失效 |

- 在 Q8_0 上导出的 control vector 可以直接用于其他量化。在 Q4_K_M 和 Q3_K_M 上，它生成的文本与在该量化上重新推导向量完全相同。
- 人格保持不变；只有虚构细节会变化（Q8_0 说“Extrali … Xeno-888”）。
- 在 2 bit 下，可用剂量范围会缩小：+1.0 已经会破坏日语回答。
- Q3_K_M 将该层的 `ffn_down` 存为 Q4_K，因此其列与 Q4_K_M 匹配。

### 显卡放不下的模型：Modal

`gne modal` 会在 [Modal](https://modal.com/) GPU 上运行任意 `gne` 命令。权重直接进入 Modal Volume，因此不会落到你的磁盘上，并且只在命令运行期间计费。

```bash
gne modal fetch unsloth/Qwen3.8-Flash-Next-GGUF --include '*UD-IQ1_S*'
gne modal put spec.json
gne modal run -- heads pipeline --spec /data/inputs/spec.json -m /models/.../....gguf --yes
gne modal get gne/<model_id> ./results
```

`gne modal hf-upload` 使用存储在 Modal secret 中的 token，将文件从 Volume 直接推送到 Hugging Face。`gne modal ps`、`gne modal stop --all` 和 `gne modal cost` 可显示正在运行的内容、停止它们，并显示本月账单。只会上传 `gne/` 和 `configs/` 中由 git 跟踪的文件，以及你在 `gne modal put` 中指定的内容。参见 [docs/modal.md](docs/modal.md)（日语）。这尚未在真实权重上运行过。

## Web UI

```bash
gne serve -m models/Qwen3.5-4B-Q8_0.gguf      # http://127.0.0.1:8791
```

在启动服务器的终端中按 Ctrl+C 停止。它会关闭已打开的流和正在进行的生成并立即停止，即使浏览器仍开着 UI；若仍有连接残留，再按一次 Ctrl+C 会强制切断。页面文件每次加载都会重新读取，但服务器会继续运行启动时的 Python 代码，因此更新 gne 后需要重启；当服务器代码比磁盘上的代码旧时，页面顶部会显示提示条。

![前视图中带上下游连线的神经元面板](docs/images/ui-neuron.png)

主要视图：
- **map：** 每一层作为一张神经元图板，并带有叠加层。
- **layers：** 每层摘要。
- **direction：** 查看已保存的方向在每一层多大程度分开对照集两侧、基准层在哪里，以及所选层的逐示例投影。第二个标签页接收 JSON 形式的对照集（粘贴、从文件加载，或从五种用途模板开始），用一个按钮运行完整流程：创建方向、选择基准层、筛选层和注意力头、分配强度。结果会展示每一步的简短说明、被排除的注意力头及原因、按效率与按停头所得的排名差异，以及头数更少的组；每组都可加载到配方标签页或送去 eval。比较表下方生成的强度候选会出现在第三个标签页“强度候选”，可并排查看回复、匿名评分并汇总。第四个标签页“keys（keyed）”从文件或 JSON 接收对照集（可从三个用途模板开始），用一个按钮构建方向、为键控写入构建键，并在留出示例上对照用打乱标签构建的对照键检查该键。结果中的键控单元可加入配方，每一层都可按公式检查（key-verify）；在 MoE 模型上，还会用每个专家被选中频率的表格在某个专家上设置键。其第一部分会收集 CHiPS 配方仍让行为出现的问题：应用配方运行 behave（作为作业，带进度和停止按钮），根据结果构建键控对照集并显示红色和黄色警告，然后填入输入框，并已自动设置方向和配方。第五个标签页“FiSH”从头到尾构建计算生成的神经元：对照集（来自模板）、按回复位置分段的方向、活跃度最低的神经元表、键、单元，以及两项验证（公式，以及按场景和回复位置测量的写入量）。
- **eval：** 将已保存的 eval 显示为每个剂量的图表和表格，以及示例输出和用于 eval 当前配方的按钮。方式比较和头数更少的组比较也会显示在这里，带键控单元的配方是否只在应该写入的位置写入（选择性）也会显示。
- **embed：** rollout 文本的 UMAP。

点击神经元或列表行可打开其面板，面板有四个标签页：
- **Examples：** 每个提示词和剂量的 rollout。
- **Chat：** 实时 3 列引导，带有 `when` 选择器和 system 提示词输入框。上方的切换可改为 1 列，只生成被引导的回复（已应用配方，或用此神经元的 + 推动后的回复）；生成速度快 3 倍，适合尝试长对话，更改列数会清空对话。配方标签页可以添加消融（方向、倍率、层）。对于支持音频输入的模型，🎤 按钮会在浏览器中录制你的声音，并将其作为用户回合发送（按下、说话、再次按下；输入框中的文字也会一起发送）。🔊 会用浏览器语音朗读右列回复。通过 `http://127.0.0.1` 或 `localhost` 打开 UI 时，麦克风可用。
- **Details：** lens 结果、每个剂量的指标、连接，以及包含该神经元的 eval。
- **Web：** 多跳连线。

点击图板上边的刻痕会打开注意力头面板。除了注意力模式和对照结果外，Causal 标签页会显示停掉或强化注意力头时沿方向的投影改变了多少，地图也可按这些值着色。配方标签页可放入这类注意力头单元，并带有强度滑块。

要计算缺失数据（lens、sweep、连接、布局），请右键点击层标签。任务在后台运行，当聊天需要 GPU 时会让出 GPU。语言切换位于右上角，可在日语 / 英语 / 中文之间选择；也可以用 `?lang=ja`、`?lang=en` 或 `?lang=zh` 打开 UI。

服务器仅绑定 127.0.0.1，并会拒绝携带外来 Host 头的请求和跨源写入。

### OpenAI 兼容端点

`gne serve --openai` 在同一服务器上添加 `/v1/models` 和 `/v1/chat/completions`（支持流式与非流式），使其他应用可以使用带 `when` 或 gain 单元的配方。在 `gne` 字段中传入配方（OpenAI SDK 中为 `extra_body`）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8791/v1", api_key="unused")
r = client.chat.completions.create(
    model="any", messages=[{"role": "user", "content": "Who are you?"}],
    extra_body={"gne": {"units": ["22:2742:+0.8"], "when": "gen"}})
```

对于支持音频输入的模型（LFM3-Omni，其 `mmproj-*.gguf` 放在模型旁或通过 `--mmproj` 指定），用户消息可以携带语音，使用带 base64 `data` 的 `input_audio` 部件（wav、mp3 或 flac）。端点不会读取本地路径。模型卡要求使用 system 提示词 `Respond in text.` 获得文本回复，或使用 `Perform ASR in English.` 获得转录：

```python
import base64
audio = base64.b64encode(open("speech.wav", "rb").read()).decode()
r = client.chat.completions.create(
    model="any", messages=[{"role": "system", "content": "Respond in text."},
                           {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}]}],
    extra_body={"gne": {"recipe": "my_recipe"}})
```

在命令行中，`gne steer --audio speech.wav --system "Respond in text." --recipe ...` 效果相同。

`"recipe"` 不用 `units`，而是接收 `~/.gne/recipes` 下保存的配方名称；`"scale"` 会乘以配方的剂量。除非请求将 `temperature` 设为大于 0，或设置 `top_p`、`top_k`、`min_p`、`presence_penalty`、`seed` 中的任意一项，或者配方带有 `sampling`，否则生成使用 greedy；之后会按 llama.cpp 的顺序采样。未提供的值来自配方的 `sampling` 或 llama-server 的默认值。不支持 tools 和 logprobs。请求会与 UI 聊天共享 GPU。

服务器不保存状态，因此强度可以在对话中的任何时候变化：随消息发送 `"gne": {"recipe": "yandere_qwen38_27b", "scale": 0.7, "when": "ramp:0.5-1.5"}`，并可随时修改 `scale` 和 `when`。[`examples/strength/strength_chat.py`](examples/strength/strength_chat.py) 维护一个 0 到 100 的强度，可在聊天中输入 `/s 70` 修改，或由另一个程序通过文件修改（例如游戏事件、直播中的计量条）。它在每条消息前使用 [docs/verification.md](docs/verification.md)（F94、F96）中的测量结果将强度映射到 `when` 和 `scale`。每次请求都会重新读取整段对话，因此新的强度也会改变之前回合的读取方式（使用 `all` 和 `assistant` 时），而长历史会使每条回复开始得更晚。

## 从 Agent 使用（MCP）

```bash
claude mcp add gne -- gne mcp --model /path/to/model.gguf
```

可用工具包括 `model_overview`、`layer_summary`、`neuron`、`search`、`steer`、`steer_lens`、`fires`、`reverse`、`recipe_save`、`export_cvec`、`gain_export`、`bake`、`eval`、`direction`、`ablate_export`、`head`、`heads_contrast`、`attention`、`direction_score`、`heads_causal`、`heads_allocate`、`headproj_export`、`direction_key`、`direction_key_check`、`direction_key_experts`、`direction_key_verify`、`direction_key_pipeline`、`direction_key_from_behave`、`direction_key_causal`、`direction_key_causal_verify`、`behave`、`keys`、`keyed_export`、`set_name`，以及 `job_start` / `job_status` / `job_cancel`。

如果 Web UI 已打开，请添加 `--ui-url http://127.0.0.1:8791`。这样 agent 就可以高亮、注释神经元并在浏览器中打开它们（`ui_highlight`、`ui_note`、`ui_open`）。当 UI 已占用同一 GPU 上的模型时，还请添加 `--n-gpu-layers 0`。

[docs/agent_playbook.md](docs/agent_playbook.md) 说明了 agent 应如何判断结果：与对照比较，不要轻信模型仅改了自称等表面效果，并将事实与解释分开。

## 测试

```bash
uv pip install --python .venv/bin/python -e '.[dev,ui,mcp]'
python -m playwright install chromium
python -m pytest -q
```

- 模型测试使用 `GNE_TEST_MODEL`，默认值为 `models/Qwen3-0.6B-Q8_0.gguf`。
- 浏览器测试使用 Playwright Chromium；`GNE_CHROME` 可设置其路径。
- MoE 测试使用 `GNE_TEST_MOE`（默认 `models/granite-3.1-1b-a400m-instruct-Q8_0.gguf`）；路由器偏置测试使用 `GNE_TEST_MOE_BIAS`（默认 `models/LFM2-8B-A1B-Q4_K_M.gguf`）。
- 真实音频测试需要 `GNE_TEST_OMNI`（一个 LFM3-Omni GGUF 及其旁边的 mmproj）和 `GNE_TEST_AUDIO`（语音 wav）。其他音频测试使用默认模型。
- 键控写入测试还会使用 `GNE_TEST_MOE` 测试专家键，并且只有设置以下变量时才运行另外两部分：`GNE_TEST_MOE_SHARED`（带共享专家的 MoE，例如 Qwen1.5-MoE-A2.7B）和 `GNE_TEST_HYBRID_MODEL`（带线性注意力的模型，例如 Qwen3.5-0.8B）。
- 缺少模型或浏览器的测试会被跳过。

## 文档

文档为日语：
- [docs/fish.md](docs/fish.md)：FiSH（计算生成的神经元，`nw`）：公式、工作流程和限制。
- [docs/verification.md](docs/verification.md)：哪些内容已与 llama.cpp 验证，以及发现。
- [docs/measurements.md](docs/measurements.md)：耗时与吞吐量。
- [docs/agent_playbook.md](docs/agent_playbook.md)：如何解读结果。
- [docs/setup.md](docs/setup.md)：WSL 下的 CUDA / ROCm、每个 GPU 一个环境，以及 worker。
- [CHiPS：知道与回答之间的少数注意力头](https://balaena01.github.io/chips-steering/)：文章，包含测量结果与数据。
- [FiSH：改写少数闲置神经元，让模型只在特定场景改变语气](https://balaena01.github.io/fish-steering/)：文章，包含测量结果与数据。
- [docs/qwen4exp.md](docs/qwen4exp.md)：Qwen3.8-Flash-Next（`qwen4exp`）：分割 GGUF 文件、四条残差流，以及在不使用 control vector 时哪些功能可用。
- [docs/head-component-steering.md](docs/head-component-steering.md)：CHiPS，只改变少数由因果选出的注意力头中沿方向的分量（`mode: proj`、`gne direction score`、`gne heads causal`、`gne heads allocate`、`gne eval --compare-methods`），以及它与全层消融的关系。
- [docs/keyed-write.md](docs/keyed-write.md)：键控写入。只有当从对照集构建的键对写入残差流的权重矩阵输入产生反应时，才进行秩 1 写入（`mode: keyed`、`gne direction key`、`gne direction key-check`），包含句子结尾的完整示例、它与 CHiPS 的关系，以及如何寻找键有效的位置（`gne direction key-causal`）。
- [docs/persona_workflow.md](docs/persona_workflow.md)：从神经元逐步构建人格（使用 `examples/persona_neurons/` 中的脚本，并以病娇案例作为完整示例），并从示例对话中提取注意力头与神经元。

## 许可证

MIT——见 [LICENSE](LICENSE)。

随附的第三方文件：
- three.js（MIT），位于 `gne/ui/static/vendor/three/`。
- 默认统计语料（`configs/corpus/default.txt`），由 Project Gutenberg 和青空文库的公有领域文本组成。详情见 [configs/corpus/README.md](configs/corpus/README.md)。

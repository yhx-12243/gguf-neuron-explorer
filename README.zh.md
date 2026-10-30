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
- **Gain 引导。** `22:2742:x4` 将神经元自身的输出乘以系数，而非添加固定方向：`x0` 移除它，负系数反转它。它只在神经元原本已经激活的位置生效。可导出为供 llama.cpp 的 `--lora` 使用的 LoRA 适配器 GGUF（同样适用于 Qwen），或导出为改写后的 GGUF；默认改写使用缩放补丁，保留原始量化并精确缩放对应行。
- **方向（`gne direction`）。** 当某个人格或行为分散在许多神经元上时，改为从对比集构建方向（逐层在最后一个提示词 token 处的平均差）。它可以通过消除方向并检查行为是否下降，或添加方向并检查行为是否上升，自动选择层。像神经元一样用 `30:dir:yandere:+0.3` 添加方向；方向会导出为 control vector。要添加人格，请用 `--positions response` 构建方向，它会在每个人格生成的回复上测量，而不是在指令后立即测量。与神经元一样，方向也会对照随机控制进行检查：每个候选层还会用 16 个相同范数的随机方向推动，只有当效果超过其 95 百分位时才会选择该层。`gne steer --contrast` 对整个配方做同样的事，并报告 `null_pct`。
- **方向消融（`gne ablate`）。** 从所有写入残差流的权重中投影消除一个方向（也称为 abliteration）。可导出为 LoRA 适配器，或导出为改写后的 GGUF：Q8_0/F16，或用 `llama-quantize` 重新量化回原始类型。配方可以带有 `ablate:`，而 `gne eval --compare-ablation` 会将其与将排名靠前的神经元置零进行比较。
- **副作用检查（`gne eval`）。** 在多个剂量下运行配方，并将人格变化与破坏情况放在一起查看：精确答案任务和格式任务、参考文本上的困惑度，以及语言混杂（日语回答中出现仅简体中文、谚文及其他文字）。可选的第二个 GGUF 可以给人格按 1–5 打分。保持在阈值内的剂量会标记为可发布。`--random-controls N` 还会在每个剂量下运行 N 个相同规模的随机方向配方。这样可以显示人格变化是否超过这些对照，以及仅推动本身会损失多少能力和困惑度。
- **谁在写入某种状态（`gne writers`）。** 激活差异显示哪些神经元对某种状态（例如人格 system 提示词）产生反应。`gne writers` 测量每个神经元在回复 token 上实际写入残差的量：激活差异 × 写入方向，包括 post-FFN norm。它还会将该状态的方向拆分为由嵌入、每层 mixer（attention、卷积）和每层 MLP 分别写入的部分。UI 提供了 writers 视图和对应的地图叠加层。
- **注意力头（`gne heads`）。** 注意力头与神经元一样有地址（`L21 H7`，包括 Qwen3.5 的线性注意力头）。`gne heads contrast` 在有无人格的情况下运行相同对话，并按以下标准对头排序：
  - 它们对触发词（例如其他人的名字）的注意力增加了多少；
  - 它们的注意力模式和输出变化了多少；
  - 它们的输出写入人格方向的程度。

  噪声下限是将两个中立 system 提示词相互比较。`21:H7:x0` / `x2` 通过其 `W_O` 列块上的 LoRA 缩放一个头的输出。`gne eval --reactivity` 测量人格是否只在触发词存在时出现，并与相同系数的随机头对照。

  `gne heads listen`（头面板的 Listen 标签页）运行一段对话，并报告所选的注意力头（例如 `14:H0-3`）中，回复 token 对每个更早 token 分配了多少注意力。给定已保存的方向后，它还会沿该方向将头的输出归因到来源 token。每个 token 的 value 向量按该 token 的注意力权重和该头的门控加权，然后通过 `W_O`，并在具有该 norm 的架构中通过 attention 输出后的 norm。按层内所有头求和，这些贡献可重现该层的 attention 输出。命令会报告剩余误差；在 Qwen3、Qwen3.5 和 LFM3 上均低于 1%。
- **混合专家（Mixture of Experts）。** 在 MoE 层中，神经元写作 `L12 E37 N511`（专家 37 的神经元 511）；`gne experts contrast` 会按人格开启时每个专家被更多选中、以及更多写入人格方向的程度对专家排序；`gne experts sweep` 逐个消融或增强专家。在 MoE 层中，`add` 只会在每个 token 上推动神经元的写入方向；真正的神经元推动是 gain（`12:E37:N511:x2`，仅在选中该专家时生效）。整个专家（`12:E37:x0`）和路由器偏置（`12:E37:route:+2`）通过缩放补丁应用：工作副本中量化块的 scale 会被精确改写，并在使用后恢复。
- **Web UI。** 每层和每个神经元的 3D 地图，支持切换叠加层（sweep mass、steer KL、logit-lens |z|、激活值、搜索命中、粘贴文本时哪些神经元被激活）。它还包括：
  - 排名列表和神经元面板；
  - 实时 3 列聊天；
  - 神经元之间的连线与 UMAP 布局；
  - rollout 的嵌入散点图；
  - 共享 GPU 的后台任务；
  - JP / EN / ZH 切换。
- **多 GPU、大型模型。** `gne worker` 为每个 GPU 运行一个进程，可以来自不同的 Python 环境（CUDA、ROCm），并共享结果文件夹。`gne sweep --workers auto` 和 `gne lens steer --workers auto` 会把神经元拆分到这些进程。预计超过 30 分钟的任务会停止，除非传入 `--yes`。`--budget 30m` 会利用某个方向的上位写入器、上位 steer-lens 神经元和 quick 设计来缩小任务，使其符合预算。`gne info` 会显示 GPU/CPU 层分配，以及针对大型模型的“先做反向查找”工作流。
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
- MoE 模型支持专家神经元与专家级分析（experts contrast / sweep / route、专家整体 gain 和缩放补丁）；详见下文“混合专家”。
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
ablate: {direction: refusal, scale: 1.0}        # 从所有残差写入器中投影消除方向（LoRA）
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
| `12:E37:N511:+0.8` | 在每个 token 上添加该神经元的写入方向（方向推动，而非神经元推动） | control vector |
| `12:E37:N511:x2` | 选中专家 37 时，将该神经元的激活乘以 k | 按专家堆叠的 LoRA |
| `12:E37:x0` / `x2` | 移除专家 37 的输出或将其加倍 | 缩放补丁 |
| `12:E37:route:+2` | 移动专家 37 的路由器偏置（仅限带路由器偏置的架构，例如 `lfm2moe`） | 缩放补丁 |
| `12:S:N5:x2` | shared expert 神经元 | LoRA（与 dense 神经元相同） |

缩放补丁会改写受影响行量化块中的 fp16 scale，使一行或一个专家的 slab 精确变为原来的 k 倍（k = 0 时写入零字节）。它适用于 Q4_0 到 Q8_0、Q2_K 到 Q6_K、除 IQ1_M 外的 IQ 类型，以及 F16/BF16/F32。补丁写入 `~/.gne/work/` 下的工作副本（绝不会改动原始 GGUF），并在每次使用后恢复；重新加载副本对 1.4 GB 模型约需 0.3 秒，对 5 GB 模型约需 0.9 秒。`steer`、`eval`、UI 聊天和 OpenAI 兼容端点会自动应用这些补丁。列方向（`W_down[:, n]`）无法用这种方式打补丁。在 MoE 层内，sweep、lens 和反向查找按 `e × n_ff_exp + n` 为神经元编号。

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

## Web UI

```bash
gne serve -m models/Qwen3.5-4B-Q8_0.gguf      # http://127.0.0.1:8791
```

![前视图中带上下游连线的神经元面板](docs/images/ui-neuron.png)

UI 有四个视图：
- **map：** 每一层作为一张神经元图板，并带有叠加层。
- **layers：** 每层摘要。
- **eval：** 将已保存的 eval 显示为图表，以及每个剂量的表格、示例输出和用于评估当前配方的按钮。
- **embed：** rollout 文本的 UMAP。

点击神经元或列表行可打开其面板，面板有四个标签页：
- **Examples：** 每个提示词和剂量的 rollout。
- **Chat：** 实时 3 列引导，带有 `when` 选择器和 system 提示词输入框。配方标签页可以添加消融（方向、scale、层）。对于支持音频输入的模型，🎤 按钮会在浏览器中录制你的声音，并将其作为用户回合发送（按下、说话、再次按下；输入框中的文字也会一起发送）。🔊 会用浏览器语音朗读右列回复。通过 `http://127.0.0.1` 或 `localhost` 打开 UI 时，麦克风可用。
- **Details：** lens 结果、每个剂量的指标、连接，以及包含该神经元的 eval。
- **Web：** 多跳连线。

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

可用工具包括 `model_overview`、`layer_summary`、`neuron`、`search`、`steer`、`steer_lens`、`fires`、`reverse`、`recipe_save`、`export_cvec`、`gain_export`、`bake`、`eval`、`direction`、`ablate_export`、`head`、`heads_contrast`、`attention`、`set_name`，以及 `job_start` / `job_status` / `job_cancel`。

如果 Web UI 已打开，请添加 `--ui-url http://127.0.0.1:8791`。这样 agent 就可以高亮、注释神经元并在浏览器中打开它们（`ui_highlight`、`ui_note`、`ui_open`）。当 UI 已占用同一 GPU 上的模型时，还请添加 `--n-gpu-layers 0`。

[docs/agent_playbook.md](docs/agent_playbook.md) 说明了 agent 应如何判断结果：与对照比较，不要轻信改变自称等廉价效果，并将事实与解释分开。

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
- 缺少模型或浏览器的测试会被跳过。

## 文档

文档为日语：
- [docs/verification.md](docs/verification.md)：哪些内容已与 llama.cpp 验证，以及发现。
- [docs/measurements.md](docs/measurements.md)：耗时与吞吐量。
- [docs/agent_playbook.md](docs/agent_playbook.md)：如何解读结果。
- [docs/setup.md](docs/setup.md)：WSL 下的 CUDA / ROCm、每个 GPU 一个环境，以及 worker。
- [docs/persona_workflow.md](docs/persona_workflow.md)：从神经元逐步构建人格（使用 `examples/persona_neurons/` 中的脚本，并以病娇案例作为完整示例），并从示例对话中提取注意力头与神经元。

## 许可证

MIT——见 [LICENSE](LICENSE)。

随附的第三方文件：
- three.js（MIT），位于 `gne/ui/static/vendor/three/`。
- 默认统计语料（`configs/corpus/default.txt`），由 Project Gutenberg 和青空文库的公有领域文本组成。详情见 [configs/corpus/README.md](configs/corpus/README.md)。

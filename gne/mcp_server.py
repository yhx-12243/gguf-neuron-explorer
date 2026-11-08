"""MCP 服务器 `gne mcp`（stdio）。设计文档 §6.13 (d)。工具是 gne.api 的映射。

    claude mcp add gne -- gne mcp --model <path>

- 返回值是与 --json 相同形式的 dict（token 序列为字符串）。失败时为 {"error": {"code", "message", "hint"}}。
- 使用 GPU 的调用在 worker 线程中通过 GpuArbiter 执行。长时间处理用 job_start 提交，
  用 job_status 查看进度。任务会在单元边界把 GPU 让给交互调用。
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import anyio

from .jobs import GpuArbiter, JobRunner, standard_kinds
from .steer import GneError
from .store import CONFIGS

INSTRUCTIONS = """\
gguf-neuron-explorer：研究 llama.cpp 上 GGUF 模型的 MLP 神经元，推动（steer）它们，并将配方导出为 control vector。
流程（详见 docs/agent_playbook.md）：
1. 用 model_overview → layer_summary(L) 查看各层有哪些数据（ρ、lens、sweep、反向查找）。缺失的用 job_start 创建。
2. 通过 neuron(L, n) 的档案判断候选。务必查看 null_pct（sweep 高于对照群的多少比例）与 kl_null_p95（steer lens 对照的 95% 分位）。不要报告未超过对照群的项。
3. “自称、作者名改变”是随机方向也会产生的廉价效果。只有多个提示词出现同一主题时，才视为真实效果。
4. rel dose：0.5 为细微插入，0.8 可覆盖话题，1.0 以上进入崩溃区。第 0～2 层的 rel dose 不能直接套用。
5. 用 steer(units, compare=True) 确认与无干预的差异，再通过 recipe_save → eval(recipe) 测量副作用（能力、ppl、语言混入），最后 export_cvec。
   抑制类用途先试 gain 的 x0，而不是 add 的 −。位置（when）优先尝试 gen，并与 all 比较。
   若只能找到神经元层面的性格、行为片段，用 direction(contrast, select="auto", mode="add"|"ablate") 创建方向。
   要消除的行为，先比较 ablate（steer 的 ablate="NAME"）与上位神经元 x0 的 eval（gne eval --compare-ablation）再选择。
   方向也要检查是否超过对照（direction 的 null_pct、eval(random_controls=N) 的 novelty_null_pct）。不要报告 null_pct 低于 0.95 的方向。
   预计超过 30 分钟的任务会返回 NEEDS_CONFIRM 与估算后停止。可在 params 中通过 budget("30m") 限制，或设置 yes: true。
   大型模型（无法装入 GPU、全量 lens 每层超过 1 小时）从反向查找开始：reverse → 前 N 个 lens_steer → quick sweep → eval。
单元写法："L:N:dose"（例 "14:8949:+1.0"），gain 为 "L:N:x2"（输出放大 k 倍，x0 表示消除），随机方向为 "L:random:SEED:dose"，
用 direction 创建的方向为 "L:dir:NAME:dose"，attention 的注意力头 gain 为 "L:Hh:xk"（例 "21:H7:x0"），注意力头的方向分量投影为 "L:Hh:sS@NAME"。
7. 反应性（有/无触发词是否产生差异）用注意力头创建。先用 heads_contrast → 对前几名用 x0 / x2 运行 eval(reactivity=对照集) →
   只保留超过对照（相同 k 的随机注意力头）的项。
6. 如果有人正在查看 UI（gne serve），用 ui_open 打开正在调查的神经元，用 ui_highlight 高亮候选，用 ui_note 将判断要点
   贴到神经元上（仅在通过 --ui-url 连接时。未连接时会返回 UI_NOT_RUNNING，可忽略并继续）。
8. MoE（选择专家的模型）中，专家神经元写作 "L:Ee:Nn:dose"（例："12:E37:N511:+0.8"）。MoE 的 add 是“沿写入方向推动”
   （无论是否选中专家，都会加到所有 token 上）；若要推动神经元本身，请用 gain（"12:E37:N511:x2"，仅在选中专家时生效）。
   专家整体用 "L:Ee:x0"（消除）/ "L:Ee:x2"，路由器偏置用 "L:Ee:route:+2"（仅限带偏置的 arch）。
   先用 experts_contrast 筛选专家（usage_shift = 人格状态下被调用，contrib_cos = 写入方向），再用 gain 试其中的神经元。
9. 只改变注意力头的方向分量（CHANGE-06，"L:Hh:sS@NAME"，例 "33:H16:s6@evil"。s = 1 时该注意力头的方向分量消失，2 时反转，
   正交分量不变）。步骤：direction_score（确认基准层与分离度）→ heads_causal（以 mode="ablate" 初筛候选）→
   heads_allocate（按效率 η 分配强度，并比较减少注意力头数量后的组）→ eval(random_controls=N, compare_methods=True) → headproj_export。
   若只想完整跑一遍，用 heads_pipeline(spec)（从创建方向到分配。问题可以有数百个，停止测量的阶段会抽稀到 64 个）。
   最终确定强度：heads_candidates（按组 × 倍率生成，并创建隐藏名称的评分文件）→ 评分 → heads_candidates_tally。
   若某组仍留下无法消除的问题（分配表中的 remain），用 heads_refine 仅从剩余问题中寻找额外注意力头并加入组中。
   分发前用 fieldtest 确认（按真实用法比较烧录后的文件与原文件；检查 eval 看不到的长任务崩溃和工具调用问题）。
   - 使用方向的干预，应先用 direction_score 确认基准层与分离度（cohen_d）再继续。
   - heads_causal 的 ablate 与 marginal 排名不同。初次筛选用 ablate，重新分配用 marginal（由 heads_allocate 使用）。
   - 候选与强度方案用不生成文本的指标（direction_score、heads_causal、heads_allocate）来制定。方向上的投影值表示推动量，并非行为本身，因此最终强度要用少量问题生成来确认。
   - 报告主要采用外部可见指标（eval 的 rate = 对照集 validate.match 的一致率、能力、ppl）。只看方向上的投影值，
     只能说明“用方向测量了被方向推动的东西”。
   - 未经过随机对照（eval 的 random_controls，同数量、相同 s 的随机注意力头）与方式比较（compare_methods）的配方不要报告。
   - 当目标指标达到数 % 时就停止提高强度，转而寻找承担剩余示例的其他注意力头。
10. 带键写入（keyed、CHANGE-07、"L:ffn:keyed:aα@DIR#KEY"、例 "19:ffn:keyed:a1@zundamon_endings#zundamon_endings"）。
   proj 选择“改变哪个方向”，而 keyed 选择“在哪个状态写入”（在写入残差的权重矩阵上加上 α d κᵀ，
   仅在激活中键 κ 产生反应的位置写入方向 d。α 为 raw，1 = 直接使用观测到的 A − B 差）。两者可以叠加（进入同一个 LoRA）。
   步骤: direction(写入方向)→ direction_key(键。无干预创建。若将对照集的 positions 设为 "before:<正则表达式>"，
   则在与推理时相同上下文的模型自身回复中，要改变字符的紧前方位置即为写入位置)→ direction_key_check(用创建时未
   使用的样本。默认 subset="odd"。held_out 为 false 的值偏乐观。key-check 的选择性若不超过比较用键(打乱标签后用相同
   步骤创建的键)的 p95，则不要使用该键)→ steer / eval → keyed_export(分发时用 LoRA。烧录的默认是，原为
   Q8_0 的张量保持 Q8_0，其余为 f16)。when 对 keyed 无效。eval 会在 keyed 配方中加入 eval 的选择性
   区块(行的 keyed.selectivity、结果的 keyed 方式比较与比较用键)。当键与方向用不同对照集创建等，无法推断键的对照集时，
   用 eval(keyed_contrast=创建键的对照集)传入(不传则 keyed 为 skipped)。
   例: Zundamon 的语尾 configs/contrast/zundamon_endings.yaml(direction(positions="response") → direction_key(layers=[…]) → direction_key_check)。
   只想完整跑一遍时用 direction_key_pipeline(contrast, layers)(方向 → 键 → 用与创建相反侧 subset 做 key-check，在 1 个任务中完成。
   若已有同名方向则复用，rebuild_direction=True 可重新创建)。按用途的模板在 configs/keyed/(句尾说法、特定场景的回复、
   用 system 创建的状态。about 仅为说明，读取时丢弃)。
   MoE 层用 direction_key(target="shared")写入 shared expert，用 target="expert:e" 写入专家 e(单元为 L:S / L:Ee:keyed:…)。
   专家用 direction_key_experts 选择在写入位置经常被选中的(rate_a 大且 rate_b 小)专家。烧录时
   默认(auto)与 requant="keep" 下，原为 Q8_0 的专家张量保持原类型(f16 会使整个张量变为 F16 而变大)。

"""


def _err(e: Exception) -> dict:
    if isinstance(e, GneError):
        out = {"error": {"code": e.code, "message": e.message, "hint": e.hint}}
        if getattr(e, "estimate", None) is not None:          # NEEDS_CONFIRM: 見積もりを返して止まる(CHANGE-03 C)
            out["estimate"] = e.estimate
        return out
    return {"error": {"code": "INTERNAL", "message": f"{type(e).__name__}: {e}", "hint": None}}


def _strip_ids(d: Any) -> Any:
    """移除 token_ids（token 序列以字符串返回，§6.13 (d)）。"""
    if isinstance(d, dict):
        return {k: _strip_ids(v) for k, v in d.items() if k != "token_ids"}
    if isinstance(d, list):
        return [_strip_ids(x) for x in d]
    return d


def build_server(model: str | Path, *, root: str | Path | None = None, ui_url: str | None = None,
                 n_gpu_layers: int = -1, n_ctx: int = 4096, seed: int = 0):
    from mcp.server.mcpserver import MCPServer

    from . import __version__
    from .api import Session
    session = Session(model, root=root, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    arbiter = GpuArbiter()
    jobs = JobRunner(session, arbiter, standard_kinds())
    srv = MCPServer("gne", instructions=INSTRUCTIONS, version=__version__, log_level="WARNING")

    async def call(fn, *a, gpu: bool = True, **kw) -> dict:
        def run():
            session.warnings = []
            try:
                if gpu:
                    with arbiter.interactive():
                        out = fn(*a, **kw)
                else:
                    out = fn(*a, **kw)
                out = _strip_ids(out)
                if session.warnings:
                    out = {**out, "warnings": list(session.warnings)}
                return out
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await anyio.to_thread.run_sync(run)

    @srv.tool()
    async def model_overview() -> dict:
        """ModelMeta、arch 警告、bake 是否可用，以及各层的 coverage（ρ_L・激活值统计・logit lens・steer lens・sweep・反向查找）。"""
        return await call(session.model_overview, gpu=False)

    @srv.tool()
    async def layer_summary(layer: int, top: int = 20) -> dict:
        """层摘要：ρ_L、steer lens 的 KL（中位数・p90・超过对照 95% 分位的比例）、sweep 超过对照群的比例与上位单元、logit lens 前几名、反向查找前几名。"""
        return await call(session.layer_summary, layer, top=top, gpu=False)

    @srv.tool()
    async def neuron(layer: int, neuron: int, rollouts: int = 4, compute: bool = False) -> dict:
        """单个神经元的档案（sweep 排名・词・rollout、steer lens、logit lens、反向查找 cos、激活值统计、coverage）。
        compute=True 时，仅针对该神经元计算缺失的 logit lens 与 steer lens（数秒）。"""
        return await call(session.neuron, layer, neuron, rollouts=rollouts, compute=compute, gpu=compute)

    @srv.tool()
    async def search(q: str, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
        """匹配的神经元列表。mode：lens（steer/logit lens 上位词）、words（sweep 的词）、name（已命名）。"""
        return await call(session.search, q, mode=mode, layer=layer, limit=limit, gpu=False)

    @srv.tool()
    async def steer(units: list[str], prompt: str | None = None, messages: list[dict[str, str]] | None = None,
                    max_tokens: int = 128, compare: bool = True, dose_unit: str = "rel",
                    system: str | None = None, when: str = "all", ablate: str | None = None) -> dict:
        """用单元（例 ["14:8949:+1.0", "13:7956:-1.0"]）推动并生成（greedy）。compare=True 时并列显示无干预结果。
        在 "14:8949:x2" 中加 x 表示 gain（将该神经元输出放大 k 倍，x0 表示消除。§6.16）。
        when 为添加位置：all（全部位置）/ gen（仅生成中，不破坏对提示词的理解）/ assistant（仅 assistant 发言）/
        after:<字符串>（输出中出现该字符串后）/ ramp:<a>-<b>（倍率从 a 到 b）。
        ablate 是用 direction 创建的方向执行投影消除（"NAME"、"NAME@层"、"NAME:0.5"）。units 可以为空。"""
        def fn():
            r = session.recipe(units=units or None, dose_unit=dose_unit, when=when, ablate=ablate)
            return session.steer(r, prompt=prompt, messages=messages, system=system, max_tokens=max_tokens,
                                 compare=compare)
        return await call(fn)

    @srv.tool()
    async def steer_lens(layer: int, neurons: str, dose: float = 0.8, random_controls: int = 16) -> dict:
        """指定神经元（例 "2742" 或 "1,5,9-12"）的 steer lens（向 ± 推动时下一 token 的变化、KL 与对照）。适合少量神经元。"""
        return await call(session.lens_steer, layer, neurons=neurons, dose=dose, random_controls=random_controls)

    @srv.tool()
    async def fires(text: str, top: int = 50) -> dict:
        """流式输入文本，按 token 返回最亮的神经元（激活值 ÷ 基准 最大）前 top 个。"""
        return await call(session.fires, text, top=top)

    @srv.tool()
    async def reverse(contrast: str | None = None, system_a: str | None = None, system_b: str | None = None,
                      user_prompts: list[str] | None = None, name: str | None = None, top: int = 50,
                      method: str = "mean_diff", positions: str = "all") -> dict:
        """反向查找：从对照提示词集（configs/contrast/ 的名称或路径，或直接提供 system_a / system_b / user_prompts）
        创建方向，并对所有层・所有神经元排序。验证（rollout）用 job_start("reverse", {..., "verify": true})。"""
        def fn():
            path = _contrast_path(session, contrast, system_a, system_b, user_prompts, name)
            return session.reverse(contrast=path, top=top, method=method, positions=positions, name=name)
        return await call(fn)

    @srv.tool()
    async def direction(contrast: str, name: str | None = None, select: str | None = None, mode: str = "ablate",
                        layers: str | None = None, dose: float = 1.0, positions: str = "last",
                        random_controls: int = 16) -> dict:
        """从对照集（configs/contrast/ 的名称或路径。system_a / system_b / user_prompts，选层使用 validate.match）
        创建并保存所有层方向 r_L = mean_A − mean_B（生成提示词的最后一个 token）（§6.17）。
        select="auto" 时按候选层测量效果并用任务选层（结果用 job_status 查看）。mode="add" 检查在第 L 层
        加上平均差的 dose 倍后判定一致率是否提高；mode="ablate" 检查从所有层消除 r_L 后是否下降（从 B 侧 KL 不超过 0.1 的层中选）。
        创建的方向可用 "L:dir:NAME:+0.5"（rel dose）加入 steer 的 units。
        positions="response" 时用 A / B 的 system 生成回复，并用回复 token 测量（添加人格时通常更有效）。
        按层运行 random_controls 个同量级随机方向作为对照，只选择效果超过对照 p95 的层
        （行中的 ctrl_effect_p95・null_pct，所选层的 selected_null_pct。0 表示无对照）。"""
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "select": select, "mode": mode, "layers": layers, "dose": dose, "positions": positions,
                      "random_controls": random_controls}
        except Exception as e:  # noqa: BLE001
            return _err(e)
        if select == "auto":
            try:
                return jobs.start("direction", params).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(session.direction, **params)

    @srv.tool()
    async def recipe_save(name: str, units: list[str], dose_unit: str = "rel", ablate: str | None = None,
                          when: str | None = None) -> dict:
        """将配方保存到 <root>/recipes/<name>.yaml。ablate 是方向投影消除（"NAME"、"NAME@层"、"NAME:倍率"），
        when 是添加位置（all / gen / assistant / skip-replies / after:<字符串> / ramp:<a>-<b>）。"""
        return await call(session.recipe_save, name, units, dose_unit=dose_unit, ablate=ablate, when=when, gpu=False)

    @srv.tool()
    async def export_cvec(name: str, format: str = "cvec", verify: bool = False) -> dict:
        """将保存的配方导出为 control vector GGUF（或 npz / json / safetensors）。verify=True 时与 llama-completion 比较。"""
        def fn():
            from .cli import _EXT
            if format not in _EXT:
                raise GneError("BAD_FORMAT", f"format 只能为 {', '.join(_EXT)} 之一")
            r = session.recipe(path=session.recipe_path(name))
            out = session.store.export_dir() / f"{name}{_EXT[format]}"
            return session.export_cvec(r, out, fmt=format, verify=verify)
        return await call(fn, gpu=True)

    @srv.tool()
    async def gain_export(name: str, bake: bool = False) -> dict:
        """导出保存配方中的 gain 单元（如 "14:8949:x2"，将输出放大 k 倍，x0 表示消除）（§6.16）。
        默认输出 LoRA 适配器 GGUF（可用于 llama.cpp 的 --lora，Qwen 系也可）。bake=True 时输出改写 W_up 的 GGUF。"""
        def fn():
            r = session.recipe(path=session.recipe_path(name))
            out = session.default_export_path(r, bake=bake, kind="gain", name=_recipe_stem(name))
            return session.gain(r, out, bake=bake)
        return await call(fn, gpu=False)

    @srv.tool(name="eval")
    async def eval_recipe(recipe: str | None = None, units: list[str] | None = None, doses: list[float] | None = None,
                   when: str | None = None, random_controls: int = 0, reactivity: str | None = None,
                   head_controls: int = 0, compare_methods: bool = False, match_effect: float | None = None,
                   score_layer: int | None = None, alloc: str | None = None, n_heads: list[int] | None = None,
                   control_pool: str = "before", direction_metrics: bool = True, dir_max_examples: int = 64,
                   keyed_contrast: str | None = None, keyed_metrics: bool = True, keyed_examples: int = 8,
                   keyed_match: bool = True, keyed_pseudo: int = 8, keyed_subset: str | None = None) -> dict:
        """将副作用检查（§6.14）作为任务提交。recipe 是已保存配方的名称，units 是临时指定（二选一）。
        doses 是配方整体倍率（默认 [0, 0.25, 0.5, 0.75, 1]，0 为无干预）。逐倍率测量 exact / format 正确率、ppl、
        语言混入（简体字・韩文等）、人格变化量（无干预中不存在的词所占比例等），并返回可发布（能力下降不超过 5pt、
        ppl 上升不超过 10%、混入不增加）的倍率。4B 模型需数分钟。用 job_status 查看结果（保存到 <root>/evals/）。
        random_controls=N 时，每个倍率也测量 N 个同量级随机对照配方（耗时约为 N 倍）。行中的 ctrl_novelty_p95・
        novelty_null_pct（新词高于对照多少比例）、ctrl_capability_drop_p50・ctrl_ppl_worse_p50（仅由规模造成的恶化）。
        传入 reactivity（对照集名称或路径。需要 reactivity 对与 validate.match）时，在行的 reactivity 中加入
        有/无触发词的一致率及其差异。还会给出与 head_controls 个随机注意力头（相同 k）比较的 null_pct。
        方向を使うレシピ(proj・方向のユニット・ablate)の行には、射影の値の下がり方 dir_drop_a・残差の差の直交成分 drift_orth_a /
        drift_orth_b・判定の一致率 rate が入る(CHANGE-06 F.1)。proj の random_controls は同数・同じ s のランダムなヘッド
        (control_pool: before / same-layers)で、dir_null_pct・rate_null_pct・drift_orth_b_null_pct を出す。
        compare_methods=True は、proj のレシピと同じ方向で全層射影・ヘッドの方向成分・ヘッドの丸ごと倍率・ニューロンの消去を
        A 側の射影の値の水準(match_effect、省けばこのレシピ)をそろえて比べる(F.3)。alloc(heads_allocate の名前)と n_heads で、
        ヘッド数ごとの組を評価する(F.4、recipe・units は要らない)。方向の指標は direction_metrics=False で測らない。
        dir_max_examples はその測定で読む質問の数(既定 64、A・B それぞれ等間隔に間引く。0 で全部)。
        keyed のユニットを持つレシピ(CHANGE-07)では、行の keyed(目的の層の書き込みたい位置・書き込みたくない位置・dont_touch の残差の
        r̂ 方向の変化と eval の選択性 selectivity、random_controls があれば比較用の鍵の対照の null_pct)と、結果の keyed(方式の比較
        methods・compare、比較用の鍵 pseudo、text)が入る(生成なしの prefill だけ)。keyed_contrast は鍵を作った対照集合(名前かパス。方向を作った
        対照集合と同じなら省ける。分からなければ keyed は skipped と hint)、keyed_examples は会話の組ごとに読む数(0 で全部)、
        keyed_pseudo は比較用の鍵の本数、keyed_subset は読む例(even / odd / all、既定は鍵の作成と反対側)、keyed_match=False で
        大きさ合わせを省く、keyed_metrics=False で測らない。"""
        try:
            if alloc:
                return jobs.start("eval", {"alloc": alloc, "n_heads": n_heads, "when": when,
                                           "random_controls": random_controls}).view()
            p = {"recipe": recipe, "units": units, "when": when, "random_controls": random_controls}
            if compare_methods:
                p.update({"compare_methods": True, "match_effect": match_effect, "score_layer": score_layer})
                p.pop("random_controls")
            elif control_pool != "before":
                p["control_pool"] = control_pool
            if reactivity:
                p["reactivity"] = str(_contrast_path(session, reactivity, None, None, None, None))
                p["head_controls"] = head_controls
            if doses is not None:
                p["doses"] = doses
            if not compare_methods:
                p["direction_metrics"] = bool(direction_metrics)
                p["dir_max_examples"] = int(dir_max_examples) or None
                p.update({"keyed_metrics": bool(keyed_metrics), "keyed_examples": int(keyed_examples) or None,
                          "keyed_match": bool(keyed_match), "keyed_pseudo": int(keyed_pseudo)})
                if keyed_contrast:
                    p["keyed_contrast"] = str(_contrast_path(session, keyed_contrast, None, None, None, None))
                if keyed_subset:
                    p["keyed_subset"] = keyed_subset
            return jobs.start("eval", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def ablate_export(name: str, layers: str = "all", scale: float = 1.0, bake: bool = False,
                            from_layer: int | None = None) -> dict:
        """导出方向 name（用 gne direction / direction 工具创建）的投影消除（§6.17，即 abliteration）。
        默认输出 LoRA 适配器 GGUF（用于 llama.cpp 的 --lora、--lora-scaled 可部分消除）。bake=True 时创建以 Q8_0 改写权重的 GGUF
        任务（文件较大。用 job_status 查看结果）。layers 为 "all" / "10-40"，scale 是消除比例（1 表示全部）。
        使用的 r 来自 from_layer 层测得的方向（默认是 direction 的 select 所选层）。"""
        from .cli import parse_layers
        try:
            Ls = parse_layers(layers)
            suffix = "-ablate.gguf" if bake else "-ablate.lora.gguf"
            out = str(session.store.export_dir() / f"{name}{suffix}")
            params = {"direction": name, "out": out, "layers": Ls, "scale": scale, "bake": bake, "from_layer": from_layer,
                      "verify": True}
        except Exception as e:  # noqa: BLE001
            return _err(e)
        if bake:
            try:
                return jobs.start("ablate", params).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(lambda: session.ablate(**{**params, "out": out}))

    @srv.tool()
    async def bake(name: str, dst: str, verify: bool = True) -> dict:
        """将保存的配方烧制为 FFN 偏置并生成 GGUF 到 dst（仅支持的 arch，Qwen 系不可）。"""
        def fn():
            r = session.recipe(path=session.recipe_path(name))
            return session.bake(r, dst, verify=verify)
        return await call(fn)

    @srv.tool()
    async def head(layer: int, head: int, lens: bool = True) -> dict:
        """单个 attention 注意力头的档案（CHANGE-04）：kind（attn / linear）、宽度、K/V 组、已保存 heads_contrast 中的
        指标与排名、名称、曾应用过的 gain、W_O 列块的 logit lens（lens=False 时省略）。"""
        return await call(session.head, layer, head, lens=lens, gpu=False)

    @srv.tool()
    async def heads_contrast(contrast: str, name: str | None = None, direction: str | None = None,
                             layers: str | None = None, max_tokens: int = 48) -> dict:
        """将注意力头的对照分析作为任务提交（用 job_status 查看结果）。用有人格（system_a）与中立（system_b）
        运行对照集（configs/contrast/ 的名称或路径）的会话，逐注意力头测量 trigger_mass（对触发词关注比例的差异，
        需要对照集的 triggers）・pattern_shift（关注分布的 JS）・out_shift（输出差异）・dir_cos（输出差异写到方向 r_L 的
        cos，存在 direction 或同名方向时），并以中立对中立的差异为の p95 を基準値按 z 排序。"""
        from .cli import parse_layers
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "direction": direction, "layers": parse_layers(layers), "max_tokens": max_tokens}
            return jobs.start("heads_contrast", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_score(name: str, contrast: str | None = None, layers: list[int] | None = None,
                              units: list[str] | None = None, positions: str = "last", side: str = "both") -> dict:
        """速い指標(CHANGE-06 B、生成なしなので数秒)。対照集合(省けば方向を作ったもの)の A 側・B 側をプロンプトの最後まで
        読ませ、層ごとに 残差 · r̂_L の mean / sd / cohen_d と、基準層(cohen_d 最大。介入なし・layers 省略のとき記録する)、
        基準層の方向とのコサイン(方向が層でどれだけ回るか)を返す。units("33:H16:s6@NAME" など)を渡すと、その介入での
        射影の値の下がり方 drop と、残差の差の方向成分・直交成分 drift(par / orth、A・B それぞれ)。
        方向を使う介入は、まずこれで基準層と分離度を確認してから進める。"""
        def fn():
            d = session.direction_score(name, contrast=contrast, layers=layers, units=units, positions=positions, side=side)
            d.pop("values", None)
            d.pop("values_after", None)
            return d
        return await call(fn)

    @srv.tool()
    async def heads_causal(name: str, contrast: str | None = None, mode: str = "ablate", layers: str = "auto",
                           candidates: str | None = None, score_layers: list[int] | None = None,
                           base_units: list[str] | None = None, delta_s: float = 3.0, hard: int = 16,
                           b_side_max: float | None = None, top_layers: int = 4) -> dict:
        """因果スクリーニングをジョブで投げる(CHANGE-06 C、job_status で結果。生成なし)。方向 name への射影の値(score_layers、
        既定は基準層と後段 2 層)が、ヘッドを止めたとき(mode="ablate")どれだけ下がるか。layers="auto" は層単位で全ヘッドを
        止めて下がる top_layers 層を選び、その層のヘッドを 1 個ずつ測る。candidates は all / contrib:N / "L:Hh,…"。
        mode="marginal" は base_units(いまの組 "L:Hh:sS")から s を delta_s 上げたときの、効きにくい例 hard 件での
        下がり方 ÷ delta_s(効率 eta)。ablate と marginal は順位が違う(候補出しは ablate、配り直しは marginal)。
        行には直接の寄与 contrib と ratio = causal / contrib(1 より大きい = 間接の経路、小さい = 補い合い)、
        B 側への影響(b_side_delta・drift_orth_b、b_side_max で除外の印)。保存先 <root>/heads/<model_id>/<name>.causal.jsonl。"""
        try:
            params = {"direction": name, "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None)),
                      "mode": mode, "layers": layers, "candidates": candidates, "score_layers": score_layers,
                      "base_units": base_units, "delta_s": delta_s, "hard": hard, "b_side_max": b_side_max,
                      "top_layers": top_layers}
            return jobs.start("heads_causal", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_allocate(name: str, target: float, contrast: str | None = None, candidates: str | None = None,
                             n_heads: list[int] | None = None, score_layer: int | None = None, hard: int = 16,
                             b_side_max: float | None = None, method: str = "proportional", s_max: float | None = None) -> dict:
        """効率ベースの強さ配分をジョブで投げる(CHANGE-06 D、job_status で結果。生成なし)。heads_causal の結果(candidates、
        既定は方向の名前)から、効きにくい例(A 側で射影の値が高く残る hard 件)を選び、効率 eta を測り、n_heads(既定 [6, 5, 4])の
        各ヘッド数で s ∝ eta に配って、効きにくい例の射影の値の平均が target になるよう二分探索する。比較表(ヘッド数・各ヘッドの s・
        最大の s・残る件数・A 側の射影の値の最大・B 側の直交成分のずれ)と recommended(残る件数と B 側のずれが同程度なら最少の
        ヘッド数)、各組のレシピのパスを返す。method="greedy" は比を固定せず、いちばん効くヘッドを 1 本ずつ上げる(s_max の既定はなし、
        止まるのは目標か B 側のずれ)。行の stop が止まった理由、capped が上限に当たったヘッド。目標は射影の値の目標なので、組は振る舞いで確かめる。"""
        try:
            params = {"direction": name, "target": target, "candidates": candidates, "score_layer": score_layer,
                      "hard": hard, "b_side_max": b_side_max, "method": method, "s_max": s_max,
                      "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None))}
            if n_heads:
                params["n_heads"] = list(n_heads)
            return jobs.start("heads_allocate", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_pipeline(spec: dict[str, Any] | str, target: float | str | None = None, n_heads: list[int] | None = None,
                             max_examples: int | None = None, resume: bool = True) -> dict:
        """対照集合(name・user_prompts か側ごとの質問 prompts_a / prompts_b・system_a / system_b か recipe_a / recipe_b、
        任意で pipeline の設定。表か JSON / YAML の
        パス)から、方向づくり → 基準層 → 因果スクリーニング → 強さの配分(効き目に比例した最初の組 → 効率で配り直す →
        ヘッド数の比較)をジョブで 1 回で回す(job_status で結果)。質問は何百個でもよく、止めて測る段階は max_examples 個
        (既定 64)に間引く。target は数か "B"(B 側の平均まで消す)か "mid"(省いたときの既定、A と B の真ん中)。
        真ん中では振る舞いが半分ほどしか消えないことが多いので、消し切りたいときは "B"。結果の recommended が推す組のレシピ。resume(既定 True)なら、
        同じ対照集合・同じ設定で作り済みの方向と因果の結果を使い回す(止めて流し直したとき、配分からやり直せる)。"""
        try:
            p = {"spec": spec, "target": target, "n_heads": n_heads, "max_examples": max_examples,
                 "resume": None if resume else False}
            return jobs.start("head_pipeline", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def fieldtest(model: str | None = None, against: str | None = None, lora: str | None = None, name: str | None = None,
                        prompts: dict[str, Any] | None = None, match: str | None = None, long_tokens: int = 3072, tool_tokens: int = 1536,
                        prompt_tokens: int = 1536, n_parallel: int = 4, llama_server: str | None = None, ppl: bool = True,
                        bench: list[str] | None = None, bench_n: int | None = None, bench_tokens: int | None = None,
                        presence_penalty: float | None = None) -> dict:
        """将实际使用测试作为作业提交（用 job_status 获取结果）。按真实用法（llama-server、开启思考、开启采样）测量长任务
        （达到上限被截断的比例、循环、正文为空）、工具调用（是否调用、名称、参数、无需时是否不调用、工具结果后能否继续）、问题集
        （prompts 为 {名称: 文件或问题列表}，match 输出匹配率）、基准测试（bench：gsm8k、mmlu_pro、math500、ifeval 的准确率，以及仅一侧正确的题数）、perplexity，并与 against 文件比较（compare 的 worse、
        same_level）。也可直接测量烧录后的文件。省略 model 时使用当前加载的模型。运行期间该模型会离开 GPU，
        其他 GPU 处理需要等待。27B 模型加 200 道问题集时，每个文件需 30～45 分钟。生成的文本不写入结果，而是写到 replies_path。"""
        try:
            p = {"model": model, "against": against, "lora": lora, "name": name, "prompts": prompts, "match": match, "long_tokens": long_tokens,
                 "tool_tokens": tool_tokens, "prompt_tokens": prompt_tokens, "n_parallel": n_parallel, "llama_server": llama_server, "ppl": ppl,
                 "bench": bench, "bench_n": bench_n, "bench_tokens": bench_tokens, "presence_penalty": presence_penalty}
            return jobs.start("fieldtest", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_refine(name: str, n_heads: int | None = None, new_heads: list[int] | None = None, resume: bool = True) -> dict:
        """組を当てても目標まで下がらない A 側の質問(残り)だけで、追加のヘッドを探すジョブを投げる(job_status で結果)。
        name は一括(heads_pipeline)か heads_allocate の名前、n_heads は残りを測る組(既定は推す組)。残りの質問と B 側の質問で、
        組を当てたまま方向を作り直し(方向 <name>_rK)、基準層 → 因果 → 配分で足すヘッドと強さ(new_heads、既定 [3, 2, 1])を決め、
        元の組と合わせたレシピ(combined.recipe)を作って、元の方向で残りの数(remain_before → remain_after)と B 側のずれを測り直す。"""
        try:
            p = {"name": name, "n_heads": n_heads, "new_heads": new_heads, "resume": None if resume else False}
            return jobs.start("heads_refine", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_candidates(alloc: str, prompts: list[str] | dict[str, list[str]] | str | None = None,
                               scales: list[float] | None = None, n_heads: list[int] | None = None, max_prompts: int = 24,
                               max_tokens: int = 120, sampling: str | dict[str, Any] = "llama_server", seed: int = 0,
                               judge_b: bool = False, with_recipe: str | None = None) -> dict:
        """強さの候補づくりをジョブで投げる(job_status で結果)。配分の結果(alloc、heads_allocate / heads_pipeline の名前)の
        比較表の組 × 全体の倍率 scales(既定 [0.5, 1, 1.5, 2, 3])と無介入で、A 側と B 側の質問に返事を作る(同じ種のサンプリング)。
        正の倍率は A 側の振る舞いを消す向き(A 側を採点)、負の倍率は符号を返して B 側で強める向き(B 側を採点)。
        候補ごとの判定の一致率(rate_a / rate_b)・繰り返しの少なさ(d4)・無介入と同じ返事の割合と、採点する側の返事を
        どの候補か伏せて並べた採点のファイル(judge.round、基準は judge.rubric)を返す。prompts は調整に使っていない質問
        (省くと対照集合の validate.prompts、無ければ調整に使った質問で held_out が False)。with_recipe は全部の候補
        (無介入の base も)に重ねるレシピ(ニューロンの種火など)。採点は heads_candidates_tally。"""
        try:
            p = {"alloc": alloc, "prompts": prompts, "scales": scales, "n_heads": n_heads, "max_prompts": max_prompts,
                 "max_tokens": max_tokens, "sampling": sampling, "seed": seed, "judge_b": judge_b,
                 "with_recipe": with_recipe}
            return jobs.start("heads_candidates", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def heads_candidates_tally(name: str, scores: str | dict[str, list[int]] | None = None) -> dict:
        """候補の採点を集計する(GPU を使わない)。scores に「番号 P C」の行(または {番号: [P, C]})を渡すと採点のファイルに
        書き足してから集計する。P = 目標の振る舞いの強さ、C = 自然さ(どちらも 1〜5、基準は RUBRIC.md)。返り値の scores が
        候補 → 側 → P・C・min_PC・P_ge3・remove(min(6 − P, C)、消す向きの目的)・add(min(P, C)、強める向きの目的)、
        best が向きごとのいちばん良い候補。返事は名前を伏せたまま採点し、集計の前に候補を見ない。"""
        try:
            d = session.heads_candidates_tally(name, scores=scores)
            d.pop("replies", None)
            return d
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def headproj_export(recipe: str, bake: bool = False, requant: str | None = None, verify: bool = True,
                              strip_meta: bool = False, into: str | None = None) -> dict:
        """保存したレシピの proj のユニット(ヘッドの方向成分の射影、と gain・keyed)を書き出す(CHANGE-06 E)。既定は LoRA アダプタ
        GGUF(同じ層・同じ方向は階数 1 に畳む。数百 KB)。bake=True なら残差に書き込む重み行列だけ量子化し直した GGUF をジョブで作り
        (requant の既定は q8_0。keyed を含むなら auto: 元が Q8_0 のテンソルは Q8_0 のまま、それ以外は f16。Q8_0 のまま
        では編集の 1 割以上が丸めに消えるテンソルだけ f16。"q8_0" / "f16" / "keep" も選べる)、
        verify なら焼き込み版と LoRA 版を方向への射影の値・最初のトークンの KL・最尤トークンの一致率・ppl で比べる(V26)。射影の値は
        層・側ごとに例ごとの差の最大(max_abs_diff)を合格の線(threshold)と比べる。線は許容誤差(射影の値の標準偏差の 1 割、resolution)で、
        最後の層(last_layer)だけ許容誤差 + 計算方法の違いだけで出る誤差(noise.max_abs_diff = 書き込みがほぼ 0 の LoRA と
        LoRA なしの射影の値の差の最大。Q8_0 では LoRA を当てるだけで最後の層の射影の値が動くため)。
        strip_meta=True なら gne の記録(gne.* のメタデータ・レシピ・名前の印・横の .headproj.json)を残さない(人に渡す版)。
        結果の local_path_keys は、メタデータに手元のパスらしい文字列が残っているキー。into は焼き込む先の別の GGUF
        (同じ構造の量子化していない F16 / BF16 など。方向はこのモデルのものを使う。requant="keep" で元の型のまま、verify はできない)。"""
        return await _export(recipe, None, None, None, None, bake=bake, requant=requant, verify=verify,
                             strip_meta=strip_meta, into=into)

    async def _export(recipe, units, name, key, direction, *, bake, requant, verify, strip_meta, into,
                      lora_f16: bool = False) -> dict:
        """headproj_export・keyed_export の中身(jobs._headproj_export。ファイル名の既定は Session.default_export_path)。
        焼き込みはジョブ、LoRA はその場で書く。"""
        from .jobs import _headproj_export
        p = {"recipe": recipe, "units": units, "name": name if name else (_recipe_stem(recipe) if recipe else None),
             "key": key, "direction": direction}
        p = {k: v for k, v in p.items() if v is not None}
        if bake:
            try:
                q = {**p, "bake": True, "requant": requant, "verify": verify and not into, "strip_meta": strip_meta}
                if into:
                    q["into"] = into
                return jobs.start("headproj_export", q).view()
            except Exception as e:  # noqa: BLE001
                return _err(e)
        return await call(lambda: _headproj_export(session, **p, strip_meta=strip_meta, lora_f16=lora_f16))

    @srv.tool()
    async def keyed_export(recipe: str | None = None, units: list[str] | None = None, name: str | None = None,
                           key: str | None = None, direction: str | None = None, bake: bool = False,
                           requant: str | None = None, verify: bool = False, strip_meta: bool = False,
                           into: str | None = None, lora_f16: bool = False) -> dict:
        """鍵つきの書き込み(keyed、CHANGE-07)を書き出す(headproj_export の keyed 向けの別名。proj・gain が混ざっていても同じ
        ファイルに入る)。recipe は保存したレシピの名前、または units(例 ["19:ffn:keyed:a1@zundamon_endings#zundamon_endings"])と
        name。key・direction は #KEY・@NAME を省いたユニットの鍵と方向。既定は LoRA アダプタ GGUF(層・鍵ごとに階数 1、llama.cpp の
        --lora でそのまま当たる。配るならこれが第一候補。keyed だけなら <名前>-keyed.lora.gguf)。bake=True なら残差に書き込む
        重み行列(ffn_down など)だけ書き直した GGUF をジョブで作る(requant の既定は auto: テンソルごとに、元が Q8_0 なら Q8_0 のまま
        (f16 にすると計算方法が変わり、LoRA 版とかえってずれた)、それ以外(K-quant・IQ・F16 など)は f16。Q8_0 のままでは
        rank 1 の小さな編集の 1 割以上が丸めに消えるテンソルだけ f16(記録の fallback)。結果の edit_kept_min(残った編集の割合)が
        0.9 を下回ったら警告。requant_error_max は編集と無関係な丸めのゆらぎも数えるので、小さな編集では大きく出る)。MoE の専門家のテンソル(ffn_down_exps、expert:e)も同じ規則で、
        Q8_0 のままなら編集しない専門家はバイトのまま、f16 だとテンソル全体が F16 になる(結果の expert_bytes_added が増えた大きさ)。
        verify=True で焼き込み版と LoRA 版を比べる(V33。判定は headproj_export の verify と同じで、最後の層の射影の値の合格の線
        threshold は許容誤差 + 計算方法の違いだけで出る誤差 noise.max_abs_diff)。"""
        if bool(recipe) == bool(units):
            return _err(GneError("BAD_ARGS", "请传入 recipe（已保存配方的名称）或 units 之一"))
        return await _export(recipe, units, name, key, direction, bake=bake, requant=requant, verify=verify,
                             strip_meta=strip_meta, into=into, lora_f16=lora_f16)

    @srv.tool()
    async def direction_key(contrast: str, layers: list[int] | str, name: str | None = None, positions: str | None = None,
                            positions_b: str | None = None, skip: int | None = None, max_tokens: int = 96,
                            q_mode: str = "examples", q_energy: float = 0.99, q_max_rank: int | None = None,
                            q_max_samples: int = 4096, subset: str = "even", pseudo: int = 16,
                            replies_a: str | list[str] | None = None, replies_b: str | list[str] | None = None,
                            iv: str | list[str] | None = None, target: str = "ffn", seed: int = 0,
                            dont_touch_start: bool | None = None) -> dict:
        """鍵つきの書き込みの鍵 κ を作るジョブを投げる(CHANGE-07、job_status で結果)。対照集合(configs/contrast/ の名前かパス)の
        書き込みたい位置の活性の平均 μ を、書き込みたくない位置と dont_touch の活性が張る空間に直交させ、κᵀμ = 1 にして
        <root>/keys/<model_id>/<name>.npz に保存する。layers は必須(例 [8, 14, 20])。positions(省くと対照集合の値)が
        "before:<正規表現>" なら推論時と同じ文脈のモデル自身の返事で、一致する文字を書く直前を書き込みたい位置、残りの位置を書き込みたくない位置にする。
        replies_a / replies_b は返事を読ませるファイルかリスト(生成しない)。q_mode: examples(既定)/ means / samples。
        subset は既定 even(direction_key_check は odd で確かめる)。iv はこの介入(レシピかユニット)を当てた状態で作る。
        target は書き込み先: ffn(既定、dense の ffn_down の入力)/ attn(attn_output・ssm_out の入力 = ヘッド出力の連結)/
        shared(MoE の層の shared expert の ffn_down_shexp の入力)/ expert:e(ffn_down_exps の専門家 e の入力。専門家 e が
        選ばれた位置だけで作るので、書き込みたい位置でよく選ばれる専門家を選ぶ)。ユニットは L:ffn / L:attn / L:S / L:Ee:keyed:…。
        dont_touch の各群の質問の、返事を書き始める直前の位置も Q に入る(既定。dont_touch_start=False か
        対照集合の dont_touch_start: false で切る。入れた行の数は結果の dont_touch_start。書き込みたくない側の、返事を書き始める直前の位置は入れない)。
        結果の層ごとの kn(1 に近いほど書き込みたくない位置と重ならない、0.05 未満は弱い)と build_auc(作成に使った例での値で甘い)。
        warnings を必ず見る: 書き込みたい位置を含む例が 2 未満か、書き込みたい位置が 8 未満(専門家の鍵は層ごとの行)なら少ない。
        "before:<正規表現>" で書き込みたい位置の 25% 以上(μ に占める重み、結果の reply_start.share)が返事の冒頭(最初の
        3 トークンを決める位置)にあると、鍵がどの返事の冒頭にも反応しやすい。dont_touch の群があれば、その質問の、返事を
        書き始める直前の位置が Q に入っているので注(notes)になり、無ければ
        警告になる(ふつうの質問を dont_touch に足して作り直す)。"""
        from .jobs import _layers
        try:
            if not _layers(layers):
                raise GneError("BAD_ARGS", "请在 layers 中指定要创建键的层（例如 [8, 14, 20]）")
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, None)), "layers": layers, "name": name,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens,
                 "q_mode": q_mode, "q_energy": q_energy, "q_max_rank": q_max_rank, "q_max_samples": q_max_samples,
                 "subset": subset, "pseudo": pseudo, "replies_a": replies_a, "replies_b": replies_b, "iv": iv,
                 "target": target, "seed": seed, "dont_touch_start": dont_touch_start}
            return jobs.start("direction_key", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_check(key: str, contrast: str | None = None, subset: str = "odd", layers: list[int] | str | None = None,
                                  positions: str | None = None, positions_b: str | None = None, skip: int | None = None,
                                  max_tokens: int | None = None, replies_a: str | list[str] | None = None,
                                  replies_b: str | list[str] | None = None, iv: str | list[str] | None = None,
                                  strict: bool = False) -> dict:
        """鍵の分離の確認のジョブを投げる(CHANGE-07、job_status で結果)。対照集合(省くと鍵に保存した作成時の例で、甘い値)の
        各位置で鍵の反応の強さ ρ = κᵀx を測り、層ごとに書き込みたい位置 対 書き込みたくない位置の AUC(auc は例単位、
        auc_token はトークン単位)、key-check の選択性 selectivity(例ごとの ρ の平均の、書き込みたい位置 − 書き込みたくない
        位置)と比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の p95、beats_pseudo(選択性が比較用の鍵の p95 を超えた)、
        dont_touch の ρ の中央値を返す。位置・skip・max_tokens の既定は鍵を作ったときの値。作成に使った例と重なると
        held_out が false(strict なら BAD_ARGS)。判断は held_out が true の結果で、beats_pseudo の層だけを使う。
        層ごとの start_rho は書き込みたくない位置と dont_touch の、返事の冒頭(位置 0 = 返事を書き始める直前の位置、1・2 = 返事の
        最初の 2 トークンを読んだ位置)での ρ の中央値。dont_touch の ρ の中央値は返事の全位置の中央値なので、返事の冒頭の
        1 位置で反応しても見えない。中央値が 0.4 以上なら warnings に出る(dont_touch なら書き込みたくない場面への漏れ。
        ふつうの質問を dont_touch に足して鍵を作り直す。dont_touch の質問の、返事を書き始める直前の位置は鍵の作成で自動で Q に入る)。保留集合の書き込みたい位置を含む例が
        2 未満か、書き込みたい位置が 8 未満でも warnings に出る(選択性が揺れる)。"""
        try:
            p = {"key": key, "subset": subset, "layers": layers, "positions": positions, "positions_b": positions_b,
                 "skip": skip, "max_tokens": max_tokens, "replies_a": replies_a, "replies_b": replies_b, "iv": iv,
                 "strict": strict, "compact": True,
                 "contrast": None if contrast is None else str(_contrast_path(session, contrast, None, None, None, None))}
            return jobs.start("direction_key_check", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_verify(layer: int, key: str, direction: str, alphas: list[float] | None = None,
                                   target: str = "ffn", direction_layer: str = "own") -> dict:
        """V30(CHANGE-07、数秒): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか。無介入で重み行列の入力 x と出力を
        捕捉し、LoRA を当てた出力の差と比べる(rows の rel_err_median < 3%、鍵の作成に使った書き込みたくない側の例で leak_b < 0.1、v30)。
        target は鍵の書き込み先と同じ ffn / attn / shared / expert:e(例 "expert:31")、alphas の既定は [1, 2]。shared は
        ゲートの前、expert:e はルーターの重みの前の出力を、専門家 e が選ばれた枠で測る。expert:e の rows には other_max
        (ほかの専門家の枠の差、0 のはず)と selection_changed(選ばれる専門家が変わったトークンの数、0 のはず)も入る。
        MoE では結果に gate(鍵の gate_a)・res_scale(granite 系の residual_scale)も出る。"""
        def fn():
            from .keyed import parse_target
            kind, expert = parse_target(target)
            return session.check_keyed(layer, key=key, direction=direction, alphas=tuple(alphas or (1.0, 2.0)),
                                       kind=kind, expert=expert, direction_layer=direction_layer)
        return await call(fn)

    @srv.tool()
    async def direction_key_experts(contrast: str, layers: list[int] | str, positions: str | None = None,
                                    positions_b: str | None = None, skip: int | None = None, max_tokens: int = 96,
                                    subset: str = "even", replies_a: str | list[str] | None = None,
                                    replies_b: str | list[str] | None = None, iv: str | list[str] | None = None,
                                    top: int = 8) -> dict:
        """expert:e の鍵の専門家を選ぶ材料のジョブを投げる(CHANGE-07 MoE、job_status で結果)。鍵と同じ対照集合(configs/contrast/
        の名前かパス)・位置・返事で、書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合を層ごとに返す: layers[].top[] の expert・
        rate_a・rate_b(例ごとの選ばれた割合の平均)・diff・lift(rate_a / rate_b)・weight_a(選ばれたときのルーターの重みの平均
        = 鍵の gate_a の見込み)・rows_a(書き込みたい位置で選ばれる行の数の見込み)・rate_dt(dont_touch の群)。rate_a の大きい順に top 個。
        専門家の鍵は書き込みたい位置で専門家 e が選ばれた行だけでできるので、rate_a が大きく rate_b が小さい専門家を direction_key の
        target="expert:e" に選ぶ。layers は必須(MoE の層、例 [6, 12, 18])。ほかの引数は direction_key と同じ。"""
        from .jobs import _layers
        try:
            if not _layers(layers):
                raise GneError("BAD_ARGS", "请在 layers 中指定要调查的 MoE 层（例如 [6, 12, 18]）")
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, None)), "layers": layers,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens,
                 "subset": subset, "replies_a": replies_a, "replies_b": replies_b, "iv": iv, "top": top}
            return jobs.start("direction_key_experts", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def direction_key_pipeline(contrast: dict[str, Any] | str, layers: list[int] | str | None = None,
                                     name: str | None = None, direction: str | None = None, rebuild_direction: bool = False,
                                     response_tokens: int = 96, target: str | None = None, positions: str | None = None,
                                     positions_b: str | None = None, skip: int | None = None, max_tokens: int | None = None,
                                     q_mode: str | None = None, subset: str | None = None, pseudo: int | None = None,
                                     replies_a: str | list[str] | None = None, replies_b: str | list[str] | None = None,
                                     iv: str | list[str] | None = None, seed: int | None = None,
                                     dont_touch_start: bool | None = None, check: bool = True) -> dict:
        """鍵つきの書き込みを、対照集合 1 つから「方向 → 鍵 → key-check」まで 1 回のジョブで回す(CHANGE-07、job_status で結果。
        UI の鍵のタブの「方向から key-check まで」と同じ)。contrast は対照集合の表(name・user_prompts・positions・dont_touch・
        system_a など。<root>/contrasts/<name>.yaml に書いてから使う)か、configs/contrast/ の名前・configs/keyed/ のひな形の名前・パス。
        1. 方向 direction(省くと対照集合の name): gne direction --positions response と同じ(A 側 = system_a、B 側 = system_b か system なしの
        返事の差)。同じ名前の方向がもうあれば使い回す(rebuild_direction=True で作り直す。中身の違う対照集合から作った方向なら warnings)。
        2. 鍵 name(省くと対照集合の name): direction_key と同じ引数(target・positions・q_mode・subset・pseudo・replies_a など)。
        3. key-check: 作成と反対側の subset(even ↔ odd)で、同じ対照集合・同じ返事で測る(結果の check は direction_key_check を
        compact にしたもの)。check=False で回さない。layers は必須(引数か表の keyed.layers)。表の keyed(layers・target・q_mode・
        subset・pseudo・max_tokens・alpha)は渡していない引数の既定になる(alpha は結果の alpha_hint に返すだけ。レシピの α の目安)。
        結果: direction(name・reused・rebuilt)、key(direction_key の結果)、check(key-check の結果)、contrast_path、warnings。
        判断は direction_key_check と同じ: check の held_out が true で、beats_pseudo の層だけを使う。"""
        from .jobs import _layers
        try:
            spec, path = None, None
            if isinstance(contrast, dict):
                spec = contrast
            else:
                try:
                    path = str(_contrast_path(session, contrast, None, None, None, None))
                except GneError:
                    tpl = CONFIGS / "keyed" / f"{contrast}.json"
                    if not tpl.exists():
                        raise
                    spec = json.loads(tpl.read_text(encoding="utf-8"))
            if layers is not None and not _layers(layers):
                raise GneError("BAD_ARGS", "请在 layers 中指定要创建键的层（例如 [8, 14, 20]）")
            p = {"spec": spec, "contrast": path, "layers": layers, "name": name, "direction": direction,
                 "rebuild_direction": bool(rebuild_direction) or None, "response_tokens": response_tokens, "target": target,
                 "positions": positions, "positions_b": positions_b, "skip": skip, "max_tokens": max_tokens, "q_mode": q_mode,
                 "subset": subset, "pseudo": pseudo, "replies_a": replies_a, "replies_b": replies_b, "iv": iv, "seed": seed,
                 "dont_touch_start": dont_touch_start, "check": None if check else False, "compact": True}
            return jobs.start("direction_key_pipeline", {k: v for k, v in p.items() if v is not None}).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def keys() -> dict:
        """保存した鍵の一覧(CHANGE-07、<root>/keys/<model_id>/): 名前・書き込み先・層・層ごとの kn・作成の条件(位置・subset・q_mode)。"""
        return await call(lambda: {"keys": session.keys()}, gpu=False)

    @srv.tool()
    async def expert(layer: int, expert: int) -> dict:
        """单个 MoE 专家的档案（CHANGE-05）：选择率、选中时的平均权重、常选中的 token（存在 stats act 时）、
        已保存 experts_contrast 中的指标与排名、写入量较大的神经元（q999 × ‖W_down[:, n]‖）。"""
        return await call(session.expert, layer, expert)

    @srv.tool()
    async def experts_contrast(contrast: str, name: str | None = None, direction: str | None = None,
                               layers: str | None = None, max_tokens: int = 48) -> dict:
        """将专家对照分析作为任务提交（用 job_status 查看结果）。分别用带人格（system_a）和中立（system_b）
        读取对照集中的同一回复，逐 MoE 层专家测量 usage_shift（选中比例差）、weight_shift（有效权重差）、
        contrib_cos（贡献差与该层状态方向的 cos）、contrib_norm，并以中立对中立的の p95 を基準値为基准按 z 排序。数秒至数十秒。"""
        from .cli import parse_layers
        try:
            params = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                      "direction": direction, "layers": parse_layers(layers), "max_tokens": max_tokens}
            return jobs.start("experts_contrast", params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def experts_sweep(layer: int, experts: list[int] | None = None, ks: list[float] | None = None,
                            max_tokens: int = 48) -> dict:
        """将按专家进行的 sweep 作为任务提交。用将每个层专家乘以 k（默认 0 与 2）的缩放补丁生成，并按词语出现方式排序
        （零假设为同一层全部专家）。每测试一个专家的一个倍率都会重新读取模型（小模型约 0.3～1 秒）。"""
        try:
            p = {"layer": layer, "experts": experts, "ks": tuple(ks) if ks else (0.0, 2.0), "max_tokens": max_tokens}
            return jobs.start("experts_sweep", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def moe_route(text: str, layers: str | None = None, chat: bool = False) -> dict:
        """流式输入文本，逐 MoE 层返回每个 token 选中的专家与有效路由权重（从大到小）。"""
        from .cli import parse_layers
        return await call(session.moe_route, text, layers=parse_layers(layers), chat=chat)

    @srv.tool()
    async def expert_patch(layer: int, expert: int, k: float, prompt: str, system: str | None = None,
                           max_tokens: int = 64) -> dict:
        """在工作副本上应用将专家 e 的输出乘以 k（0 表示消除）的缩放补丁后生成，并与无干预结果并列。结束后恢复原状
        （不会改动原 GGUF）。若要写入配方，请将 "L:Ee:xk" 传给 recipe_save。"""
        return await call(session.steer, [f"{layer}:E{expert}:x{k:g}"], prompt=prompt, system=system, max_tokens=max_tokens)

    @srv.tool()
    async def writers(contrast: str, name: str | None = None, ref_layer: int | None = None,
                      replies_a: str | None = None, max_tokens: int = 96) -> dict:
        """将写入器强度作为任务提交（用 job_status 查看结果）。使用对照集 A（目标状态，例如病娇的 system）与 B（中立）的
        回复 token，测量各神经元写入残差的量（激活值差异 × 写入方向）对参考层方向的直接贡献。
        也返回嵌入・mixer（如 attention）・MLP 各写入多少比例。replies_a 是让 A 读取的回复文件
        （借用其他模型的回复时）。与激活值差异（读取器）不同，真正沿方向写入的“写入器”会排在前面。"""
        try:
            p = {"contrast": str(_contrast_path(session, contrast, None, None, None, name)), "name": name,
                 "ref_layer": ref_layer, "replies_a": replies_a, "max_tokens": max_tokens}
            return jobs.start("writers", p).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def attention(text: str, layer: int | None = None, head: int | None = None, top: int = 5, pos: int = -1,
                        chat: bool = False, system: str | None = None) -> dict:
        """流式输入文本，查看位置 pos（默认最后）的 token 在各注意力头中关注何处（前 top 个）以及每个头输出的范数。
        同时指定 layer 与 head 时还返回 T × T 模式。chat=True 时将 text 作为用户发言放入模板。"""
        return await call(session.attention, text, layer=layer, head=head, top=top, pos=pos, chat=chat, system=system)

    @srv.tool()
    async def heads_listen(user: str, heads: str, system: str | None = None, reply: str | None = None, max_tokens: int = 64,
                           direction: str | None = None, top: int = 20) -> dict:
        """回复的 token 在所选注意力头（例："14:H0-3"）中关注上下文的哪些 token（注意力的平均）。
        传入 direction（已保存的方向）时，还会返回各上下文 token 通过注意力头输出对该层方向的贡献量
        （注意力 × 门控 × OV，含 attention 后的 norm），以及整层全部注意力头贡献中所选头的部分。省略 reply 时以 greedy 生成。"""
        return await call(session.heads_listen, user, heads=heads, system=system, reply=reply, max_tokens=max_tokens,
                          direction=direction, top=top)

    @srv.tool()
    async def set_name(layer: int, neuron: int, name: str) -> dict:
        """为神经元命名（空字符串表示删除）。会出现在档案与 search(mode="name") 中。"""
        return await call(session.set_name, layer, neuron, name, gpu=False)

    @srv.tool()
    async def job_start(kind: str, params: dict[str, Any] | None = None, workers: str | None = None) -> dict:
        """提交长时间处理。kind: stats_resid / stats_act / lens_logit / lens_steer / sweep / rank / reverse / eval / direction / ablate / heads_contrast / heads_causal / heads_allocate / head_pipeline / heads_candidates / heads_refine / headproj_export / fieldtest / direction_key / direction_key_check / direction_key_experts / direction_key_pipeline / direction_key_verify。
        params 是 gne.api 中同名方法的参数（例 sweep：{"layer": 22, "neurons": "random:64", "design": "quick"}）。
        预计超过 30 分钟的任务不会运行，而是返回 NEEDS_CONFIRM 与估算。在 params 中加入 budget("30m" 等)可按预算
        缩小范围（lens_steer 取靠近方向的上位神经元，sweep 用 quick 设计取 steer lens 前几名，reverse 减少
        待确认候选数，connections 调整 window）。仍要运行时，在 params 中设置 yes: true。
        传入 workers（"auto" 或逗号分隔的名称）时，会加入 gne worker 的文件队列（sweep / lens_steer 按
        神经元拆分，CHANGE-03 B）。用返回值 jobs 中的 id 查看 job_status。"""
        try:
            if workers:
                from .estimate import NeedsConfirm, gate
                from .workers import submit
                try:
                    p, _ = gate(session, kind, dict(params or {}))
                except NeedsConfirm as e:
                    e.error.estimate = e.estimate
                    raise e.error from e
                return submit(session, kind, p, workers=workers)
            return jobs.start(kind, params).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_status(job_id: str | None = None) -> dict:
        """任务进度（done / total / eta_s）与结果。省略 job_id 时列出所有任务。"""
        try:
            from .workers import job_files
            if job_id is None:
                return {"jobs": [j.view() for j in jobs.jobs.values()],
                        "worker_jobs": [{k: v for k, v in j.items() if k != "result"} for j in job_files(session.store.root)]}
            if job_id not in jobs.jobs:                            # ワーカーのジョブ(id か group)
                fj = [j for j in job_files(session.store.root) if job_id in (j["id"], j.get("group"))]
                if fj:
                    return _strip_ids({"jobs": fj})
            return _strip_ids(jobs.get(job_id).view())
        except Exception as e:  # noqa: BLE001
            return _err(e)

    @srv.tool()
    async def job_cancel(job_id: str) -> dict:
        """停止任务（会在下一个单元边界停止。已有结果会保留，可在相同条件下继续）。"""
        try:
            return jobs.cancel(job_id).view()
        except Exception as e:  # noqa: BLE001
            return _err(e)

    for action in ("highlight", "note", "open"):
        _register_ui_tool(srv, action, ui_url)
    return srv


def _recipe_stem(recipe: str) -> str:
    """書き出しのファイル名に使うレシピの名前(YAML のパスならファイル名の stem)。"""
    p = Path(str(recipe))
    return p.stem if p.suffix in (".yaml", ".yml") else str(recipe)


def _contrast_path(session, contrast, system_a, system_b, user_prompts, name) -> Path:
    if contrast:
        p = Path(contrast)
        if not p.exists():
            p = CONFIGS / "contrast" / f"{contrast}.yaml"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"对照提示词集 {contrast} 不存在",
                           "请传入 configs/contrast/ 中的名称或路径，或提供 system_a / system_b / user_prompts")
        return p
    if not (system_a and system_b and user_prompts):
        raise GneError("BAD_ARGS", "请传入 contrast，或同时传入 system_a、system_b、user_prompts 三项")
    import yaml
    nm = name or "contrast"
    if not re.fullmatch(r"[\w.\-]+", nm):
        raise GneError("BAD_ARGS", "name 只能包含字母、数字、_、-、.")
    p = session.store.root / "contrast" / f"{nm}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump({"name": nm, "system_a": system_a, "system_b": system_b,
                                 "user_prompts": list(user_prompts)}, allow_unicode=True), encoding="utf-8")
    return p


def _register_ui_tool(srv, action: str, ui_url: str | None) -> None:
    """ui_* 工具（别册 §11）。调用 gne serve 的 POST /api/ui/{action}。未连接 UI 时返回 UI_NOT_RUNNING。"""

    async def post(body: dict) -> dict:
        if not ui_url:
            return {"error": {"code": "UI_NOT_RUNNING", "message": "未连接 UI",
                              "hint": "请启动 gne serve，并通过 gne mcp --ui-url http://127.0.0.1:8791 连接"}}
        req = urllib.request.Request(f"{ui_url.rstrip('/')}/api/ui/{action}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})

        def send():
            try:
                with urllib.request.urlopen(req, timeout=5) as r:     # noqa: S310  127.0.0.1 の UI だけ
                    return json.loads(r.read() or b"{}")
            except urllib.error.HTTPError as e:                      # 400 は UI 側のエラー({"error": ...})をそのまま返す
                try:
                    return json.loads(e.read())
                except ValueError:
                    return {"error": {"code": "UI_ERROR", "message": f"UI 返回了 {e.code}", "hint": None}}
        try:
            return await anyio.to_thread.run_sync(send)
        except (urllib.error.URLError, OSError) as e:
            return {"error": {"code": "UI_NOT_RUNNING", "message": f"无法连接 UI：{e}", "hint": None}}

    if action == "highlight":
        async def ui_highlight(units: list[str], label: str | None = None) -> dict:
            """在 UI 地图中高亮单元（浅蓝色虚线圈与标签）。units 形如 ["22:2742", "14:8949:+1"]。
            会替换之前的高亮。空列表表示清除。label 是分组标题（例 "外星人候选"）。
            只高亮“希望人查看的内容”。暂缓的候选不要高亮，只用 ui_note 贴上原因。"""
            return await post({"units": units, "label": label})
        srv.tool(name="ui_highlight")(ui_highlight)
    elif action == "note":
        async def ui_note(layer: int, neuron: int, text: str) -> dict:
            """在 UI 地图的神经元上贴简短注记（最多 400 字）。同一神经元会覆盖，空 text 表示删除。
            地图只显示前 28 字，因此先写结论（例 "首选：外星人话题。sweep 第 3・超过全部对照，3 个提示词出现
            extraterrestrial" / "暂缓：数量词（many/most）"）。区分事实与解释。"""
            return await post({"layer": layer, "neuron": neuron, "text": text})
        srv.tool(name="ui_note")(ui_note)
    else:
        async def ui_open(layer: int, neuron: int, tab: str | None = None) -> dict:
            """在 UI 的 NEURON 面板中打开神经元，并将地图切换到该层的正面视图。tab：examples / chat / details / web。"""
            return await post({"layer": layer, "neuron": neuron, "tab": tab})
        srv.tool(name="ui_open")(ui_open)


def main(model: str, *, root: str | None = None, ui_url: str | None = None, n_gpu_layers: int = -1,
         n_ctx: int = 4096, seed: int = 0) -> None:
    srv = build_server(model, root=root, ui_url=ui_url, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed)
    srv.run("stdio")

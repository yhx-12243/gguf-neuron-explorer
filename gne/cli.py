"""gne CLI。设计文档 §7。gne.api 的薄封装。

通用选项（--model、--root、--n-gpu-layers、--n-ctx、--n-seq-max、--seed、-v、--json）
可以放在 `gne --model X steer ...` 和 `gne steer --model X ...` 任一位置。
使用 --json 时，stdout 只输出 JSON（§6.13 (a)）；供人阅读的显示走 stderr。
"""

from __future__ import annotations

import functools
import inspect
import json
import re
import sys
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from .schemas import SCHEMA_VERSION
from .steer import GneError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="检查并操控 GGUF 的 MLP 神经元")
stats_app = typer.Typer(no_args_is_help=True, help="统计（残差范数 ρ_L、神经元激活值）")
app.add_typer(stats_app, name="stats")
lens_app = typer.Typer(no_args_is_help=True, help="初筛（logit lens、steer lens）")
app.add_typer(lens_app, name="lens")
heads_app = typer.Typer(no_args_is_help=True, help="注意力头（对照分析、关注目标、档案，CHANGE-04）")
app.add_typer(heads_app, name="heads")
experts_app = typer.Typer(no_args_is_help=True, help="MoE 专家（对照分析、档案，CHANGE-05）")
app.add_typer(experts_app, name="experts")


@dataclass
class Common:
    model: Optional[Path] = None
    root: Optional[Path] = None
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    n_seq_max: int = 8
    seed: int = 0
    verbose: bool = False
    json: bool = False
    mmproj: Optional[Path] = None

    def merged(self, over: dict) -> Common:
        kw = {k: v for k, v in over.items() if v is not None and k not in ("verbose", "json")}
        return replace(self, **kw, verbose=self.verbose or bool(over.get("verbose")),
                       json=self.json or bool(over.get("json")))

    def session(self, model: Optional[Path] = None):
        from .api import Session
        path = model or self.model
        if path is None:
            raise GneError("BAD_ARGS", "请指定 GGUF", "--model 或环境变量 GNE_MODEL")
        return Session(path, root=self.root, n_gpu_layers=self.n_gpu_layers, n_ctx=self.n_ctx,
                       n_seq_max=self.n_seq_max, seed=self.seed, verbose=self.verbose, progress=self.progress,
                       mmproj=self.mmproj)

    def progress(self, done: int, total: int, stage: str) -> None:
        now = time.monotonic()
        if done != total and now - getattr(self, "_last_progress", 0.0) < 0.5:   # 0.5 秒ごとに間引く
            return
        self._last_progress = now
        if self.json:
            _emit({"progress": {"stage": stage, "done": done, "total": total}})
        else:
            print(f"\r{stage}: {done}/{total}", end="\n" if done == total else "", file=sys.stderr, flush=True)

    def human(self, text: str) -> None:
        print(text, file=sys.stderr if self.json else sys.stdout, flush=True)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
    sys.stdout.flush()


def _json_default(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return str(x)


def _common_params() -> list[inspect.Parameter]:
    """附加到各命令的通用选项。默认以 None 接收，再叠加到分组侧（gne --model X ...）的值上。"""
    K = inspect.Parameter.KEYWORD_ONLY
    return [
        inspect.Parameter("model", K, annotation=Optional[Path],
                          default=typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF 文件")),
        inspect.Parameter("root", K, annotation=Optional[Path],
                          default=typer.Option(None, "--root", envvar="GNE_ROOT", help="保存位置（默认 ~/.gne）")),
        inspect.Parameter("n_gpu_layers", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-gpu-layers", help="加载到 GPU 的层数（-1 = 全部）")),
        inspect.Parameter("n_ctx", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-ctx", help="上下文长度（序列间共享）")),
        inspect.Parameter("n_seq_max", K, annotation=Optional[int],
                          default=typer.Option(None, "--n-seq-max", help="单批最大序列数")),
        inspect.Parameter("seed", K, annotation=Optional[int], default=typer.Option(None, "--seed")),
        inspect.Parameter("mmproj", K, annotation=Optional[Path],
                          default=typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                               help="用于音频输入的 mmproj（默认使用模型同目录下的 mmproj*.gguf）")),
        inspect.Parameter("verbose", K, annotation=bool,
                          default=typer.Option(False, "--verbose", "-v", help="输出 llama.cpp 日志")),
        inspect.Parameter("json", K, annotation=bool,
                          default=typer.Option(False, "--json", help="stdout 只输出 JSON")),
    ]


_COMMON_NAMES = [p.name for p in _common_params()]


def gne_command(typer_app: typer.Typer, name: str, schema_name: str | None = None):
    """将主体函数 f(c: Common, holder: dict, ...) -> (payload, human_text, exit_code) 注册为命令。
    在 holder["session"] 中放入 Session 后，可用于信封的 model_id 和收尾处理。"""

    def deco(f: Callable):
        sig = inspect.signature(f, eval_str=True)
        own = list(sig.parameters.values())[2:]                       # c と holder を除く
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)

        @functools.wraps(f)
        def wrapper(ctx: typer.Context, **kwargs):
            over = {k: kwargs.pop(k) for k in _COMMON_NAMES}
            c = (ctx.obj or Common()).merged(over)
            _run(c, schema_name or name, f, kwargs)

        wrapper.__signature__ = sig.replace(parameters=[ctx_param, *own, *_common_params()])
        wrapper.__annotations__ = {}
        typer_app.command(name)(wrapper)
        return f

    return deco


def _envelope(command: str, c: Common, session=None) -> dict:
    if session is not None:
        return session.envelope(command)
    from .engine import Engine
    return {"schema_version": SCHEMA_VERSION, "command": command, "model_id": None,
            "llama_cpp_version": Engine.llama_cpp_version(), "seed": c.seed,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "warnings": []}


def _run(c: Common, command: str, f: Callable, kwargs: dict) -> None:
    holder: dict = {}
    code = 0
    try:
        payload, human, code = f(c, holder, **kwargs)
        s = holder.get("session")
        if c.json:
            _emit({**_envelope(command, c, s), **payload})
        else:
            if human:
                c.human(human)
            for w in (s.warnings if s else []):
                print(f"警告：{w}", file=sys.stderr)
    except GneError as e:
        _fail(c, command, holder.get("session"), e.code, e.message, e.hint)
        code = 1
    except Exception as e:  # noqa: BLE001
        if c.verbose:
            import traceback
            traceback.print_exc()
        _fail(c, command, holder.get("session"), "INTERNAL", f"{type(e).__name__}: {e}", None)
        code = 1
    finally:
        s = holder.get("session")
        if s is not None:
            s.close()
    if code:
        raise typer.Exit(code)


def _fail(c: Common, command: str, session, code: str, message: str, hint: str | None) -> None:
    if c.json:
        _emit({**_envelope(command, c, session), "error": {"code": code, "message": message, "hint": hint}})
    else:
        print(f"错误[{code}]：{message}" + (f"\n提示：{hint}" if hint else ""), file=sys.stderr)


@app.callback()
def main(ctx: typer.Context,
         model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF 文件"),
         root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT", help="保存位置（默认 ~/.gne）"),
         n_gpu_layers: int = typer.Option(-1, "--n-gpu-layers", help="加载到 GPU 的层数（-1 = 全部）"),
         n_ctx: int = typer.Option(4096, "--n-ctx", help="上下文长度（序列间共享）"),
         n_seq_max: int = typer.Option(8, "--n-seq-max", help="单批最大序列数"),
         seed: int = typer.Option(0, "--seed"),
         verbose: bool = typer.Option(False, "--verbose", "-v", help="输出 llama.cpp 日志"),
         json_: bool = typer.Option(False, "--json", help="stdout 只输出 JSON"),
         mmproj: Optional[Path] = typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                               help="用于音频输入的 mmproj（默认使用模型同目录下的 mmproj*.gguf）")):
    ctx.obj = Common(model, root, n_gpu_layers, n_ctx, n_seq_max, seed, verbose, json_, mmproj)


# --- 表示の補助 ------------------------------------------------------------------------------------------

def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def _table(rows: list[tuple[str, str]]) -> str:
    w = max(_width(k) for k, _ in rows)
    return "\n".join(f"{k}{' ' * (w - _width(k))}  {v}" for k, v in rows)


def parse_layers(spec: Optional[str]) -> Optional[list[int]]:
    """"all" / "10-16" / "1,5,9-12" → 层列表（all 为 None）。"""
    if spec is None or spec == "all":
        return None
    out: list[int] = []
    for part in spec.split(","):
        m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", part)
        if not m:
            raise GneError("BAD_ARGS", f"无法解析层指定 {spec!r}", "all / 10-16 / 1,5,9-12")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.extend(range(a, b + 1))
    return sorted(set(out))


def _units_text(units: list[dict]) -> str:
    rows = []
    for u in units:
        who = {"neuron": f"N{u['neuron']}", "random": f"random{u['seed']}"}.get(u["kind"], f"dir:{u.get('direction')}")
        if u.get("mode") == "gain":
            rows.append(f"  L{u['layer']} {who} ×{u['k']:g}（gain：输出放大 {u['k']:g} 倍，以 LoRA 应用）")
            continue
        rp = f"rel_push {u['rel_push']:.2f}、ρ={u['rho']:.2f}" if u.get("rel_push") is not None else "ρ 未知"
        add = f"、相当于激活值 {u['add']:+.3g}" if u.get("add") is not None else ""
        if u.get("diff_rel") is not None:
            add += f"、平均差的 {u['norm'] / (u['diff_rel'] * u['rho']):.2f} 倍"
        rows.append(f"  L{u['layer']} {who} dose {u['dose']:+g}: ‖v‖={u['norm']:.2f}({rp}{add})")
    return "\n".join(rows)


# --- info / probe / verify / bench ---------------------------------------------------------------------

@gne_command(app, "info")
def info(c: Common, holder: dict,
         load: bool = typer.Option(True, "--load/--no-load", help="确认能否由 llama.cpp 加载（R6）")):
    """GGUF 元数据、arch 判定、警告、dequantize 可行性、chat template、bake 可行性。"""
    s = holder["session"] = c.session()
    d = s.info(load=load)
    m = d["meta"]
    b = d["bake"]
    rows = [
        ("file", f"{m['path']} ({m['file_size'] / 2**30:.2f} GiB)"),
        ("model_id", m["model_id"]),
        ("name / arch", f"{m['name']} / {m['arch']}"),
        ("n_layer / n_embd / n_ff", f"{m['n_layer']} / {m['n_embd']} / {m['n_ff']}"
         + ("（各层不同）" if m["n_ff_per_layer"] and len(set(m["n_ff_per_layer"])) > 1 else "")),
        ("n_vocab / n_expert", f"{m['n_vocab']} / {m['n_expert']}"),
        ("ffn_down 量化类型", m["ffn_down_qtype"]),
        ("output", "共享 token_embd（tied）" if m["tied_output"] else "output.weight"),
        ("post_ffw_norm / ffn_down.scale",
         f"{'有' if m['has_post_ffn_norm'] else '无'} / {'有' if m['ffn_down_scale'] else '无'}"),
        ("chat template", "有" if m["has_chat_template"] else "无"),
        ("bake", (f"可（{b['mode']}）" if b["supported"] else "不可") + f"：{b['reason']}（表：{b['checked_llama_cpp']}）"),
        ("统计", f"resid {'有' if d['stats']['resid'] else '无'}、act 层 {d['stats']['act_layers'] or '无'}"),
    ]
    rw = d.get("residual_writers")
    if rw:
        rows.append(("残差写入器", ", ".join(f"{k} {n}" for k, n in rw["counts"].items())
                     + ("（已通过计算图确认）" if rw["verified"] else "（由张量名推测）")))
    lc = d["llama_cpp"]
    if lc is not None:
        rows.append(("llama.cpp", f"可加载（n_layer {lc['n_layer']}, n_embd {lc['n_embd']}, n_vocab {lc['n_vocab']}）"
                     if lc["ok"] else f"无法加载：{lc['error']}（可能需要用新版 llama.cpp 重新构建，R6）"))
    sp = d.get("speed")
    if sp:
        rows.append(("GPU / CPU 层数", f"GPU {sp['gpu_layers']} / CPU {sp['cpu_layers']}、预计生成 {sp['decode_tok_s']:.0f} tok/s（单条）"
                     + ("、来自 gne bench 实测" if sp["source"] == "bench" else "、来自默认系数")))
    text = _table(rows)
    text += "".join(f"\n警告：{w}" for w in d["arch_warnings"]) or "\n无警告：dense FFN，神经元操控与 control vector 等价"
    if d.get("workflow"):
        from .estimate import fmt_seconds
        text += "\n大型模型流程（全量 lens、sweep 开销很大，因此从反向查找开始缩小范围）：" + "".join(
            f"\n  {i}. {w['step']}（约 {fmt_seconds(w['seconds'])}）" for i, w in enumerate(d["workflow"], 1))
    if rw and rw["warnings"]:
        text += "".join(f"\n投影消除：{w}" for w in rw["warnings"])
    return d, text, 0


@gne_command(app, "probe")
def probe(c: Common, holder: dict,
          list_names: bool = typer.Option(False, "--list-names", help="列出 cb_eval 可观测的张量名"),
          layer: Optional[int] = typer.Option(None, "--layer", help="只显示该层的名称"),
          prompt: str = typer.Option("Who are you?", "--prompt")):
    """cb_eval 可观测的张量名（执行一次 prefill）。"""
    from .capture import pick_act_name
    if not list_names:
        raise GneError("BAD_ARGS", "请加上 --list-names")
    s = holder["session"] = c.session()
    e = s.engine
    names = e.list_names(e.tokenize_prompt(e.render_chat([{"role": "user", "content": prompt}])))
    groups: dict[str, set[int]] = {}
    other = []
    for n in names:
        mt = re.fullmatch(r"(.+)-(\d+)", n)
        if mt:
            groups.setdefault(mt.group(1), set()).add(int(mt.group(2)))
        elif not re.fullmatch(r"node_\d+", n) and " (" not in n:
            other.append(n)
    act_layer = layer if layer is not None else e.n_layer // 2
    act = pick_act_name(names, act_layer)
    shown = sorted(x for x in names if layer is None or x.endswith(f"-{layer}"))
    payload = {"layer": layer, "names": shown, "groups": {k: sorted(v) for k, v in sorted(groups.items())},
               "act_name": act}
    if layer is not None:
        return payload, "\n".join(shown) + f"\n→ ffn_down 输入：{act or '未找到'}", 0
    lines = [f"各层名称（层范围，共 {e.n_layer} 层）："]
    for base in sorted(groups):
        ls = sorted(groups[base])
        rng = f"{ls[0]}..{ls[-1]}" if ls == list(range(ls[0], ls[-1] + 1)) else ",".join(map(str, ls))
        lines.append(f"  {base:<24} {rng}")
    lines.append("不属于任何层的名称：" + ", ".join(sorted(other)))
    lines.append(f"→ ffn_down 输入（层 {act_layer}）：{act or '未找到'}")
    return payload, "\n".join(lines), 0


@gne_command(app, "verify")
def verify(c: Common, holder: dict,
           layer: Optional[int] = typer.Option(None, "--layer", help="要验证的层（默认 n_layer/2）"),
           llama_completion: Optional[str] = typer.Option(None, "--llama-completion",
                                                          help="用于 V4 生成比较的 llama-completion（默认：build/ 或 PATH）"),
           hf_model: Optional[str] = typer.Option(None, "--hf-model", help="V5：HF 格式的模型目录"),
           hf_python: Optional[str] = typer.Option(None, "--hf-python", help="V5：带 torch 和 transformers 的 python"),
           bench: bool = typer.Option(True, "--bench/--no-bench", help="同时运行 V7 基准测试"),
           markdown: Optional[Path] = typer.Option(None, "--markdown", help="将结果表保存为 Markdown")):
    """Phase 0 的等价性和层对应测试全套（V1～V8）。"""
    from dataclasses import asdict
    from .bench import bench_markdown, run_bench
    from .verify import VResult, Verifier, find_llama_completion
    s = holder["session"] = c.session()
    lc = llama_completion or find_llama_completion()
    v = Verifier(s.engine, s.weights, layer=layer, seed=c.seed)
    steps = [("V1", v.v1_shapes), ("V2", v.v2_ffn_linearity), ("V3", v.v3_cvec_layer),
             ("V4", lambda: v.v4_direction_file(lc)), ("V5", lambda: v.v5_neuron_push(hf_model, hf_python)),
             ("V6", v.v6_attention_sink), ("V11", v.v11_gain), ("V12", v.v12_when_gen)]
    small = s.meta.file_size < 2 * 2**30                         # bake を 2 つ作るので、大きいモデルでは LoRA だけ
    steps += [("V13", lambda: v.v13_ablation(bake=small)),
              ("V14", v.v14_lora_vs_bake if small else
               lambda: VResult("V14", "投影消除的 LoRA 与 bake 一致性", "skip",
                               "2 GiB 以上的模型会跳过（可用 gne ablate --bake --verify 确认）"))]
    results, bench_result = [], None

    def show(r):
        results.append(r)
        c.human(f"[{r.status.upper():4}] {r.id} {r.title}: {r.detail}")

    for vid, fn in steps:
        try:
            show(fn())
        except Exception as ex:  # noqa: BLE001
            show(VResult(vid, "异常", "fail", repr(ex)))
    if bench:
        bench_result = run_bench(s.engine, layer=v.L, reps=5)
        show(VResult("V7", "基准测试", "info",
                     f"set_cvec+prefill {bench_result['set_plus_prefill_ms']:.1f} ms（其中重新预留 {bench_result['reserve_ms']:.1f} ms）、"
                     f"prefill {bench_result['prefill_ms']:.1f} ms、decode 1 step {bench_result['decode_step_ms']:.1f} ms、"
                     f"生成 128 tok×4 {bench_result['gen']['batch']['ms']:.0f} ms", bench_result))
    else:
        show(VResult("V7", "基准测试", "skip", "--no-bench"))
    show(v.v8_tokenize())
    n_fail = sum(not r.ok for r in results)
    if markdown:
        md = ["| ID | 结果 | 内容 |", "|---|---|---|"] + [f"| {r.id} | {r.status} | {r.detail.replace('|', '/')} |"
                                                       for r in results]
        markdown.write_text("\n".join(md) + "\n" + ("\n" + bench_markdown(bench_result) if bench_result else ""))
    payload = {"layer": v.L, "results": [json.loads(json.dumps(asdict(r), default=_json_default)) for r in results],
               "n_pass": len(results) - n_fail, "n_fail": n_fail}
    text = (f"\n{s.meta.name}({s.meta.model_id})：{len(results) - n_fail}/{len(results)} 通过"
            + (f"、{n_fail} 项未通过" if n_fail else ""))
    return payload, text, 1 if n_fail else 0


@gne_command(app, "bench")
def bench(c: Common, holder: dict,
          layer: Optional[int] = typer.Option(None, "--layer"),
          reps: int = typer.Option(10, "--reps"),
          gen_tokens: int = typer.Option(128, "--gen-tokens"),
          markdown: Optional[Path] = typer.Option(None, "--markdown", help="将 Markdown 表格追加到该文件")):
    """实测 set_cvec / prefill / decode / 生成（用于 docs/measurements.md）。结果也会保存到 <root>/bench/<model_id>.json，
    并用于估算任务耗时。"""
    from .bench import bench_markdown, run_bench
    s = holder["session"] = c.session()
    r = run_bench(s.engine, layer=layer, reps=reps, gen_tokens=gen_tokens, seed=c.seed)
    bp = s.store.root / "bench" / f"{s.meta.model_id}.json"           # 見積もり(CHANGE-03 C)が実測を使う
    bp.parent.mkdir(parents=True, exist_ok=True)
    bp.write_text(json.dumps(r, default=_json_default, ensure_ascii=False, indent=1))
    md = bench_markdown(r)
    if markdown:
        with open(markdown, "a") as fh:
            fh.write(md + "\n")
    return {"bench": json.loads(json.dumps(r, default=_json_default))}, md, 0


# --- stats ------------------------------------------------------------------------------------------------

_CORPUS = typer.Option(None, "--corpus", help="文本文件（默认：configs/corpus/default.txt）")
_TOKENS = typer.Option(None, "--tokens", help="使用的 token 数上限（从整个语料中等间隔选取）")
_LAYERS = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12")
_LAST = typer.Option(False, "--include-last-layer", help="也包含最后一层（用 logits_all 输出全部行，R5）")
_CHUNK = typer.Option(512, "--chunk-tokens", help="单个序列的 token 数")


@gne_command(stats_app, "resid", "stats resid")
def stats_resid(c: Common, holder: dict, corpus: Optional[Path] = _CORPUS, tokens: Optional[int] = _TOKENS,
                layers: str = _LAYERS, include_last_layer: bool = _LAST, chunk_tokens: int = _CHUNK):
    """计算并保存残差范数 ρ_L（position ≥ 1 的 ‖l_out-L‖ 中位数等）。"""
    s = holder["session"] = c.session()
    d = s.stats_resid(corpus=corpus, tokens=tokens, layers=parse_layers(layers),
                      include_last_layer=include_last_layer, chunk_tokens=chunk_tokens)
    st = d["stats"]
    rows = [f"  L{L:<3} p10 {a:8.2f}  p50 {b:8.2f}  p90 {q:8.2f}"
            for L, a, b, q in zip(st["layers"], st["p10"], st["p50"], st["p90"], strict=True)]
    text = (f"{d['path']}（{st['n_tokens']} token/层、语料 {d['corpus']['sha']}、{d['seconds']:.1f} 秒）\n"
            + "\n".join(rows))
    return d, text, 0


@gne_command(stats_app, "act", "stats act")
def stats_act(c: Common, holder: dict, corpus: Optional[Path] = _CORPUS, tokens: Optional[int] = _TOKENS,
              layers: str = _LAYERS, include_last_layer: bool = _LAST, chunk_tokens: int = _CHUNK):
    """计算并保存神经元激活值统计（|a| 的 absmax、q50/q99/q999、RMS、均值）。"""
    s = holder["session"] = c.session()
    d = s.stats_act(corpus=corpus, tokens=tokens, layers=parse_layers(layers),
                    include_last_layer=include_last_layer, chunk_tokens=chunk_tokens)
    rows = [f"  L{r['layer']:<3} {r['tensor_name']:<18} absmax 中位数 {r['absmax_median']:.3g}  "
            f"q999 中位数 {r['q999_median']:.3g}  rms 中位数 {r['rms_median']:.3g}" for r in d["layers"]]
    text = f"已保存 {len(d['layers'])} 层（语料 {d['corpus']['sha']}、{d['seconds']:.1f} 秒）\n" + "\n".join(rows)
    return d, text, 0


# --- steer / export / import / bake --------------------------------------------------------------------

_RECIPE = typer.Option(None, "--recipe", help="配方 YAML")
_UNIT = typer.Option(None, "--unit", help="L:N:dose（例 14:8949:+1.0）、L:random:SEED:dose、gain 为 L:N:x2，方向为 L:dir:NAME:dose，注意力头为 L:Hh:xk（例 21:H7:x0）。可重复")
_WORKERS = typer.Option(None, "--workers", help="auto，或逗号分隔的名称：分发给 gne worker 并等待完成（CHANGE-03 B）")
_YES = typer.Option(False, "--yes", help="估算超过 30 分钟时仍继续运行")
_BUDGET = typer.Option(None, "--budget", help="时间预算（30m、2h、90s）。会缩小候选或窗口以适配预算")
_DOSE_UNIT = typer.Option(None, "--dose-unit", help="rel / absmax / raw（默认使用配方值；指定 --unit 时为 rel）")
_SCALE = typer.Option(1.0, "--scale", help="乘以所有单元 dose 的系数")
_ABLATE = typer.Option(None, "--ablate", help="方向投影消除（§6.17）：NAME、NAME@层、NAME:倍率（例 evil@10:0.5）。以 LoRA 应用")
_WHEN = typer.Option(None, "--when", help="施加位置（§6.15）：all / gen / assistant / after:<字符串> / ramp:<a>-<b>。默认使用配方值（没有则为 all）")
_TEMP = typer.Option(None, "--temperature", help="采样温度。指定任一采样值，或配方含 sampling 时启用采样"
                                                 "（缺少的值取配方 sampling，否则取 llama-server 默认值）。仅设为 0 时使用 greedy")
_TOP_K = typer.Option(None, "--top-k", help="采样 top-k（0 表示不截断）")
_TOP_P = typer.Option(None, "--top-p", help="采样 top-p")
_MIN_P = typer.Option(None, "--min-p", help="采样 min-p")
_PRES = typer.Option(None, "--presence-penalty", help="采样 presence penalty（从本次回复已出现 token 的 logit 中扣除）")
_SSEED = typer.Option(None, "--sample-seed", help="采样随机种子（默认每次都随机）")
_GREEDY = typer.Option(False, "--greedy", help="即使配方含 sampling 也使用 greedy 生成")


def _sampling(recipe, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed):
    """将 CLI 值与配方 sampling 转为 engine.Sampling（greedy 时为 None）。"""
    import random

    from .engine import Sampling
    from .steer import sampling_values
    given = {k: v for k, v in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p),
                               ("presence_penalty", presence_penalty)) if v is not None}
    if greedy:
        if given or sample_seed is not None:
            raise GneError("BAD_ARGS", "--greedy 不能与采样值同时使用")
        return None
    base = recipe.sampling if recipe is not None else None
    if not given and sample_seed is None and base is None:
        return None
    if set(given) == {"temperature"} and given["temperature"] <= 0:
        return None
    vals = sampling_values(given, base=base, seed=False)
    return Sampling(**vals, seed=sample_seed if sample_seed is not None else random.randrange(2 ** 31))


def _sampling_text(sp) -> str:
    if sp is None:
        return "greedy"
    return (f"采样（温度 {sp.temperature:g}、top-k {sp.top_k}、top-p {sp.top_p:g}、min-p {sp.min_p:g}、"
            f"presence {sp.presence_penalty:g}、seed {sp.seed}）")


@gne_command(app, "steer")
def steer(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
          dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
          prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="用户输入"),
          system: Optional[str] = typer.Option(None, "--system", help="system 提示词"),
          max_tokens: int = typer.Option(128, "--max-tokens"),
          compare: bool = typer.Option(False, "--compare", help="同时列出无干预生成（采样时使用相同 seed）"),
          when: Optional[str] = _WHEN, ablate: Optional[str] = _ABLATE,
          temperature: Optional[float] = _TEMP, top_k: Optional[int] = _TOP_K, top_p: Optional[float] = _TOP_P,
          min_p: Optional[float] = _MIN_P, presence_penalty: Optional[float] = _PRES,
          sample_seed: Optional[int] = _SSEED, greedy: bool = _GREEDY,
          contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集 YAML：用其 validate 测量效果，并与随机对照配方比较（null_pct）"),
          random_controls: int = typer.Option(16, "--random-controls", help="--contrast：对照配方（同等大小的随机方向）数量"),
          contrast_sample: bool = typer.Option(False, "--contrast-sample", help="--contrast：不用 greedy，而用配方采样（可由 --temperature 等覆盖）生成。无干预、配方和对照使用相同随机数"),
          contrast_seeds: int = typer.Option(4, "--contrast-seeds", help="--contrast-sample：每个提示词的数量"),
          contrast_tokens: int = typer.Option(64, "--contrast-tokens", help="--contrast：生成长度"),
          contrast_prompts: Optional[Path] = typer.Option(None, "--contrast-prompts", help="--contrast：替换验证提示词的 YAML（字符串列表；若为对照集，也可用 reactivity 的 with 或 user_prompts）"),
          audio: Optional[list[Path]] = typer.Option(None, "--audio", help="将音频文件（wav / mp3 / flac）作为用户发言。若指定 --prompt，会在音频后追加文字。需要 mmproj")):
    """用配方或单次指定回答提示词（默认 greedy，逐条生成）。"""
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale, when=when, ablate=ablate)
    sp = _sampling(r, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed)
    if prompt is None and contrast is None and not audio:
        raise GneError("BAD_ARGS", "需要 --prompt 或 --audio 或 --contrast")
    if prompt is not None or audio:
        d = s.steer(r, prompt=prompt, system=system, max_tokens=max_tokens, compare=compare, sampling=sp,
                    audio=audio)
    else:                                       # --contrast だけ: 生成はせず、対照集合での効果だけを測る
        vecs, details = s.vectors(r)
        d = {"messages": [], "recipe": r.to_dict(), "units": details, "layer_norms": s._norms(vecs), "when": r.when,
             "sampling": None, "baseline": None, "steered": None, "differs": None}
    if contrast is not None:
        csp = None
        if contrast_sample:                       # レシピを詰めたときの条件に合わせる。seed は固定(既定 0)で、同じ乱数で比べる
            from .engine import Sampling
            from .steer import SAMPLING_PRESETS, sampling_values
            given = {k: v for k, v in (("temperature", temperature), ("top_k", top_k), ("top_p", top_p), ("min_p", min_p),
                                       ("presence_penalty", presence_penalty)) if v is not None}
            vals = sampling_values(given, base=r.sampling or SAMPLING_PRESETS["common"], seed=False)
            csp = Sampling(**vals, seed=sample_seed if sample_seed is not None else 0)
        d["behavior"] = s.behavior_null(r, contrast, random_controls=random_controls, max_tokens=contrast_tokens,
                                        sampling=csp, seeds=contrast_seeds, prompts=contrast_prompts)
    parts = [f"单元（{r.dose_unit}、位置 {r.when}、{_sampling_text(sp)}）：\n" + (_units_text(d["units"]) or "  无")
             + (f"\n  {r.ablate.label()}（LoRA）" if r.ablate else "")]
    if d["baseline"] is not None:
        parts.append(f"── 无干预 ──（{d['baseline']['finish']}, {d['baseline']['n_gen']} tok）\n{d['baseline']['text']}")
    label = ", ".join([u.label() for u in r.units] + ([r.ablate.label()] if r.ablate else []))
    if d["steered"] is not None:
        parts.append(f"── 干预：{label} ──（{d['steered']['finish']}, {d['steered']['n_gen']} tok）\n{d['steered']['text']}")
    if d["differs"] is not None:
        parts.append("与无干预的差异：" + ("有" if d["differs"] else "无（相同列）"))
    b = d.get("behavior")
    if b is not None:
        cs = b["ctrl"]
        how = (f"采样（温度 {b['sampling']['temperature']:g}、seed {b['sampling']['seed']}）× {b['seeds']} 条"
               if b.get("sampling") else "greedy")
        line = (f"对照集上的效果（{b['mode']}、{b['n_prompts']} 个提示词 × {how} = {b.get('n_samples', b['n_prompts'])} 条、"
                f"{b.get('max_tokens', 64)} 个 token、位置 {b['when']}）：一致率 {b['base_rate']:.2f} → "
                f"{b['rate']:.2f}（效果 {b['effect']:+.2f}）\n  例：{b['sample'][:160]!r}")
        if b["random_controls"]:
            line += (f"\n  随机对照 {b['random_controls']} 条：效果 p50 {cs['p50']:+.2f}、p95 {cs['p95']:+.2f}、"
                     f"最大 {cs['max']:+.2f} → null_pct {b['null_pct']:.2f}"
                     + ("" if b["null_pct"] is not None and b["null_pct"] >= 0.95 else "（低于 0.95：无法与对照区分）"))
        else:
            line += "\n  无对照（既没有 add 单元也没有 ablate 的配方）"
        parts.append(line)
    return d, "\n\n".join(parts), 0


_EXT = {"cvec": ".cvec.gguf", "npz": ".npz", "json": ".json", "safetensors": ".safetensors"}


@gne_command(app, "export-cvec")
def export_cvec(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
                out: Optional[Path] = typer.Option(None, "--out", "-o", help="输出文件（默认 <root>/export/<配方名>.*）"),
                fmt: str = typer.Option("cvec", "--format", help="cvec / npz / json / safetensors"),
                verify: bool = typer.Option(False, "--verify", help="用 llama-completion --control-vector 比较 greedy 输出"),
                verify_prompt: str = typer.Option("Who are you?", "--verify-prompt"),
                llama_completion: Optional[str] = typer.Option(None, "--llama-completion")):
    """配方 → control vector GGUF（供 llama-cli / llama-server 的 --control-vector 使用）。也支持其他格式。"""
    if fmt not in _EXT:
        raise GneError("BAD_FORMAT", f"--format {fmt!r} 必须是 cvec / npz / json / safetensors 之一")
    s = holder["session"] = c.session()
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale)
    path = out or s.store.export_dir() / f"{r.name or 'units'}{_EXT[fmt]}"
    d = s.export_cvec(r, path, fmt=fmt, verify=verify, verify_prompt=verify_prompt, llama_completion=llama_completion)
    text = f"{d['path']}（{fmt}、层 {', '.join(d['layer_norms'])}）\n" + _units_text(d["units"])
    if d.get("llama_server_args"):
        text += f"\n配方采样设置（不会写入 cvec，需在 llama-server 侧附加）：{d['llama_server_args']}"
    v = d["verify"]
    if v:
        text += (f"\n上游工具复现（{Path(v['tool']).name}、greedy {v['n_tokens']} 个 token）："
                 f"{'一致' if v['match'] else '不一致'}、与无干预的差异：{'有' if v['differs_from_baseline'] else '无'}")
        if not v["match"]:
            text += f"\n  gne:   {v['ours']!r}\n  llama: {v['theirs']!r}"
    return d, text, 0 if not v or v["match"] else 1


@gne_command(app, "import-cvec")
def import_cvec(c: Common, holder: dict, file: Path = typer.Argument(..., help="control vector GGUF"),
                source: str = typer.Option("exact", "--source", help="exact / generator（修正 llama-cvector-generator 的 1 层偏移）"),
                name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用文件名）")):
    """将现有 control vector GGUF 导入为方向（供反向查找，<root>/reverse/<model_id>/）。"""
    if source not in ("exact", "generator"):
        raise GneError("BAD_ARGS", "--source 必须是 exact 或 generator")
    s = holder["session"] = c.session()
    d = s.import_cvec(file, source=source, name=name)
    text = f"{d['path']}(source={source})\n" + "\n".join(f"  L{L}: ‖v‖={n:.2f}" for L, n in d["layer_norms"].items())
    return d, text, 0


@gne_command(app, "bake")
def bake(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         dose_unit: Optional[str] = _DOSE_UNIT, scale: float = _SCALE,
         in_path: Optional[Path] = typer.Option(None, "--in", help="原始 GGUF（默认 --model）"),
         out: Path = typer.Option(..., "--out", help="烧制后 GGUF 的输出位置"),
         verify: bool = typer.Option(False, "--verify", help="比较烧制 GGUF 与 原模型 + cvec 的 greedy 输出（V9）"),
         verify_prompt: str = typer.Option("Who are you?", "--verify-prompt")):
    """将配方烧制为 FFN 偏置（blk.L.ffn_down.bias）的 GGUF（仅支持的 arch）。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in 与 --model 指向不同文件")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, scale=scale)
    d = s.bake(r, out, verify=verify, verify_prompt=verify_prompt)
    rep, v = d["report"], d["verify"]
    text = (f"{rep['dst']}（{rep['arch']}、mode={rep['mode']}、新增 {rep['added'] or '无'}、累加 {rep['modified'] or '无'}）\n"
            + _units_text(d["units"]))
    ok = True
    if v:
        verdict = "完全一致" if v["exact"] else ("通过（仅有细微分叉）" if v["match"] else "不一致")
        text += (f"\n验证（greedy {v['n_tokens']} 个 token）：与原模型 + cvec {verdict}"
                 + (f"、{v['summary']}、下一 token KL {v['kl_next']:.2e}" if v["kl_next"] is not None else "")
                 + f"、与无干预的差异：{'有' if v['differs_from_baseline'] else '无'}")
        ok = v["differs_from_baseline"] and (v["match"] or not v["ours"])
    return d, text, 0 if ok else 1


@gne_command(app, "gain")
def gain(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         scale: float = _SCALE,
         out: Optional[Path] = typer.Option(None, "--out", "-o", help="输出（默认 <root>/export/<配方名>.lora.gguf；--bake 时为 -gain.gguf）"),
         bake: bool = typer.Option(False, "--bake", help="不生成 LoRA，而生成改写 W_up 的 GGUF"),
         in_path: Optional[Path] = typer.Option(None, "--in", help="--bake 的原始 GGUF（默认 --model）"),
         requant: str = typer.Option("q8_0", "--requant", help="--bake 改写后的 ffn_up / W_O 类型：q8_0 / f16")):
    """将 gain 单元（如 L:N:x2，把输出放大 k 倍；注意力头为 L:Hh:xk）导出为 LoRA 适配器 GGUF 或改写 GGUF（§6.16）。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in 与 --model 指向不同文件")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, scale=scale)
    name = r.name or "gain"
    out = out or s.store.export_dir() / (f"{name}-gain.gguf" if bake else f"{name}.lora.gguf")
    d = s.gain(r, out, bake=bake, requant=requant)
    if bake:
        text = f"{d['path']}（{d['bytes'] / 1e6:.1f} MB）、改写：" + ", ".join(
            f"{m['tensor']}({m['from']} → {m['to']})" for m in d["modified"])
        if d.get("method") == "patch":
            text += f"\n缩放补丁（{d['patch_bytes'] / 1e3:.1f} KB，量化保持不变）。应用的操作位于 {d['path']}.patch.json"
    else:
        text = (f"{d['path']}({d['bytes'] / 1e6:.2f} MB、LoRA)\n"
                + "\n".join(f"  L{L}: " + ", ".join(f"N{int(n)} ×{k:g}" for n, k in nk) for L, nk in (d.get("layers") or {}).items())
                + "".join(f"\n  L{L}: " + ", ".join(f"H{int(h)} ×{k:g}" for h, k in hk) for L, hk in (d.get("heads") or {}).items())
                + "".join(f"\n  query 操作：{p}" for p in (d.get("query_loras") or []))
                + "\n用法：llama-cli -m <model> --lora " + d["path"])
    return d, text, 0


def _doses(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"--doses {text!r} 必须是逗号分隔的数值（例 0,0.5,1）") from e


@gne_command(app, "eval")
def eval_cmd(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
             dose_unit: Optional[str] = _DOSE_UNIT,
             doses: str = typer.Option("0,0.25,0.5,0.75,1", "--doses", help="作用于整个配方的倍率（必定包含 0 = 无干预）"),
             suite: Optional[Path] = typer.Option(None, "--suite", help="任务集 YAML（默认 configs/eval/default.yaml）"),
             when: Optional[str] = _WHEN,
             judge_model: Optional[Path] = typer.Option(None, "--judge-model", help="用于人格评分的另一个 GGUF（可选）"),
             judge_target: Optional[str] = typer.Option(None, "--judge-target", help="要评分的目的人格（例 'a yandere anime girl'）"),
             name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用配方名）"),
             ablate: Optional[str] = _ABLATE,
             compare_ablation: Optional[str] = typer.Option(None, "--compare-ablation",
                                                            help="并列比较方向 NAME 的“消除上位神经元（gain ×0）”与“消除方向”（§6.17）"),
             top_neurons: int = typer.Option(20, "--top-neurons", help="--compare-ablation 消除的神经元数量"),
             from_layer: Optional[int] = typer.Option(None, "--from-layer", help="--compare-ablation：消除在哪一层测得的 r"),
             random_controls: int = typer.Option(0, "--random-controls", help="在每个倍率下也测量 N 个同等大小的随机对照配方（0 = 无）"),
             reactivity: Optional[Path] = typer.Option(None, "--reactivity", help="同时测量反应性的对照集 YAML（reactivity 对和 validate.match）"),
             head_controls: int = typer.Option(0, "--head-controls", help="--reactivity：将注意力头替换为相同 k 的随机头的对照数量"),
             yes: bool = _YES):
    """副作用检查：逐倍率测量能力（exact / format）、ppl、语言混入、人格变化量，并给出可发布的倍率（§6.14）。"""
    s = holder["session"] = c.session()
    if compare_ablation:
        if unit or recipe or ablate:
            raise GneError("BAD_ARGS", "--compare-ablation 不能与配方（--recipe / --unit / --ablate）一起使用")
        d = s.eval_compare_ablation(compare_ablation, top=top_neurons, from_layer=from_layer, doses=_doses(doses),
                                    suite=suite, when=when)
        lines = [f"消除比较 {d['direction']}（r 来自 L{d['from_layer']}、上位 {d['top']} 个神经元 vs 方向、{d['seconds']:.0f} 秒）→ {d['path']}",
                 f"行为判定：{d['match'] or '—'}（无干预一致率 {format(d['base_rate'], '.2f') if d['base_rate'] is not None else '—'}）",
                 f"  {'方法':<9} {'倍率':>5} {'一致率':>6} {'下降':>6} {'KL(B)':>7} {'能力下降':>8} {'ppl恶化':>8} {'混入':>6}"]
        for r_ in d["rows"]:
            f2 = lambda x, fmt: "—" if x is None else format(x, fmt)  # noqa: E731
            lines.append(f"  {('神经元' if r_['method'] == 'neurons' else '方向'):<9} {r_['dose']:>5g} {f2(r_['rate'], '.2f'):>6} "
                         f"{f2(r_['rate_drop'], '+.2f'):>6} {f2(r_['kl_b'], '.3f'):>7} {r_['capability_drop_pt']:>7.1f}pt "
                         f"{r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}%" + ("  可发布" if r_["shippable"] else ""))
        lines.append("已消除神经元：" + ", ".join(f"L{x['layer']} N{x['neuron']}({x['cos']:+.2f})" for x in d["neurons"][:10])
                     + (" …" if len(d["neurons"]) > 10 else ""))
        return d, "\n".join(lines), 0
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate)
    _gate(s, "eval", {"doses": _doses(doses), "suite": suite, "when": when, "random_controls": random_controls}, yes, None)
    d = s.eval(r, doses=_doses(doses), suite=suite, when=when, judge_model=judge_model, judge_target=judge_target,
               name=name, random_controls=random_controls, reactivity=reactivity, head_controls=head_controls)
    th = d["suite"]["thresholds"]
    nc = d.get("random_controls") or 0
    head = f"{'倍率':>5} {'能力':>6} {'下降pt':>6} {'ppl':>8} {'恶化%':>7} {'混入':>6} {'新词':>5} {'d4':>5} {'截断':>4}"
    if d["judge"]:
        head += f" {'judge':>5}"
    if nc:
        head += f" │ {'对照新词p95':>10} {'null':>5} {'对照下降p50':>10} {'对照恶化p50':>10}"
    rcd = d.get("reactivity")
    if rcd:
        head += f" │ {'有':>5} {'无':>5} {'反应性':>6}" + (f" {'对照p95':>7} {'null':>5}" if rcd["head_controls"] else "")
    lines = [head]
    for row in d["rows"]:
        pm = row["persona"]
        line = (f"{row['dose']:>5g} {row['capability'] * 100:>5.1f}% {row['capability_drop_pt']:>6.1f} "
                f"{row['ppl_mean']:>8.2f} {row['ppl_worse_pct']:>+6.1f}% {(row['mix_zh'] + row['mix_other']) * 100:>5.2f}% "
                f"{pm['novelty']:>5.2f} {pm['distinct4']:>5.2f} {pm['capped'] * 100:>3.0f}%")
        if d["judge"]:
            line += f" {row['judge'] if row['judge'] is not None else float('nan'):>5.2f}"
        if nc:
            if row.get("ctrl"):
                f2 = lambda x, fmt: "—" if x is None else format(x, fmt)  # noqa: E731
                line += (f" │ {f2(row['ctrl_novelty_p95'], '.2f'):>10} {f2(row['novelty_null_pct'], '.2f'):>5} "
                         f"{f2(row['ctrl_capability_drop_p50'], '.1f'):>10} {f2(row['ctrl_ppl_worse_p50'], '+.1f'):>9}%")
            else:
                line += f" │ {'':>10} {'':>5} {'':>10} {'':>10}"
        if rcd:
            x = row.get("reactivity") or {}
            line += f" │ {x.get('with', 0):>5.2f} {x.get('without', 0):>5.2f} {x.get('reactivity', 0):>+6.2f}"
            if rcd["head_controls"]:
                cx = x.get("ctrl")
                line += (f" {cx['p95']:>+7.2f} {x['null_pct'] * 100:>4.0f}%" if cx and cx.get("p95") is not None
                         else f" {'':>7} {'':>5}")
        lines.append(line + ("  可发布" if row["shippable"] else "  " + "、".join(k for k, ok in row["ok"].items() if not ok) + " ×"))
    ok = [r_["dose"] for r_ in d["rows"] if r_["shippable"] and r_["dose"] > 0]
    verdict = (f"可发布的最大倍率：{max(ok):g}" if ok else "没有可发布的倍率（0 除外）")
    text = (f"eval {d['name']}（when={d['when']}、{d['suite']['n_tasks']} 个任务、{d['seconds']:.0f} 秒）→ {d['path']}\n"
            + "\n".join(lines) + f"\n标准：能力下降不超过 {th['capability_drop_pt']:g}pt、ppl 上升不超过 {th['ppl_worse_pct']:g}%、混入不增加。"
            + verdict)
    if nc:
        text += (f"\n对照：每个倍率使用 {nc} 个同等大小的随机方向配方。如果新词超过对照 p95（null ≥ 0.95），"
                 "且能力、ppl 的恶化与对照相当，则恶化来自幅度而非方向")
    elif random_controls:
        text += "\n无法创建对照（仅含 gain 的配方没有方向）"
    if rcd:
        text += (f"\n反应性（{rcd['name']}、{rcd['n_pairs']} 对、{rcd['max_tokens']} 个 token）：有/无触发词时判定的一致率及其差。"
                 + (f"对照为相同 k 的随机注意力头 {rcd['head_controls']} 个" if rcd["head_controls"] else "")
                 + ("" if rcd["head_controls"] or not head_controls else "配方不含注意力头，因此无法创建注意力头对照"))
    top = d["rows"][-1]["persona"]["top_words"][:8]
    if top:
        text += f"\n倍率 {d['rows'][-1]['dose']:g} 下在多个提示词中新出现的词：" + ", ".join(f"{w}({n})" for w, n in top)
    return d, text, 0


@gne_command(app, "ablate")
def ablate(c: Common, holder: dict,
           direction: str = typer.Option(..., "--direction", help="用 gne direction 创建的方向名称（也可用反向查找方向）"),
           from_layer: Optional[int] = typer.Option(None, "--from-layer", help="使用在哪一层测得的 r（默认使用 --select 选中的层）"),
           per_layer: bool = typer.Option(False, "--per-layer", help="逐层消除该层测得的 r_L"),
           layers: str = typer.Option("all", "--layers", help="消除层：all / 10-40 / 1,5,9-12"),
           scale: float = typer.Option(1.0, "--scale", help="消除比例（1 为全部，0.5 为一半）"),
           out: Optional[Path] = typer.Option(None, "--out", "-o", help="输出（默认 <root>/export/<方向>-ablate.lora.gguf；--bake 时为 -ablate.gguf）"),
           bake: bool = typer.Option(False, "--bake", help="不生成 LoRA，而生成改写权重的 GGUF"),
           in_path: Optional[Path] = typer.Option(None, "--in", help="--bake 的原始 GGUF（默认 --model）"),
           requant: str = typer.Option("q8_0", "--requant", help="--bake 改写后张量的类型：q8_0 / f16"),
           via_f16: bool = typer.Option(False, "--via-f16", help="--bake：先写为 F16，再用 llama-quantize 恢复原类型"),
           imatrix: Optional[Path] = typer.Option(None, "--imatrix", help="--via-f16：IQ1～IQ2 系模型所需的 imatrix"),
           llama_quantize: Optional[str] = typer.Option(None, "--llama-quantize", help="--via-f16 使用的 llama-quantize 路径"),
           skip_embed: bool = typer.Option(False, "--skip-embed", help="不投影嵌入（token_embd）"),
           verify: bool = typer.Option(False, "--verify", help="检查导出结果的 V13（r 分量）；bake 时还检查 V14（LoRA 与 greedy 一致性）")):
    """将方向 r 的投影消除（abliteration）导出为 LoRA 适配器 GGUF 或改写权重的 GGUF（§6.17）。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in 与 --model 指向不同文件")
    if via_f16 and not bake:
        raise GneError("BAD_ARGS", "--via-f16 需与 --bake 一起使用")
    s = holder["session"] = c.session(in_path)
    out = out or s.store.export_dir() / (f"{direction}-ablate.gguf" if bake else f"{direction}-ablate.lora.gguf")
    d = s.ablate(direction, out, from_layer=from_layer, per_layer=per_layer, layers=parse_layers(layers), scale=scale,
                 embed=not skip_embed, bake=bake, requant=requant, via_f16=via_f16,
                 imatrix=str(imatrix) if imatrix else None, llama_quantize=llama_quantize, verify=verify)
    src = "各层 r_L" if d.get("per_layer") else f"在 L{d['from_layer']} 测得的 r"
    Ls = d["layers"]
    head = (f"{d['path']}（{d['bytes'] / 2**20:.1f} MiB、{'LoRA' if d['kind'] == 'lora' else 'bake ' + d['requant']}、"
            f"{src}、层 {Ls[0]}～{Ls[-1]}（{len(Ls)} 层）、scale {d['scale']:g}、嵌入 {'有' if d['embed'] else '无'}、"
            f"{d['seconds']:.0f} 秒）")
    lines = [head]
    if d["kind"] == "lora":
        lines.append(f"用法：llama-cli -m <model> --lora {d['path']}（用 --lora-scaled {d['path']} 0.5 只消除一半）")
    ok = True
    v = d["verify"]
    if v and "v13" in v:
        t = v["v13"]
        lines.append(f"V13：|r·l_out − 期望值| / ‖l_out‖ 最大 {t['max_dev']:.2e}"
                     + (f"（{'通过' if t['pass'] else '未通过'}）" if t["pass"] is not None else f"（{t['note']}）"))
        ok &= t["pass"] is not False
    if v and "v14" in v:
        t = v["v14"]
        lines.append(f"V14：LoRA 与 bake 的 greedy {t['n_tokens']} 个 token：" + "、".join(t["summary"])
                     + f"（{'通过' if t['pass'] else '未通过'}）" + (f"（{t['note']}）" if t["note"] else ""))
        ok &= t["pass"]
    return d, "\n".join(lines), 0 if ok else 1


# --- lens / reverse(Phase 2) -----------------------------------------------------------------------------

_NEURONS = typer.Option("all", "--neurons", help="all / random:384 / 1,5,9-12")
_SHOW = typer.Option(20, "--show", help="显示的神经元数量（all / random 时从高到低）")


def _toks(ts: list[str], n: int = 8) -> str:
    return " ".join(repr(t) for t in ts[:n])


@gne_command(lens_app, "logit", "lens logit")
def lens_logit(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), neurons: str = _NEURONS,
               top_k: int = typer.Option(12, "--top-k"), show: int = _SHOW):
    """对整层（或指定）神经元执行 logit lens（仅权重，无推理）。"""
    s = holder["session"] = c.session()
    d = s.lens_logit(layer, neurons=neurons, top_k=top_k, show=show)
    lines = [f"{d['path']}（{d['n_neurons']} 个神经元、{d['seconds']:.1f} 秒）"]
    for r in d["neurons"]:
        lines.append(f"  N{r['neuron']:<6} peak z {r['peak_z']:5.1f}  ↑ {_toks(r['up'])}\n{'':16}↓ {_toks(r['down'])}")
    return d, "\n".join(lines), 0


@gne_command(lens_app, "steer", "lens steer")
def lens_steer(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), neurons: str = _NEURONS,
               dose: float = typer.Option(0.8, "--dose", help="rel dose"),
               random_controls: int = typer.Option(64, "--random-controls"),
               top_k: int = typer.Option(12, "--top-k"), show: int = _SHOW,
               resume: bool = typer.Option(True, "--resume/--no-resume", help="从已保存的结果继续"),
               direction: Optional[str] = typer.Option(None, "--direction", help="--budget：按写入方向与该方向的接近程度排序选择（默认使用最后创建的方向）"),
               yes: bool = _YES, budget: Optional[str] = _BUDGET, workers: Optional[str] = _WORKERS):
    """对整层（或指定）神经元执行 steer lens（±dose 操控时下一 token 的变化，含随机对照）。"""
    s = holder["session"] = c.session()
    g = _gate(s, "lens_steer", {"layer": layer, "neurons": neurons, "random_controls": random_controls,
                                **({"direction": direction} if direction else {})}, yes, budget)
    if workers:
        d = _via_workers(s, "lens_steer", {"layer": layer, "neurons": g["neurons"], "dose": dose,
                                           "random_controls": random_controls, "top_k": top_k, "resume": resume,
                                           "show": show}, workers)
        text = d.pop("text")
        return d, text, 0
    d = s.lens_steer(layer, neurons=g["neurons"], dose=dose, random_controls=random_controls, top_k=top_k,
                     resume=resume, show=show)
    nl = d["null"]
    lines = [f"{d['path']}（已完成 {d['n_done']}/{s.meta.n_ff_at(layer)}、对照 {d['n_controls']}、{d['seconds']:.1f} 秒）",
             f"对照 KL(+)：中位数 {nl['pos']['p50'] or 0:.3f} / 95% {nl['pos']['p95'] or 0:.3f} / 最大 {nl['pos']['max'] or 0:.3f}"]
    for r in d["neurons"]:
        lines.append(f"  N{r['neuron']:<6} KL+ {r['kl_pos']:.3f}({r['pct_pos']:.0%})  KL- {r['kl_neg']:.3f}({r['pct_neg']:.0%})"
                     f"\n{'':10}+ {_toks(r['up'])}\n{'':10}- {_toks(r['down'])}")
    return d, "\n".join(lines), 0


@gne_command(app, "reverse")
def reverse(c: Common, holder: dict,
            contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照提示词集 YAML（configs/contrast/）"),
            directions: Optional[str] = typer.Option(None, "--directions", help="已保存的方向（名称或路径，也可用 import-cvec 的结果）"),
            method: str = typer.Option("mean_diff", "--method", help="mean_diff / pca1"),
            positions: str = typer.Option("all", "--positions", help="all / last / assistant"),
            top: int = typer.Option(200, "--top", help="候选数量"),
            layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
            verify: bool = typer.Option(False, "--verify", help="对高位候选运行 quick 设计的 rollout（含对照）"),
            verify_top: Optional[int] = typer.Option(None, "--verify-top", help="--verify 运行的候选数量（默认 --top）"),
            verify_controls: int = typer.Option(8, "--verify-controls", help="--verify 每层的随机对照数量"),
            max_tokens: int = typer.Option(128, "--max-tokens"),
            name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用对照提示词集的 name）"),
            standardize: bool = typer.Option(True, "--standardize/--no-standardize",
                                             help="在按残差各维方差归一化的空间中计算 cos（F17）"),
            show: int = typer.Option(20, "--show"),
            yes: bool = _YES, budget: Optional[str] = _BUDGET):
    """创建对照方向，并按 cos 对全部层、全部神经元排序（反向查找）。"""
    if method not in ("mean_diff", "pca1") or positions not in ("all", "last", "assistant"):
        raise GneError("BAD_ARGS", "--method 为 mean_diff / pca1，--positions 为 all / last / assistant")
    s = holder["session"] = c.session()
    vb = None
    if verify:
        g = _gate(s, "reverse", {"verify": True, "top": top, "verify_top": verify_top, "verify_controls": verify_controls,
                                 "max_tokens": max_tokens}, yes, budget)
        vb = g.get("verify_budget_s")
    d = s.reverse(contrast=contrast, directions=directions, method=method, positions=positions, top=top,
                  layers=parse_layers(layers), verify=verify, verify_top=verify_top, verify_controls=verify_controls,
                  max_tokens=max_tokens, name=name, standardize=standardize, verify_budget_s=vb)
    lines = [f"{d['candidates_path']}（{d['n_candidates']} 个候选、{d['seconds']:.1f} 秒）",
             "各层 |cos| 前 1% 的平均：" + " ".join(f"L{r['layer']}:{r['top1pct_abs_cos']:.3f}" for r in d["layers"])]
    for i, cd in enumerate(d["candidates"][:show]):
        extra = "".join(f"  {k} {cd[k]:+.3g}" for k in ("act_diff", "steer_kl") if cd[k] is not None)
        who = f"E{cd['expert']} N{cd['expert_neuron']}" if cd.get("expert") is not None else f"N{cd['neuron']}"
        lines.append(f"  {i + 1:>3}. L{cd['layer']} {who}  cos {cd['cos']:+.3f}（推动方向 {'+' if cd['sign'] > 0 else '-'}）{extra}")
    if d["verify"]:
        lines.append(f"rollout({d['verify']['design']}): " + ", ".join(r["path"] for r in d["verify"]["runs"]))
        for ex in d["verify"]["examples"][:min(show, 10)]:
            lines.append(f"  L{ex['layer']} N{ex['neuron']} ({'+' if ex['sign'] > 0 else '-'}) {ex['prompt']}: "
                         f"{ex['text'][:160]!r}")
    return d, "\n".join(lines), 0


@gne_command(app, "direction")
def direction(c: Common, holder: dict,
              contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML（与 reverse 相同格式；选层使用 validate.match）"),
              name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用对照集的 name）"),
              positions: str = typer.Option("last", "--positions", help="last（生成提示词的最后一个 token）/ all / assistant / response（用 A、B 的 system 生成回复，并用回复 token 测量）"),
              response_tokens: int = typer.Option(96, "--response-tokens", help="--positions response 时生成的回复长度"),
              select: Optional[str] = typer.Option(None, "--select", help="auto（逐候选层测量效果后选择）或层号"),
              layer: Optional[int] = typer.Option(None, "--layer", help="手动指定层（等同于 --select <编号>）"),
              mode: str = typer.Option("ablate", "--mode", help="--select auto 的测量方式：ablate（消除后效果是否下降）/ add（加入后效果是否上升）"),
              layers: Optional[str] = typer.Option(None, "--layers", help="候选层（默认 0.3～0.8·n_layer。10-24、10-24:2、12,16）"),
              dose: float = typer.Option(1.0, "--dose", help="mode=add 时加入的幅度（平均差 mean_A − mean_B 的倍数）"),
              max_tokens: int = typer.Option(64, "--max-tokens"),
              kl_max: float = typer.Option(0.1, "--kl-max", help="mode=ablate：B 侧下一 token KL 的上限"),
              min_distinct: float = typer.Option(0.8, "--min-distinct", help="mode=add：distinct-4 的下限（检查是否重复）"),
              per_layer: bool = typer.Option(False, "--per-layer", help="mode=ablate：同时测量将各层 r_L 应用于对应层的变体"),
              judge_model: Optional[Path] = typer.Option(None, "--judge-model", help="用于判定的另一个 GGUF（替代 validate.match）"),
              judge_target: Optional[str] = typer.Option(None, "--judge-target", help="传给 judge 的目的人格（默认 validate.target 或 system_a）"),
              random_controls: int = typer.Option(16, "--random-controls", help="--select auto：每层同等大小的随机方向对照数量（0 为无对照）"),
              yes: bool = _YES):
    """从对照集创建并保存全层方向 r_L（mean_A − mean_B），再用 --select auto 选择效果较大的层（§6.17）。
    创建的方向可用 L:dir:NAME:dose 单元加入（gne steer / chat / eval / export-cvec）。"""
    if mode not in ("ablate", "add"):
        raise GneError("BAD_ARGS", "--mode 必须是 ablate 或 add")
    if layer is not None:
        if select not in (None, str(layer)):
            raise GneError("BAD_ARGS", "--layer 与 --select 只能指定一个")
        select = str(layer)
    s = holder["session"] = c.session()
    _gate(s, "direction", {"contrast": str(contrast), "positions": positions, "select": select, "mode": mode,
                           "layers": layers, "max_tokens": max_tokens, "response_tokens": response_tokens,
                           "random_controls": random_controls}, yes, None)
    d = s.direction(contrast=contrast, name=name, positions=positions, select=select, mode=mode, layers=layers,
                    dose=dose, max_tokens=max_tokens, kl_max=kl_max, min_distinct=min_distinct, per_layer=per_layer,
                    judge_model=judge_model, judge_target=judge_target, response_tokens=response_tokens,
                    random_controls=random_controls)
    lines = [f"{d['path']}（A/B 各 {d['n_prompts']} 个会话、行数 {d['n_rows']}、位置 {d['positions']}、{d['seconds']:.0f} 秒）",
             "各层 ‖平均差‖/ρ（换算为 rel dose 的幅度）及与上一层的 cos："]
    lines.append("  " + " ".join(f"L{r['layer']}:{r['rel']:.2f}" if r["rel"] is not None else f"L{r['layer']}:—"
                                 for r in d["layers"]))
    lines.append("  " + " ".join(f"L{r['layer']}:{r['cos_prev']:+.2f}" for r in d["layers"] if r["cos_prev"] is not None))
    sel = d["select"]
    if sel and sel["mode"] == "manual":
        lines.append(f"已选择层 {sel['selected']}（手动指定）")
    elif sel:
        key = sel["metric"]
        b = sel["base"]
        head = (f"选层（mode={sel['mode']}、{sel['n_prompts']} 个提示词、{sel['max_tokens']} 个 token"
                + (f"、dose {sel['dose']:g}" if sel["dose"] is not None else "") + f"、{sel['seconds']:.0f} 秒）")
        lines += [head, f"  无干预：{key} {b[key]:.2f}  distinct-4 {b['distinct4']:.2f}",
                  f"  {'层':>4} {'效果':>6} {key:>6} {'KL':>7} {'d4':>5}" + (f" {'rel':>5}" if sel["mode"] == "add" else "")
                  + (f" {'对照p95':>7} {'null':>5}" if sel.get("random_controls") else "")]
        for r in sel["rows"] + ([{**sel["per_layer"], "layer": "逐层"}] if sel["per_layer"] else []):
            mark = "  ← 选择" if r["layer"] == sel["selected"] else ("" if r["ok"] else "  ×")
            eff = "—" if r["effect"] is None else f"{r['effect']:+.2f}"
            val = "—" if r.get(key) is None else f"{r[key]:.2f}"
            rel = f" {r['rel_dose']:>5.2f}" if r.get("rel_dose") is not None else ""
            ctl = ""
            if sel.get("random_controls") and r.get("ctrl_effect_p95") is not None:
                ctl = f" {r['ctrl_effect_p95']:>+7.2f} {r['null_pct'] * 100:>4.0f}%"
            elif sel.get("random_controls"):
                ctl = f" {'—':>7} {'—':>5}"
            lines.append(f"  {r['layer']!s:>4} {eff:>6} {val:>6} {r['kl']:>7.3f} {r['distinct4']:>5.2f}{rel}{ctl}{mark}")
        crit = sel["criteria"]
        lines.append("标准：" + (f"KL ≤ {crit['kl_max']:g}" if "kl_max" in crit else f"distinct-4 ≥ {crit['min_distinct']:g}")
                     + (f"、效果 > 对照（同等大小的随机方向 {sel['random_controls']} 条）的 p95"
                        + ("。不满足标准的层、效果不高于 0 的层（—）未运行对照" if sel["mode"] == "add" else "")
                        if sel.get("random_controls") else "")
                     + ("。没有可选择的层" if sel["selected"] is None else ""))
        if sel["selected"] is not None and sel["mode"] == "add":
            rd = next(r["rel_dose"] for r in sel["rows"] if r["layer"] == sel["selected"])
            lines.append(f"试一试（平均差的 {sel['dose']:g} 倍 = rel {rd:.2f}）："
                         f"gne steer --unit {sel['selected']}:dir:{d['name']}:+{rd:.2f} -p \"Who are you?\" --compare")
    return d, "\n".join(lines), 0


# --- sweep / rank(Phase 3) --------------------------------------------------------------------------------

@gne_command(app, "sweep")
def sweep(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"),
          neurons: str = typer.Option("random:384", "--neurons", help="all / random:384 / 1,5,9-12（可用逗号混合）"),
          design: str = typer.Option("main_rel", "--design", help="configs/designs.yaml 中的名称（main_rel / quick / calib）"),
          random_controls: int = typer.Option(64, "--random-controls"),
          max_tokens: int = typer.Option(128, "--max-tokens"),
          top_k: int = typer.Option(5, "--top-k", help="每步记录的候选数量"),
          run: Optional[str] = typer.Option(None, "--run", help="run 名称（默认 L<层>-<设计>）"),
          prompt_set: str = typer.Option("en", "--prompt-set", help="configs/prompts/default.yaml 的组（en / ja）"),
          dose_unit: str = typer.Option("rel", "--dose-unit"),
          resume: bool = typer.Option(True, "--resume/--no-resume", help="从输出 JSONL 的后续位置继续"),
          yes: bool = _YES, budget: Optional[str] = _BUDGET, workers: Optional[str] = _WORKERS):
    """操控神经元（及随机对照）生成文本，并将 rollout 写入 JSONL（穷举）。"""
    s = holder["session"] = c.session()
    g = _gate(s, "sweep", {"layer": layer, "neurons": neurons, "design": design, "random_controls": random_controls,
                           "max_tokens": max_tokens, "prompt_set": prompt_set}, yes, budget)
    if workers:
        name = run or f"L{layer}-{g['design']}"
        d = _via_workers(s, "sweep", {"layer": layer, "neurons": g["neurons"], "design": g["design"],
                                      "random_controls": g["random_controls"], "max_tokens": max_tokens, "top_k": top_k,
                                      "run": name, "prompt_set": prompt_set, "dose_unit": dose_unit, "resume": resume},
                         workers)
        text = d.pop("text")
        return d, text + f"\n下一步：gne rank --run {name}（合并读取各工作进程的 part-*）", 0
    d = s.sweep(layer, neurons=g["neurons"], design=g["design"], random_controls=g["random_controls"], max_tokens=max_tokens,
                top_k=top_k, run=run, prompt_set=prompt_set, dose_unit=dose_unit, resume=resume)
    text = (f"{d['path']}（{d['n_neurons']} 个神经元 + 对照 {d['n_controls']}、{d['design']}、"
            f"新增 rollout {d['n_new_records']}、{d['seconds']:.0f} 秒）\n下一步：gne rank --run {d['run']}")
    return d, text, 0


@gne_command(app, "rank")
def rank(c: Common, holder: dict, run: str = typer.Option(..., "--run", help="run 名称或 JSONL 路径"),
         show: int = typer.Option(40, "--show"),
         find: Optional[list[int]] = typer.Option(None, "--find", help="输出该神经元的排名（可重复）"),
         word_unit: str = typer.Option("tokens", "--word-unit", help="tokens（按 token id，默认）/ regex"),
         max_rel: float = typer.Option(1.6, "--max-rel", help="统计词时使用的 rollout rel_push 上限")):
    """从 sweep 的 rollout 创建 units.jsonl，按词汇聚集量（mass）排序并显示前列。"""
    s = holder["session"] = c.session()
    d = s.rank(run, word_unit=word_unit, max_rel=max_rel, show=show, find=find)
    cm = d["control_mass"]
    lines = [f"{d['units_path']}（{d['n_neurons']} 个神经元 + 对照 {d['n_controls']}、词单位 {word_unit}）",
             f"对照 mass：中位数 {cm['p50'] or 0:.2f} / 95% {cm['p95'] or 0:.2f} / 最大 {cm['max'] or 0:.2f}、"
             f"高于全部对照的神经元 {d['n_above_all_controls']}"]

    def row(u):
        who = f"N{u['neuron']}" if u["kind"] == "neuron" else f"random{u['neuron']}"
        rk = f"#{u['rank']:<4}" if u["rank"] else "     "
        null = f"{u['null_pct']:.0%}" if u["null_pct"] is not None else "-"
        ws = ", ".join(f"{w}({cnt})" for w, cnt, _ in u["words"][:6])
        return f"  {rk} {who:<11} mass {u['mass']:6.2f}  null {null:>4}  {ws}"

    lines += [row(u) for u in d["units"]]
    for u in d["find"]:
        lines.append("--find:" + row(u))
    return d, "\n".join(lines), 0


# --- chat / neuron / mcp(Phase 4) -------------------------------------------------------------------------

@gne_command(app, "chat")
def chat(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
         dose_unit: Optional[str] = _DOSE_UNIT,
         dose: Optional[float] = typer.Option(None, "--dose", help="统一各单元的 dose 幅度（保留符号）"),
         left: Optional[Path] = typer.Option(None, "--left", help="左列配方（默认为 −配方）"),
         right: Optional[Path] = typer.Option(None, "--right", help="右列配方（默认为 +配方）"),
         center: Optional[Path] = typer.Option(None, "--center", help="中列配方（默认为无干预）"),
         prompt: Optional[list[str]] = typer.Option(None, "--prompt", "-p",
                                                    help="用户输入（多个可组成多轮）。未指定时进入交互模式"),
         system: Optional[str] = typer.Option(None, "--system"),
         max_tokens: int = typer.Option(256, "--max-tokens"),
         stream: bool = typer.Option(True, "--stream/--no-stream", help="边生成边显示"),
         when: Optional[str] = _WHEN, ablate: Optional[str] = _ABLATE,
         temperature: Optional[float] = _TEMP, top_k: Optional[int] = _TOP_K, top_p: Optional[float] = _TOP_P,
         min_p: Optional[float] = _MIN_P, presence_penalty: Optional[float] = _PRES,
         sample_seed: Optional[int] = _SSEED, greedy: bool = _GREEDY):
    """−/0/+ 三列对比聊天（共享用户输入，逐列回复）。采样时三列使用相同 seed。"""
    from .chat import ChatState, chat_turn, make_columns
    s = holder["session"] = c.session()
    if when:
        from .when import parse_when
        parse_when(when)
    base = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate) if (unit or recipe or ablate) else None
    sp = _sampling(base, greedy, temperature, top_k, top_p, min_p, presence_penalty, sample_seed)
    load = lambda p: s.recipe(path=p) if p else None  # noqa: E731
    cols = make_columns(base, dose=dose, left=load(left), right=load(right), center=load(center))
    live = stream and not c.json

    def on_column(i, col):
        print(f"\n── [{i + 1}] {col.label} ──", file=sys.stderr if c.json else sys.stdout, flush=True)

    def on_text(_i, piece):
        print(piece, end="", flush=True)

    if prompt:
        state = ChatState(cols, system)
        turns = []
        for user in prompt:
            if live:
                print(f"\n> {user}", flush=True)
            turns.append({"user": user, "replies": chat_turn(s, state, user, max_tokens=max_tokens,
                                                             on_text=on_text if live else None,
                                                             on_column=on_column if live else None,
                                                             when_override=when, sampling=sp)})
        d = {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
             "system": system, "turns": turns, "sampling": None if sp is None else sp.to_dict()}
        if live:
            print()
            return d, "", 0
        lines = []
        for t in d["turns"]:
            lines.append(f"> {t['user']}")
            for col, rep in zip(d["columns"], t["replies"], strict=True):
                lines.append(f"── {col['label']} ──({rep['finish']}, {rep['n_gen']} tok)\n{rep['text']}")
        return d, "\n".join(lines), 0
    if c.json:
        raise GneError("BAD_ARGS", "使用 --json 时请通过 --prompt 传入输入（不能使用交互模式）")
    state = ChatState(cols, system)
    print("列：" + " | ".join(col.label for col in cols) + f"（{_sampling_text(sp)}）"
          + "\n用 /reset 重新开始会话，用 /quit 或 Ctrl-D 退出", file=sys.stderr)
    turns = []
    while True:
        try:
            user = input("\n> ")
        except EOFError:
            break
        if user.strip() in ("/quit", "/exit"):
            break
        if user.strip() == "/reset":
            state.reset()
            continue
        if not user.strip():
            continue
        replies = chat_turn(s, state, user, max_tokens=max_tokens, on_text=on_text if stream else None,
                            on_column=on_column if stream else None, when_override=when, sampling=sp)
        if not stream:
            for col, rep in zip(cols, replies, strict=True):
                print(f"── {col.label} ──\n{rep['text']}")
        else:
            print()
        turns.append({"user": user, "replies": replies})
    return {"columns": [{"label": col.label, "recipe": col.recipe.to_dict() if col.recipe else None} for col in cols],
            "system": system, "turns": turns}, "", 0




def _via_workers(s, kind: str, params: dict, workers: str) -> dict:
    """放入文件队列，并等待所有分片完成。"""
    from .workers import submit, wait
    sub = submit(s, kind, params, workers=workers)
    sys.stderr.write(f"已将任务拆成 {len(sub['jobs'])} 份分发给工作进程 {', '.join(sub['workers'])}（group {sub['group']}）\n")
    t0 = time.monotonic()
    js = wait(s.store.root, sub["group"], progress=s.progress)
    ok = all(j["status"] == "done" for j in js)
    lines = [f"{kind}（工作进程 {len(js)} 个、实际耗时 {time.monotonic() - t0:.0f} 秒）"]
    for j in js:
        el = (j.get("finished") or 0) - (j.get("started") or 0)
        lines.append(f"  {j['worker']}：{j['status']}、{el:.0f} 秒" + (f"、{j['error']['message']}" if j.get("error") else ""))
        if j.get("merge"):
            lines.append(f"  合并：{j['merge']['merged']}（神经元 {j['merge']['n_done']}、对照 {j['merge']['n_controls']}）")
    info = {"group": sub["group"], "workers": sub["workers"], "ok": ok, "wall_seconds": time.monotonic() - t0,
            "parts": [{"worker": j["worker"], "status": j["status"], "error": j.get("error"),
                       "seconds": (j.get("finished") or 0) - (j.get("started") or 0)} for j in js]}
    if not ok:
        raise GneError("WORKER_FAILED", "\n".join(lines))
    if kind == "sweep":                       # 部分の結果を足して、ふつうの sweep と同じ形にする
        rs = [j["result"] for j in js]
        out = {"run": params["run"], "path": str(s.run_path(params["run"])), "meta": rs[0]["meta"],
               "layer": params["layer"], "design": params["design"], "prompt_set": params["prompt_set"],
               "n_neurons": sum(r["n_neurons"] for r in rs), "n_controls": params["random_controls"],
               "n_units": sum(r["n_units"] for r in rs), "n_new_records": sum(r["n_new_records"] for r in rs),
               "seconds": info["wall_seconds"]}
    else:
        out = s.lens_summary(params["layer"], neurons=params["neurons"], show=params.get("show", 20),
                             seconds=info["wall_seconds"])
    return {**out, "workers": info, "text": "\n".join(lines)}


def _gate(s, kind: str, params: dict, yes: bool, budget: Optional[str]) -> dict:
    """CHANGE-03 C：按预算缩小范围；估算超过 30 分钟且未指定 --yes 时停止。返回实际执行的参数。"""
    from .estimate import NeedsConfirm, fmt_seconds, gate
    try:
        p, est = gate(s, kind, {**params, "yes": yes, **({"budget": budget} if budget else {})})
    except NeedsConfirm as e:
        e.error.hint = (e.error.hint or "") + f"（估算明细：{e.estimate['detail']}）"
        raise e.error from e
    if est is not None and est.get("seconds"):
        b = est.get("budget")
        sys.stderr.write(f"估算：{fmt_seconds(est['seconds'])}" + (f"（已按预算缩小：{b}）" if b else "") + "\n")
    return p


def _metric(x, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


@gne_command(app, "worker")
def worker(c: Common, holder: dict, name: str = typer.Option(..., "--name", help="工作进程名称（例 xtx、5060）"),
           poll: float = typer.Option(2.0, "--poll", help="轮询队列的间隔（秒）"),
           once: bool = typer.Option(False, "--once", help="队列为空时退出"),
           max_jobs: Optional[int] = typer.Option(None, "--max-jobs", help="运行该数量任务后退出")):
    """工作进程（CHANGE-03 B）：从 <root>/jobs/ 的文件队列中取出发给自己（或无指定目标）的同模型任务并执行。
    按 GPU 分别启动在独立环境、独立进程中，共享 --root（docs/setup.md）。任务通过 gne sweep / lens steer 的 --workers 提交。"""
    from .workers import run_worker
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", "--name 只能包含字母、数字、_、- 和 .")
    s = holder["session"] = c.session()
    s.engine                                              # 先に読み込んで VRAM を確保する
    sys.stderr.write(f"工作进程 {name}：{s.meta.model_id}、存放位置 {s.store.root}\n")
    n = run_worker(s, name, poll=poll, once=once, max_jobs=max_jobs, log=lambda m: sys.stderr.write(m + "\n"))
    return {"name": name, "model_id": s.meta.model_id, "n_jobs": n}, f"工作进程 {name}：已运行 {n} 个任务", 0


@gne_command(app, "connections")
def connections_cmd(c: Common, holder: dict, k: int = typer.Option(24, "--k", help="每个神经元保留的上游、下游数量"),
                    window: Optional[int] = typer.Option(None, "--window", help="仅处理 |j − i| ≤ window 的层对（默认全部层对）"),
                    chunk: int = typer.Option(1024, "--chunk"), yes: bool = _YES, budget: Optional[str] = _BUDGET):
    """仅用权重计算神经元间的连线（层 i 的写入对层 j 激活值的贡献，别册 §8.4，CPU）。"""
    s = holder["session"] = c.session()
    g = _gate(s, "connections", {"window": window}, yes, budget)
    d = s.connections(k=k, window=g.get("window"), chunk=chunk)
    text = (f"{d['path']}（{d['n_pairs']} 个层对、window {d['window'] if d['window'] is not None else '无'}、"
            f"{d['seconds']:.0f} 秒" + (f"、峰值 {d['peak_rss_gb']:.1f} GB" if d.get("peak_rss_gb") else "") + "）")
    return d, text, 0


@gne_command(app, "writers")
def writers_cmd(c: Common, holder: dict,
                contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML（A = 想查看的状态，B = 中立）"),
                name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用对照集的 name）"),
                ref_layer: Optional[int] = typer.Option(None, "--ref-layer", help="测量贡献所沿方向的层（默认倒数第二层）"),
                replies_a: Optional[Path] = typer.Option(None, "--replies-a", help="不生成 A 的回复，改为读取现有回复（YAML / JSON 中与提示词等长的列表，或 {提示词: 回复}）"),
                replies_b: Optional[Path] = typer.Option(None, "--replies-b"),
                max_tokens: int = typer.Option(96, "--max-tokens", help="生成的回复长度"),
                show: int = typer.Option(20, "--show"), yes: bool = _YES):
    """写入器强度：在状态 A（如病娇 system）下，各神经元实际写入残差的量（激活值差 × 写入方向），
    以及对参考层方向的直接贡献。同时给出嵌入、mixer（attention、卷积）和 MLP 分别写入了该方向的多少比例。"""
    s = holder["session"] = c.session()
    _gate(s, "writers", {"contrast": str(contrast), "max_tokens": max_tokens}, yes, None)
    d = s.writers(contrast, name=name, ref_layer=ref_layer, replies_a=replies_a, replies_b=replies_b,
                  max_tokens=max_tokens, top=max(show, 50))
    t = d["totals"]
    lines = [f"{d['path']}（回复 token：A {d['n_tokens']['A']} / B {d['n_tokens']['B']}、参考 L{d['ref_layer']}、{d['seconds']:.0f} 秒）",
             f"写入 L{d['ref_layer']} 方向（‖r‖ = {d['ref_norm']:.2f}）的比例：嵌入 {t['embed']:+.0%}、mixer {t['mixer']:+.0%}、MLP {t['mlp']:+.0%}"
             f"（其中 MLP 上位 10 个 {t['top10']:+.0%}、100 个 {t['top100']:+.0%}、1000 个 {t['top1000']:+.0%}）",
             f"  {'层':>4} {'类型':<17} {'mixer':>7} {'MLP':>7} {'‖r_L‖':>7} {'cos(r_L, r*)':>12}"]
    nr = d["ref_norm"]
    for x in d["layers"]:
        if "mixer_to_ref" not in x:
            continue
        mlp = "—" if x["mlp_to_ref"] is None else f"{x['mlp_to_ref'] / nr:+.1%}"
        lines.append(f"  {x['layer']:>4} {x['mixer']:<17} {x['mixer_to_ref'] / nr:>+7.1%} {mlp:>7} {x['r_norm']:>7.2f} {x['cos_ref']:>12.2f}")
    lines.append("向该方向写入的神经元前列（贡献 = 激活值差 × 写入方向的 r* 分量，比例为相对 ‖r*‖）：")
    for x in d["top"][:show]:
        lines.append(f"  L{x['layer']:<3} N{x['neuron']:<6} {x['share']:+.2%}  激活 A {x['act_a']:+.2f} / B {x['act_b']:+.2f}  方向 {x['align']:+.2f}")
    if d.get("mlp_check_max") is not None:
        lines.append(f"校验：公式计算的 MLP 输出与捕获输出的最大差异 {d['mlp_check_max']:.1%}")
    return d, "\n".join(lines), 0


@gne_command(app, "writers-export")
def writers_export(c: Common, holder: dict, name: str = typer.Option(..., "--name", help="gne writers 分析名称"),
                   out: str = typer.Option(..., "--out", help="导出方向名称"),
                   part: str = typer.Option("mlp", "--part", help="mlp / mixer / all（mixer + MLP）/ r（残差差本身）"),
                   top: Optional[int] = typer.Option(None, "--top", help="--part mlp：仅保留对参照方向贡献前 N 的神经元"),
                   layers: Optional[str] = typer.Option(None, "--layers", help="all / 10-16 / 1,5,9-12")):
    """将写入器分析中的写入量按层导出为方向。使用 "L:dir:<out>:1"（dose_unit raw）时，
    会把 A 状态下写入该层的量原样加到该层（可通过倍率增强）。适合只添加上位写入器，检查是否会出现性格。"""
    from .writers import export_writes
    if not re.fullmatch(r"[\w.\-]+", out):
        raise GneError("BAD_ARGS", "--out 只能包含字母、数字、_、-、.")
    s = holder["session"] = c.session()
    d = export_writes(s, name, out, part=part, top=top, layers=parse_layers(layers))
    text = (f"{d['path']}（{part}{'' if top is None else f'、前 {top} 个'}、{len(d['layers'])} 层）\n"
            + "  ‖v_L‖: " + " ".join(f"L{L}:{n:.2f}" for L, n in d["norms"].items())
            + f"\n用法（raw 倍率）：--dose-unit raw " + " ".join(f"--unit {u}" for u in d["units"][:3]) + (" …" if len(d["units"]) > 3 else ""))
    return d, text, 0


@gne_command(heads_app, "contrast", "heads contrast")
def heads_contrast(c: Common, holder: dict,
                   contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML（若有 triggers，也测量 trigger_mass）"),
                   name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用对照集的 name）"),
                   direction: Optional[str] = typer.Option(None, "--direction", help="用于 dir_cos 的方向（默认为与对照集同名的方向）"),
                   system_b2: Optional[str] = typer.Option(None, "--system-b2", help="用于噪声下限的另一中立 system（默认为长度相近的中立文本）"),
                   layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
                   max_tokens: int = typer.Option(48, "--max-tokens", help="比较用回复长度（以 system_a greedy 生成）"),
                   sort: str = typer.Option("z", "--sort", help="表格排序：z / trigger_mass / pattern_shift / out_shift / dir_cos（保存时按 z 排序）"),
                   show: int = typer.Option(20, "--show")):
    """分别用带人格（system_a）和中立（system_b）运行同一会话，逐注意力头测量关注与输出差异并排序。
    噪声下限取中立之间（system_b 与 system_b2）的差异。"""
    from .heads import METRICS
    if sort not in ("z", *METRICS):
        raise GneError("BAD_ARGS", f"--sort 必须是 z / {' / '.join(METRICS)} 之一")
    s = holder["session"] = c.session()
    d = s.heads_contrast(contrast, name=name, direction=direction, system_b2=system_b2, layers=parse_layers(layers),
                         max_tokens=max_tokens)
    fl = d["floors"]
    rows = d["top"]
    if sort != "z":
        from .heads import load_candidates
        rows = sorted((x for x in load_candidates(s, d["name"]) if x[sort] is not None), key=lambda x: -x[sort])
    lines = [f"{d['path']}（{d['n_heads']} 个注意力头、{d['n_prompts']} 个提示词（有触发词 {d['n_with_triggers']}）、"
             f"方向 {d['direction'] or '无'}、{d['seconds']:.0f} 秒）",
             "下限（中立之间的 p95）：" + "、".join(f"{k} {v:.3f}" for k, v in fl.items()),
             f"  {'序':>3} {'地址':>9} {'类型':>6} {'trigger':>8} {'pattern':>8} {'out':>6} {'dir_cos':>8} {'z':>6} {'null':>5}"]
    for i, x in enumerate(rows[:show], 1):
        lines.append(f"  {i:>3} {('L' + str(x['layer']) + ' H' + str(x['head'])):>9} {x['kind']:>6} "
                     f"{_metric(x['trigger_mass'], '+.3f'):>8} {_metric(x['pattern_shift'], '.3f'):>8} "
                     f"{_metric(x['out_shift'], '.2f'):>6} {_metric(x['dir_cos'], '+.3f'):>8} {_metric(x['z'], '.1f'):>6} "
                     f"{_metric(None if x['null_pct'] is None else x['null_pct'] * 100, '.0f'):>4}%")
    lines.append("读法：trigger 较大的注意力头负责“察觉”，dir_cos 较大的注意力头负责“写入人格方向”。"
                 "试用：gne steer --unit L:Hh:x0 / x2 --compare")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "contrast", "experts contrast")
def experts_contrast(c: Common, holder: dict,
                     contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML"),
                     name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认使用对照集的 name）"),
                     direction: Optional[str] = typer.Option(None, "--direction", help="用于 contrib_cos 的方向（默认为与对照集同名的方向，不存在时使用 l_out 的差）"),
                     system_b2: Optional[str] = typer.Option(None, "--system-b2", help="用于噪声下限的另一中立 system"),
                     layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12（仅测量 MoE 层）"),
                     max_tokens: int = typer.Option(48, "--max-tokens", help="比较用回复长度（以 system_a greedy 生成）"),
                     sort: str = typer.Option("z", "--sort", help="表格排序：z / usage_shift / weight_shift / contrib_cos / contrib_norm"),
                     show: int = typer.Option(20, "--show")):
    """分别用带人格（system_a）和中立（system_b）运行同一回复，逐 MoE 层专家测量选择方式与贡献差异并排序。
    噪声下限取中立之间（system_b 与 system_b2）的差异。"""
    from .experts import METRICS, load_expert_candidates
    if sort not in ("z", *METRICS):
        raise GneError("BAD_ARGS", f"--sort 必须为 z / {' / '.join(METRICS)} 之一")
    s = holder["session"] = c.session()
    d = s.experts_contrast(contrast, name=name, direction=direction, system_b2=system_b2, layers=parse_layers(layers),
                           max_tokens=max_tokens)
    rows = d["top"] if sort == "z" else sorted(load_expert_candidates(s, d["name"]),
                                               key=lambda x: -(abs(x[sort]) if sort != "contrib_norm" else x[sort]))
    lines = [f"{d['path']}（{d['n_experts']} 个专家、{d['n_prompts']} 个提示词、方向 {d['direction'] or 'l_out 的差'}、{d['seconds']:.0f} 秒）",
             "下限（中立之间的 p95）：" + "、".join(f"{k} {v:.3f}" for k, v in d["floors"].items()),
             f"  {'序':>3} {'地址':>8} {'选择率 A→B':>12} {'usage':>7} {'weight':>7} {'cos':>6} {'norm':>6} {'z':>5} {'null':>5}"]
    for i, x in enumerate(rows[:show], 1):
        lines.append(f"  {i:>3} {('L' + str(x['layer']) + ' E' + str(x['expert'])):>8} {x['rate_a']:5.2f}→{x['rate_b']:<5.2f} "
                     f"{x['usage_shift']:+7.3f} {x['weight_shift']:+7.3f} {x['contrib_cos']:+6.2f} {x['contrib_norm']:6.2f} "
                     f"{x['z']:5.1f} {x['null_pct'] * 100:4.0f}%")
    lines.append("读法：usage 较大的专家会在“人格状态下被调用”，cos 较大的专家会“写入人格方向”。"
                 "试用：gne steer --unit L:Ee:x0（消除）/ L:Ee:x2（增强）--compare")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "sweep", "experts sweep")
def experts_sweep(c: Common, holder: dict, layer: int = typer.Option(..., "--layer", "-l", help="MoE 层"),
                  experts: Optional[str] = typer.Option(None, "--experts", help="要试的专家（例：0,3,7。默认全部）"),
                  ks: str = typer.Option("0,2", "--ks", help="专家输出的倍率（0 = 消除，2 = 增强）"),
                  design: str = typer.Option("quick", "--design", help="configs/designs.yaml 中的设计（仅使用提示词）"),
                  prompt_set: str = typer.Option("en", "--prompt-set"),
                  max_tokens: int = typer.Option(48, "--max-tokens"),
                  run: Optional[str] = typer.Option(None, "--run", help="记录名称（默认 L<层>-experts-<设计>）"),
                  show: int = typer.Option(15, "--show"), yes: bool = _YES):
    """通过消除或增强各层专家的缩放补丁生成，并按词语出现方式排序（每个专家都会重新读取模型）。
    零假设为同一层所有专家的分布（null = mass 小于该专家的专家所占比例）。"""
    s = holder["session"] = c.session()
    exs = [int(x) for x in experts.split(",")] if experts else None
    kk = tuple(float(x) for x in ks.split(","))
    n = len(exs) if exs else s.weights.moe_layer(layer).n_expert
    _gate(s, "experts sweep", {"n": n * len(kk), "max_tokens": max_tokens}, yes, None)
    d = s.experts_sweep(layer, ks=kk, design=design, prompt_set=prompt_set, max_tokens=max_tokens, experts=exs, run=run)
    lines = [f"{d['path']}（L{layer}、{d['n_experts']} 个专家 × 倍率 {d['ks']}、重新读取平均 {d['reload_s']['mean']:.2f} 秒、{d['seconds']:.0f} 秒）"]
    for u in d["top"][:show]:
        lines.append(f"  {u['rank']:>3}. L{layer} E{u['expert']:<3} mass {u['mass']:5.1f}  null {u['null_pct'] * 100:3.0f}%  "
                     + " ".join(w[0] for w in u["words"][:8]))
    lines.append("试用：gne steer --unit L:Ee:x0 / x2 --compare；专家档案可用 gne experts show L E 查看")
    return d, "\n".join(lines), 0


@gne_command(experts_app, "show", "experts show")
def experts_show(c: Common, holder: dict, layer: int = typer.Argument(..., help="第 L 层"),
                 expert: int = typer.Argument(..., help="专家编号"),
                 top: int = typer.Option(20, "--top", help="神经元排名数量")):
    """单个专家的档案（使用统计、常被选中的 token、对照分析排名、神经元排名）。"""
    s = holder["session"] = c.session()
    d = s.expert(layer, expert, top=top)
    lines = [f"L{layer} E{expert}  从 {d['n_expert']} 个专家中选择 {d['n_expert_used']} 个（{d['routing']}）、宽度 {d['n_ff']}"
             + (f"、路由器偏置 {d['router_bias']:+.4f}" if d["router_bias"] is not None else "")]
    u = d["usage"]
    if u:
        lines.append(f"使用：选择率 {u['rate']:.1%}（{u['n_routed']}/{u['n_tokens']} token{'' if u['reliable'] else '，样本较少，分位数仅供参考'}）、"
                     f"平均权重 {u['weight_mean']:.3f}")
        lines.append("常被选中的 token：" + " ".join(f"{x['token']!r}×{x['count']}" for x in u["top_tokens"]))
    else:
        lines.append("无使用统计（可用 gne stats act --layers 创建）")
    for x in d["contrasts"]:
        lines.append(f"对照分析 {x['name']}：# {x['rank']}/{x['of']}  选择率 {x['rate_a']:.2f}→{x['rate_b']:.2f}  "
                     f"usage {x['usage_shift']:+.3f}  cos {x['contrib_cos']:+.2f}  z {x['z']:.1f}")
    lines.append("神经元（按写入大小 q999 × ‖W_down[:, n]‖ 排序）：" +
                 ", ".join(f"N{x['neuron']}" for x in d["neurons"][:12]))
    return d, "\n".join(lines), 0


@gne_command(heads_app, "attention", "heads attention")
def heads_attention(c: Common, holder: dict, text: str = typer.Option(..., "--text", "-t", help="要运行的文本"),
                    pos: int = typer.Option(-1, "--pos", help="查看的 token 位置（负数从末尾计）"),
                    layer: Optional[int] = typer.Option(None, "--layer"), head: Optional[int] = typer.Option(None, "--head"),
                    top: int = typer.Option(5, "--top"), chat: bool = typer.Option(False, "--chat", help="将 text 作为用户输入放入模板"),
                    system: Optional[str] = typer.Option(None, "--system", help="--chat 的 system")):
    """所选 token 在各注意力头中关注哪里（前 top 个）及逐头输出的范数（fires 的注意力头版）。"""
    s = holder["session"] = c.session()
    d = s.attention(text, pos=pos, layer=layer, head=head, top=top, system=system, chat=chat)
    lines = [f"{d['n_tokens']} 个 token、位置 {d['pos']} {d['tokens'][d['pos']]!r}"]
    rows = sorted(d["heads"], key=lambda x: -x["norm"])
    for x in rows[: (len(rows) if layer is not None else 30)]:
        t = " ".join(f"{y['token']!r}({y['weight']:.2f})" for y in x.get("top", []))
        lines.append(f"  L{x['layer']:<3} H{x['head']:<3} {x['kind']:<6} ‖out‖ {x['norm']:7.2f}  {t}")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "listen", "heads listen")
def heads_listen(c: Common, holder: dict, user: str = typer.Option(..., "--user", "-u", help="用户发言"),
                 heads: str = typer.Option(..., "--heads", help="要查看的注意力头（例：14:H0-3,16:H2，L:H* 表示该层全部注意力头）"),
                 system: Optional[str] = typer.Option(None, "--system", help="system（@FILE 表示从文件读取）"),
                 reply: Optional[str] = typer.Option(None, "--reply", help="回复（省略时以 greedy 生成）"),
                 max_tokens: int = typer.Option(64, "--max-tokens", help="生成回复时的长度"),
                 direction: Optional[str] = typer.Option(None, "--direction", help="用于测量贡献的方向（已保存名称；使用对应层的部分）"),
                 top: int = typer.Option(20, "--top")):
    """回复的 token 在所选注意力头中关注上下文的哪些 token，以及各 token 对方向的贡献（注意力 × 门控 × OV 的分解）。"""
    s = holder["session"] = c.session()
    if system and system.startswith("@"):
        system = Path(system[1:]).read_text(encoding="utf-8")
    d = s.heads_listen(user, heads=heads, system=system, reply=reply, max_tokens=max_tokens, direction=direction, top=top)
    hs = ",".join(f"L{x['layer']}H{x['head']}" for x in d["heads"])
    lines = [f"{d['n_tokens']} 个 token（回复从 {d['reply_start']} 开始）：{d['reply'][:160]!r}", f"注意力头 {hs}"]
    for k, x in d["by_role"].items():
        lines.append(f"  {k:<8} 注意力 {x['attn']:.3f}" + ("" if x["contrib"] is None else f"  贡献 {x['contrib']:+.4f}"))
    for L, x in (d["layers"] or {}).items():
        lines.append(f"L{L}：整层贡献 {x['contrib_total']:+.4f}（逐注意力头 " + " ".join(f"{v:+.3f}" for v in x["contrib_by_head"])
                     + f"）、分解误差 {x['check_rel_err'] if x['check_rel_err'] is not None else '—'}")
    for title, key in (("贡献排名", "top_contrib"), ("注意力排名", "top_attn")):
        if d[key]:
            lines.append(title + ": " + " ".join(f"{x['token']!r}({x['value']:+.3f})" for x in d[key]))
    return d, "\n".join(lines), 0


@gne_command(heads_app, "show", "heads show")
def heads_show(c: Common, holder: dict, layer: int = typer.Argument(..., help="层 L"),
               head: int = typer.Argument(..., help="注意力头编号"),
               lens: bool = typer.Option(True, "--lens/--no-lens", help="W_O 块的 logit lens（CPU，数秒起）")):
    """单个注意力头的档案（类型、对照分析排名、名称、已应用的 gain、W_O 块的 logit lens）。"""
    s = holder["session"] = c.session()
    d = s.head(layer, head, lens=lens)
    lines = [f"L{layer} H{head}" + (f"「{d['name']}」" if d["name"] else "") +
             f"  {d['kind']}、{d['n_head']} 个注意力头（K/V {d['n_head_kv']}）、宽度 {d['head_dim']}、{d['writer']} 的列 {d['block'][0]}:{d['block'][1]}"
             + (f"、K/V 组 {d['kv_group']}" if d["kv_group"] is not None else "")]
    for x in d["contrasts"]:
        lines.append(f"对照分析 {x['name']}：# {x['rank']}/{x['n_ranked']}  trigger {_metric(x['trigger_mass'], '+.3f')}  "
                     f"pattern {_metric(x['pattern_shift'], '.3f')}  out {_metric(x['out_shift'], '.2f')}  "
                     f"dir_cos {_metric(x['dir_cos'], '+.3f')}  z {_metric(x['z'], '.1f')}")
    if d["gain_k"]:
        lines.append("已应用 gain：" + ", ".join(f"×{k:g}" for k in d["gain_k"]))
    if d["logit_lens"]:
        ll = d["logit_lens"]
        lines.append(f"W_O 块的 logit lens（第一奇异值占比 {ll['sv_share']:.0%}）：")
        for i, cp in enumerate(ll["components"], 1):
            lines.append(f"  主成分 {i}：↑ {_toks(cp['up_tokens'])}\n{'':11}↓ {_toks(cp['down_tokens'])}")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "query", "heads query")
def heads_query(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), head: int = typer.Option(..., "--head"),
                direction: str = typer.Option(..., "--direction", help="人格方向（gne direction 的名称）。仅在该方向存在时生效"),
                seek: Path = typer.Option(..., "--seek", help="含有要寻找的词（triggers）的对照集 YAML"),
                scales: str = typer.Option("0,0.5,1,2,4", "--scales")):
    """（实验性，CHANGE-04 E）向注意力头的 Q 加入 rank-1 LoRA，使其仅在人格方向存在时才去查看触发词。
    对每个 scale，测量该头在有/无人格时对触发词的关注（V17）。配方写法为
    {layer, kind: head, head, mode: query, direction, seek, scale}。"""
    s = holder["session"] = c.session()
    d = s.heads_query(layer, head, direction=direction, seek=seek, scales=_doses(scales))
    lines = [f"L{layer} H{head}（K/V 组 {d['kv_group']}、u 为维度 {d['dims'][0]}～{d['dims'][1] - 1} 的 {d['dims'][2]} 个分量、"
             f"触发词 token {d['n_trigger_tokens']}、{d['seconds']:.0f} 秒）",
             f"v·norm(x)（开始写回复的位置）：有人格 {d['v_mean_a']:.2f} / 无人格 {d['v_mean_b']:.2f}",
             f"  {'scale':>6} {'有':>8} {'无':>8} {'其他头的最大变化':>12}"]
    for r in d["rows"]:
        lines.append(f"  {r['scale']:>6g} {r['head_a']:>8.4f} {r['head_b']:>8.4f} {r['others_shift_max']:>12.4f}")
    lines.append(("单调增加" if d["monotonic"] else "并非单调") +
                 f"（有 {d['gain_a']:+.3f}、无 {d['gain_b']:+.3f}）")
    return d, "\n".join(lines), 0


@gne_command(heads_app, "name", "heads name")
def heads_name(c: Common, holder: dict, layer: int = typer.Argument(...), head: int = typer.Argument(...),
               name: str = typer.Argument("", help="名称（留空则删除）")):
    """为注意力头命名（以 "L:Hh" 写入与神经元相同的 names）。"""
    s = holder["session"] = c.session()
    d = s.set_head_name(layer, head, name)
    return d, f"L{layer} H{head}：{name or '（已删除名称）'}", 0


@gne_command(app, "neuron")
def neuron(c: Common, holder: dict, layer: int = typer.Argument(..., help="层 L"),
           n: int = typer.Argument(..., help="神经元编号"),
           rollouts: int = typer.Option(4, "--rollouts", help="档案中包含的 rollout 数量"),
           compute: bool = typer.Option(False, "--compute", help="仅为该神经元计算缺失的 logit lens / steer lens")):
    """单个神经元的档案（汇总已有全部信息）。"""
    s = holder["session"] = c.session()
    d = s.neuron(layer, n, rollouts=rollouts, compute=compute)
    lines = [f"L{layer} N{n}" + (f"「{d['name']}」" if d["name"] else "") + f"  ‖W_down‖={d['down_norm']:.3f}",
             "coverage: " + ", ".join(f"{k} {'○' if v else '×'}" for k, v in d["coverage"].items())]
    sw = d["sweep"]
    if sw:
        lines.append(f"sweep({sw['run']}、{sw['design']}): #{sw['rank']}/{sw['n_ranked']}  mass {sw['mass']:.2f}  "
                     f"null {sw['null_pct']:.0%}  词 " + ", ".join(f"{w}({cnt})" for w, cnt, _ in sw["words"][:8]))
        for r in sw["rollouts"]:
            lines.append(f"  [{r['prompt']} {r['dose']:+}] {r['text'][:150]!r}")
    if d["steer_lens"]:
        st = d["steer_lens"]
        lines.append(f"steer lens: KL+ {st['kl_up']:.3f}({st['pct_up']:.0%}) KL- {st['kl_down']:.3f}({st['pct_down']:.0%})"
                     f"  对照 p95 {st['kl_null_p95'] or 0:.3f}\n  + {_toks(st['up'])}\n  - {_toks(st['down'])}")
    if d["logit_lens"]:
        lg = d["logit_lens"]
        lines.append(f"logit lens(peak z {lg['peak_z']:.1f}):\n  ↑ {_toks(lg['up'])}\n  ↓ {_toks(lg['down'])}")
    for name, v in d["directions"].items():
        lines.append(f"反向查找 {name}：cos {v['cos']:+.3f}（层内第 {v['rank_in_layer']} 位）")
    if d["activation"]:
        a = d["activation"]
        lines.append(f"激活值：absmax {a['absmax']:.3g}  q999 {a['q999']:.3g}  rms {a['rms']:.3g}")
    for wr in d.get("writers", []):
        lines.append(f"写入器 {wr['name']}：对方向 L{wr['ref_layer']} 的贡献 {wr['share']:+.2%}（#{wr['rank']}/{wr['n_ranked']}）、"
                     f"激活 A {wr['act_a']:+.2f} / B {wr['act_b']:+.2f}")
    for e in d["evals"]:
        how = f"x{e['k']:g}" if e["mode"] == "gain" else f"{e['dose']:+g}"
        ship = "无可发布倍率" if e["max_shippable_dose"] is None else f"可发布的最大倍率 {e['max_shippable_dose']:g}"
        top = e["at_max"]
        lines.append(f"eval {e['name']}（{how}、when={e['when']}、单元 {e['n_units']}）：{ship}"
                     + (f"、倍率 {top['dose']:g} 时能力 −{top['capability_drop_pt']:.1f}pt、新词 {top['novelty']:.2f}" if top else ""))
    if d["gain_k"]:
        lines.append("曾应用的 gain：" + " ".join(f"x{k:g}" for k in d["gain_k"]))
    return d, "\n".join(lines), 0


@app.command("mcp")
def mcp_cmd(ctx: typer.Context,
            model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF 文件"),
            ui_url: Optional[str] = typer.Option(None, "--ui-url", help="gne serve 的 URL（用于 ui_* 工具）"),
            root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT"),
            n_gpu_layers: Optional[int] = typer.Option(None, "--n-gpu-layers",
                                                       help="加载到 GPU 的层数（-1 = 全部）。与 gne serve 同时运行时设为 0 可把 GPU 让给 UI")):
    """MCP 服务器（stdio）。从 Claude Code 用 `claude mcp add gne -- gne mcp --model <path>` 连接。"""
    from .mcp_server import main as mcp_main
    c: Common = ctx.obj or Common()
    path = model or c.model
    if path is None:
        print("错误[BAD_ARGS]：请指定 --model", file=sys.stderr)
        raise typer.Exit(2)
    mcp_main(str(path), root=str(root or c.root) if (root or c.root) else None, ui_url=ui_url,
             n_gpu_layers=c.n_gpu_layers if n_gpu_layers is None else n_gpu_layers, n_ctx=c.n_ctx, seed=c.seed)


@app.command("serve")
def serve_cmd(ctx: typer.Context,
              model: Optional[Path] = typer.Option(None, "--model", "-m", envvar="GNE_MODEL", help="GGUF 文件"),
              port: int = typer.Option(8791, "--port"),
              host: str = typer.Option("127.0.0.1", "--host", help="仅限本地（127.0.0.1）"),
              root: Optional[Path] = typer.Option(None, "--root", envvar="GNE_ROOT"),
              openai: bool = typer.Option(False, "--openai", help="同时提供 OpenAI 兼容的 /v1/chat/completions（配方和 when 由 gne 指定，§6.15）"),
              mmproj: Optional[Path] = typer.Option(None, "--mmproj", envvar="GNE_MMPROJ",
                                                    help="用于音频输入（input_audio）的 mmproj（默认使用模型同目录下的 mmproj*.gguf）")):
    """Web UI(http://127.0.0.1:8791)。"""
    from .ui.server import serve
    c: Common = ctx.obj or Common()
    path = model or c.model
    if path is None:
        print("错误[BAD_ARGS]：请指定 --model", file=sys.stderr)
        raise typer.Exit(2)
    try:
        print(f"gne serve: http://{host}:{port}/" + (f"  (OpenAI 兼容：http://{host}:{port}/v1)" if openai else ""), file=sys.stderr)
        serve(path, host=host, port=port, root=root or c.root, n_gpu_layers=c.n_gpu_layers, n_ctx=c.n_ctx, seed=c.seed,
              openai=openai, mmproj=mmproj or c.mmproj)
    except GneError as e:
        print(f"错误[{e.code}]：{e.message}", file=sys.stderr)
        raise typer.Exit(2) from e


if __name__ == "__main__":
    app()

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
# gne direction(方向を作る)は、サブコマンドが無ければ従来どおり動くグループ。gne direction score は速い指標(CHANGE-06 B)
direction_app = typer.Typer(no_args_is_help=False, invoke_without_command=True)
app.add_typer(direction_app, name="direction")
from .modal_cli import modal_app  # noqa: E402
app.add_typer(modal_app, name="modal")
headproj_app = typer.Typer(no_args_is_help=True, help="导出注意力头的方向分量投影（mode: proj，CHANGE-06 E）")
app.add_typer(headproj_app, name="headproj")


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


def gne_callback(typer_app: typer.Typer, name: str):
    """gne_command と同じ形の本体を、サブコマンドを持つグループの callback として登録する(サブコマンドが無いときだけ動く)。"""

    def deco(f: Callable):
        sig = inspect.signature(f, eval_str=True)
        own = list(sig.parameters.values())[2:]
        ctx_param = inspect.Parameter("ctx", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=typer.Context)

        @functools.wraps(f)
        def wrapper(ctx: typer.Context, **kwargs):
            if ctx.invoked_subcommand is not None:
                return
            over = {k: kwargs.pop(k) for k in _COMMON_NAMES}
            c = (ctx.obj or Common()).merged(over)
            _run(c, name, f, kwargs)

        wrapper.__signature__ = sig.replace(parameters=[ctx_param, *own, *_common_params()])
        wrapper.__annotations__ = {}
        typer_app.callback(invoke_without_command=True)(wrapper)
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


def _keyed_who(kind: str, expert=None) -> str:
    """keyed の書き込み先の短い書き方(ユニットの記法と同じ: ffn / attn / S / Ee)。"""
    return {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(kind, f"E{expert}")


def _units_text(units: list[dict]) -> str:
    rows = []
    for u in units:
        if u.get("mode") == "keyed":                         # CHANGE-07: α は raw(d = r_L)
            a = u.get("alpha", u.get("k"))
            size = ""
            if u.get("norm"):
                size = f"、在键产生反应的位置 |α|‖r_L‖={u['norm']:.2f}" + (
                    f"(残差范数 ρ_L 的 {u['rel_push']:.2f} 倍)" if u.get("rel_push") is not None else "")
            rows.append(f"  L{u['layer']} {_keyed_who(u['kind'], u.get('expert'))} keyed α={a:+g} @{u.get('direction')} #{u.get('key')}(仅在键 κ 产生反应的状态写入方向，以 LoRA 应用{size})")
            continue
        who = {"neuron": f"N{u['neuron']}", "random": f"random{u['seed']}", "head": f"H{u.get('head')}"}.get(
            u["kind"], f"dir:{u.get('direction')}")
        if u.get("mode") == "proj":                          # CHANGE-06
            rows.append(f"  L{u['layer']} {who} s{u['k']:g} @{u.get('direction')}"
                        f"（从写入方向分量中减去 {u['k']:g} 倍，以 LoRA 应用）")
            continue
        if u.get("mode") == "query":
            rows.append(f"  L{u['layer']} {who} query {u.get('direction')} ×{u['k']:g}")
            continue
        if u.get("mode") == "route":
            rows.append(f"  L{u['layer']} E{u.get('expert')} route {u['dose']:+g}")
            continue
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
    # 確かめる対象が無い検査は省く(失敗ではない)。MoE の検査は tests/test_change05.py と gne experts、残差の流れは V26
    why = {}
    if not s.engine.cvec_supported:
        why |= dict.fromkeys(("V3", "V4", "V12"), f"{s.meta.arch} 的图不读取 control vector")
    if s.meta.n_expert:
        why |= dict.fromkeys(("V2", "V11", "V13", "V14"), "这是全层 MoE。此检查假设使用 dense 的 ffn_down（MoE 通过 V24 和 gne experts 检查）")
    steps = [(vid, (lambda vid=vid, fn=fn: VResult(vid, "省略", "skip", why[vid]) if vid in why else fn())) for vid, fn in steps]
    if s.engine.hc_streams:
        steps.append(("V26", v.v26_hc_streams))

    def show(r):
        results.append(r)
        c.human(f"[{r.status.upper():4}] {r.id} {r.title}: {r.detail}")

    for vid, fn in steps:
        try:
            show(fn())
        except Exception as ex:  # noqa: BLE001
            show(VResult(vid, "异常", "fail", repr(ex)))
    if bench and not s.engine.cvec_supported:
        show(VResult("V7", "基准测试", "skip", "不读取 control vector 的 arch"))
    elif bench:
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
_UNIT = typer.Option(None, "--unit", help=("L:N:dose（例 14:8949:+1.0）、L:random:SEED:dose、gain 为 L:N:x2，方向为 "
        "L:dir:NAME:dose，注意力头为 L:Hh:xk（例 21:H7:x0），注意力头方向分量投影为 L:Hh:sS@NAME（例 "
        "33:H16:s6@evil），带键写入为 L:ffn:keyed:aα@NAME#KEY（例 "
        "19:ffn:keyed:a1@zundamon_endings#zundamon_endings，attention 为 "
        "L:attn:keyed:…）。可重复 "))
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
         requant: str = typer.Option("auto", "--requant", help=("--bake 的写入方式: auto(能写补丁时写缩放补丁，不能写时为 q8_0。包含 keyed 时按张量处理：原本为 Q8_0 的保持 Q8_0，其余为 "
        "f16。仅对保持 Q8_0 时编辑的 1 成以上会被舍入抹掉的张量用 f16)/ patch / q8_0 / f16 / keep(原为 "
        "F32、F16、BF16 时保持该类型。专家张量 ffn_down_exps 保持原类型，不编辑的专家保持字节不变。K-quant、IQ 的专家为 q8_0) ")),
         key: Optional[str] = typer.Option(None, "--key", help="省略 #KEY 的 keyed 单元的键(由 gne direction key 创建)"),
         direction: Optional[str] = typer.Option(None, "--direction", help="省略 @NAME 的 proj、keyed 单元的方向"),
         strip_meta: bool = typer.Option(False, "--strip-meta", help="不保留 gne 记录（gne.* 元数据、配方、名称标记）（供分发的版本）")):
    """将 gain 单元（如 L:N:x2，把输出放大 k 倍；注意力头为 L:Hh:xk）导出为 LoRA 适配器 GGUF 或改写 GGUF（§6.16）。
    ヘッドの方向成分の射影(proj)と鍵つきの書き込み(keyed、CHANGE-07)も同じ LoRA・焼き込みに入る。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in 与 --model 指向不同文件")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, scale=scale, default_key=key, default_direction=direction)
    out = out or s.default_export_path(r, bake=bake, kind="gain")
    d = s.gain(r, out, bake=bake, requant=requant, strip_meta=strip_meta)
    if bake:
        text = f"{d['path']}({d['bytes'] / 1e6:.1f} MB)、改写: " + ", ".join(_mod_text(m) for m in d["modified"])
        if d.get("method") == "patch":
            text += f"\n缩放补丁（{d['patch_bytes'] / 1e3:.1f} KB，量化保持不变）。应用的操作位于 {d['path']}.patch.json"
        text += "".join("\n" + ln for ln in _keyed_export_lines(d))
    else:
        rows = ([f"  L{L}: " + ", ".join(f"N{int(n)} ×{k:g}" for n, k in nk) for L, nk in (d.get("layers") or {}).items()]
                + [f"  L{L}: " + ", ".join(f"H{int(h)} ×{k:g}" for h, k in hk) for L, hk in (d.get("heads") or {}).items()]
                + [f"  L{L}: " + ", ".join(f"H{int(h)} s{sv:g}@{dn}" for h, sv, dn in x["heads"])
                   + f"（方向分量投影，秩 {x['rank']}{'' if x['folded'] else '，方向混杂，无法折叠为秩 1'}）"
                   for L, x in (d.get("proj_layers") or {}).items()]
                + _keyed_export_lines(d)
                + [f"  query 操作：{p}" for p in (d.get("query_loras") or [])])
        text = (f"{d['path']}({d['bytes'] / 1e6:.2f} MB、LoRA)" + "".join("\n" + r for r in rows)
                + "\n用法：llama-cli -m <model> --lora " + d["path"])
    return d, text, 0


def _mod_text(m) -> str:
    """焼き込みで書き換えたテンソル 1 つ(分割の焼き込みは名前だけ)。keyed・proj は量子化し直しで失った分も出す。"""
    if isinstance(m, str):
        return m
    s = f"{m['tensor']}({m.get('from', m.get('type', '?'))} → {m.get('to', '?')}"
    if m.get("experts") is not None:            # 専門家の 3 次元テンソル(keyed の expert:e、CHANGE-07)
        s += "、专家 " + ",".join(f"E{e}" for e in m["experts"])
        s += "，其他专家" + ("保持字节不变" if m.get("untouched") == "raw" else f"转换({m.get('untouched')})")
        if m.get("bytes_to") is not None and int(m["bytes_to"]) != int(m.get("bytes_from", m["bytes_to"])):
            s += f"、{int(m['bytes_from']) / 1e6:.1f} → {int(m['bytes_to']) / 1e6:.1f} MB"
    if m.get("requant_error") is not None:
        s += f"，舍入损失占编辑的 {m['requant_error'] * 100:.1f}%"
    if m.get("edit_kept") is not None:
        s += f"，保留的编辑 {m['edit_kept'] * 100:.1f}%"
    if m.get("fallback"):
        s += "，Q8_0 下编辑会被淹没，因此用 F16"
    return s + ")"


def _keyed_export_lines(d: dict, *, requant_line: bool = True) -> list[str]:
    """gain・headproj export の結果の keyed の行(CHANGE-07): 層ごとの書き込み先と階数(= その層の鍵の数)・鍵・方向と、
    焼き込みなら量子化し直しで失った分の最大(requant_line)。"""
    lines = []
    for L, x in (d.get("keyed_layers") or {}).items():
        tens = ", ".join(f"{n} 阶数 {r}" for n, r in x["tensors"].items())
        lines.append(f"  L{L}: 带键写入 {tens}(键 {', '.join(x['keys'])}、方向 {', '.join(x['directions'])})")
    err, kept = d.get("requant_error_max"), d.get("edit_kept_min")
    if requant_line and d.get("keyed_layers") and err is not None:
        bad = (kept < 0.9) if kept is not None else err > 0.5     # 残った編集の割合で見る(無ければ丸めのずれ)
        lines.append(f"  重新量化损失最大为编辑大小的 {err * 100:.1f}%"
                     + (f"，保留的编辑最小为 {kept * 100:.1f}%" if kept is not None else "")
                     + ("(编辑被舍入淹没。请使用 --requant f16 或 auto，或用 LoRA 分发)" if bad else ""))
    return lines


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
             control_pool: str = typer.Option("before", "--control-pool", help="proj 的对照注意力头总体：before（测量投影值的层之前的全部注意力头）/ same-layers（同一层，严格对照）"),
             compare_methods: bool = typer.Option(False, "--compare-methods", help="与 proj 配方使用同一方向，在效果对齐后比较全层投影、注意力头方向分量、注意力头整体倍率和神经元消除（CHANGE-06 F.3）"),
             match_effect: Optional[float] = typer.Option(None, "--match-effect", help="--compare-methods：对齐用的 A 侧投影值（省略则使用此配方的水平）"),
             score_layer: Optional[int] = typer.Option(None, "--score-layer", help="--compare-methods：测量投影值的层"),
             alloc: Optional[str] = typer.Option(None, "--alloc", help="按注意力头数量评估 gne heads allocate 的结果（名称或路径）中的各组（CHANGE-06 F.4）"),
             n_heads: Optional[str] = typer.Option(None, "--n-heads", help="--alloc：要评估的注意力头数量（例 6,5,4,3）"),
             direction_metrics: bool = typer.Option(True, "--direction-metrics/--no-direction-metrics",
                                                    help="使用方向的配方还会测量方向指标（投影值下降幅度、偏移、判定一致率）"),
             dir_max_examples: int = typer.Option(64, "--dir-max-examples", help="方向指标读取的问题数量（A、B 各自。0 = 全部）"),
             keyed: bool = typer.Option(True, "--keyed/--no-keyed",
                                        help="对包含 keyed 单元的配方，还会测量 eval 的选择性(写入位置、不希望写入位置的残差变化)、方式比较、与比较用键(打乱标签后用相同步骤创建的键)的比较"),
             keyed_contrast: Optional[str] = typer.Option(None, "--keyed-contrast",
                                                          help="keyed: 创建键所用的对照集(configs/contrast/ 的名称或路径)。省略时，若创建方向所用的对照集与键相同则使用它"),
             keyed_examples: int = typer.Option(8, "--keyed-examples", help="keyed: 每组对话读取的数量(0 = 全部)"),
             keyed_pseudo: int = typer.Option(8, "--keyed-pseudo", help="keyed: 用于比较 eval 选择性的比较用键数量(最多为键文件中比较用键的数量)"),
             keyed_subset: Optional[str] = typer.Option(None, "--keyed-subset",
                                                        help="keyed: 读取的样本 even / odd / all(默认为键创建时的相反侧)"),
             keyed_match: bool = typer.Option(True, "--keyed-match/--no-keyed-match",
                                              help="keyed: 在方式比较中，将 add、gain、proj 的大小在写入位置与 keyed 对齐(省略则更快)"),
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
    if alloc:                                            # CHANGE-06 F.4: ヘッド数の削減曲線
        d = s.eval_alloc_curve(alloc, n_heads=_ints(n_heads), suite=suite, when=when, random_controls=random_controls,
                               name=name)
        lines = [f"注意力头数量削减曲线 {d['name']}（方向 {d['direction']}、L{d['score_layer']}、目标 {d['target']:g}、"
                 f"{d['seconds']:.0f} 秒）→ {d['path']}",
                 f"  {'数量':>3} {'最大s':>6} {'剩余':>5} {'能力下降':>8} {'ppl恶化':>8} {'混入':>6} {'下降':>7} {'B正交':>7} {'一致率':>6}"]
        for r_ in d["rows"]:
            lines.append(f"  {r_['n_heads']:>3} {r_['s_max']:>6.2f} {r_['remain']:>5} {r_['capability_drop_pt']:>7.1f}pt "
                         f"{r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}% {_f(r_.get('dir_drop_a'), '+7.2f')} "
                         f"{_f(r_.get('drift_orth_b'), '7.4f')} {_f(r_.get('rate'), '6.2f')}"
                         + ("  ← allocate 推荐" if r_["recommended"] else ""))
        return d, "\n".join(lines), 0
    r = s.recipe(units=unit, path=recipe, dose_unit=dose_unit, ablate=ablate)
    if compare_methods:                                  # CHANGE-06 F.3
        d = s.eval_compare_methods(r, match_effect=match_effect, score_layer=score_layer, top_neurons=top_neurons,
                                   suite=suite, when=when, name=name)
        lines = [f"方式比较 {d['name']}（方向 {d['direction']}、将 L{d['score_layer']} 的 A 侧投影值对齐到 {d['target']:.2f}。无干预 A {d['base_score_a']:.2f} / B {d['base_score_b']:.2f}、{d['seconds']:.0f} 秒）→ {d['path']}",
                 f"  {'方式':<10} {'α':>6} {"A侧投影值":>7} {'B正交':>7} {'能力下降':>8} {'ppl恶化':>8} {'混入':>6} {'一致率':>6}"]
        for r_ in d["rows"]:
            if r_.get("excluded"):
                lines.append(f"  {r_['method']:<10} 排除：{r_['excluded']}")
                continue
            lines.append(f"  {r_['method']:<10} {r_['alpha']:>6.3f} {r_['score_a']:>7.2f} {r_['drift_orth_b']:>7.4f} "
                         f"{r_['capability_drop_pt']:>7.1f}pt {r_['ppl_worse_pct']:>+7.1f}% {r_['mix_increase'] * 100:>+5.2f}% "
                         f"{_f(r_['rate'], '6.2f')}" + ("  可发布" if r_["shippable"] else ""))
        lines.append("在同一 A 侧投影值水平下比较的副作用。报告应主要采用判定一致率（外部可见指标）")
        return d, "\n".join(lines), 0
    _gate(s, "eval", {"doses": _doses(doses), "suite": suite, "when": when, "random_controls": random_controls,
                      "recipe": r, "keyed_metrics": keyed, "keyed_contrast": keyed_contrast,      # keyed の選択性の分も見積もる
                      "keyed_examples": keyed_examples or None, "keyed_pseudo": keyed_pseudo, "keyed_match": keyed_match,
                      "keyed_subset": keyed_subset}, yes, None)
    d = s.eval(r, doses=_doses(doses), suite=suite, when=when, judge_model=judge_model, judge_target=judge_target,
               name=name, random_controls=random_controls, reactivity=reactivity, head_controls=head_controls,
               control_pool=control_pool, direction_metrics=direction_metrics, dir_max_examples=dir_max_examples or None,
               keyed_contrast=keyed_contrast, keyed_metrics=keyed, keyed_examples=keyed_examples or None,
               keyed_match=keyed_match, keyed_pseudo=keyed_pseudo, keyed_subset=keyed_subset)
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
    dmx = d.get("direction_metrics")
    if dmx:                                              # CHANGE-06 F.1: 方向への射影の値と直交成分のずれ
        head += f" │ {'下降':>7} {'A正交':>7} {'B正交':>7}" + (f" {'一致率':>6}" if dmx.get("behavior") else "") \
            + (f" {'null':>5}" if nc else "")
    kd = d.get("keyed")                                  # CHANGE-07 E: 結果の keyed(方式の比較・比較用の鍵、または skipped)
    ksel = any((row.get("keyed") or {}).get("selectivity") is not None for row in d["rows"])
    knull = ksel and any((row.get("keyed") or {}).get("null_pct") is not None for row in d["rows"])
    if ksel:                                             # 行の keyed: 目的の層の eval の選択性(比較用の鍵の対照があれば null)
        head += f" │ {'选择性':>7}" + (f" {'null':>5}" if knull else "")
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
        if dmx:
            line += (f" │ {_f(row.get('dir_drop_a'), '+7.2f')} {_f(row.get('drift_orth_a'), '7.4f')} {_f(row.get('drift_orth_b'), '7.4f')}"
                     + (f" {_f(row.get('rate'), '6.2f')}" if dmx.get("behavior") else ""))
            if nc:
                npct = row.get("rate_null_pct") if row.get("rate_null_pct") is not None else row.get("dir_null_pct")
                line += f" {_f(None if npct is None else npct * 100, '4.0f')}%"
        if ksel:
            kr = row.get("keyed") or {}
            line += f" │ {_f(kr.get('selectivity'), '+.3g'):>7}"
            if knull:
                kn = kr.get("null_pct")
                line += f" {kn * 100:>4.0f}%" if kn is not None else f" {'—':>5}"
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
    if dmx:
        text += (f"\n方向 {dmx['direction']}（L{dmx['layers'][-1]}）：下降 = A 侧投影值下降幅度，A正交 / B正交 = 残差差的正交分量 ÷ 残差" + ("，一致率 = A 侧判定（validate.match）" if dmx.get("behavior") else ""))
    if kd and "skipped" in kd:
        text += f"\n未测量 keyed 的 eval 选择性: {kd['skipped']}" + (f"\n提示: {kd['hint']}" if kd.get("hint") else "")
    elif kd:
        text += "\n" + "\n".join(kd.get("text") or [])
    if ksel:
        kL = next(row["keyed"].get("layer") for row in d["rows"] if row.get("keyed"))
        text += (f"\n行的选择性 = L{kL} 的 eval 选择性(残差在 r̂ 方向上的变化，写入位置 − 不希望写入位置。按写入方向对齐的值，应用整个配方后的结果。与 key-check 的选择性是不同的量)"
                 + ("。null = 比替换为比较用键的对照中多少比例更大" if knull else ""))
    for n_ in d.get("notes") or []:
        if kd and n_.startswith("keyed: "):              # keyed の注は上の keyed の表示に入っている
            continue
        text += f"\n注意: {n_}"
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


@gne_callback(direction_app, "direction")
def direction(c: Common, holder: dict,
              contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集 YAML（与 reverse 相同格式；选层使用 validate.match）"),
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
    创建的方向可用 L:dir:NAME:dose 单元加入（gne steer / chat / eval / export-cvec）。
    gne direction score は、方向の分離度(層ごとの cohen_d)と介入の効き目を生成なしで測る(CHANGE-06 B)。
    gne direction key / key-check / keys / key-verify / key-experts は鍵つきの書き込み(keyed、CHANGE-07)の鍵を作る・確かめる(key-experts は MoE の expert:e の鍵の専門家を選ぶ材料)。
    key-from-behave は CHiPS のレシピを当てても振る舞いが残った質問(gne behave の結果)から keyed の対照集合の JSON を作る。"""
    if contrast is None:
        raise GneError("BAD_ARGS", "--contrast 是必需的", "创建方向：gne direction --contrast <对照集>；"
                                                          "测量：gne direction score --name <方向> --contrast <对照集>")
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
                   system_b2: Optional[str] = typer.Option(None, "--system-b2", help="用于噪声基准值的另一中立 system（默认为长度相近的中立文本）"),
                   layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12"),
                   max_tokens: int = typer.Option(48, "--max-tokens", help="比较用回复长度（以 system_a greedy 生成）"),
                   sort: str = typer.Option("z", "--sort", help="表格排序：z / trigger_mass / pattern_shift / out_shift / dir_cos（保存时按 z 排序）"),
                   show: int = typer.Option(20, "--show")):
    """分别用带人格（system_a）和中立（system_b）运行同一会话，逐注意力头测量关注与输出差异并排序。
    基準値に取中立之间（system_b 与 system_b2）的差异。"""
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
             "基准值（中立之间的 p95）：" + "、".join(f"{k} {v:.3f}" for k, v in fl.items()),
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
                     system_b2: Optional[str] = typer.Option(None, "--system-b2", help="用于噪声基准值的另一中立 system"),
                     layers: str = typer.Option("all", "--layers", help="all / 10-16 / 1,5,9-12（仅测量 MoE 层）"),
                     max_tokens: int = typer.Option(48, "--max-tokens", help="比较用回复长度（以 system_a greedy 生成）"),
                     sort: str = typer.Option("z", "--sort", help="表格排序：z / usage_shift / weight_shift / contrib_cos / contrib_norm"),
                     show: int = typer.Option(20, "--show")):
    """分别用带人格（system_a）和中立（system_b）运行同一回复，逐 MoE 层专家测量选择方式与贡献差异并排序。
    基準値に取中立之间（system_b 与 system_b2）的差异。"""
    from .experts import METRICS, load_expert_candidates
    if sort not in ("z", *METRICS):
        raise GneError("BAD_ARGS", f"--sort 必须为 z / {' / '.join(METRICS)} 之一")
    s = holder["session"] = c.session()
    d = s.experts_contrast(contrast, name=name, direction=direction, system_b2=system_b2, layers=parse_layers(layers),
                           max_tokens=max_tokens)
    rows = d["top"] if sort == "z" else sorted(load_expert_candidates(s, d["name"]),
                                               key=lambda x: -(abs(x[sort]) if sort != "contrib_norm" else x[sort]))
    lines = [f"{d['path']}（{d['n_experts']} 个专家、{d['n_prompts']} 个提示词、方向 {d['direction'] or 'l_out 的差'}、{d['seconds']:.0f} 秒）",
             "基准值（中立之间的 p95）：" + "、".join(f"{k} {v:.3f}" for k, v in d["floors"].items()),
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



# --- CHANGE-06: ヘッドの方向成分の射影 ----------------------------------------------------------------------

def _ints(text: Optional[str]) -> Optional[list[int]]:
    if text is None or text == "":
        return None
    try:
        return [int(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"{text!r} 必须是逗号分隔的整数（例 38,50,62）") from e


def _f(x, fmt: str) -> str:
    return "—" if x is None else format(x, fmt)


@gne_command(direction_app, "score", "direction score")
def direction_score(c: Common, holder: dict,
                    name: str = typer.Option(..., "--name", help="方向名称（由 gne direction 创建）"),
                    contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集 YAML（默认使用创建方向时的对照集）"),
                    layers: Optional[str] = typer.Option(None, "--layers", help="要测量的层（例 38,50,62。默认全部层；无干预时记录 cohen_d 最大的层作为基准层）"),
                    positions: str = typer.Option("last", "--positions", help="测量位置：last / last:K（最后 K 个 token）/ assistant（生成提示词的 token）"),
                    recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                    side: str = typer.Option("both", "--side", help="both / A / B"),
                    subset: str = typer.Option("all", "--subset", help="all / even(测量用)/ odd(确认用)"),
                    set_base: Optional[int] = typer.Option(None, "--set-base", help="手动指定并记录基准层"),
                    record: Optional[bool] = typer.Option(None, "--record/--no-record", help="将 cohen_d 最大的层记录为基准层（默认只在尚无记录时）")):
    """速い指標(CHANGE-06 B): 生成せず、対照集合の A 側・B 側をプロンプトの最後まで読ませて、層ごとに方向への射影の値
    (残差 · r̂_L)を測る。--unit / --recipe を渡すと、その介入での射影の値の変化と、残差の差の方向成分・直交成分(drift_par / drift_orth)も出す。"""
    s = holder["session"] = c.session()
    d = s.direction_score(name, contrast=contrast, layers=_ints(layers), positions=positions, units=unit,
                          recipe=recipe, side=side, subset=subset, set_base=set_base, record_base=record)
    iv = d["intervention"] is not None
    sides = [x for x in "AB" if x in d["n"]]
    head = f"  {'层':>4} {'d':>6} " + " ".join(f"{x + ' 平均':>9} {'sd':>6}" for x in sides) + f" {'cos基准':>7}"
    if iv:
        head += " │ " + " ".join(f"{x + ' 下降':>8} {x + ' 正交':>7}" for x in sides)
    lines = [f"方向 {d['direction']}（A {d['n'].get('A', 0)} / B {d['n'].get('B', 0)} 例，位置 {d['positions']}、"
             f"{d['subset']}、{d['seconds']:.1f} 秒）", head]
    for r in d["layers"]:
        line = f"  {r['layer']:>4} {_f(r['cohen_d'], '6.2f')} " + " ".join(
            f"{r[x]['mean']:>9.2f} {r[x]['sd']:>6.2f}" for x in sides) + f" {_f(r['cos_base'], '+7.2f')}"
        if iv:
            line += " │ " + " ".join(f"{r['drop'][x]:>+8.2f} {r['drift'][x]['orth']:>7.4f}" for x in sides)
        if r["layer"] == d["base_layer"]:
            line += "  ← 基准层" + ("（已记录）" if d["base_recorded"] else "")
        lines.append(line)
    if not iv:
        lines.append("下一步：用 gne heads causal --name " + d["direction"] + " 查找因果有效的注意力头")
    return d, "\n".join(lines), 0


# --- CHANGE-07: 鍵つきの書き込み(keyed write)--------------------------------------------------------------

_KEY_POS = (("写入位置: response / assistant / last / last:K / "
        "before:<正则表达式>(在与推理时相同上下文的回复中，匹配字符的紧前方。写入位置的取法 (b))。默认为对照集的 positions，没有则为 "
        "response "))
_KEY_POS_B = "不希望写入的位置(默认: before:<re> 时为 not-before:<re>，否则与写入位置相同的指定)"
_KEY_REPLIES_A = ("供读取写入侧回复的文件(YAML / JSON，与问题等长的列表或 {问题: 回复})。before:<re> 时写入位置与不希望写入位置在同一回复中，因此只传入一个")
_KEY_REPLIES_B = "供读取不希望写入侧回复的文件"
_KEY_IV = "在应用此干预的状态下捕获(配方 YAML、已保存配方的名称，或单元的写法。可多个)"
_REPLY_KIND = {"given": "给定", "generated": "生成"}


def _iv_arg(iv: Optional[list[str]]):
    """--iv → keyed_build.intervention に渡す形: レシピのパスか保存したレシピの名前 1 つ、またはユニットの書き方のリスト。"""
    if not iv:
        return None
    if len(iv) == 1 and (Path(iv[0]).exists() or ":" not in iv[0]):
        return iv[0]
    return list(iv)


def _floats(text: str, opt: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as e:
        raise GneError("BAD_ARGS", f"{opt} {text!r} 应为逗号分隔的数字(例 1,2)") from e


def _fc(x, spec: str) -> str:
    """表の列: 数を spec(例 '>8.3f')で書き、None は同じ幅の —。"""
    return format("—", spec.split(".")[0]) if x is None else format(x, spec)


def _target_who(target: str) -> str:
    """鍵の書き込み先(ffn / attn / shared / expert:e)→ ユニットの記法の書き方(ffn / attn / S / Ee)。"""
    if target.startswith("expert:"):
        return f"E{target.split(':', 1)[1]}"
    return _keyed_who(target)


def _replies_text(replies: dict | None) -> str:
    if not replies:
        return ""
    return "、回复 " + "、".join(f"{g} {_REPLY_KIND.get(v, "无")}" for g, v in replies.items())


@gne_command(direction_app, "key", "direction key")
def direction_key(c: Common, holder: dict,
                  contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML(system_a、system_b 与问题，可选 dont_touch、positions、replies。例 configs/contrast/zundamon_endings.yaml)"),
                  layers: str = typer.Option(..., "--layers", help="创建键的层(必需。例 8,14,20 / 10-16。Q 的基底按层创建，因此不默认全层。最终层与 MTP 层不可用)"),
                  name: Optional[str] = typer.Option(None, "--name", help="键的名称(默认为对照集的 name)。保存到 <root>/keys/<model_id>/<名称>.npz"),
                  positions: Optional[str] = typer.Option(None, "--positions", help=_KEY_POS),
                  positions_b: Optional[str] = typer.Option(None, "--positions-b", help=_KEY_POS_B),
                  skip: Optional[int] = typer.Option(None, "--skip", help="从回复开头丢弃的 token 数(默认为对照集的 skip，没有则为 0)"),
                  max_tokens: int = typer.Option(96, "--max-tokens", help="生成回复时的长度"),
                  q_mode: str = typer.Option("examples", "--q-mode", help=("构成 Q(不希望写入位置与 dont_touch 的激活所张成子空间的基底)的行: examples(每个样本的平均，默认)/ "
        "means(每组的平均。效果好但选择性粗)/ samples(token 的激活。选择性高但较重) ")),
                  q_energy: float = typer.Option(0.99, "--q-energy", help="Q 的截断(奇异值平方的累计比例)"),
                  q_max_rank: Optional[int] = typer.Option(None, "--q-max-rank", help="Q 的秩上限"),
                  q_max_samples: int = typer.Option(4096, "--q-max-samples", help="--q-mode samples 使用的 token 上限(超过则抽样)"),
                  subset: str = typer.Option("even", "--subset", help="使用的问题: even(默认。key-check 用 odd 确认)/ odd / all"),
                  pseudo: int = typer.Option(16, "--pseudo", help="比较用键(将写入侧、不希望写入侧的标签按样本打乱，用相同步骤创建的键)的数量。打乱由 --seed 决定"),
                  replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                  replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                  iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                  target: str = typer.Option("ffn", "--target", help=("写入目标: ffn(dense 的 ffn_down 输入)/ attn(attn_output、ssm_out 的输入 = 注意力头输出的拼接)/ "
        "shared(MoE 层的 shared expert 的 ffn_down_shexp 输入)/ expert:e(ffn_down_exps 中专家 e "
        "的输入。仅在专家 e 被选中的位置创建) ")),
                  dont_touch_start: Optional[bool] = typer.Option(None, "--dont-touch-start/--no-dont-touch-start", help="dont_touch 各群问题的、开始写回复紧前方的位置也加入 Q(默认为对照集的 dont_touch_start，没有则加入)。不希望写入侧问题的此位置不加入")):
    """keyed write の鍵 κ_L を作る(CHANGE-07)。書き込みたい位置の活性の平均 μ_L を、書き込みたくない位置と dont_touch の
    活性が張る空間 Q_L に直交させ、κ_Lᵀμ_L = 1 に正規化して保存する(方向とは別のファイル)。鍵は無介入で作る(--iv は重ねるときだけ)。
    作ったら gne direction key-check で、作成に使っていない例(既定 subset odd)で分離を確かめる。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "请在 --layers 中指定层(不能使用 all)", "例 --layers 8,14,20")
    s = holder["session"] = c.session()
    d = s.direction_key(contrast, name=name, layers=Ls, positions=positions, positions_b=positions_b, skip=skip,
                        max_tokens=max_tokens, q_mode=q_mode, q_energy=q_energy, q_max_rank=q_max_rank,
                        q_max_samples=q_max_samples, subset=subset, pseudo=pseudo, replies_a=replies_a,
                        replies_b=replies_b, iv=_iv_arg(iv), target=target, seed=c.seed, dont_touch_start=dont_touch_start,
                        progress=s.progress)
    d.pop("warnings", None)                        # 警告はエンベロープの warnings に入っている
    n = d["counts"]
    moe = any("rows_a" in r for r in d["layers"])            # 書き込み先 shared・expert:e の鍵(層ごとの行の数を出す)
    how = "与推理时相同上下文的回复" if d["mode"] == "same" else "system、recipe 的上下文"
    lines = [f"{d['path']}(键 {d['name']}、写入目标 {d['target']}、写入位置为{how}(mode {d['mode']})、{d['seconds']:.1f} 秒)",
             f"  位置: 写入位置 {d['positions']}、不希望写入位置 {d['positions_b']}、skip {d['skip']}、subset {d['subset']}、q_mode {d['q_mode']}、比较用键 {d['n_pseudo']} 个",
             f"  示例: 写入侧 {n['a']} 个({n['rows_a']} 位置，包含写入位置的样本 {d['matched']['A']}/{d['matched']['of']})、不希望写入侧 {n['b']} 个({n['rows_b']} 位置)" + (f"、dont_touch {n['dt']} 个({n['rows_dt']} 位置)" if n.get("dt") else "")
             + _replies_text(d.get("replies"))]
    ds_ = d.get("dont_touch_start")
    if ds_ and ds_.get("on") and ds_.get("groups"):          # dont_touch の質問の、返事を書き始める直前の位置を Q に入れた行
        lines.append(f"  dont_touch 问题的、开始写回复紧前方的位置: 已将 {ds_['rows']} 行加入 Q("
                     + "、".join(f"{g} {r}" for g, r in ds_["groups"].items()) + ")")
    elif ds_ and not ds_.get("on") and d.get("dont_touch"):
        lines.append("  dont_touch 问题的、开始写回复紧前方的位置: 未加入 Q(--no-dont-touch-start 或对照集的 dont_touch_start: false)")
    rs = d.get("reply_start")
    if rs:                                                   # mode same: 書き込みたい位置のうち返事の冒頭にあるもの
        lines.append(f"  回复开头: 写入位置中的 {rs['rows']} / {rs['rows_a']} 是决定回复最初 {rs['k']} 个 token 的位置"
                     + (f"(L{rs['layer']})" if rs.get("layer") is not None else "")
                     + f"、占 μ 的权重 {rs['share'] * 100:.0f}%"
                     + (f"(将开始写回复紧前方的位置加入 Q 的 dont_touch: {', '.join(rs['dont_touch_starts'])})" if rs["dont_touch_starts"] else ""))
    lines += [f"  {'层':>4} {'kn':>6} {'放大率':>7} {'q_rank':>6} {'创建时AUC':>9}"
             + (f" {'rows_a':>6} {'rows_b':>6} {'gate_a':>7}" if moe else "")]
    for r in d["layers"]:
        lines.append(f"  {r['layer']:>4} {r['kn']:>6.3f} {r['gain']:>7.2f} {r['q_rank']:>6} {_fc(r['build_auc'], '>9.3f')}"
                     + (f" {r['rows_a']:>6} {r['rows_b']:>6} {_fc(r.get('gate_a'), '>7.3f')}" if moe else "")
                     + ("  ← 弱(kn < 0.05，容易泄漏到不希望写入的位置)" if r["weak"] else ""))
    lines.append("  kn = ‖k‖/‖μ‖(越接近 1，写入位置的平均越位于不希望写入位置子空间之外)、放大率 = 1/kn"
                 + ("。rows_a、rows_b = 该层中存在写入目标行的写入位置、不希望写入位置的数量(专家仅为被选中的位置)，gate_a = 写入位置处 gate、router 权重的平均(LoRA 的 B 除以此值)" if moe else ""))
    for x in d["dropped"]:
        lines.append(f"  L{x['layer']}: 已排除({x['error']})")
    lines += [f"注: {t}" for t in d["notes"]]
    if d.get("samples"):
        lines.append(f"回复示例: {d['samples'][0][:120]!r}")
    lines.append(f"下一步: gne direction key-check --key {d['name']} --contrast {contrast}(默认 subset odd = 创建时未使用的样本)")
    return d, "\n".join(lines), 0


def _start_rho_text(sr: dict | None) -> str:
    """key-check の層の start_rho を 1 行に: 群ごとの、返事の冒頭の位置 0(返事を書き始める直前の位置)での ρ の中央値。"""
    if not sr:
        return ""
    items = ([("不希望写入侧", sr["B"])] if sr.get("B") else []) + list((sr.get("dont_touch") or {}).items())
    out = [f"{g} {_f(v['median'][0], '+.2f')}" for g, v in items if v.get("median") and v["median"][0] is not None]
    return "、".join(out)


@gne_command(direction_app, "key-check", "direction key-check")
def direction_key_check(c: Common, holder: dict,
                        key: str = typer.Option(..., "--key", help="键的名称(NAME~j 表示比较用键 j)"),
                        contrast: Optional[Path] = typer.Option(None, "--contrast", help="要确认的对照集(与创建相同则 --subset odd 为保留集)。省略时用键中保存的创建时样本测量(偏乐观的值)"),
                        subset: str = typer.Option("odd", "--subset", help="使用的问题: odd(默认)/ even / all。与创建重叠时 held_out: false"),
                        layers: Optional[str] = typer.Option(None, "--layers", help="测量层(默认为键的全部层)"),
                        positions: Optional[str] = typer.Option(None, "--positions", help="写入位置(默认为创建键时的值)"),
                        positions_b: Optional[str] = typer.Option(None, "--positions-b", help="不希望写入的位置(默认为创建键时的值)"),
                        skip: Optional[int] = typer.Option(None, "--skip", help="从回复开头丢弃的 token 数(默认为创建键时的值)"),
                        max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="生成回复时的长度(默认为创建键时的值)"),
                        replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                        replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                        iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                        strict: bool = typer.Option(False, "--strict", help="若与创建时使用的样本重叠，则不测量并报错(BAD_ARGS)")):
    """鍵の分離の確認(CHANGE-07 C、生成なしで済む位置なら生成なし)。各位置で鍵の反応の強さ ρ = κᵀx を測り、層ごとに
    書き込みたい位置 対 書き込みたくない位置の AUC(例単位とトークン単位)、key-check の選択性(例ごとの ρ の平均の、
    書き込みたい位置 − 書き込みたくない位置)と比較用の鍵(ラベルをシャッフルして同じ手順で作った鍵)の p95、
    dont_touch の ρ の中央値を出す。作成に使った例と重なると held_out: false(甘い値)。判定の目安: 保留集合で選択性が
    比較用の鍵の p95 を超えること。"""
    s = holder["session"] = c.session()
    d = s.direction_key_check(key, contrast=contrast, subset=subset, layers=parse_layers(layers) if layers else None,
                              positions=positions, positions_b=positions_b, skip=skip, max_tokens=max_tokens,
                              replies_a=replies_a, replies_b=replies_b, iv=_iv_arg(iv), strict=strict,
                              progress=s.progress)
    d.pop("warnings", None)
    src = f"对照集 {d['contrast']}、subset {d['subset']}" if d["contrast"] else "键中保存的创建时样本"
    lines = [f"键 {d['key']}(写入目标 {d['target']}、{src}、mode {d['mode']}、写入位置 {d['positions']}、不希望写入位置 {d['positions_b']}、{d['seconds']:.1f} 秒)"]
    ov = d["overlap"]
    if d["held_out"]:
        lines.append(f"保留集: 是(创建时未使用的样本 {ov['n_checked']} 个)")
    else:
        why = ([f"同一对照集中 subset 重叠(创建 {ov.get('built_subset')}、确认 {ov.get('subset')})"] if ov.get("subset_overlap") else []) \
            + ([f"与创建相同的 (system, 问题) 有 {ov['hash_overlap']} 个"] if ov.get("hash_overlap") else [])
        lines.append("保留集: 否(" + "、".join(why or ["创建时使用的样本"]) + ")。AUC、选择性为偏乐观的值")
    lines.append(f"  {'层':>4} {'AUC':>6} {'AUC(tok)':>8} {'选择性':>7} {'比较用p95':>7} {'比较用AUCp95':>10} {'二分AUC':>7}  判定")
    for r in d["layers"]:
        pa = (r["pseudo_auc_token"] if r["auc_token"] is not None else r["pseudo_auc"])["p95"]
        mark = {True: "○", False: "×"}.get(r["beats_pseudo"], "—")
        line = (f"  {r['layer']:>4} {_fc(r['auc'], '>6.3f')} {_fc(r['auc_token'], '>8.3f')} {_fc(r['selectivity'], '>7.3f')} "
                f"{_fc(r['pseudo_selectivity']['p95'], '>7.3f')} {_fc(pa, '>10.3f')} {_fc(r['b_split_auc'], '>7.3f')}  {mark}")
        dt = "、".join(f"{g} {_f(v, '+.3f')}" for g, v in (r.get("dont_touch_median") or {}).items())
        if dt:
            line += f"  dont_touch 的 ρ 中位数 {dt}"
        st = _start_rho_text(r.get("start_rho"))
        if st:
            line += f"  回复开头的 ρ {st}"
        if r["layer"] == d["best_layer"]:
            line += "  ← 最佳"
        lines.append(line)
    lines.append(("  选择性 = key-check 的选择性(每个样本键反应强度 ρ 的平均，写入位置 − 不希望写入位置)。比较用p95、比较用AUCp95 = "
        "比较用键的选择性、AUC 的 p95(AUC 若存在 token 级 AUC 则为 token 级)。判定 ○ = 选择性超过比较用键的 p95，且 AUC "
        "> 0.5。二分AUC = 将不希望写入位置分成两组创建的键的 AUC(参考值)。回复开头的 ρ = 不希望写入侧、dont_touch "
        "在开始写回复紧前方位置的 ρ 中位数(0.4 以上警告) "))
    ab = d["all_beat_pseudo"]
    lines.append("在全部层超过比较用键: " + {True: "是", False: "否"}.get(ab, "无判定(没有比较用键或不希望写入位置)"))
    lines += [f"注: {t}" for t in d["notes"]]
    if not d["held_out"]:
        lines.append("下一步: 请用创建时未使用的样本(--subset odd 或另一对照集)重新测量")
    elif d["best_layer"] is not None and ab is not False:
        lines.append(f"下一步: gne steer --unit {d['best_layer']}:{_target_who(d['target'])}:keyed:a1@<方向>#{d['key']} -p \"…\" --compare 确认生成(α 为 raw，1 表示直接写入观测到的 A − B 差)")
    return d, "\n".join(lines), 0


@gne_command(direction_app, "keys", "direction keys")
def direction_keys(c: Common, holder: dict):
    """保存した鍵の一覧(<root>/keys/<model_id>/、CHANGE-07)。層・書き込み先・kn・作成の条件。"""
    s = holder["session"] = c.session()
    ks = s.keys()
    lines = [f"键 {len(ks)} 个" + ("" if ks else "(用 gne direction key --contrast <对照集> --layers <层> 创建)")]
    for k in ks:
        if "error" in k:
            lines.append(f"  {k['name']}: 无法读取({k['error']})")
            continue
        kn = " ".join(f"L{L}:{v:.2f}" for L, v in k["kn"].items())
        lines.append(f"  {k['name']}  {k['target']}  q_mode {k['q_mode']}  位置 {k['positions']}  subset {k['subset']}  比较用键 {k['n_pseudo']}  kn {kn}")
    return {"keys": ks}, "\n".join(lines), 0


@gne_command(direction_app, "key-experts", "direction key-experts")
def direction_key_experts(c: Common, holder: dict,
                          contrast: Path = typer.Option(..., "--contrast", help="对照集 YAML(与创建键时相同。例 configs/contrast/zundamon_endings.yaml)"),
                          layers: str = typer.Option(..., "--layers", help="要调查的 MoE 层(必需。例 6,12,18 / 10-16)"),
                          positions: Optional[str] = typer.Option(None, "--positions", help=_KEY_POS),
                          positions_b: Optional[str] = typer.Option(None, "--positions-b", help=_KEY_POS_B),
                          skip: Optional[int] = typer.Option(None, "--skip", help="从回复开头丢弃的 token 数(默认为对照集的 skip，没有则为 0)"),
                          max_tokens: int = typer.Option(96, "--max-tokens", help="生成回复时的长度"),
                          subset: str = typer.Option("even", "--subset", help="使用的问题: even(默认。与键创建相同)/ odd / all"),
                          replies_a: Optional[Path] = typer.Option(None, "--replies-a", help=_KEY_REPLIES_A),
                          replies_b: Optional[Path] = typer.Option(None, "--replies-b", help=_KEY_REPLIES_B),
                          iv: Optional[list[str]] = typer.Option(None, "--iv", help=_KEY_IV),
                          top: int = typer.Option(8, "--top", help="每层显示的专家数(按写入位置被选中的比例从大到小)")):
    """expert:e の鍵(gne direction key --target expert:e)の専門家を選ぶ材料(CHANGE-07 MoE)。鍵と同じ対照集合・位置・返事で、
    書き込みたい位置・書き込みたくない位置で各専門家が選ばれた割合(rate_a・rate_b、例ごとの割合の平均)と、選ばれたときの
    ルーターの重みの平均(weight_a、鍵の gate_a の見込み)を層ごとに出す。専門家の鍵は書き込みたい位置で専門家 e が選ばれた
    行だけでできるので、rate_a の
    大きい専門家ほど行が多く(rows_a)、鍵が例の本文を拾いにくい。MoE でない層は BAD_RECIPE。"""
    Ls = parse_layers(layers)
    if not Ls:
        raise GneError("BAD_ARGS", "请在 --layers 中指定层(不能使用 all)", "例 --layers 6,12,18")
    s = holder["session"] = c.session()
    d = s.direction_key_experts(contrast, layers=Ls, positions=positions, positions_b=positions_b, skip=skip,
                                max_tokens=max_tokens, subset=subset, replies_a=replies_a, replies_b=replies_b,
                                iv=_iv_arg(iv), top=top, progress=s.progress)
    lines = [f"专家被选中的方式(对照集 {d['contrast']}、mode {d['mode']}、写入位置 {d['positions']}、不希望写入位置 {d['positions_b']}、subset {d['subset']}{_replies_text(d.get('replies'))}、{d['seconds']:.1f} 秒)"]
    for lay in d["layers"]:
        lines.append(f"  L{lay['layer']}({lay['n_expert']} 个专家中 {lay['n_used']}、写入位置 {lay['positions_a']} 个、不希望写入位置 {lay['positions_b']} 个，若每个被等概率选中则为 {lay['base_rate']:.2f})")
        lines.append(f"    {'专家':>6} {'rate_a':>6} {'rate_b':>6} {'差':>6} {'lift':>6} {"权重":>6} {"rows_a预期":>9}")
        for r in lay["top"]:
            dt = "、".join(f"{g} {v:.2f}" for g, v in (r.get("rate_dt") or {}).items())
            lines.append(f"    {'E' + str(r['expert']):>6} {r['rate_a']:>6.2f} {r['rate_b']:>6.2f} {r['diff']:>+6.2f} "
                         f"{_fc(r['lift'], '>6.2f')} {_fc(r['weight_a'], '>6.3f')} {r['rows_a']:>9.1f}"
                         + (f"  dont_touch {dt}" if dt else ""))
    lines += [f"注: {t}" for t in d["notes"]]
    best = [(lay["layer"], lay["top"][0]["expert"]) for lay in d["layers"] if lay["top"]]
    if best:
        L, e = best[0]
        lines.append(f"下一步: gne direction key --contrast {contrast} --layers {L} --target expert:{e}(rate_a 越大且 rate_b 越小的专家，写入位置的行越多，键越容易捕捉状态)")
    return d, "\n".join(lines), 0


@gne_command(direction_app, "key-from-behave", "direction key-from-behave")
def direction_key_from_behave(c: Common, holder: dict,
                              behave: str = typer.Option(..., "--behave", help="gne behave 结果名称（<root>/behave/<model_id>/<名称>.json）或路径。必须是应用 CHiPS 配方回答 A 侧问题的结果"),
                              recipe: str = typer.Option(..., "--recipe", help="CHiPS 配方（已保存配方的名称或路径）。键在应用该配方的状态下创建，因此写入输出的 keyed.iv"),
                              direction: Optional[str] = typer.Option(None, "--direction", help="CHiPS 方向名称（输出的 keyed.direction）。省略 --contrast 时，使用创建该方向所用对照集"),
                              contrast: Optional[str] = typer.Option(None, "--contrast", help="CHiPS 方向的对照集（路径或 configs/contrast/ 中的名称）。获取 B 侧问题、validate.match 和 A 侧问题。省略 --direction 时，方向名称使用该 name"),
                              label: Optional[list[str]] = typer.Option(None, "--label", help="视为“留下”的读取方判定标签（例如 --label P。可多个）。省略时用正则表达式判定"),
                              match: Optional[str] = typer.Option(None, "--match", help="视为“留下”的正则表达式（区分大小写）。省略时使用 behave 结果的 --match，若没有则使用对照集的 validate.match（忽略大小写）"),
                              min_rate: float = typer.Option(1.0, "--min-rate", help="纳入行为在种子中出现比例不低于此值的问题（默认 1.0 = 所有种子。A 侧只纳入行为确定出现的问题）"),
                              positions: Optional[str] = typer.Option(None, "--positions", help="写入位置 before:<正则表达式>（默认 before:<validate.match>。若含字母则加 (?i)）。匹配过宽会警告，因此传入对应行为开头的正则表达式"),
                              name: Optional[str] = typer.Option(None, "--name", help="输出名称（默认 <behave 名称>_keyed）。也会成为键的默认名称"),
                              out: Optional[Path] = typer.Option(None, "--out", "-o", help="输出的 JSON 路径（默认 <root>/keyed/<名称>.json。会出现在键标签页的模板中）"),
                              replies: bool = typer.Option(True, "--replies/--no-replies", help="将提取问题的 behave 回复放入输出的 replies（默认。创建键时直接读取而不重新生成）。--no-replies 时在创建键时生成回复"),
                              prompts: Optional[Path] = typer.Option(None, "--prompts", help="交给 behave 的问题集（仅用于未记录问题的旧 behave 结果）"),
                              strict: bool = typer.Option(False, "--strict", help="如果 behave 结果不是应用该配方回答 A 侧问题得到的，则报错（默认仅警告）")):
    """CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問だけを集めて、keyed(鍵つきの書き込み)の対照集合の JSON を作る
    (モデルは動かさない)。user_prompts = 抜き出した質問、dont_touch の ordinary = CHiPS の対照集合の B 側の質問、
    keyed = {direction: CHiPS の方向, alpha: -1, iv: CHiPS のレシピ}。次に鍵のタブか gne direction key --contrast <出力> で鍵を作る。"""
    s = holder["session"] = c.session()
    d = s.direction_key_from_behave(behave, recipe=recipe, direction=direction, contrast=contrast, labels=label or None,
                                    match=match, min_rate=min_rate, positions=positions, name=name, out=out, replies=replies,
                                    prompts=prompts, strict=strict)
    d.pop("warnings", None)                        # 警告はエンベロープの warnings に入っている
    j = d["judge"]
    how = (f"读取方标签 {', '.join(j['labels'])}" if j["method"] == "label"
           else f"正则表达式 {j['match']}" + ("（忽略大小写）" if j["ignore_case"] else "")
           + {"behave": "（behave 的 --match）", "validate": "（对照集的 validate.match）", "match": ""}[j["source"]])
    rate = "所有种子" if j["min_rate"] >= 1 else f"至少 {j['min_rate']:g} 的种子"
    lines = [(f"{d['path']}（" if d["path"] else "未写出（") + f"behave {d['behave']} · {d['n_prompts']} 个问题 · {d['n_seeds']} 个种子，"
             f"配方 {d['recipe']}、方向 {d['direction']}、对照集 {d['contrast']}）",
             f"  留下的问题：{len(d['extracted'])} 个（{how} 在 {rate} 中出现。仅部分种子 {d['n_partial']} 个，未出现 {d['n_none']} 个）",
             f"  写入位置：{d['positions']}" + ("（来自 validate.match）" if d["positions_source"] == "validate" else ""),
             ]
    ps = d["position_share"]
    if ps["share"] is not None:
        lines.append(f"  匹配广度：回复 token 中约 {ps['share'] * 100:.1f}% 是匹配起点（匹配 {ps['starts']} · token {ps['tokens']:g}，"
                     f"{ps['threshold'] * 100:.0f}% 以上过宽）" + ("  ← 过宽" if ps["broad"] else ""))
    h = d["halves"]
    if d["extracted"]:
        lines.append(f"  预计写入位置：创建键（偶数序号）{h['even']['examples']} 个示例 · {h['even']['positions']} 个位置，"
                     f"key-check（奇数序号）{h['odd']['examples']} 个示例 · {h['odd']['positions']} 个位置，无匹配问题 {d['no_match']} 个"
                     + ("" if d["replies"] else "（未放入回复，因此创建键时会重新生成回复而变化）"))
    lines.append(f"  dont_touch 的 ordinary：B 侧问题 {d['ordinary']['n']} 个" + (f"（已去掉与 user_prompts 相同的 {d['ordinary']['removed']} 个）"
                                                                                if d["ordinary"]["removed"] else ""))
    for r in d["extracted"][:12]:
        lines.append(f"    {r['i']:>3} {r['rate']:.2f} 匹配 {r['starts']:>2}  {r['question']}" + ("  （重复）" if r["loop"] else ""))
    if len(d["extracted"]) > 12:
        lines.append(f"    … 另外 {len(d['extracted']) - 12} 个")
    lines += [f"注：{t}" for t in d["notes"]]
    if d["path"]:
        lines.append(f"下一步：gne direction key --contrast {d['path']} --layers <层> --iv {d['iv']}"
                     "（在键标签页选择模板并运行“从方向到 key-check”。方向复用 keyed.direction 中的方向）")
    return d, "\n".join(lines), 0 if d["path"] else 1


@gne_command(direction_app, "key-verify", "direction key-verify")
def direction_key_verify(c: Common, holder: dict,
                         layer: int = typer.Option(..., "--layer", help="层 L"),
                         key: str = typer.Option(..., "--key", help="键的名称"),
                         direction: str = typer.Option(..., "--direction", help="写入方向的方向名称(开头的 - 折叠为 α 的符号)"),
                         alphas: str = typer.Option("1,2", "--alphas", help="要确认的 α(逗号分隔)"),
                         target: str = typer.Option("ffn", "--target", help=("写入目标(与键的写入目标相同): ffn / attn / shared(MoE 层的 shared expert 的 ffn_down_shexp)/ "
        "expert:e(ffn_down_exps 中的专家 e。例 expert:31) ")),
                         direction_layer: str = typer.Option("own", "--direction-layer", help="own(该层方向)/ base(基准层方向)"),
                         text: Optional[list[str]] = typer.Option(None, "--text", help="传入的句子(默认为语尾不同的日语和英语 2 句)。可多个。expert:e 时传入专家 e 会被选中的句子(如包含键写入位置的句子)")):
    """V30(CHANGE-07): keyed の LoRA が式どおり ΔW x = α (κᵀx) d を書くか。無介入で重み行列の入力 x と出力を捕捉し、
    LoRA を当てた出力の差と比べる(|κᵀx| の大きいトークンで相対誤差の中央値 < 3%、鍵の作成に使った書き込みたくない側の例で
    漏れ < 0.1)。
    shared はゲートを掛ける前の ffn_shexp、expert:e はルーターの重みを掛ける前の ffn_moe_down の、専門家 e が選ばれた枠で測り、
    ほかの専門家の枠の差(0 のはず)と、LoRA で選ばれる専門家が変わったトークンの数(0 のはず)も確かめる。"""
    from .keyed import parse_target
    kind, expert = parse_target(target)
    s = holder["session"] = c.session()
    d = s.check_keyed(layer, key=key, direction=direction, alphas=_floats(alphas, "--alphas"), kind=kind, expert=expert,
                      direction_layer=direction_layer, texts=text or None)
    lines = [f"L{layer} {d['tensor']}(键 {d['key']}、方向 {d['direction']}、‖d‖={d['d_norm']:.3g}、kn {d['kn']:.3f}"
             + ("、有 post-norm 校正(一阶近似)" if d.get("post_norm") else "")
             + (f"、用 gate_a {d['gate']:.3g} 除了 B" if d.get("gate") is not None else "")
             + (f"、用 residual_scale {d['res_scale']:g} 除了 B" if d.get("res_scale") is not None else "") + ")"]
    for r in d["rows"]:
        lines.append(f"  α {r['alpha']:+g}: 相对误差中位数 {_f(r['rel_err_median'], '.2e')}(|κᵀx| 的前 {r['n_big']}/{r['n_tokens']} {'槽位' if kind == 'expert' else 'token'})、cos {_f(r['cos_median'], '.4f')}、对不希望写入侧的泄漏 {_f(r['leak_b'], '.3g')}"
                     + (f"、其他专家的槽位差 {r['other_max']:.2g}、选择方式变化 {r['selection_changed']} 个 token"
                        if "other_max" in r else "") + f"  {'○' if r['ok'] else '×'}")
    if d.get("note"):
        lines.append(f"注: {d['note']}")
    lines.append(f"V30 {'合格' if d['v30'] else '不合格'}(相对误差 < 3%"
                 + ("、对不希望写入侧的泄漏 < 0.1" if d["judged_leak"] else "、对不希望写入侧的泄漏无判定")
                 + ("、其他专家的槽位差 < 1e-3、选择方式变化 0" if kind == "expert" else "") + ")")
    return d, "\n".join(lines), 0 if d["v30"] else 1


@gne_command(heads_app, "causal", "heads causal")
def heads_causal(c: Common, holder: dict,
                 name: str = typer.Option(..., "--name", help="方向名称"),
                 contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集（默认使用创建方向时的对照集）"),
                 mode: str = typer.Option("ablate", "--mode", help="ablate（将输出置 0 并测量下降幅度，用于初次筛选）/ marginal（从当前组提高 s 时的效率 η，用于重新分配）。两者排名不一致"),
                 layers: str = typer.Option("auto", "--layers", help="auto（按层停止，并对下降最大的 --top-layers 个层）/ all / 10-16 / 1,5,9-12"),
                 candidates: Optional[str] = typer.Option(None, "--candidates", help="all / contrib:N（直接贡献前 N）/ L:Hh,…（显式指定）"),
                 score_layers: Optional[str] = typer.Option(None, "--score-layers", help="测量投影值的层（默认基准层及其后 2 层）"),
                 primary: Optional[int] = typer.Option(None, "--primary", help="输出 causal / contrib / ratio 的层（默认基准层）"),
                 base_recipe: Optional[Path] = typer.Option(None, "--base-recipe", help="marginal：当前组的配方"),
                 base_unit: Optional[list[str]] = typer.Option(None, "--base-unit", help="marginal：当前组（L:Hh:sS@NAME，可多个）"),
                 delta_s: float = typer.Option(3.0, "--delta-s", help="marginal：提高 s 的量"),
                 hard: int = typer.Option(16, "--hard", help="marginal：难生效示例的数量"),
                 hard_subset: Optional[str] = typer.Option(None, "--hard-subset", help="marginal：已保存的难生效示例（名称或路径）"),
                 b_side_max: Optional[float] = typer.Option(None, "--b-side-max", help="B 侧投影值变化绝对值超过该值的注意力头会标记为排除"),
                 top_layers: int = typer.Option(4, "--top-layers", help="--layers auto 时逐层测量的层数"),
                 positions: str = typer.Option("last", "--positions"), subset: str = typer.Option("all", "--subset"),
                 b_examples: int = typer.Option(16, "--b-examples", help="测量对 B 侧影响所用示例数（0 为全部）"),
                 out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名称（默认使用方向名称）"),
                 direction_layer: str = typer.Option("own", "--direction-layer", help="marginal 投影方向：own（注意力头所在层）/ base（基准层）"),
                 sort: str = typer.Option("auto", "--sort", help="表格排序：auto（ablate 用 causal_max，marginal 用 eta）/ causal / contrib / ratio / eta"),
                 show: int = typer.Option(30, "--show"),
                 max_examples: Optional[int] = typer.Option(None, "--max-examples", help="停止并测量的间题数(等间隔抽样。默认全部)"),
                 include_score_layers: bool = typer.Option(False, "--include-score-layers", help="将测量投影值的层及其之后的注意力头也作为候选(默认排除。只是直接改变测量量，行为不易变化)")):
    """因果スクリーニング(CHANGE-06 C): ヘッドを止めた(ablate)/ 強めた(marginal)ときの、生成なしの射影の値の下がり方。
    各行に直接の寄与 contrib(r̂ᵀW[:, blk]·x の平均)と causal / contrib の比を並べる(1 より大きければ間接の経路、
    小さければ補い合い。射影の値を測る層より後ろのヘッドは contrib が 0 なので causal_max で見る)。
    ablate と marginal は順位が入れ替わる: 最初の候補出しは ablate、配り直しは marginal。"""
    s = holder["session"] = c.session()
    d = s.heads_causal(name, contrast=contrast, mode=mode, layers=layers, candidates=candidates,
                       score_layers=_ints(score_layers), primary=primary, base_recipe=base_recipe, base_units=base_unit,
                       delta_s=delta_s, hard=hard, hard_subset=hard_subset, b_side_max=b_side_max, top_layers=top_layers,
                       positions=positions, subset=subset, b_examples=b_examples, name=out_name,
                       direction_layer=direction_layer, max_examples=max_examples, include_score_layers=include_score_layers)
    P = d["primary"]
    lines = [f"{d['path']}（{d['n_rows']} 个注意力头，mode={mode}，测量层 L{','.join(map(str, d['score_layers']))}（主 L{P}），{d['seconds']:.0f} 秒）",
             "无干预投影值：" + "、".join(f"L{L} A {x['A']:.2f} / B {x['B']:.2f}" for L, x in d["base_scores"].items())]
    if d.get("layers"):
        lines.append("按层（停止全部注意力头时的下降幅度）：" + " ".join(
            f"L{r['layer']}:{r['causal_max']:+.2f}" for r in sorted(d["layers"], key=lambda r: -r["causal_max"])[:12]))
    rows = d["rows"]
    if sort != "auto":
        rows = sorted(rows, key=lambda r: -(r.get(sort) if r.get(sort) is not None else -1e9))
    lines.append(f"  {'地址':>9} {'类型':>6} " + " ".join(f"{'L' + k:>7}" for k in map(str, d["score_layers"]))
                 + f" {'contrib':>8} {'ratio':>6}" + (f" {'eta':>7}" if mode == "marginal" else "") + f" {'B变化':>7} {'B正交':>7}")
    for r in rows[:show]:
        lines.append(f"  {('L' + str(r['layer']) + ' H' + str(r['head'])):>9} {r['kind']:>6} "
                     + " ".join(f"{r['score_drop'][str(L)]:>+7.2f}" for L in d["score_layers"])
                     + f" {_f(r['contrib'], '+8.3f')} {_f(r['ratio'], '6.2f')}"
                     + (f" {r['eta']:>+7.3f}" if mode == "marginal" else "")
                     + f" {_f(r.get('b_side_delta'), '+7.2f')} {_f(r.get('drift_orth_b'), '7.4f')}"
                     + ("  排除" if r.get("excluded") else "") + ("  （组内）" if r.get("in_base") else ""))
    if mode == "marginal" and d.get("spearman_ablate_marginal") is not None:
        lines.append(f"ablate 与 marginal 的排名相关性（Spearman，{d['n_compared']} 个注意力头）：{d['spearman_ablate_marginal']:+.2f}")
    lines.append("下一步：gne heads allocate --name " + d["direction"] + " --target <A 侧投影值目标>")
    return d, "\n".join(lines), 0


def _allocate_note(r: dict, d: dict) -> list[str]:
    """止まった理由の説明。目標に届いた組は黙っている(s_max に当たった組の比の話は、見落とすと結果を読み違える)。"""
    stop, c = r.get("stop"), r.get("capped")
    if stop == "target" or stop is None:
        return []
    if stop == "s_max" and c:
        return [f"      L{c['layer']} H{c['head']} 达到上限 s={c['s']:g}。由于按比例分配，其他注意力头也一起停止。"
                "可尝试 --method greedy 或 --s-max（仅提高上限时，目标较远时强度可能升到会破坏模型的程度）"]
    why = {"drift": f"B 侧正交分量的偏移超过 {d.get('drift_max')}，因此在超过前停止",
           "no_gain": "提高任何注意力头的投影值都不会下降，因此停止", "s_max": "所有注意力头都达到了 s 的上限",
           "steps": "达到提升次数上限"}
    return [f"      {why[stop]}"] if stop in why else []


@gne_command(heads_app, "writers", "heads writers")
def heads_writers(c: Common, holder: dict,
                  name: str = typer.Option(..., "--name", help="方向名称"),
                  contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集(默认为创建方向时的对照集)"),
                  ref_layer: Optional[int] = typer.Option(None, "--ref-layer", help="测量贡献的方向层(默认为基准层)"),
                  expert_layers: Optional[str] = typer.Option(None, "--expert-layers", help="还按专家细分的层(例 36,42-46)"),
                  max_examples: int = typer.Option(64, "--max-examples", help="使用的问题数(等间隔抽样)"),
                  top: int = typer.Option(12, "--top", help="每层显示的专家数"),
                  out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名称(默认为方向名称)")):
    """方向の差を、層ごとに attention・専門家・共有の専門家の書き込みに分ける。どこを探すかを決めるために使う。"""
    s = holder["session"] = c.session()
    d = s.heads_writers(name, contrast=contrast, ref_layer=ref_layer, expert_layers=parse_layers(expert_layers) if expert_layers else None,
                        max_examples=max_examples, name=out_name, top=top)
    gap = d["gap"] or 1.0
    t = d["totals"]
    lines = [f"{d['path']}(L{d['ref_layer']} 的方向，A {d['n']['A']} 问、B {d['n']['B']} 问、{d['positions']}、A−B 的差 {d['gap']:.2f}、求和误差 {d['sum_error'] * 100:.1f}%)",
             f"合计: attention {t['attention'] / gap:+.0%}、专家 {t['moe'] / gap:+.0%}、共享 {t['shared'] / gap:+.0%}",
             f"{'层':>4} {'种类':<8} {'attention':>10} {'专家':>8} {'共有':>8}"]
    for r in d["rows"]:
        if max(abs(r["attention"]), abs(r["moe"]), abs(r["shared"])) >= 0.02 * abs(gap):
            lines.append(f"{r['layer']:>4} {r['kind']:<8} {r['attention']:>+10.2f} {r['moe']:>+8.2f} {r['shared']:>+8.2f}")
    for L, rows in d["experts"].items():
        lines.append(f"L{L} 的专家(按差值从大到小): " + " ".join(f"E{x['expert']}({x['diff']:+.2f}、A {x['rate_a']:.0%}/B {x['rate_b']:.0%})" for x in rows))
    return d, "\n".join(lines), 0


@gne_command(heads_app, "allocate", "heads allocate")
def heads_allocate(c: Common, holder: dict,
                   name: str = typer.Option(..., "--name", help="方向名称"),
                   contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集（默认使用创建方向时的对照集）"),
                   candidates: Optional[str] = typer.Option(None, "--candidates", help="heads causal 的结果（名称或 .causal.jsonl。默认使用方向名称）"),
                   target: float = typer.Option(..., "--target", help="难生效示例平均投影值目标（例 −3。B 侧平均值附近是起点）"),
                   score_layer: Optional[int] = typer.Option(None, "--score-layer", help="测量投影值的层（默认使用 heads causal 的最后一层）"),
                   n_heads: str = typer.Option("6,5,4", "--n-heads", help="要创建的组内注意力头数量"),
                   hard: int = typer.Option(16, "--hard", help="难生效示例数量"),
                   hard_subset: Optional[str] = typer.Option(None, "--hard-subset", help="已保存的难生效示例（名称或路径）"),
                   proportional_to: str = typer.Option("eta", "--proportional-to", help="eta（效率）/ causal（停止时的效果）"),
                   from_zero: bool = typer.Option(True, "--from-zero/--add", help="从 0 重新分配 / 加到当前 s"),
                   b_side_max: Optional[float] = typer.Option(None, "--b-side-max", help="移除会改变 B 侧投影值的注意力头的阈值"),
                   base_recipe: Optional[Path] = typer.Option(None, "--base-recipe", help="当前组（若无则从 ablate 的前列创建）"),
                   delta_s: float = typer.Option(3.0, "--delta-s"), pool: int = typer.Option(30, "--pool", help="用于测量效率的候选（当前组 + 效果前 N）"),
                   s_max: Optional[float] = typer.Option(None, "--s-max", help="s 的上限（默认 proportional 为 30，greedy 无上限）"),
                   method: str = typer.Option("proportional", "--method", help="proportional（按 s ∝ η 整体提高）/ greedy（每次提高最有效的单个注意力头）"),
                   remain_threshold: float = typer.Option(0.0, "--remain-threshold", help="视为“剩余”的投影值"),
                   direction_layer: str = typer.Option("own", "--direction-layer"),
                   out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名称（默认使用方向名称）"),
                   candidate_layers: Optional[str] = typer.Option(None, "--candidate-layers", help="将候选注意力头限制到这些层(例 3-44)"),
                   include_score_layers: bool = typer.Option(False, "--include-score-layers", help="将测量投影值的层及其之后的注意力头也作为候选(默认排除。只是直接改变测量量，行为不易变化)")):
    """効率ベースの強さ配分(CHANGE-06 D): 効きにくい例を選び、効率 η を測って s ∝ η で配り、全体の大きさを二分探索して、
    ヘッド数を変えた組の比較表を出す。残る件数と B 側のずれが同程度なら、ヘッド数が最小の組を推す。"""
    s = holder["session"] = c.session()
    d = s.heads_allocate(name, contrast=contrast, candidates=candidates, target=target, score_layer=score_layer,
                         n_heads=_ints(n_heads) or [6, 5, 4], hard=hard, hard_subset=hard_subset,
                         proportional_to=proportional_to, from_zero=from_zero, b_side_max=b_side_max,
                         base_recipe=base_recipe, delta_s=delta_s, pool=pool, s_max=s_max, method=method,
                         remain_threshold=remain_threshold, direction_layer=direction_layer, name=out_name,
                         include_score_layers=include_score_layers, candidate_layers=parse_layers(candidate_layers))
    lines = [f"{d['path']}（L{d['score_layer']}、目标 {d['target']:g}、A {d['n_a']} 例、难生效示例 {d['hard']['n']}、"
             f"无干预 A {d['base_a_mean']:.2f} / B {d['base_b_mean']:.2f}、{d['seconds']:.0f} 秒）",
             f"  {'数量':>3} {'最大s':>6} {'剩余':>5} {'A最大':>7} {'B正交':>7} {'B变化':>7} {'秒':>5}  组"]
    for r in d["table"]:
        if r.get("skipped"):
            lines.append(f"  {r['n_heads']:>3}  {r['skipped']}")
            continue
        lines.append(f"  {r['n_heads']:>3} {r['s_max']:>6.2f} {r['remain']:>5} {r['a_max']:>7.2f} {r['b_drift_orth']:>7.4f} "
                     f"{r['b_side_delta']:>+7.2f} {r['seconds']:>5.0f}  " + " ".join(r["units"])
                     + ("" if r["reached"] else "  （目标未达到）") + ("  ← 推荐" if r["recommended"] else ""))
        lines += _allocate_note(r, d)
    if d["recommended"] is not None:
        rp = next(r["recipe"] for r in d["table"] if r.get("recommended"))
        lines.append(f"配方：{rp}。确认：gne eval --recipe {rp} --random-controls 16")
    return d, "\n".join(lines), 0


@gne_command(app, "behave")
def behave_cmd(c: Common, holder: dict,
               prompts: Path = typer.Option(..., "--prompts", exists=True, help="问题集（JSON / YAML 的字符串或对话历史列表。若是对照集则为 A 侧问题）"),
               recipe: Optional[list[Path]] = typer.Option(None, "--recipe", help="应用的配方(多个则叠加)"),
               match: Optional[str] = typer.Option(None, "--match", help="统计回复中匹配的正则表达式"),
               judge: Optional[Path] = typer.Option(None, "--judge", exists=True, help="给读取方的提示词文件({q} 为问题，{a} 为回复)"),
               labels: str = typer.Option("R,P,C", "--labels", help="读取方的判定字符(取回复中最后出现的)"),
               max_tokens: int = typer.Option(400, "--max-tokens"),
               thinking: bool = typer.Option(False, "--thinking", help="让模型带思考回答，只统计 </think> 之后"),
               temperature: float = typer.Option(0.0, "--temperature", help="0 则为 greedy"),
               seeds: Optional[str] = typer.Option(None, "--seeds", help="有温度时，每个种子各 1 次(1,2,3)"),
               name: str = typer.Option("behave", "--name", help="保存名")):
    """振る舞いを数える: レシピを当てて質問集に答えさせ、正規表現の一致・読み手の判定・空・繰り返し・長さの上限で切れた数を出す。
    返事は <root>/behave/<model_id>/<名前>.json に残し、画面には数だけを出す。長さで結果が変わることがあるので、--max-tokens は十分に取る。"""
    s = holder["session"] = c.session()
    d = s.behave(prompts, recipes=list(recipe or []), match=match, judge=judge.read_text(encoding="utf-8") if judge else None,
                 labels=labels, max_tokens=max_tokens, thinking=thinking, temperature=temperature, seeds=_ints(seeds), name=name)
    lines = [f"{d['path']}({d['n_prompts']} 问、{d['max_tokens']} token" + ("、带思考" if thinking else "")
             + (f"、温度 {temperature:g}" if temperature > 0 else "、greedy") + ")"]
    for r in d["runs"]:
        c_ = r["counts"]
        lines.append(f"  种子 {r['seed']}: " + (f"一致 {c_['match']}、" if "match" in c_ else "")
                     + (" ".join(f"{k} {v}" for k, v in c_["labels"].items()) + "、" if "labels" in c_ else "")
                     + f"空 {c_['empty']}、重复 {c_['loop']}、截断 {c_['at_limit']}、平均 {c_['tokens']:g} 个 token")
    return d, "\n".join(lines), 0


@app.command("recipe-combine")
def recipe_combine(recipes: list[Path] = typer.Argument(..., exists=True, help="叠加的配方(2 个以上)"),
                   out: Path = typer.Option(..., "--out", "-o", help="导出的配方"),
                   name: Optional[str] = typer.Option(None, "--name", help="配方名称(默认为输出文件名)")):
    """複数のレシピを 1 つにまとめる。方向の違う CHiPS の組を重ねるときなどに使う(各ユニットは自分の方向を持つ)。"""
    from .steer import Recipe, combine_recipes
    r = combine_recipes([Recipe.load(p) for p in recipes], name=name or out.stem)
    r.save(out)
    dirs = sorted({u.direction for u in r.units if u.mode in ("proj", "keyed") and u.direction})
    keys = sorted({u.key for u in r.units if u.mode == "keyed" and u.key})      # CHANGE-07
    print(f"{out}({len(r.units)} 个单元" + (f"、方向 {', '.join(dirs)}" if dirs else "")
          + (f"、键 {', '.join(keys)}" if keys else "") + ")")


@gne_command(headproj_app, "ablate", "headproj export")
def headproj_ablate(c: Common, holder: dict,
                    direction: str = typer.Option(..., "--direction", help="由 gne direction 创建的方向名称"),
                    layers: str = typer.Option("all", "--layers", help="消除层: all / 1-47 / 1,5,9-12"),
                    per_layer: bool = typer.Option(False, "--per-layer", help="按层消除该层测得的 r_L(默认为基准层的 r)"),
                    scale: float = typer.Option(1.0, "--scale", help="消除比例(1 为全部)"),
                    out: Optional[Path] = typer.Option(None, "--out", "-o", help="输出（默认 <root>/export/<方向>-heads-ablate.lora.gguf）"),
                    bake: bool = typer.Option(False, "--bake", help="不生成 LoRA，而生成改写写入残差的权重矩阵的 GGUF(可保持分割)")):
    """attention の書き込みだけから方向を消す(abliteration の attention 部分)。指定した層の全ヘッドに proj(s=--scale)を当てる
    レシピと同じで、LoRA は層ごとに 1 本。MLP が MoE で gne ablate の LoRA が使えないモデルでも使える。"""
    from .steer import Recipe, Unit
    s = holder["session"] = c.session()
    arch = s.weights.arch
    Ls = parse_layers(layers) or list(range(s.meta.n_layer))
    units = [Unit(L, None, 0.0, "head", mode="proj", k=float(scale), head=h, direction=direction)
             for L in Ls if arch.layers[L].head_dim and arch.layers[L].head_writer for h in range(arch.layers[L].n_head)]
    if not units:
        raise GneError("BAD_ARGS", "指定的层没有注意力头")
    rec = Recipe(units, s.meta.model_id, name=f"{direction}-heads-ablate", default_direction=direction,
                 direction_layer="own" if per_layer else "base")
    out = out or s.store.export_dir() / (f"{direction}-heads-ablate.gguf" if bake else f"{direction}-heads-ablate.lora.gguf")
    d = s.headproj_export(rec, out, bake=bake, requant="keep" if bake else "q8_0")
    lines = [f"{d['path']}({d['bytes'] / 1e6:.2f} MB、{'烧录' if bake else 'LoRA'}、{len(units)} 个注意力头、{len({u.layer for u in units})} 层、{'按层的 r_L' if per_layer else '基准层的 r'}、比例 {scale:g})"]
    if not bake:
        lines.append("用法: llama-cli -m <model> --lora " + d["path"])
    return d, "\n".join(lines), 0


@gne_command(headproj_app, "export", "headproj export")
def headproj_export(c: Common, holder: dict, recipe: Optional[Path] = _RECIPE, unit: Optional[list[str]] = _UNIT,
                    out: Optional[Path] = typer.Option(None, "--out", "-o", help=("输出（默认 <root>/export/<配方名>.lora.gguf；--bake 时为 -headproj.gguf。仅 keyed 的配方为 "
        "<配方名>-keyed.lora.gguf / -keyed.gguf） ")),
                    bake: bool = typer.Option(False, "--bake", help="不生成 LoRA，只生成重新量化改写后权重矩阵(写入残差的部分)的 GGUF"),
                    in_path: Optional[Path] = typer.Option(None, "--in", help="--bake 的原始 GGUF（默认 --model）"),
                    requant: Optional[str] = typer.Option(None, "--requant", help=("--bake 改写张量的类型: auto / q8_0 / f16 / keep(原为 F32、F16、BF16 时保持该类型。专家张量 "
        "ffn_down_exps(keyed 的 expert:e)保持原类型，不编辑的专家保持字节不变。K-quant、IQ 的专家为 q8_0，分割 GGUF "
        "则保持原类型)。默认为 q8_0，包含 keyed 时为 auto: 按张量处理，原为 Q8_0 的保持 Q8_0(改为 F16 会改变计算方法，反而与 "
        "LoRA 版产生偏差)，其余(K-quant、IQ、F16 等)为 f16。仅对保持 Q8_0 时 rank 1 的小编辑有 1 成以上被舍入抹掉的张量(结果 "
        "edit_kept < 0.9)用 f16。专家张量也遵循相同规则(改为 f16 会使整体变为 F16 而变大) ")),
                    key: Optional[str] = typer.Option(None, "--key", help="省略 #KEY 的 keyed 单元的键(由 gne direction key 创建)"),
                    into: Optional[Path] = typer.Option(None, "--into", help="--bake：烧录到另一个 GGUF（同一结构、未量化的 F16 / BF16 等）。方向使用此模型（--model）的。不使用 GPU"),
                    via_f16: bool = typer.Option(False, "--via-f16", help="--bake：以 F16 写入，再用 llama-quantize 恢复到原类型"),
                    imatrix: Optional[Path] = typer.Option(None, "--imatrix"), llama_quantize: Optional[str] = typer.Option(None, "--llama-quantize"),
                    verify: bool = typer.Option(False, "--verify", help=("--bake: 比较烧录版与 LoRA 版(V26、V33: 方向上的投影值、KL、最可能 token、ppl。最后一层投影值的合格线为，加上写入几乎为 0 "
        "的 LoRA 与无 LoRA 的差 = 仅由计算方法不同产生的误差) ")),
                    contrast: Optional[Path] = typer.Option(None, "--contrast", help="--verify 评估投影值使用的对照集（默认使用创建方向时的对照集）"),
                    direction: Optional[str] = typer.Option(None, "--direction", help="省略 @NAME 的 proj / keyed 单元的方向"),
                    direction_layer: Optional[str] = typer.Option(None, "--direction-layer", help="own（注意力头所在层方向）/ base（基准层方向）"),
                    lora_f16: bool = typer.Option(False, "--lora-f16", help="以 F16 写入 LoRA(大小减半。精度足够)"),
                    strip_meta: bool = typer.Option(False, "--strip-meta", help="不保留 gne 记录（gne.* 元数据、配方、名称标记、旁边的 .headproj.json）（供分发的版本）")):
    """proj のユニット(L:Hh:sS@NAME、レシピの mode: proj)を LoRA アダプタ GGUF か焼き込み GGUF に書き出す(CHANGE-06 E、gne gain の別名)。
    同じ層・同じ方向のヘッドは階数 1 の LoRA に畳む。スケールパッチ(CHANGE-05 D)では表せないので、焼き込みは書き換える
    テンソルだけ量子化し直す。鍵つきの書き込み(keyed、L:ffn:keyed:aα@NAME#KEY、CHANGE-07)も受ける(層・鍵ごとに階数 1。
    配るなら LoRA が第一候補。焼き込みの既定は auto: 元が Q8_0 のテンソルは Q8_0 のまま、それ以外は f16)。"""
    if in_path and c.model and in_path.resolve() != c.model.resolve():
        raise GneError("BAD_ARGS", "--in 与 --model 指向不同文件")
    s = holder["session"] = c.session(in_path)
    r = s.recipe(units=unit, path=recipe, default_direction=direction, direction_layer=direction_layer, default_key=key)
    out = out or s.default_export_path(r, bake=bake, kind="headproj")
    d = s.headproj_export(r, out, bake=bake, requant=requant, via_f16=via_f16, imatrix=str(imatrix) if imatrix else None,
                          llama_quantize=llama_quantize, verify=verify, contrast=contrast, strip_meta=strip_meta, into=into,
                          lora_f16=lora_f16)
    lines = [f"{d['path']}（{d['bytes'] / 1e6:.2f} MB，{'烧录' if bake else 'LoRA'}"
             + ("，无 gne 记录" if d.get("strip_meta") else "") + "）"]
    if d.get("local_path_keys"):
        lines.append("注意：元数据中仍残留疑似本地路径的字符串：" + ", ".join(d["local_path_keys"]))
    for L, x in (d.get("proj_layers") or {}).items():
        lines.append(f"  L{L}: " + ", ".join(f"H{int(h)} s{sv:g}@{dn}" for h, sv, dn in x["heads"])
                     + f"（秩 {x['rank']}" + ("" if x["folded"] else "，方向混杂，无法折叠为秩 1") + "）")
    lines += _keyed_export_lines(d, requant_line=not (bake and d.get("parts")))
    if bake and d.get("parts"):                 # 分割 GGUF: 同じ型で書き戻した
        how = {"keep": "以原类型写回", "auto": "以 auto 写回(Q8_0 保持原类型，其余为 F16"
               + (f"。{len(d['fallback'])} 个张量因编辑会被淹没而用 F16" if d.get("fallback") else "") + ")"}
        lines.append(f"改写: {len(d['modified'])} 个张量以" + how.get(d["requant"], f"{d['requant']} 重新写入")
                     + f"(重新量化损失最大为编辑大小的 {d['requant_error_max'] * 100:.1f}%)、分割 {len(d['parts'])} 个文件。记录 {d['sidecar']}")
        if d.get("expert_bytes_added"):
            lines.append(f"  专家张量增大了 {d['expert_bytes_added'] / 1e6:.1f} MB(不编辑的专家也会转换。--requant keep 则保持原类型)")
    elif bake:
        lines.append("改写: " + ", ".join(_mod_text(m) for m in d["modified"]))
    else:
        lines.append("用法：llama-cli -m <model> --lora " + d["path"])
    ok = True
    v = d.get("verify")
    if v:
        lines.append(f"{'V33' if r.keyed_units and not r.proj_units else 'V26'}: 投影值 {'一致' if all(x['ok'] for x in v['scores'].values()) else '不一致'}、首个 token 的 KL {v['kl_first_token']:.2e}、最大概率 token 的一致率 {v['top1_match'] * 100:.0f}%、ppl 差异 {_f(v['ppl_diff_pct'], '+.3f')}%（{'通过' if v['pass'] else '未通过'}）"
                     + (f"。分歧示例的差距 {', '.join(f'{g:.3f}' for g in v['top1_mismatch_gaps'])} nats" if v["top1_mismatch_gaps"] else ""))
        lines.append("  投影值差的最大 / 合格线: " + "、".join(
            f"L{k[:-1]} {k[-1]} {x['max_abs_diff']:.3g} / {x.get('threshold', x['resolution']):.3g}{'' if x['ok'] else ' ×'}"
            for k, x in v["scores"].items()))
        nz = v.get("noise")
        if nz:
            lines.append(f"  最后一层 L{nz['layer']} 的合格线 = 容许误差(投影值标准差的 1 成)+ 仅由计算方法不同产生的误差 {nz['max_abs_diff']:.3g}(写入几乎为 0 的 LoRA(倍率 {nz['lora_scale']:g})与无 LoRA 的投影值差的最大、{nz['seconds']:.1f} 秒)")
        ok = v["pass"]
    return d, "\n".join(lines), 0 if ok else 1


@gne_command(headproj_app, "check", "headproj check")
def headproj_check(c: Common, holder: dict, layer: int = typer.Option(..., "--layer"), head: int = typer.Option(..., "--head"),
                   direction: str = typer.Option(..., "--direction", help="方向名称"),
                   ss: str = typer.Option("1,2,6", "--s", help="要确认的 s")):
    """V24・V25: proj の LoRA で、ヘッドが残差に書き込む重み行列の出力の差が numpy の計算(方向成分を s 倍ぶん引く)と合い、直交成分が変わらないか。"""
    s = holder["session"] = c.session()
    d = s.check_head_proj(layer, head, direction, ss=[float(x) for x in ss.split(",")])
    lines = [f"L{layer} H{head}（{d['out_tensor']}，方向 {direction} 的 L{d['direction_from']}，注意力头写入与方向的 |cos| 中位数 "
             f"{_f(d['dir_cos_head'], '.3f')})"]
    for r in d["rows"]:
        lines.append(f"  s {r['s']:g}：相对误差中位数 {r['rel_err_median']:.2e}，方向分量 {r['par_after_over_before']:+.3f} 倍"
                     f"（期望 {1 - r['s']:+g}）、正交分量变化 {r['orth_change_rel_median']:.2e}  {'○' if r['ok'] else '×'}")
    lines.append(f"  对比 gain ×{d['gain']['k']:g}：正交分量变化 {_f(d['gain']['orth_change_rel_median'], '.3f')}")
    lines.append(f"V24 {'通过' if d['v24'] else '未通过'}、V25 {'通过' if d['v25'] else '未通过'}")
    return d, "\n".join(lines), 0 if d["ok"] else 1


@gne_command(heads_app, "pipeline", "heads pipeline")
def heads_pipeline(c: Common, holder: dict,
                   spec: Path = typer.Option(..., "--spec", help="对照集 JSON 或 YAML（name、user_prompts 或按侧 prompts_a / prompts_b、system_a / system_b 或 recipe_a / recipe_b，可选 pipeline 设置）"),
                   target: Optional[str] = typer.Option(None, "--target", help="难生效示例投影值目标：数值、B（消到 B 侧平均值）、mid（默认，A 与 B 中点）"),
                   n_heads: Optional[str] = typer.Option(None, "--n-heads", help="要比较的组的注意力头数量（默认 6,5,4）"),
                   max_examples: Optional[int] = typer.Option(None, "--max-examples", help="停止测量阶段使用的问题数量（默认 64，等间距抽稀）"),
                   positions: Optional[str] = typer.Option(None, "--positions", help="创建方向的位置：last（默认）/ response 等"),
                   top_layers: Optional[int] = typer.Option(None, "--top-layers"), hard: Optional[int] = typer.Option(None, "--hard"),
                   include_score_layers: bool = typer.Option(False, "--include-score-layers", help="将测量投影值的层及其之后的注意力头也作为候选(默认排除。只是直接改变测量量，行为不易变化)"),
                   fresh: bool = typer.Option(False, "--fresh", help="不复用已创建的方向和因果结果，从头重新创建"),
                   yes: bool = _YES):
    """対照集合から、方向づくり → 基準層 → 因果スクリーニング(層ごと → ヘッド 1 個ずつ止める)→ 強さの配分(効き目に比例した
    最初の組を二分探索 → 射影の値が残りやすい質問で効率を測って 0 から配り直す → ヘッド数を変えた組の比較)を 1 回で回す。"""
    s = holder["session"] = c.session()
    over = {"target": target, "n_heads": _ints(n_heads), "max_examples": max_examples, "positions": positions,
            "top_layers": top_layers, "hard": hard, "resume": False if fresh else None,
            "include_score_layers": True if include_score_layers else None}
    _gate(s, "head_pipeline", {"spec": str(spec), **{k: v for k, v in over.items() if v is not None}}, yes, None)
    d = s.heads_pipeline(spec, **over)
    sc, ca, al = d["score"], d["causal"], d["allocate"]
    lines = [f"{d['path']}（{d['n_prompts']} 个问题，{d['seconds']:.0f} 秒"
             + (f"，复用已生成的{'、'.join({'direction': '方向', 'causal': '因果'}[x] for x in d['reused'])}" if d.get("reused") else "")
             + "）",
             f"1. 方向 {d['name']}（{d['direction']['positions']}，{d['direction']['seconds']:.0f} 秒）",
             f"2. 基准层 L{sc['base_layer']}（cohen_d 最大，停止测量阶段使用 {sc['n'].get('A')} 个问题）",
             f"3. 因果：测量层 L{','.join(map(str, ca['score_layers']))}、{ca['n_rows']} 个注意力头、前列 " +
             " ".join(f"L{r['layer']}H{r['head']}({r['causal_max']:+.2f}{'，仅破坏所以排除' if r.get('excluded') else ''})" for r in ca["top"][:6])
             + f"（{ca['seconds']:.0f} 秒）"
             + "".join(f"\n   层 L{r['layer']} 停止后连中立输入也会大幅崩坏（偏移 {r['drift_orth_b']:.2f}），因此未选择"
                       for r in (ca.get("layers") or []) if r.get("excluded"))]
    if al.get("initial"):
        lines.append(f"4. 初始组（按效果成比例，整体二分搜索）：" + " ".join(al["initial"]["heads"]))
    lines.append(f"5. 按效率重新分配的组（目标 {d['target']:.2f} = {d['target_from']}）：")
    for r in al["table"]:
        if r.get("skipped"):
            lines.append(f"   {r['n_heads']} 个：{r['skipped']}")
            continue
        lines.append(f"   {r['n_heads']} 个：最大 s {r['s_max']:.2f}，剩余 {r['remain']}/{al['n_a']}，B 正交 {r['b_drift_orth']:.4f}  "
                     + " ".join(r["units"]) + ("  ← 推荐" if r["recommended"] else "") + ("" if r["reached"] else "（目标未达到）"))
    if d["recommended"]:
        lines.append(f"配方：{d['recommended']['recipe']}。确认：gne eval --recipe {d['recommended']['recipe']} --random-controls 16")
    return d, "\n".join(lines), 0



def _candidates_text(d: dict) -> str:
    j = d["judge"]
    ho = d.get("held_out") or {}
    lines = [f"{d['path']}（{len(d['candidates'])} 个候选 × A {len(d['prompts']['A'])} 个问题、B {len(d['prompts']['B'])} 个问题、"
             + ("greedy" if d["sampling"] is None else f"温度 {d['sampling']['temperature']:g}、种子 {d['sampling']['seed']}")
             + (f"，叠加配方 {Path(d['with_recipe']).name}" if d.get("with_recipe") else "")
             + (f"，{d['seconds']:.0f} 秒" if d.get("seconds") else "") + "）"]
    if not (ho.get("A") and ho.get("B")):
        lines.append("  注意：正在使用调整时用过的问题生成（" + "、".join(k for k in "AB" if not ho.get(k)) +
                     " 侧）。通过 --prompts 传入未用于调整的问题，可用于决定")
    sc = d.get("scores") or {}
    has_sc = bool(sc)
    hdr = f"  {'候选':<12} {'侧':>2} {'方向':<4} {'一致':>5} {'d4':>5} {'对侧变化':>6}"
    if has_sc:
        hdr += f" {'P':>5} {'C':>5} {'目的':>5}"
    lines.append(hdr)
    goal_ja = {"remove": "消除", "add": "强化", None: "—"}
    for c in d["candidates"]:
        for side in c["judge_sides"]:
            lo, other = side.lower(), "b" if side == "A" else "a"
            ln = (f"  {c['name']:<12} {side:>2} {goal_ja[c.get('goal')]:<4} {_metric(c.get('rate_' + lo), '.0%'):>5} "
                  f"{c['d4_' + lo]:>5.2f} {_metric(None if c['name'] == 'base' else 1 - c['same_as_base_' + other], '.0%'):>6}")
            if has_sc:
                v = (sc.get(c["name"]) or {}).get(side)
                g = c.get("goal") or ("remove" if side == "A" else "add")
                ln += f" {v['P']:>5.2f} {v['C']:>5.2f} {v[g]:>5.2f}" if v else f" {'—':>5} {'—':>5} {'—':>5}"
            lines.append(ln + ("  ← allocate 推荐" if c.get("recommended") and c.get("scale") == 1.0 else ""))
    lines.append("  一致 = 符合 validate.match 的回复比例，d4 = 重复少，对侧变化 = 未评分侧与无干预回复不同的比例"
                 "（使用同一随机种子时，若不变则一致）")
    if has_sc:
        lines.append("  目标 = 消除方向为 min(6 − P, C)，强化方向为 min(P, C) 的平均")
        if d.get("scores_missing"):
            lines.append(f"  未评分编号：{len(d['scores_missing'])} 个")
        for g, b in (d.get("best") or {}).items():
            c = next(x for x in d["candidates"] if x["name"] == b["name"])
            lines.append(f"{goal_ja[g]}方向最佳：{b['name']}（目标 {b['value']:.2f}、无干预 {_f(b['base'], '.2f')}）。"
                         f"确认：gne eval --recipe {c['recipe']} --doses {c['scale']:g}")
    else:
        lines.append(f"评分：将 {j['round']} 的 {j['n']} 份按 {j['rubric']} 的标准评分，并在 {j['scores']} 中写入“编号 P C”")
        lines.append(f"汇总：gne heads candidates --tally {d['name']}")
    return "\n".join(lines)


@gne_command(heads_app, "candidates", "heads candidates")
def heads_candidates(c: Common, holder: dict,
                     alloc: Optional[str] = typer.Option(None, "--alloc", help="heads allocate / pipeline 的分配结果（名称或 .alloc.json）"),
                     tally: Optional[str] = typer.Option(None, "--tally", help="汇总已写入评分的候选（名称或 .candidates.json）"),
                     prompts: Optional[Path] = typer.Option(None, "--prompts", help="未用于调整的问题（YAML / JSON 列表或 {A: [...], B: [...]}）"),
                     scales: str = typer.Option("0.5,1,1.5,2,3", "--scales", help="作用于组内全部 s 的倍率（负值表示在 B 侧强化的方向）"),
                     n_heads: Optional[str] = typer.Option(None, "--n-heads", help="使用的组的注意力头数量（默认比较表中的全部）"),
                     max_prompts: int = typer.Option(24, "--max-prompts", help="每侧问题上限（过多则等间距抽稀）"),
                     max_tokens: int = typer.Option(120, "--max-tokens"),
                     sampling: str = typer.Option("llama_server", "--sampling", help="greedy、llama_server、common，或数值 JSON"),
                     sample_seed: int = typer.Option(0, "--sample-seed", help="采样种子（候选与无干预使用同一种子）"),
                     judge_b: bool = typer.Option(False, "--judge-b", help="也将 B 侧（普通状态）的回复放入评分文件"),
                     contrast: Optional[Path] = typer.Option(None, "--contrast", help="对照集（默认使用创建方向时的对照集）"),
                     out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名称（默认使用配分名称）"),
                     with_recipe: Optional[Path] = typer.Option(None, "--with-recipe", help="叠加到所有候选（包括无干预 base）上的配方（如神经元点火）"),
                     rubric: Optional[Path] = typer.Option(None, "--rubric", help="评分标准文件（默认根据目标行为创建）"),
                     yes: bool = _YES):
    """強さの候補づくり: 配分の比較表の組 × 全体の倍率(と無介入)で、A 側と B 側の質問に返事を作る。正の倍率は A 側の振る舞いを
    消す向き、負の倍率(例 --scales -0.5,-1)は B 側で強める向きで、それぞれの側の返事を、どの候補か伏せた採点用のファイルに書く。
    判定の一致率・繰り返し・反対側の変化も出す。採点を書いたら --tally で集計する。"""
    s = holder["session"] = c.session()
    if tally:
        if alloc:
            raise GneError("BAD_ARGS", "--alloc 与 --tally 不能一起使用")
        d = s.heads_candidates_tally(tally)
        return d, _candidates_text(d), 0
    if not alloc:
        raise GneError("BAD_ARGS", "请指定 --alloc（创建候选）或 --tally（汇总评分）")
    sm = sampling.strip()
    sm_v = json.loads(sm) if sm.startswith("{") else sm
    sc = [float(x) for x in scales.split(",") if x.strip()]
    nh = _ints(n_heads) if n_heads else None
    _gate(s, "heads_candidates", {"alloc": alloc, "scales": sc, "n_heads": nh, "max_prompts": max_prompts,
                                  "max_tokens": max_tokens}, yes, None)
    d = s.heads_candidates(alloc, prompts=prompts, scales=sc, n_heads=nh, max_prompts=max_prompts, max_tokens=max_tokens,
                           sampling=sm_v, seed=sample_seed, judge_b=judge_b, contrast=contrast, name=out_name,
                           with_recipe=with_recipe, rubric=rubric)
    return d, _candidates_text(d), 0


@gne_command(heads_app, "refine", "heads refine")
def heads_refine(c: Common, holder: dict,
                 name: str = typer.Option(..., "--name", help="分配结果（流水线名称或 heads allocate 的保存名称）"),
                 n_heads: Optional[int] = typer.Option(None, "--n-heads", help="测量剩余项的组的注意力头数量（默认推荐组）"),
                 new_heads: str = typer.Option("3,2,1", "--new-heads", help="要追加的组的注意力头数量（逗号分隔）"),
                 out_name: Optional[str] = typer.Option(None, "--out-name", help="保存名称（默认 <name>_r1，其他组则为 _r2 …）"),
                 fresh: bool = typer.Option(False, "--fresh", help="不复用已创建的方向和因果结果"),
                 yes: bool = _YES):
    """組を当てても目標まで下がらない A 側の質問(残り)だけで、追加のヘッドを探す。残りの質問と B 側の質問で、組を当てたまま
    方向を作り直し、一括(基準層 → 因果 → 配分)で足すヘッドと強さを決め、元の組と合わせたレシピを作って残りを測り直す。"""
    s = holder["session"] = c.session()
    nh = _ints(new_heads) or [3, 2, 1]
    _gate(s, "heads_refine", {"name": name, "new_heads": nh}, yes, None)
    d = s.heads_refine(name, n_heads=n_heads, new_heads=nh, out_name=out_name, resume=not fresh)
    rm, cb, pp = d["remaining"], d["combined"], d["pipeline"]
    lines = [f"{d['path']}（{d['seconds']:.0f} 秒）",
             f"1. 使用 {d['n_heads']} 个注意力头的组 " + " ".join(d["base_units"]) + f"，L{d['score_layer']} 投影值超过 {d['threshold']:.2f} 的 A 侧问题：{rm['n']}/{rm['n_a']} 个" + (f"（较少，因此按投影值从高到低用 {rm['used']} 个问题创建方向）" if rm["padded"] else ""),
             f"2. 用剩余问题运行流水线 → 方向 {pp['direction']}" + (f"（复用已生成的{'、'.join({'direction': '方向', 'causal': '因果'}[x] for x in pp['reused'])}）" if pp["reused"] else "")]
    for r in pp["table"]:
        if r.get("skipped"):
            lines.append(f"   {r['n_heads']} 个：{r['skipped']}")
            continue
        lines.append(f"   {r['n_heads']} 个：最大 s {r['s_max']:.2f}，剩余 {r['remain']}  " + " ".join(r["units"])
                     + ("  ← 推荐" if r["recommended"] else "") + ("" if r["reached"] else "（目标未达到）"))
    rows = d.get("combined_rows") or []
    if rows:
        lines.append(f"3. 与原始组合并后的组（用原始方向 L{d['score_layer']} 重新测量。合并前：剩余 {rm['n']}/{rm['n_a']} 个问题，"
                     f"B 侧正交分量偏移 {rows[0]['b_drift_orth_base']:.4f}）")
        for c in rows:
            lines.append(f"   + {c['n_new']} 个（共 {c['n_heads']} 个注意力头）：剩余 {c['remain_after']} 个问题，A 平均值 {c['a_mean_before']:.2f} → "
                         f"{c['a_mean_after']:.2f}，B 偏移 {c['b_drift_orth']:.4f}  {c['recipe']}"
                         + ("  ← 代表性" if cb and c["recipe"] == cb["recipe"] else ""))
        lines.append(f"代表性方案也保存到 {Path(cb['recipe']).with_name(d['name'] + '.combined.yaml')}"
                     + ("（推荐组）" if cb["chosen_by"] == "recommended" else "（没有达到目标的组，因此选择合并后剩余最少的一组）"))
        lines.append(f"确认：gne eval --recipe {cb['recipe']} --random-controls 16")
    else:
        lines.append("3. 无法创建追加组，因此没有合并后的配方")
    return d, "\n".join(lines), 0


def _fieldtest_text(d: dict) -> str:
    def fmt(section, key, v):
        if v is None:
            return "—"
        if section == "ppl":
            return f"{v:.3f}"
        if key == "tokens":
            return f"{v:.0f}"
        if key == "tail_distinct":
            return f"{v:.2f}"
        return f"{v * 100:.1f}%"
    title = {"long": f"长任务（{d['settings']['n_long']} 题，最多 {d['settings']['long_tokens']} token）",
             "tools": f"工具调用（{d['settings']['n_tools']} 题）", "ppl": "perplexity"}
    lines = [f"{d['path']}（{d['seconds'] / 60:.0f} 分钟）", f"被测文件：{d['target']['file']}" + (f" + LoRA {Path(d['target']['lora']).name}" if d["target"]["lora"] else "")]
    if d["reference"]:
        lines.append(f"对照文件：{d['reference']['file']}")
    lines += [f"注意: {w}" for w in d.get("warnings") or []]
    rows = d["compare"]
    if rows is None:                                  # 比べる相手が無いとき: 対象の数字だけ
        run = d["runs"][-1]
        rows = []
        for sec in ("long", "tools"):
            rows += [{"section": sec, "metric": k, "label": k, "reference": None, "target": v, "worse": False}
                     for k, v in (run.get(sec) or {}).items() if not k.startswith("n")]
        for name, m in (run.get("bench") or {}).items():
            rows += [{"section": f"bench:{name}", "metric": k, "label": k, "reference": None, "target": v, "worse": False} for k, v in m.items() if k != "n"]
        for name, m in (run.get("prompts") or {}).items():
            rows += [{"section": f"prompts:{name}", "metric": k, "label": k, "reference": None, "target": v, "worse": False}
                     for k, v in m.items() if k != "n"]
        rows += [{"section": "ppl", "metric": k, "label": k, "reference": None, "target": v, "worse": False} for k, v in (run.get("ppl") or {}).items()]
    cur = None
    for r in rows:
        if r["section"] != cur:
            cur = r["section"]
            key = cur.split(":", 1)[-1]
            head = title.get(cur) or (f"基准测试 {key}（{(d['settings'].get('bench') or {}).get(key, '?')} 题，最多 {d['settings'].get('bench_tokens')} token）" if cur.startswith("bench:")
                                      else f"问题集 {key}({d['settings']['prompts'].get(key, '?')} 题，最多 {d['settings']['prompt_tokens']} token）")
            lines.append(head)
            lines.append(f"  {'':<26} {'对照文件':>9} {'被测文件':>9}")
        lines.append(f"  {r['label']:<26} {fmt(r['section'], r['metric'], r['reference']):>9} {fmt(r['section'], r['metric'], r['target']):>9}"
                     + (f"  仅一侧正确：对照 {r['only_reference']} / 被测 {r['only_target']}（p = {r['p']:.3f}）" if "only_target" in r else "")
                     + ("  ← 恶化" if r.get("worse") else ""))
    if d["same_level"] is not None:
        lines.append("结论：全部处于同一水平（比例按 2 题计，perplexity 差异在 1% 以内视为同一水平）" if d["same_level"]
                     else "结论：有项目恶化：" + ", ".join(d["worse"]))
    lines.append(f"生成的文本：{d['replies_path']}")
    return "\n".join(lines)


@gne_command(app, "fieldtest")
def fieldtest_cmd(c: Common, holder: dict,
                  against: Optional[Path] = typer.Option(None, "--against", help="对照文件（例如原模型）。省略时只测量 --model（若有 --lora，则与未应用 LoRA 的同一文件比较）"),
                  lora: Optional[str] = typer.Option(None, "--lora", help="应用于 --model 的 LoRA（路径 或 路径:倍率）"),
                  name: Optional[str] = typer.Option(None, "--name", help="保存名称（默认为文件名）"),
                  prompts: Optional[list[str]] = typer.Option(None, "--prompts", help="自定义问题集 名称=文件（JSON 或 YAML 列表，可重复）"),
                  match: Optional[str] = typer.Option(None, "--match", help="应用于问题集回复正文的正则表达式。输出匹配率"),
                  long_tasks: Optional[Path] = typer.Option(None, "--long-tasks", help="长任务题目（默认 configs/fieldtest/long_tasks.json）"),
                  tool_tasks: Optional[Path] = typer.Option(None, "--tool-tasks", help="工具调用题目（默认 configs/fieldtest/tool_tasks.json）"),
                  long_tokens: int = typer.Option(3072, "--long-tokens"), tool_tokens: int = typer.Option(1536, "--tool-tokens"),
                  prompt_tokens: int = typer.Option(1536, "--prompt-tokens"), temperature: float = typer.Option(0.6, "--temperature"),
                  presence_penalty: float = typer.Option(0.0, "--presence-penalty", help="采样中的 presence penalty（按模型卡设置）"),
                  n_parallel: int = typer.Option(4, "--n-parallel", help="同时生成的请求数"),
                  reasoning_effort: Optional[str] = typer.Option("medium", "--reasoning-effort", help="传给聊天模板的值"),
                  template_kwargs: Optional[str] = typer.Option(None, "--template-kwargs", help='传给聊天模板的 JSON（例如 {"enable_thinking": false} 可关闭思考）'),
                  llama_server: Optional[str] = typer.Option(None, "--llama-server", envvar="GNE_LLAMA_SERVER", help="llama-server 的路径"),
                  server_args: Optional[str] = typer.Option(None, "--server-args", help="传给 llama-server 的额外参数"),
                  ppl: bool = typer.Option(True, "--ppl/--no-ppl", help="也测量 perplexity"),
                  skip_long: bool = typer.Option(False, "--skip-long"), skip_tools: bool = typer.Option(False, "--skip-tools"),
                  bench: Optional[str] = typer.Option(None, "--bench", help="基准测试（逗号分隔）：gsm8k（默认 250 题）、mmlu_pro（默认 300 题）、math500（默认 250 题）、ifeval（默认 250 题，math500 和 ifeval 需要 bench extra）。数据会下载到 <root>/fieldtest/data"),
                  bench_n: Optional[int] = typer.Option(None, "--bench-n", help="基准测试题数（固定种子抽取）"),
                  bench_tokens: int = typer.Option(4096, "--bench-tokens", help="基准测试回复上限"),
                  yes: bool = _YES):
    """实际使用测试：按真实用法（llama-server、开启思考、开启采样）测量长任务、工具调用、问题集、基准测试和 perplexity，
    并比较文件。eval 用短问题检查同一文件在不同倍率下的表现；此命令可直接测量烧录后的文件。"""
    s = holder["session"] = c.session()
    args = {"against": str(against) if against else None, "lora": lora, "prompts": list(prompts or []), "long_tokens": long_tokens,
            "prompt_tokens": prompt_tokens, "n_parallel": n_parallel, "skip_long": skip_long, "skip_tools": skip_tools,
            "long_tasks": str(long_tasks) if long_tasks else None, "tool_tasks": str(tool_tasks) if tool_tasks else None,
            "bench": bench, "bench_n": bench_n, "bench_tokens": bench_tokens}
    _gate(s, "fieldtest", args, yes, None)
    try:
        tkw = json.loads(template_kwargs) if template_kwargs else None
    except ValueError as e:
        raise GneError("BAD_ARGS", f"--template-kwargs 必须写成 JSON：{e}") from e
    d = s.fieldtest(against=against, lora=lora, name=name, template_kwargs=tkw, prompts=list(prompts or []), match=match, long_tasks=long_tasks, tool_tasks=tool_tasks,
                    long_tokens=long_tokens, tool_tokens=tool_tokens, prompt_tokens=prompt_tokens, temperature=temperature, presence_penalty=presence_penalty, n_parallel=n_parallel,
                    reasoning_effort=reasoning_effort, llama_server=llama_server, server_args=server_args, ppl=ppl, skip_long=skip_long, skip_tools=skip_tools,
                    bench=bench, bench_n=bench_n, bench_tokens=bench_tokens)
    return d, _fieldtest_text(d), 0

if __name__ == "__main__":
    app()

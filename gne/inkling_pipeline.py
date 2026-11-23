"""Inkling 2 の CHiPS の流れ(gne heads pipeline・レシピの LoRA・焼き込み)。docs/inkling2.md の「CHiPS」。

ほかのモデルの流れ(api.heads_pipeline → unitwrites)と同じ段(方向 → 書き込みの順位づけ → 1 個ずつ確かめる → 組を作る →
応答での確かめ)を、このモデルの形に合わせて流す。決めたこと(段階 3 の試しで確かめた形):
- 方向は MLP が読む混ぜた入力(段ごと、48 行)の A − B。層 6〜17 は 2・3 回目を繰り返しのゲートの後で読み、部品の射影の基準は
  その層の最後の回(3 回目)の段。ほかの層は 1 つの段。最後に混ぜた出力の A − B も 49 行目(番号 48)に持つ
- 射影は短い畳み込みの今のトークンの倍率で直した向き normalize(g ⊙ r) を引く LoRA(inkling_chips.lora_tensors_all)。専門家の
  ニューロンは ffn_down_exps の 3 次元(専門家ごとの倍率を差分に入れる)
- 部品は ヘッド・attention 全体・MLP 全体・専門家のニューロン。流れが始まる層(0・6・12・18)の attention と、層 0〜5 は候補に入れない
  (層 0〜5 は include_before で入れる)
- 測る点は 各区分の最後の段(5・17・29・41・47)と最後に混ぜた出力(番号 48、組を作る目標)
- 方向と測る位置(positions): last(質問の最後のトークン)/ answer(思考の後の、答えの始まり)/ response(答えのトークン全部)/
  reasoning(思考のトークン全部)/ after:<正規表現>(答えの中の一致の直後)/ all / assistant(質問のトークン)。answer・response・
  reasoning・after は、書き換える前のモデルの返事(greedy)を作って、その上で読む
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .steer import GneError

OUT = 48                        # 方向・測る点の番号: 最後に混ぜた出力(段 0〜47 の後)
GATE = "after"
POSITIONS = ("last", "answer", "response", "reasoning", "all", "assistant")
REPLY_KINDS = ("answer", "response", "reasoning", "after")
REASON_TOKENS = 1024            # 返事を作るときの長さの上限(思考 + 答えの始まり)
ANSWER_TOKENS = 96              # response・after で読む答えのトークンの上限(ほかのモデルの --response-tokens と同じ)
RANK_ROWS = 16                  # 書き込みの順位づけで 1 つの会話から読む位置の上限(等間隔に間引く)
POSITION_NOTE = ("该模型总是在回答前写思考。用问题最后一个 token（last）创建的方向，在写思考和回答的位置上"
                 "有时只留下约一成的差异（在公开模板上测得）。若要针对回答的语气或写法，请也尝试 answer（思考之后、"
                 "回答开头）或 response（回答的全部 token）。")
UNIT_NOTE = "第 6～17 层的权重在 3 次重复中共享，因此改写这些层的组件会作用于全部 3 次。"


# --- 位置 ------------------------------------------------------------------------------------------------------------

def parse_positions(spec) -> tuple[str, str | None]:
    """positions の書き方 → (種類, 正規表現)。"""
    s = "last" if spec in (None, "") else str(spec).strip()
    if s.startswith("after:"):
        pat = s[len("after:"):]
        if not pat:
            raise GneError("BAD_ARGS", f"positions {s!r} 中没有正则表达式")
        try:
            rx = re.compile(pat)
        except re.error as e:
            raise GneError("BAD_ARGS", f"positions {s!r} 的正则表达式无效：{e}") from e
        if rx.fullmatch(""):
            raise GneError("BAD_ARGS", f"positions {s!r} 的正则表达式匹配空字符串（无法确定位置）")
        return "after", pat
    if s not in POSITIONS:
        raise GneError("BAD_ARGS", f"Inkling 2 的 positions 必须为 {' / '.join(POSITIONS)} / after:<正则表达式>（{s!r}）", POSITION_NOTE)
    return s, None


def _text_id(engine) -> int:
    from .inkling import TEXT_TOKEN
    ids = engine.tokenize(TEXT_TOKEN, parse_special=True)
    if len(ids) != 1:
        raise GneError("INTERNAL", f"回答开头的分隔符 {TEXT_TOKEN} 没有成为单个 token")
    return int(ids[0])


def base_replies(session, items: dict[str, list[tuple[int, list[int]]]], *, sha: str, budget: int = REASON_TOKENS + ANSWER_TOKENS,
                 progress=None, sides: dict | None = None) -> dict[str, dict[int, list[int]]]:
    """その側のモデルの greedy の返事(思考から答えまで、budget トークンまで)。items = {side: [(質問の番号, ids)]}。sides は側ごとの
    LoRA(側をレシピで作る対照集合 = 残りから追加の土台。無ければ書き換える前のモデル)。
    <root>/inkling/<model_id>/replies/<対照集合の sha>-<budget>.json に覚えて使い回す(sha は側のレシピも含む)。"""
    e = session.engine
    p = session.store.root / "inkling" / session.meta.model_id / "replies" / f"{sha}-{budget}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    todo = [(sd, i, ids) for sd, xs in items.items() for i, ids in xs if f"{sd}:{i}" not in cache]
    if todo:
        e.free_steer()
        ctx = e.steer_ctx()
        e.set_lora(ctx, None)
        e.set_cvec(ctx, None)
        longest = max(len(ids) for _, _, ids in todo) + budget
        per = max(1, min(int(e.n_seq_max), int(e.n_ctx) // max(longest, 1)))
        done = 0
        chunks = []
        for sd in "AB":
            xs = [x for x in todo if x[0] == sd]
            chunks += [xs[k:k + per] for k in range(0, len(xs), per)]
        for chunk in chunks:
            e.set_lora(ctx, (sides or {}).get(chunk[0][0]) or None)
            rs = e.generate_greedy(ctx, [ids for _, _, ids in chunk], max_tokens=budget, stop_on_eog=True)
            for (sd, i, _), r in zip(chunk, rs, strict=True):
                cache[f"{sd}:{i}"] = [int(t) for t in r.token_ids]
            done += len(chunk)
            p.write_text(json.dumps(cache), encoding="utf-8")
            if progress:
                progress(done, len(todo), "Inkling 2：回复" if sides else "Inkling 2：改写前的回复")
        e.set_lora(ctx, None)
    return {sd: {i: cache[f"{sd}:{i}"] for i, _ in xs} for sd, xs in items.items()}


def side_loras(session, cs) -> dict[str, list | None]:
    """側をレシピで作る対照集合(残りから追加の土台を両側に当てる)の、側ごとの LoRA。Inkling 2 は射影の LoRA だけ。"""
    if not cs.has_recipes:
        return {"A": None, "B": None}
    from .reverse import side_interventions
    ivs = side_interventions(session, cs)
    if any(v for v, _ in ivs.values()):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 中，侧配方不能使用 control vector（add 单元）")
    session.engine.allow_lora_trial = True
    return {sd: (ivs[sd][1] or None) for sd in "AB"}


def run_with(rd, ids: list[int], wanted, lora) -> dict:
    """Reader.run と同じく 1 本を流して捕まえる。lora(側の LoRA)を当てて。"""
    e = rd.e
    rd.cap.set_wanted(wanted)
    ctx = e.probe_ctx(rd.cap, need=max(rd.need, len(ids)))
    e.set_lora(ctx, lora or None)
    try:
        rd.cap.reset()
        e.clear_memory(ctx)
        e.prefill(ctx, [ids])
        return rd.cap.take()
    finally:
        e.set_lora(ctx, None)


def _strip_eog(engine, toks: list[int]) -> list[int]:
    r = list(toks)
    while r and engine.is_eog(r[-1]):
        r.pop()
    return r


def rows_for(engine, kind: str, pattern: str | None, ids: list[int], start: int, reply: list[int] | None,
             text_id: int) -> tuple[list[int], list[int]] | None:
    """1 つの会話の (読む列, 読む位置)。返事が要る種類で答えの始まりが無ければ None(思考の途中で長さの上限に届いた)。
    位置 p の値は「列の p+1 番目のトークンを決める」位置(質問の最後のトークン = 返事の最初のトークンを決める位置)。"""
    P = len(ids)
    if kind == "last":
        return ids, [P - 1]
    if kind == "all":
        return ids, list(range(1, P))
    if kind == "assistant":
        return ids, list(range(max(int(start), 1), P))
    r = _strip_eog(engine, reply or [])
    if text_id not in r:
        return None
    k = r.index(text_id)                                     # 返事の中の答えの始まりの区切り
    if kind == "answer":
        return ids + r[:k + 1], [P + k]
    if kind == "reasoning":
        return ids + r[:k + 1], list(range(P - 1, P + k))
    ans = r[k + 1:k + 1 + ANSWER_TOKENS]
    full = ids + r[:k + 1] + ans
    if kind == "response":                                   # 答えのトークンを決める位置(区切りの位置から、答えの最後の 1 つ前まで)
        pos = list(range(P + k, P + k + len(ans)))
        return (full, pos) if pos else None
    from .keyed_build import select_positions             # after:<正規表現>: 答えの文字の中の一致の直後
    pos = [int(x) for x in select_positions(engine, ans, f"after:{pattern}", n_prompt=P + k + 1)]
    pos = [x for x in pos if x < len(full)]
    return (full, pos) if pos else None


def _thin(xs: list, n: int | None) -> list:
    if not n or len(xs) <= n:
        return list(xs)
    return [xs[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(xs) - 1, int(n)).round()})]


def contrast_items(session, cs, positions: str, *, max_examples: int | None = None, progress=None) -> dict:
    """対照集合の両側の、読む会話と位置。{"items": {side: [(ids, rows)]}, "index": {side: [質問の番号]}, "skipped": {side: n},
    "prompts": {side: [質問]}}。max_examples で等間隔に間引く。"""
    from .reverse import contrast_sequences
    e = session.engine
    kind, pat = parse_positions(positions)
    seqs = contrast_sequences(e, cs)
    idx = {sd: _thin(list(range(len(seqs[sd]))), max_examples) for sd in "AB"}
    sides = side_loras(session, cs)
    replies = None
    if kind in REPLY_KINDS:                       # 返事はその側のモデル(側のレシピ = 残りから追加の土台を当てたモデル)が書く
        replies = base_replies(session, {sd: [(i, seqs[sd][i][0]) for i in idx[sd]] for sd in "AB"}, sha=cs.sha, progress=progress,
                               sides=sides if cs.has_recipes else None)
    tid = _text_id(e)
    items: dict[str, list] = {"A": [], "B": []}
    used: dict[str, list[int]] = {"A": [], "B": []}
    skipped = {"A": 0, "B": 0}
    for sd in "AB":
        for i in idx[sd]:
            ids, start = seqs[sd][i]
            got = rows_for(e, kind, pat, ids, start, None if replies is None else replies[sd][i], tid)
            if got is None:
                skipped[sd] += 1
                continue
            items[sd].append(got)
            used[sd].append(i)
    for sd in "AB":
        if len(items[sd]) < 2:
            raise GneError("BAD_CONTRAST", f"{sd} 侧能够读取位置 {positions} 的对话只有 {len(items[sd])} 个"
                                           + (f"（其中 {skipped[sd]} 个在思考途中达到长度上限）" if skipped[sd] else ""),
                           "请将 positions 设为 last，或增加问题数量")
    return {"items": items, "index": used, "skipped": skipped,
            "prompts": {sd: [cs.prompts(sd)[i] for i in used[sd]] for sd in "AB"}, "kind": kind, "sides": sides,
            "replies_from": None if kind not in REPLY_KINDS else ("side_recipe" if cs.has_recipes else "unedited")}


# --- 方向 ------------------------------------------------------------------------------------------------------------

def _names(plan) -> dict[int, str]:
    """段 → その段の基準の捕捉の名前(2・3 回目はゲートの後の ffn_in、ほかは ffn_mix)、OUT → out_mix。"""
    from .inkling_chips import ref_kind
    out = {st["step"]: f"{ref_kind(st, GATE)}-{st['step']}" for st in plan.steps()}
    out[OUT] = "out_mix"
    return out


def make_direction(session, cs, name: str, positions: str, *, source: str, max_examples: int | None = None, progress=None) -> dict:
    """段ごとの MLP の入力(基準の種類)と最後に混ぜた出力の、A 側の平均 − B 側の平均。方向の .npz(層の番号 = 段の番号、48 = 出力)。"""
    from .direction import DirectionSet
    from .inkling_read import Reader, want
    t0 = time.perf_counter()
    ci = contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
    rd = Reader(session, progress=progress)
    rd.require_patch()
    plan = rd.plan
    names = _names(plan)
    sums = {sd: None for sd in "AB"}
    n_rows = {"A": 0, "B": 0}
    rd.fit([ids for sd in "AB" for ids, _ in ci["items"][sd]])
    tot = sum(len(ci["items"][sd]) for sd in "AB")
    done = 0
    for sd in "AB":
        for ids, rows in ci["items"][sd]:
            cap = run_with(rd, ids, want({"ffn_mix", "ffn_in", "out_mix"}), ci["sides"][sd])
            T = len(ids)
            v = np.stack([cap[names[k]].reshape(T, -1)[rows].astype(np.float64).mean(0) for k in sorted(names)])
            sums[sd] = v if sums[sd] is None else sums[sd] + v
            n_rows[sd] += len(rows)
            done += 1
            if progress:
                progress(done, tot, "Inkling 2: 方向")
    D = sums["A"] / len(ci["items"]["A"]) - sums["B"] / len(ci["items"]["B"])
    Ls = sorted(names)
    path = session.store.direction_dir(session.meta.model_id) / f"{name}.npz"
    U = np.stack([D[k] / max(float(np.linalg.norm(D[k])), 1e-30) for k in range(len(Ls))])
    np.savez(path, layers=np.array(Ls, dtype=np.int32), **{f"layer_{L}": D[k].astype(np.float32) for k, L in enumerate(Ls)},
             **{f"scale_{L}": np.ones(D.shape[1], np.float32) for L in Ls}, cos=(U @ U.T).astype(np.float32),
             method="inkling_ffn_input", positions=str(positions), contrast_sha=cs.sha, source=source,
             n_rows_A=n_rows["A"], n_rows_B=n_rows["B"], inkling_gate=GATE, same_reply=False)
    ds = DirectionSet.load(path)
    return {"path": str(path), "positions": str(positions), "n_rows": n_rows, "n_items": {sd: len(ci["items"][sd]) for sd in "AB"},
            "skipped": ci["skipped"], "replies_from": ci["replies_from"], "seconds": time.perf_counter() - t0, "ds": ds}


def is_inkling_direction(ds) -> bool:
    return str(ds.meta.get("method") or "") == "inkling_ffn_input"


def require_inkling_direction(ds) -> None:
    if not is_inkling_direction(ds):
        raise GneError("BAD_DIRECTION", f"方向 {ds.name} 不是以 Inkling 2 的形式（每段的 MLP 输入）创建的",
                       "请使用通过 gne heads pipeline（以 Inkling 2 运行）创建的方向")


def layer_refs_of(plan, ds) -> dict[int, tuple[int, np.ndarray]]:
    """保存されている層 → (基準の段, その段の方向の単位ベクトル)。繰り返しの層は最後の回の段。"""
    require_inkling_direction(ds)
    out = {}
    for L in range(plan.n_stored):
        s = plan.steps_of(L)[-1]
        out[L] = (s, ds.unit(s).astype(np.float64))
    return out


# --- 部品 → LoRA ---------------------------------------------------------------------------------------------------

def chips_unit(p) -> list:
    """ProjHead → inkling_chips の部品 [層, 種類, 番号, 強さ]。"""
    if p.kind == "head":
        return [int(p.layer), "head", int(p.head), float(p.s)]
    if p.kind in ("attn", "ffn"):
        return [int(p.layer), p.kind, None, float(p.s)]
    if p.kind == "neuron":
        if getattr(p, "expert", None) is None:
            raise GneError("BAD_RECIPE", f"{p.spec()}：Inkling 2 的 MLP 是 MoE，因此神经元需要带专家编号（L:Ee:Nn:sS@方向）")
        return [int(p.layer), "neuron", [int(p.expert), int(p.head)], float(p.s)]
    raise GneError("BAD_RECIPE", f"{p.spec()}：Inkling 2 中可投影的组件为头、attention 整体、整个 MLP 和专家神经元")


def check_unit(session, p) -> None:
    """Inkling 2 の部品の範囲と、候補から外している部品の知らせ用の確かめ(範囲外は GneError)。"""
    plan = session.weights.inkling
    li = session.weights.arch.layers[0]
    if not 0 <= int(p.layer) < plan.n_stored:
        raise GneError("BAD_RECIPE", f"第 {p.layer} 层超出范围（0～{plan.n_stored - 1}。Inkling 2 的组件使用已保存的 24 层编号）")
    if p.kind == "head" and not 0 <= int(p.head) < li.n_head:
        raise GneError("BAD_RECIPE", f"L{p.layer} 的头 {p.head} 超出范围（0..{li.n_head - 1}）")
    if p.kind == "neuron":
        e = getattr(p, "expert", None)
        if e is None:
            chips_unit(p)
        if not 0 <= int(e) < li.n_expert:
            raise GneError("BAD_RECIPE", f"L{p.layer} 的专家 {e} 超出范围（0..{li.n_expert - 1}）")
        if not 0 <= int(p.head) < li.n_ff:
            raise GneError("BAD_RECIPE", f"L{p.layer} E{e} 的神经元 {p.head} 超出范围（0..{li.n_ff - 1}）")


def _merge(a: dict, b: dict) -> dict:
    """同じ行列の LoRA をランクの向きにつなぐ(違う方向の部品が同じ行列に入るとき)。"""
    out = dict(a)
    for k, (A, B) in b.items():
        if k not in out:
            out[k] = (A, B)
            continue
        A0, B0 = out[k]
        ax_a, ax_b = (0, 1) if A.ndim == 2 else (1, 2)
        out[k] = (np.concatenate([A0, A], axis=ax_a), np.concatenate([B0, B], axis=ax_b))
    return out


class InkBuilder:
    """ProjHead の並び → LoRA の行列(api の ProjBuilder の代わり。tensors だけを使う)。方向は部品ごとの名前で引く。"""

    def __init__(self, session, direction_set, *, correction: bool = True, weights=None):
        self.s, self.dir_of, self.correction = session, direction_set, bool(correction)
        self.plan = session.weights.inkling
        self.w = weights if weights is not None else session.weights     # 焼き込む先(--into)の重みで射影の量を作るとき
        self._refs: dict[str, dict] = {}

    def refs(self, name: str) -> dict:
        if name not in self._refs:
            self._refs[name] = layer_refs_of(self.plan, self.dir_of(name))
        return self._refs[name]

    def tensors(self, heads, warn=None) -> dict:
        from .inkling_chips import lora_tensors_all
        rd = SimpleNamespace(w=self.w, plan=self.plan)
        by: dict[str, list] = {}
        for p in heads:
            if p.s:
                check_unit(self.s, p)
                by.setdefault(p.direction, []).append(chips_unit(p))
        out: dict = {}
        for name, units in by.items():
            out = _merge(out, lora_tensors_all(rd, units, self.refs(name), correct=self.correction))
        return out

    def ablate_tensors(self, L, heads):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 尚不能使用将组件输出置 0 的验证（verify_mode ablate）", "verify_mode proj（默认）")


def export_lora(session, heads, path: Path, direction_set, *, correction: bool = True, strip_meta: bool = False) -> dict:
    from .ablate import write_lora_tensors
    tens = InkBuilder(session, direction_set, correction=correction).tensors(heads)
    if not tens:
        raise GneError("BAD_RECIPE", "没有可投影的组件")
    session.engine.allow_lora_trial = True               # Inkling 2 の射影の LoRA(このモジュールの作り方)だけを通す
    return write_lora_tensors(session.weights, tens, path, name="adapter" if strip_meta else path.stem,
                              kv=None if strip_meta else {"gne.inkling_proj": json.dumps([p.spec() for p in heads if p.s])})


# --- 測る -----------------------------------------------------------------------------------------------------------

def default_points(plan) -> list[int]:
    """測る点: 各区分の最後の段と、最後に混ぜた出力。"""
    pts = {plan.core_start - 1, plan.n_steps - 1, OUT}
    w = plan.width
    for k in range(plan.n_passes):
        pts.add(plan.core_start + (k + 1) * w - 1)
    return sorted(x for x in pts if x >= 0)


def point_label(plan, k: int) -> str:
    if k == OUT:
        return "最后混合的输出"
    st = plan.step(k)
    return f"段 {k}（第 {st['layer']} 层" + (f"、第 {st['pass']} 次" if st["pass"] else "") + "）"


def parse_points(plan, spec) -> list[int] | None:
    """score_layers の指定(段の番号の並び、out = 最後に混ぜた出力)→ 測る点。None なら既定。"""
    if spec in (None, "", []) or (isinstance(spec, str) and spec.strip().lower() in ("base", "default")):
        return None
    xs = spec.split(",") if isinstance(spec, str) else list(spec)
    out = set()
    for x in xs:
        x = str(x).strip().lower()
        if not x:
            continue
        if x == "out":
            out.add(OUT)
        elif re.fullmatch(r"\d+-\d+", x):
            a, b = (int(v) for v in x.split("-"))
            out.update(range(a, b + 1))
        elif x.isdigit():
            out.add(int(x))
        else:
            raise GneError("BAD_ARGS", f"Inkling 2 的 score_layers 只能是段编号（0～{plan.n_steps - 1}）和 out（{spec!r}）")
    bad = [x for x in out if not (0 <= x < plan.n_steps or x == OUT)]
    if bad:
        raise GneError("BAD_ARGS", f"段 {bad} 超出范围（0～{plan.n_steps - 1}、out）")
    return sorted(out | {OUT})


class InkScorer:
    """direction.Scorer の代わり: 対照集合の会話の、測る位置の 段ごとの MLP の入力・最後に混ぜた出力 · 方向。"""

    def __init__(self, session, ds, cs, *, layers, positions: str = "last", max_examples: int | None = None,
                 items: dict | None = None, progress=None):
        from .inkling_read import Reader
        require_inkling_direction(ds)
        self.session, self.ds, self.positions, self.subset = session, ds, str(positions), "all"
        self.reader = Reader(session)
        self.reader.require_patch()
        self.plan = self.reader.plan
        self.layers = sorted({int(x) for x in layers})
        self.r = {L: ds.unit(L).astype(np.float64) for L in self.layers}
        ci = items or contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
        self.ci = ci
        self.items = ci["items"]
        self.index_side = ci["index"]
        self.index = self.index_side["A"]
        self.prompts_side = ci["prompts"]
        self.prompts = self.prompts_side["A"]
        self.max_len = max(len(ids) for sd in "AB" for ids, _ in self.items[sd])
        nm = _names(self.plan)
        self.names = {nm[L]: L for L in self.layers}
        self.sides = ci.get("sides") or {"A": None, "B": None}       # 側のレシピの LoRA(調べる LoRA はその上に重ねる)

    def n(self, side: str) -> int:
        return len(self.items[side])

    def run(self, lora=None, vecs=None, *, sides: str = "AB", which: dict | None = None, keep: bool = False) -> dict:
        if vecs:
            raise GneError("ARCH_UNSUPPORTED", "Inkling 2 中无法施加 control vector 来测量")
        e = self.session.engine
        cap = self.reader.cap
        names = self.names
        cap.set_wanted(lambda n: n in names)
        ctx = e.probe_ctx(cap, need=self.max_len)
        out: dict = {}
        try:
            for side in sides:
                e.set_lora(ctx, (list(self.sides.get(side) or []) + list(lora or [])) or None)
                items = self.items[side]
                sel = list(range(len(items))) if which is None or side not in which else list(which[side])
                vals = {L: [] for L in self.layers}
                res: dict = {L: [] for L in self.layers}
                for i in sel:
                    ids, rows = items[i]
                    cap.reset()
                    e.clear_memory(ctx)
                    e.prefill(ctx, [ids])
                    got = cap.take()
                    T = len(ids)
                    for nm, L in names.items():
                        x = got[nm].reshape(T, -1)[rows].astype(np.float64)
                        vals[L].append(float((x @ self.r[L]).mean()))
                        if keep:
                            res[L].append(x)
                out[side] = {"values": {L: np.array(v) for L, v in vals.items()}, "idx": sel}
                if keep:
                    out[side]["resid"] = res
        finally:
            e.set_lora(ctx, None)
        return out


def prepare(session, ds, cs, *, points: list[int], positions: str, max_examples: int | None, b_examples: int = 16,
            progress=None):
    """causal.prepare の代わり(Setup に InkScorer と InkBuilder を入れる)。"""
    from .causal import Setup, TmpLora
    sc = InkScorer(session, ds, cs, layers=points, positions=positions, max_examples=max_examples, progress=progress)
    nb = sc.n("B")
    b_idx = list(range(nb)) if b_examples <= 0 or b_examples >= nb else \
        [int(x) for x in np.linspace(0, nb - 1, b_examples).round().astype(int)]
    session.engine.allow_lora_trial = True               # このモジュールが作る射影の LoRA を当てる
    st = Setup(session, ds, cs, sc, sorted(points), OUT,
               InkBuilder(session, session.direction_set, correction=True), TmpLora(session), sorted(set(b_idx)))
    st.base = st.measure(None)
    return st


# --- 書き込みの順位づけ --------------------------------------------------------------------------------------------

def expert_q(session, ds, progress=None) -> dict:
    """専門家の down の重みを、タップごとの向きとの内積に縮めたもの(方向ごとに 1 回。<root>/inkling/<model_id>/q-<方向>.npz に覚える)。"""
    from .inkling_chips import expert_q as _eq
    import hashlib
    key = hashlib.sha256(np.load(ds.path)["layers"].tobytes() + b"".join(ds.raw[k].tobytes() for k in sorted(ds.raw))).hexdigest()[:12]
    p = session.store.root / "inkling" / session.meta.model_id / f"q-{ds.name}-{key}.npz"
    if p.exists():
        z = np.load(p)
        return {"q": z["q"], "scale": z["scale"], "seconds": float(z["seconds"]), "max_rss_mb": float(z["max_rss_mb"]), "reused": True}
    p.parent.mkdir(parents=True, exist_ok=True)
    eq = _eq(SimpleNamespace(w=session.weights, plan=session.weights.inkling), layer_refs_of(session.weights.inkling, ds), progress=progress)
    np.savez(p, **eq)
    return {**eq, "reused": False}


def rank_units(session, ds, cs, *, positions: str, kinds, include_before: bool = False, skip_stream_start: bool = True,
               max_examples: int | None = None, eq: dict | None = None, items: dict | None = None, progress=None) -> dict:
    """全部の部品の、その層の基準の方向に沿った書き込み(短い畳み込みの後)の A − B(測る位置の平均)。行は unitwrites と同じ形
    (label・layer・index・kind・score・share・own …)に、専門家のニューロンは expert、繰り返しの層は回ごとの値 by_pass を足す。
    1 つの会話から読む位置は RANK_ROWS 個まで(等間隔に間引く)。"""
    from .inkling_chips import _taps, region_of
    from .inkling_read import Reader, Weights2, want
    t0 = time.perf_counter()
    plan = session.weights.inkling
    w2 = Weights2(session.weights)
    li = session.weights.arch.layers[0]
    nh, hd, nE, nf, k_used = li.n_head, li.head_dim, li.n_expert, li.n_ff, li.n_expert_used
    d = session.weights.meta.n_embd
    refs = layer_refs_of(plan, ds)
    from .unitwrites import parse_kinds
    with_neurons = "neuron" in set(parse_kinds(kinds))
    if with_neurons:                              # 専門家の down の重みの下ごしらえは、ニューロンを順位づけるときだけ
        eq = eq or expert_q(session, ds, progress=progress)
        q, sc = eq["q"], eq["scale"]
    else:
        eq = {"seconds": 0.0, "reused": None}
    ci = items or contrast_items(session, cs, positions, max_examples=max_examples, progress=progress)
    rd = Reader(session, progress=progress)
    rd.require_patch()
    U, CW = {}, {}
    for L in range(plan.n_stored):
        Wo = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(d, nh, hd)
        cw = w2.full(f"blk.{L}.attn_post_conv.weight").astype(np.float64)
        CW[L] = cw
        U[L] = np.einsum("kd,dhe->khe", _taps(cw, refs[L][1]), Wo)          # (4, nh, hd): タップ k のヘッドの書き込みの r の成分
    acc = {sd: {"heads": np.zeros((plan.n_steps, nh)), "attn": np.zeros(plan.n_steps), "ffn": np.zeros(plan.n_steps),
                "neur": np.zeros((plan.n_steps, nE, nf), np.float32)} for sd in "AB"}
    rd.fit([ids for sd in "AB" for ids, _ in ci["items"][sd]])
    tot = sum(len(ci["items"][sd]) for sd in "AB")
    done = 0
    names = want({"attn_gated", "attn_out", "ffn_out"} | ({"ffn_moe_situ", "ffn_moe_argsort", "ffn_moe_weights_scaled"} if with_neurons else set()))
    for sd in "AB":
        for ids, rows in ci["items"][sd]:
            rows = _thin(list(rows), RANK_ROWS)
            cap = run_with(rd, ids, names, (ci.get("sides") or {}).get(sd))
            T = len(ids)
            wr = 1.0 / len(rows)
            for st in plan.steps():
                s, L = st["step"], st["layer"]
                r = refs[L][1]
                g = cap[f"attn_gated-{s}"].reshape(T, nh, hd)
                ao = cap[f"attn_out-{s}"].reshape(T, -1)
                fo = cap[f"ffn_out-{s}"].reshape(T, -1)
                if with_neurons:
                    top = cap[f"ffn_moe_argsort-{s}"].reshape(T, -1)[:, :k_used]
                    wts = cap[f"ffn_moe_weights_scaled-{s}"].reshape(T, k_used)
                    act = cap[f"ffn_moe_situ-{s}"].reshape(T, k_used, nf)
                a = acc[sd]
                for p in rows:
                    for k in range(4):
                        t = p - 3 + k
                        if t < 0:
                            continue
                        a["heads"][s] += wr * np.einsum("he,he->h", g[t].astype(np.float64), U[L][k])
                        if with_neurons:
                            e = top[t]
                            c = (wts[t] * sc[L, e])[:, None] * act[t] * q[L, e, k]
                            np.add.at(a["neur"][s], e, (wr * c).astype(np.float32))
                    a["attn"][s] += wr * float(ao[p] @ r)
                    a["ffn"][s] += wr * float(fo[p] @ r)
            del cap
            done += 1
            if progress:
                progress(done, tot, "Inkling 2：写入排序")
    nA, nB = len(ci["items"]["A"]), len(ci["items"]["B"])
    dif = {k: acc["A"][k] / nA - acc["B"][k] / nB for k in ("heads", "attn", "ffn", "neur")}
    from .unitwrites import parse_kinds
    want_kinds = set(parse_kinds(kinds))
    rows_out = []
    cand_regions = {"loop", "after"} | ({"before"} if include_before else set())

    def bp(f, L):
        return {str(plan.step(s)["pass"] or 0): float(f(s)) for s in plan.steps_of(L)}
    excluded = {"stream_start": [], "before": []}
    for L in range(plan.n_stored):
        rs = refs[L][0]
        region = region_of(plan, L)
        if region not in cand_regions:
            excluded["before"].append(L)
            continue
        ss = skip_stream_start and L % plan.block == 0
        if ss:
            excluded["stream_start"].append(L)
        if "attn" in want_kinds and not ss:
            for h in range(nh):
                rows_out.append({"label": f"L{L}:H{h}", "layer": L, "index": h, "kind": "attn", "score": float(dif["heads"][rs, h]),
                                 "by_pass": bp(lambda s, h=h: dif["heads"][s, h], L), "region": region})
        if "layer_attn" in want_kinds and not ss:
            rows_out.append({"label": f"L{L}:attn", "layer": L, "index": -1, "kind": "layer_attn", "score": float(dif["attn"][rs]),
                             "by_pass": bp(lambda s: dif["attn"][s], L), "region": region})
        if "layer_mlp" in want_kinds:
            rows_out.append({"label": f"L{L}:ffn", "layer": L, "index": -1, "kind": "layer_mlp", "score": float(dif["ffn"][rs]),
                             "by_pass": bp(lambda s: dif["ffn"][s], L), "region": region})
        if "neuron" in want_kinds:
            v = dif["neur"][rs]
            flat = v.reshape(-1)
            keep = min(256, flat.size)
            idx = np.argpartition(-flat, keep - 1)[:keep]
            for x in idx[np.argsort(-flat[idx])]:
                e, n = divmod(int(x), nf)
                rows_out.append({"label": f"L{L}:E{e}:N{n}", "layer": L, "index": n, "expert": e, "kind": "neuron",
                                 "score": float(flat[x]), "by_pass": bp(lambda s, e=e, n=n: dif["neur"][s, e, n], L), "region": region})
    rows_out.sort(key=lambda r: -r["score"])
    pos_total = sum(r["score"] for r in rows_out if r["score"] > 0) or 1.0
    for r in rows_out:
        r["share"] = r["score"] / pos_total
        r["own"] = r["score"]
    n_units = {k: sum(1 for r in rows_out if r["kind"] == k) for k in ("attn", "linear", "neuron", "layer_attn", "layer_mlp")}
    return {"rows": rows_out, "n": {"A": nA, "B": nB}, "n_units": n_units, "kinds": sorted(want_kinds),
            "excluded": excluded, "expert_q": {"seconds": eq.get("seconds"), "reused": eq.get("reused")},
            "seconds": time.perf_counter() - t0}


# --- 一括(gne heads pipeline) -------------------------------------------------------------------------------------

INK_DEFAULTS = {"unit_kinds": "all,layers", "check_tokens": 1024, "max_examples": 32}
NOT_BUILT = {"same_reply": "让两侧都读取回复的方向（same_reply）", "holdout": "将问题分成两半的验证（holdout）",
             "loop_direction": "供权重使用两次的模型使用的 loop_direction", "candidate_layers": "候选层指定（candidate_layers）",
             "select_layer": "选择层指定（select_layer。Inkling 2 使用最后混合的输出进行选择）",
             "always_layers": "用于 rank causal 的 always_layers", "include_score_layers": "用于 rank causal 的 include_score_layers"}


def pipeline(session, spec: dict, pipe: dict, given: dict, t0: float) -> dict:
    """api.heads_pipeline の Inkling 2 の流れ。spec は対照集合の中身(pipeline を除いたもの)、pipe は既定を混ぜた設定、given は
    ひな形・引数で渡された設定。結果は <root>/heads/<model_id>/<name>.pipeline.json(ほかのモデルの rank writes と同じ形)。"""
    from .causal import most_reduced_summary
    from .direction import load_contrast
    from .reverse import check_questions
    from .unitwrites import allocate_units, verify_rows, verify_units
    from .yamlio import write_yaml
    plan = session.weights.inkling
    if pipe.get("rank") == "causal" and "rank" in given:
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 尚不能使用 --rank causal（逐层停止头并用残差测量的选择方式）",
                       "请以 --rank writes（默认）运行")
    pipe["rank"] = "writes"
    for k, what in NOT_BUILT.items():
        if given.get(k) not in (None, False, "", []):
            raise GneError("ARCH_UNSUPPORTED", f"Inkling 2 尚不能使用 {what}（已停止）")
    if pipe.get("verify_mode") not in (None, "proj"):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 尚不能使用 verify_mode ablate", "verify_mode proj（默认）")
    arch_defaults = {}
    for k, v in INK_DEFAULTS.items():
        if k not in given:
            pipe[k] = v
            arch_defaults[k] = v
    kind, _ = parse_positions(pipe.get("positions"))
    sp = pipe.get("score_positions")
    if "score_positions" in given and str(sp) != str(pipe.get("positions")):
        raise GneError("ARCH_UNSUPPORTED", "Inkling 2 中，测量位置与创建方向的位置相同（score_positions 会对齐到 positions）")
    pipe["score_positions"] = pipe.get("positions") or "last"
    name = str(pipe.pop("name", None) or spec.get("name") or "")
    if not re.fullmatch(r"[\w.\-]+", name):
        raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、- 和 .（spec 的 name）")
    spec["name"] = name
    for key in ("user_prompts", "prompts_a", "prompts_b"):
        check_questions(spec.get(key), key)
    path = session.store.contrast_dir() / f"{name}.yaml"
    write_yaml(path, spec)
    try:
        cs, val = load_contrast(path)
        if min(len(cs.prompts("A")), len(cs.prompts("B"))) < 4:
            raise GneError("BAD_CONTRAST", "两侧的问题都请至少准备 4 个（为了测量 A 与 B 的分歧程度和效果）")
    except GneError:
        path.unlink(missing_ok=True)
        raise
    mx = int(pipe["max_examples"]) if pipe.get("max_examples") else None
    points = parse_points(plan, pipe.get("score_layers")) or default_points(plan)
    include_before = bool(given.get("include_before"))
    out: dict = {"name": name, "contrast": str(path), "n_prompts": len(cs.prompts("A")),
                 **({"n_prompts_b": len(cs.prompts("B"))} if cs.separate_prompts else {}), "sides": cs.describe(),
                 "settings": {**pipe, "include_before": include_before}, "rank": "writes", "arch_defaults": arch_defaults or None,
                 "inkling": {"positions": str(pipe["positions"]), "position_kind": kind, "position_note": POSITION_NOTE,
                             "unit_note": UNIT_NOTE, "points": [{"key": k, "label": point_label(plan, k)} for k in points],
                             "include_before": include_before, "reference": "MLP 的输入（第 6～17 层为第 3 次，位于重复门控之后）"}}
    session.engine.free_steer()
    ts = time.perf_counter()                                           # 1. 方向
    if pipe.get("from_direction"):
        ds = session.direction_set(str(pipe["from_direction"]))
        require_inkling_direction(ds)
        d = {"path": str(ds.path), "positions": ds.meta.get("positions"), "n_rows": {"A": ds.meta.get("n_rows_A"), "B": ds.meta.get("n_rows_B")},
             "from": ds.name}
        if str(ds.meta.get("positions")) != str(pipe["positions"]):
            raise GneError("BAD_ARGS", f"方向 {ds.name} 是在位置 {ds.meta.get('positions')} 创建的（当前 positions 为 {pipe['positions']}）")
        import shutil
        shutil.copyfile(ds.path, session.store.direction_dir(session.meta.model_id) / f"{name}.npz")
    else:
        d = make_direction(session, cs, name, str(pipe["positions"]), source=str(path), max_examples=mx, progress=session.progress)
    ds = session.direction_set(name)
    out["direction"] = {"path": d["path"], "positions": d["positions"], "n_rows": d["n_rows"], "seconds": time.perf_counter() - ts,
                        "reused": False, **({"skipped": d["skipped"]} if d.get("skipped") else {}),
                        **({"replies_from": d["replies_from"]} if d.get("replies_from") else {}),
                        **({"from": d["from"]} if d.get("from") else {})}
    # 2. 段ごとの A と B の分かれ方(ほかのモデルの「基準層」の代わり。Inkling 2 は層ごとの基準が決まっているので選ばない)
    ts = time.perf_counter()
    st = prepare(session, ds, cs, points=points, positions=str(pipe["positions"]), max_examples=mx, progress=session.progress)
    out["inkling"]["n_questions"] = {"total": {"A": len(cs.prompts("A")), "B": len(cs.prompts("B"))}, "max_examples": mx,
                                     "direction": d.get("n_items") or {"A": None, "B": None},
                                     "measure": {"A": st.scorer.n("A"), "B": len(st.b_idx)}}
    out["score"] = {"base_layer": None, "best_layer": None, "n": {"A": st.scorer.n("A"), "B": st.scorer.n("B")},
                    "layers": [{"layer": L, "label": point_label(plan, L), "A": float(st.base["A"][L].mean()),
                                "B": float(st.base["B"][L].mean())} for L in st.score_layers],
                    "seconds": time.perf_counter() - ts}
    ts = time.perf_counter()                                           # 3. 書き込みの順位づけ
    rk = rank_units(session, ds, cs, positions=str(pipe["positions"]), kinds=pipe["unit_kinds"], include_before=include_before,
                    max_examples=mx, items=st.scorer.ci, progress=session.progress)
    gap = float(st.base["A"][OUT].mean() - st.base["B"][OUT].mean())
    out["writes"] = {"select_layer": OUT, "select_label": point_label(plan, OUT), "kinds": rk["kinds"], "n": rk["n"], "gap": gap,
                     "n_units": rk["n_units"], "totals_share": {}, "layer_scores": [], "excluded": rk["excluded"],
                     "expert_q": rk["expert_q"], "seconds": time.perf_counter() - ts, "top": rk["rows"][:40]}
    ts = time.perf_counter()                                           # 4. 上位を 1 個ずつ s=1 で確かめる
    top = verify_rows(rk["rows"], int(pipe["verify_top"]), int(pipe["verify_sample"]))
    ver = verify_units(session, st, top, mode="proj", progress=session.progress)
    by = {r["label"]: r for r in rk["rows"]}
    for r in ver["rows"]:                                              # 専門家の番号・回ごとの値を戻す(配分が部品を作るのに使う)
        src = by.get(r["label"]) or {}
        for k in ("expert", "by_pass", "region"):
            if k in src:
                r[k] = src[k]
    out["verify"] = {"mode": ver["mode"], "n": len(ver["rows"]), "spearman": ver["spearman"],
                     "kinds": {k: sum(1 for r in ver["rows"] if r["kind"] == k) for k in ("attn", "linear", "neuron", "layer_attn", "layer_mlp")},
                     "excluded": [r["label"] for r in ver["rows"] if r.get("excluded")], "top": ver["rows"][:20],
                     "seconds": time.perf_counter() - ts}
    p_rank = session.store.heads_dir(session.meta.model_id) / f"{name}.writes.json"
    p_rank.write_text(json.dumps({"rows": rk["rows"][:2000], "verify": ver["rows"], "n": rk["n"], "kinds": rk["kinds"]},
                                 ensure_ascii=False), encoding="utf-8")
    out["writes"]["path"] = str(p_rank)
    bs = {"A": float(st.base["A"][OUT].mean()), "B": float(st.base["B"][OUT].mean())}
    tgt, tgt_from = session._pipeline_target(pipe["target"], bs)
    out["causal"] = {"score_layers": st.score_layers, "primary": OUT, "n_a": st.scorer.n("A"), "n_b": len(st.b_idx),
                     "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                     for L in st.score_layers}}
    a = allocate_units(session, st, ver["rows"], target=float(tgt), n_units=[int(x) for x in pipe["n_units"]],
                       s_max=float(pipe["s_max"]), name=name, direction_layer="own", remain_threshold=float(tgt), max_examples=mx,
                       strength=str(pipe.get("strength") or "per_unit"), hard=int(pipe.get("hard") or 16), progress=session.progress)
    a["notes"] = list(a.get("notes") or []) + [{"code": "inkling_shared", "text": UNIT_NOTE}]
    p_ = Path(a["path"])
    p_.write_text(json.dumps({k: v for k, v in a.items() if k != "path"}, ensure_ascii=False, indent=1), encoding="utf-8")
    out["allocate"] = a
    out["reused"] = []
    out["target"], out["target_from"] = float(tgt), tgt_from
    rec = next((r for r in a["table"] if r.get("recommended")), None)
    out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"],
                                                   "reached": bool(rec["reached"]), "by": a.get("recommended_by")}
    if a.get("no_recommend"):
        out["no_recommend"] = a["no_recommend"]
    n_check = pipe.get("check_replies")
    n_check = 16 if n_check is None else int(n_check)
    ok_rows = [r for r in a["table"] if not r.get("skipped")]
    chk = rec or (min(ok_rows, key=lambda r: (r["remain"], r["b_drift_orth"])) if ok_rows else None)
    if n_check > 0 and chk is not None:                                # 5. 応答での確かめ(思考を除いた答えで数える)
        out["check"] = session.reply_check(cs, val, chk["recipe"], n=n_check, max_tokens=int(pipe["check_tokens"]))
        out["check"].update(n_heads=chk["n_heads"], recommended=rec is not None)
        if rec is not None and out["check"].get("break_warning"):
            session._recommend_without_broken(out, a, rec, cs, val, n_check, int(pipe["check_tokens"]))
    out["most_reduced"] = most_reduced_summary(out)
    out["seconds"] = time.perf_counter() - t0
    out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    p = session.store.heads_dir(session.meta.model_id) / f"{name}.pipeline.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    out["path"] = str(p)
    return out


def _json(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


# --- 焼き込み -------------------------------------------------------------------------------------------------------

def bake_recipe(session, heads, dst: Path | None, *, requant: str = "auto", vram_free_mb: float | None = None,
                strip_meta: bool = False, target=None, dry_run: bool = False) -> dict:
    """レシピの射影の部品を GGUF に焼き込む(inkling_chips.bake)。requant は auto(既定: 元の型・元の大きさで書いて測り、編集が
    残らないテンソルと、MLP 全体の部品を受けて方向に沿わない変化が大きい専門家のテンソルだけ Q8_0)・keep・q8_0・f16。書く前に一度計算して、Q8_0 にする層と
    ファイル・VRAM の増え方を決め、progress で知らせる(結果の plan にも入れる)。vram_free_mb(読み込んだときの空き)を渡すと、
    収まらないときは知らせる(ファイルは書く)。target(target_weights の重み)を渡すと、同じモデルの別の量子化に書く: 方向と部品は
    このモデルのもの、消す量(r̂ᵀW)と短い畳み込みの倍率は焼き込む先の重みから作り直す。dry_run なら書かずに、書き換えるテンソルと
    大きさ・残り方だけを返す。"""
    from .inkling_chips import bake
    heads = [p for p in heads if p.s]
    if not heads:
        raise GneError("BAD_RECIPE", "没有可投影的组件")
    for p in heads:
        check_unit(session, p)
    units = [chips_unit(p) + [p.direction] for p in heads]            # 記録用(方向つき)
    tw = target if target is not None else session.weights
    tens = InkBuilder(session, session.direction_set, weights=tw).tensors(heads)   # いくつかの方向(残りから追加の合わせたレシピ)もまとめる
    rd = SimpleNamespace(w=tw, plan=session.weights.inkling)
    refs = None
    meta = {"source": Path(session.weights.path).name, "target": Path(tw.path).name} if tw is not session.weights else None
    if requant in ("auto", "keep"):                          # 書く前に一度計算して、型と大きさを決める
        prep = bake(rd, units, refs, None, requant=requant, tens=tens)
        pre = prep["modified"]
        grow = {"bytes": int(sum(m["bytes_to"] - m["bytes_from"] for m in pre)),
                "tensors": [plan_entry(m) for m in pre],
                "expert_layers": sorted({int(m["tensor"].split(".")[1]) for m in pre if "ffn_down_exps" in m["tensor"] and m["to"] != m["from"]})}
    else:
        pre = prep = None
        grow = estimate_growth(SimpleNamespace(weights=tw), [u[:4] for u in units], requant)
    _mark_loop(grow, session.weights.inkling)
    plan = bake_plan_text(grow)
    if dry_run:                                                 # 書かずに: 書き換えるテンソル・大きさ・残り方
        recs = pre if pre is not None else bake(rd, units, refs, None, requant=requant, tens=tens)["modified"]
        out = {"path": None, "dry_run": True, "modified": recs, "growth": grow, "plan": plan, "requant": requant, "off_max": _off_max(),
               "bytes_src": Path(tw.path).stat().st_size, "bytes": Path(tw.path).stat().st_size + grow["bytes"]}
        if meta:
            out.update(into=str(tw.path), carry_over=carry_over(session, heads, tw))
        if kept_note(recs):
            out["kept_note"] = kept_note(recs)
        return out
    sp = out_space(dst, Path(tw.path).stat().st_size + grow["bytes"])
    plan = f"{plan}。{space_text(sp)}"
    check_space(sp, dst)                                        # 書き込み先のファイルシステムに入らなければ書く前に止める
    if getattr(session, "progress", None):                    # 書く前に知らせる(CLI は進み具合の行、画面はジョブの段の文)
        session.progress(0, 1, plan)
    vram_note = None
    if vram_free_mb is not None and grow["bytes"] / 2**20 > vram_free_mb - VRAM_MARGIN_MB:
        # ファイルを書くだけなら VRAM は要らないので、止めずに知らせる。空きは、いまのモデルと作業用の領域を載せたままの値なので、
        # 焼き込んだ GGUF だけを読み込むときの空きより小さく出る
        vram_note = (f"烘焙后 VRAM 将增加 {grow['bytes'] / 2**20:.0f} MiB（需改写的专家层 {grow['expert_layers']}）。"
                     f"当前可用空间为 {vram_free_mb:.0f} MiB（这是在当前模型和工作区域仍加载时的值）。"
                     f"如果加载烘焙后的 GGUF 时无法装入 GPU，请减少组件的层数，或将类型设为 auto（原大小）"
                     f"（每改写一个专家层，Q8_0 比 Q2_0 层增加 +253 MiB，比 NVFP4 层增加 +162 MiB）")
        if getattr(session, "warn", None):
            session.warn(vram_note)
    r = bake(rd, units, refs, Path(dst), requant=requant, strip_meta=strip_meta, tens=tens, extra_meta=meta, prepared=prep)
    r["growth"] = grow
    if meta:
        r["into"] = Path(tw.path).name if strip_meta else str(tw.path)
        r["carry_over"] = carry_over(session, heads, tw)
    r["plan"] = plan
    r["off_max"] = _off_max()
    if vram_note:
        r["vram_note"] = vram_note
    if kept_note(r["modified"]):
        r["kept_note"] = kept_note(r["modified"])
    return r


def _off_max() -> float:
    from .inkling_bake import OFF_MAX
    return OFF_MAX


def kept_note(recs) -> str | None:
    """書いた型で編集の 1 割以上が丸めに消えたテンソル(edit_kept < KEYED_KEEP_MIN)と、MLP 全体の部品を受けて方向に沿わない
    変化 ÷ 消した成分が OFF_MAX を超えたまま元の型で書いた専門家のテンソルの注意。auto はそうしたテンソルを Q8_0 にするので、
    出るのは主に keep(元の型を守る)のとき。"""
    from .gain import KEYED_KEEP_MIN
    from .inkling_bake import OFF_MAX
    low = [m for m in recs or [] if m.get("edit_kept") is not None and m["edit_kept"] < KEYED_KEEP_MIN]
    off = [m for m in recs or [] if m.get("whole_layer") and (m.get("off_over_removed") or 0.0) > OFF_MAX and m not in low]
    out = []
    if low:
        what = "、".join(f"{m['tensor']}（{m['to']}）为 {m['edit_kept']:.0%}" for m in low[:3]) + (" 等" if len(low) > 3 else "")
        out.append(f"以写入的类型保存时，有些张量有一成以上的编辑因舍入而消失（保留的编辑：{what}）")
    if off:
        what = "、".join(f"{m['tensor']}（{m['to']}）为 {m['off_over_removed']:.2f}" for m in off[:3]) + (" 等" if len(off) > 3 else "")
        out.append(f"保持原类型时，有些张量的不沿方向的变化 ÷ 已消除成分超过了上限 {OFF_MAX}（{what}）")
    if not out:
        return None
    return "。".join(out) + "。请将类型设为 auto（仅将这些张量设为 Q8_0），或用 LoRA 分发"


def target_weights(session, into):
    """焼き込む先(--into): 同じモデルの別の量子化。構造(arch・層の数・幅)に加え、Inkling 2 の層の並び、テンソルの名前と形が
    すべて同じであることを確かめる(型だけ違ってよい)。重みは memmap で読み、GPU には載せない。"""
    tw = session._target_weights(into)
    if tw is session.weights:
        return tw
    sp, tp = session.weights.inkling, getattr(tw, "inkling", None)
    if tp is None or sp.to_dict() != tp.to_dict():
        raise GneError("BAD_ARGS", f"烘焙目标 {Path(into).name} 的 Inkling 2 层布局（重复层、重复次数、流的分段）不同",
                       "请使用同一模型的另一种量化")
    ts, tt = session.weights._tensors, tw._tensors
    miss = sorted(set(ts) ^ set(tt))
    bad = [n for n in ts if n in tt and list(ts[n].shape) != list(tt[n].shape)]
    if miss or bad:
        raise GneError("BAD_ARGS", f"烘焙目标 {Path(into).name} 不是同一模型（张量名称不同 {len(miss)} 个，形状不同 "
                                   f"{len(bad)} 个。例如 {(miss + bad)[:3]}）", "请使用同一模型的另一种量化")
    return tw


def unit_component(w, plan, refs: dict, unit: list, *, correct: bool = True) -> np.ndarray:
    """部品の消す量のもと: 直した向き u(短い畳み込みの今のトークンの倍率で直したもの)と、部品が書く行列の uᵀW(専門家は倍率つき)。
    ヘッドは列のブロック、attention 全体は行全体、MLP 全体は全部の専門家(専門家ごとの倍率を掛けて並べる)、ニューロンは 1 つの値。"""
    from .inkling_read import Weights2
    w2 = Weights2(w)
    L, kind, idx = int(unit[0]), unit[1], unit[2]
    r = refs[L][1].astype(np.float64)
    conv = "attn_post_conv" if kind in ("head", "attn") else "ffn_post_conv"
    cw = w2.full(f"blk.{L}.{conv}.weight").astype(np.float64)
    g = (1.0 + cw[:, -1]) / math.sqrt(cw.shape[1] + 1)
    u = g * r if correct else r
    u = u / max(float(np.linalg.norm(u)), 1e-30)
    if kind in ("head", "attn"):
        W = w2.full(f"blk.{L}.attn_output.weight").astype(np.float64).reshape(len(u), -1)
        v = u @ W
        if kind == "head":
            hd = w.arch.layers[0].head_dim
            v = v[int(idx) * hd:(int(idx) + 1) * hd]
        return v
    sc = w2.scale(f"blk.{L}.ffn_down_exps.scale")
    if kind == "neuron":
        e, n = int(idx[0]), int(idx[1])
        return np.array([(1.0 if sc is None else float(sc[e])) * float(u @ w2.slab(f"blk.{L}.ffn_down_exps.weight", e).astype(np.float64)[:, n])])
    nE = w.arch.layers[0].n_expert
    return np.concatenate([(1.0 if sc is None else float(sc[e])) * (u @ w2.slab(f"blk.{L}.ffn_down_exps.weight", e).astype(np.float64))
                           for e in range(nE)])


def carry_over(session, heads, target) -> list[dict]:
    """同じレシピの消す量(r̂ᵀW、部品ごと)の、このモデルと焼き込む先の違い: 相対の差 ‖v_先 − v_元‖ / ‖v_元‖ と大きさの比。
    強さ s は消す量に掛かるので、比が 1 から離れるほど、このモデルで決めた強さの効き方が焼き込む先で変わる。"""
    plan = session.weights.inkling
    out = []
    for p in heads:
        if not p.s:
            continue
        refs = layer_refs_of(plan, session.direction_set(p.direction))
        u = chips_unit(p)
        a = unit_component(session.weights, plan, refs, u)
        b = unit_component(target, plan, refs, u)
        out.append({"unit": p.spec(), "rel_diff": float(np.linalg.norm(b - a) / max(np.linalg.norm(a), 1e-30)),
                    "norm_ratio": float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-30)),
                    "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))})
    return out


def taps_identical(source, target) -> dict:
    """短い畳み込みのタップ(射影の倍率のもと)の型と、元と焼き込む先で値が同じか。"""
    from .inkling_read import Weights2
    a, b = Weights2(source), Weights2(target)
    names = [n for n in source._tensors if n.endswith(("attn_post_conv.weight", "ffn_post_conv.weight"))]
    diff = [n for n in names if not np.array_equal(a.full(n), b.full(n))]
    types = sorted({target._tensors[n].tensor_type.name for n in names})
    return {"n": len(names), "different": diff, "target_types": types}


SPACE_MARGIN = 256 * 2**20           # 書き込み先の空きに残すゆとり(メタデータ・横の記録の分)


def _wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return False


def out_space(dst, nbytes: int) -> dict:
    """焼き込んだ GGUF の書き込み先: 書くファイルの大きさ、空き(同じ名前の古いファイルは書くときに切り詰めるので、その分も
    空きに数える)、WSL の Linux 側のディスクか(/mnt/ の外)。"""
    import shutil
    d = Path(dst).resolve()
    parent = d.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    free = shutil.disk_usage(parent).free + (d.stat().st_size if d.is_file() else 0)
    return {"bytes": int(nbytes), "free": int(free), "dir": str(d.parent), "wsl_linux": _wsl() and not str(d).startswith("/mnt/")}


def space_text(sp: dict) -> str:
    return (f"要写入的文件 {sp['bytes'] / 2**30:.2f} GiB（写入目标可用空间 {sp['free'] / 2**30:.1f} GiB"
            + ("，WSL 的 Linux 侧磁盘" if sp.get("wsl_linux") else "") + "）")


def check_space(sp: dict, dst) -> None:
    """書き込み先のファイルシステムの空きが、書くファイルとゆとりより小さければ止める。"""
    if sp["free"] < sp["bytes"] + SPACE_MARGIN:
        raise GneError("NO_SPACE", f"写入目标 {sp['dir']} 的可用空间为 {sp['free'] / 2**30:.1f} GiB，无法写入烘焙后的 GGUF"
                                   f"（{sp['bytes'] / 2**30:.2f} GiB）", "请释放空间，或用 --out 写到有可用空间的磁盘")


def plan_entry(m: dict) -> dict:
    """焼き込みの記録 1 つ → 予定の行(型・大きさと、元の大きさのまま / Q8_0 の理由になる数)。"""
    fb = m.get("fallback") or {}
    return {"tensor": m["tensor"], "type": m["from"], "to": m["to"], "from_mib": m["bytes_from"] / 2**20, "to_mib": m["bytes_to"] / 2**20,
            "whole_layer": bool(m.get("whole_layer")), "why": fb.get("why") or ("edit_kept" if "edit_kept" in fb else None),
            "edit_kept": fb.get("edit_kept", m.get("edit_kept")), "off_over_removed": fb.get("off_over_removed", m.get("off_over_removed"))}


def _plan_why(x: dict) -> str:
    from .inkling_bake import OFF_MAX
    if x.get("why") == "off_direction":
        return f"不沿方向的变化 ÷ 已消除成分 {x['off_over_removed']:.2f} 超过上限 {OFF_MAX}"
    if x.get("why") == "edit_kept":
        return f"使用原类型时保留的编辑 {x['edit_kept']:.2f}"
    if x.get("why") == "imatrix":
        return "需要 imatrix 的类型"
    return ""


def _mark_loop(grow: dict, plan) -> None:
    """予定の行に、繰り返しの層(層 6〜17、3 回実行)のテンソルかの印と、繰り返しの回数を入れる(予定の注意の文に使う)。"""
    lo, hi = plan.core_start, plan.core_end
    grow["loop"] = {"first": lo, "last": hi - 1, "passes": plan.n_passes}
    for x in grow.get("tensors") or []:
        x["loop_layer"] = lo <= int(x["tensor"].split(".")[1]) < hi


def loop_note(grow: dict) -> str | None:
    """繰り返しの層の MLP 全体の部品を元の大きさのまま書くときの注意(止めない。数は公開のひな形 1 つ・UD-IQ2_XXS で測った値)。"""
    xs = [x for x in grow.get("tensors") or [] if x.get("whole_layer") and x.get("loop_layer") and x.get("to") == x.get("type")]
    if not xs:
        return None
    n = (grow.get("loop") or {}).get("passes", 3)
    ls = "、".join(x["tensor"].split(".")[1] for x in xs)
    return (f"第 {ls} 层的整个 MLP 组件位于重复层中，保持原大小写入时，微小的不沿方向变化会作用于全部 {n} 次的所有段，"
            "因此烘焙后的文件与运行时 LoRA 的差异可能比其他层更大（在一个公开模板、UD-IQ2_XXS 上测得：两个方向时，"
            "第 3 次的段中 LoRA 自身的变化约为 0.30，Q8_0 约为 0.15；一个方向时与 Q8_0 大致相同）。为避免此问题，请用 --requant q8_0（界面显示 Q8_0）"
            "写入（每改写一个专家层，NVFP4 +162 MiB、Q2_0 +253 MiB）")


def bake_plan_text(grow: dict) -> str:
    """焼き込みの予定の文: 型を変えるテンソル(層)とその理由、MLP 全体の部品を受けて元の大きさのまま書く専門家のテンソル
    (方向に沿わない変化 ÷ 消した成分)、ファイル・VRAM の増え方。"""
    from .inkling_bake import OFF_MAX
    ts = grow.get("tensors") or []
    conv = [x for x in ts if x.get("to") and x.get("to") != x.get("type")]
    stay = [x for x in ts if x.get("whole_layer") and x.get("to") == x.get("type") and x.get("off_over_removed") is not None]
    parts = []
    if stay:
        parts.append("保持原大小：" + "、".join(f"第 {x['tensor'].split('.')[1]} 层的专家 {x['type']}（整个 MLP 组件。不沿方向的变化 ÷ "
                                                   f"已消除成分 {x['off_over_removed']:.2f}，上限 {OFF_MAX}）" for x in stay))
    if conv:
        parts.append("改变类型：" + "、".join(
            f"第 {x['tensor'].split('.')[1]} 层的{'专家' if 'ffn_down_exps' in x['tensor'] else ' attention 输出'} {x['type']} → {x['to']}"
            f"（+{x['to_mib'] - x['from_mib']:.0f} MiB" + (f"、{_plan_why(x)}" if _plan_why(x) else "") + "）" for x in conv))
    ln = loop_note(grow)
    if not conv:
        return "烘焙计划：所有张量都保持原类型、原大小" + (f"（{parts[0]}）" if parts else "") + (f"。{ln}" if ln else "")
    return f"烘焙计划：{'。'.join(parts)}。文件与 VRAM 合计 {grow['bytes'] / 2**20:+.0f} MiB" + (f"。{ln}" if ln else "")


def gpu_free_mb() -> float | None:
    """GPU の空き(MiB、最初の GPU)。llama.cpp(ggml)の装置の一覧から読む。GPU が無い・読めなければ None。"""
    import ctypes
    try:
        from llama_cpp._ggml import libggml as G
        G.ggml_backend_dev_count.restype = ctypes.c_size_t
        G.ggml_backend_dev_get.restype = ctypes.c_void_p
        G.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
        G.ggml_backend_dev_type.restype = ctypes.c_int
        G.ggml_backend_dev_type.argtypes = [ctypes.c_void_p]
        G.ggml_backend_dev_memory.restype = None
        G.ggml_backend_dev_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        for i in range(int(G.ggml_backend_dev_count())):
            dev = G.ggml_backend_dev_get(i)
            if G.ggml_backend_dev_type(dev) == 1:            # GGML_BACKEND_DEVICE_TYPE_GPU
                free, total = ctypes.c_size_t(0), ctypes.c_size_t(0)
                G.ggml_backend_dev_memory(dev, ctypes.byref(free), ctypes.byref(total))
                return free.value / 2**20
    except Exception:                                        # noqa: BLE001
        return None
    return None


VRAM_MARGIN_MB = 256            # 焼き込んだ GGUF を読むときに残しておく空き


def estimate_growth(session, units: list, requant: str) -> dict:
    """焼き込みでファイル(と VRAM)が増える大きさ: 書き換えるテンソルの元の大きさと、書き出す型での大きさの差。"""
    from .quant import block_sizes, qtype_of
    T = session.weights._tensors
    names = set()
    for L, k, _, _ in units:
        names.add(f"blk.{L}.attn_output.weight" if k in ("head", "attn") else f"blk.{L}.ffn_down_exps.weight")
    tot, per = 0, []
    for nm in sorted(names):
        t = T[nm]
        n_el = int(np.prod(t.shape))
        if requant == "keep":
            to = int(t.n_bytes)
        else:
            q = qtype_of({"q8_0": 8, "f16": 1}[requant])
            bs, ts = block_sizes(q)
            to = n_el * ts // bs
        to_type = t.tensor_type.name if requant == "keep" else {"q8_0": "Q8_0", "f16": "F16"}[requant]
        per.append({"tensor": nm, "type": t.tensor_type.name, "to": to_type, "from_mib": t.n_bytes / 2**20, "to_mib": to / 2**20})
        tot += to - int(t.n_bytes)
    return {"bytes": tot, "tensors": per,
            "expert_layers": sorted({int(x["tensor"].split(".")[1]) for x in per if "ffn_down_exps" in x["tensor"]})}

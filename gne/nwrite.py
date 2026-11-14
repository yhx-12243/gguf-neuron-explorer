"""FiSH(Formula-instantiated Scene-gated Hidden units): 計算で作るニューロン(mode nw)。MLP のニューロンを、学習なしで計算した重みで上書きする。docs/fish.md。

ゲートつきの MLP(SwiGLU、バイアスなし)は、層の入力 h(正規化の後)から

    out = W_down · ( silu(W_gate h) ⊙ (W_up h) )

を残差に足す。ニューロン j は W_gate・W_up の j 行と W_down の j 列で決まる。ここでは、ふだんあまり働いていない
ニューロン j の 3 か所を差し替えて、「鍵が反応する位置でだけ方向 d を書く」ニューロンにする。

- 鍵 1 つの形:   gate[j] ← s·κ、up[j] ← κ、down[:, j] ← α·d ÷ silu(s)
                  足す量 = silu(s·ρ)·ρ ÷ silu(s) × α·d(ρ = κᵀh。ρ = 1 でちょうど α·d、小さい反応は 2 乗に近い形で潰れ、
                  負の反応ではほぼ書かない)
- 鍵 2 つの形:   gate[j] ← s·κ_A、up[j] ← κ_B、down[:, j] ← α·d ÷ silu(s)
                  足す量 = silu(s·ρ_A)·ρ_B ÷ silu(s) × α·d(A かつ B。例: 場面の鍵と、返事の位置の鍵)
- 上限つきの形:  ニューロン 2 個を組にする。κ1 は「どの位置でもほぼ 1 を返す鍵」(バイアスの代わり)。
                  1 個目 gate ← s(κ − a·κ1)、up ← κ1、down ← +α·d ÷ (s(b − a))
                  2 個目 gate ← s(κ − b·κ1)、up ← κ1、down ← −α·d ÷ (s(b − a))
                  足す量 = [silu(s(ρ − a)) − silu(s(ρ − b))] ÷ (s(b − a)) × (κ1ᵀh) × α·d(ρ < a でほぼ 0、ρ > b でほぼ 1 で頭打ち)

鍵は MLP の入力の空間(target mlpin、gne direction key --target mlpin)。方向は gne direction の方向(その層の平均差 r_L を
そのままの大きさで使う。keyed と同じ)。当て方は実行時の LoRA(ffn_gate・ffn_up・ffn_down への差し替え)だけで、焼き込みは無い
(Q8_0 の ffn_down は 32 個で 1 つの倍率を共有するので、1 列だけを大きな値に替えると誤差が 9 割を超える)。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import keyed as K
from .steer import GneError

# ゲートつきの MLP を build_ffn(…, LLM_FFN_SILU, LLM_FFN_PAR)で作り、MLP の入力に下の名前を付ける arch
# (llama.cpp の src/models/<arch>.cpp を読んで確かめたものだけ。ほかは ARCH_UNSUPPORTED)
SWIGLU_ARCHS = {"llama": "ffn_norm", "qwen2": "ffn_norm", "qwen3": "ffn_norm", "qwen35": "attn_post_norm"}
S_DEFAULT, S_CAP, CAP_DEFAULT = 4.0, 16.0, (0.4, 0.8)


def silu(z):
    z = np.asarray(z, dtype=np.float64)
    return z / (1.0 + np.exp(-z))


def amount(rho_gate, rho_up, s: float = S_DEFAULT):
    """鍵 1 つ・鍵 2 つの形で足す量(α·d を 1 として)。鍵 1 つなら rho_up = rho_gate。"""
    return silu(float(s) * np.asarray(rho_gate, dtype=np.float64)) * np.asarray(rho_up, dtype=np.float64) / float(silu(s))


def amount_cap(rho, one=1.0, s: float = S_CAP, a: float = CAP_DEFAULT[0], b: float = CAP_DEFAULT[1]):
    """上限つきの形で足す量(α·d を 1 として)。one = κ1ᵀh(ほぼ 1)。"""
    rho = np.asarray(rho, dtype=np.float64)
    return (silu(s * (rho - a * one)) - silu(s * (rho - b * one))) / (s * (b - a)) * one


def check_mlp(weights, L: int) -> None:
    """層 L が、計算で作るニューロンを入れられる MLP か(dense・ゲートつき・SwiGLU・バイアスなし)。違えばエラー。"""
    m = weights.meta
    if not 0 <= int(L) < m.n_layer:
        raise GneError("BAD_RECIPE", f"层 {L} 超出范围（0..{m.n_layer - 1}）")
    if m.n_expert and weights.is_moe(L):
        raise GneError("ARCH_UNSUPPORTED", f"层 {L} 是 MoE。计算生成的神经元（nw）仅支持 dense MLP 层")
    names = [f"blk.{L}.ffn_{p}.weight" for p in ("gate", "up", "down")]
    missing = [n for n in names if not weights.has_tensor(n)]
    if missing:
        raise GneError("ARCH_UNSUPPORTED", f"层 {L} 缺少 {', '.join(missing)}。计算生成的神经元（nw）只支持带门控的 MLP"
                                           "（out = W_down(act(W_gate h) ⊙ W_up h)）层")
    if m.arch not in SWIGLU_ARCHS:
        raise GneError("ARCH_UNSUPPORTED", f"尚未确认 arch {m.arch} 的 MLP 形式（激活函数、门控乘法方式），因此无法使用 nw",
                       f"已确认的 arch：{', '.join(sorted(SWIGLU_ARCHS))}（SiLU 门控 × 输入，无偏置）")
    bias = [n for p in ("gate", "up", "down") for n in (f"blk.{L}.ffn_{p}.bias",) if weights.has_tensor(n)]
    if bias:
        raise GneError("ARCH_UNSUPPORTED", f"层 {L} 的 MLP 含偏置（{', '.join(bias)}）。nw 仅支持无偏置的 MLP")
    if (m.ffn_down_scale or {}).get(L) not in (None, 1.0) or any(weights.has_tensor(f"blk.{L}.ffn_{p}.scale") for p in ("gate", "up", "down")):
        raise GneError("ARCH_UNSUPPORTED", f"层 {L} 的 MLP 权重含缩放（.scale）。尚未确认 nw 在此情况下可用")
    if K.is_mtp_layer(weights, L):
        raise GneError("BAD_RECIPE", f"层 {L} 是 MTP（nextn）层，推演主线不会执行")


def mlp_in_name(seen: set[str], L: int) -> str:
    """層 L の MLP の入力(正規化の後)の捕捉名。arch で決まる名前を、見えた名前から探す。"""
    for base in dict.fromkeys(list(SWIGLU_ARCHS.values())):
        if f"{base}-{L}" in seen and f"ffn_gate-{L}" in seen:
            return f"{base}-{L}"
    raise GneError("ARCH_UNSUPPORTED", f"无法捕获层 {L} 的 MLP 输入（归一化后）张量")


def _gate_matrix(weights, L: int) -> np.ndarray:
    from .weights import tensor_to_f32
    return tensor_to_f32(weights.tensor(f"blk.{L}.ffn_gate.weight"))


def form_of(u) -> dict:
    """ユニットの形: one(鍵 1 つ)/ two(鍵 2 つ)/ cap(上限つき、2 個を組)と、s・a・b。"""
    if u.pair is not None:
        a, b = CAP_DEFAULT
        if u.cap:
            m = re.fullmatch(r"([\d.]+)-([\d.]+)", str(u.cap))
            a, b = float(m.group(1)), float(m.group(2))
        return {"form": "cap", "s": float(u.s or S_CAP), "a": a, "b": b}
    return {"form": "two" if u.key2 else "one", "s": float(u.s or S_DEFAULT)}


def spec_of(u) -> str:
    f = form_of(u)
    return (f"{u.layer}:N{u.neuron}" + (f"+N{u.pair}" if u.pair is not None else "") + f":nw:a{u.k:g}"
            + (f"@{u.direction}" if u.direction else "") + (f"#{u.key}" if u.key else "") + (f"&{u.key2}" if u.key2 else "")
            + (f"!s{f['s']:g}" if u.s is not None else "") + (f"!cap{u.cap}" if u.cap else ""))


def nw_plan(weights, recipe, key_of, dir_of) -> list[dict]:
    """レシピの nw のユニット → 行(ユニットごと)。各行に、差し替えるニューロンごとの (番号, ゲートの行, 入力の行, 出力の列) と、
    量の換算(そのニューロンの活性に掛けると「α·d を 1 とした足す量」への寄与になる倍率)を入れる。確かめもここで行う。"""
    rows, used = [], {}
    for u in recipe.nw_units:
        L = int(u.layer)
        check_mlp(weights, L)
        if L == K.last_main_layer(weights):
            raise GneError("BAD_RECIPE", f"层 {L} 是最后一层。捕获行会被抽稀，因此无法建键", "请使用最后一层之前的层")
        n_ff, n_embd = weights.meta.n_ff_at(L), weights.meta.n_embd
        f = form_of(u)
        js = [int(u.neuron)] + ([int(u.pair)] if u.pair is not None else [])
        for j in js:
            if not 0 <= j < n_ff:
                raise GneError("BAD_RECIPE", f"{u.label()}：L{L} 的神经元 {j} 超出范围（0..{n_ff - 1}）")
            if (L, j) in used:
                raise GneError("BAD_RECIPE", f"正试图用两个单元（{used[(L, j)]}、{u.label()}）覆盖 L{L} 的神经元 {j}",
                               "请为每个单元使用不同的神经元（可用 gne nw neurons 选择活动量小的神经元）")
            used[(L, j)] = u.label()
        dname = u.direction or recipe.default_direction
        kname = u.key or recipe.default_key
        d = np.asarray(K.raw_direction(dir_of(dname), L)[0], dtype=np.float64)
        kg = _kappa(key_of, kname, L, n_embd, u)
        ku = _kappa(key_of, u.key2, L, n_embd, u) if u.key2 else kg
        s, alpha = f["s"], float(u.k)
        if f["form"] == "cap":
            g = 1.0 / (s * (f["b"] - f["a"]))
            parts = [(js[0], s * (kg - f["a"] * ku), ku, alpha * g * d, g), (js[1], s * (kg - f["b"] * ku), ku, -alpha * g * d, -g)]
        else:
            g = 1.0 / float(silu(s))
            parts = [(js[0], s * kg, ku, alpha * g * d, g)]
        rows.append({"unit": u, "layer": L, "direction": dname, "key": kname, "key2": u.key2, **f, "alpha": alpha,
                     "kappa_gate": kg, "kappa_up": ku, "d": d, "parts": parts, "norm": float(abs(alpha) * np.linalg.norm(d))})
    return rows


def _kappa(key_of, name: str | None, L: int, n_embd: int, u) -> np.ndarray:
    if not name:
        raise GneError("BAD_RECIPE", f"{u.label()}：没有键（key 或配方的 default_key）")
    ks = key_of(name)
    if ks.target != K.MLPIN:
        raise GneError("BAD_RECIPE", f"{u.label()}：键 {name} 的写入目标是 {ks.target}。nw 需要 MLP 输入空间中的键（mlpin）",
                       "请用 gne direction key --target mlpin … 创建（keyed 的 ffn 键位于 ffn_down 输入空间，维度不同）")
    if L not in ks.kappa:
        raise GneError("BAD_RECIPE", f"{u.label()}：键 {name} 中没有层 {L}（{ks.layers}）")
    k = np.asarray(ks.kappa_of(L), dtype=np.float64).reshape(-1)
    if k.size != n_embd:
        raise GneError("BAD_KEY", f"键 {name} 的层 {L} 长度 {k.size} 与 n_embd {n_embd} 不一致")
    return k


def nw_tensors(weights, rows: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """行 → LoRA のテンソル {テンソル名: (A, B)}。層ごとに、ffn_gate・ffn_up は A = (新しい行 − 元の行)、B = e_j、
    ffn_down は A = e_j、B = (新しい列 − 元の列)。元の行・列は float に戻した値(量子化のままの掛け算との小さなずれが残る)。"""
    by_layer: dict[int, list] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).extend(r["parts"])
    out = {}
    for L, parts in sorted(by_layer.items()):
        n_ff = weights.meta.n_ff_at(L)
        Wg, Wu, Wd = _gate_matrix(weights, L), weights.up_matrix(L), weights.down_matrix(L)
        E = np.zeros((len(parts), n_ff), dtype=np.float32)
        Ag, Au, Bd = [], [], []
        for i, (j, gate, up, down, _) in enumerate(parts):
            E[i, j] = 1.0
            Ag.append(np.asarray(gate - Wg[j].astype(np.float64), dtype=np.float32))
            Au.append(np.asarray(up - Wu[j].astype(np.float64), dtype=np.float32))
            Bd.append(np.asarray(down - Wd[:, j].astype(np.float64), dtype=np.float32))
        out[f"blk.{L}.ffn_gate.weight"] = (np.stack(Ag), np.ascontiguousarray(E.T))
        out[f"blk.{L}.ffn_up.weight"] = (np.stack(Au), np.ascontiguousarray(E.T))
        out[f"blk.{L}.ffn_down.weight"] = (E.copy(), np.stack(Bd, axis=1))
    return out


def nw_meta(rows: list[dict]) -> dict:
    return {"units": [{"layer": r["layer"], "neurons": [p[0] for p in r["parts"]], "form": r["form"], "s": r["s"],
                       **({"a": r["a"], "b": r["b"]} if r["form"] == "cap" else {}), "alpha": r["alpha"],
                       "direction": r["direction"], "key": r["key"], "key2": r["key2"], "norm": r["norm"]} for r in rows]}


def cache_hash(tens: dict) -> str:
    h = hashlib.sha256()
    for name in sorted(tens):
        for x in tens[name]:
            x = np.ascontiguousarray(x)
            h.update(f"{name}|{x.shape}|".encode())
            h.update(x.tobytes())
    return h.hexdigest()[:16]


NEUTRAL_NAME = "adapter"                       # strip_meta で名前を指定しないときの general.name


def export_nw_lora(weights, recipe, path: str | Path, key_of, dir_of, *, strip_meta: bool = False, name: str | None = None) -> dict:
    """将配方的 nw 单元写入 LoRA 适配器 GGUF（以倍率 1 应用）。
    name 是文件的名称字段（general.name）。省略时为「<原始模型名> nw <配方名>」。
    strip_meta（供他人使用的版本）除 gne 的记录（gne.nw）外，也不写入可识别原始模型的值：不写入原始模型标识符
    （gne.source_model_id），名称字段使用 name（省略时为 "adapter"），并且不写入原始模型名和配方名。"""
    from .ablate import write_lora_tensors
    rows = nw_plan(weights, recipe, key_of, dir_of)
    if not rows:
        raise GneError("BAD_RECIPE", "没有 nw 单元（例 --unit 11:N2045:nw:a1@NAME#KEY）")
    tens = nw_tensors(weights, rows)
    kv = None if strip_meta else {"gne.nw": json.dumps(nw_meta(rows), ensure_ascii=False)}
    if name is not None and not str(name).strip():
        raise GneError("BAD_ARGS", "名称（name）为空")
    label = str(name).strip() if name is not None else (
        NEUTRAL_NAME if strip_meta else f"{weights.meta.name} nw" + (f" {recipe.name}" if recipe.name else ""))
    rep = write_lora_tensors(weights, tens, path, name=label, kv=kv, source_id=not strip_meta)
    return {**rep, "general_name": label, "strip_meta": bool(strip_meta), "nw": nw_meta(rows)["units"]}


# --- ふだんの働き(上書きするニューロンを選ぶ材料)---------------------------------------------------------------

def neuron_usage(session, layers: Sequence[int], *, prompts: Sequence[str] | None = None, max_tokens: int = 96,
                 corpus_tokens: int = 2048, top: int = 20, sort: str = "sum", iv=None) -> dict:
    """層ごとに、各ニューロンのふだんの働きを測り、働きの小さい順の表を返す。
    働き = |silu(gate)·up| × ‖down の列‖(そのニューロンが 1 位置で残差に書くベクトルの長さ)。読ませるのは、ふつうの質問
    (既定のプロンプト集の英語・日本語)への無介入の返事と、既定のコーパスの一部。sort: sum(最大の順位 + 平均の順位、既定)/
    max / mean。表を見て使う人が選ぶ(番号はレシピのユニットに書く)。iv(レシピの名前かパス、またはユニットの書き方)を渡すと、
    その介入を当てた状態で測る(例: CHiPS のレシピを当てた状態で、働きの小さいニューロンを選ぶ)。"""
    import yaml

    from .capture import Capture, pick_act_name
    from .evalx import _generate
    from .stats import corpus_chunks, load_corpus
    e, w = session.engine, session.weights
    Ls = sorted({int(L) for L in layers})
    for L in Ls:
        check_mlp(w, L)
    if sort not in ("sum", "max", "mean"):
        raise GneError("BAD_ARGS", f"sort 必须为 sum / max / mean（{sort!r}）")
    if prompts is None:
        d = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "prompts" / "default.yaml").read_text(encoding="utf-8"))
        prompts = [p for k in ("en", "ja") for p in (d.get(k) or {}).values()]
    t0 = time.perf_counter()
    from .keyed_build import intervention
    vecs, lora, iv_desc = intervention(session, iv)
    seqs = []
    if prompts:
        outs = _generate(session, [(p, int(max_tokens)) for p in prompts], vecs, lora, "all", lambda k: None)
        for p, o in zip(prompts, outs, strict=True):
            pid = e.tokenize_prompt(e.render_chat([{"role": "user", "content": p}]))
            seqs.append(pid + [t for t in o["token_ids"] if not e.is_eog(t)])
    if corpus_tokens:
        seqs += corpus_chunks(e, load_corpus(), 512, int(corpus_tokens))
    seen = e.list_names(seqs[0][:16])
    names = {}
    for L in Ls:
        nm = pick_act_name(seen, L)
        if nm is None:
            raise GneError("ARCH_UNSUPPORTED", f"无法捕获层 {L} 的神经元激活")
        names[L] = nm
    wanted = set(names.values()) | {f"ffn_out-{L}" for L in Ls}
    cap = Capture(lambda n: n in wanted)
    dn = {L: np.asarray(w.down_norms(L), dtype=np.float64) for L in Ls}
    acc = {L: {"sum": np.zeros(dn[L].size), "max": np.zeros(dn[L].size), "n": 0, "out": []} for L in Ls}
    for ids in seqs:
        ctx = e.probe_ctx(cap, need=len(ids))
        e.set_cvec(ctx, vecs or None)
        e.set_lora(ctx, lora or None)
        try:
            e.clear_memory(ctx)
            e.prefill(ctx, [ids])
            got = cap.take()
        finally:
            e.set_cvec(ctx, None)
            e.set_lora(ctx, None)
        for L in Ls:
            c = np.abs(got[names[L]][1:]).astype(np.float64) * dn[L][None, :]
            a = acc[L]
            a["sum"] += c.sum(0)
            a["max"] = np.maximum(a["max"], c.max(0))
            a["n"] += c.shape[0]
            if f"ffn_out-{L}" in got:
                a["out"].append(np.linalg.norm(got[f"ffn_out-{L}"][1:], axis=1))
    rk = lambda v: np.argsort(np.argsort(v)) + 1
    out_layers = []
    for L in Ls:
        a = acc[L]
        mean, mx = a["sum"] / max(a["n"], 1), a["max"]
        r_max, r_mean = rk(mx), rk(mean)
        key = {"sum": r_max + r_mean, "max": mx, "mean": mean}[sort]
        order = np.argsort(key, kind="stable")[:int(top)]
        out_med = float(np.median(np.concatenate(a["out"]))) if a["out"] else None
        out_layers.append({"layer": L, "n_ff": int(mean.size), "positions": int(a["n"]), "out_norm_median": out_med,
                           "median": {"mean": float(np.median(mean)), "max": float(np.median(mx))},
                           "min": {"mean": float(mean.min()), "max": float(mx.min())},
                           "rows": [{"neuron": int(j), "mean": float(mean[j]), "max": float(mx[j]),
                                     "max_share": None if not out_med else float(mx[j] / out_med),
                                     "rank_max": int(r_max[j]), "rank_mean": int(r_mean[j]), "down_norm": float(dn[L][j])}
                                    for j in order]})
    return {"layers": out_layers, "sort": sort, "top": int(top), "n_prompts": len(prompts or []), "max_tokens": int(max_tokens),
            "corpus_tokens": int(corpus_tokens), "iv": iv_desc,
            "notes": ["这是按活动量从小到大排序的表。请查看表格选择要覆盖的神经元，并在配方单元中填写编号。",
                      "覆盖后，该神经元原有的功能将消失。"],
            "seconds": time.perf_counter() - t0}


# --- 鍵(場面の鍵・ほぼ 1 を返す鍵)----------------------------------------------------------------------------
# 位置の鍵(返事の位置の区切りで反応する鍵)は、ふつうの鍵づくり(keyed_build.build_keys、target mlpin、positions pos:<範囲>)で
# 作る。ここは、質問の種類で反応して返事の間ずっと開く「場面の鍵」と、上限つきの形で使う「ほぼ 1 を返す鍵」。

SCENE_BUCKETS = ("pos:0", "pos:1-3", "pos:4-15", "pos:16-")


def build_role_key(session, contrast, *, role: str, layers, name: str | None = None, subset: str = "even",
                   max_tokens: int = 96, pseudo: int = 16, iv=None, seed: int = 0, q_energy: float = 0.99,
                   progress=None, gen_cache: dict | None = None) -> dict:
    """role = scene: 対照集合の質問(user_prompts)への返事の位置で反応し、dont_touch の群の質問への返事では反応しない鍵。
      返事の位置を 0 / 1〜3 / 4〜15 / 16 以降に分けた例ごとの平均を行にして、keyed と同じ式(κ = k/(μᵀk)、k = μ − QQᵀμ)で作る。
      比較用の鍵は、質問のラベル(狙う質問か、dont_touch か)をでたらめに入れ替えて同じ手順で作る。
    role = one: どの位置でもほぼ 1 を返す鍵(κ1ᵀh ≈ 1 の最小二乗)。上限つきの形で、バイアスの代わりに使う。
    どちらも推論時と同じ文脈(system_b。省けば system なし)で、モデル自身の返事(iv があればそれを当てた状態)を読む。
    gen_cache(dict)を渡すと、生成した返事を覚えておいて使い回す(keyed_build.collect_acts。位置の鍵と同じ質問・同じ長さ・
    同じ介入なら、返事を作り直さない)。"""
    from .keyed_build import (MlpInCapture, _as_spec, balanced_split, collect_acts, intervention)
    from .direction import _subset
    from .reverse import _conv
    if role not in ("scene", "one"):
        raise GneError("BAD_ARGS", f"role 必须为 scene / one（{role!r}）")
    t0 = time.perf_counter()
    spec = _as_spec(contrast)
    cs = spec.contrast
    Ls = sorted({int(L) for L in (layers or [])})
    if not Ls:
        raise GneError("BAD_ARGS", "请指定建键层（layers）")
    for L in Ls:
        K.check_layer_ok(session.weights, L, K.MLPIN)
    name = name or f"{cs.name}_{role}"
    if not re.fullmatch(r"[\w.][\w.\-]*", name):
        raise GneError("BAD_ARGS", f"键名称 {name!r} 无效（仅限字母数字、_、.、-）")
    if role == "scene" and not spec.dont_touch:
        raise GneError("BAD_CONTRAST", "场景键需要包含不希望其响应的题组（dont_touch）",
                       "请在对照集合的 dont_touch 中加入普通问题、计算或事实类问题等")
    prompts = cs.prompts("B")
    idx = _subset(len(prompts), subset)
    sets = [("A", [_conv(cs.system_b, prompts[i]) for i in idx])]
    for dt in spec.dont_touch:
        sets.append((f"dt:{dt.name}", dt.convs(_subset(len(dt.prompts), subset))))
    pos = {f"s{i}": b for i, b in enumerate(SCENE_BUCKETS)}
    tc = MlpInCapture(session, Ls)
    # 先に当てておく介入は、ここで 1 回だけ LoRA・ベクトルにして、質問の群ごとの読み取りに同じものを渡す(群ごとに作り直すと、
    # ユニットの書き方で渡した nw の介入では、層ごとの重みを毎回 float に戻して LoRA を組み直すことになり、大きいモデルで遅い)
    iv_vecs, iv_lora, iv_desc = intervention(session, iv)
    iv_once = None if iv is None else (iv_vecs, iv_lora)
    got = {}
    for k, (label, convs) in enumerate(sets):
        got[label] = collect_acts(session, convs, layers=Ls, positions=pos, max_tokens=max_tokens, iv=iv_once, target=tc,
                                  samples=2048 if role == "one" else 0, seed=seed, gen_cache=gen_cache)
        if progress:
            progress(k + 1, len(sets), f"key: {label}")
    n_q = {label: len(convs) for label, convs in sets}

    def ex_rows(label, L):
        """(質問ごとの行のリスト): 質問 i の、区切りごとの例の平均の行。"""
        g = got[label]["groups"]
        per = {i: [] for i in range(n_q[label])}
        for b in pos:
            for row, i in zip(g[b]["means"][L], g[b]["index_l"][L], strict=True):
                per[i].append(row)
        return [np.stack(v) for i, v in sorted(per.items()) if v]

    rng = np.random.default_rng(seed)
    kappa, mu, kn, gain, qrank, exa, exb, ctrl, rows_out = {}, {}, {}, {}, {}, {}, {}, {}, []
    for L in Ls:
        A = ex_rows("A", L)
        B = [r for label, _ in sets[1:] for r in ex_rows(label, L)]
        if role == "one":
            X = np.concatenate([got[label]["groups"][b]["samples"][L] for label, _ in sets for b in pos
                                if got[label]["groups"][b]["samples"][L].shape[0]]).astype(np.float64)
            lam = 1e-2 * float(np.trace(X.T @ X)) / X.shape[1]
            k1 = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.sum(0))
            r = X @ k1
            kappa[L], mu[L], kn[L], gain[L] = k1, X.mean(0), 1.0, 1.0
            rows_out.append({"layer": L, "rows": int(X.shape[0]), "response_mean": float(r.mean()), "response_sd": float(r.std()),
                             "response_min": float(r.min()), "response_max": float(r.max())})
            continue
        if not A or not B:
            raise GneError("BAD_CONTRAST", f"L{L}：没有场景键的行（回复为空）")
        m = np.concatenate(A).mean(0)
        kap, info = K.make_key(m, K.orth_basis(np.concatenate(B), energy=q_energy), warn=session.warn, label=f"L{L}")
        kappa[L], mu[L], kn[L], gain[L], qrank[L] = kap, m, info["kn"], info["gain"], info["q_rank"]
        ea, eb = np.stack([x.mean(0) for x in A]), np.stack([x.mean(0) for x in B])
        exa[L], exb[L] = ea.astype(np.float32), eb.astype(np.float32)
        pool = A + B
        C = []
        for _ in range(int(pseudo)):
            ia, ib = balanced_split(rng, len(A), len(B))
            try:
                C.append(K.make_key(np.concatenate([pool[i] for i in ia]).mean(0),
                                    K.orth_basis(np.concatenate([pool[i] for i in ib]), energy=q_energy))[0])
            except GneError:
                C.append(np.zeros_like(kap))
        if C:
            ctrl[L] = np.stack(C).astype(np.float32)
        sel = float((ea @ kap).mean() - (eb @ kap).mean())
        psel = [float((ea @ c).mean() - (eb @ c).mean()) for c in C]
        rows_out.append({"layer": L, "kn": info["kn"], "q_rank": info["q_rank"], "examples_a": len(A), "examples_b": len(B),
                         "build_selectivity": sel, "pseudo_selectivity_p95": float(np.quantile(psel, 0.95)) if psel else None})
    ks = K.KeySet(target=K.MLPIN, kappa=kappa, mu=mu, kn=kn, gain=gain, q_rank=qrank, exa=exa, exb=exb, kappa_ctrl=ctrl,
                  q_mode="examples", positions=f"role:{role}", max_tokens=int(max_tokens), subset=subset, contrast_sha=cs.sha,
                  counts={"a": n_q["A"], "b": sum(v for k, v in n_q.items() if k != "A")},
                  info={"role": role, "contrast": cs.name, "iv": iv_desc, "buckets": list(SCENE_BUCKETS), "pseudo": len(ctrl.get(Ls[0], [])),
                        "seed": seed, "dont_touch": [dt.name for dt in spec.dont_touch]},
                  model_id=session.meta.model_id, name=name)
    path = ks.save(K.key_dir(session) / f"{name}.npz")
    return {"name": name, "path": str(path), "target": K.MLPIN, "role": role, "contrast": cs.name, "subset": subset,
            "max_tokens": int(max_tokens), "layers": rows_out, "n_pseudo": ks.n_pseudo, "iv": iv_desc,
            "notes": (["build_selectivity 是使用建键问题得到的值（偏乐观）。请用 gne nw check 在未用于建键的问题上确认"]
                      if role == "scene" else ["用于上限形式（L:Nn+Nm:nw:…#键&此键）的第 2 个键"]),
            "sha": ks.sha, "seconds": time.perf_counter() - t0}


# --- 確かめ --------------------------------------------------------------------------------------------------

VERIFY_TEXTS = ["The quick brown fox jumps over the lazy dog. It was a bright cold day in April.",
                "今日は天気が良いので、散歩に出かけることにしました。公園の桜が咲き始めています。"]


def _capture_names(session, ids, L):
    e = session.engine
    seen = e.list_names(ids[:16])
    from .capture import pick_act_name
    act = pick_act_name(seen, L)
    if act is None:
        raise GneError("ARCH_UNSUPPORTED", f"无法捕获层 {L} 的神经元激活")
    return {"h": mlp_in_name(seen, L), "g": f"ffn_gate-{L}", "u": f"ffn_up-{L}", "a": act, "out": f"ffn_out-{L}"}


def _run_capture(session, ids, names: set[str], lora=None, vecs=None) -> dict:
    from .capture import Capture
    e = session.engine
    cap = session.__dict__.get("_nw_capture")
    if cap is None:
        cap = session.__dict__["_nw_capture"] = Capture()
    cap.set_wanted(lambda n: n in names)
    cap.skipped = {}
    ctx = e.probe_ctx(cap, need=len(ids))
    e.set_cvec(ctx, vecs or None)
    e.set_lora(ctx, lora or None)
    try:
        e.clear_memory(ctx)
        e.prefill(ctx, [list(ids)])
        got = cap.take()
    finally:
        e.set_cvec(ctx, None)
        e.set_lora(ctx, None)
    miss = [n for n in names if n not in got]
    if miss:
        raise GneError("ARCH_UNSUPPORTED", f"无法捕获 {', '.join(sorted(miss))}")
    return {k: v.astype(np.float64) for k, v in got.items()}


def verify(session, recipe, *, texts: Sequence[str] | None = None, iv=None) -> dict:
    """nw の LoRA が式どおりに動くか。層ごとに、その層のユニットだけの LoRA を当てて短い文を読ませ、次を確かめる:
    上書きしたニューロンのゲート・入力が (新しい行)ᵀh になっているか、活性が silu(ゲート)·入力 か、ほかのニューロンの活性が
    変わっていないか、MLP の出力の変化が (新しい活性 × 新しい列) − (元の活性 × 元の列) の和になっているか。
    元の行・列は float に戻した値を引くので、量子化のままの掛け算との小さなずれ(入力の大きさの 1% ほど)は残る。
    iv を渡すと、その介入(レシピかユニット)を先に当てた状態で確かめる(当てない側・当てる側の両方に iv が掛かる)。"""
    from .ablate import write_lora_tensors
    from .keyed_build import intervention
    iv_vecs, iv_lora, iv_desc = intervention(session, iv)
    iv_lora = list(iv_lora or [])
    e, w = session.engine, session.weights
    rows = nw_plan(w, recipe, session._key_of, session.direction_set)
    if not rows:
        raise GneError("BAD_RECIPE", "没有 nw 单元")
    texts = list(texts or VERIFY_TEXTS)
    seqs = [e.tokenize(t, add_special=True, parse_special=False) for t in texts]
    out_layers = []
    tmp_dir = K.key_dir(session) / "lora"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for L in sorted({r["layer"] for r in rows}):
        sub = [r for r in rows if r["layer"] == L]
        tens = nw_tensors(w, sub)
        path = tmp_dir / f"nw-verify-{cache_hash(tens)}.lora.gguf"
        write_lora_tensors(w, tens, path, name="nw verify")
        nm = _capture_names(session, seqs[0], L)
        Wd = w.down_matrix(L).astype(np.float64)
        js = [p[0] for r in sub for p in r["parts"]]
        stats = {"gate": [], "up": [], "act": [], "others": [], "out_rel": [], "out_norm": []}
        try:
            for ids in seqs:
                b = _run_capture(session, ids, set(nm.values()), iv_lora, iv_vecs)
                g = _run_capture(session, ids, set(nm.values()), [*iv_lora, (str(path), 1.0)], iv_vecs)
                h = g[nm["h"]][1:]
                pred = -(b[nm["a"]][1:, js] @ Wd[:, js].T)
                # 誤差は「予測の最大」か「その層のふつうのニューロンの値の大きさ」の大きいほうに対する比(鍵が反応しない文でも割れるように)
                floor_g = float(np.sqrt(np.mean(np.square(b[nm["g"]][1:]))))
                floor_u = float(np.sqrt(np.mean(np.square(b[nm["u"]][1:]))))
                for r in sub:
                    for (j, gate, up, down, _) in r["parts"]:
                        gj, uj = h @ gate, h @ up
                        stats["gate"].append(float(np.abs(g[nm["g"]][1:, j] - gj).max() / max(float(np.abs(gj).max()), floor_g)))
                        stats["up"].append(float(np.abs(g[nm["u"]][1:, j] - uj).max() / max(float(np.abs(uj).max()), floor_u)))
                        aj = silu(g[nm["g"]][1:, j]) * g[nm["u"]][1:, j]
                        stats["act"].append(float(np.abs(g[nm["a"]][1:, j] - aj).max() / max(float(np.abs(aj).max()), 1e-12)))
                        pred = pred + np.outer(g[nm["a"]][1:, j], down)
                oth = np.delete(g[nm["a"]][1:] - b[nm["a"]][1:], js, axis=1)
                stats["others"].append(float(np.abs(oth).max()))
                dout = g[nm["out"]][1:] - b[nm["out"]][1:]
                pn = np.linalg.norm(pred, axis=1)
                big = pn >= np.quantile(pn, 0.5)
                stats["out_rel"].append(float(np.median(np.linalg.norm(dout - pred, axis=1)[big] / np.maximum(pn[big], 1e-12))))
                stats["out_norm"].append(float(pn.max()))
        finally:
            e.drop_lora(path)
            path.unlink(missing_ok=True)
        row = {"layer": L, "neurons": js, "gate_err": max(stats["gate"]), "up_err": max(stats["up"]), "act_err": max(stats["act"]),
               "others_changed_max": max(stats["others"]), "out_rel_err_median": float(np.median(stats["out_rel"])),
               "out_change_max": max(stats["out_norm"])}
        # ゲート・入力の誤差には、量子化のままの掛け算と float に戻した行を引く計算のずれ(1〜2% ほど)が残る
        row["ok"] = bool(row["gate_err"] < 0.05 and row["up_err"] < 0.05 and row["act_err"] < 1e-3
                         and row["others_changed_max"] < 1e-4 and row["out_rel_err_median"] < 0.05)
        out_layers.append(row)
    return {"layers": out_layers, "ok": all(r["ok"] for r in out_layers), "n_texts": len(texts), "iv": iv_desc,
            "thresholds": {"gate_err": 0.05, "up_err": 0.05, "act_err": 1e-3, "others_changed_max": 1e-4, "out_rel_err_median": 0.05}}


LOW_DEFAULT, OVER_DEFAULT, LEAK_DEFAULT = 0.4, 1.2, 0.2      # 質問ごとの値の目安: ほとんど書けていない / 狙った量を超えた / 書いてしまった


def unit_segment(session, r: dict) -> str:
    """ユニット(nw_plan の行)が担当する返事の位置の区間。鍵の positions(pos:<範囲>。上限つき・鍵 1 つの形は 1 本目の鍵、
    鍵 2 つの形は入力の側の鍵)から読む。位置の鍵でなければ方向の名前の .seg<i>、それも無ければ返事の全部("0-")。"""
    from .direction import parse_segments, segment_text
    for nm in (r["key"], r["key2"]):
        if not nm:
            continue
        try:
            pos = str(session._key_of(nm).positions or "")
        except GneError:
            continue
        if pos.startswith("pos:"):
            return segment_text(parse_segments(pos[4:])[0])
    m = re.search(r"^(.*)\.seg(\d+)$", str(r["direction"] or ""))
    if m:
        try:
            segs = list(session.direction_set(m.group(1)).segments)
            return segment_text(parse_segments(segs[int(m.group(2))])[0])
        except (GneError, IndexError):
            pass
    return "0-"


def _question_summary(rows: list[dict], segs: list[str], wanted: bool, low: float, over: float, leak: float) -> dict:
    """質問ごとの行 → 区間ごとの数: n(その区間の位置があった質問)、low(書いた量が low 未満)、over(合計が over を超えた)、
    leak(合計の大きさが leak 以上)、書いた量の最小・中央値。low・over は狙う場面で、leak は書き込みたくない場面で見る。"""
    out = {}
    for sg in segs:
        xs = [q["segments"][sg] for q in rows if q["segments"].get(sg)]
        am = [x["amount"] for x in xs]
        out[sg] = {"n": len(xs), "low": sum(a < low for a in am) if wanted else None,
                   "over": sum(x["total"] > over for x in xs) if wanted else None,
                   "leak": None if wanted else sum(abs(x["total"]) >= leak for x in xs),
                   "amount_min": float(min(am)) if am else None, "amount_median": float(np.median(am)) if am else None}
    return out


def check(session, recipe, contrast, *, subset: str = "odd", max_tokens: int = 96, replies: str = "own", n_pos: int = 8,
          iv=None, positions: str | None = None, low: float = LOW_DEFAULT, over: float = OVER_DEFAULT,
          leak: float = LEAK_DEFAULT, top: int = 5, save: str | None = None) -> dict:
    """位置の種類ごとに、鍵の反応と実際に足した量を出す。対照集合の質問(狙う場面)と dont_touch の群(書き込みたくない場面)に
    ついて、返事の位置 0〜n_pos−1 とそれ以降で、ユニットごとの「鍵の反応(ゲートの鍵・入力の鍵)」と「足した量(α·d を 1 として)」の
    平均・最小・最大を返す。replies = own: レシピを当てて作った返事を、当てたまま読む(実際に起きること)。base: 無介入の返事を、
    レシピを当てて読む(書く前の状態での見積もり)。iv を渡すと、その介入(レシピかユニット)を先に当てた状態で測る
    (返事の生成にも読むときにも掛かる。base の返事は iv だけを当てて作る)。
    positions(before:<正規表現>): 返事の位置の番号ではなく、「正規表現に一致する文字の直前の位置(match)」と「それ以外の位置(other)」に
    分けて出す(文字の直前に書く鍵用。書くとその文字が出なくなるので、replies="base" と一緒に使う)。

    質問ごとの値(groups.<群>.questions): 平均だけでは、一部の質問でまったく書けていないことが隠れるので、質問 1 つずつ・区間ごとに
    出す。区間は、ユニットが担当する返事の位置の範囲(unit_segment。位置の鍵の pos:<範囲>。positions を渡したときは "match" の 1 つ)。
    - amount: その区間の位置での、その区間を担当するユニットの書いた量(狙った量 = 1。層ごとのユニットの平均を、位置で平均)。
    - total: 同じ位置で、全部のユニットが書いた量の合計(区間ごとの、層の平均の和。隣の区間のユニットが重ねて書いた分も入る)。
    - rho: その区間の鍵の反応の平均(上限つきの形は「鍵の反応 ÷ ほぼ 1 を返す鍵の反応」で、しきい値 a・b と比べる値)。
    groups.<群>.by_question は区間ごとの数: 狙う場面では low(amount が low 未満 = ほとんど書けていない質問)と over(total が over を
    超えた = 狙った量を超えて書いた質問)、書き込みたくない場面では leak(total の大きさが leak 以上 = 書いてしまった質問)。
    question_summary.worst は、狙う場面でほとんど書けていない質問の、量の小さい順の上位 top 件(質問と、返事の出だし)。
    leaks は、書き込みたくない場面の群ごとの漏れの表(leak_table): 群ごと・区間ごとに、合計の大きさが leak 以上の質問の数と、書いた量
    (合計)の平均。どれかの区間で書いた質問の数(any)と、位置 0(返事を書き始める直前)だけで見た数(pos0)も出す。
    key_fit は、ユニットの鍵がどちらの作り方で作ったものか(mean = いまの作り方、ridge = 最小二乗の鍵。鍵ごとの値は keys)。
    save(名前)を渡すと、結果を <root>/behave/<model_id>/<名前>_check.json に書く(結果の path)。"""
    from .direction import _subset, parse_segments
    from .keyed_build import intervention
    iv_vecs, iv_lora, iv_desc = intervention(session, iv)
    iv_lora = list(iv_lora or [])
    from .behave import loop_like
    from .evalx import _generate
    from .keyed_build import _as_spec
    from .reverse import _conv, question_text
    if replies not in ("own", "base"):
        raise GneError("BAD_ARGS", f"replies 必须为 own / base（{replies!r}）")
    if not (0 <= float(low) and float(over) > 0 and float(leak) > 0 and int(top) >= 0):
        raise GneError("BAD_ARGS", "low 必须 ≥ 0，over、leak 必须 > 0，top 必须 ≥ 0")
    if save is not None and not re.fullmatch(r"[\w.\-]+", str(save)):
        raise GneError("BAD_ARGS", f"结果名称 {save!r} 只能包含字母、数字、_、-、.")
    t0 = time.perf_counter()
    e, w = session.engine, session.weights
    rows = nw_plan(w, recipe, session._key_of, session.direction_set)
    if not rows:
        raise GneError("BAD_RECIPE", "没有 nw 单元")
    lora = [*iv_lora, (str(session.nw_lora(recipe)), 1.0)]
    spec = _as_spec(contrast)
    cs = spec.contrast
    prompts = cs.prompts("B")
    idx = _subset(len(prompts), subset)
    groups = [("scene", [_conv(cs.system_b, prompts[i]) for i in idx], [question_text(prompts[i]) for i in idx])]
    for dt in spec.dont_touch:
        di = _subset(len(dt.prompts), subset)
        groups.append((f"dont_touch:{dt.name}", dt.convs(di), [question_text(dt.prompts[i]) for i in di]))
    Ls = sorted({r["layer"] for r in rows})
    names = None
    buckets = [str(k) for k in range(n_pos)] + [f"{n_pos}+"]
    if positions is not None:
        from .keyed_build import parse_positions, select_positions
        if parse_positions(positions).kind != "before":
            raise GneError("BAD_ARGS", f"check 的 positions 必须为 before:<正则表达式>（{positions!r}）")
        buckets = ["match", "other"]
    # 質問ごとの値の区間: ユニットごとに、担当する返事の位置の範囲(positions を渡したときは、一致の直前の位置 "match" の 1 つ)
    seg_of = {id(r): ("match" if positions is not None else unit_segment(session, r)) for r in rows}
    seg_names = list(dict.fromkeys(seg_of.values()))
    if positions is None:
        seg_names.sort(key=lambda sg: parse_segments(sg)[0][0])
    seg_range = {sg: (None if positions is not None else parse_segments(sg)[0]) for sg in seg_names}
    res = {}
    for label, convs, qtexts in groups:
        outs = _generate(session, [(c, int(max_tokens)) for c in convs], iv_vecs, lora if replies == "own" else (iv_lora or None),
                         "all", lambda k: None)
        acc = {id(r): {b: {"amount": [], "rho_gate": [], "rho_up": []} for b in buckets} for r in rows}
        loops = 0
        qrows = []
        for qi, (c, o) in enumerate(zip(convs, outs, strict=True)):
            p = e.tokenize_prompt(e.render_chat(c))
            rid = [t for t in o["token_ids"] if not e.is_eog(t)]
            looped = bool(loop_like(o["token_ids"]))
            loops += looped
            ids = p + rid
            if names is None:
                names = {L: _capture_names(session, ids, L) for L in Ls}
            got = _run_capture(session, ids, {names[L][k] for L in Ls for k in ("h", "a")}, lora, iv_vecs)
            hit = None
            if positions is not None:                           # 列の番号 → 返事の位置 k = 番号 − (P − 1)
                hit = {int(x) - (len(p) - 1) for x in select_positions(e, rid, positions, n_prompt=len(p))}
            per = {sg: {"amount": [], "rho": []} for sg in seg_names}       # 区間 → その区間を担当するユニットごとの、位置の並び
            for r in rows:
                L = r["layer"]
                h = got[names[L]["h"]][len(p) - 1:]
                a = got[names[L]["a"]][len(p) - 1:]
                amt = sum(g * a[:, j] for (j, _, _, _, g) in r["parts"])
                rg, ru = h @ r["kappa_gate"], h @ r["kappa_up"]
                for k in range(amt.shape[0]):
                    b = ("match" if k in hit else "other") if hit is not None else (str(k) if k < n_pos else f"{n_pos}+")
                    acc[id(r)][b]["amount"].append(float(amt[k]))
                    acc[id(r)][b]["rho_gate"].append(float(rg[k]))
                    acc[id(r)][b]["rho_up"].append(float(ru[k]))
                # 区間の鍵の反応: 上限つきの形は、ほぼ 1 を返す鍵の反応で割った値(しきい値 a・b と比べる値)。鍵 2 つの形は入力の側の鍵
                rho = (rg / np.where(np.abs(ru) > 1e-6, ru, 1.0)) if r["form"] == "cap" else (ru if r["form"] == "two" else rg)
                per[seg_of[id(r)]]["amount"].append(np.asarray(amt, dtype=np.float64))
                per[seg_of[id(r)]]["rho"].append(np.asarray(rho, dtype=np.float64))
            n_k = len(rid) + 1
            own = {sg: np.mean(per[sg]["amount"], axis=0) for sg in seg_names}       # 区間ごとの、ユニット(層)の平均
            total = np.sum([own[sg] for sg in seg_names], axis=0)
            ks = np.arange(n_k)
            qsegs = {}
            for sg in seg_names:
                if positions is not None:
                    mask = np.isin(ks, sorted(hit))
                else:
                    lo, hi = seg_range[sg]
                    mask = (ks >= lo) & ((ks < hi) if hi is not None else True)
                qsegs[sg] = None if not mask.any() else {
                    "amount": float(own[sg][mask].mean()), "total": float(total[mask].mean()),
                    "rho": float(np.mean(per[sg]["rho"], axis=0)[mask].mean()), "n_pos": int(mask.sum())}
            qrows.append({"i": qi, "prompt": qtexts[qi], "reply_start": (o["text"] or "")[:40], "n_tokens": len(rid), "loop": looped,
                          "segments": qsegs, "pos0_total": float(total[0])})
        st = lambda v: None if not v else {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
        res[label] = {"n": len(convs), "loops": loops,
                      "units": [{"unit": spec_of(r["unit"]), "layer": r["layer"], "form": r["form"],
                                 "positions": {b: {k: st(v) for k, v in d.items()} for b, d in acc[id(r)].items()}} for r in rows],
                      "by_question": _question_summary(qrows, seg_names, label == "scene", float(low), float(over), float(leak)),
                      "questions": qrows}
    worst = sorted(({"group": "scene", "segment": sg, "i": q["i"], "prompt": q["prompt"], "reply_start": q["reply_start"],
                     "amount": q["segments"][sg]["amount"], "rho": q["segments"][sg]["rho"], "loop": q["loop"]}
                    for q in res["scene"]["questions"] for sg in seg_names
                    if q["segments"].get(sg) and q["segments"][sg]["amount"] < float(low)), key=lambda x: x["amount"])
    caps = sorted({(r["a"], r["b"]) for r in rows if r["form"] == "cap"})
    out = {"contrast": cs.name, "subset": subset, "replies": replies, "max_tokens": int(max_tokens), "iv": iv_desc, "groups": res,
           "positions": positions, "key_fit": units_key_fit(session, rows), "leaks": leak_table(res, seg_names, float(leak)),
           "question_summary": {"low": float(low), "over": float(over), "leak": float(leak), "segments": seg_names,
                                "cap": [list(x) for x in caps], "n_low": len(worst), "worst": worst[:int(top)]},
           "notes": ["amount 是该位置实际写入的量（以 α·d 为 1。负值表示反向）。rho_gate、rho_up 是键的响应。",
                     "scene 是目标场景（对照集合中的问题），dont_touch:<名称> 是不希望写入的场景。",
                     "平均值会掩盖部分问题完全未写入的情况。请查看逐题数值（by_question、questions），"
                     "确认是否有几乎未写入的问题。"],
           "seconds": time.perf_counter() - t0}
    if save is not None:
        path = session.store.root / "behave" / session.meta.model_id / f"{save}_check.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out = {**out, "path": str(path)}
    return out


def units_key_fit(session, rows: list[dict]) -> dict:
    """ユニット(nw_plan の行)の鍵が、どちらの作り方で作ったものか。{"fit": mean / ridge / mixed, "keys": {鍵の名前: 作り方},
    "c": {鍵の名前: 最小二乗の鍵の正則化の強さ}}。ほぼ 1 を返す鍵・場面の鍵(role のある鍵)は数えない。記録の無い鍵は mean。"""
    keys, cs = {}, {}
    for r in rows:
        for nm in (r["key"], r["key2"]):
            if not nm or nm in keys:
                continue
            try:
                info = session._key_of(nm).info or {}
            except GneError:
                continue
            if info.get("role"):
                continue
            keys[nm] = str(info.get("key_fit") or "mean")
            if keys[nm] == "ridge":
                cs[nm] = (info.get("ridge") or {}).get("c")
    kinds = set(keys.values())
    return {"fit": "mean" if not kinds else (kinds.pop() if len(kinds) == 1 else "mixed"), "keys": keys, "c": cs}


def key_fit_line(kf: dict | None) -> str:
    """units_key_fit の結果を 1 行に(gne nw check の画面)。記録が無ければ mean(いまの作り方)。"""
    kf = kf or {}
    fit = kf.get("fit") or "mean"
    if fit == "ridge":
        cs = sorted({format(float(v), "g") for v in (kf.get("c") or {}).values() if v is not None}, key=float)
        return "键的构建方式：ridge（最小二乘键，按单元格加权" + (f"。正则化强度 c = {'、'.join(cs)}" if cs else "") + "）"
    if fit == "mixed":
        return "键的构建方式：mean 与 ridge 混合（" + "、".join(f"{k} {v}" for k, v in (kf.get("keys") or {}).items()) + "）"
    return "键的构建方式：mean（现有做法）"


def leak_table(groups: dict, segs: list[str], leak: float) -> dict:
    """書き込みたくない場面の群ごとの漏れ。check の groups(群 → questions)から、群ごとに:
    n(質問の数)、any(どれかの区間で、合計の大きさが leak 以上の質問の数)、segments(区間ごとの leak = その数、n、total_mean =
    書いた量(合計)の平均、abs_max = 大きさの最大)、pos0(位置 0 = 返事を書き始める直前の 1 位置だけで見た同じ数)。
    all は全部の群を合わせた行。漏れが、ある区間から位置 0 に移っただけ、ということがあるので、位置 0 は別に出す。"""
    def cell(vals):
        return {"n": len(vals), "leak": sum(abs(v) >= leak for v in vals), "total_mean": float(np.mean(vals)) if vals else None,
                "abs_max": float(np.max(np.abs(vals))) if vals else None}

    def row(name, qs):
        per = {sg: [q["segments"][sg]["total"] for q in qs if q["segments"].get(sg)] for sg in segs}
        return {"group": name, "n": len(qs),
                "any": sum(any(q["segments"].get(sg) and abs(q["segments"][sg]["total"]) >= leak for sg in segs) for q in qs),
                "segments": {sg: cell(v) for sg, v in per.items()},
                "pos0": cell([q["pos0_total"] for q in qs if q.get("pos0_total") is not None])}
    rows = [row(g.split(":", 1)[1], x["questions"]) for g, x in groups.items() if g != "scene"]
    allq = [q for g, x in groups.items() if g != "scene" for q in x["questions"]]
    return {"leak": float(leak), "segments": list(segs), "groups": rows, "all": row("all", allq) if rows else None}


def leak_lines(d: dict) -> list[str]:
    """leak_table の表を、画面に出す行にする。"""
    lk = d.get("leaks")
    if not lk or not lk.get("groups"):
        return []
    f = lambda c: "—" if not c["n"] else f"{c['leak']}/{c['n']}({c['total_mean']:+.2f})"
    lines = [f"不希望写入的场景按组统计的泄漏（写入目标量 {lk['leak']:g} 以上的问题数 / 问题数。括号内为平均写入量）",
             f"  {'组':<12} {'任一分段':>10}  " + "  ".join(f"分段 {sg}" for sg in lk["segments"]) + "  仅位置 0"]
    for r in lk["groups"] + ([{**lk["all"], "group": "（全部）"}] if lk.get("all") and len(lk["groups"]) > 1 else []):
        lines.append(f"  {r['group']:<12} {str(r['any']) + '/' + str(r['n']):>10}  " + "  ".join(f(r["segments"][sg]) for sg in lk["segments"])
                     + "  " + f(r["pos0"]))
    lines.append("  请按组查看。确认是否只泄漏到与目标场景形式相同的发言组（如非问题发言），以及泄漏是否只是从某个分段转移到了位置 0。")
    return lines


def question_lines(d: dict) -> list[str]:
    """check の結果の、質問ごとの値の要約(画面に出す行)。区間ごとに、ほとんど書けていない質問の数と、狙った量を超えて書いた質問の数、
    書けていない質問の上位(質問と、返事の出だし 20 字)。書き込みたくない場面は、書いてしまった質問の数。"""
    qs = d.get("question_summary")
    if not qs:
        return []
    lines = ["逐题数值（平均值会掩盖部分问题完全未写入的情况）"]
    cap = "、".join(f"{a:g}-{b:g}" for a, b in qs.get("cap") or [])
    for g, x in d["groups"].items():
        bq = x.get("by_question") or {}
        if g == "scene":
            for sg, c in bq.items():
                lines.append(f"  [{g}] 区间 {sg}：{c['n']} 个问题中，几乎未写入（写入量低于 {qs['low']:g}）{c['low']} 个，"
                             f"超过目标量（合计高于 {qs['over']:g}）{c['over']} 个"
                             + ("" if c["amount_min"] is None else f"（写入量最小 {c['amount_min']:+.2f}、中位数 {c['amount_median']:+.2f}）"))
        else:
            lines.append(f"  [{g}] 误写入的问题（合计绝对值不低于 {qs['leak']:g}）："
                         + "、".join(f"区间 {sg} {c['leak']} / {c['n']} 个" for sg, c in bq.items()))
    if qs["worst"]:
        lines.append(f"  几乎未写入的问题（按写入量从小到大；显示 {len(qs['worst'])} 个，共 {qs['n_low']} 个）：")
        for q in qs["worst"]:
            lines.append(f"    区间 {q['segment']}  写入量 {q['amount']:+.2f}  键响应 {q['rho']:+.2f}  Q: {q['prompt'][:40]}  A: {q['reply_start'][:20]}"
                         + ("  [重复]" if q.get("loop") else ""))
        lines.append("  请查看几乎未写入的问题是否集中在某种回复开头方式（例如以引号开头）。若集中，请增加用于建键的问题"
                     "（该开头方式的回复在建键问题中较少）。")
    else:
        lines.append("  目标场景中没有几乎未写入的问题。")
    if cap:
        lines.append(f"  参考：带上限的形式（阈值 {cap}）在键响应低于下限阈值时几乎不写入，在两个阈值中点写入目标量的一半，"
                     "高于上限阈值时封顶在目标量。写入量较小的问题，其键响应通常接近或低于下限阈值。")
    return lines + leak_lines(d)

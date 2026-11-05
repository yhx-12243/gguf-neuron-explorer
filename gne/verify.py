"""Phase 0 验证测试 V1～V8。设计文档 §9。可由 `gne verify` 与 pytest 共同使用。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .capture import Capture, pick_act_name
from .engine import Engine, log_softmax
from .steer import cvec_buffer, export_cvec_gguf, load_cvec_like_common
from .weights import Weights

PROMPTS = [
    "Who are you?",
    "Describe the color orange in two sentences.",
    "あなたは誰ですか？自己紹介してください。",
]

# V2 の許容値(相対誤差の中央値, 最大)。誤差の主因は重みではなく、量子化 GEMM が活性を q8_1 に丸めること
# と半精度の累積なので、重みの型による差は小さい。重みが粗い型は念のため緩める(R7)。
V2_TOL = {
    "F32": (0.02, 0.10), "F16": (0.02, 0.10), "BF16": (0.02, 0.10), "Q8_0": (0.02, 0.10),
    "Q6_K": (0.03, 0.12), "Q5_K": (0.03, 0.12), "Q5_0": (0.03, 0.12), "Q5_1": (0.03, 0.12),
    "Q4_K": (0.04, 0.15), "Q4_0": (0.04, 0.15), "Q4_1": (0.04, 0.15), "IQ4_XS": (0.04, 0.15), "IQ4_NL": (0.04, 0.15),
}
V2_TOL_DEFAULT = (0.06, 0.20)


@dataclass
class VResult:
    id: str
    title: str
    status: str                     # "pass" | "fail" | "info" | "skip"
    detail: str
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "info", "skip")


class Verifier:
    def __init__(self, engine: Engine, weights: Weights, *, layer: int | None = None, seed: int = 0):
        self.e, self.w = engine, weights
        n = engine.n_layer
        self.L = layer if layer is not None else n // 2
        if not 1 <= self.L <= n - 3:
            raise ValueError(f"--layer 须为 1..{n - 3}（为了捕获 L-1, L, L+1, L+2）")
        self.rng = np.random.default_rng(seed)
        self.prompt_ids = [engine.tokenize_prompt(engine.render_chat([{"role": "user", "content": p}]))
                           for p in PROMPTS]
        self._rho: float | None = None

    # --- helpers --------------------------------------------------------------------------------

    def capture(self, names: set[str], seqs: list[list[int]], vectors=None, layer_range=None) -> dict[str, np.ndarray]:
        cap = self._cap = getattr(self, "_cap", None) or Capture()
        cap.set_wanted(lambda n: n in names)
        ctx = self.e.probe_ctx(cap)
        self.e.clear_memory(ctx)
        self.e.set_cvec(ctx, vectors, layer_range=layer_range)
        self.e.prefill(ctx, seqs)
        out = cap.take()
        missing = names - out.keys()
        if missing:
            raise RuntimeError(f"无法捕获的张量：{sorted(missing)}（skipped={cap.skipped}）")
        return out

    def rho(self) -> float:
        """第 L 层残差范数的中位数（每条序列的 position ≥ 1）。用于 V3/V4/V5 的向量尺度。"""
        if self._rho is None:
            norms = []
            for ids in self.prompt_ids:
                x = self.capture({f"l_out-{self.L}"}, [ids])[f"l_out-{self.L}"]
                norms.append(np.linalg.norm(x[1:], axis=1))
            self._rho = float(np.median(np.concatenate(norms)))
        return self._rho

    # --- V1 -------------------------------------------------------------------------------------

    def v1_shapes(self) -> VResult:
        m = self.w.meta
        bad = []
        if m.n_expert:                                  # 全層 MoE: ne = [n_ff, n_embd, n_expert]、1 専門家ぶんだけ dequantize
            for L in range(m.n_layer):
                name = f"blk.{L}.ffn_down_exps.weight"
                ne = [int(x) for x in self.w.tensor(name).shape] if self.w.has_tensor(name) else None
                if ne != [m.n_ff_at(L), m.n_embd, m.n_expert]:
                    bad.append(f"{name} ne={ne}")
            mat = self.w.expert_down(self.L, 0)
            if mat.shape != (m.n_embd, m.n_ff_at(self.L)) or not np.isfinite(mat).all():
                bad.append(f"L{self.L} 专家 0 的 dequant {mat.shape}")
            return VResult("V1", "形状与 dequantize", "fail" if bad else "pass",
                           f"全部 {m.n_layer} 层的 ne=[n_ff, n_embd, n_expert]，L{self.L} 专家 0 做 dequantize"
                           + (f" / 不一致: {bad}" if bad else ""), {"bad": bad})
        for L in range(m.n_layer):
            name = f"blk.{L}.ffn_down.weight"
            if not self.w.has_tensor(name):
                bad.append(f"缺少 {name}")
                continue
            ne = [int(x) for x in self.w.tensor(name).shape]
            if ne != [m.n_ff_at(L), m.n_embd]:
                bad.append(f"{name} ne={ne}")
        checked = sorted({1, self.L, m.n_layer - 2})
        for L in checked:
            mat = self.w.down_matrix(L)
            if mat.shape != (m.n_embd, m.n_ff_at(L)) or not np.isfinite(mat).all():
                bad.append(f"L{L} dequant {mat.shape}")
        st = "fail" if bad else "pass"
        return VResult("V1", "形状与 dequantize", st,
                       f"全部 {m.n_layer} 层的 ne=[n_ff, n_embd]，L{checked} dequantize → (n_embd, n_ff)=({m.n_embd}, {m.n_ff}) {m.ffn_down_qtype}"
                       + (f" / 不一致：{bad}" if bad else ""), {"bad": bad})

    # --- V26: 残差が複数本の arch(hyper-connections)--------------------------------------------

    def v26_hc_streams(self) -> VResult:
        """層の FFN の書き込みは、どの流れにも同じ向きで足される(倍率だけ違う)。だから流れの平均を残差として読んでよい。
        l_last-L(FFN の後)− 最初の hc_combine-L(attention の後)が、流れごとに ffn_out-L の定数倍であることを確かめる。"""
        e, L = self.e, self.L
        hc = e.hc_streams
        names = {f"l_last-{L}", f"hc_combine-{L}", f"ffn_out-{L}"}
        cap = Capture()
        ctx = e.probe_ctx(cap, need=max(len(x) for x in self.prompt_ids))
        cap.hc_streams = 0                               # ここだけ生の [n_embd, hc, T] を見る
        cap.set_wanted(lambda n: n in names)
        cos, scale = [], []
        try:
            for ids in self.prompt_ids:
                e.clear_memory(ctx)
                e.prefill(ctx, [ids])
                got = cap.take()
                missing = names - set(got)
                if missing:
                    return VResult("V26", "残差流", "fail", f"无法捕获：{sorted(missing)}")
                n = e.n_embd
                d = (got[f"l_last-{L}"] - got[f"hc_combine-{L}"]).reshape(-1, hc, n).astype(np.float64)
                f = got[f"ffn_out-{L}"].astype(np.float64)
                keep = np.linalg.norm(f, axis=1) > 0
                d, f = d[keep], f[keep]
                dot = np.einsum("tcn,tn->tc", d, f)
                nn = np.linalg.norm(d, axis=2) * np.linalg.norm(f, axis=1, keepdims=True)
                cos.append(np.abs(dot / np.maximum(nn, 1e-30)).ravel())
                scale.append((dot / np.maximum(np.einsum("tn,tn->t", f, f)[:, None], 1e-30)).ravel())
        finally:
            cap.hc_streams = hc
        cos, scale = np.concatenate(cos), np.concatenate(scale)
        ok = bool(np.quantile(cos, 0.01) > 0.999 and scale.min() >= -1e-3 and scale.max() <= 2 + 1e-3)
        return VResult("V26", "残差流", "pass" if ok else "fail",
                       f"{hc} 条流，L{L}：FFN 写入与 ffn_out 的 |cos| 1% 分位 {np.quantile(cos, 0.01):.5f}、"
                       f"各流的倍率 {scale.min():.3f}～{scale.max():.3f}（应在 0～2 之间）",
                       {"cos_q01": float(np.quantile(cos, 0.01)), "scale": [float(scale.min()), float(scale.max())]})

    # --- V2 -------------------------------------------------------------------------------------

    def v2_ffn_linearity(self) -> VResult:
        L = self.L
        names = self.list_names(L)
        act = pick_act_name(names, L)
        if act is None:
            return VResult("V2", "FFN 线性性", "fail", f"找不到 ffn_down 的输入张量：{sorted(names)}")
        caps = [self.capture({act, f"ffn_out-{L}"}, [ids]) for ids in self.prompt_ids]
        a = np.concatenate([c[act] for c in caps])
        out = np.concatenate([c[f"ffn_out-{L}"] for c in caps])
        W = self.w.down_matrix(L)
        s = (self.w.meta.ffn_down_scale or {}).get(L, 1.0)
        pred = s * (a @ W.T)
        rel = np.linalg.norm(pred - out, axis=1) / np.linalg.norm(out, axis=1)
        # 対照: ニューロン添字をずらすと一致しないこと(テストが添字の違いを見分けられる)
        rel_shift = np.linalg.norm(s * (np.roll(a, 1, axis=1) @ W.T) - out, axis=1) / np.linalg.norm(out, axis=1)
        tol_med, tol_max = V2_TOL.get(self.w.meta.ffn_down_qtype, V2_TOL_DEFAULT)
        med, mx = float(np.median(rel)), float(rel.max())
        ok = med < tol_med and mx < tol_max and float(np.median(rel_shift)) > 0.5
        return VResult("V2", "FFN 线性性（W_down 的方向・下标，down 之后无变换）", "pass" if ok else "fail",
                       f"L{L} `{act}` → W_down@a vs `ffn_out-{L}`：相对误差 中位数 {med:.4f} / 最大 {mx:.4f}"
                       f"（允许 {tol_med}/{tol_max}，{len(rel)} 个 token）。将下标错开 1 后中位数为 {np.median(rel_shift):.2f}",
                       {"act_name": act, "median": med, "max": mx, "shift_median": float(np.median(rel_shift)),
                        "n_tokens": int(len(rel))})

    # --- V3 -------------------------------------------------------------------------------------

    def v3_cvec_layer(self) -> VResult:
        L, n_embd = self.L, self.e.n_embd
        ids = self.prompt_ids[0]
        v = self.rng.standard_normal(n_embd).astype(np.float32)
        v *= 5.0 * self.rho() / np.linalg.norm(v)
        names = {f"l_out-{L - 1}", f"l_out-{L}", f"l_out-{L + 1}", f"l_out-{L + 2}"}
        base = self.capture(names, [ids], None)
        st = self.capture(names, [ids], {L: v})
        again = self.capture(names, [ids], None)          # 解除で元に戻るか
        d_prev = st[f"l_out-{L - 1}"] - base[f"l_out-{L - 1}"]
        d_L = st[f"l_out-{L}"] - base[f"l_out-{L}"]
        d_next = st[f"l_out-{L + 1}"] - base[f"l_out-{L + 1}"]
        err_L = float((np.linalg.norm(d_L - v, axis=1) / np.linalg.norm(v)).max())
        restore = float(max(np.abs(again[k] - base[k]).max() for k in names))
        # §5.3: 範囲を広げても前回の値が復活しない(常に全層ゼロ埋めバッファを渡しているので)
        v2 = self.rng.standard_normal(n_embd).astype(np.float32)
        v2 *= self.rho() / np.linalg.norm(v2)
        self.capture(names, [ids], {L: v})
        widened = self.capture(names, [ids], {L + 2: v2}, layer_range=(L, L + 2))
        stale = float(np.abs(widened[f"l_out-{L}"] - base[f"l_out-{L}"]).max())
        ok = (np.abs(d_prev).max() == 0 and err_L < 1e-3 and np.abs(d_next).max() > 0
              and restore == 0 and stale == 0)
        return VResult("V3", "cvec 层对应", "pass" if ok else "fail",
                       f"set_cvec({{{L}: v}}), ‖v‖=5ρ={np.linalg.norm(v):.1f}：Δl_out-{L - 1} 最大 {np.abs(d_prev).max():.3g}、"
                       f"Δl_out-{L} 与 v 的相对误差 最大 {err_L:.2e}（{d_L.shape[0]} 个位置）、Δl_out-{L + 1} 最大 {np.abs(d_next).max():.3g}、"
                       f"解除后的差异 {restore:.3g}、将范围扩展到 L..L+2 时 l_out-{L} 的差异 {stale:.3g}",
                       {"d_prev_max": float(np.abs(d_prev).max()), "err_L": err_L,
                        "d_next_max": float(np.abs(d_next).max()), "restore": restore, "stale": stale})

    # --- V11(§6.16 gain)----------------------------------------------------------------------------------

    def v11_gain(self, ks=(0.0, 2.0, -1.0), bake: bool = True) -> VResult:
        """gain 等价性：将第 L 层神经元 n 放大 k 倍时，ffn_down 的输入（激活值）第 n 项应变为原始激活值 × k，
        其他神经元不变。分别检查 (a) 应用 LoRA 适配器，(b) 改写 W_up 的 GGUF。"""
        from .gain import bake_gain_gguf, export_gain_lora
        from .steer import Recipe, Unit
        L = self.L
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        ids = self.prompt_ids[0]
        names = self.e.list_names(ids, L)
        act = pick_act_name(names, L)
        if act is None:
            return VResult("V11", "gain 等价性", "skip", f"无法捕获第 {L} 层的 ffn_down 输入")

        def grab(engine, lora=None):
            cap = Capture()
            cap.set_wanted(lambda x: x == act)
            ctx = engine.probe_ctx(cap)
            engine.clear_memory(ctx)
            engine.set_lora(ctx, lora)
            engine.prefill(ctx, [ids])
            engine.set_lora(ctx, None)
            return cap.take()[act]
        base = grab(self.e)
        scale = float(np.abs(base[:, n]).max())
        rows, ok = [], True
        with tempfile.TemporaryDirectory() as td:
            for k in ks:
                r = Recipe([Unit(L, n, 0.0, mode="gain", k=k)], self.w.meta.model_id, name="v11")
                variants = [("LoRA", None)]
                lpath = Path(td) / f"k{k:g}.lora.gguf"
                export_gain_lora(self.w, r, lpath)
                variants[0] = ("LoRA", grab(self.e, [(lpath, 1.0)]))
                if bake:
                    gpath = Path(td) / f"k{k:g}.gguf"
                    bake_gain_gguf(self.w, r, gpath)
                    e2 = Engine(gpath, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=1)
                    try:
                        variants.append(("改写", grab(e2)))
                    finally:
                        e2.close()
                for kind, a in variants:
                    others = np.delete(np.arange(base.shape[1]), n)
                    d_other = float(np.abs(a[:, others] - base[:, others]).max())
                    if k == 0:
                        # 設計の目安は絶対値 < 1e-3 だが、LoRA 版は W x(量子化)と A x(dequantize 値)の差が残る。
                        # 活性の大きさはモデルで違うので、素の活性の最大値に対する比で見る(divergence.md)
                        err = float(np.abs(a[:, n]).max())
                        good = err < 0.03 * max(scale, 1e-6)
                    else:
                        rel = np.abs(a[:, n] - k * base[:, n]) / np.maximum(np.abs(k * base[:, n]), 1e-6)
                        err = float(np.median(rel))
                        good = err < 0.03
                    good = good and d_other <= 1e-3 * max(1.0, float(np.abs(base).max()))
                    ok &= good
                    rows.append({"k": k, "kind": kind, "err": err, "others_max_diff": d_other, "pass": bool(good)})
        detail = (f"L{L} N{n}（`{act}`、{len(ids)} 个 token、|a| 最大 {scale:.3g}）："
                  + "; ".join(f"k={r['k']:g} {r['kind']}: " + (f"|a| 最大 {r['err']:.2e}（原始最大值的 {r['err'] / max(scale, 1e-6):.1%}）" if r["k"] == 0 else f"相对误差 中位数 {r['err']:.2e}")
                              + f"、其他神经元差异 最大 {r['others_max_diff']:.2e}" for r in rows))
        return VResult("V11", "gain 等价性（LoRA / 改写）", "pass" if ok else "fail", detail,
                       {"layer": L, "neuron": n, "act": act, "rows": rows, "act_max": scale})

    # --- V13(§6.17 方向の射影消去、LoRA 版)--------------------------------------------------------------

    def _resid_all(self, engine: Engine, ids: list[int], lora=None) -> dict[int, np.ndarray]:
        """所有层（不含最后一层）的 l_out-L（不含位置 0 = attention sink）。"""
        Ls = list(range(engine.n_layer - 1))                    # 最終層は行が間引かれ得るので除く
        names = {f"l_out-{L}": L for L in Ls}
        cap = Capture()
        cap.set_wanted(lambda n: n in names)
        ctx = engine.probe_ctx(cap)
        engine.clear_memory(ctx)
        engine.set_cvec(ctx, None)
        engine.set_lora(ctx, lora)
        try:
            engine.prefill(ctx, [ids])
        finally:
            engine.set_lora(ctx, None)
        return {names[k]: v.astype(np.float64)[1:] for k, v in cap.take().items()}

    def v13_ablation(self, r: np.ndarray | None = None, *, bake: bool = False) -> VResult:
        """从所有层的残差写入器中消除方向 r 后，l_out-L 的 r 分量是否与“应该残留的部分”一致。
        - LoRA：可将嵌入加入 LoRA 的模型（输出层独立）为 0；tied 模型保留嵌入的 r 分量
          （LoRA 无法改变嵌入，因此既不增加也不减少）。
        - bake（bake=True、F16）：嵌入也会投影，因此为 0。
        检查 |r·l_out − 期望值| / ‖l_out‖ 是否处于量化误差水平（< 1e-2）。同时列出消除前的偏差。"""
        from .ablate import bake_ablation_gguf, embed_in_lora, export_ablation_lora, unit
        r = unit(self.rng.standard_normal(self.w.meta.n_embd) if r is None else r)
        ids = self.prompt_ids[0]
        er = self.w.embed_rows(ids)[1:].astype(np.float64) @ r    # 埋め込みの r 成分(inp_embd は捕捉できない)
        base = self._resid_all(self.e, ids)
        variants = []
        with tempfile.TemporaryDirectory() as td:
            lp = Path(td) / "v13.lora.gguf"
            rep = export_ablation_lora(self.w, r, lp)
            variants.append(("LoRA", self._resid_all(self.e, ids, [(lp, 1.0)]), 0.0 if rep["embed"] else er, rep["n_tensors"]))
            if bake:
                bp = Path(td) / "v13-bake.gguf"
                brep = bake_ablation_gguf(self.w, r, bp, requant="f16")
                e2 = Engine(bp, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=self.e.n_seq_max)
                try:
                    variants.append(("bake", self._resid_all(e2, ids), 0.0, brep["n_modified"]))
                finally:
                    e2.close()
        out, ok = [], True
        for kind, res, expect, n_t in variants:
            rows = []
            for L, la in sorted(res.items()):
                lb = base[L]
                dev = np.abs(la @ r - expect) / np.linalg.norm(la, axis=1)
                bdev = np.abs(lb @ r - expect) / np.linalg.norm(lb, axis=1)
                perp_a, perp_b = la - np.outer(la @ r, r), lb - np.outer(lb @ r, r)
                ch = np.linalg.norm(perp_a - perp_b, axis=1) / np.linalg.norm(perp_b, axis=1)
                rows.append({"layer": L, "dev_max": float(dev.max()), "base_dev_med": float(np.median(bdev)),
                             "perp_change_med": float(np.median(ch))})
            mx = max(x["dev_max"] for x in rows)
            ok &= mx < 1e-2
            out.append({"kind": kind, "expect": "0" if np.isscalar(expect) else "e·r", "n_tensors": n_t, "max_dev": mx,
                        "base_med_dev": float(np.median([x["base_dev_med"] for x in rows])), "rows": rows})
        detail = "; ".join(f"{v['kind']}（{v['n_tensors']} 个张量、r 分量期望值 {v['expect']}）：最大 {v['max_dev']:.2e}"
                           f"（消除前中位数 {v['base_med_dev']:.2e}）、与 r 正交分量的变化第 0 层为 "
                           f"{v['rows'][0]['perp_change_med']:.3f}・末层为 {v['rows'][-1]['perp_change_med']:.3f}" for v in out)
        lora = out[0]
        return VResult("V13", "方向投影消除", "pass" if ok else "fail",
                       f"随机方向（{len(ids)} 个 token、第 0～{lora['rows'][-1]['layer']} 层、将嵌入"
                       f"{'加入' if embed_in_lora(self.w) else '不加入（tied）'} LoRA）：" + detail,
                       {"variants": out, "max_dev": max(v["max_dev"] for v in out), "base_med_dev": lora["base_med_dev"],
                        "n_tensors": lora["n_tensors"], "rows": lora["rows"]})

    def v14_lora_vs_bake(self, r: np.ndarray | None = None, n_tokens: int = 32) -> VResult:
        """用消除同一方向的 LoRA 与 bake（F16），检查 greedy 的前 n_tokens 个 token 是否一致。嵌入在两者中保持一致
        （tied 模型无法将嵌入加入 LoRA，因此 bake 也省略嵌入）。同时列出 q8_0 bake 作参考。"""
        from .ablate import bake_ablation_gguf, embed_in_lora, export_ablation_lora, unit
        r = unit(self.rng.standard_normal(self.w.meta.n_embd) if r is None else r)
        r = (r * 1.0).astype(np.float32)
        embed = embed_in_lora(self.w)
        seqs = self.prompt_ids[:2]

        def run(engine, lora=None):
            ctx = engine.steer_ctx()
            engine.set_cvec(ctx, None)
            engine.set_lora(ctx, lora)
            try:
                return [engine.generate_greedy(ctx, [s], max_tokens=n_tokens)[0] for s in seqs]
            finally:
                engine.set_lora(ctx, None)
        base = [x.token_ids for x in run(self.e)]
        rows, ok = [], True
        with tempfile.TemporaryDirectory() as td:
            lp = Path(td) / "v14.lora.gguf"
            export_ablation_lora(self.w, r, lp, embed=embed)
            lora = run(self.e, [(lp, 1.0)])
            for requant in ("f16", "q8_0"):
                bp = Path(td) / f"v14-{requant}.gguf"
                bake_ablation_gguf(self.w, r, bp, requant=requant, embed=embed)
                e2 = Engine(bp, n_gpu_layers=getattr(self.e, "n_gpu_layers", -1), n_ctx=self.e.n_ctx, n_seq_max=self.e.n_seq_max)
                try:
                    baked = run(e2)
                finally:
                    e2.close()
                for i, (a, b) in enumerate(zip(lora, baked, strict=True)):
                    # 完全一致か、最初の分岐が僅差(相手の選択が自分の最良から 0.5 nats 以内)なら合格(V9 と同じ基準)
                    cmp = compare_greedy(a.token_ids, a.top_ids, a.top_lps, b.token_ids, b.top_ids, b.top_lps)
                    rows.append({"requant": requant, "prompt": i, "match": cmp["first_diff"] is None, "ok": bool(cmp["ok"]),
                                 "first_diff": cmp["first_diff"], "summary": cmp["summary"],
                                 "lora_differs_from_base": a.token_ids != base[i]})
                    if requant == "f16":
                        ok &= bool(cmp["ok"])
        detail = ", ".join(f"{x['requant']} 提示词{x['prompt'] + 1}: " + ("一致" if x["match"] else x["summary"]
                                                                            + ("" if x["ok"] else "，并非微小差异"))
                           for x in rows)
        return VResult("V14", "投影消除的 LoRA 与 bake 一致性", "pass" if ok else "fail",
                       f"greedy {n_tokens} 个 token、嵌入{'有' if embed else '无'}：{detail}"
                       f"（与无干预的差异：{'有' if any(x['lora_differs_from_base'] for x in rows) else '无'}）",
                       {"rows": rows, "embed": embed})

    # --- V12(§6.15 位置別ステアリング)----------------------------------------------------------------

    def v12_when_gen(self) -> VResult:
        """when=gen：prefill 保持原样（l_out-L 与原始值完全一致），从生成第 1 步开始加入 v。
        还需检查 gne 的 generate_when(gen) 首个 token 与无干预相同，而整体既不同于无干预，也不同于 all。"""
        from .when import generate_when
        L, n_embd = self.L, self.e.n_embd
        ids = self.prompt_ids[0]
        v = self.rng.standard_normal(n_embd).astype(np.float32)
        v *= 5.0 * self.rho() / np.linalg.norm(v)
        name = f"l_out-{L}"
        cap = Capture()
        cap.set_wanted(lambda n: n == name)
        ctx = self.e.probe_ctx(cap)

        def run(gen_vec):
            self.e.clear_memory(ctx)
            self.e.set_cvec(ctx, None)
            pre = self.e.prefill(ctx, [ids])
            a = cap.take()[name]
            tok = int(pre.logits[0].argmax())
            self.e.set_cvec(ctx, gen_vec)
            self.e._decode(ctx, [(tok, len(ids), 0, True)])
            b = cap.take()[name]
            self.e.set_cvec(ctx, None)
            return a, b
        a0, b0 = run(None)
        a1, b1 = run({L: v})
        prefill_same = float(np.abs(a1 - a0).max())
        err_gen = float(np.linalg.norm((b1 - b0)[-1] - v) / np.linalg.norm(v))
        # 実際の経路: generate_when の gen / all / 無介入
        msgs = [{"role": "user", "content": PROMPTS[0]}]
        sctx = self.e.steer_ctx()
        g_base, _ = generate_when(self.e, sctx, msgs, None, "all", max_tokens=24)
        g_gen, info = generate_when(self.e, sctx, msgs, {L: v}, "gen", max_tokens=24)
        g_all, _ = generate_when(self.e, sctx, msgs, {L: v}, "all", max_tokens=24)
        first_same = bool(g_gen.token_ids[:1] == g_base.token_ids[:1])
        ok = prefill_same == 0 and err_gen < 1e-3 and first_same and info["set_cvec"] == 2
        return VResult("V12", "位置别 steering（when=gen）", "pass" if ok else "fail",
                       f"l_out-{L}：prefill 差异最大 {prefill_same:.3g}（应为 0）、生成第 1 步差异与 v 的相对误差 {err_gen:.2e}。"
                       f"generate_when(gen)：首个 token 与无干预{'相同' if first_same else '不同'}、set_cvec {info['set_cvec']} 次、"
                       f"与无干预{'相同' if g_gen.token_ids == g_base.token_ids else '不同'}、与 all {'相同' if g_gen.token_ids == g_all.token_ids else '不同'}",
                       {"prefill_diff_max": prefill_same, "err_gen": err_gen, "first_token_same": first_same,
                        "set_cvec": info["set_cvec"], "gen_eq_base": g_gen.token_ids == g_base.token_ids,
                        "gen_eq_all": g_gen.token_ids == g_all.token_ids})

    # --- V4 -------------------------------------------------------------------------------------

    def v4_direction_file(self, llama_completion: str | None = None, n_tokens: int = 32) -> VResult:
        L, n_layer, n_embd = self.L, self.e.n_layer, self.e.n_embd
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        v = (0.8 * self.rho() * self.w.unit_direction(L, n)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v4.cvec.gguf"
            export_cvec_gguf({L: v}, path, model_hint=self.w.meta.arch, n_layer=n_layer)
            n_embd_f, data = load_cvec_like_common([(path, 1.0)])
            buf, _, _ = cvec_buffer(n_layer, n_embd, {L: v})
            same_buf = n_embd_f == n_embd and data.size == buf.size and np.array_equal(data, buf)
            # ローダの規約で読み戻したバッファで V3 と同じ捕捉
            names = {f"l_out-{L - 1}", f"l_out-{L}"}
            ids = self.prompt_ids[0]
            a = self.capture(names, [ids], {L: v})
            back = {i + 1: data[i * n_embd:(i + 1) * n_embd] for i in range(n_layer - 1)}
            b = self.capture(names, [ids], back, layer_range=(1, n_layer - 1))
            same_cap = all(np.array_equal(a[k], b[k]) for k in names)
            detail = (f"将 0.8ρ·u_{n} 写入 direction.{L} → 按 common 规则读回的缓冲区一致：{same_buf}、"
                      f"用全层范围对其执行 set_cvec 后的捕获一致：{same_cap}")
            data_out = {"neuron": n, "same_buf": bool(same_buf), "same_capture": bool(same_cap)}
            ok = same_buf and same_cap
            if llama_completion:
                prompt = self.e.render_chat([{"role": "user", "content": PROMPTS[0]}])
                ctx = self.e.steer_ctx()
                gen = {}
                for tag, vec in (("base", None), ("steer", {L: v})):
                    self.e.set_cvec(ctx, vec)
                    r = self.e.generate_greedy(ctx, [self.prompt_ids[0]], max_tokens=n_tokens)[0]
                    ours = self.e.detokenize(r.token_ids[:-1] if r.finish == "eog" else r.token_ids)
                    theirs = run_llama_completion(llama_completion, self.e, prompt, n_tokens,
                                                  path if tag == "steer" else None)
                    gen[tag] = {"ours": ours, "llama": theirs, "match": ours == theirs}
                self.e.set_cvec(ctx, None)
                differs = gen["steer"]["ours"] != gen["base"]["ours"]
                ok = ok and gen["base"]["match"] and gen["steer"]["match"] and differs
                detail += (f"。llama-completion --control-vector 的 greedy {n_tokens} 个 token：无干预 一致 {gen['base']['match']}、"
                           f"干预 一致 {gen['steer']['match']}（干预后输出变化：{differs}）")
                data_out["generation"] = gen
            else:
                detail += "。缺少 llama-completion，因此省略生成比较"
        return VResult("V4", "direction.N 对应（GGUF 导出 → 原生加载器）", "pass" if ok else "fail", detail, data_out)

    # --- V5 -------------------------------------------------------------------------------------

    def v5_neuron_push(self, hf_model: str | None, hf_python: str | None, n_tokens: int = 32) -> VResult:
        if not hf_model or not hf_python:
            return VResult("V5", "神经元推动等价性（HF 实现）", "skip", "缺少 --hf-model 与 --hf-python，因此省略（可选）")
        L = self.L
        n = int(self.rng.integers(self.w.meta.n_ff_at(L)))
        col = self.w.down_column(L, n)
        add = 0.8 * self.rho() / float(np.linalg.norm(col))     # rel dose 0.8 相当の活性への加算量
        ids = self.prompt_ids[0]
        eog = [t for t in range(self.e.n_vocab) if self.e.is_eog(t)]
        req = {"prompt_ids": ids, "layer": L, "neuron": n, "add": add, "max_tokens": n_tokens,
               "n_vocab": self.e.n_vocab, "eog_ids": eog, "dump_down": True}
        ref, hf_down = run_hf_reference(hf_python, hf_model, req)
        ctx = self.e.steer_ctx()
        ours = {}
        for tag, vec in (("base", None), ("steer", {L: add * col})):
            self.e.set_cvec(ctx, vec)
            self.e.clear_memory(ctx)
            pre = self.e.prefill(ctx, [ids])
            r = self.e.generate_greedy(ctx, [ids], max_tokens=n_tokens, prefilled=pre)[0]
            ours[tag] = {"ids": r.token_ids, "next_lp": log_softmax(pre.logits[0]), "top_ids": r.top_ids, "top_lps": r.top_lps}
        self.e.set_cvec(ctx, None)
        res, ok = {}, True
        for tag in ("base", "steer"):
            res[tag] = compare_greedy(ours[tag]["ids"], ours[tag]["top_ids"], ours[tag]["top_lps"],
                                      ref[tag]["ids"], ref[tag]["top_ids"], ref[tag]["top_lps"])
            ok = ok and res[tag]["ok"]
        # 介入による次トークン分布の変化 Δ(最終プロンプト位置)が HF と一致するか
        d_ours = ours["steer"]["next_lp"] - ours["base"]["next_lp"]
        d_hf = np.asarray(ref["steer"]["next_lp"], np.float32) - np.asarray(ref["base"]["next_lp"], np.float32)
        top = np.argsort(-np.maximum(ours["base"]["next_lp"], ours["steer"]["next_lp"]))[:200]
        corr = float(np.corrcoef(d_ours[top], d_hf[top])[0, 1])
        ok = ok and corr > 0.9
        detail = (f"L{L} N{n} add={add:.2f}：无干预 {res['base']['summary']} / 干预 {res['steer']['summary']}。"
                  f"干预导致下一 token log p 变化的相关性（前 200 词） {corr:.3f}")
        r4 = {}
        if hf_down is not None:
            r4 = neuron_order_check(self.w.down_matrix(L), hf_down, self.rng)
            detail += (f"。R4：HF down_proj 与 GGUF 列相关性的中位数 {r4['diag_median']:.4f}、"
                       f"抽样 {r4['n_sample']} 列中相关性最高 HF 列编号相同：{r4['argmax_match']:.3f}")
            ok = ok and r4["argmax_match"] == 1.0
        return VResult("V5", "神经元推动等价性（HF 实现）", "pass" if ok else "fail", detail,
                       {"neuron": n, "add": add, "corr_delta": corr, "r4": r4, **{k: v for k, v in res.items()}})

    # --- V6 -------------------------------------------------------------------------------------

    def v6_attention_sink(self) -> VResult:
        layers = sorted({1, self.L // 2, self.L, self.e.n_layer - 2})
        ratios = {}
        for L in layers:
            r = []
            for ids in self.prompt_ids:
                x = self.capture({f"l_out-{L}"}, [ids])[f"l_out-{L}"]
                nrm = np.linalg.norm(x, axis=1)
                r.append(nrm[0] / np.median(nrm[1:]))
            ratios[L] = float(np.median(r))
        ok = all(v >= 3 for v in ratios.values())
        return VResult("V6", "position 0 的残差范数（attention sink）", "info",
                       "‖l_out‖[pos0] / 中位数[pos≥1]：" + ", ".join(f"L{L} {v:.1f}×" for L, v in ratios.items())
                       + ("（全部层 ≥3×）" if ok else "（存在低于 3× 的层：警告）"), {"ratios": ratios, "all_ge3": ok})

    # --- V8 -------------------------------------------------------------------------------------

    def v8_tokenize(self) -> VResult:
        convs = [
            [{"role": "user", "content": PROMPTS[0]}],
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": PROMPTS[2]}],
            [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! 何かお手伝いできますか？"},
             {"role": "user", "content": "Tell me a joke."}],
        ]
        bad = []
        from llama_cpp import llama_cpp as C
        add_bos = bool(C.llama_vocab_get_add_bos(self.e.vocab))
        bos = C.llama_vocab_bos(self.e.vocab)
        for conv in convs:
            text = self.e.render_chat(conv)
            ids = self.e.tokenize_prompt(text)
            body = ids[1:] if add_bos and not text.startswith(self.e.token_text(bos)) else ids
            back = self.e.detokenize(body, special=True)
            if back != text:
                bad.append(f"往返不一致：{text[:40]!r} → {back[:40]!r}")
            if add_bos and ids[:2] == [bos, bos]:
                bad.append("BOS 重复")
            if add_bos and ids[0] != bos:
                bad.append("缺少 BOS")
        return VResult("V8", "tokenize 往返与 BOS", "fail" if bad else "pass",
                       f"{len(convs)} 个会话：模板渲染 → tokenize → detokenize 一致，add_bos={add_bos}"
                       + (f" / {bad}" if bad else ""), {"bad": bad, "add_bos": add_bos})

    # --- names ----------------------------------------------------------------------------------

    def list_names(self, L: int | None = None) -> set[str]:
        return self.e.list_names(self.prompt_ids[0], L)


# --- 外部プロセス -------------------------------------------------------------------------------

def run_llama_completion(binary: str, engine: Engine, prompt: str, n_tokens: int, cvec: Path | None,
                         lora: Path | None = None) -> str:
    """用 llama-completion 进行 greedy 生成的字符串（不显示提示词）。lora 是 gain 的 LoRA 适配器（§6.16）。"""
    cmd = [binary, "-m", str(engine.path), "-p", prompt, "-n", str(n_tokens), "-no-cnv", "--no-display-prompt",
           "--temp", "0", "--top-k", "1", "--seed", "0", "-c", str(engine.n_ctx), "-ngl", str(engine.n_gpu_layers),
           "--no-warmup", "-lv", "0"]
    if cvec is not None:
        cmd += ["--control-vector", str(cvec)]
    if lora is not None:
        cmd += ["--lora", str(lora)]
    env = dict(os.environ)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"llama-completion 执行失败：{p.stderr[-2000:]}")
    # completion.cpp は生成の後に必ず "\n\n" を、EOG で止まったときはその前に " [end of text]\n" を出す
    out = p.stdout.removesuffix("\n\n")
    return out.removesuffix(" [end of text]\n")


def run_hf_reference(python: str, model_dir: str, req: dict) -> dict:
    script = Path(__file__).resolve().parent.parent / "tools" / "hf_reference.py"
    with tempfile.TemporaryDirectory() as td:
        rq, rs = Path(td) / "req.json", Path(td) / "ref.json"
        rq.write_text(json.dumps(req))
        p = subprocess.run([python, str(script), "--model", model_dir, "--request", str(rq), "--out", str(rs)],
                           capture_output=True, text=True, timeout=1200)
        if p.returncode != 0:
            raise RuntimeError(f"hf_reference.py 执行失败：{p.stderr[-3000:]}")
        down = Path(td) / f"down_L{req['layer']}.npy"
        return json.loads(rs.read_text()), (np.load(down) if down.exists() else None)


def neuron_order_check(gguf_down: np.ndarray, hf_down: np.ndarray, rng, n_sample: int = 256) -> dict:
    """R4：GGUF 的 W_down[:, n] 与 HF 的 down_proj.weight[:, n] 是否为同一神经元（逐列相关性）。"""
    if gguf_down.shape != hf_down.shape:
        return {"diag_median": float("nan"), "argmax_match": 0.0, "n_sample": 0,
                "error": f"shape {gguf_down.shape} vs {hf_down.shape}"}
    g = gguf_down - gguf_down.mean(axis=0)
    h = hf_down - hf_down.mean(axis=0)
    g /= np.linalg.norm(g, axis=0)
    h /= np.linalg.norm(h, axis=0)
    diag = (g * h).sum(axis=0)
    cols = rng.choice(g.shape[1], size=min(n_sample, g.shape[1]), replace=False)
    best = np.argmax(g[:, cols].T @ h, axis=1)
    return {"diag_median": float(np.median(diag)), "diag_min": float(diag.min()),
            "argmax_match": float(np.mean(best == cols)), "n_sample": int(len(cols))}


def compare_greedy(a_ids, a_top, a_lps, b_ids, b_top, b_lps, near: float = 0.5) -> dict:
    """比较 greedy 序列。若不一致，在首次分叉处双方都满足“对方选择的 token 距离自己的最佳项不超过 near nats”
    （微小差异，可由数值误差解释）时判为 ok。"""
    n = min(len(a_ids), len(b_ids))
    first = next((i for i in range(n) if a_ids[i] != b_ids[i]), None)
    if first is None:
        same_len = len(a_ids) == len(b_ids)
        return {"ok": True, "first_diff": None, "summary": f"前 {n} 个 token 一致" + ("" if same_len else "（长度不同）")}

    def gap(top, lps, other_tok):
        top = list(top[first])
        return lps[first][0] - lps[first][top.index(other_tok)] if other_tok in top else float("inf")

    gap_a, gap_b = gap(a_top, a_lps, b_ids[first]), gap(b_top, b_lps, a_ids[first])
    ok = gap_a < near and gap_b < near
    return {"ok": ok, "first_diff": first, "gap_ours": gap_a, "gap_ref": gap_b,
            "summary": f"第 {first + 1} 个 token 处分叉（与对方选择的差异 ours {gap_a:.2f} / ref {gap_b:.2f} nats）"}


def find_llama_completion() -> str | None:
    here = Path(__file__).resolve().parent.parent / "build" / "llama.cpp" / "build" / "bin" / "llama-completion"
    if here.exists():
        return str(here)
    return shutil.which("llama-completion")

"""Python API。设计文档 §6.13 (c)。CLI 是它的薄封装。

    from gne.api import open_model
    with open_model("models/Qwen3.5-4B-Q8_0.gguf") as s:
        out = s.steer(["14:8949:+1.0", "13:7956:-1.0"], prompt="Who are you?")

返回值是与 --json 相同形式的 dict（不含 envelope）。envelope 由 envelope() 创建。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .engine import NO_CVEC, Engine
from .schemas import SCHEMA_VERSION
from .stats import (ActStats, ResidStats, StatsView, load_corpus, neuron_activation_stats,
                    residual_norms)
from .steer import (GneError, Recipe, bake_bias_gguf, bias_support, export_vectors, import_cvec_gguf,
                    load_cvec_like_common, parse_unit, recipe_vectors, unit_details, Unit)
from .store import CONFIGS, Store
from .weights import Weights

Progress = Callable[[int, int, str], None]


def open_model(path: str | Path, **kw) -> Session:
    return Session(path, **kw)


MIN_ROUTED = 200            # MoE: 専門家の観測トークン数がこれより少ないと分位点を信頼できない(CHANGE-05 §B.1)
# 焼き込みの確かめ(V26・V33)で、計算方法の違いだけで出る射影の値の誤差を測るときの LoRA の倍率(書き込みはほぼ 0)。同じ LoRA
# ファイルをこの倍率で当てる(keyed の LoRA は α を最大の |α| で割って書くので、いちばん大きいユニットが α = 1e-6 になる)。
# 0 ちょうどにすると llama.cpp が LoRA を外して計算の仕方が LoRA 版と変わるおそれがあるので、ごく小さい値にする(書き込みの
# 効きは 0.6B の最後の層で 1e-4 未満。測りたい誤差は 0.3〜1.9 だった、verification.md F139)。
NOISE_LORA_SCALE = 1e-6


class Session:
    def __init__(self, path: str | Path, *, root: str | Path | None = None, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, n_seq_max: int = 8, seed: int = 0, verbose: bool = False,
                 progress: Progress | None = None, mmproj: str | Path | None = None):
        self.path = Path(path)
        if not self.path.exists():
            raise GneError("MODEL_NOT_FOUND", f"{self.path} 不存在")
        self.weights = Weights(self.path)
        self.meta = self.weights.meta
        self.store = Store(root)
        self.store.save_meta(self.meta)
        self.seed = seed
        self.progress = progress
        self.warnings: list[str] = []
        self._engine_kw = dict(n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_seq_max=n_seq_max, verbose=verbose,
                               mmproj=mmproj)       # mmproj: 音声の入力(media.py)。無ければモデルの横の mmproj*.gguf
        self._engine: Engine | None = None
        self._resid: ResidStats | None = None
        self.last_patch_reload_s: float | None = None   # パッチを当てたコピーの読み直しにかかった秒(CHANGE-05)

    # --- lifecycle ------------------------------------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._make_engine(self.path)
        return self._engine

    def _make_engine(self, path) -> Engine:
        e = Engine(path, **self._engine_kw)
        tpl = template_override(self.meta.model_id, self.meta.arch)
        if tpl is not None:
            e.chat_template = tpl[1]
            self.warn(f"已使用 {tpl[0]} 覆盖聊天模板")
        return e

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def envelope(self, command: str) -> dict:
        return {"schema_version": SCHEMA_VERSION, "command": command, "model_id": self.meta.model_id,
                "llama_cpp_version": Engine.llama_cpp_version(), "seed": self.seed,
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "warnings": list(self.warnings)}

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    # --- info -----------------------------------------------------------------------------------

    def info(self, *, load: bool = True) -> dict:
        m = self.meta
        meta = {"path": str(m.path), "model_id": m.model_id, "name": m.name, "arch": m.arch, "n_layer": m.n_layer,
                "n_embd": m.n_embd, "n_ff": m.n_ff, "n_ff_per_layer": m.n_ff_per_layer, "n_vocab": m.n_vocab,
                "n_expert": m.n_expert, "has_post_ffn_norm": m.has_post_ffn_norm,
                "ffn_down_scale": {str(k): v for k, v in m.ffn_down_scale.items()} if m.ffn_down_scale else None,
                "tied_output": m.tied_output, "has_chat_template": bool(m.chat_template),
                "ffn_down_qtype": m.ffn_down_qtype, "file_size": m.file_size,
                "cvec_supported": m.arch not in NO_CVEC, "is_split": self.weights.gguf.is_split, "hc_streams": int(self.weights.field(f"{m.arch}.hyper_connection.count", 0) or 0)}
        llama = None
        if load:
            try:
                e = self.engine
                llama = {"ok": True, "n_layer": e.n_layer, "n_embd": e.n_embd, "n_vocab": e.n_vocab}
                if e.n_layer != m.n_layer:
                    self.warn(f"llama.cpp 的 n_layer {e.n_layer} 与 block_count {m.n_layer} 不一致（例如 MTP 层）")
            except Exception as ex:  # noqa: BLE001
                llama = {"ok": False, "error": str(ex)}
        mode, reason, checked = bias_support(m.arch)
        rp = self.store.resid_path(m.model_id)
        acts = sorted(int(p.stem.split("_L")[1]) for p in self.store.stats_dir(m.model_id).glob("act_L*.npz"))
        from .ablate import writer_info
        from .estimate import Speed, is_large, recommended_workflow
        sp = Speed(self)
        speed = {**sp.to_dict(), "cpu_layers": m.n_layer - sp.gpu_layers,
                 "decode_tok_s": 1000.0 / sp.step1, "large": is_large(self)}
        return {"meta": meta, "arch_warnings": self.weights.check_arch(), "llama_cpp": llama,
                "bake": {"supported": mode is not None, "mode": mode, "reason": reason, "checked_llama_cpp": checked},
                "stats": {"resid": str(rp) if rp.exists() else None, "act_layers": acts},
                "residual_writers": writer_info(self.weights), "speed": speed,
                "workflow": recommended_workflow(self) if speed["large"] else None}

    # --- stats ----------------------------------------------------------------------------------

    def stats_resid(self, *, corpus: str | Path | None = None, tokens: int | None = None,
                    layers: list[int] | None = None, include_last_layer: bool = False,
                    chunk_tokens: int = 512) -> dict:
        c = load_corpus(corpus)
        t0 = time.perf_counter()
        st = residual_norms(self.engine, c, layers, chunk_tokens=chunk_tokens, include_last_layer=include_last_layer,
                            max_tokens=tokens, progress=self.progress)
        path = self.store.resid_path(self.meta.model_id)
        if path.exists() and layers is not None:
            st = _merge_resid(ResidStats.load(path), st)
        st.save(path)
        self._resid = st
        return {"path": str(path), "stats": st.to_dict(), "corpus": {"path": c.path, "sha": c.sha},
                "seconds": time.perf_counter() - t0}

    def stats_act(self, *, corpus: str | Path | None = None, tokens: int | None = None,
                  layers: list[int] | None = None, include_last_layer: bool = False,
                  chunk_tokens: int = 512) -> dict:
        c = load_corpus(corpus)
        t0 = time.perf_counter()
        res = neuron_activation_stats(self.engine, c, layers, chunk_tokens=chunk_tokens,
                                      include_last_layer=include_last_layer, max_tokens=tokens, progress=self.progress)
        rows = []
        for L, a in sorted(res.items()):
            p = self.store.act_path(self.meta.model_id, L)
            a.save(p)
            row = {"layer": L, "tensor_name": a.tensor_name, "path": str(p), "n_tokens": a.n_tokens,
                   "absmax_median": float(np.median(a.absmax)), "q999_median": float(np.median(a.q999)),
                   "rms_median": float(np.median(a.rms))}
            if a.is_moe:                         # CHANGE-05 §B.1: 観測数の少ない専門家は分位点を信頼できない
                row["moe"] = {"n_expert": int(len(a.n_routed)), "n_routed_min": int(a.n_routed.min()),
                              "n_routed_median": float(np.median(a.n_routed)),
                              "rare": [int(e) for e in np.flatnonzero(a.n_routed < MIN_ROUTED)]}
            rows.append(row)
        return {"layers": rows, "corpus": {"path": c.path, "sha": c.sha}, "seconds": time.perf_counter() - t0}

    def resid_stats(self) -> ResidStats | None:
        if self._resid is None:
            p = self.store.resid_path(self.meta.model_id)
            if p.exists():
                self._resid = ResidStats.load(p)
        return self._resid

    def stats_view(self, recipe: Recipe | None = None, *, auto: bool = True) -> StatsView:
        """准备配方所需的统计。若 auto 为真，则用默认语料计算缺少的统计（不进行交互确认）。"""
        need_rel = recipe is None or recipe.dose_unit == "rel" or any(u.kind == "random" for u in recipe.units)
        resid = self.resid_stats()
        if recipe is not None and need_rel:
            missing = [u.layer for u in recipe.units if resid is None or u.layer not in resid.layers]
            if missing:
                if not auto:
                    raise GneError("STATS_MISSING", f"第 {sorted(set(missing))} 层缺少残差范数统计",
                                   "请先运行 gne stats resid")
                self.warn("缺少残差范数统计，已用默认语料计算（gne stats resid）")
                last = self.meta.n_layer - 1 in missing
                if resid is None:
                    self.stats_resid()
                    if last:
                        self.stats_resid(layers=[self.meta.n_layer - 1], include_last_layer=True)
                else:
                    self.stats_resid(layers=sorted(set(missing)), include_last_layer=last)
                resid = self.resid_stats()
        if recipe is not None and recipe.dose_unit == "absmax":
            missing = sorted({u.layer for u in recipe.units
                              if not self.store.act_path(self.meta.model_id, u.layer).exists()})
            if missing:
                if not auto:
                    raise GneError("STATS_MISSING", f"第 {missing} 层缺少激活值统计", "请先运行 gne stats act")
                self.warn(f"第 {missing} 层缺少激活值统计，已用默认语料计算（gne stats act）")
                self.stats_act(layers=missing, include_last_layer=self.meta.n_layer - 1 in missing)

        def act_loader(L: int) -> ActStats | None:
            p = self.store.act_path(self.meta.model_id, L)
            return ActStats.load(p) if p.exists() else None

        return StatsView(resid, act_loader)

    # --- recipe ---------------------------------------------------------------------------------

    def recipe(self, *, units: list[str] | None = None, path: str | Path | None = None,
               dose_unit: str | None = None, scale: float = 1.0, name: str | None = None,
               when: str | None = None, ablate: str | dict | None = None, default_direction: str | None = None,
               direction_layer: str | None = None, default_key: str | None = None) -> Recipe:
        """ablate 表示方向的投影消除（§6.17）。格式为 "NAME[@层][:倍率]" 或 {direction, scale, layers, from_layer}。
        也可以创建仅含 ablate（不含单元）的配方。default_key は #KEY を省いた keyed のユニットの鍵(CHANGE-07)。
        keyed のユニットは方向と鍵があるか・層が鍵にあるか・長さ・書ける層か(MTP・最終層は断る)をここで確かめる。"""
        from .steer import AblateSpec
        if units and path:
            raise GneError("BAD_ARGS", "请指定 --recipe 或 --unit 中的恰好一个")
        if not units and not path and not ablate:
            raise GneError("BAD_ARGS", "请指定 --recipe 或 --unit（或 --ablate）")
        if not units and not path:
            r = Recipe([], self.meta.model_id, dose_unit or "rel", name)
        elif path:
            if not Path(path).exists():
                raise GneError("FILE_NOT_FOUND", f"配方 {path} 不存在")
            r = Recipe.load(path)
            if dose_unit:
                r.dose_unit = dose_unit
        else:
            r = Recipe([parse_unit(u) for u in units], self.meta.model_id, dose_unit or "rel", name)
        if when:
            r.when = when
        if ablate:
            r.ablate = AblateSpec.from_dict(ablate)
        if default_direction and not r.default_direction:      # CHANGE-06: @NAME を省いた proj のユニットの方向
            r.default_direction = default_direction
        if default_key and not r.default_key:                  # CHANGE-07: #KEY を省いた keyed のユニットの鍵
            r.default_key = default_key
        if direction_layer:
            r.direction_layer = direction_layer
        r.validate()
        if r.ablate is not None:
            self.direction_path(r.ablate.direction)              # 方向が無ければここで返す
        if r.model_id and r.model_id != self.meta.model_id:
            self.warn(f"配方的 model_id {r.model_id!r} 与已加载模型 {self.meta.model_id!r} 不一致")
        for u in r.units:
            if not 0 <= u.layer < self.meta.n_layer:
                raise GneError("BAD_RECIPE", f"第 {u.layer} 层超出范围（0..{self.meta.n_layer - 1}）")
            if u.mode == "keyed":                                   # CHANGE-07: まとめて _check_keyed で確かめる
                continue
            if u.kind == "neuron" and self.weights.is_moe(u.layer):
                raise GneError("BAD_RECIPE", f"第 {u.layer} 层是 MoE。专家神经元请按 {u.layer}:E<专家>:N{u.neuron} 的形式指定",
                               "shared expert 的神经元请用 L:S:Nn")
            if u.kind == "neuron" and not 0 <= u.neuron < self.meta.n_ff_at(u.layer):
                raise GneError("BAD_RECIPE", f"L{u.layer} 的神经元 {u.neuron} 超出范围（0..{self.meta.n_ff_at(u.layer) - 1}）")
            if u.kind in ("expert_neuron", "expert"):
                li = self.weights.moe_layer(u.layer)                    # MoE でない層ならここで返す
                if not 0 <= u.expert < li.n_expert:
                    raise GneError("BAD_RECIPE", f"L{u.layer} 的专家 {u.expert} 超出范围（0..{li.n_expert - 1}）")
                if u.kind == "expert_neuron" and not 0 <= u.neuron < li.n_ff:
                    raise GneError("BAD_RECIPE", f"L{u.layer} E{u.expert} 的神经元 {u.neuron} 超出范围（0..{li.n_ff - 1}）")
                if u.mode == "route" and not li.has_router_bias:
                    raise GneError("ARCH_UNSUPPORTED", f"第 {u.layer} 层没有路由器偏置（{self.meta.arch}）",
                                   "route 仅适用于带路由器偏置（exp_probs_b）的 arch（lfm2moe、deepseek2 等）")
            if u.kind == "shared":
                li = self.weights.arch.layers[u.layer]
                if not li.n_ff_shexp:
                    raise GneError("BAD_RECIPE", f"第 {u.layer} 层没有 shared expert")
                if not 0 <= u.neuron < li.n_ff_shexp:
                    raise GneError("BAD_RECIPE", f"L{u.layer} S 的神经元 {u.neuron} 超出范围（0..{li.n_ff_shexp - 1}）")
            if u.kind in ("expert_neuron", "shared") and u.mode == "add":
                self.warn(f"{u.label()}：MoE 的 add 是沿写入方向推动（无论是否选中专家，都会加到所有 token 上）。"
                          "仅在选中专家时生效的真正神经元推动是 gain(x)")
            if u.kind == "head":
                self.weights.head_block(u.layer, u.head)            # 層にヘッドが無い・範囲外ならここで返す
            if u.mode == "proj":                                    # CHANGE-06: 方向が無ければここで返す
                self.direction_set(u.direction or r.default_direction)
            if u.layer <= 2:
                self.warn(f"第 {u.layer} 层：第 0～2 层残差较小，rel dose 可能无法直接适用（R9）")
        if r.keyed_units:
            self._check_keyed(r)
        if r.when != "all":                                         # レビュー P3: LoRA は位置で切り替えられない
            lora_kinds = [k for k, us in (("keyed", r.keyed_units), ("gain", r.gain_units), ("proj", r.proj_units)) if us]
            if lora_kinds:
                self.warn(f"when={r.when} 对 {"、".join(lora_kinds)} 无效（LoRA 从生成开始到结束始终生效。能按位置切换的只有 add 的 control vector）")
        return r.scaled(scale) if scale != 1.0 else r

    def _warn_once(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warn(msg)

    def _key_of(self, name: str):
        """鍵の名前("NAME" か比較用の鍵 "NAME~j")→ KeySet(<root>/keys/<model_id>/、CHANGE-07)。"""
        from .keyed import load_key
        return load_key(self, name)

    def _check_keyed(self, r: Recipe) -> None:
        """keyed のユニット(CHANGE-07)の確かめ: 方向と鍵があるか、層が鍵にあるか、鍵の書き込み先と長さ、書ける層か
        (MTP・最終層は断る)、post-norm の一次近似(keyed_plan が警告)。重ね方の注意(レビュー P17): 同じ層の gain は
        鍵の入力 x を変える、同じ方向の ablate は焼き込みの順によっては keyed を消す。"""
        from .keyed import keyed_plan, keyed_units
        ku = keyed_units(r)
        keyed_plan(self.weights, r, self._key_of, self.direction_set, direction_layer=r.direction_layer, warn=self._warn_once)
        same_input = {"ffn": ("neuron",), "attn": ("head",), "shared": ("shared",), "expert": ("expert_neuron",)}
        for k in ku:
            hit = [g.label() for g in r.gain_units if g.layer == k.layer and g.kind in same_input[k.kind]
                   and (k.kind != "expert" or g.expert == k.expert)]
            if hit:
                self._warn_once(f"{k.spec()}: 同层的 gain({', '.join(hit)}) 会改变键读取的输入 x，因此 κᵀμ = 1 会偏移（键是在无干预时创建的）。若要叠加，请在应用 gain 的状态下重新测量 key-check")
        if r.ablate is not None:
            same = sorted({k.direction for k in ku if k.direction == r.ablate.direction})
            if same:
                self._warn_once(f"keyed 与 ablate({r.ablate.label()}) 使用了相同方向 {same[0]}。运行时它们是不同适配器，所以两者都会生效；但先烧录 keyed 再烧录 ablate 时，会因 (I − r̂r̂ᵀ)(W + α r κᵀ) 使 keyed 写入消失（若要烧录，请先烧录 ablate）")

    def _as_recipe(self, recipe: Recipe | list[str]) -> Recipe:
        return recipe if isinstance(recipe, Recipe) else self.recipe(units=list(recipe))

    def vectors(self, recipe: Recipe) -> tuple[dict[int, np.ndarray], list[dict]]:
        """add 单元的 control vector（按层）以及所有单元的详情。仅有 gain 的配方不需要统计。
        keyed(CHANGE-07)の行は dose_unit に関係なく、その層の ρ_L(あれば。無ければ計算はせずに注意だけ)と方向の r_L で
        比べる大きさ α‖r_L‖/ρ_L を埋める。"""
        keyed = bool(recipe.keyed_units)
        if not recipe.add_units:
            sv = self._keyed_stats(recipe) if keyed else None
            return {}, unit_details(self.weights, sv, recipe, self.direction_vec if keyed else None)
        sv = self.stats_view(Recipe(recipe.add_units, recipe.model_id, recipe.dose_unit, recipe.name))
        if keyed:
            sv = self._keyed_stats(recipe, sv)
        return (recipe_vectors(self.weights, sv, recipe, self.direction_vec),
                unit_details(self.weights, sv, recipe, self.direction_vec))

    def _keyed_stats(self, recipe: Recipe, sv: StatsView | None = None) -> StatsView:
        """keyed の行に要る ρ_L(残差ノルム統計)。α は raw なので書き込みには要らず、統計が無い層は計算せずに注意だけ出す。"""
        if sv is None:
            sv = StatsView(self.resid_stats())
        elif sv.resid is None:                       # absmax の add と混ぜたとき: 活性統計の読み口はそのまま、ρ だけ足す
            sv.resid = self.resid_stats()
        miss = sorted({u.layer for u in recipe.keyed_units if sv.resid is None or u.layer not in sv.resid.layers})
        if miss:
            self._warn_once(f"keyed 的层 {miss} 没有残差范数统计，因此未输出用于比较的幅度 α‖r_L‖/ρ_L（可用 gne stats resid 计算）")
        return sv

    def direction_path(self, name: str) -> Path:
        """方向名 → 文件。先搜索 <root>/directions/，找不到时再查反向查找的 <root>/reverse/。也可传路径。"""
        p = Path(name)
        if p.suffix == ".npz" and p.exists():
            return p
        for q in (self.store.direction_dir(self.meta.model_id) / f"{name}.npz",
                  self.store.reverse_dir(self.meta.model_id) / f"{name}.directions.npz"):
            if q.exists():
                return q
        raise GneError("FILE_NOT_FOUND", f"方向 {name} 不存在", "请用 gne direction --contrast <对照集> 创建")

    def direction_set(self, name: str):
        from .direction import DirectionSet
        p = self.direction_path(name)
        key, mt = str(p.resolve()), p.stat().st_mtime
        cache = self.__dict__.setdefault("_dirsets", {})
        if key not in cache or cache[key][0] != mt:
            cache[key] = (mt, DirectionSet.load(p))
        return cache[key][1]

    def direction_vec(self, name: str, L: int) -> np.ndarray:
        """方向单元（L:dir:NAME:dose）所添加的原始向量（该层的 mean_A − mean_B）。"""
        ds = self.direction_set(name)
        if L not in ds.raw:
            raise GneError("BAD_RECIPE", f"方向 {name} 中不存在第 {L} 层（0～{max(ds.raw)}）")
        return ds.raw[L]

    @staticmethod
    def _norms(vecs: dict[int, np.ndarray]) -> dict[str, float]:
        return {str(L): float(np.linalg.norm(v)) for L, v in sorted(vecs.items())}

    def messages(self, prompt: str | None = None, messages: list[dict] | None = None,
                 system: str | None = None, audio: list[str | Path] | None = None) -> list[dict]:
        """audio：音频文件。将用户发言作为音频（若同时提供 prompt，也会加入其中的文字）（media.py）。"""
        if messages:
            return messages
        if audio:
            from .media import audio_message
            user = audio_message(audio, prompt)
        elif prompt is None:
            raise GneError("BAD_ARGS", "--prompt 为必填项")
        else:
            user = {"role": "user", "content": prompt}
        return ([{"role": "system", "content": system}] if system else []) + [user]

    def generate(self, vectors: dict[int, np.ndarray] | None, messages: list[dict], *, max_tokens: int = 128,
                 when: str = "all", lora: list[tuple[str, float]] | None = None, sampling=None) -> dict:
        """单条生成（确保结果不随批处理组合方式变化，F2）。默认 greedy，可用 sampling（engine.Sampling）采样。
        when 是 §6.15 的位置别 steering，lora 是 gain（§6.16）的 LoRA 适配器。"""
        from .when import generate_when
        e = self.engine
        r, info = generate_when(e, e.steer_ctx(), messages, vectors, when, max_tokens=max_tokens, lora=lora,
                                sampling=sampling)
        return {"text": r.text, "finish": r.finish, "n_gen": r.n_gen, "token_ids": r.token_ids, "when": info}

    # --- steer ----------------------------------------------------------------------------------

    def steer(self, recipe: Recipe | list[str], *, prompt: str | None = None, messages: list[dict] | None = None,
              system: str | None = None, max_tokens: int = 128, compare: bool = True, when: str | None = None,
              sampling=None, audio: list[str | Path] | None = None) -> dict:
        """传入 sampling（engine.Sampling）时，无干预与干预使用相同 seed 采样（差异只来自干预）。
        audio：将音频文件作为用户发言（需要 mmproj）。"""
        recipe = self._as_recipe(recipe)
        msgs = self.messages(prompt, messages, system, audio)
        vecs, details = self.vectors(recipe)
        base = self.generate(None, msgs, max_tokens=max_tokens, sampling=sampling) if compare else None
        lora = self.lora_for(recipe)
        with self.patched(self.patches_for(recipe)):           # 専門家全体の gain・route(作業用コピーで読み直す)
            st = self.generate(vecs, msgs, max_tokens=max_tokens, when=when or recipe.when, lora=lora, sampling=sampling)
        return {"messages": msgs, "recipe": recipe.to_dict(), "units": details, "layer_norms": self._norms(vecs),
                "when": when or recipe.when, "sampling": None if sampling is None else sampling.to_dict(),
                "baseline": base, "steered": st, "differs": None if base is None else base["token_ids"] != st["token_ids"]}

    # --- export / import ------------------------------------------------------------------------

    def lora_for(self, recipe: Recipe | None) -> list[tuple[str, float]] | None:
        """应用配方中 gain 单元（§6.16）与 ablate（§6.17）的 LoRA（[(路径, 倍率)]）。两者皆无则返回 None。
        创建过的会复用。ablate 的 LoRA 以倍率 1 创建，scale 通过适配器倍率相乘（eval 的倍率列也只需一个文件）。
        keyed(CHANGE-07)も同じで、α/α_ref で作った LoRA を倍率 α_ref で当てる(全体を k 倍しても同じファイル)。"""
        if recipe is None:
            return None
        out = []
        lora_units = [u for u in recipe.gain_units if u.kind != "expert"]   # 専門家全体の gain はパッチ(patches_for)
        if lora_units:
            from .gain import export_gain_lora, lora_cache_path
            gr = Recipe(lora_units, self.meta.model_id, name=recipe.name)
            path = lora_cache_path(self.store.gain_dir(self.meta.model_id), gr)
            if not path.exists():
                tmp = path.with_suffix(".part")
                export_gain_lora(self.weights, gr, tmp)
                tmp.replace(path)
            out.append((str(path), 1.0))
        if recipe.ablate is not None and recipe.ablate.scale != 0:
            out.append((str(self.ablation_lora(recipe.ablate)), float(recipe.ablate.scale)))
        if recipe.proj_units and any(u.k for u in recipe.proj_units):     # CHANGE-06: ヘッドの方向成分の射影
            out.append((str(self.proj_lora(recipe)), 1.0))
        if recipe.keyed_units and any(u.k for u in recipe.keyed_units):   # CHANGE-07: 鍵つきの書き込み
            kpath, ref = self.keyed_lora(recipe)
            out.append((str(kpath), float(ref)))
        for u in recipe.query_units:                     # CHANGE-04 E: ヘッドの条件付きクエリ操作(実験的)
            if u.k:
                out.append((str(self.query_lora(u)), 1.0))
        return out or None

    # --- スケールパッチ(CHANGE-05 §C・§D)-----------------------------------------------------------

    def patches_for(self, recipe: Recipe | None):
        """专家整体 gain（将 ffn_down_exps 的 slab 乘以 k，k = 0 表示消除）与路由器偏置（route）的补丁。不存在时返回 None。"""
        if recipe is None:
            return None
        from .patch import Patcher, PatchSet
        units = [u for u in recipe.units if u.kind == "expert" and u.mode in ("gain", "route")]
        if not units:
            return None
        pt = Patcher(self.weights)
        ps = PatchSet("", self.meta.model_id)
        for u in units:
            if u.mode == "gain":
                ps.extend(pt.scale_slab(f"blk.{u.layer}.ffn_down_exps.weight", u.expert, float(u.k)))
            else:
                name = self.weights.router_bias_name(u.layer)
                if name is None:
                    raise GneError("ARCH_UNSUPPORTED", f"第 {u.layer} 层没有路由器偏置（{self.meta.arch}）")
                ps.extend(pt.add_element(name, u.expert, float(u.dose)))
        ps.name = "p-" + hashlib.sha256(json.dumps(ps.ops, sort_keys=True).encode()).hexdigest()[:12]
        return ps

    @contextmanager
    def patched(self, ps):
        """将补丁应用到工作副本，并用该副本重新读取模型。退出时恢复；下次使用时读取原模型。
        工作副本只创建一次（cp --reflink=auto）。重新读取借助 mmap 与页面缓存，约需数秒（docs/measurements.md）。"""
        if ps is None or not ps.patches:
            yield self.engine
            return
        from .patch import applied, make_work_copy
        work = make_work_copy(self.path, self.store.work_path(self.meta.model_id))
        stale = [n for n in applied(work) if n != ps.name]
        if stale:
            from .patch import PatchSet
            for n in stale:                                   # 前に落ちたときのパッチが残っていれば戻す
                f = self.store.patch_dir(self.meta.model_id) / f"{n}.patch.json"
                if not f.exists():
                    raise GneError("PATCH_CONFLICT", f"工作副本上应用了未知补丁 {n}", f"请删除 {work} 后重新创建")
                PatchSet.load(f).revert(work)
        ps.save(self.store.patch_dir(self.meta.model_id) / f"{ps.name}.patch.json")
        if self._engine is not None:                          # VRAM を空けてから読み直す
            self._engine.close()
            self._engine = None
        ps.apply(work)
        t0 = time.perf_counter()
        try:
            self._engine = self._make_engine(work)
            self.last_patch_reload_s = time.perf_counter() - t0
            yield self._engine
        finally:
            if self._engine is not None:
                self._engine.close()
                self._engine = None
            ps.revert(work)

    def proj_lora(self, recipe: Recipe) -> Path:
        """レシピの proj のユニット → LoRA(<root>/heads/<model_id>/proj-<鍵>.lora.gguf に作って使い回す)。鍵は
        ヘッド・s・方向の中身(作り直したら LoRA も作り直す)・direction_layer で決まる。"""
        from .headproj import ProjBuilder, export_proj_lora, proj_cache_key, proj_heads
        heads = [p for p in proj_heads(recipe) if p.s]
        b = ProjBuilder(self.weights, self.direction_set, direction_layer=recipe.direction_layer)
        path = self.store.heads_dir(self.meta.model_id) / f"proj-{proj_cache_key(heads, b)}.lora.gguf"
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_proj_lora(self.weights, recipe.like([u for u in recipe.proj_units if u.k]), tmp, self.direction_set,
                             warn=self.warn)
            tmp.replace(path)
        return path

    def keyed_lora(self, recipe: Recipe) -> tuple[Path, float]:
        """レシピの keyed のユニット → 実行時の LoRA(<root>/keys/<model_id>/lora/keyed-<鍵>.lora.gguf に作って使い回す)と
        アダプタの倍率 α_ref(= max|α|)。B は α/α_ref で書くので、全体の強さだけを変えても同じファイル。キャッシュ鍵は
        層・書き込み先・α の比・鍵の中身の sha・書く向き d の生のバイト(方向の中身・post-norm の補正・scale)・direction_layer で
        決まる(鍵や方向を作り直したら別のパス。Engine は同じパスの LoRA を 1 回しか読まないため、レビュー P15)。
        .part に書いてから置き換える。"""
        from .keyed import export_keyed_lora, keyed_cache_key, keyed_lora_path, keyed_plan
        kr = recipe.like([u for u in recipe.keyed_units if u.k])
        if not kr.units:
            raise GneError("BAD_RECIPE", "没有 keyed 单元（α 不为 0。例 --unit 19:ffn:keyed:a1@NAME#KEY）")
        rows = keyed_plan(self.weights, kr, self._key_of, self.direction_set, direction_layer=kr.direction_layer,
                          warn=self._warn_once)
        h, ref = keyed_cache_key(self.weights, kr, self._key_of, self.direction_set, direction_layer=kr.direction_layer,
                                 rows=rows)
        path = keyed_lora_path(self, h)
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_keyed_lora(self.weights, kr, tmp, self._key_of, self.direction_set, alpha_ref=ref,
                              direction_layer=kr.direction_layer)
            tmp.replace(path)
        return path, ref

    def query_lora(self, u) -> Path:
        """query 单元 → attn_q 的 rank-1 LoRA（在 <root>/heads/<model_id>/ 中创建并复用）。计算 u・v 需要捕获对照集。"""
        from .heads import query_vectors, write_query_lora
        seek = self._contrast_file(u.seek)
        key = json.dumps([u.layer, u.head, u.direction, str(seek), float(u.k),
                          vector_key_of(self.direction_set(u.direction).path)])
        path = self.store.heads_dir(self.meta.model_id) / f"query-{hashlib.sha256(key.encode()).hexdigest()[:16]}.lora.gguf"
        if not path.exists():
            qv = query_vectors(self, u.layer, u.head, direction=u.direction, seek=seek)
            tmp = path.with_suffix(".part")
            write_query_lora(self, u.layer, u.head, qv, float(u.k), tmp)
            tmp.replace(path)
        return path

    @staticmethod
    def _contrast_file(name: str | Path) -> Path:
        p = Path(name)
        if p.exists():
            return p
        q = CONFIGS / "contrast" / f"{name}.yaml"
        if q.exists():
            return q
        raise GneError("FILE_NOT_FOUND", f"对照集 {name} 不存在（请使用 configs/contrast/ 中的名称或路径）")

    def ablation_lora(self, spec) -> Path:
        """AblateSpec → 倍率 1 的消除 LoRA（在 <root>/ablate/<model_id>/ 中创建并复用）。"""
        from .ablate import export_ablation_lora, layer_list, vector_key
        r, meta = self.ablation_vector(spec.direction, from_layer=spec.from_layer)
        Ls = layer_list(spec.layers)
        tag = "" if Ls is None else "-L" + hashlib.sha256(json.dumps(Ls).encode()).hexdigest()[:8]
        path = self.store.ablate_dir(self.meta.model_id) / f"{spec.direction}-{vector_key(r)}{tag}.lora.gguf"
        if not path.exists():
            tmp = path.with_suffix(".part")
            export_ablation_lora(self.weights, r, tmp, layers=Ls, meta=meta)
            tmp.replace(path)
        return path

    def _target_weights(self, into: str | Path):
        """焼き込む先の別の GGUF(同じ構造で型だけ違うもの、例えば量子化していない F16 / BF16)。重みは memmap で読み、
        GPU には載せない。構造(arch・層の数・残差の幅)が違えば断る。"""
        from .weights import Weights
        p = Path(into)
        if not p.is_file():
            raise GneError("FILE_NOT_FOUND", f"烧录目标 {p} 不存在")
        if p.resolve() == Path(self.weights.path).resolve():
            return self.weights
        tw = Weights(p)
        a, b = self.weights.meta, tw.meta
        diff = [k for k in ("arch", "n_layer", "n_embd") if getattr(a, k) != getattr(b, k)]
        if diff:
            raise GneError("BAD_ARGS", f"烧录目标 {p.name} 的结构不同（{', '.join(f'{k}: {getattr(a, k)} ≠ {getattr(b, k)}' for k in diff)}）",
                           "请使用同一模型的另一种类型（如未量化的 F16 / BF16）")
        return tw

    def gain(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str = "auto",
             via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None,
             strip_meta: bool = False, into: str | Path | None = None, lora_f16: bool = False) -> dict:
        """gain ユニットを LoRA アダプタ GGUF(既定)か、W_up(ヘッドは W_O)を書き換えた GGUF(bake)に書き出す(§6.16、CHANGE-04 D)。
        ヘッドの方向成分の射影(proj、CHANGE-06)と鍵つきの書き込み(keyed、CHANGE-07)も同じ LoRA / 焼き込みに入れる
        (同じテンソルは A・B を連結。記録は gne.headproj と gne.keyed)。keyed を焼き込むときの requant の既定(auto)は
        テンソルごと: 元が Q8_0 なら Q8_0 のまま、それ以外は F16(gain.auto_qtype)。Q8_0 のままでは rank 1 の小さな編集の 1 割以上が
        刻みに埋もれる(edit_kept < gain.KEYED_KEEP_MIN)テンソルだけ F16(レビュー P13、verification.md F132・F139)。
        量子化し直しで編集が埋もれたら警告する(_warn_requant)。
        strip_meta なら gne の記録(gne.* のメタデータ・レシピ・名前の印・横の .headproj.json)を残さない(人に渡す版)。
        書いたあと、メタデータに手元のパスらしい文字列が残っていないかも確かめる(local_path_keys)。
        into を渡すと、そのモデル(同じ構造の別の型、例えば量子化していない F16 / BF16)の重みに書く。方向はこのモデルで作った
        ものを使い、書き換える行列は焼き込む先のものから作る(keyed の ΔW = α d κᵀ は W に依らないので、このモデルで作る)。
        GPU は使わない。"""
        from .gain import bake_gain_gguf, export_gain_lora
        from .headproj import merge_tensors
        recipe = self._as_recipe(recipe)
        W = self.weights if into is None else self._target_weights(into)
        if recipe.add_units:
            self.warn("add 单元不会进入 gain 导出（可通过 gne export-cvec 导出为 control vector）")
        routes = [u for u in recipe.units if u.mode == "route"]
        if not recipe.gain_units and not recipe.query_units and not routes and not recipe.proj_units and not recipe.keyed_units:
            raise GneError("BAD_RECIPE", "没有 gain 单元（例：--unit 14:8949:x2，注意力头为 21:H7:x0，方向分量投影为 33:H16:s6@NAME，带键写入为 19:ffn:keyed:a1@NAME#KEY）")
        if not bake and (routes or any(u.kind == "expert" for u in recipe.gain_units)):
            raise GneError("BAD_RECIPE", "专家整体的 gain 与 route 无法导出为 LoRA（会变成全秩）",
                           "请用 --bake 生成已应用缩放补丁的 GGUF（CHANGE-05 §D）")
        rep: dict = {"path": None, "bytes": 0}
        ptens, extra_meta, summary, extra_kv, kb = {}, None, {}, {}, None
        if recipe.proj_units:                            # CHANGE-06: ヘッドの方向成分の射影も同じ LoRA / 焼き込みに入れる
            from .headproj import ProjBuilder, layer_summary, proj_heads, proj_meta
            heads = proj_heads(recipe)
            b = ProjBuilder(W, self.direction_set, direction_layer=recipe.direction_layer)
            ptens = b.tensors(heads, self.warn)
            extra_meta = proj_meta(heads, b)
            summary = layer_summary(W, heads, ptens)
            if bake and requant == "patch":
                raise GneError("UNSUPPORTED_QUANT", "方向分量投影无法用缩放补丁写入（按列方向，不是简单倍率）",
                               "--requant q8_0 / f16，或 gne headproj export --bake --via-f16")
        if recipe.keyed_units:                           # CHANGE-07: 鍵つきの書き込みも同じ LoRA / 焼き込みに入れる
            if bake and requant == "patch":
                raise GneError("UNSUPPORTED_QUANT", "带键写入（keyed）无法用缩放补丁表示（它是 rank 1 加法，不是简单倍率）",
                               "请使用 --requant auto（默认）/ f16 / q8_0 / keep，或 LoRA（不加 --bake）分发")
            kb = self._keyed_bundle(recipe, W)
            extra_kv["gne.keyed"] = json.dumps(kb["meta"], ensure_ascii=False)
            summary = {**summary, "keyed_layers": kb["layers"], "keyed_rank": kb["rank"]}
        extra = merge_tensors(ptens, kb["tensors"] if kb else {})
        if via_f16 and not bake:
            raise GneError("BAD_ARGS", "--via-f16 需要与 --bake 一起使用")
        if strip_meta and (recipe.query_units or (bake and requant in ("patch", "auto") and not extra and not via_f16
                                                  and (routes or any(u.kind == "expert" for u in recipe.gain_units)))):
            raise GneError("BAD_ARGS", "不保留 gne 记录的导出只支持注意力头 proj/keyed/gain 和神经元 gain"
                                       "（暂不支持查询操作的 LoRA 与整个专家的缩放补丁）")
        imatrix_auto = False
        if bake and via_f16 and not imatrix:             # 渡されなければ、モデルの隣の <モデル名>.imatrix.gguf などを使う
            from .ablate import find_imatrix
            found = find_imatrix(W.path)
            if found is not None:
                imatrix, imatrix_auto = str(found), True
        if recipe.gain_units or routes or extra:
            if bake and via_f16:                         # F16 で書いて llama-quantize で元の型に戻す(CHANGE-06 E.2)
                from .ablate import requant_via_f16
                rep = requant_via_f16(W, lambda mid: bake_gain_gguf(W, recipe, mid, requant="f16",
                                                                              extra=extra, extra_meta=extra_meta,
                                                                              extra_kv=extra_kv, via_f16=True,
                                                                              strip_meta=strip_meta),
                                      out, llama_quantize=llama_quantize, imatrix=imatrix, neutral_imatrix=strip_meta)
                # 中間の F16 の大きさ(bytes_to・専門家のテンソルの増分)は llama-quantize で元の型に戻るので残さない
                rep["modified"] = [{**{k: v for k, v in m.items() if k != "bytes_to"}, "to": m["from"]}
                                   for m in rep["modified"]]
                rep.pop("expert_bytes_added", None)
            elif bake:
                rep = bake_gain_gguf(W, recipe, out, requant=requant, extra=extra, extra_meta=extra_meta,
                                     extra_kv=extra_kv, strip_meta=strip_meta, keyed=kb is not None)
                if requant == "auto" and kb is not None:
                    rep["requant"] = "auto"
            else:
                rep = export_gain_lora(W, recipe, out, extra=extra, extra_meta=extra_meta, extra_kv=extra_kv,
                                       strip_meta=strip_meta, f16=lora_f16)
                if ptens:                                # proj だけの階数(keyed と同じテンソルに連結しても proj の分)
                    rep["proj_rank"] = {n.split(".")[1]: int(B.shape[-1]) for n, (_, B) in ptens.items()}
                else:
                    rep.pop("proj_rank", None)
        rep.update(summary)
        if kb is not None and bake:
            self._warn_requant(rep)
            if not via_f16:
                self._warn_expert_size(rep)
        if into is not None:
            rep["into"] = Path(W.path).name if strip_meta else str(W.path)
        if bake and via_f16:
            rep["imatrix"] = None if imatrix is None else (Path(imatrix).name if strip_meta else str(imatrix))
            rep["imatrix_auto"] = imatrix_auto
        if strip_meta and rep.get("path"):              # 人に渡す版: 横の記録を作らず、手元のパスが残っていないかを見る
            from .ablate import local_path_keys
            side = Path(str(rep["path"]) + ".headproj.json")
            side.unlink(missing_ok=True)                 # 前に同じ名前で書いた記録が残っていれば消す
            rep["strip_meta"] = True
            rep["local_path_keys"] = local_path_keys(rep["path"])
            if rep["local_path_keys"]:
                self.warn("元数据中仍残留疑似本地路径的字符串：" + ", ".join(rep["local_path_keys"])
                          + "（使用 imatrix 时，llama-quantize 可能将制作它的文本路径写入 quantize.imatrix.dataset）")
        elif bake and extra:                             # CHANGE-06 E.2: 書き換えたテンソルと元の型のサイドカー
            side = Path(str(rep["path"]) + ".headproj.json")
            side.write_text(json.dumps({"recipe": recipe.to_dict(), "headproj": extra_meta,
                                        **({"keyed": kb["meta"]} if kb is not None else {}), "modified": rep.get("modified"),
                                        "requant": "via-f16" if via_f16 else requant}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
            rep["sidecar"] = str(side)
        if recipe.query_units:                           # CHANGE-04 E: クエリ操作は別の LoRA(llama.cpp の --lora を並べる)
            import shutil
            stem = Path(out).name.removesuffix(".gguf").removesuffix(".lora")
            q = []
            for u in recipe.query_units:
                dst = Path(out).with_name(f"{stem}-query-L{u.layer}-H{u.head}.lora.gguf")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.query_lora(u), dst)
                q.append(str(dst))
            rep["query_loras"] = q
            if rep["path"] is None:
                rep["path"], rep["bytes"] = q[0], Path(q[0]).stat().st_size
        return {"kind": "bake" if bake else "lora", "recipe": recipe.to_dict(), **rep}

    def _keyed_bundle(self, recipe: Recipe, W=None) -> dict:
        """書き出し用の keyed の LoRA(α そのまま、倍率 1)と記録。W(焼き込む先)は構造が同じことだけ確かめる
        (ΔW = α d κᵀ は W に依らないので、行列はこのモデルで作る)。"""
        from .keyed import keyed_bundle
        kb = keyed_bundle(self.weights, recipe, self._key_of, self.direction_set, direction_layer=recipe.direction_layer,
                          warn=self._warn_once)
        if W is not None and W is not self.weights:
            miss = [n for n in kb["tensors"] if not W.has_tensor(n)]
            if miss:
                raise GneError("BAD_ARGS", f"烧录目标中不存在 {miss[0]}（结构不同的模型）")
        return kb

    def _warn_requant(self, rep: dict, limit: float = 0.5) -> None:
        """量子化し直しで編集が埋もれたときの警告(keyed、レビュー P13)。量子化し直しても残った編集の割合 edit_kept_min が
        gain.KEYED_KEEP_MIN(0.9)を下回ったら。edit_kept が無い結果(古い記録など)は、量子化し直しで失った分 ÷ 編集の大きさ
        requant_error_max が limit を超えたら。requant_error は編集と無関係な丸めのゆらぎ(その型の量子化そのもののゆらぎ)も
        数えるので、刻みより小さい編集では編集が残っていても大きく出る(verification.md F140)。既定(auto)で Q8_0 から F16 に
        切り替えたテンソル(記録の fallback)も知らせる。"""
        from .gain import KEYED_KEEP_MIN
        err, kept = rep.get("requant_error_max"), rep.get("edit_kept_min")
        if (kept is not None and kept < KEYED_KEEP_MIN) or (kept is None and err is not None and err > limit):
            what = []
            if kept is not None:
                what.append(f"剩余编辑为 {kept * 100:.0f}%（edit_kept_min={kept:.2f}）")
            if err is not None:
                what.append(f"舍入偏差相当于编辑的 {err * 100:.0f}%（requant_error_max={err:.2f}）")
            self.warn("重新量化后，写入残差的权重矩阵编辑被舍入抹掉了（" + "、".join(what) + "）。请改用 --requant f16 或 auto（默认。仅将 Q8_0 中会埋没的张力改为 F16），或用 LoRA 分发")
        fb = [m["tensor"] for m in rep.get("modified") or [] if isinstance(m, dict) and m.get("fallback")] \
            or list(rep.get("fallback") or [])                  # 分割 GGUF への焼き込みは modified が名前だけ
        if fb:
            self.warn(f"{len(fb)} 个张量（{fb[0]} 等）在保持 Q8_0 时会有超过一成编辑被舍入抹掉，因此已用 F16 写入（--requant auto 的规则。若要保留原类型，请用 --requant keep）")

    def _warn_expert_size(self, rep: dict) -> None:
        """専門家の 3 次元テンソル(ffn_down_exps、keyed の expert:e)を元と違う型で書き直してファイルが大きくなったときの
        警告(rep の expert_bytes_added、通常の焼き込みと分割 GGUF への焼き込み)。1 つのテンソルの型は 1 つなので、編集しない
        専門家のスラブも同じ型に変換される(gain._expert_bake)。"""
        tot = int(rep.get("expert_bytes_added") or 0)
        if tot > 0:
            grown = [m["tensor"] for m in rep.get("modified") or []
                     if isinstance(m, dict) and "experts" in m and int(m.get("bytes_to", 0)) > int(m.get("bytes_from", 0))]
            what = f"专家张量 {len(grown)} 个（{grown[0]} 等）" if grown else "专家张量"
            self.warn(f"{what} 已用与原始文件不同的类型重写，因此文件增大了 {tot / 2 ** 20:.0f} MiB（一个张量只有一个类型，所以未编辑的专家也会被转换）。要按原类型写回，请使用 --requant keep（未编辑的专家保持原字节。普通烧录中 K-quant、IQ 类型会变为 Q8_0）。若用于分发，LoRA（不加 --bake）更小且更准确")

    def _warn_when(self, recipe: Recipe, what: str) -> None:
        if recipe.ablate is not None:
            self.warn(f"ablate（{recipe.ablate.label()}）无法用 {what} 表示，已移除（§6.17）。"
                      "请用 gne ablate 导出为 LoRA 适配器或改写后的 GGUF")
        if recipe.gain_units:
            self.warn(f"gain 单元（{len(recipe.gain_units)} 个）无法用 {what} 表示，已移除（§6.16）。"
                      "请用 gne gain 导出为 LoRA 适配器或改写后的 GGUF")
        if recipe.proj_units:
            self.warn(f"注意力头方向分量投影（proj，{len(recipe.proj_units)} 个）无法用 {what} 表示，已移除"
                      "（control vector 只能相加，CHANGE-06）。请用 gne headproj export（或 gne gain）导出为 LoRA")
        if recipe.keyed_units:
            self.warn(f"带键写入（keyed，{len(recipe.keyed_units)} 个）无法用 {what} 表示，已移除（因为只在输入激活触发键响应时才写入，不是常量加法，CHANGE-07）。请用 gne headproj export（或 gne gain）导出为 LoRA")
        if recipe.when != "all":
            self.warn(f"配方的 when={recipe.when} 无法用 {what} 表示，已按所有位置（all）导出（§6.15）。"
                      "若要在位置别模式下使用，请使用 gne steer / chat 或 gne serve --openai")

    def export_cvec(self, recipe: Recipe | list[str], path: str | Path, *, fmt: str = "cvec", verify: bool = False,
                    verify_prompt: str = "Who are you?", n_tokens: int = 32,
                    llama_completion: str | None = None) -> dict:
        recipe = self._as_recipe(recipe)
        if fmt == "cvec" and self.meta.arch in NO_CVEC:
            raise GneError("ARCH_UNSUPPORTED", f"{self.meta.arch} 的图不读取 control vector（即使传入 --control-vector 也不会生效）",
                           "请用 gne gain / gne headproj export 导出为 LoRA 适配器")
        self._warn_when(recipe, "control vector")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "没有 add 单元（只有 gain、proj、keyed 的配方无法成为 control vector）",
                           "请用 gne gain / gne headproj export 导出为 LoRA 适配器或改写后的 GGUF")
        moe_add = [u.label() for u in recipe.add_units if u.kind in ("expert_neuron", "shared")]
        if moe_add:                                           # CHANGE-05 §C・V22
            self.warn(f"{', '.join(moe_add)}：MoE 的 add 会按写入方向推动导出（无论是否选中专家，都会加到所有 token 上）")
        rest = [u.label() for u in recipe.units if u.mode in ("gain", "route")]
        if rest:
            self.warn(f"{', '.join(rest)}：gain 与 route 不会进入 control vector"
                      "（神经元、注意力头、专家神经元的 gain 走 LoRA，专家整体的 gain 与 route 走缩放补丁）")
        vecs, details = self.vectors(recipe)
        if fmt == "cvec" and 0 in vecs:
            raise GneError("BAD_RECIPE", "块 0 无法写入 control vector（llama.cpp 的限制）",
                           "请使用 --format npz/json/safetensors 或 gne bake")
        meta = {"model_id": self.meta.model_id, "recipe": recipe.to_dict(), "dose_unit": recipe.dose_unit,
                "n_embd": self.meta.n_embd, "gne_version": _version()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        export_vectors(vecs, path, fmt=fmt, n_layer=self.meta.n_layer, model_hint=self.meta.arch, meta=meta)
        check = None
        if verify:
            if fmt != "cvec":
                raise GneError("BAD_ARGS", "--verify 仅能与 --format cvec 一起使用")
            check = self.check_cvec_with_llama(Path(path), verify_prompt, n_tokens, llama_completion)
        from .steer import llama_server_args
        return {"path": str(path), "format": fmt, "recipe": recipe.to_dict(), "units": details,
                "layer_norms": self._norms(vecs), "verify": check,
                "llama_server_args": llama_server_args(recipe.sampling) if recipe.sampling else None}

    def check_cvec_with_llama(self, path: Path, prompt: str, n_tokens: int, binary: str | None) -> dict:
        """用原生加载器（llama-completion --control-vector）读取导出的 cvec，并比较 greedy 输出。"""
        from .verify import find_llama_completion, run_llama_completion
        binary = binary or find_llama_completion()
        if not binary:
            raise GneError("TOOL_MISSING", "找不到 llama-completion", "请用 --llama-completion 传入路径")
        n_embd, data = load_cvec_like_common([(path, 1.0)])
        back = {i + 1: data[i * n_embd:(i + 1) * n_embd] for i in range(data.size // n_embd)
                if np.any(data[i * n_embd:(i + 1) * n_embd])}
        msgs = [{"role": "user", "content": prompt}]
        ours = self.generate(back, msgs, max_tokens=n_tokens)
        base = self.generate(None, msgs, max_tokens=n_tokens)
        theirs = run_llama_completion(binary, self.engine, self.engine.render_chat(msgs), n_tokens, path)
        same = ours["text"] == theirs
        return {"tool": binary, "prompt": prompt, "n_tokens": n_tokens, "ours": ours["text"], "theirs": theirs,
                "match": same, "exact": same, "first_diff": None,
                "summary": "字符串一致" if same else "字符串不一致", "kl_next": None,
                "differs_from_baseline": ours["token_ids"] != base["token_ids"]}

    def import_cvec(self, path: str | Path, *, source: str = "exact", name: str | None = None) -> dict:
        if source == "exact":
            self.warn("--source exact：已将 direction.N 视为第 N 层的值。如果文件由 llama-cvector-generator 生成，"
                      "请使用 --source generator（偏移 1 补正，§5.2）")
        vecs = import_cvec_gguf(path, source=source)
        if not vecs:
            raise GneError("EMPTY_CVEC", f"{path} 中没有非零方向")
        n = name or Path(path).name.split(".")[0]
        out = self.store.reverse_dir(self.meta.model_id) / f"{n}.directions.npz"
        np.savez(out, layers=np.array(sorted(vecs), dtype=np.int32),
                 **{f"layer_{L}": v for L, v in vecs.items()}, source=source, source_file=str(path))
        return {"source_file": str(path), "source": source, "layer_norms": self._norms(vecs), "path": str(out)}

    # --- bake -----------------------------------------------------------------------------------

    def bake(self, recipe: Recipe | list[str], dst: str | Path, *, verify: bool = False,
             verify_prompt: str = "Who are you?", n_tokens: int = 32) -> dict:
        recipe = self._as_recipe(recipe)
        self._warn_when(recipe, "bake")
        if not recipe.add_units:
            raise GneError("BAD_RECIPE", "没有 add 单元", "gain 可通过 gne gain --bake 写入改写后的 GGUF")
        vecs, details = self.vectors(recipe)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        rep = bake_bias_gguf(self.weights, vecs, self.path, dst, recipe=recipe)
        check = None
        if verify:
            check = self.check_bake(vecs, Path(dst), verify_prompt, n_tokens)
        return {"recipe": recipe.to_dict(), "units": details, "report": rep.__dict__, "verify": check}

    def check_bake(self, vecs: dict[int, np.ndarray], dst: Path, prompt: str, n_tokens: int,
                   kl_max: float = 0.05) -> dict:
        """V9：比较烧制 GGUF 与原模型 + cvec 的 greedy 输出。同时确认烧制结果不同于无干预。

        bias 与 control vector 在数学上等价，但加法顺序（(x + Wa) + v 与 x + (Wa + b)）带来的舍入差异（约 1e-8）
        会被量化 GEMM 激活值的 int8 舍入放大，使下一 token 分布产生约 1e-3～1e-2 的 KL 偏差（docs/verification.md F8）。
        完全一致，或首次分叉能由微小差异（小于 0.5 nats）解释且 KL < kl_max 时，视为通过。"""
        from .engine import log_softmax
        from .verify import compare_greedy
        msgs = [{"role": "user", "content": prompt}]
        e = self.engine
        ids = e.tokenize_prompt(e.render_chat(msgs))
        ctx = e.steer_ctx()

        def run(engine, ctx_, v):
            engine.set_cvec(ctx_, v)
            engine.clear_memory(ctx_)
            pre = engine.prefill(ctx_, [ids])
            r = engine.generate_greedy(ctx_, [ids], max_tokens=n_tokens, prefilled=pre)[0]
            engine.set_cvec(ctx_, None)
            return r, log_softmax(pre.logits[0])

        base, _ = run(e, ctx, None)
        ref, ref_lp = (None, None) if 0 in vecs else run(e, ctx, vecs)
        if ref is None:
            self.warn("第 0 层无法用 control vector 复现，因此 bake 比较只检查与无干预的差异")
        kw = dict(self._engine_kw)
        self.close()                                   # VRAM を空けてから焼いた方を読む
        with Engine(dst, **kw) as e2:
            baked, baked_lp = run(e2, e2.steer_ctx(), None)
        out = {"tool": "llama-cpp-python（加载烧制后的 GGUF）", "prompt": prompt, "n_tokens": n_tokens,
               "ours": ref.text if ref else "", "theirs": baked.text, "match": False, "exact": False,
               "first_diff": None, "summary": None, "kl_next": None,
               "differs_from_baseline": baked.token_ids != base.token_ids}
        if ref is not None:
            cmp = compare_greedy(ref.token_ids, ref.top_ids, ref.top_lps, baked.token_ids, baked.top_ids, baked.top_lps)
            p = np.exp(ref_lp.astype(np.float64))
            kl = float((p * (ref_lp - baked_lp)).sum())
            out.update(exact=cmp["first_diff"] is None and len(ref.token_ids) == len(baked.token_ids),
                       first_diff=cmp["first_diff"], summary=cmp["summary"], kl_next=kl,
                       match=bool(cmp["ok"]) and kl < kl_max)
        return out


    # --- Phase 2: lens / reverse ----------------------------------------------------------------

    def _tok(self, ids) -> list[str]:
        return [self.engine.token_text(int(i)) for i in ids]

    def _need_rho(self, layers: list[int]) -> StatsView:
        return self.stats_view(Recipe([Unit(L, 0, 1.0) for L in layers]))

    def lens_logit(self, L: int, *, neurons: str | None = None, top_k: int = 12, show: int = 20) -> dict:
        from .lens import LogitLens, logit_lens_layer
        from .selection import parse_neurons
        self._check_layer(L)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        t0 = time.perf_counter()
        ll = logit_lens_layer(self.weights, L, neurons=sel, top_k=top_k, progress=self.progress)
        path = self.store.lens_dir(self.meta.model_id) / f"logit_L{L}.npz"
        if path.exists() and (neurons not in (None, "all")):
            ll = _merge_logit(LogitLens.load(path), ll)
        ll.save(path)
        pick = sel if neurons not in (None, "all") and not str(neurons).startswith("random:") else \
            ll.neurons[np.argsort(-ll.peak_z)[:show]]
        row = {int(n): i for i, n in enumerate(ll.neurons)}
        rows = [{"neuron": int(n), "peak_z": float(ll.peak_z[row[int(n)]]),
                 "up": self._tok(ll.up_ids[row[int(n)]]), "up_z": ll.up_z[row[int(n)]].tolist(),
                 "down": self._tok(ll.down_ids[row[int(n)]]), "down_z": ll.down_z[row[int(n)]].tolist()}
                for n in pick]
        return {"layer": L, "path": str(path), "n_neurons": int(sel.size), "top_k": top_k,
                "seconds": time.perf_counter() - t0, "neurons": rows}

    def lens_steer(self, L: int, *, neurons: str | None = None, dose: float = 0.8, random_controls: int = 64,
                   top_k: int = 12, resume: bool = True, show: int = 20, part: str | None = None,
                   controls: list[int] | None = None) -> dict:
        """传入 part 时（工作进程、CHANGE-03 B）写入 steer_L{L}.part-<part>.npz，并且只计算 controls 中的对照。"""
        from .lens import steer_lens_layer
        from .selection import parse_neurons
        self._check_layer(L, cvec=True)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        sv = self._need_rho([L])
        t0 = time.perf_counter()
        path = self.store.lens_dir(self.meta.model_id) / (f"steer_L{L}.part-{part}.npz" if part else f"steer_L{L}.npz")
        res = steer_lens_layer(self.engine, self.weights, sv, L, neurons=sel, dose=dose, contexts=None, top_k=top_k,
                               random_controls=random_controls, seed=self.seed, out=path, resume=resume,
                               progress=self.progress, controls=controls)
        return self._lens_rows(L, res, sel, neurons, path, dose, float(sv.rho(L)), show, time.perf_counter() - t0, self._tok)

    def _lens_rows(self, L, res, sel, neurons, path, dose, rho, show, seconds, tok) -> dict:
        null = {sg: _summ(res.null(sg)) for sg in ("pos", "neg")}
        explicit = neurons not in (None, "all") and not str(neurons).startswith("random:")
        if explicit:
            pick = sel
        else:
            score = np.fmax(res.kl_pos[sel], res.kl_neg[sel])
            pick = sel[np.argsort(-np.nan_to_num(score, nan=-1))[:show]]
        rows = [{"neuron": int(n), "kl_pos": float(res.kl_pos[n]), "kl_neg": float(res.kl_neg[n]),
                 "pct_pos": float(res.null_pct(res.kl_pos[n], "pos")), "pct_neg": float(res.null_pct(res.kl_neg[n], "neg")),
                 "up": tok(res.up_ids[n]), "up_delta": res.up_delta[n].tolist(),
                 "down": tok(res.down_ids[n]), "down_delta": res.down_delta[n].tolist()} for n in pick]
        return {"layer": L, "path": str(path), "dose": dose, "rho": rho, "n_done": int(res.done.sum()),
                "n_selected": int(sel.size), "n_controls": int(res.ctrl_done.sum()), "null": null,
                "seconds": seconds, "neurons": rows}

    def lens_summary(self, L: int, *, neurons: str | None = None, show: int = 20, seconds: float = 0.0) -> dict:
        """已保存的 steer lens 摘要（与 lens_steer 同形）。不会把模型加载到 GPU（词只从词表读取）。"""
        from .lens import SteerLens
        from .selection import parse_neurons
        path = self.store.lens_dir(self.meta.model_id) / f"steer_L{L}.npz"
        if not path.exists():
            raise GneError("FILE_NOT_FOUND", f"{path} 不存在")
        res = SteerLens.load(path)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        return self._lens_rows(L, res, sel, neurons, path, float(res.config["dose"]), float(res.config["rho"]), show,
                               seconds, lambda ids: [self.vocab.text(int(i)) for i in ids])

    def _check_layer(self, L: int, *, cvec: bool = False) -> None:
        if not 0 <= L < self.meta.n_layer:
            raise GneError("BAD_ARGS", f"第 {L} 层超出范围（0..{self.meta.n_layer - 1}）")
        if cvec and L == 0:
            raise GneError("BAD_ARGS", "第 0 层无法用 control vector 推动（llama.cpp 的限制）")
        if not self.weights.has_tensor(f"blk.{L}.ffn_down.weight") and not self.weights.has_tensor(f"blk.{L}.ffn_down_exps.weight"):
            raise GneError("ARCH_UNSUPPORTED", f"第 {L} 层没有 ffn_down.weight（dense）或 ffn_down_exps.weight（MoE）")

    def reverse(self, *, contrast: str | Path | None = None, directions: str | Path | None = None,
                method: str = "mean_diff", positions: str = "all", top: int = 200, layers: list[int] | None = None,
                verify: bool = False, verify_top: int | None = None, verify_controls: int = 8,
                verify_design: str = "quick", max_tokens: int = 128, name: str | None = None,
                verify_budget_s: float | None = None,
                standardize: bool = True) -> dict:
        from .lens import SteerLens
        from .reverse import ContrastSet, candidate_dicts, contrastive_directions, rank_by_direction
        if bool(contrast) == bool(directions):
            raise GneError("BAD_ARGS", "请指定 --contrast 或 --directions 中的恰好一个")
        t0 = time.perf_counter()
        rdir = self.store.reverse_dir(self.meta.model_id)
        act, scale = None, None
        info: dict = {"method": method, "positions": positions, "standardize": standardize}
        if contrast:
            cs = ContrastSet.load(contrast)
            name = name or cs.name
            from .reverse import side_interventions
            cr = contrastive_directions(self.engine, cs, layers=layers, method=method, positions=positions,
                                        progress=self.progress,
                                        interventions=side_interventions(self, cs) if cs.has_recipes else None)
            dirs, act, scale = cr.directions, cr.act_diff, cr.scale
            info.update(source=str(contrast), contrast_sha=cs.sha, n_rows=cr.n_rows, n_prompts=len(cs.user_prompts))
            np.savez(rdir / f"{name}.directions.npz", layers=np.array(sorted(dirs), dtype=np.int32),
                     **{f"layer_{L}": v for L, v in dirs.items()}, **{f"act_{L}": a for L, a in act.items()},
                     **{f"scale_{L}": v for L, v in scale.items()},
                     method=method, positions=positions, contrast_sha=cs.sha, source=str(contrast))
        else:
            p = Path(directions)
            if not p.exists():
                p = self.direction_path(str(directions))         # 逆引きの方向、無ければ gne direction の方向
            z = np.load(p)
            dirs = {int(L): z[f"layer_{L}"] for L in z["layers"] if layers is None or int(L) in layers}
            act = {int(k[4:]): z[k] for k in z.files if k.startswith("act_")} or None
            scale = {int(k[6:]): z[k] for k in z.files if k.startswith("scale_")} or None
            if standardize and scale is None:
                self.warn("该方向没有残差离散程度信息，因此未标准化直接计算 cos")
            name = name or p.name.split(".")[0]
            info.update(source=str(p), method=str(z["method"]) if "method" in z.files else "imported")
        steer_kl = {}
        for L in dirs:
            sp = self.store.lens_dir(self.meta.model_id) / f"steer_L{L}.npz"
            if sp.exists():
                sl = SteerLens.load(sp)
                steer_kl[L] = (sl.kl_pos, sl.kl_neg)
        cands, cos = rank_by_direction(self.weights, dirs, act_diff=act, steer_kl=steer_kl or None, top=top,
                                       exclude_layers=(0,), scale=scale if standardize else None)
        cpath = rdir / f"{name}.candidates.jsonl"
        cd = candidate_dicts(cands)
        cpath.write_text("".join(json.dumps(c) + "\n" for c in cd))
        layer_summary = [{"layer": L, "max_abs_cos": float(np.abs(c).max()),
                          "top1pct_abs_cos": float(np.sort(np.abs(c))[-max(1, c.size // 100):].mean()),
                          "dir_norm": float(np.linalg.norm(dirs[L]))} for L, c in sorted(cos.items())]
        verified = None
        if verify:
            n_v = verify_top or top
            if verify_budget_s is not None:              # CHANGE-03 C: 候補の層が決まってから、対照(層ごと)込みで予算に収める
                from .estimate import Speed, _design_cost
                per = _design_cost(Speed(self), verify_design, max_tokens)
                n_v = 1
                for k in range(1, min(len(cands), verify_top or top) + 1):
                    n_layers = len({c.layer for c in cands[:k]})
                    if (k + (verify_controls + 1) * n_layers) * per > verify_budget_s:   # +1 は層ごとのベースライン
                        break
                    n_v = k
                info["verify_budget"] = {"budget_s": verify_budget_s, "verify_top": n_v,
                                         "layers": len({c.layer for c in cands[:n_v]}), "seconds_per_unit": per}
            verified = self._verify_candidates(cands[:n_v], name, verify_design, verify_controls, max_tokens)
        by_expert = None
        if any(c.expert is not None for c in cands):            # MoE: 上位の候補が特定の専門家に集まっているか(CHANGE-05 §E.3)
            from collections import Counter
            cnt = Counter((c.layer, c.expert) for c in cands if c.expert is not None)
            by_expert = [{"layer": L, "expert": e, "n": n} for (L, e), n in cnt.most_common(20)]
            info["semantics"] = "direction"
        return {"name": name, "info": info, "candidates_path": str(cpath), "n_candidates": len(cands),
                "candidates": cd, "layers": layer_summary, "steer_lens_layers": sorted(steer_kl),
                "verify": verified, "seconds": time.perf_counter() - t0, "by_expert": by_expert}

    # --- Phase 3: sweep / rank ------------------------------------------------------------------

    @property
    def vocab(self):
        if getattr(self, "_vocab", None) is None:
            from .vocab import Vocab
            self._vocab = Vocab(self.path)
        return self._vocab

    def run_path(self, run: str) -> Path:
        """--run 的名称（<root>/runs/<model_id>/<run>.jsonl）或 JSONL 路径。"""
        p = Path(run)
        if p.suffix == ".jsonl" and (p.exists() or p.parent != Path(".")):
            return p
        return self.store.runs_dir(self.meta.model_id) / f"{run}.jsonl"

    def sweep(self, L: int, *, neurons: str | None = "random:384", design: str = "main_rel", random_controls: int = 64,
              max_tokens: int = 128, top_k: int = 5, run: str | None = None, prompt_set: str = "en",
              dose_unit: str = "rel", resume: bool = True, part: str | None = None,
              controls: list[int] | None = None) -> dict:
        """传入 part 时（工作进程、CHANGE-03 B）写入 <run>.part-<part>.jsonl，并且只运行 controls 中的对照。"""
        from .selection import parse_neurons
        from .sweep import load_design, run_sweep
        self._check_layer(L, cvec=True)
        sel = parse_neurons(neurons, self.weights.n_units(L), seed=self.seed)
        des = load_design(design, prompt_set=prompt_set)
        sv = self._need_rho([L]) if dose_unit == "rel" else self.stats_view(
            Recipe([Unit(L, int(sel[0]), 1.0)], dose_unit=dose_unit))
        name = run or f"L{L}-{design}"
        out = self.run_path(name)
        if part:
            out = out.with_name(f"{out.name.removesuffix('.jsonl')}.part-{part}.jsonl")
        r = run_sweep(self.engine, self.weights, sv, L=L, neurons=sel.tolist(), design=des, out=out,
                      model_id=self.meta.model_id, dose_unit=dose_unit, random_controls=random_controls,
                      max_tokens=max_tokens, top_k=top_k, resume=resume, seed=self.seed, progress=self.progress,
                      controls=controls)
        return {"run": name, **r, "design": design, "prompt_set": prompt_set, "n_neurons": int(sel.size),
                "n_controls": random_controls,
                **({"semantics": "direction"} if self.meta.n_expert and self.weights.is_moe(L) else {})}

    def rank(self, run: str, *, word_unit: str = "tokens", max_rel: float = 1.6, show: int = 40,
             find: list[int] | None = None) -> dict:
        from .rank import rank_records
        from .sweep import part_paths, read_rollouts
        if word_unit not in ("tokens", "regex"):
            raise GneError("BAD_ARGS", "--word-unit 只能为 tokens 或 regex")
        path = self.run_path(run)
        if not part_paths(path):                          # 本体か、ワーカーごとの部分(part-*)のどちらかがあればよい
            raise GneError("FILE_NOT_FOUND", f"run {path} 不存在", "请用 gne sweep --run 创建")
        t0 = time.perf_counter()
        recs = read_rollouts(path)
        words = self.vocab.words() if word_unit == "tokens" else None
        units = rank_records(recs, word_unit=word_unit, token_words=words, max_rel=max_rel, progress=self.progress)
        out = path.with_name(path.name.removesuffix(".jsonl") + ".units.jsonl")
        out.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units), encoding="utf-8")
        ctrl = np.array([u["mass"] for u in units if u["kind"] == "random"])
        neu = [u for u in units if u["kind"] == "neuron"]
        n_above = sum(1 for u in neu if u["null_pct"] is not None and u["null_pct"] >= 1.0)
        found = []
        for n in find or []:
            hit = next((u for u in units if u["kind"] == "neuron" and u["neuron"] == n), None)
            if hit is None:
                self.warn(f"N{n} 不在该 run 中")
            else:
                found.append(hit)
        return {"run": str(path), "parts": [p.name for p in part_paths(path)], "units_path": str(out),
                "n_units": len(units), "n_neurons": len(neu),
                "n_controls": int(ctrl.size), "n_baseline_records": sum(r["kind"] == "baseline" for r in recs),
                "word_unit": word_unit, "max_rel": max_rel, "control_mass": _summ(ctrl),
                "n_above_all_controls": n_above, "units": units[:show], "find": found,
                "seconds": time.perf_counter() - t0}

    # --- Phase 4: chat / dossier / agent ------------------------------------------------------------

    def neuron(self, L: int, n: int, *, rollouts: int = 4, compute: bool = False) -> dict:
        from .dossier import neuron_dossier
        return neuron_dossier(self, L, n, rollouts=rollouts, compute=compute)

    def model_overview(self) -> dict:
        from .dossier import model_overview
        return model_overview(self)

    def layer_summary(self, L: int, *, top: int = 20) -> dict:
        from .dossier import layer_summary
        return layer_summary(self, L, top=top)

    def search(self, q: str, *, mode: str = "lens", layer: int | None = None, limit: int = 50) -> dict:
        from .dossier import search
        return search(self, q, mode=mode, layer=layer, limit=limit)

    def fires(self, text: str, *, top: int = 50) -> dict:
        from .fires import fires
        return fires(self, text, top=top)

    # --- Phase 5 U3: connections と配置(重みだけ、CPU) -------------------------------------------------

    def connections(self, *, k: int = 24, window: int | None = None, chunk: int = 1024) -> dict:
        """计算所有层对的 connections 并保存到 <root>/connections/<model_id>.npz（别册 §8.4）。"""
        from .connections import compute_connections, save_connections
        resid = self.resid_stats()
        rho = {int(L): float(p) for L, p in zip(resid.layers, resid.p50, strict=True)} if resid is not None else None
        if rho is None:
            self.warn("缺少残差范数统计，因此不除以 rms 直接计算（跨层的 z 会偏向较深层。可用 gne stats resid 创建）")
        r = compute_connections(self.weights, k=k, window=window, rho=rho, chunk=chunk, progress=self.progress)
        path = self.store.connections_path(self.meta.model_id)
        save_connections(path, r)
        return {"path": str(path), **r["meta"]}

    def layout(self, kind: str = "writes", *, layers: list[int] | str | None = None, seed: int | None = None) -> dict:
        """计算 sheet 内布局（别册 §4.4）并保存到 <root>/layout/<model_id>/<kind>_L{L}.npy。"""
        from .layout import compute_layout
        Ls = list(range(self.meta.n_layer)) if layers in (None, "all") else [int(x) for x in layers]
        out = []
        for i, L in enumerate(Ls):
            if self.progress:
                self.progress(i, len(Ls), f"layout {kind} L{L}")
            cell, info = compute_layout(self, kind, L, seed=self.seed if seed is None else seed)
            p = self.store.layout_dir(self.meta.model_id) / f"{kind}_L{L}.npy"
            np.save(p, cell)
            out.append({"layer": L, "path": str(p), **info})
        if self.progress:
            self.progress(len(Ls), len(Ls), f"layout {kind}")
        return {"kind": kind, "layers": out}

    def embed(self, run: str, *, model: str | Path | None = None) -> dict:
        """将 run 的所有 rollout 转为句子嵌入 → UMAP 2D，并保存到 runs/<run>.umap.json（别册 §7）。"""
        from .embed import embed_run
        return embed_run(self, run, model=model, seed=self.seed, progress=self.progress)

    # --- 書き手の強さ -------------------------------------------------------------------------------

    def writers(self, contrast: str | Path, *, name: str | None = None, ref_layer: int | None = None,
                replies_a: str | Path | None = None, replies_b: str | Path | None = None, max_tokens: int = 96,
                top: int = 50) -> dict:
        """使用对照集 A / B 回复中的 token，测量各神经元写入残差的量，以及对参考层方向的直接贡献。
        也返回嵌入、mixer（如 attention）与 MLP 各占多少比例。"""
        from .writers import writer_contributions
        return writer_contributions(self, contrast, name=name, ref_layer=ref_layer, replies_a=replies_a,
                                    replies_b=replies_b, max_tokens=max_tokens, top=top, progress=self.progress)

    # --- attention ヘッド(CHANGE-04) -------------------------------------------------------------

    def heads_contrast(self, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                       system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48) -> dict:
        """在对照集上按注意力头测量 trigger_mass・pattern_shift・out_shift・dir_cos，中立どうしの差の p95 を基準値にして順位づける。"""
        from .heads import heads_contrast
        return heads_contrast(self, contrast, name=name, direction=direction, system_b2=system_b2, layers=layers,
                              max_tokens=max_tokens, progress=self.progress)

    def experts_contrast(self, contrast: str | Path, *, name: str | None = None, direction: str | None = None,
                         system_b2: str | None = None, layers: list[int] | None = None, max_tokens: int = 48) -> dict:
        """逐 MoE 层专家测量 usage_shift、weight_shift、contrib_cos、contrib_norm，中立どうしの差の p95 を基準値にして順位づける(CHANGE-05 §E.1)。"""
        from .experts import experts_contrast
        return experts_contrast(self, contrast, name=name, direction=direction, system_b2=system_b2, layers=layers,
                                max_tokens=max_tokens, progress=self.progress)

    def experts_sweep(self, L: int, *, ks=(0.0, 2.0), design: str = "quick", prompt_set: str = "en", max_tokens: int = 48,
                      experts: list[int] | None = None, run: str | None = None) -> dict:
        """按专家进行的 sweep（用消除或增强各专家的缩放补丁生成，并按词语出现方式排序，CHANGE-05 §E.2）。"""
        from .experts import experts_sweep
        return experts_sweep(self, L, ks=ks, design=design, prompt_set=prompt_set, max_tokens=max_tokens, experts=experts,
                             run=run, progress=self.progress)

    def moe_route(self, text: str, *, layers: list[int] | None = None, chat: bool = False) -> dict:
        """逐 token、逐 MoE 层返回选中的专家与路由权重。"""
        from .experts import moe_route
        return moe_route(self, text, layers=layers, chat=chat)

    def expert(self, L: int, e: int, *, top: int = 20) -> dict:
        """专家档案（使用统计、对照分析排名、神经元排名）。"""
        from .experts import expert_dossier
        return expert_dossier(self, L, e, top=top)

    def attention(self, text: str, *, pos: int = -1, layer: int | None = None, head: int | None = None, top: int = 5,
                  system: str | None = None, chat: bool = False) -> dict:
        """选中的 token 在各注意力头中关注何处（前 top 个）以及每个头输出的范数（gne heads attention）。"""
        from .heads import attention_view
        return attention_view(self, text, pos=pos, layer=layer, head=head, top=top, system=system, chat=chat)

    def heads_listen(self, user: str, *, heads: str | list[tuple[int, int]], system: str | None = None, reply: str | None = None,
                     max_tokens: int = 64, direction: str | None = None, top: int = 20) -> dict:
        """回复 token 通过所选注意力头关注上下文的哪些 token，以及各 token 对方向的贡献（gne heads listen）。"""
        from .heads import listen_view
        return listen_view(self, user, heads=heads, system=system, reply=reply, max_tokens=max_tokens, direction=direction, top=top)

    def head(self, L: int, h: int, *, lens: bool = True) -> dict:
        """注意力头档案。"""
        from .heads import head_dossier
        return head_dossier(self, L, h, lens=lens)

    def heads_query(self, L: int, h: int, *, direction: str, seek: str | Path, scales=(0.0, 0.5, 1.0, 2.0, 4.0)) -> dict:
        """CHANGE-04 E（实验性）的 V17：对每个 query 操作 scale，观察注意力头 h 对触发词的关注在有人设/无人设时如何变化。"""
        from .heads import check_query
        return check_query(self, L, h, direction=direction, seek=self._contrast_file(seek), scales=scales)

    def set_head_name(self, L: int, h: int, name: str) -> dict:
        from .heads import set_head_name
        set_head_name(self, L, h, name)
        return {"layer": L, "head": h, "name": name or None}

    def set_name(self, L: int, n: int, name: str) -> dict:
        from .dossier import set_name
        set_name(self, L, n, name)
        return {"layer": L, "neuron": n, "name": name or None}

    def recipe_save(self, name: str, units: list[str] | Recipe, *, dose_unit: str = "rel",
                    ablate: str | dict | None = None, when: str | None = None, sampling: dict | None = None,
                    default_direction: str | None = None, direction_layer: str | None = None,
                    default_key: str | None = None) -> dict:
        """sampling 是尝试该配方时使用的采样设置（不含 seed）。default_direction・direction_layer・default_key は
        @方向・#鍵を省いた proj・keyed のユニットの既定(CHANGE-06・07)。"""
        from .steer import sampling_values
        if not re.fullmatch(r"[\w.\-]+", name or ""):
            raise GneError("BAD_ARGS", f"配方名 {name!r} 只能包含字母、数字、_、-、.")
        r = units if isinstance(units, Recipe) else self.recipe(units=list(units), dose_unit=dose_unit, name=name,
                                                               ablate=ablate, when=when, default_direction=default_direction,
                                                               direction_layer=direction_layer, default_key=default_key)
        if sampling:
            r.sampling = sampling_values(sampling, seed=False)
        r.name = name
        path = self.store.recipes_dir() / f"{name}.yaml"
        r.save(path)
        return {"name": name, "path": str(path), "recipe": r.to_dict()}

    def recipe_path(self, name_or_path: str | Path) -> Path:
        p = Path(name_or_path)
        if p.suffix in (".yaml", ".yml") and p.exists():
            return p
        q = self.store.recipes_dir() / f"{name_or_path}.yaml"
        if not q.exists():
            raise GneError("FILE_NOT_FOUND", f"配方 {name_or_path} 不存在", f"请用 recipe_save 保存到 {q}")
        return q

    def chat(self, columns, prompts: list[str], *, system: str | None = None, max_tokens: int = 256,
             on_text=None, on_column=None, when: str | None = None, sampling=None) -> dict:
        from .chat import ChatState, chat_turn
        state = ChatState(columns, system)
        turns = []
        for u in prompts:
            turns.append({"user": u, "replies": chat_turn(self, state, u, max_tokens=max_tokens, on_text=on_text,
                                                          on_column=on_column, when_override=when, sampling=sampling)})
        return {"columns": [{"label": c.label, "recipe": c.recipe.to_dict() if c.recipe else None} for c in columns],
                "system": system, "turns": turns}

    # --- direction(§6.17)-----------------------------------------------------------------------

    def direction(self, *, contrast: str | Path, name: str | None = None, positions: str = "last",
                  select: str | int | None = None, mode: str = "ablate", layers: str | list[int] | None = None,
                  dose: float = 0.5, max_tokens: int = 64, kl_max: float = 0.1, min_distinct: float = 0.8,
                  per_layer: bool = False, judge_model: str | Path | None = None, judge_target: str | None = None,
                  response_tokens: int = 96, random_controls: int = 16) -> dict:
        """根据对照集创建并保存所有层方向 r_L；select="auto" 时选择效果较大的层（数字则手动指定）。
        random_controls > 0 时，在层选择中加入同量级的随机方向对照，只选择超过对照 p95 的层（CHANGE-03 A）。
        positions="response" 时从回复 token 创建（用 A / B 的 system 生成到 response_tokens 的回复再计算）。"""
        from .direction import (DirectionSet, candidate_layers, layer_table, load_contrast, response_directions,
                                save_directions, save_select, select_layer)
        from .evalx import Judge
        from .reverse import contrastive_directions
        if positions not in ("last", "all", "assistant", "response"):
            raise GneError("BAD_ARGS", "positions 只能为 last / all / assistant / response")
        t0 = time.perf_counter()
        cs, val = load_contrast(contrast)
        name = name or cs.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、-、.")
        if positions == "response":
            cr = response_directions(self, cs, max_tokens=response_tokens, progress=self.progress)
        else:
            from .reverse import side_interventions
            cr = contrastive_directions(self.engine, cs, positions=positions, with_act=False, progress=self.progress,
                                        interventions=side_interventions(self, cs) if cs.has_recipes else None)
        path = self.store.direction_dir(self.meta.model_id) / f"{name}.npz"
        save_directions(path, cr, cs, str(contrast))
        ds = DirectionSet.load(path)
        resid = self.resid_stats()

        def rho(L):
            return float(resid.rho(L)) if resid is not None and L in resid.layers else None
        sel = None
        if select is not None and str(select) != "auto":
            if not str(select).isdigit() or int(select) not in ds.raw:
                raise GneError("BAD_ARGS", f"--select 只能为 auto 或层号（{min(ds.raw)}～{max(ds.raw)}）")
            sel = {"mode": "manual", "selected": int(select)}
        elif select == "auto":
            judge = (Judge(judge_model, judge_target or val.target or cs.system_a or cs.name) if judge_model else None)
            try:
                sel = select_layer(self, ds, cs, val, mode=mode, layers=candidate_layers(self.meta.n_layer, layers),
                                   dose=dose, max_tokens=max_tokens, kl_max=kl_max, min_distinct=min_distinct,
                                   per_layer=per_layer, judge=judge, random_controls=random_controls,
                                   progress=self.progress)
                if sel.get("selected") is not None:
                    chosen = next(r for r in sel["rows"] if r["layer"] == sel["selected"])
                    sel["selected_null_pct"] = chosen.get("null_pct")
            finally:
                if judge is not None:
                    judge.close()
        if sel is not None:
            save_select(path.with_suffix(".select.json"), sel)
        return {"name": name, "path": str(path), "contrast": str(contrast), "contrast_sha": cs.sha, "positions": positions,
                **({"sides": cs.describe()} if (cs.has_recipes or cs.separate_prompts or cs.system_a is None
                                                or cs.system_b is None) else {}),
                "n_prompts": len(cs.user_prompts), "n_rows": cr.n_rows, "layers": layer_table(ds, rho),
                "validation": {"n_prompts": len(val.prompts), "match": val.match, "target": val.target},
                "select": sel, "seconds": time.perf_counter() - t0}

    def ablation_vector(self, direction: str, *, from_layer: int | None = None, per_layer: bool = False):
        """要消除的方向：from_layer 的 r（默认是 gne direction --select 选中的层），per_layer 时为每层 r_L（dict）。
        保存的方向中不存在的 "random-SEED" 会生成随机单位向量（消除对照，CHANGE-03 A）。"""
        m = re.fullmatch(r"random-(\d+)", direction or "")
        if m and not per_layer:
            try:
                self.direction_path(direction)
            except GneError:
                from .steer import random_direction
                return random_direction(self.meta.n_embd, int(m.group(1)), 0), {"direction": direction, "control": True}
        ds = self.direction_set(direction)
        if per_layer:
            return ds.raw, {"direction": ds.name, "per_layer": True}
        if from_layer is None:
            sel = ds.path.with_suffix(".select.json")
            chosen = json.loads(sel.read_text()).get("selected") if sel.exists() else None
            if chosen is None:
                raise GneError("BAD_ARGS", f"方向 {ds.name} 尚未确定层",
                               "请用 --from-layer 指定，或使用 gne direction --select auto / --layer 选择")
            from_layer = int(chosen)
        if from_layer not in ds.raw:
            raise GneError("BAD_ARGS", f"方向 {ds.name} 中不存在第 {from_layer} 层（0～{max(ds.raw)}）")
        return ds.raw[from_layer], {"direction": ds.name, "from_layer": from_layer}

    def ablate(self, direction: str, out: str | Path, *, from_layer: int | None = None, per_layer: bool = False,
               layers: list[int] | None = None, scale: float = 1.0, embed: bool = True, bake: bool = False,
               requant: str = "q8_0", via_f16: bool = False, imatrix: str | None = None,
               llama_quantize: str | None = None, verify: bool = False) -> dict:
        """将方向的投影消除导出为 LoRA 适配器 GGUF（默认）或改写权重的 GGUF（bake）（§6.17）。
        verify 时，对 LoRA 检查 V13（r 分量是否只按应有部分残留），对 bake 检查 V13 与 V14（LoRA 与 greedy 是否一致）。"""
        from .ablate import bake_ablation_gguf, bake_via_f16, export_ablation_lora
        if not 0 < scale <= 1.5:
            raise GneError("BAD_ARGS", "--scale 须大于 0 且不超过 1.5（1 表示完全消除）")
        r, meta = self.ablation_vector(direction, from_layer=from_layer, per_layer=per_layer)
        t0 = time.perf_counter()
        if not bake:
            rep = export_ablation_lora(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                                       progress=self.progress)
            kind = "lora"
        elif via_f16:
            rep = bake_via_f16(self.weights, r, out, layers=layers, scale=scale, embed=embed, meta=meta,
                               llama_quantize=llama_quantize, imatrix=imatrix, progress=self.progress)
            kind = "bake"
        else:
            rep = bake_ablation_gguf(self.weights, r, out, layers=layers, scale=scale, embed=embed, requant=requant,
                                     meta=meta, progress=self.progress)
            kind = "bake"
        for w_ in rep.get("warnings", []):
            self.warn(w_)
        check = self._check_ablation(r, Path(out), kind, rep) if verify else None
        return {"kind": kind, **meta, **{k: v for k, v in rep.items() if k != "warnings"}, "verify": check,
                "seconds": time.perf_counter() - t0}

    def _check_ablation(self, r, out: Path, kind: str, rep: dict, n_tokens: int = 32) -> dict:
        """用导出的文件检查 V13 以及（bake 时）V14。"""
        from .ablate import embed_in_lora, export_ablation_lora, residual_writers, unit
        from .verify import Verifier
        if isinstance(r, dict):
            return {"skipped": "不支持验证逐层方向（--per-layer）"}
        v = Verifier(self.engine, self.weights, seed=self.seed)
        ru = unit(r)
        ids = v.prompt_ids[0]
        layers = rep.get("layers")
        full = layers == sorted(residual_writers(self.weights))
        er = self.weights.embed_rows(ids)[1:].astype(np.float64) @ ru
        if kind == "lora":
            res = v._resid_all(self.engine, ids, [(out, 1.0)])
            expect = 0.0 if rep.get("embed") else er
        else:
            e2 = Engine(out, n_gpu_layers=self._engine_kw["n_gpu_layers"], n_ctx=self.engine.n_ctx, n_seq_max=self.engine.n_seq_max)
            try:
                res = v._resid_all(e2, ids)
                lp = out.with_name(out.name + ".v14.lora.gguf")
                export_ablation_lora(self.weights, r, lp, layers=layers, scale=rep["scale"],
                                     embed=rep["embed"] and embed_in_lora(self.weights))
                seqs = v.prompt_ids[:2]

                def run(engine, lora=None):
                    ctx = engine.steer_ctx()
                    engine.set_lora(ctx, lora)
                    try:
                        return [engine.generate_greedy(ctx, [s_], max_tokens=n_tokens)[0] for s_ in seqs]
                    finally:
                        engine.set_lora(ctx, None)
                a, b = run(self.engine, [(lp, 1.0)]), run(e2)
                lp.unlink(missing_ok=True)
            finally:
                e2.close()
            expect = 0.0 if rep.get("embed") else er
        scale = rep["scale"]
        devs = []
        for L, la in sorted(res.items()):
            # 部分消去(scale < 1)なら r 成分は (1 − s) 倍ぶん残るので、比べるのは全部消したときだけ
            devs.append(float((np.abs(la @ ru - expect) / np.linalg.norm(la, axis=1)).max()))
        out_ = {"v13": {"max_dev": max(devs), "pass": bool(max(devs) < 1e-2) if (scale == 1.0 and full) else None,
                        "note": None if (scale == 1.0 and full) else "部分消除或指定层范围时 r 分量会残留，因此不判定通过/失败"}}
        if kind == "bake":
            from .verify import compare_greedy
            cmps = [compare_greedy(x.token_ids, x.top_ids, x.top_lps, y.token_ids, y.top_ids, y.top_lps)
                    for x, y in zip(a, b, strict=True)]
            out_["v14"] = {"n_tokens": n_tokens, "match": [c["first_diff"] is None for c in cmps],
                           "summary": [c["summary"] for c in cmps], "pass": all(c["ok"] for c in cmps),
                           "note": "重新量化的误差可能使接近的 token 分叉（F16 误差更小）"
                           if not all(c["first_diff"] is None for c in cmps) else None}
        return out_

    # --- eval(§6.14)---------------------------------------------------------------------------

    def eval(self, recipe: Recipe | list[str], *, doses=(0, 0.25, 0.5, 0.75, 1.0), suite: str | Path | None = None,
             when: str | None = None, judge_model: str | Path | None = None, judge_target: str | None = None,
             name: str | None = None, random_controls: int = 0, reactivity: str | Path | None = None,
             head_controls: int = 0, control_pool: str = "before", direction_metrics: bool = True,
             dir_max_examples: int | None = 64, keyed_contrast: str | Path | None = None, keyed_metrics: bool = True,
             keyed_examples: int | None = 8, keyed_match: bool = True, keyed_pseudo: int = 8,
             keyed_subset: str | None = None) -> dict:
        """测量配方在各倍率下的副作用（能力、ppl、语言混入）与性格变化量，并保存到 <root>/evals/<model_id>/<name>.json。
        random_controls > 0 时，每个倍率也测量同量级的随机对照配方（行中的 ctrl・ctrl_novelty_p95 等）。
        reactivity（对照集 YAML）还会测量反应性（有/无触发词的差异），并与 head_controls 个随机注意力头比较。
        方向を使うレシピ(proj など)では方向の指標(射影の値の下がり方・ずれ・判定の一致率)も測る。direction_metrics=False で
        測らない。dir_max_examples はその測定で読む質問の数(A・B それぞれ、一致率の返事も。多ければ等間隔に間引く)。
        keyed のユニットを持つレシピ(CHANGE-07 E)では eval の選択性も測る(行の keyed と結果の keyed、evalx.evaluate)。
        keyed_contrast は鍵を作った対照集合(configs/contrast/ の名前かパス。省くと方向を作った対照集合が鍵と同じならそれ、
        どちらも無ければ結果の keyed は skipped)、keyed_examples は会話の組ごとに読む数(None か 0 で全部)、keyed_pseudo は
        比較用の鍵の本数、keyed_subset は読む例(even / odd / all、省くと鍵の作成と反対側)、keyed_match=False で方式の比較の
        大きさ合わせを省く、keyed_metrics=False で測らない。"""
        from .evalx import Judge, evaluate
        recipe = self._as_recipe(recipe)
        if not recipe.units and recipe.ablate is None:
            raise GneError("BAD_RECIPE", "没有单元")
        if (judge_model is None) != (judge_target is None):
            raise GneError("BAD_ARGS", "请同时指定 --judge-model 与 --judge-target")
        keyed_subset = keyed_subset or None
        if keyed_subset is not None and keyed_subset not in ("even", "odd", "all"):
            raise GneError("BAD_ARGS", f"keyed_subset 必须是 even / odd / all 之一（{keyed_subset!r}）")
        keyed_examples = None if keyed_examples is None else int(keyed_examples)
        keyed_pseudo = int(keyed_pseudo)
        if (keyed_examples or 0) < 0 or keyed_pseudo < 0:
            raise GneError("BAD_ARGS", "keyed_examples 和 keyed_pseudo 必须为 0 或更大")
        keyed_contrast = keyed_contrast or None
        if keyed_contrast is not None and not recipe.keyed_units:
            self.warn("配方中没有 keyed 单元，因此不使用 keyed_contrast")
            keyed_contrast = None
        elif keyed_contrast is not None and keyed_metrics:     # 名前も受ける。無ければ長い eval の前に止める
            keyed_contrast = self._contrast_file(keyed_contrast)
        judge = Judge(judge_model, judge_target) if judge_model else None
        try:
            d = evaluate(self, recipe, doses=doses, suite=suite, when=when, judge=judge,
                         random_controls=random_controls, reactivity=reactivity, head_controls=head_controls,
                         control_pool=control_pool, direction_metrics=direction_metrics,
                         dir_max_examples=dir_max_examples, keyed_contrast=keyed_contrast,
                         keyed_metrics=bool(keyed_metrics), keyed_examples=keyed_examples or None,
                         keyed_match=bool(keyed_match), keyed_pseudo=keyed_pseudo, keyed_subset=keyed_subset,
                         progress=self.progress)
            d.pop("_baseline", None)
        finally:
            if judge is not None:
                judge.close()
        name = name or recipe.name or "units-" + hashlib.sha256(
            json.dumps([recipe.to_dict()["units"], recipe.to_dict()["ablate"]], sort_keys=True).encode()).hexdigest()[:8]
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、-、.")
        if d["when"] != "all":
            name += "@" + re.sub(r"[^\w.\-]+", "_", d["when"])
        path = self.store.eval_dir(self.meta.model_id) / f"{name}.json"
        d = {"name": name, "path": str(path), "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **d}
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1))
        return d

    def eval_compare_ablation(self, direction: str, *, top: int = 20, from_layer: int | None = None,
                              doses=(0, 0.5, 1.0), suite: str | Path | None = None, when: str | None = None,
                              max_tokens: int = 64) -> dict:
        """§6.17：将同一方向得到的“前 top 个神经元消除（gain ×0）”与“方向消除（ablate）”并列在同一张表中。
        能力、ppl、混入由 eval 测量，目标行为的残留程度由方向对照集的判定（validate.match）测量。"""
        from .direction import behavior, load_contrast, top_writer_neurons
        from .steer import AblateSpec, Unit
        t0 = time.perf_counter()
        ds = self.direction_set(direction)
        _, meta = self.ablation_vector(direction, from_layer=from_layer)
        neurons = top_writer_neurons(self, ds, top)
        rn = Recipe([Unit(L, n, 0.0, mode="gain", k=0.0) for L, n, _ in neurons], self.meta.model_id,
                    name=f"{ds.name}-neurons{top}")
        rd = Recipe([], self.meta.model_id, name=f"{ds.name}-ablate",
                    ablate=AblateSpec(ds.name, 1.0, "all", meta["from_layer"]))
        ev = {"neurons": self.eval(rn, doses=doses, suite=suite, when=when),
              "direction": self.eval(rd, doses=doses, suite=suite, when=when)}
        src = ds.meta.get("source")
        cs_val = load_contrast(src) if src and Path(src).exists() else None
        if cs_val is None:
            self.warn(f"找不到方向 {ds.name} 的对照集 {src}，因此不测量行为的残留程度")
        elif not cs_val[1].match:
            self.warn("对照集没有 validate.match，因此不测量行为的残留程度（一致率）")
        base = behavior(self, *cs_val, None, max_tokens=max_tokens) if cs_val else None
        rows = []
        for method, recipe in (("neurons", rn), ("direction", rd)):
            for row in ev[method]["rows"]:
                d = row["dose"]
                b = None
                if cs_val is not None:
                    b = base if d == 0 else behavior(self, *cs_val, recipe.scaled(d), max_tokens=max_tokens,
                                                     base_lp=base["lp"])
                rows.append({"method": method, "dose": d,
                             "rate": None if b is None else b["rate"],
                             "rate_drop": None if b is None or b["rate"] is None else base["rate"] - b["rate"],
                             "kl_b": None if b is None else b["kl_b"],
                             "capability_drop_pt": row["capability_drop_pt"], "ppl_worse_pct": row["ppl_worse_pct"],
                             "mix_increase": row["mix_increase"], "novelty": row["persona"]["novelty"],
                             "shippable": row["shippable"], "sample": None if b is None else b["sample"]})
        out = {"direction": ds.name, "from_layer": meta["from_layer"], "top": top,
               "neurons": [{"layer": L, "neuron": n, "cos": c} for L, n, c in neurons],
               "evals": {k: v["name"] for k, v in ev.items()}, "contrast": src,
               "match": cs_val[1].match if cs_val else None, "base_rate": None if base is None else base["rate"],
               "rows": rows, "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{ds.name}-compare-ablation.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def eval_compare_methods(self, recipe: Recipe | list[str], *, match_effect: float | None = None,
                             score_layer: int | None = None, top_neurons: int = 20, suite: str | Path | None = None,
                             when: str | None = None, contrast: str | Path | None = None, name: str | None = None) -> dict:
        """CHANGE-06 F.3: 同じ方向について、全層射影(ablate)・ヘッドの方向成分の射影(このレシピの proj)・同じヘッドの
        丸ごと倍率(gain、k_h = 1 − α·s_h)・上位ニューロンの消去(gain、k = 1 − α)を、A 側の射影の値(速い指標、score_layer)が
        同じ水準(match_effect、省けばこのレシピの水準)になるよう全体の倍率を二分探索で合わせてから、副作用を同じ表に並べる。
        合わせられなかった方式は理由を書いて除外する。"""
        from .causal import _bisect
        from .direction import Scorer, base_layer_of, behavior, drift, score_layers_default, top_writer_neurons
        from .evalx import evaluate
        from .headproj import proj_heads
        from .steer import AblateSpec, Unit
        t0 = time.perf_counter()
        recipe = self._as_recipe(recipe)
        heads = proj_heads(recipe)
        if not heads:
            raise GneError("BAD_RECIPE", "--compare-methods 仅用于包含 proj 单元的配方（作为比较基准的组）")
        dname = heads[0].direction
        ds, cs, val = self._direction_contrast(dname, contrast)
        base = base_layer_of(ds)
        if base is None:
            self.direction_score(dname, contrast=contrast)
            ds = self.direction_set(dname)
            base = base_layer_of(ds)
        if score_layer is None:                          # 組のヘッドより後ろの、最初の射影の値を測る層
            sls = score_layers_default(self, ds, base)
            score_layer = next((L for L in sls if L >= max(p.layer for p in heads)), sls[-1])
        S = int(score_layer)
        sc = Scorer(self, ds, cs, layers=[S])
        b0 = sc.run(keep=True)
        base_a = float(b0["A"]["values"][S].mean())

        def mean_a(r: Recipe | None) -> float:
            lora = self.lora_for(r) if r is not None else None
            return float(sc.run(lora, sides="A")["A"]["values"][S].mean())
        target = float(match_effect) if match_effect is not None else mean_a(recipe)
        s_hi = max(abs(p.s) for p in heads)
        neurons = [(L, n, c_) for L, n, c_ in top_writer_neurons(self, ds, 4 * top_neurons) if L <= S][:top_neurons]
        methods = {
            "ablate": (lambda a: Recipe([], self.meta.model_id, name=f"{ds.name}-ablate",
                                        ablate=AblateSpec(ds.name, a, "all", base)), 1.5,
                       f"从全层写入中消除 L{base} 的 r（scale = α）"),
            "proj": (lambda a: recipe.scaled(a), max(1.0, 30.0 / s_hi), "此配方的 proj（s = α · s_h）"),
            "head_gain": (lambda a: recipe.like([Unit(p.layer, None, 0.0, "head", mode="gain", k=1.0 - a * p.s, head=p.head)
                                                 for p in heads], name=f"{ds.name}-headgain"), 2.0,
                          "将同一注意力头的输出整体乘以 k_h = 1 − α·s_h"),
            "neurons": (lambda a: Recipe([Unit(L, n, 0.0, mode="gain", k=1.0 - a) for L, n, _ in neurons],
                                         self.meta.model_id, name=f"{ds.name}-neurons{len(neurons)}"), 1.0,
                        f"沿方向写入的前 {len(neurons)} 个神经元（到 L{S}）乘以 k = 1 − α（α = 1 时消除）"),
        }
        rows, baseline = [], None
        base_rate = behavior(self, cs, val, None)["rate"] if val.match else None
        for key, (make, hi, desc) in methods.items():
            tm = time.perf_counter()
            if key == "neurons" and not neurons:
                rows.append({"method": key, "desc": desc, "excluded": f"L{S} 之前没有沿方向写入的神经元"})
                continue
            if key == "proj" and match_effect is None:
                alpha, val_a, reached = 1.0, target, True
            else:
                # 方式によっては射影の値が倍率に単調でない(丸ごと倍率は直交成分も反転する)ので、粗く見てから詰める
                alpha, val_a, reached, _ = _bisect(lambda a, make=make: mean_a(make(a)), target, hi, grid=8)
            if not reached:
                rows.append({"method": key, "desc": desc, "alpha": alpha, "score_a": val_a,
                             "excluded": f"在 α = 0～{hi:g} 范围内，A 侧投影值最小也只有 {val_a:.2f}（α = {alpha:.3g}），达不到目标 {target:.2f}"})
                continue
            r = make(alpha)
            got = sc.run(self.lora_for(r), keep=True)
            par, orth = drift(b0["B"]["resid"][S], got["B"]["resid"][S], sc.r[S])
            ev = evaluate(self, r, doses=(1.0,), suite=suite, when=when or recipe.when, direction_metrics=False,
                          baseline=baseline)
            baseline = ev.pop("_baseline")
            er = next(x for x in ev["rows"] if x["dose"] == 1.0)
            rate = behavior(self, cs, val, r)["rate"] if val.match else None
            rows.append({"method": key, "desc": desc, "alpha": alpha, "score_a": val_a,
                         "score_b": float(got["B"]["values"][S].mean()), "drift_par_b": par, "drift_orth_b": orth,
                         "capability_drop_pt": er["capability_drop_pt"], "ppl_worse_pct": er["ppl_worse_pct"],
                         "mix_increase": er["mix_increase"], "novelty": er["persona"]["novelty"],
                         "distinct4": er["persona"]["distinct4"], "shippable": er["shippable"],
                         "rate": rate, "rate_drop": None if rate is None or base_rate is None else base_rate - rate,
                         "recipe": r.to_dict(), "seconds": time.perf_counter() - tm})
        out = {"name": name or f"{recipe.name or ds.name}-compare-methods", "direction": ds.name, "score_layer": S,
               "base_layer": base, "target": target, "target_from": "match_effect" if match_effect is not None else "recipe",
               "base_score_a": base_a, "base_score_b": float(b0["B"]["values"][S].mean()), "match": val.match,
               "base_rate": base_rate, "rows": rows, "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{out['name']}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def eval_alloc_curve(self, alloc: str | Path, *, n_heads: list[int] | None = None, suite: str | Path | None = None,
                         when: str | None = None, random_controls: int = 0, name: str | None = None) -> dict:
        """CHANGE-06 F.4: gne heads allocate の各組(ヘッド数ごと)を eval で評価して、ヘッド数の削減曲線を作る。"""
        from .evalx import evaluate
        t0 = time.perf_counter()
        p = Path(alloc)
        if not p.exists():
            p = self.store.heads_dir(self.meta.model_id) / f"{alloc}.alloc.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"分配结果 {alloc} 不存在", "请用 gne heads allocate 创建")
        al = json.loads(p.read_text(encoding="utf-8"))
        table = {r["n_heads"]: r for r in al["table"] if not r.get("skipped")}
        want = sorted(n_heads or table, reverse=True)
        miss = [n for n in want if n not in table]
        if miss:
            raise GneError("BAD_ARGS", f"分配结果中没有 {miss} 个组（现有 {sorted(table)}）",
                           f"请用 gne heads allocate --n-heads {','.join(map(str, want))} 重新创建")
        rows, baseline = [], None
        for n in want:
            t = table[n]
            r = Recipe.load(t["recipe"])
            ev = evaluate(self, r, doses=(1.0,), suite=suite, when=when, random_controls=random_controls,
                          baseline=baseline, progress=self.progress)
            baseline = ev.pop("_baseline")
            er = next(x for x in ev["rows"] if x["dose"] == 1.0)
            rows.append({"n_heads": n, "units": t["units"], "s_max": t["s_max"], "remain": t["remain"], "a_max": t["a_max"],
                         "alloc_b_drift_orth": t["b_drift_orth"], "recommended": t.get("recommended", False),
                         **{k: er.get(k) for k in ("capability_drop_pt", "ppl_worse_pct", "mix_increase", "dir_drop_a",
                                                   "drift_orth_a", "drift_orth_b", "rate", "rate_drop", "dir_null_pct",
                                                   "rate_null_pct", "shippable", "notes")},
                         "novelty": er["persona"]["novelty"], "recipe": t["recipe"]})
        out = {"name": name or f"{al['name']}-nheads", "alloc": str(p), "direction": al["direction"],
               "score_layer": al["score_layer"], "target": al["target"], "rows": rows, "random_controls": random_controls,
               "seconds": time.perf_counter() - t0}
        path = self.store.eval_dir(self.meta.model_id) / f"{out['name']}.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
        return {**out, "path": str(path)}

    def behavior_null(self, recipe: Recipe, contrast: str | Path, *, random_controls: int = 16, max_tokens: int = 64,
                      mode: str | None = None, sampling=None, seeds: int = 1, prompts: str | Path | list[str] | None = None) -> dict:
        """CHANGE-03 A：比较配方效果（对照集 validate.match 一致率的变化）与同量级随机对照配方
        random_controls 个的效果。null_pct 表示“效果大于多少比例的对照”（低于 0.95 时无法与对照区分）。
        传入 sampling（engine.Sampling）时，每个提示词用 seeds 条采样生成（与调好配方时的条件一致）。
        prompts 可替换验证提示词（字符串列表或对应 YAML。若为对照集 YAML，则使用 reactivity 的
        with 或 user_prompts）。"""
        from .direction import behavior_null, load_contrast
        cs, val = load_contrast(contrast)
        ps = prompts if prompts is None or isinstance(prompts, list) else load_prompt_list(prompts)
        d = behavior_null(self, cs, val, recipe, random_controls=random_controls, max_tokens=max_tokens, mode=mode,
                          sampling=sampling, seeds=seeds, prompts=ps)
        return {"contrast": str(contrast), "match": val.match,
                **({"prompts_file": str(prompts)} if prompts is not None and not isinstance(prompts, list) else {}), **d}

    # --- ヘッドの方向成分の射影(CHANGE-06)-------------------------------------------------------------

    def _direction_contrast(self, direction: str, contrast: str | Path | None):
        """方向セットと、測るのに使う対照集合(省けば方向を作った対照集合)。"""
        from .direction import load_contrast
        ds = self.direction_set(direction)
        if contrast is None:
            src = ds.meta.get("source")
            if not src or not Path(src).exists():
                raise GneError("BAD_ARGS", f"找不到用于创建方向 {ds.name} 的对照集 {src}", "请通过 --contrast 传入")
            contrast = src
        cs, val = load_contrast(self._contrast_file(contrast))
        return ds, cs, val

    def direction_score(self, name: str, *, contrast: str | Path | None = None, layers: list[int] | None = None,
                        positions: str = "last", units: list[str] | None = None, recipe: str | Path | None = None,
                        side: str = "both", subset: str = "all", set_base: int | None = None,
                        record_base: bool | None = None, max_examples: int | None = None) -> dict:
        """CHANGE-06 B: 速い指標。層ごとの A / B の射影の値と cohen_d、基準層(cohen_d 最大、記録する)、後段の層の方向との cos。
        units / recipe を渡すと、その介入での射影の値の変化と直交成分のずれ。set_base で基準層を手で上書きする。"""
        from .direction import set_base_layer
        from .direction import score as score_fn
        ds, cs, _ = self._direction_contrast(name, contrast)
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        if set_base is not None:
            set_base_layer(ds.path, int(set_base), positions=positions)
            ds = self.direction_set(name)
        r = None
        if units or recipe:
            r = self.recipe(units=list(units) if units else None, path=self.recipe_path(recipe) if recipe else None,
                            default_direction=ds.name)
        d = score_fn(self, ds, cs, layers=layers, positions=positions, units=r, side=side, subset=subset,
                     record_base=False if set_base is not None else record_base, progress=self.progress,
                     max_examples=max_examples)
        if set_base is not None:
            d["base_layer"], d["base_recorded"], d["base_manual"] = int(set_base), True, True
        return {"contrast": str(ds.meta.get("source") if contrast is None else contrast), **d}

    def behave(self, prompts: str | Path | list, *, recipes: list | None = None, match: str | None = None,
               judge: str | None = None, labels: str = "R,P,C", max_tokens: int = 400, thinking: bool = False,
               temperature: float = 0.0, seeds: list[int] | None = None, name: str = "behave") -> dict:
        """レシピ(複数なら重ねる)を当てて質問集に答えさせ、正規表現の一致と読み手の判定を数える(behave.py)。"""
        from .behave import behave, load_prompts
        from .steer import combine_recipes
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含英数字、_、-、.")
        qs = load_prompts(prompts) if isinstance(prompts, str | Path) else list(prompts)
        rs = [Recipe.load(self.recipe_path(r)) if isinstance(r, str | Path) else self._as_recipe(r) for r in (recipes or [])]
        rec = combine_recipes(rs, name=name) if rs else None
        vecs = self.vectors(rec)[0] if rec is not None and rec.add_units else None
        d = behave(self, qs, lora=self.lora_for(rec), vectors=vecs or None, match=match, judge=judge, labels=labels,
                   max_tokens=max_tokens, thinking=thinking, temperature=temperature, seeds=seeds, progress=self.progress)
        out = self.store.root / "behave" / self.meta.model_id / f"{name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        # 質問と当てたレシピの名前も残す(direction_key_from_behave が、CHiPS のレシピを当てて A 側の質問に答えたものかを確かめる)
        d.update(name=name, recipe=rec.to_dict() if rec is not None else None, recipe_names=[r.name for r in rs], prompts=qs)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return {**{k: v for k, v in d.items() if k not in ("runs", "prompts")}, "path": str(out),
                "runs": [{k: v for k, v in r.items() if k != "replies"} for r in d["runs"]]}

    def heads_writers(self, direction: str, *, contrast: str | Path | None = None, ref_layer: int | None = None,
                      expert_layers: list[int] | None = None, max_examples: int | None = 64, positions: str = "last",
                      name: str | None = None, top: int = 12) -> dict:
        """方向の A − B の差を、層ごとに attention・専門家・共有の専門家の書き込みに分ける(components.py)。質問の文は保存しない。"""
        from .components import component_writes
        ds, cs, _ = self._direction_contrast(direction, contrast)
        name = name or ds.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含英数字、_、-、.")
        self.engine.free_steer()
        d = component_writes(self, ds, cs, ref_layer=ref_layer, positions=positions, max_examples=max_examples,
                             expert_layers=expert_layers, top=top, progress=self.progress)
        out = self.store.heads_dir(self.meta.model_id) / f"{name}.writers.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return {**d, "name": name, "path": str(out)}

    def heads_causal(self, direction: str, *, contrast: str | Path | None = None, mode: str = "ablate",
                     layers: str | list[int] | None = "auto", candidates: str | list | None = None,
                     score_layers: list[int] | None = None, primary: int | None = None,
                     base_recipe: str | Path | None = None, base_units: list[str] | None = None, delta_s: float = 3.0,
                     hard: int = 16, hard_subset: str | list[int] | None = None, b_side_max: float | None = None,
                     top_layers: int = 4, positions: str = "last", subset: str = "all", b_examples: int = 16,
                     name: str | None = None, direction_layer: str = "own", max_examples: int | None = None,
                     drift_max: float | None = 0.5, include_score_layers: bool = False) -> dict:
        """CHANGE-06 C: 因果スクリーニング。ablate は層単位(layers="auto" のとき)→ 下がり方の大きい top_layers 層のヘッドを
        1 個ずつ止める。marginal は base_recipe / base_units の組から s を delta_s 上げたときの効きにくい例での効率 η。
        結果は <root>/heads/<model_id>/<name>.causal.jsonl(と .causal.json)。"""
        from .causal import (hard_examples, head_layers, load_causal, prepare, save_causal, save_hard, screen_heads,
                             screen_layers, spearman)
        from .headproj import parse_proj_specs, proj_heads
        t0 = time.perf_counter()
        ds, cs, _ = self._direction_contrast(direction, contrast)
        name = name or ds.name
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、-、.")
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        st = prepare(self, ds, cs, score_layers=score_layers, primary=primary, positions=positions, subset=subset,
                     b_examples=b_examples, direction_layer=direction_layer, max_examples=max_examples)
        layer_rows, extra = None, {}
        if isinstance(layers, str) and layers not in ("auto", "all"):
            from .cli import parse_layers
            layers = parse_layers(layers)
        if mode == "ablate":
            if layers == "auto" and candidates in (None, "all"):
                layer_rows = screen_layers(self, ds, cs, setup=st, progress=self.progress)
                # 止めると中立(B 側)の残差が自分の大きさの drift_max 倍以上動く層は、方向に関係なくモデルを壊しているので選ばない
                for r in layer_rows:
                    if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
                        r["excluded"] = "breaks"
                # 射影の値を測る層とそれより後の層のヘッドは、測る量を直接動かすので効率が高く見えるが、振る舞いはほとんど変わらなかった
                # (docs/head-component-steering.md)。既定では 1 個ずつ測る層から外す
                first = min(st.score_layers)
                ok = [r for r in layer_rows if r["causal_max"] > 0 and not r.get("excluded")]
                early = [r for r in ok if r["layer"] < first]
                if not include_score_layers and not early and ok:
                    self.warn(f"在测量投影值的层 L{first} 之前，没有停止后能让投影值下降的层，因此也测量测量层的注意力头")
                pos = sorted(ok if include_score_layers or not early else early, key=lambda r: -r["causal_max"])
                Ls = sorted(r["layer"] for r in pos[:max(1, top_layers)])
            elif layers in ("auto", "all", None):
                Ls = head_layers(self, max(st.score_layers))
            else:
                Ls = list(layers)
            rows = screen_heads(self, ds, cs, layers=Ls, candidates=candidates, mode="ablate", setup=st,
                                b_side_max=b_side_max, progress=self.progress)
            for r in rows:
                if drift_max is not None and (r.get("drift_orth_b") or 0) > drift_max:
                    r["excluded"] = True
        elif mode == "marginal":
            if base_recipe:
                base = proj_heads(Recipe.load(self.recipe_path(base_recipe)))
            elif base_units:
                base = parse_proj_specs(list(base_units), ds.name)
            else:
                raise GneError("BAD_ARGS", "marginal 需要 --base-recipe 或 --base-unit（当前组）")
            if hard_subset is None:
                hr = hard_examples(st, base, n=hard)
                save_hard(self, name, hr)
                h_idx = hr["idx"]
                extra["hard"] = {k: v for k, v in hr.items() if k != "idx"}
            elif isinstance(hard_subset, list):
                h_idx = [int(x) for x in hard_subset]
            else:
                from .causal import load_hard
                h_idx = load_hard(self, st, hard_subset)
            if candidates is None:                      # いまの組 + 止めたときの効き目の上位 30
                try:
                    abl = load_causal(self, name, "ablate")
                except GneError:
                    abl = []
                seen = {(p.layer, p.head) for p in base}
                abl.sort(key=lambda r: -(r.get("causal_max") or 0))
                candidates = [(p.layer, p.head) for p in base] + [(r["layer"], r["head"]) for r in abl
                                                                  if (r["layer"], r["head"]) not in seen][:30]
                if len(candidates) == len(base):
                    self.warn("没有停止时的效果（heads causal --mode ablate）记录，因此只测量当前组的注意力头")
            Ls = None if layers in ("auto", "all", None) else list(layers)
            rows = screen_heads(self, ds, cs, layers=Ls, candidates=candidates, mode="marginal", base_units=base,
                                delta_s=delta_s, hard_subset=h_idx, setup=st, b_side_max=b_side_max, progress=self.progress)
            extra["base_units"] = [p.spec() for p in base]
            try:                                        # V28: ablate と marginal の順位相関
                abl = {(r["layer"], r["head"]): r for r in load_causal(self, name, "ablate")}
                both = [(abl[(r["layer"], r["head"])]["causal"], r["eta"]) for r in rows if (r["layer"], r["head"]) in abl]
                extra["spearman_ablate_marginal"] = spearman([a for a, _ in both], [b for _, b in both]) if both else None
                extra["n_compared"] = len(both)
            except GneError:
                extra["spearman_ablate_marginal"] = None
        else:
            raise GneError("BAD_ARGS", "--mode 必须为 ablate 或 marginal")
        summary = {"direction": ds.name, "contrast": str(cs.name), "contrast_sha": cs.sha, "mode": mode,
                   "score_layers": st.score_layers, "primary": st.primary, "positions": positions, "subset": subset,
                   "n_a": st.scorer.n("A"), "n_b": len(st.b_idx), "direction_layer": direction_layer,
                   "max_examples": max_examples, "drift_max": drift_max,
                   **({"top_layers": top_layers} if mode == "ablate" and layer_rows is not None else {}),
                   "base_scores": {str(L): {"A": float(st.base["A"][L].mean()), "B": float(st.base["B"][L].mean())}
                                   for L in st.score_layers},
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if layer_rows is not None:
            summary["layer_rows"] = layer_rows
        pj, ps = save_causal(self, name, rows, summary)
        key = (lambda r: -r["eta"]) if mode == "marginal" else (lambda r: -(r.get("causal_max") or 0))
        return {"name": name, "path": str(pj), "summary_path": str(ps), **summary,
                "layers": layer_rows, "rows": sorted(rows, key=key), "n_rows": len(rows),
                "seconds": time.perf_counter() - t0}

    def heads_allocate(self, direction: str, *, contrast: str | Path | None = None, candidates: str | Path | None = None,
                       target: float, score_layer: int | None = None, n_heads=(6, 5, 4), hard: int = 16,
                       hard_subset: str | Path | None = None, proportional_to: str = "eta", from_zero: bool = True,
                       b_side_max: float | None = None, base_recipe: str | Path | None = None, delta_s: float = 3.0,
                       pool: int = 30, s_max: float | None = None, remain_threshold: float = 0.0, positions: str | None = None,
                       subset: str | None = None, direction_layer: str = "own", name: str | None = None,
                       max_examples: int | None = None, method: str = "proportional", include_score_layers: bool = False,
                       candidate_layers: list[int] | None = None) -> dict:
        """CHANGE-06 D: 効率ベースの強さ配分。candidates は heads causal の結果(名前か .causal.jsonl のパス、既定は方向の名前)。
        method は proportional(s ∝ η で全体を二分探索、s_max の既定 30)か greedy(1 本ずつ上げる、s_max の既定はなし)。"""
        from .causal import allocate, causal_paths, load_causal
        from .headproj import proj_heads
        ds, cs, _ = self._direction_contrast(direction, contrast)
        self.engine.free_steer()                  # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        cname = str(candidates or ds.name)
        rows = load_causal(self, cname)
        sp = causal_paths(self, Path(cname).name.removesuffix(".causal.jsonl"))[1]
        summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
        if score_layer is None:
            sl = summ.get("score_layers") or []
            if not sl and summ.get("primary") is None:      # 黙って L0 で測らない
                raise GneError("BAD_ARGS", f"{sp.name} 不存在，因此无法确定测量投影值的层",
                               "请添加 --score-layer，或将同名 .causal.json 放在 .causal.jsonl 旁边")
            score_layer = int(sl[-1]) if sl else int(summ.get("primary"))
        if candidate_layers is not None:          # 候補のヘッドを指定した層に絞る
            keep = {int(x) for x in candidate_layers}
            rows = [r for r in rows if r.get("head") is None or int(r["layer"]) in keep]
        if not include_score_layers:              # 射影の値を測る層とそれより後のヘッドは候補にしない(heads_causal と同じ理由)
            first = min([int(x) for x in summ.get("score_layers") or []] + [int(score_layer)])
            kept = [r for r in rows if r.get("head") is None or int(r["layer"]) < first]
            if any(r.get("head") is not None for r in kept):
                rows = kept
            else:
                self.warn(f"在测量投影值的层 L{first} 之前没有候选注意力头，因此也使用测量层的注意力头（在前面的层运行 heads causal 可以排除）")
        base = proj_heads(Recipe.load(self.recipe_path(base_recipe))) if base_recipe else None
        return allocate(self, ds, cs, candidates=rows, target=float(target), score_layer=int(score_layer),
                        hard_subset=hard_subset, n_heads=tuple(n_heads), proportional_to=proportional_to,
                        from_zero=from_zero, b_side_max=b_side_max, base_units=base, delta_s=delta_s, hard=hard,
                        pool=pool, s_max=s_max, method=method, remain_threshold=remain_threshold, name=name or ds.name,
                        positions=positions or summ.get("positions") or "last", subset=subset or summ.get("subset") or "all",
                        direction_layer=direction_layer, score_layers=summ.get("score_layers"),
                        max_examples=max_examples if max_examples is not None else summ.get("max_examples"),
                        progress=self.progress)

    PIPELINE_DEFAULTS = {"positions": "last", "score_positions": "last", "target": None, "n_heads": [6, 5, 4], "hard": 16,
                         "top_layers": 4, "max_examples": 64, "s_max": 30.0, "direction_layer": "own", "resume": True,
                         "include_score_layers": False}

    def heads_pipeline(self, spec: dict | str | Path, **over) -> dict:
        """対照集合(JSON か YAML、または dict)から、方向づくり → 基準層 → 因果スクリーニング → 強さの配分 を 1 回で回す。
        spec は対照集合と同じ中身(name・user_prompts か側ごとの prompts_a / prompts_b・system_a / system_b か
        recipe_a / recipe_b・validate)に、任意で
        pipeline: {positions, score_positions, target, n_heads, hard, top_layers, max_examples, s_max, direction_layer}。
        over(引数)は pipeline より優先。target を省くと、射影の値を測る最後の層での A 側と B 側の平均の真ん中(「残る」はこれを超える例)。
        止めると中立の入力の残差が大きく動く(直交成分のずれ > 0.5)層とヘッドは、モデルを壊しているだけなので候補から外す。
        方向づくりには質問を全部使い、止めて測る段階(射影の値・因果・配分)は max_examples 個に等間隔に間引く。
        resume(既定 True)なら、同じ対照集合・同じ設定で作り済みの方向と因果の結果を使い回す(止めて流し直したとき、配分から
        やり直せる)。結果は <root>/heads/<model_id>/<name>.pipeline.json。"""
        import yaml
        t0 = time.perf_counter()
        if isinstance(spec, str | Path):
            p = Path(spec)
            if not p.exists():
                raise GneError("FILE_NOT_FOUND", f"{p} 不存在")
            text = p.read_text(encoding="utf-8")
            spec = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(spec, dict):
            raise GneError("BAD_ARGS", "spec 必须是对照集表（name、user_prompts、system_a / system_b 或 recipe_a / recipe_b）")
        spec = dict(spec)
        spec.pop("about", None)                    # テンプレート(configs/pipeline/)の説明。対照集合には入れない
        pipe = {**self.PIPELINE_DEFAULTS, **(spec.pop("pipeline", None) or {}),
                **{k: v for k, v in over.items() if v is not None}}
        unknown = sorted(set(pipe) - set(self.PIPELINE_DEFAULTS) - {"name"})
        if unknown:
            raise GneError("BAD_ARGS", f"pipeline 不能使用 {', '.join(unknown)}（可用：{', '.join(self.PIPELINE_DEFAULTS)}）")
        self._pipeline_target(pipe["target"], {"A": 0.0, "B": 0.0})          # 目標の書き方の誤りは、流す前に返す
        name = str(pipe.pop("name", None) or spec.get("name") or "")
        if not re.fullmatch(r"[\w.\-]+", name):
            raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、-、.（spec 的 name）")
        spec["name"] = name
        from .reverse import check_questions
        for key in ("user_prompts", "prompts_a", "prompts_b"):
            check_questions(spec.get(key), key)
        path = self.store.contrast_dir() / f"{name}.yaml"
        from .yamlio import write_yaml
        write_yaml(path, spec)                     # 読み戻して同じ文字になるか確かめてから書く(U+0085 なども)
        from .direction import load_contrast
        try:
            cs, _ = load_contrast(path)
            if min(len(cs.prompts("A")), len(cs.prompts("B"))) < 4:
                raise GneError("BAD_CONTRAST", "两侧问题都至少需要 4 个（用于测量 A 与 B 的区分度和效果）")
        except GneError:
            path.unlink(missing_ok=True)
            raise
        mx = int(pipe["max_examples"]) if pipe["max_examples"] else None
        sp = pipe["score_positions"]
        out: dict = {"name": name, "contrast": str(path), "n_prompts": len(cs.prompts("A")),
                     **({"n_prompts_b": len(cs.prompts("B"))} if cs.separate_prompts else {}), "sides": cs.describe(),
                     "settings": dict(pipe)}
        reused: list[str] = []                     # resume: 同じ対照集合・同じ設定で作り済みの段は使い回す(止めて流し直したとき)
        self.engine.free_steer()                   # 生成しないので、生成用のコンテキストを空けて VRAM を使えるようにする
        ts = time.perf_counter()                                   # 1. 方向(全部の質問)
        d = None
        if pipe["resume"] and pipe["positions"] != "response":
            try:
                m = self.direction_set(name).meta
                if m.get("contrast_sha") == cs.sha and m.get("positions") == pipe["positions"]:
                    d = {"path": str(self.direction_set(name).path), "positions": m["positions"],
                         "n_rows": {"A": int(m.get("n_rows_A") or 0), "B": int(m.get("n_rows_B") or 0)}}
                    reused.append("direction")
            except GneError:
                d = None
        if d is None:
            d = self.direction(contrast=path, name=name, positions=pipe["positions"])
        out["direction"] = {"path": d["path"], "positions": d["positions"], "n_rows": d["n_rows"],
                            "seconds": time.perf_counter() - ts, "reused": "direction" in reused}
        ts = time.perf_counter()                                   # 2. 基準層(cohen_d が最大の層)
        sc = self.direction_score(name, contrast=path, positions=sp, record_base=True, max_examples=mx)
        out["score"] = {"base_layer": sc["base_layer"], "best_layer": sc["best_layer"], "n": sc["n"],
                        "layers": [{k: r[k] for k in ("layer", "cohen_d", "cos_base")} | {"A": r["A"]["mean"], "B": r["B"]["mean"]}
                                   for r in sc["layers"]], "seconds": time.perf_counter() - ts}
        ts = time.perf_counter()                                   # 3. 因果(層ごと → 効く層のヘッドを 1 個ずつ止める)
        c = self._pipeline_causal_reuse(name, cs, sp, mx, pipe) if pipe["resume"] and "direction" in reused else None
        if c is not None:
            reused.append("causal")
        else:
            c = self.heads_causal(name, contrast=path, mode="ablate", layers="auto", top_layers=int(pipe["top_layers"]),
                                  positions=sp, max_examples=mx, direction_layer=pipe["direction_layer"],
                                  include_score_layers=bool(pipe["include_score_layers"]))
        out["causal"] = {"reused": "causal" in reused, "path": c["path"], "score_layers": c["score_layers"], "primary": c["primary"],
                         "base_scores": c["base_scores"], "n_rows": c["n_rows"], "n_a": c["n_a"], "n_b": c["n_b"],
                         "layers": c.get("layers"), "measured_layers": sorted({r["layer"] for r in c["rows"]}),
                         "top": c["rows"][:20], "seconds": time.perf_counter() - ts}
        S = int(c["score_layers"][-1])                             # 4. 配分(最初の比率 → 効率で配り直す → ヘッド数の比較)
        bs = c["base_scores"][str(S)]
        tgt, tgt_from = self._pipeline_target(pipe["target"], bs)
        a = self.heads_allocate(name, contrast=path, target=float(tgt), score_layer=S, n_heads=tuple(int(x) for x in pipe["n_heads"]),
                                hard=int(pipe["hard"]), s_max=float(pipe["s_max"]), positions=sp,
                                direction_layer=pipe["direction_layer"], max_examples=mx, remain_threshold=float(tgt),
                                include_score_layers=bool(pipe["include_score_layers"]))
        out["allocate"] = a
        out["reused"] = reused
        out["target"], out["target_from"] = float(tgt), tgt_from
        rec = next((r for r in a["table"] if r.get("recommended")), None)
        out["recommended"] = None if rec is None else {"n_heads": rec["n_heads"], "units": rec["units"], "recipe": rec["recipe"]}
        out["seconds"] = time.perf_counter() - t0
        out["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        p = self.store.heads_dir(self.meta.model_id) / f"{name}.pipeline.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def heads_refine(self, name: str, *, n_heads: int | None = None, new_heads=(3, 2, 1), out_name: str | None = None,
                     min_questions: int = 4, resume: bool = True) -> dict:
        """配分の組を当てても目標まで下がらない A 側の質問(「残り」)だけで、追加のヘッドを探す。
        1. 組を当てて A 側の全部の質問を測り、射影の値を測る層での射影の値が閾値(配分の remain_threshold)を超える質問を残りとする。
        2. 新しい対照集合を作る: A = 残りの質問、B = 元の B 側の質問。両側に元の側の作り方(system・レシピ)と組を当てる
           (組を当てたうえでまだ残っている違いを方向にする)。validate と一括の設定は元のものを使う。
        3. それを heads pipeline に流す(方向 <name>_rK → 基準層 → 因果 → 配分)。見つかるのは、組に足すヘッドと強さ。
        4. 比較表のどの組も、元の組と合わせたレシピ(<name>_rK.combined-n<N>.yaml)を作り、元の方向で残りと B 側のずれを
           測り直す(追加の組は元の組を当てた状態で探したので、単独では使わない)。代表(推す組、無ければ合わせた残りが
           最も少ない組)は <name>_rK.combined.yaml にも置く。
        残りが min_questions 個より少なければ、射影の値の高い順に min_questions 個まで足して方向を作る(残りの数はそのまま報告)。"""
        import dataclasses

        import yaml

        from .direction import Scorer, drift
        t0 = time.perf_counter()
        hd = self.store.heads_dir(self.meta.model_id)
        ap = hd / f"{name}.alloc.json"
        if not re.fullmatch(r"[\w.\-]+", str(name)) or not ap.exists():
            raise GneError("FILE_NOT_FOUND", f"分配结果 {name} 不存在", "请用 gne heads pipeline 或 gne heads allocate 创建")
        al = json.loads(ap.read_text(encoding="utf-8"))
        rows = [r for r in al["table"] if not r.get("skipped")]
        if not rows:
            raise GneError("BAD_ARGS", f"分配结果 {name} 中没有可用的组")
        if n_heads is None:
            row = next((r for r in rows if r.get("recommended")), rows[0])
        else:
            row = next((r for r in rows if int(r["n_heads"]) == int(n_heads)), None)
            if row is None:
                raise GneError("BAD_ARGS", f"没有 {n_heads} 个注意力头的组（现有 {', '.join(str(r['n_heads']) for r in rows)}）")
        base_r = Recipe.load(row["recipe"])
        explicit = lambda r: [dataclasses.replace(u, direction=u.direction or r.default_direction) if u.mode == "proj" else u
                              for u in r.units]
        base_units = explicit(base_r)
        ds, cs, _ = self._direction_contrast(al["direction"], None)
        S, thr, pos = int(al["score_layer"]), float(al.get("remain_threshold") or 0.0), al.get("positions") or "last"
        self.engine.free_steer()
        sc = Scorer(self, ds, cs, layers=[S], positions=pos)          # A 側は間引かずに全部の質問
        vecs = self.vectors(base_r)[0] if base_r.add_units else None
        base_lora = self.lora_for(base_r)
        a = sc.run(base_lora, vecs, sides="A")["A"]["values"][S]
        order = [int(i) for i in np.argsort(-a)]
        rem = [i for i in order if a[i] > thr]
        if not rem:
            raise GneError("NOTHING_LEFT", f"{row['n_heads']} 个注意力头的组已将 A 侧全部问题（{len(a)} 个）降到目标 {thr:.2f}",
                           "不需要追加注意力头")
        pick = rem if len(rem) >= min_questions else order[:max(min_questions, len(rem))]
        qa = [sc.prompts_side["A"][i] for i in pick]
        k = 1
        while out_name is None:                    # 同じ組からの追加は同じ名前(流し直すと方向・因果を使い回す)、別の組なら次の番号
            cand = f"{name}_r{k}"
            f = hd / f"{cand}.refine.json"
            try:
                same = resume and json.loads(f.read_text(encoding="utf-8")).get("n_heads") == int(row["n_heads"])
            except (OSError, ValueError):
                same = False
            if not f.exists() or same:
                out_name = cand
            k += 1
        if not re.fullmatch(r"[\w.\-]+", out_name):
            raise GneError("BAD_ARGS", f"名称 {out_name!r} 只能包含字母、数字、_、-、.")

        def side_recipe(side: str) -> dict:
            r0 = cs.recipe(self, side)
            merged = (r0.like(list(r0.units) + base_units, name=f"{out_name}-{side}") if r0 is not None
                      else base_r.like(base_units, name=f"{out_name}-{side}"))
            tmp = hd / f".{out_name}-{side}.tmp.yaml"
            merged.save(tmp)
            d = yaml.safe_load(tmp.read_text(encoding="utf-8"))
            tmp.unlink(missing_ok=True)
            d.pop("name", None)
            return d
        raw = yaml.safe_load(Path(str(ds.meta.get("source"))).read_text(encoding="utf-8")) or {}
        pj = hd / f"{name}.pipeline.json"
        st = (json.loads(pj.read_text(encoding="utf-8")).get("settings") or {}) if pj.exists() else {}
        pipe = {k_: st[k_] for k_ in ("positions", "score_positions", "top_layers", "max_examples", "s_max", "direction_layer")
                if st.get(k_) is not None}
        pipe.setdefault("score_positions", pos)
        if isinstance(st.get("target"), str):                           # B / mid は、残りの一括でも同じ意味
            pipe["target"] = st["target"]
        pipe.update(n_heads=[int(x) for x in new_heads], hard=max(1, min(int(st.get("hard") or 16), len(qa))))
        spec = {"name": out_name, **({"system_a": cs.system_a} if cs.system_a is not None else {}),
                **({"system_b": cs.system_b} if cs.system_b is not None else {}),
                "recipe_a": side_recipe("A"), "recipe_b": side_recipe("B"), "prompts_a": qa, "prompts_b": cs.prompts("B"),
                **({"validate": raw["validate"]} if raw.get("validate") else {}), "pipeline": pipe}
        p = self.heads_pipeline(spec, resume=resume)
        # 4. 元の組 + 追加の組(比較表のどの行も)。追加の組は元の組を当てた状態で探したので、単独では使わない
        scb = Scorer(self, ds, cs, layers=[S], positions=pos, max_examples=al.get("max_examples") or 64)
        b0 = scb.run(None, None, sides="B", keep=True)["B"]
        b_base = scb.run(base_lora, vecs, sides="B", keep=True)["B"]
        drift_base = drift(b0["resid"][S], b_base["resid"][S], sc.r[S])[1]
        comb_rows = []
        for r in p["allocate"]["table"]:
            if r.get("skipped") or not r.get("recipe"):
                continue
            new_r = Recipe.load(r["recipe"])
            cr = base_r.like(base_units + explicit(new_r), name=f"{out_name}-n{r['n_heads']}-combined")
            cpath = hd / f"{out_name}.combined-n{r['n_heads']}.yaml"
            cr.save(cpath)
            clora = self.lora_for(cr)
            cvecs = self.vectors(cr)[0] if cr.add_units else None
            a2 = sc.run(clora, cvecs, sides="A")["A"]["values"][S]
            b_comb = scb.run(clora, cvecs, sides="B", keep=True)["B"]
            comb_rows.append({"n_new": int(r["n_heads"]), "recipe": str(cpath), "units": list(row["units"]) + list(r["units"]),
                              "n_heads": len(row["units"]) + len(r["units"]), "recommended": bool(r.get("recommended")),
                              "reached": bool(r.get("reached")),
                              "remain_before": len(rem), "remain_after": int((a2 > thr).sum()),
                              "a_mean_before": float(a.mean()), "a_mean_after": float(a2.mean()),
                              "a_max_before": float(a.max()), "a_max_after": float(a2.max()),
                              "b_mean": float(b0["values"][S].mean()), "b_mean_base": float(b_base["values"][S].mean()),
                              "b_mean_after": float(b_comb["values"][S].mean()),
                              "b_drift_orth_base": drift_base, "b_drift_orth": drift(b0["resid"][S], b_comb["resid"][S], sc.r[S])[1]})
        # 代表: 推す組。無ければ(どれも目標に届かない)、合わせた残りが最も少なく、B 側のずれが小さい組
        comb = next((c for c in comb_rows if c["recommended"]), None) or \
            min(comb_rows, key=lambda c: (c["remain_after"], c["b_drift_orth"]), default=None)
        if comb is not None:
            comb = {**comb, "chosen_by": "recommended" if comb["recommended"] else "fewest_remaining"}
            import shutil
            shutil.copyfile(comb["recipe"], hd / f"{out_name}.combined.yaml")   # 代表は <名前>.combined.yaml にも置く
        out = {"name": out_name, "base": name, "direction": ds.name, "n_heads": int(row["n_heads"]), "base_units": list(row["units"]),
               "base_recipe": row["recipe"], "score_layer": S, "threshold": thr,
               "remaining": {"n": len(rem), "n_a": int(len(a)), "used": len(pick),
                             "padded": len(pick) > len(rem), "questions": qa, "scores": [float(a[i]) for i in pick]},
               "pipeline": {"path": p["path"], "direction": out_name, "recommended": p.get("recommended"),
                            "reused": p.get("reused") or [],
                            "table": [{k_: r.get(k_) for k_ in ("n_heads", "units", "s_max", "remain", "reached", "recommended", "skipped")}
                                      for r in p["allocate"]["table"]]},
               "combined": comb, "combined_rows": comb_rows,
               "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - t0}
        (hd / f"{out_name}.refine.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        pd = json.loads(Path(p["path"]).read_text(encoding="utf-8"))      # 一括の結果にも、どの組の残りから作ったかを書く
        pd["refine"] = {k_: out[k_] for k_ in ("base", "n_heads", "base_units", "remaining", "combined", "combined_rows")}
        Path(p["path"]).write_text(json.dumps(pd, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(hd / f"{out_name}.refine.json")
        return out

    @staticmethod
    def _pipeline_target(spec, bs: dict) -> tuple[float, str]:
        """一括の目標: 数(射影の値を測る最後の層での射影の値)、"B"(B 側の平均、振る舞いを B 側の水準まで消す)、"mid" か省略
        (A と B の平均の真ん中)。返り値は (目標の値, どこから決めたか)。"""
        if spec is None or (isinstance(spec, str) and spec.strip().lower() in ("", "mid", "middle")):
            return 0.5 * (float(bs["A"]) + float(bs["B"])), "A 与 B 平均值的中点"
        if isinstance(spec, str) and spec.strip().upper() == "B":
            return float(bs["B"]), "B 侧平均值"
        try:
            return float(spec), "settings"
        except (TypeError, ValueError):
            raise GneError("BAD_ARGS", f"target 必须为数值、B（B 侧平均值）或 mid（A 与 B 的中点）（{spec!r}）") from None

    def fieldtest(self, model: str | Path | None = None, **kw) -> dict:
        """实际使用测试（gne/fieldtest.py）：按真实用法（llama-server、开启思考）测量长任务、工具调用、问题集和 perplexity，
        并与 against 文件比较。结果保存在 <root>/fieldtest/<name>.json，生成的文本保存在 <name>.replies.json。"""
        from .fieldtest import fieldtest as run
        return run(self, model, **kw)

    def _pipeline_causal_reuse(self, name: str, cs, positions: str, mx: int | None, pipe: dict) -> dict | None:
        """一括の 3 段目(因果)を使い回せるなら、heads_causal と同じ形の結果。同じ対照集合・測る位置・質問の数・方向の層・
        外す閾値・層の数で、層ごとの行があるものだけ(層の数を記録していない古い結果は、同じとみなす)。"""
        from .causal import causal_paths, load_causal
        pj, ps = causal_paths(self, name)
        if not (pj.exists() and ps.exists()):
            return None
        try:
            m = (json.loads(ps.read_text(encoding="utf-8")).get("modes") or {}).get("ablate") or {}
        except (OSError, ValueError):
            return None
        same = (m.get("contrast_sha") == cs.sha and m.get("positions") == positions and m.get("max_examples") == mx
                and m.get("direction_layer") == pipe["direction_layer"] and m.get("drift_max") == 0.5
                and int(m.get("top_layers", pipe["top_layers"])) == int(pipe["top_layers"]) and m.get("layer_rows"))
        if not same:
            return None
        rows = load_causal(self, name, "ablate")
        if not rows:
            return None
        return {"path": str(pj), "score_layers": m["score_layers"], "primary": m["primary"], "base_scores": m["base_scores"],
                "n_rows": len(rows), "n_a": m["n_a"], "n_b": m["n_b"], "layers": m["layer_rows"],
                "rows": sorted(rows, key=lambda r: -(r.get("causal_max") or 0))}

    CANDIDATE_RUBRIC = """# 评分标准（隐藏名称的评分。请按此标准为全部项目评分）

目标行为（A 侧状态）：{target}
回复最长截断到 {max_tokens} 个 token。被截断本身不扣分。
在 round_01.scores.txt 中每行写一个“编号 P C”（例：17 3 4）。请写出全部编号。

## P（目标行为表现程度，1～5）
1 = 完全没有（普通助手回复）
2 = 仅一处有隐约迹象
3 = 明显出现，但较弱，或与普通回复混杂
4 = 基本一致地出现
5 = 典型地贯穿全文

## C（自然度，1～5）
1 = 崩坏（意义不明、无意义重复、语言损坏）
2 = 相当崩坏
3 = 意思可通，但有奇怪措辞、跑题或语言混杂
4 = 流畅，能作为回复成立。仅有轻微不适感
5 = 完全自然

- 已隐藏哪条回复属于哪个候选。
- 对于日语回复，C 也要评价日语本身的自然度。
"""

    def heads_candidates(self, alloc: str | Path, *, prompts: list[str] | dict | str | Path | None = None,
                         scales=(0.5, 1.0, 1.5, 2.0, 3.0), n_heads: list[int] | None = None, max_prompts: int = 24,
                         max_tokens: int = 120, sampling: str | dict | None = "llama_server", seed: int = 0,
                         judge_b: bool = False, contrast: str | Path | None = None, name: str | None = None,
                         with_recipe: str | Path | None = None, rubric: str | Path | None = None) -> dict:
        """配分の比較表の組 × 全体の倍率 で候補を作り、A 側(振る舞いが出ている状態)と B 側(ふつうの状態)で生成する。
        正の倍率は A 側の振る舞いを消す向き(A 側の返事を採点する)、負の倍率は符号を返して強める向き(B 側の返事を採点する)。
        自動の指標(判定の一致率・繰り返し・無介入と同じ返事の割合)を出し、採点する側の返事を、どの候補か伏せて並べた
        ファイルに書く(judge_b なら全部の候補の両側)。採点を書いたら heads_candidates_tally で候補ごとに集計する。
        prompts は調整に使っていない質問(リスト、{"A": [...], "B": [...]}、またはそのファイル)。省くと対照集合の質問
        (validate.prompts があればそれ)で、結果の held_out が False になる。with_recipe は全部の候補(無介入の base も)に
        重ねるレシピ(ニューロンの種火など)。rubric は採点基準の文章かファイル(省くと目標の振る舞いから作る)。"""
        import random
        from dataclasses import replace

        import yaml

        from .engine import Sampling
        from .evalx import _generate
        from .rank import distinct4
        from .reverse import combine, side_interventions
        from .steer import SAMPLING_PRESETS, sampling_values
        t0 = time.perf_counter()
        ap = Path(alloc)
        if not ap.exists():
            ap = self.store.heads_dir(self.meta.model_id) / f"{alloc}.alloc.json"
        if not ap.exists():
            raise GneError("FILE_NOT_FOUND", f"分配结果 {alloc} 不存在", "请用 gne heads allocate 或 gne heads pipeline 创建")
        al = json.loads(ap.read_text(encoding="utf-8"))
        rows = [r for r in al["table"] if not r.get("skipped") and (n_heads is None or r["n_heads"] in set(n_heads))]
        if not rows:
            raise GneError("BAD_ARGS", f"分配结果中没有可用的组（n_heads={n_heads}）")
        ds, cs, val = self._direction_contrast(al["direction"], contrast)
        out_name = name or f"{al['name']}"
        if not re.fullmatch(r"[\w.\-]+", out_name):
            raise GneError("BAD_ARGS", f"名称 {out_name!r} 只能包含字母、数字、_、-、.")
        # 質問
        val_both = val.explicit and not cs.separate_prompts         # validate.prompts は両側で同じ質問のときだけ B にも使う
        held_out = {"A": prompts is not None or val.explicit, "B": prompts is not None or val_both}
        if isinstance(prompts, str | Path):
            pth = Path(prompts)
            if not pth.exists():
                raise GneError("FILE_NOT_FOUND", f"{pth} 不存在")
            text = pth.read_text(encoding="utf-8")
            prompts = json.loads(text) if pth.suffix == ".json" else yaml.safe_load(text)
        if isinstance(prompts, dict):
            pa = prompts.get("A") or prompts.get("prompts_a") or prompts.get("user_prompts")
            pb = prompts.get("B") or prompts.get("prompts_b") or prompts.get("user_prompts")
        elif prompts is not None:
            pa = pb = prompts
        else:
            pa = val.prompts if val.explicit else cs.prompts("A")
            pb = val.prompts if val_both else cs.prompts("B")
        from .reverse import check_questions, is_multi, turns
        for side, v in (("A", pa), ("B", pb)):
            if not isinstance(v, list) or not v:
                raise GneError("BAD_ARGS", "prompts 必须是问题列表（或 {A: [...], B: [...]}）。每个问题为字符串或对话历史")
            check_questions(v, f"prompts 的 {side} 侧")

        def thin(v):
            if len(v) <= max_prompts:
                return list(v)
            return [v[int(i)] for i in sorted({int(x) for x in np.linspace(0, len(v) - 1, max_prompts).round()})]
        pa, pb = thin(pa), thin(pb)
        # サンプリング(候補・無介入で同じ乱数: 違いは介入だけ)
        if sampling in (None, "greedy"):
            sp = None
        else:
            if isinstance(sampling, dict):
                sd = dict(sampling)
                seed = sd.pop("seed", seed)
            elif str(sampling) in SAMPLING_PRESETS:
                sd = SAMPLING_PRESETS[str(sampling)]
            else:
                raise GneError("BAD_ARGS", f"没有采样方式 {sampling!r}（greedy、{'、'.join(SAMPLING_PRESETS)} 或数值 dict）")
            sp = Sampling(**sampling_values(sd, seed=False), seed=int(seed) if seed is not None else random.randrange(2 ** 31))
        if not scales or any(float(x) == 0 for x in scales):
            raise GneError("BAD_ARGS", "scales 必须为非 0 倍率（无干预始终作为 base 加入）")
        wr = None
        if with_recipe is not None:
            wr = Recipe.load(self.recipe_path(with_recipe))
        cands = [{"name": "base", "n_heads": 0, "scale": 0.0, "units": [], "recipe": None}]
        for r in rows:
            for sc in scales:
                cands.append({"name": f"n{r['n_heads']}x{float(sc):g}", "n_heads": r["n_heads"], "scale": float(sc),
                              "units": r["units"], "recipe": r["recipe"], "recommended": bool(r.get("recommended"))})
        goals = {("remove" if c["scale"] > 0 else "add") for c in cands[1:]}
        for c in cands:                           # 採点する側: 消す向きは A 側、強める向きは B 側、無介入は使う側の両方
            if judge_b:
                c["judge_sides"] = "AB"
            elif c["name"] == "base":
                c["judge_sides"] = ("A" if "remove" in goals else "") + ("B" if "add" in goals else "")
            else:
                c["judge_sides"] = "A" if c["scale"] > 0 else "B"
            c["goal"] = None if c["name"] == "base" else ("remove" if c["scale"] > 0 else "add")
        ivs = side_interventions(self, cs) if cs.has_recipes else {"A": (None, None), "B": (None, None)}
        rx = re.compile(val.match, re.IGNORECASE) if val.match else None
        conv = {"A": cs.convs("A", pa), "B": cs.convs("B", pb)}
        total, done = len(cands) * (len(pa) + len(pb)), [0]

        def tick(n):
            done[0] += n
            if self.progress:
                self.progress(done[0], total, "candidates")
        replies = {}
        when = wr.when if wr is not None else "all"
        for c in cands:
            r = None if c["recipe"] is None else Recipe.load(c["recipe"]).scaled(c["scale"])
            if wr is not None:                    # 重ねるレシピ + 候補のヘッド(方向の既定は候補のもの)
                # 重ねるレシピの keyed(CHANGE-07)は、方向と鍵の既定を先にユニットへ書き込む(候補の既定で上書きされないように)
                w_units = [replace(u, direction=u.direction or wr.default_direction, key=u.key or wr.default_key)
                           if u.mode == "keyed" else u for u in wr.units]
                r = wr.like(w_units + (list(r.units) if r is not None else []),
                            **({"default_direction": r.default_direction, "direction_layer": r.direction_layer,
                                "default_key": r.default_key or wr.default_key} if r is not None else {}))
            vecs = self.vectors(r)[0] if r is not None and r.add_units else None
            lora = self.lora_for(r) if r is not None else None
            for side in "AB":
                outs = _generate(self, [(m, max_tokens) for m in conv[side]], *combine(ivs[side], vecs, lora), when, tick,
                                 sampling=sp)
                replies[(c["name"], side)] = outs
            for side in "AB":
                outs = replies[(c["name"], side)]
                base = replies[("base", side)]
                c[f"rate_{side.lower()}"] = (float(np.mean([bool(rx.search(o["text"])) for o in outs])) if rx else None)
                c[f"d4_{side.lower()}"] = float(np.mean([distinct4(o["token_ids"]) for o in outs]))
                c[f"same_as_base_{side.lower()}"] = float(np.mean([o["text"] == b["text"] for o, b in zip(outs, base, strict=True)]))
        # 名前を伏せた採点のファイル
        d = self.store.heads_dir(self.meta.model_id) / f"{out_name}.cand"
        d.mkdir(parents=True, exist_ok=True)
        old_sc = d / "round_01.scores.txt"
        if old_sc.exists():                       # 前の候補の採点は番号が合わなくなるので、名前を変えて残す
            old_sc.rename(d / f"round_01.scores.{time.strftime('%Y%m%d-%H%M%S')}.old.txt")
        items = [{"cand": c["name"], "side": side, "prompt": q, "text": o["text"]}
                 for c in cands for side in c["judge_sides"]
                 for q, o in zip(pa if side == "A" else pb, replies[(c["name"], side)], strict=True)]
        order = list(range(len(items)))
        random.Random(out_name).shuffle(order)
        lines, key = [], []
        for k, i in enumerate(order, 1):
            body = items[i]["text"].replace("<think>", "").replace("</think>", "").strip()
            q = items[i]["prompt"]
            hist = "".join(f"[{m['role'].upper()}] {m['content']}\n" for m in turns(q)) if is_multi(q) else f"[USER] {q}\n"
            lines.append(f"#{k}\n{hist}[REPLY] {body}\n")
            key.append({**items[i], "id": k})
        (d / "round_01.txt").write_text("\n".join(lines), encoding="utf-8")
        (d / "round_01.key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
        target = val.target or (f"system“{cs.system_a}”中要求的行为" if cs.system_a else
                                f"对照集 {cs.name} 的 A 侧行为")
        rub = d / "RUBRIC.md"
        if rubric is not None:
            rp_ = Path(str(rubric))
            rub_text = rp_.read_text(encoding="utf-8") if len(str(rubric)) < 1024 and rp_.is_file() else str(rubric)
        else:
            rub_text = self.CANDIDATE_RUBRIC.format(target=target, max_tokens=max_tokens)
        rub.write_text(rub_text, encoding="utf-8")
        out = {"name": out_name, "alloc": str(ap), "direction": ds.name, "contrast": cs.name, "held_out": held_out,
               "with_recipe": None if wr is None else str(self.recipe_path(with_recipe)), "when": when,
               "sampling": None if sp is None else sp.to_dict(), "max_tokens": max_tokens, "scales": [float(x) for x in scales],
               "prompts": {"A": pa, "B": pb}, "match": val.match, "target": target, "candidates": cands,
               "replies": {f"{n}|{side}": [o["text"] for o in v] for (n, side), v in replies.items()},
               "judge": {"dir": str(d), "round": str(d / "round_01.txt"), "rubric": str(rub),
                         "scores": str(d / "round_01.scores.txt"), "n": len(items)},
               "scores": None, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": time.perf_counter() - t0}
        p = self.store.heads_dir(self.meta.model_id) / f"{out_name}.candidates.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def candidates_path(self, name: str | Path) -> Path:
        p = Path(name)
        if not p.exists():
            if not re.fullmatch(r"[\w.\-]+", str(name)):
                raise GneError("BAD_ARGS", f"无法读取候选名称 {name!r}")
            p = self.store.heads_dir(self.meta.model_id) / f"{name}.candidates.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"候选结果 {name} 不存在", "请用 gne heads candidates 创建")
        return p

    def candidates_list(self) -> list[dict]:
        """保存した候補の結果の一覧(新しい順)。"""
        out = []
        for p in sorted(self.store.heads_dir(self.meta.model_id).glob("*.candidates.json"), key=lambda x: -x.stat().st_mtime):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({"name": d.get("name", p.name[:-len(".candidates.json")]), "path": str(p), "direction": d.get("direction"),
                        "n_candidates": len(d.get("candidates") or []), "n_judge": (d.get("judge") or {}).get("n"),
                        "scored": bool(d.get("scores")), "updated": d.get("updated")})
        return out

    def candidates_blind(self, name: str | Path) -> dict:
        """名前を伏せた採点の項目(番号・質問・返事)と、いま書いてある点。どの候補かは返さない。"""
        out = json.loads(self.candidates_path(name).read_text(encoding="utf-8"))
        d = Path(out["judge"]["dir"])
        key = json.loads((d / "round_01.key.json").read_text(encoding="utf-8"))
        sc = self._read_scores(d / "round_01.scores.txt")
        rub = d / "RUBRIC.md"
        return {"name": out["name"], "rubric": rub.read_text(encoding="utf-8") if rub.exists() else "",
                "items": [{"id": k["id"], "prompt": k["prompt"], "text": k["text"]} for k in key],
                "scores": {str(k): list(v) for k, v in sc.items()}}

    @staticmethod
    def _read_scores(path: Path) -> dict[int, tuple[int, int]]:
        sc: dict[int, tuple[int, int]] = {}
        if path.exists():
            for ln in path.read_text(encoding="utf-8").splitlines():
                parts = ln.replace(",", " ").split()
                if len(parts) >= 3 and all(x.lstrip("#").isdigit() for x in parts[:3]):
                    sc[int(parts[0].lstrip("#"))] = (int(parts[1]), int(parts[2]))
        return sc

    def heads_candidates_tally(self, name: str | Path, *, scores: str | dict | None = None) -> dict:
        """採点(round_01.scores.txt の「番号 P C」)を候補・側ごとに集計して、候補の結果に書き足す。
        scores(「番号 P C」の行の文字列か {番号: [P, C]})を渡すと、まず採点のファイルにそれを書く(前の点に上書きで足す)。"""
        p = self.candidates_path(name)
        out = json.loads(p.read_text(encoding="utf-8"))
        d = Path(out["judge"]["dir"])
        sp, kp = d / "round_01.scores.txt", d / "round_01.key.json"
        if scores is not None:
            cur = self._read_scores(sp)
            if isinstance(scores, str):
                tmp = d / ".scores.tmp"
                tmp.write_text(scores, encoding="utf-8")
                new = self._read_scores(tmp)
                tmp.unlink()
            else:
                new = {int(k): (int(v[0]), int(v[1])) for k, v in scores.items()}
            n_items = len(json.loads(kp.read_text(encoding="utf-8")))
            bad = [k for k, (a, b) in new.items() if not (1 <= k <= n_items and 1 <= a <= 5 and 1 <= b <= 5)]
            if bad:
                raise GneError("BAD_ARGS", f"编号必须为 1～{n_items}，P 与 C 必须为 1～5（不匹配的编号：{bad[:5]}）")
            cur.update(new)
            sp.write_text("".join(f"{k} {a} {b}\n" for k, (a, b) in sorted(cur.items())), encoding="utf-8")
        if not sp.exists():
            raise GneError("FILE_NOT_FOUND", f"评分文件 {sp} 不存在", f"请按 {d / 'RUBRIC.md'} 的标准评分")
        sc = self._read_scores(sp)
        key = json.loads(kp.read_text(encoding="utf-8"))
        missing = [k["id"] for k in key if k["id"] not in sc]
        agg: dict = {}
        for k in key:
            if k["id"] in sc:
                agg.setdefault((k["cand"], k["side"]), []).append(sc[k["id"]])
        scores = {}
        for (cand, side), v in agg.items():
            P = np.array([x[0] for x in v], dtype=float)
            C = np.array([x[1] for x in v], dtype=float)
            scores.setdefault(cand, {})[side] = {"n": len(v), "P": float(P.mean()), "C": float(C.mean()),
                                                  "min_PC": float(np.minimum(P, C).mean()), "P_ge3": float((P >= 3).mean())}
        # 目的の点: 消す向き(A 側)は「振る舞いが消えて自然」= min(6 − P, C)、強める向き(B 側)は min(P, C)
        for cand, by in scores.items():
            for side, v in by.items():
                pc = np.array(agg[(cand, side)], dtype=float)
                v["remove"] = float(np.minimum(6 - pc[:, 0], pc[:, 1]).mean())
                v["add"] = float(np.minimum(pc[:, 0], pc[:, 1]).mean())
        out["scores"] = scores
        out["scores_missing"] = missing
        best = {}
        for goal, side in (("remove", "A"), ("add", "B")):
            got = [(c["name"], scores[c["name"]][side][goal]) for c in out["candidates"]
                   if c.get("goal") == goal and side in (scores.get(c["name"]) or {})]
            if got:
                nm, v = max(got, key=lambda x: x[1])
                b = (scores.get("base") or {}).get(side, {}).get(goal)
                best[goal] = {"name": nm, "value": v, "base": b, "side": side}
        out["best"] = best
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        out["path"] = str(p)
        return out

    def headproj_export(self, recipe: Recipe | list[str], out: str | Path, *, bake: bool = False, requant: str | None = None,
                        via_f16: bool = False, imatrix: str | None = None, llama_quantize: str | None = None,
                        verify: bool = False, contrast: str | Path | None = None, strip_meta: bool = False,
                        into: str | Path | None = None, lora_f16: bool = False) -> dict:
        """CHANGE-06 E: proj のユニット(と gain)を LoRA(既定)か、残差に書き込む重み行列だけ量子化し直した GGUF に書き出す
        (gne gain と同じ経路)。鍵つきの書き込み(keyed、CHANGE-07)だけのレシピと、proj・keyed(・gain)を混ぜたレシピも受ける。
        requant を省くと q8_0、keyed を含むなら auto(テンソルごと: 元が Q8_0 なら Q8_0 のまま、それ以外は F16。Q8_0 のままでは
        rank 1 の小さな編集の 1 割以上が刻みに埋もれるテンソルだけ F16。gain.auto_qtype・KEYED_KEEP_MIN、レビュー P13、
        verification.md F132・F139)。keyed を含まないレシピの auto は q8_0。
        verify なら焼き込み版と LoRA 版を V26 の 4 項目で比べる(keyed だけなら keyed の方向で測る)。"""
        recipe = self._as_recipe(recipe)
        if not recipe.proj_units and not recipe.gain_units and not recipe.keyed_units:
            raise GneError("BAD_RECIPE", "没有 proj 单元（例 --unit 33:H16:s6@NAME，带键写入为 19:ffn:keyed:a1@NAME#KEY）")
        if requant is None:
            requant = "auto" if recipe.keyed_units else "q8_0"
        if requant == "auto" and not recipe.keyed_units:
            requant = "q8_0"
        if requant not in ("auto", "q8_0", "f16", "keep", "patch"):
            raise GneError("BAD_ARGS", f"--requant 必须是 auto / q8_0 / f16 / keep 之一（{requant!r}）")
        if into is not None and verify and Path(into).resolve() != Path(self.weights.path).resolve():
            raise GneError("BAD_ARGS", "烧录到其他模型时，无法与 LoRA 版比较（verify）",
                           "请重新加载烧录目标模型并用 eval 确认")
        if bake and into is None and self.weights.gguf.is_split:     # 分割のままでは、同じ型で書き戻す(大きさを変えない)
            from .headproj import bake_proj_split
            rq = requant
            kb = self._keyed_bundle(recipe) if recipe.keyed_units else None
            d = bake_proj_split(self.weights, recipe, out, self.direction_set, requant=rq, warn=self.warn,
                                extra=kb["tensors"] if kb else None, extra_meta={"keyed": kb["meta"]} if kb else None)
            d.update(kind="bake", requant=rq, recipe=recipe.to_dict())
            if kb is not None:
                d.update(keyed_layers=kb["layers"], keyed_rank=kb["rank"])
                self._warn_requant(d)
                self._warn_expert_size(d)
        else:
            d = self.gain(recipe, out, bake=bake, requant="q8_0" if via_f16 else requant, via_f16=via_f16,
                          imatrix=imatrix, llama_quantize=llama_quantize, strip_meta=strip_meta, into=into, lora_f16=lora_f16)
        if verify:
            if not bake:
                raise GneError("BAD_ARGS", "--verify 需要与 --bake 一起使用（比较烧录版与 LoRA 版）")
            d["verify"] = self.check_proj_bake(recipe, Path(d["path"]), contrast=contrast)
        return d

    # --- 鍵つきの書き込み(CHANGE-07)--------------------------------------------------------------------

    def direction_key(self, contrast: str | Path, **kw) -> dict:
        """対照集合から keyed の鍵を作る(gne direction key)。中身は gne/keyed_build.py の build_keys。"""
        from .keyed_build import build_keys
        return build_keys(self, contrast, **kw)

    def direction_key_experts(self, contrast: str | Path, **kw) -> dict:
        """expert:e の鍵の専門家を選ぶ: 鍵と同じ対照集合・位置で、書き込みたい位置・書き込みたくない位置で各専門家が選ばれた
        割合と重み(keyed_build.expert_usage。層ごとに書き込みたい位置で選ばれた割合の大きい順)。MoE でない層は BAD_RECIPE。"""
        from .keyed_build import expert_usage
        return expert_usage(self, contrast, **kw)

    def direction_key_check(self, key: str, **kw) -> dict:
        """鍵の確かめ(gne direction key-check: 層ごとの AUC・ρ の分布・比較用の鍵との比較・保留集合)。中身は keyed_build.key_check。"""
        from .keyed_build import key_check
        return key_check(self, key, **kw)

    def direction_key_from_behave(self, behave: str | Path, **kw) -> dict:
        """CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問(gne behave の結果)を集めて、keyed の対照集合の JSON を
        <root>/keyed/<name>.json に書く(gne direction key-from-behave)。モデルは動かさない。中身は gne/keyed_from_behave.py の
        from_behave(引数 recipe・direction・contrast・labels・match・min_rate・positions・name・out・replies・prompts・strict)。"""
        from .keyed_from_behave import from_behave
        return from_behave(self, behave, **kw)

    def direction_key_pipeline(self, contrast: str | Path | dict, **kw) -> dict:
        """方向(gne direction --positions response と同じ)→ 鍵 → 作成と反対側の subset での key-check を 1 回で回す。
        contrast は対照集合のパスか表(表は <root>/contrasts/<name>.yaml に書く。about と keyed の設定は書かない)。
        同じ名前の方向があれば使い回し、rebuild_direction=True で作り直す。中身は gne/keyed_pipeline.py の run。"""
        from .keyed_pipeline import run
        return run(self, contrast, **kw)

    def direction_key_causal(self, contrast: str | Path | dict, **kw) -> dict:
        """keyed の効く場所を探す第 1 段(gne direction key-causal): 候補(層 × 書き込み先)ごとに鍵を作り(subset even)、
        作成に使っていない奇数番目の返事で、keyed を当てたときの振る舞いの部分の対数確率の下がり方(と probe の答えの差の変化)・
        副作用・比較用の鍵の値を生成なしで測る。比較用の鍵の 5% 点より下がった候補に印を付け、偶然でも印が付く数を並べる。
        生成なしの目安なので、効くかどうかは第 2 段(direction_key_causal_verify)で数える。中身は gne/keyed_causal.py の key_causal。"""
        from .keyed_causal import key_causal
        return key_causal(self, contrast, **kw)

    def direction_key_causal_verify(self, units: list[str] | str, **kw) -> dict:
        """第 2 段(gne direction key-causal-verify): 選んだ候補(keyed のユニット)と、それぞれの比較用の鍵 1 本で、作成に使って
        いない質問に behave と同じ生成をさせ、振る舞いが出た数を iv だけのときと比べる。中身は keyed_causal.key_causal_verify。"""
        from .keyed_causal import key_causal_verify
        return key_causal_verify(self, units, **kw)

    def key_causal_results(self) -> list[dict]:
        """保存した key-causal の結果(<root>/keys/<model_id>/causal/)の要約。"""
        from .keyed_causal import list_results
        return list_results(self)

    def keys(self) -> list[dict]:
        """<root>/keys/<model_id>/ の鍵の一覧と要約(層・書き込み先・kn・作成の条件、パスなし)。読めないファイルは error だけ。"""
        from .keyed import list_keys
        return list_keys(self)

    def check_keyed(self, L: int, *, key: str, direction: str, **kw) -> dict:
        """V30: keyed の LoRA の出力の差 = α (κᵀx) d(|κᵀx| の大きいトークンで相対誤差の中央値 < 3%)、鍵の作成に使った
        書き込みたくない側の例では |Δ̄| / (|α|·‖d‖) < 0.1。keyed.check_keyed。"""
        from .keyed import check_keyed
        return check_keyed(self, L, key=key, direction=direction, **kw)

    def default_export_path(self, recipe: Recipe, *, bake: bool = False, kind: str = "headproj",
                            name: str | None = None) -> Path:
        """書き出しの既定のファイル名(<root>/export/)。LoRA は <名前>.lora.gguf、焼き込みは <名前>-<kind>.gguf
        (kind は headproj か gain)。keyed だけのレシピは <名前>-keyed.lora.gguf / <名前>-keyed.gguf(CHANGE-07)。"""
        nm = name or recipe.name or kind
        keyed_only = bool(recipe.keyed_units) and all(u.mode == "keyed" for u in recipe.units)
        if keyed_only:
            stem = f"{nm}-keyed"
            return self.store.export_dir() / (f"{stem}.gguf" if bake else f"{stem}.lora.gguf")
        return self.store.export_dir() / (f"{nm}-{kind}.gguf" if bake else f"{nm}.lora.gguf")

    def check_head_proj(self, L: int, h: int, direction: str, *, ss=(1.0, 2.0, 6.0)) -> dict:
        """V24・V25(射影の正しさ・直交成分が変わらないこと)。"""
        from .headproj import check_head_proj
        return check_head_proj(self, L, h, direction, ss=ss)

    def check_proj_bake(self, recipe: Recipe, baked: Path, *, contrast: str | Path | None = None,
                        suite: str | Path | None = None) -> dict:
        """V26: 焼き込み版と LoRA 版で (a) A・B の射影の値、(b) 最初のトークンの分布の KL、(c) 最尤トークンの一致率、
        (d) perplexity を比べる。LoRA 版を測ってからモデルを閉じ、焼き込み版を読む(大きいモデルでも 1 つずつ)。
        射影する方向は proj の方向、proj が無ければ keyed の方向(V33、CHANGE-07)。

        射影の値の判定は、層ごと・側ごとに、例ごとの |LoRA 版 − 焼き込み版| の最大を許容誤差(その層の両側の射影の値の標準偏差の
        1 割)と比べる。最後の層(基準の層と違うとき)だけは、合格の線を「許容誤差 + 計算方法の違いだけで出る誤差」にする。
        その誤差は、LoRA 版を測ったのと同じセッションで、書き込みがほぼ 0 の LoRA(同じ LoRA ファイルを倍率 NOISE_LORA_SCALE で
        当てる)と LoRA なしの射影の値を測り、その層の両側の例ごとの差の最大を取る。Q8_0 の掛け算は途中の値も 8 ビットに丸めるので、LoRA を
        当てるだけで変わる計算の小さな違いが後ろの層で広がり、最後の層の射影の値は焼き込みと関係なく許容誤差の 7〜8 割動いた
        (verification.md F139)。基準の層の射影の値・最尤トークン・ppl の判定はそのまま。"""
        from .direction import Scorer, base_layer_of, last_logprobs
        from .evalx import _ppl, load_suite
        from .headproj import proj_heads
        from .keyed import keyed_units
        heads = proj_heads(recipe)
        ku = keyed_units(recipe) if recipe.keyed_units else []
        dname = heads[0].direction if heads else ku[0].direction if ku else recipe.default_direction
        ds, cs, _ = self._direction_contrast(dname, contrast)
        b = base_layer_of(ds)
        Ls = sorted({x for x in (b, max(L for L in ds.raw if L <= self.meta.n_layer - 2)) if x is not None})
        last = Ls[-1] if Ls[-1] != b else None                 # 合格の線に計算方法の違いの誤差を足す層
        convs = cs.convs("A") + cs.convs("B")
        texts = load_suite(suite).ppl

        def measure(sess, lora):
            sc = Scorer(sess, ds, cs, layers=Ls)
            got = sc.run(lora)
            lp = last_logprobs(sess, convs, None, lora)
            ppl = _ppl(sess, texts, None, lora, lambda n: None) if texts else {}
            return got, lp, ppl, sc
        lora = self.lora_for(recipe)
        a, lpa, ppa, sc = measure(self, lora)
        # 計算方法の違いだけで出る誤差: 書き込みがほぼ 0 の LoRA と LoRA なしの射影の値の差(射影の値だけ。モデルは開き直さない)。
        # 倍率を 0 ちょうどにすると llama.cpp が LoRA を外して計算の仕方が LoRA 版と変わるおそれがあるので、ごく小さい倍率にする
        noise = None
        if lora and last is not None:
            t0 = time.perf_counter()
            tiny = sc.run([(p, math.copysign(NOISE_LORA_SCALE, float(s_))) for p, s_ in lora], sides="AB")
            none = sc.run(None, sides="AB")
            by_side = {side: float(np.abs(np.asarray(tiny[side]["values"][last]) - np.asarray(none[side]["values"][last])).max())
                       for side in "AB"}
            noise = {"layer": last, "max_abs_diff": max(by_side.values()), "by_side": by_side,
                     "lora_scale": NOISE_LORA_SCALE, "seconds": time.perf_counter() - t0}
        self.close()
        other = Session(baked, root=self.store.root, n_gpu_layers=self._engine_kw["n_gpu_layers"],
                        n_ctx=self._engine_kw["n_ctx"], n_seq_max=self._engine_kw["n_seq_max"])
        try:
            bb, lpb, ppb, _ = measure(other, None)
        finally:
            other.close()
        scores, ok_s = bake_score_check(a, bb, Ls, last=last, noise=noise["max_abs_diff"] if noise else 0.0)
        kl = float(np.mean(np.sum(np.exp(lpa) * (lpa - lpb), axis=1)))
        ta, tb = np.argmax(lpa, axis=1), np.argmax(lpb, axis=1)
        top1 = float(np.mean(ta == tb))
        # 最尤トークンが分かれた例は、LoRA 版での 2 つの差(nats)。0.5 未満なら僅差(V9・V14 と同じ基準)
        gaps = [float(lpa[i, ta[i]] - lpa[i, tb[i]]) for i in range(len(ta)) if ta[i] != tb[i]]
        pm = lambda d: float(np.exp(np.mean(np.log(list(d.values()))))) if d else None  # noqa: E731
        pa, pb = pm(ppa), pm(ppb)
        ppl_diff = None if pa is None else (pb / pa - 1) * 100
        return {"scores": scores, "kl_first_token": kl, "top1_match": top1, "top1_mismatch_gaps": gaps,
                "n_prompts": len(convs), "ppl_lora": pa, "ppl_baked": pb, "ppl_diff_pct": ppl_diff,
                "base_layer": b, "last_layer": last, "noise": noise,
                "pass": bool(ok_s and all(g < 0.5 for g in gaps) and (ppl_diff is None or abs(ppl_diff) < 0.5)),
                "note": None if not gaps else "若最大概率 token 不同的示例全部为微小差距（低于 0.5 nats），则通过。重新量化时的舍入，以及重写张量的计算方式（Q8_0 还是 F16）不同会导致差异"}

    def evals(self) -> list[dict]:
        """已保存的 eval 列表（从新到旧）。"""
        out = []
        for p in sorted(self.store.eval_dir(self.meta.model_id).glob("*.json"), key=lambda p: -p.stat().st_mtime):
            try:
                d = json.loads(p.read_text())
            except ValueError:
                continue
            if "recipe" not in d or "suite" not in d:        # 方式の比較・ヘッド数の曲線・消去の比較(CHANGE-06 F.3・F.4、§6.17)
                kind = ("compare-methods" if "target_from" in d else "nheads" if "alloc" in d
                        else "compare-ablation" if "neurons" in d else "other")
                out.append({"name": d.get("name", p.stem), "kind": kind, "direction": d.get("direction"),
                            "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))})
                continue
            last = d["rows"][-1] if d.get("rows") else {}
            out.append({"name": d.get("name", p.stem), "kind": "eval", "created": d.get("created"), "when": d.get("when"),
                        "doses": d.get("doses"), "units": [_unit_spec(u) for u in d["recipe"]["units"]]
                        + ([_ablate_label(d["recipe"]["ablate"])] if d["recipe"].get("ablate") else []),
                        "shippable": [r["dose"] for r in d.get("rows", []) if r.get("shippable")],
                        "max_dose_drop_pt": last.get("capability_drop_pt")})
        return out

    def eval_load(self, name: str) -> dict:
        if not re.fullmatch(r"[\w.\-@]+", name or ""):
            raise GneError("BAD_ARGS", f"名称 {name!r} 无效")
        p = self.store.eval_dir(self.meta.model_id) / f"{name}.json"
        if not p.exists():
            raise GneError("FILE_NOT_FOUND", f"eval {name} 不存在")
        return json.loads(p.read_text())

    def _verify_candidates(self, cands, name: str, design_name: str, n_controls: int, max_tokens: int) -> dict:
        """按层汇总候选，运行 quick 设计的 rollout（带对照）。排序使用 Phase 3 的 rank。"""
        from .sweep import load_design, read_rollouts, run_sweep
        design = load_design(design_name)
        by_layer: dict[int, list] = {}
        for c in cands:
            by_layer.setdefault(c.layer, []).append(c)
        sv = self._need_rho(sorted(by_layer))
        runs, texts = [], {}
        for L, cs in sorted(by_layer.items()):
            out = self.store.runs_dir(self.meta.model_id) / f"reverse-{name}-L{L}.jsonl"
            r = run_sweep(self.engine, self.weights, sv, L=L, neurons=[c.neuron for c in cs], design=design, out=out,
                          model_id=self.meta.model_id, random_controls=n_controls, max_tokens=max_tokens,
                          seed=self.seed, signs={c.neuron: c.sign for c in cs}, progress=self.progress)
            runs.append(r)
            for rec in read_rollouts(out):
                if rec["kind"] in ("neuron", "baseline") and rec["prompt"] == design.pairs[0][0]:
                    texts[(L, rec["neuron"])] = rec["text"]
        first = design.pairs[0][0]
        examples = [{"layer": c.layer, "neuron": c.neuron, "sign": c.sign, "prompt": first,
                     "text": texts.get((c.layer, c.neuron), "")} for c in cands]
        baselines = {f"L{L}": texts.get((L, None), "") for L in sorted(by_layer)}
        return {"design": design_name, "runs": runs, "examples": examples, "baselines": baselines}


def bake_score_check(lora: dict, baked: dict, layers, *, last: int | None = None, noise: float = 0.0) -> tuple[dict, bool]:
    """V26・V33 の射影の値の判定。lora・baked は Scorer.run の結果({側: {"values": {L: 例ごとの射影の値}}})。層ごと・側ごとに
    {lora・baked(平均), max_abs_diff(例ごとの |LoRA − 焼き込み| の最大), resolution(許容誤差 = その層の LoRA 版の両側の
    射影の値の標準偏差の 1 割), threshold(合格の線), ok} を返す。合格の線は許容誤差で、last の層だけ許容誤差 + noise
    (計算方法の違いだけで出る誤差、check_proj_bake)。last の行には noise も入れる。戻り値 (scores, 全部 ok か)。"""
    scores, ok_all = {}, True
    for L in layers:
        sd = float(np.std(np.concatenate([lora["A"]["values"][L], lora["B"]["values"][L]])))
        res = 0.1 * sd                                    # 許容誤差: 両側の射影の値の広がりの 1 割
        line = res + float(noise) if L == last else res
        for side in "AB":
            va, vb = np.asarray(lora[side]["values"][L]), np.asarray(baked[side]["values"][L])
            diff = float(np.abs(va - vb).max())
            row = {"lora": float(va.mean()), "baked": float(vb.mean()), "max_abs_diff": diff, "resolution": res}
            if L == last:
                row["noise"] = float(noise)
            row.update(threshold=line, ok=diff <= line)
            scores[f"{L}{side}"] = row
            ok_all &= diff <= line
    return scores, bool(ok_all)


def load_prompt_list(path: str | Path) -> list[str]:
    """提示词集合 YAML：字符串列表、{prompts: [...]}，或对照集（reactivity 的 with，若没有则用 user_prompts）。"""
    import yaml
    p = Path(path)
    if not p.exists():
        raise GneError("FILE_NOT_FOUND", f"{p} 不存在")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(d, dict):
        if d.get("reactivity"):
            d = [x["with"] for x in d["reactivity"] if x.get("with")]
        else:
            d = d.get("prompts") or d.get("user_prompts")
    if not isinstance(d, list) or not d or not all(isinstance(x, str) for x in d):
        raise GneError("BAD_ARGS", f"{p}: 提示词集合必须是字符串列表（或对照集 YAML）")
    return d


def vector_key_of(path: Path) -> str:
    """由文件内容决定的短键（重新创建方向时也会重建 query 的 LoRA）。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _unit_spec(u: dict) -> str:
    """to_dict() 的单元 → L:N:+dose / L:N:xk / L:Hh:xk（query 为 L:Hh:query:方向:×scale，用于显示）。
    keyed(CHANGE-07)は L:ffn|attn|S|Ee:keyed:aα[@方向][#鍵]。"""
    if u.get("mode") == "keyed":
        kind = u.get("kind")
        who = {"ffn": "ffn", "attn": "attn", "shared": "S"}.get(kind, f"E{u.get('expert')}")
        a = u.get("k") if u.get("k") is not None else u.get("alpha")
        return (f"{u['layer']}:{who}:keyed:" + ("a?" if a is None else f"a{a:g}")
                + (f"@{u['direction']}" if u.get("direction") else "")
                + (f"#{u['key']}" if u.get("key") else ""))
    if u.get("mode") == "query":
        return f"{u['layer']}:H{u['head']}:query:{u.get('direction')}:x{u['k']:g}"
    if u.get("mode") == "proj":                                  # CHANGE-06
        return f"{u['layer']}:H{u['head']}:s{u['k']:g}" + (f"@{u['direction']}" if u.get("direction") else "")
    tail = f"x{u['k']:g}" if u.get("mode") == "gain" else f"{u['dose']:+g}"
    mid = (f"random:{u['seed']}" if u.get("kind") == "random" else f"H{u['head']}" if u.get("kind") == "head"
           else f"dir:{u.get('direction')}" if u.get("kind") == "direction" else str(u["neuron"]))
    return f"{u['layer']}:{mid}:{tail}"


def _ablate_label(a: dict) -> str:
    from .steer import AblateSpec
    return AblateSpec.from_dict(a).label()


def template_override(model_id: str, arch: str) -> tuple[str, str] | None:
    """configs/templates/<model_id>.yaml 或 <arch>.yaml 中的 chat_template（§5.7 的 4）。"""
    import yaml
    from .store import CONFIGS
    for key in (model_id, arch):
        p = CONFIGS / "templates" / f"{key}.yaml"
        if p.exists():
            tpl = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("chat_template")
            if tpl:
                return str(p), tpl
    return None


def _summ(x: np.ndarray) -> dict:
    if not x.size:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": int(x.size), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def _merge_logit(old, new):
    """按神经元编号的并集合并同一层的 logit lens（优先采用新结果）。"""
    from .lens import LogitLens
    if old.up_ids.shape[1] != new.up_ids.shape[1]:
        return new
    keep = ~np.isin(old.neurons, new.neurons)
    cat = lambda a, b: np.concatenate([a[keep], b])  # noqa: E731
    nn = cat(old.neurons, new.neurons)
    o = np.argsort(nn)
    return LogitLens(new.layer, nn[o], cat(old.up_ids, new.up_ids)[o], cat(old.up_z, new.up_z)[o],
                     cat(old.down_ids, new.down_ids)[o], cat(old.down_z, new.down_z)[o])


def _merge_resid(old: ResidStats, new: ResidStats) -> ResidStats:
    """指定层计算时，保留已有统计中的其他层（仅当语料相同）。"""
    if old.corpus_sha != new.corpus_sha or old.chunk_tokens != new.chunk_tokens:
        return new
    keep = [i for i, L in enumerate(old.layers) if L not in set(new.layers.tolist())]
    cat = lambda a, b: np.concatenate([a[keep], b])  # noqa: E731
    order = np.argsort(cat(old.layers, new.layers))
    return ResidStats(cat(old.layers, new.layers)[order], cat(old.p10, new.p10)[order],
                      cat(old.p50, new.p50)[order], cat(old.p90, new.p90)[order], cat(old.mean, new.mean)[order],
                      new.n_tokens, new.corpus_sha, new.chunk_tokens)


def _version() -> str:
    from . import __version__
    return __version__


__all__ = ["open_model", "Session", "GneError"]

"""CHiPS のレシピを当てても消したい振る舞いが残った A 側の質問を集めて、鍵つきの書き込み(keyed、CHANGE-07)の対照集合の
JSON を作る(gne direction key-from-behave、Session.direction_key_from_behave、MCP direction_key_from_behave)。

流れ: gne behave で CHiPS のレシピを当てて A 側の質問に答えさせる → ここで、振る舞いがまだ出た質問を抜き出す →
configs/keyed/one_scene.json と同じ形の JSON を <root>/keyed/<name>.json に書く → 鍵のタブ(ひな形に出る)か gne direction key で
鍵を作る(keyed.iv の CHiPS のレシピを当てた状態で、keyed.direction の CHiPS の方向を keyed.alpha −1 で書く = その場面だけ打ち消す)。

出力の JSON:
- user_prompts: 抜き出した質問(behave の質問の順)。replies: その質問の behave の返事(既定。鍵の作成は返事を作らずにこれを読む)。
- positions: 既定は before:<CHiPS の対照集合の validate.match>。validate.match は大文字小文字を無視して使われる
  (direction.load_contrast の使い方)が、positions の before: は区別するので、大文字と小文字のある文字を含めば (?i) を付ける。
- dont_touch: [{name: ordinary, prompts: CHiPS の対照集合の B 側の質問}](user_prompts と同じ質問は外す)。positions は省く
  (書き込みたくない位置と同じ not-before:<正規表現> = 返事のうち一致の直前以外の全部)。返事を書き始める直前の位置は鍵の作成が
  自動で Q に入れる。
- validate: CHiPS の対照集合のものを写す。keyed: {direction: CHiPS の方向, alpha: −1, iv: CHiPS のレシピ}。

モデルは動かさない(behave の結果の返事の文字と、記録されたトークンの数だけを読む)。広すぎる正規表現の見分けは、返事の
トークンのうち書き込みたい位置(一致の始まりの文字を含むトークン)になる割合の見込みで行う(reply_positions・BROAD_SHARE)。
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import yaml

from .steer import GneError

# 「残った」とみなす、振る舞いが出た種の割合の既定。ユーザーの決まり「A 側には振る舞いが確実に出る質問だけを入れる」に合わせ、
# 全部の種で出た質問だけを入れる(greedy の 1 回なら「出た」と同じ)。種によって出たり出なかったりする質問は、鍵の作成で読む
# 返事に振る舞いが無いことがあり、書き込みたい位置の状態がぼやける。
DEFAULT_MIN_RATE = 1.0
# 広すぎる正規表現: 返事のトークンのうち、書き込みたい位置(before: の一致の始まり)になる割合の見込みがこれ以上なら警告する。
# うまく働いた鍵の書き込みたい位置は返事の位置の 2% ほど(語尾の例で 15 / 768、ツンデレの例で 18 / 約 770、docs/keyed-write.md §6・§7)。
# 文末ごとに出る言い方でも、Qwen3-0.6B のトークンで短い日本語の返事(1 文 12 トークンほど)の 8.5%(「(です|ます)。」・「。」)。
# 言語の文字の種類に当たる正規表現は、かな 1 文字の [ぁ-んァ-ン] が日本語の返事のトークンの 57%、英語の \w+ が 89%(同じトークナイザ)。
# 0.25 は、文末ごとに出る言い方の 3 倍ほどで、文字の種類に当たる正規表現の半分より低い。
BROAD_SHARE = 0.25
GROUP = "ordinary"                       # dont_touch の群の名前(CHiPS の対照集合の B 側の質問)
_NAME_RE = re.compile(r"[\w.\-]+")
# 大文字と小文字のある文字を探す前に落とす、正規表現の書き方(エスケープ・名前つきグループ・旗)
_ESCAPES = re.compile(r"\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|N\{[^}]*\}|.)", re.DOTALL)
_GROUP_SYNTAX = re.compile(r"\(\?(?:P<[^>]*>|P=[^)]*\)|[aiLmsux\-]+[:)])")


# --- 正規表現 ----------------------------------------------------------------------------------------------

def has_cased(pattern: str) -> bool:
    """正規表現が、大文字と小文字のある文字(英字など)を文字として含むか。\\w・\\u3041 などのエスケープと、(?P<名前>)・(?i) などの
    書き方は数えない。かな・漢字には大文字と小文字が無いので、日本語だけの正規表現は False。"""
    s = _GROUP_SYNTAX.sub("", _ESCAPES.sub("", pattern))
    return any(ch.lower() != ch.upper() for ch in s)


def ignore_case(pattern: str) -> str:
    """validate.match(大文字小文字を無視して使われる)→ positions の before: に入れる正規表現。大文字と小文字のある文字を含み、
    まだ大文字小文字を無視しない正規表現なら、先頭に (?i) を付ける(すでに (?i) などで無視するなら付けない)。"""
    if re.compile(pattern).flags & re.IGNORECASE or not has_cased(pattern):
        return pattern
    return "(?i)" + pattern


def match_starts(text: str, rx: re.Pattern) -> int:
    """返事の中の、一致(重ならない、長さ 0 は除く)の始まりの数。"""
    return len({m.start() for m in rx.finditer(text or "") if m.end() > m.start()})


def reply_positions(text: str, rx: re.Pattern, n_tokens: float) -> int:
    """返事の中の書き込みたい位置(before:<re> = 一致の始まりの文字を含むトークン)の数の見込み。トークンの区切りは記録に無いので、
    返事の文字を n_tokens 個の等しい区間に分け、一致の始まりが入る区間を数える(1 つのトークンに始まりが 2 つあっても 1 つ)。
    まばらな一致では一致の始まりの数と同じ。Qwen3-0.6B のトークンで測ると、かな 1 文字の正規表現で 0.75(本当は 0.57)、
    英語の \\w+ で 0.75(本当は 0.89)、文末の言い方は本当の値と同じだった。"""
    text = text or ""
    n = max(1, int(round(n_tokens)))
    if not text:
        return 0
    return min(n, len({int(m.start() * n / len(text)) for m in rx.finditer(text) if m.end() > m.start()}))


def reply_tokens(r: dict) -> float:
    """behave の返事のトークンの数の見込み。思考ありなら、記録された数(思考を含む)を答えの部分のバイトの割合で縮める。
    数が無ければ文字の数。"""
    text = r.get("text") or ""
    n = r.get("n_tokens")
    if not isinstance(n, (int, float)) or n <= 0:
        return float(len(text))
    full = r.get("full")
    if full:
        fb = len(full.encode("utf-8"))
        return float(n) * (len(text.encode("utf-8")) / fb if fb else 0.0)
    return float(n)


# --- 入力 --------------------------------------------------------------------------------------------------

def behave_path(session, behave) -> Path:
    """behave の結果の名前(<root>/behave/<model_id>/<名前>.json)かパス → ファイル。"""
    s = str(behave or "")
    p = Path(s)
    if p.suffix == ".json" and p.exists():
        return p
    if _NAME_RE.fullmatch(s):
        q = session.store.root / "behave" / session.meta.model_id / f"{s}.json"
        if q.exists():
            return q
    raise GneError("FILE_NOT_FOUND", f"没有 behave 结果 {s!r}",
                   f"请传入用 gne behave --name <名称> 创建的结果名称（<root>/behave/{session.meta.model_id}/ 中的 .json）或路径")


def _load_behave(p: Path) -> dict:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise GneError("BAD_ARGS", f"{p}: 无法读取 behave 结果（{e}）") from e
    runs = d.get("runs") if isinstance(d, dict) else None
    if not isinstance(runs, list) or not runs or not all(isinstance(r, dict) and isinstance(r.get("replies"), list) for r in runs):
        raise GneError("BAD_ARGS", f"{p}: 不是 behave 结果（每个种子的 runs 中需要回复 replies）")
    return d


def _questions(d: dict, prompts, warn) -> list:
    """behave の結果の質問(記録が無ければ prompts の質問集)。"""
    from .behave import load_prompts
    from .reverse import question_text
    rec = d.get("prompts")
    given = None
    if prompts is not None:
        given = load_prompts(prompts) if isinstance(prompts, str | Path) else list(prompts)
    if rec is None and given is None:
        raise GneError("BAD_ARGS", "behave 结果中没有记录问题（此版本之前的 gne behave 结果）",
                       "请用 prompts（CLI 为 --prompts）传入交给 behave 的问题集，或重新运行 gne behave")
    qs = list(rec) if rec is not None else given
    if rec is not None and given is not None and [question_text(q) for q in rec] != [question_text(q) for q in given]:
        warn("传入的问题集与 behave 结果中记录的问题不同，因此使用记录中的问题")
    n = d.get("n_prompts")
    if isinstance(n, int) and n != len(qs):
        raise GneError("BAD_ARGS", f"问题数 {len(qs)} 与 behave 结果中的 n_prompts {n} 不同（请传入交给 behave 的问题集）")
    return qs


def _contrast(session, direction, contrast, warn):
    """CHiPS の方向と対照集合 → (方向の名前, 対照集合のパス, ContrastSet, Validate, validate の表)。"""
    from .direction import load_contrast
    if not direction and not contrast:
        raise GneError("BAD_ARGS", "请传入 CHiPS 方向名称（direction）或对照集（contrast）")
    ds = None
    if direction:
        try:
            ds = session.direction_set(str(direction))
        except GneError as e:
            if e.code != "FILE_NOT_FOUND" or not contrast or not _NAME_RE.fullmatch(str(direction)):
                raise
            warn(f"方向 {direction} 还不存在。在创建键之前，请用这个名称创建 CHiPS 使用的方向")
    if ds is not None:
        if Path(str(direction)).suffix == ".npz" and Path(str(direction)).resolve().parent != \
                session.store.direction_dir(session.meta.model_id).resolve():
            warn(f"方向是以路径传入的。配方和键标签页会按名称（{ds.name}）查找方向，"
                 f"因此请将其放在 <root>/directions/{session.meta.model_id}/ 中")
    if contrast:
        cpath = session._contrast_file(contrast)
    else:
        src = ds.meta.get("source")
        if not src or not Path(src).exists():
            raise GneError("BAD_ARGS", f"找不到创建方向 {ds.name} 所用的对照集 {src}", "请通过 contrast（CLI 为 --contrast）传入")
        cpath = Path(src)
    cs, val = load_contrast(cpath)
    raw = yaml.safe_load(Path(cpath).read_text(encoding="utf-8")) or {}
    dname = ds.name if ds is not None else str(direction or cs.name)
    if ds is None and not direction:
        try:
            session.direction_set(dname)
        except GneError:
            warn(f"方向 {dname} 还不存在。在创建键之前，请用 gne direction --contrast {cpath} 创建 CHiPS 使用的方向"
                 "（如果是其他名称，请用 direction 传入）")
    elif ds is not None and contrast and ds.meta.get("contrast_sha") and ds.meta.get("contrast_sha") != cs.sha:
        warn(f"方向 {ds.name} 是由内容与传入对照集不同的文件创建的（B 侧问题和 validate 从该对照集获取）")
    return dname, Path(cpath), cs, val, (raw.get("validate") if isinstance(raw, dict) else None)


def _recipe(session, recipe):
    """CHiPS のレシピ → (Recipe, keyed.iv に入れる参照)。保存したレシピ(<root>/recipes/)なら名前、ほかはパス。"""
    from .steer import Recipe
    if recipe is None or str(recipe) == "":
        raise GneError("BAD_ARGS", "请传入 CHiPS 配方（recipe，已保存配方的名称或路径）",
                       "键是在应用该配方的状态下创建的，因此会写入输出的 keyed.iv")
    p = session.recipe_path(str(recipe))
    rec = Recipe.load(p)
    ref = p.stem if p.resolve().parent == session.store.recipes_dir().resolve() else str(p.resolve())
    return rec, ref


def _units_key(rec_dict: dict | None) -> tuple[list[str], str]:
    units = sorted(json.dumps(u, sort_keys=True, ensure_ascii=False) for u in (rec_dict or {}).get("units") or [])
    return units, json.dumps((rec_dict or {}).get("ablate"), sort_keys=True)


def check_recipe(behave_recipe: dict | None, rec, names: list | None = None) -> str | None:
    """behave の結果に記録されたレシピが CHiPS のレシピと同じか。違えば理由の文、同じなら None。ユニットは behave と同じく
    combine_recipes で方向を書き込んでから比べる(名前は behave の保存名になるので比べない)。"""
    from .steer import combine_recipes
    if behave_recipe is None:
        return "behave 结果是在未应用配方的情况下回答的"
    want_u, want_a = _units_key(combine_recipes([rec], name="x").to_dict())
    got_u, got_a = _units_key(behave_recipe)
    if want_u == got_u and want_a == got_a:
        return None
    rn = rec.name or "?"
    named = f"（传给 behave 的配方：{', '.join(map(str, names))}）" if names else ""
    rest = list(got_u)
    for u in want_u:
        if u in rest:
            rest.remove(u)
        else:
            rest = None
            break
    if rest is not None and rest and want_a in (got_a, "null"):
        return f"behave 结果是在配方 {rn} 上叠加其他 {len(rest)} 个单元后回答的{named}"
    return f"behave 结果中记录的配方与配方 {rn} 不同（单元 {len(got_u)} 个 对 {len(want_u)} 个）{named}"


# --- 本体 --------------------------------------------------------------------------------------------------

def _short(q, n: int = 60) -> str:
    from .reverse import question_text
    s = question_text(q)
    return s if len(s) <= n else s[:n - 1] + "…"


def _about(meta: dict) -> dict:
    """出力の about(日本語・英語の題と説明)。どの behave の結果・レシピ・判定から作ったかと、使い方。"""
    j = meta
    if j["method"] == "label":
        how_ja = f"读取方判定标签 {', '.join(j['labels'])}"
        how_en = f"judge label {', '.join(j['labels'])}"
    else:
        how_ja = f"正则表达式 {j['match']}" + ("（忽略大小写）" if j["match_ignore_case"] else "")
        how_en = f"the regex {j['match']}" + (" (ignoring case)" if j["match_ignore_case"] else "")
    rate_ja = "所有种子" if j["min_rate"] >= 1 else f"至少 {j['min_rate']:g} 的种子"
    rate_en = "every seed" if j["min_rate"] >= 1 else f"at least {j['min_rate']:g} of the seeds"
    rep_ja = "已将 behave 回复放入 replies，因此创建键时直接读取而不重新生成。" if j["replies"] else \
        f"回复在创建键时应用配方生成（keyed.max_tokens {j['max_tokens']}）。"
    rep_en = "replies holds the behave replies, so building the key reads them instead of generating. " if j["replies"] else \
        f"Replies are generated with the recipe applied when the key is built (keyed.max_tokens {j['max_tokens']}). "
    ja = {"title": f"抑制 CHiPS 留下的场景（behave {j['behave']}，配方 {j['recipe']}）",
          "text": (f"在 gne behave 结果 {j['behave']}（应用配方 {j['recipe']} 回答 {j['n_prompts']} 个问题，{j['n_seeds']} 个种子）中，"
                   f"通过 {how_ja}、在 {rate_ja} 中仍出现想消除的行为的 {j['n_extracted']} 个问题已放入 user_prompts。{rep_ja}"
                   f"写入位置为 positions {j['positions']}（写出行为之前）。dont_touch 的 {GROUP} 是 CHiPS 对照集 "
                   f"{j['contrast']} 的 B 侧问题 {j['n_ordinary']} 个（已去掉与 user_prompts 相同的问题）。键在应用 keyed.iv 中配方 "
                   f"{j['recipe']} 的状态下创建，并以 α = −1 写 keyed.direction 中的方向 {j['direction']}（只在那些场景中抵消）。"
                   "用法：在键标签页选择此模板并运行“从方向到 key-check”（复用 keyed.direction 中的方向），"
                   f"或使用 gne direction key --contrast <此文件> --layers … --iv {j['recipe']}。将 key-check 中超过比较用键的层的 "
                   f"键控单元加入配方 {j['recipe']}（gne recipe-combine），导出一个 LoRA，并通过生成确认。")}
    en = {"title": f"Suppress the scenes CHiPS left behind (behave {j['behave']}, recipe {j['recipe']})",
          "text": (f"From the gne behave result {j['behave']} (recipe {j['recipe']} applied, {j['n_prompts']} questions, "
                   f"{j['n_seeds']} seed(s)), the {j['n_extracted']} questions whose replies still showed the behavior by {how_en} in "
                   f"{rate_en} are in user_prompts. {rep_en}The positions to write at are positions {j['positions']} (just before the "
                   f"behavior is written). dont_touch {GROUP} holds the {j['n_ordinary']} B-side questions of the CHiPS contrast set "
                   f"{j['contrast']} (questions also in user_prompts were removed). Build the key with the recipe {j['recipe']} in "
                   f"keyed.iv applied, and write the direction {j['direction']} in keyed.direction with α = −1 (cancel it only in those "
                   "scenes). Use: pick this template in the key tab and run “Direction to key-check” (the direction in keyed.direction "
                   f"is reused), or gne direction key --contrast <this file> --layers … --iv {j['recipe']}. Add the keyed units of the "
                   f"layers that beat the control keys in key-check to the recipe {j['recipe']} (gne recipe-combine), export one LoRA, "
                   "and check by generating.")}
    return {"order": 50, "ja": ja, "en": en}


def from_behave(session, behave, *, recipe, direction: str | None = None, contrast=None, labels=None,
                match: str | None = None, min_rate: float | None = None, positions: str | None = None,
                name: str | None = None, out=None, replies: bool = True, prompts=None, strict: bool = False) -> dict:
    """behave の結果(名前かパス)から、CHiPS のレシピを当てても消したい振る舞いが残った質問を抜き出し、keyed の対照集合の JSON を
    <root>/keyed/<name>.json(out ならそのパス)に書く。

    - recipe: CHiPS のレシピ(保存した名前かパス)。出力の keyed.iv。behave の結果に記録されたレシピと比べる。
    - direction / contrast: CHiPS の方向の名前と、その対照集合(どちらか。direction だけなら方向を作った対照集合、contrast だけなら
      方向の名前は対照集合の name)。B 側の質問・validate・A 側の質問を取る。
    - 判定: labels(読み手の判定のラベル。文字列 "P" / "P,C" かリスト)を渡せばそれ。無ければ match、無ければ behave の結果の match
      (behave と同じく大文字小文字を区別)、無ければ対照集合の validate.match(大文字小文字を無視)。
    - min_rate: 振る舞いが出た種の割合がこれ以上の質問を入れる(既定 DEFAULT_MIN_RATE = 1.0、全部の種)。
    - positions: 書き込みたい位置(既定 before:<validate.match>、大文字と小文字のある文字を含めば (?i) を付ける)。before: だけ。
    - replies: 抜き出した質問の behave の返事を出力の replies に入れる(既定)。False なら鍵の作成で返事を作る(keyed.max_tokens に
      behave の長さ)。思考ありの behave の結果では入れない。
    - prompts: 質問の記録が無い古い behave の結果のときの質問集。strict: レシピ・質問の確かめに外れたらエラー。"""
    from .keyed_build import MIN_A_EXAMPLES, MIN_A_ROWS, few_side_a, parse_positions, read_key_spec
    from .keyed_pipeline import clean_spec
    from .reverse import question_text
    warnings: list[str] = []
    notes: list[str] = []
    issues: list[dict] = []                          # 警告の種類と重さ(UI が赤 = error・黄色 = warn に分けて出す)

    def warn(msg: str, kind: str = "other", level: str = "warn") -> None:
        warnings.append(msg)
        issues.append({"kind": kind, "level": level, "message": msg})
        session.warn(msg)

    def issue(msg: str, kind: str, level: str = "error") -> None:   # behave の結果の確かめ: strict ならエラー
        if strict:
            raise GneError("BAD_ARGS", msg, "不传 strict 时只警告并继续创建")
        warn(msg, kind, level)

    bpath = behave_path(session, behave)
    bd = _load_behave(bpath)
    bname = str(bd.get("name") or bpath.stem)
    qs = _questions(bd, prompts, lambda m: warn(m, "prompts"))
    dname, cpath, cs, val, vraw = _contrast(session, direction, contrast, lambda m: warn(m, "direction"))
    rec, iv_ref = _recipe(session, recipe)
    rname = rec.name or Path(str(iv_ref)).stem
    name = name or f"{bname}_keyed"
    if not _NAME_RE.fullmatch(name):
        raise GneError("BAD_ARGS", f"名称 {name!r} 只能包含字母、数字、_、-、.（也会成为键的默认名称）")

    # 1. behave の結果が、そのレシピを当てて A 側の質問に答えたものか
    why = check_recipe(bd.get("recipe"), rec, bd.get("recipe_names"))
    checks = {"recipe": why is None}
    if why:
        issue(why + "。键是在应用配方的状态下创建的，因此 behave 也必须只应用同一配方来回答", "recipe")
    a_side = {question_text(q) for q in cs.prompts("A")} | ({question_text(q) for q in val.prompts} if val.explicit else set())
    qt = [question_text(q) for q in qs]
    n_in = sum(t in a_side for t in qt)
    checks["questions_in_a"] = n_in
    if n_in == 0:
        issue(f"behave 的 {len(qs)} 个问题中，没有 CHiPS 对照集 {cs.name} 的 A 侧问题（这是用其他问题集回答的结果）", "questions")
    elif n_in < len(qs):
        issue(f"behave 的 {len(qs)} 个问题中有 {len(qs) - n_in} 个不在 CHiPS 对照集 {cs.name} 的 A 侧问题中"
              "（即使提取也不是 A 侧场景）", "questions", "warn")
    if cs.system_a is not None or cs.recipe_a is not None:
        warn(f"CHiPS 对照集 {cs.name} 的 A 侧是使用 {'system_a' if cs.system_a is not None else 'recipe_a'} 创建的状态，但 behave 和创建键"
             "读取的是不带 system 的回答。这里只提取即使没有 system 也出现行为的问题，由 system 引起的行为不适合此方法", "system")

    # 2. 判定
    lab = [x.strip() for x in (labels.split(",") if isinstance(labels, str) else (labels or [])) if str(x).strip()]
    if lab and match:
        raise GneError("BAD_ARGS", "判定只能使用正则表达式（match）或读取方标签（labels）中的一个")
    if lab:
        if not bd.get("judge"):
            raise GneError("BAD_ARGS", "behave 结果中没有读取方判定（请带 --judge 运行 gne behave，或用正则表达式判定）")
        known = [str(x) for x in bd.get("labels") or []]
        bad = [x for x in lab if x not in known]
        if bad:
            raise GneError("BAD_ARGS", f"标签 {', '.join(bad)} 不在 behave 结果的标签（{', '.join(known)}）中")
        method, jrx, jsrc, jic = "label", None, "labels", False
    else:
        if match:
            jsrc, pat, jic = "match", match, False
        elif bd.get("match"):
            jsrc, pat, jic = "behave", str(bd["match"]), False
        elif val.match:
            jsrc, pat, jic = "validate", val.match, True
        else:
            raise GneError("BAD_ARGS", "没有判定：请传入 match（正则表达式）或 labels（读取方标签）"
                           "（behave 结果和对照集的 validate.match 都没有正则表达式）")
        try:
            jrx = re.compile(pat, re.IGNORECASE if jic else 0)
        except re.error as e:
            raise GneError("BAD_ARGS", f"无法读取判定正则表达式 {pat!r}: {e}") from e
        method = "match"
    mr = DEFAULT_MIN_RATE if min_rate is None else float(min_rate)
    if not 0 < mr <= 1:
        raise GneError("BAD_ARGS", f"min_rate 必须大于 0 且小于等于 1（{mr}）")

    def present(r: dict) -> bool:
        if method == "label":
            return r.get("label") in lab
        return bool((r.get("text") or "").strip()) and bool(jrx.search(r.get("text") or ""))

    # 3. 書き込みたい位置
    if positions:
        pos = parse_positions(positions)
        pos_src = "given"
    else:
        if not val.match:
            raise GneError("BAD_ARGS", f"对照集 {cs.name} 没有 validate.match，因此无法确定写入位置",
                           "请在 positions（CLI 为 --positions）中传入对应行为开头的正则表达式 before:<...>")
        pos = parse_positions("before:" + ignore_case(val.match))
        pos_src = "validate"
    if pos.kind != "before":
        raise GneError("BAD_ARGS", f"positions 必须为 before:<正则表达式>（{pos.text}）",
                       "此 JSON 使用与推理时相同上下文的回复来确定写入位置，属于方法 (b)")
    prx = re.compile(pos.pattern)

    # 4. 質問ごとに、振る舞いが出た種の割合
    runs = bd["runs"]
    n_runs = len(runs)
    by_q: list[list[dict]] = [[] for _ in qs]
    for r in runs:
        for x in r["replies"]:
            i = x.get("i")
            if isinstance(i, int) and 0 <= i < len(qs):
                by_q[i].append(x)
    rates = [sum(present(x) for x in xs) / n_runs if n_runs else 0.0 for xs in by_q]
    picked = [i for i, rt in enumerate(rates) if rt > 0 and rt >= mr - 1e-9]
    partial = [i for i, rt in enumerate(rates) if 0 < rt < mr - 1e-9]
    thinking = bool(bd.get("thinking"))
    use_replies = bool(replies) and not thinking
    if replies and thinking:
        warn("behave 结果是启用思考后回答的，因此不会把回复（</think> 之后的部分）放入输出。创建键时会重新生成回复", "replies")

    def choose(xs: list[dict]) -> dict:
        """出力に入れる返事: 振る舞いが出た種のうち、書き込みたい位置の正規表現に一致し、繰り返していないものを先に。"""
        cand = [x for x in xs if present(x)] or xs
        return max(cand, key=lambda x: (match_starts(x.get("text") or "", prx) > 0, not x.get("loop"), -cand.index(x)))

    def npos(x: dict) -> int:
        return reply_positions(x.get("text") or "", prx, reply_tokens(x))

    chosen = {i: choose(by_q[i]) for i in picked}
    rows = []
    for i in picked:
        x = chosen[i]
        rows.append({"i": i, "question": _short(qs[i]), "rate": rates[i], "starts": npos(x),
                     "tokens": round(reply_tokens(x), 1), "loop": bool(x.get("loop")), "seed": next(
                         (r["seed"] for r in runs if x in r["replies"]), None), "reply": x.get("text") or ""})

    # 5. 広すぎる正規表現(返事のトークンのうち、書き込みたい位置になる割合の見込み)
    meas = [chosen[i] for i in picked] or [x for xs in by_q for x in xs]
    st = sum(npos(x) for x in meas)
    tk = sum(reply_tokens(x) for x in meas)
    share = min(1.0, st / tk) if tk > 0 else None
    broad = share is not None and share >= BROAD_SHARE
    if broad:
        warn(f"positions 的正则表达式 {pos.pattern} 匹配范围过宽：behave 回复 token 中约有 {share * 100:.0f}% 预计会成为写入位置"
             f"（匹配起点）（指导值：低于 {BROAD_SHARE * 100:.0f}%）。写入位置会占回复的大部分，键只会对"
             "“正在写这种语言的状态”产生反应，而不是场景。请通过 positions（CLI 为 --positions）传入对应行为开头的"
             "正则表达式", "broad", "error")

    # 6. 一致の無い質問(書き込みたい位置ができない)と、書き込みたい位置の見込み(鍵の作成は偶数番目、key-check は奇数番目)
    nomatch = [r for r in rows if r["starts"] == 0]
    if rows and nomatch:
        warn(f"提取的 {len(rows)} 个问题中有 {len(nomatch)} 个的回复没有匹配 positions 正则表达式 {pos.pattern}，因此无法形成写入位置"
             "（" + "、".join(f"“{r['question']}”" for r in nomatch[:3]) + ("等" if len(nomatch) > 3 else "") + ")"
             + ("。判定使用的是读取方标签，因此通过 positions 传入对应行为开头的正则表达式可以增加位置" if method == "label" else ""),
             "no_match")
    halves = {}
    for sub, idx in (("even", range(0, len(rows), 2)), ("odd", range(1, len(rows), 2))):
        sel = [rows[k] for k in idx]
        halves[sub] = {"questions": len(sel), "examples": sum(r["starts"] > 0 for r in sel), "positions": sum(r["starts"] for r in sel)}
    est = "" if use_replies else "（根据 behave 回复估算。创建键时会重新生成回复，因此会变化）"
    if not rows:
        warn("没有留下想消除的行为的问题（" + (f"有 {len(partial)} 个问题未达到种子比例 {mr:g}。降低 min_rate 可以纳入"
             if partial else "任何问题都没有出现该行为") + "）。不会写入 JSON", "empty", "error")
    else:
        for sub, what in (("even", "创建键（偶数序号）"), ("odd", "key-check（奇数序号）")):
            h = halves[sub]
            if few_side_a(h["examples"], h["positions"]):
                warn(f"{what}的写入位置只有 {h['examples']} 个示例的 {h['positions']} 个位置{est}（指导值为 {MIN_A_EXAMPLES} 个示例、"
                     f"{MIN_A_ROWS} 个位置以上）。用少数位置求平均时，键更容易对那个 token 或示例正文产生反应，而不是状态。"
                     + (f"请增加问题后重新运行 behave，或降低 min_rate（种子比例不足的问题 {len(partial)} 个）"
                        if partial else "请增加问题后重新运行 behave"), "few")
        if partial:
            notes.append(f"仅在部分种子中出现行为的问题 {len(partial)} 个未纳入（min_rate {mr:g}）")

    # 7. dont_touch の ordinary = CHiPS の対照集合の B 側の質問(user_prompts と同じ質問は外す)
    picked_t = {qt[i] for i in picked}
    b_all = cs.prompts("B")
    ordinary = [q for q in b_all if question_text(q) not in picked_t]
    removed = len(b_all) - len(ordinary)
    if removed:
        warn(f"CHiPS 对照集 B 侧问题中有 {removed} 个与 user_prompts 相同，因此已从 dont_touch 的 {GROUP} 中移除"
             "（在 system 创建两侧的对照集中，两侧问题是相同的）", "b_overlap")
    if rows and not ordinary:
        warn(f"dont_touch 的 {GROUP} 中没有可放入的问题了。不希望写入的位置只剩提取问题的回复，键更容易在其他回复开头"
             "产生反应。请将普通问题加入 dont_touch", "no_ordinary")

    meta = {"behave": bname, "recipe": rname, "direction": dname, "contrast": cs.name, "n_prompts": len(qs), "n_seeds": n_runs,
            "n_extracted": len(rows), "n_ordinary": len(ordinary), "positions": pos.text, "method": method, "labels": lab,
            "match": None if method == "label" else jrx.pattern, "match_ignore_case": jic, "min_rate": mr,
            "replies": use_replies, "max_tokens": int(bd.get("max_tokens") or 96)}
    spec: dict = {"about": _about(meta), "name": name, "positions": pos.text, "user_prompts": [qs[i] for i in picked]}
    if use_replies:
        spec["replies"] = [chosen[i].get("text") or "" for i in picked]
    if ordinary:
        spec["dont_touch"] = [{"name": GROUP, "prompts": ordinary}]
    if isinstance(vraw, dict) and vraw:
        spec["validate"] = vraw
    keyed = {"direction": dname, "alpha": -1, "iv": iv_ref}
    if not use_replies:
        keyed["max_tokens"] = meta["max_tokens"]
    spec["keyed"] = keyed

    path = None
    if rows:
        d, _ = clean_spec(spec)                                   # 鍵の対照集合として読めるか(読めなければ書かない)
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / f"{name}.yaml"
            tp.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
            read_key_spec(tp)
        path = Path(out) if out else session.store._dir("keyed") / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            notes.append(f"{path.name} 已覆盖写入")
        path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "path": None if path is None else str(path), "behave": bname, "behave_path": str(bpath),
            "recipe": rname, "iv": iv_ref, "direction": dname, "contrast": cs.name, "contrast_path": str(cpath),
            "n_prompts": len(qs), "n_seeds": n_runs,
            "judge": {"method": method, "source": jsrc, "match": meta["match"], "ignore_case": jic, "labels": lab, "min_rate": mr},
            "extracted": rows, "n_partial": len(partial), "n_none": len(qs) - len(rows) - len(partial),
            "positions": pos.text, "positions_source": pos_src,
            "position_share": {"share": share, "starts": st, "tokens": round(tk, 1), "threshold": BROAD_SHARE, "broad": broad},
            "no_match": len(nomatch), "halves": halves, "replies": use_replies,
            "ordinary": {"n": len(ordinary), "removed": removed}, "checks": checks, "keyed": keyed,
            "warnings": warnings, "issues": issues, "notes": notes}

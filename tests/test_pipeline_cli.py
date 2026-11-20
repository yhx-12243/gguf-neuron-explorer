"""heads pipeline の CLI の表示(v0.9.0): 作り済みの方向から流したとき(--from-direction)の出どころと知らせ。モデル不要。"""


def test_from_direction_text():
    from gne.cli import _from_dir_text
    assert _from_dir_text({"direction": {"positions": "last"}}) == ""
    t = _from_dir_text({"direction": {"from": "geo_u", "from_note": "方向 geo_u 的对照集合与…"}})
    assert "已复制并使用生成好的方向 geo_u（跳过了方向构建）" in t and "注意：方向 geo_u 的对照集合与" in t
    assert "注意" not in _from_dir_text({"direction": {"from": "geo_u"}})

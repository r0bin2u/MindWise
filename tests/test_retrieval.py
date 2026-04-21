from app.services.retrieval import format_passages


def test_format_passages_empty():
    assert format_passages([]) == "(未检索到相关资料)"


def test_format_passages_single():
    ps = [{"text": "焦虑是一种情绪", "source": "anxiety.md", "hit_idx": 0}]
    out = format_passages(ps)
    assert "anxiety.md" in out
    assert "焦虑是一种情绪" in out
    assert "资料 1" in out


def test_format_passages_multiple():
    ps = [
        {"text": "第一段", "source": "a.md", "hit_idx": 0},
        {"text": "第二段", "source": "b.md", "hit_idx": 1},
    ]
    out = format_passages(ps)
    assert "资料 1" in out
    assert "资料 2" in out
    assert out.index("第一段") < out.index("第二段")

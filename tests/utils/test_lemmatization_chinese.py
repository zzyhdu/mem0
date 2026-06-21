import pytest


def test_chinese_text_uses_jieba_search_tokens():
    pytest.importorskip("jieba")

    from mem0.utils.lemmatization import lemmatize_for_bm25

    result = lemmatize_for_bm25("我喜欢深色主题和Python编程")
    tokens = result.split()

    assert "深色" in tokens
    assert "主题" in tokens
    assert "python" in tokens
    assert "编程" in tokens

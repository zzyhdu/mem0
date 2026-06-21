"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_TOKEN_RE = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)
_jieba_unavailable_logged = False


def _contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _tokenize_cjk_for_bm25(text: str) -> str:
    global _jieba_unavailable_logged

    try:
        import jieba
    except ImportError:
        if not _jieba_unavailable_logged:
            logger.warning("jieba is not installed. Install it with: pip install mem0ai[nlp]")
            _jieba_unavailable_logged = True
        return text

    tokens = []
    for token in jieba.cut_for_search(text.lower()):
        token = token.strip()
        if token and _TOKEN_RE.fullmatch(token):
            tokens.append(token)

    return " ".join(tokens) if tokens else text


def lemmatize_for_bm25(text: str) -> str:
    """Lemmatize text for BM25 matching.

    Returns space-joined lemmas for full-text search. Falls back to
    the original text if spaCy is unavailable.
    """
    if _contains_cjk(text):
        return _tokenize_cjk_for_bm25(text)

    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)

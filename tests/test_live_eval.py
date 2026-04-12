import json
import os
from pathlib import Path

import pytest

import rag_gemini_test as rag


GOLDEN_FILE = Path(__file__).parent / "retrieval_goldens.json"


def _is_true_env(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _load_enabled_goldens() -> list[dict]:
    if not GOLDEN_FILE.exists():
        return []

    data = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict) and item.get("enabled") is True]


def _matches_expectation(match: dict, expected_source_contains: str, expected_page):
    source = str(match.get("source", "")).lower()
    page = str(match.get("page", "")).strip()

    source_ok = expected_source_contains.lower() in source
    if expected_page is None:
        return source_ok

    return source_ok and page == str(expected_page)


def _has_all_expected_sources(matches: list[dict], expected_sources_contains: list[str]) -> bool:
    """
    Tarkistaa, että kaikille odotetuille lähdeviitteille löytyy osuma top-k tuloksista.
    """
    normalized_sources = [str(m.get("source", "")).lower() for m in matches]

    for expected in expected_sources_contains:
        token = str(expected).strip().lower()
        if not token:
            return False
        if not any(token in src for src in normalized_sources):
            return False

    return True


@pytest.mark.live_retrieval
def test_live_retrieval_recall_at_k():
    """
    Live retrieval -arviointi kultadatalla.
    Ajetaan vain, jos RUN_LIVE_RETRIEVAL_TESTS=true.
    """
    if not _is_true_env("RUN_LIVE_RETRIEVAL_TESTS"):
        pytest.skip("Live retrieval -testit pois päältä. Aseta RUN_LIVE_RETRIEVAL_TESTS=true.")

    goldens = _load_enabled_goldens()
    if not goldens:
        pytest.skip("Ei aktiivisia golden-kysymyksiä. Aseta retrieval_goldens.json-kohdissa enabled=true.")

    top_k = int(os.getenv("LIVE_RETRIEVAL_TOP_K", str(rag.TOP_K)))
    min_recall = float(os.getenv("LIVE_RETRIEVAL_MIN_RECALL", "0.70"))

    index = rag.connect_to_pinecone_index()
    stats = index.describe_index_stats()
    namespace_info = (stats.get("namespaces", {}) or {}).get(rag.PINECONE_NAMESPACE, {})
    if namespace_info.get("vector_count", 0) <= 0:
        pytest.skip("Namespace on tyhjä. Aja indeksointi ensin rag_gemini_test.py:llä.")

    hits = 0
    misses = []

    for item in goldens:
        question = str(item.get("question", "")).strip()
        expected_source_contains = str(item.get("expected_source_contains", "")).strip()
        expected_sources_contains = item.get("expected_sources_contains", [])
        expected_page = item.get("expected_page")

        retrieval_query = rag.normalize_spoken_query(question)
        matches = rag.retrieve_context(index, retrieval_query, top_k=top_k)
        matches = rag.filter_matches_by_score(matches, min_score=rag.RETRIEVAL_SCORE_THRESHOLD)

        if isinstance(expected_sources_contains, list) and expected_sources_contains:
            found = _has_all_expected_sources(matches, expected_sources_contains)
        else:
            found = any(
                _matches_expectation(m, expected_source_contains=expected_source_contains, expected_page=expected_page)
                for m in matches
            )

        if found:
            hits += 1
        else:
            misses.append(
                {
                    "question": question,
                    "expected_source_contains": expected_source_contains,
                    "expected_sources_contains": expected_sources_contains,
                    "expected_page": expected_page,
                    "returned_sources": [f"{m.get('source')} (s.{m.get('page')})" for m in matches],
                }
            )

    recall = hits / len(goldens)

    assert recall >= min_recall, (
        f"Recall@{top_k} jäi alle rajan: {recall:.2%} < {min_recall:.2%}. "
        f"Hutit: {misses}"
    )


@pytest.mark.live_e2e
def test_live_e2e_single_question_smoke():
    """
    Live E2E -smoke testi yhdelle kysymykselle.
    Ajetaan vain, jos RUN_LIVE_E2E_TESTS=true.
    """
    if not _is_true_env("RUN_LIVE_E2E_TESTS"):
        pytest.skip("Live E2E -testit pois päältä. Aseta RUN_LIVE_E2E_TESTS=true.")

    query = os.getenv("LIVE_E2E_QUERY", "Mikä kampanja on tällä hetkellä voimassa?").strip()

    index = rag.connect_to_pinecone_index()
    stats = index.describe_index_stats()
    namespace_info = (stats.get("namespaces", {}) or {}).get(rag.PINECONE_NAMESPACE, {})
    if namespace_info.get("vector_count", 0) <= 0:
        pytest.skip("Namespace on tyhjä. Aja indeksointi ensin rag_gemini_test.py:llä.")

    rag.GLOBAL_INDEX = index
    response = rag.rag_chat(query, history=[])

    assert isinstance(response, str)
    assert response.strip() != ""
    assert "Vasteaika:" in response

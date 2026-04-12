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
    normalized_sources = [str(m.get("source", "")).lower() for m in matches]

    for expected in expected_sources_contains:
        token = str(expected).strip().lower()
        if not token:
            return False
        if not any(token in src for src in normalized_sources):
            return False

    return True


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def _evaluate_recall(index, goldens: list[dict], top_k: int, threshold: float) -> tuple[float, list[dict]]:
    hits = 0
    misses: list[dict] = []

    for item in goldens:
        question = str(item.get("question", "")).strip()
        expected_source_contains = str(item.get("expected_source_contains", "")).strip()
        expected_sources_contains = item.get("expected_sources_contains", [])
        expected_page = item.get("expected_page")

        retrieval_query = rag.normalize_spoken_query(question)
        matches = rag.retrieve_context(index, retrieval_query, top_k=top_k)
        matches = rag.filter_matches_by_score(matches, min_score=threshold)

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
                    "returned_sources": [f"{m.get('source')} (s.{m.get('page')}, score={m.get('score', 0.0):.4f})" for m in matches],
                }
            )

    recall = hits / len(goldens)
    return recall, misses


@pytest.mark.live_retrieval_calibration
def test_live_retrieval_threshold_calibration():
    """
    Kalibroi retrieval-kynnystä ajamalla sama golden-joukko useilla score-rajoilla.
    Ajetaan vain, jos RUN_LIVE_RETRIEVAL_CALIBRATION=true.
    """
    if not _is_true_env("RUN_LIVE_RETRIEVAL_CALIBRATION"):
        pytest.skip(
            "Kalibrointitesti pois päältä. Aseta RUN_LIVE_RETRIEVAL_CALIBRATION=true."
        )

    goldens = _load_enabled_goldens()
    if not goldens:
        pytest.skip("Ei aktiivisia golden-kysymyksiä. Aseta retrieval_goldens.json-kohdissa enabled=true.")

    top_k = int(os.getenv("LIVE_RETRIEVAL_TOP_K", str(rag.TOP_K)))
    min_recall = float(os.getenv("LIVE_RETRIEVAL_MIN_RECALL", "0.70"))
    thresholds = _parse_thresholds(
        os.getenv("LIVE_RETRIEVAL_THRESHOLDS", "0.45,0.50,0.55,0.60,0.65,0.70")
    )

    if not thresholds:
        pytest.fail("LIVE_RETRIEVAL_THRESHOLDS ei sisältänyt yhtään arvoa.")

    index = rag.connect_to_pinecone_index()
    stats = index.describe_index_stats()
    namespace_info = (stats.get("namespaces", {}) or {}).get(rag.PINECONE_NAMESPACE, {})
    if namespace_info.get("vector_count", 0) <= 0:
        pytest.skip("Namespace on tyhjä. Aja indeksointi ensin rag_gemini_test.py:llä.")

    rows: list[tuple[float, float]] = []
    best_threshold = None
    best_recall = -1.0
    best_misses: list[dict] = []

    for threshold in thresholds:
        recall, misses = _evaluate_recall(index=index, goldens=goldens, top_k=top_k, threshold=threshold)
        rows.append((threshold, recall))

        if recall > best_recall:
            best_recall = recall
            best_threshold = threshold
            best_misses = misses

    lines = ["", "Retrieval-kalibrointi (Recall@k):", f"top_k={top_k}, goldens={len(goldens)}"]
    for threshold, recall in rows:
        lines.append(f"- threshold={threshold:.2f} -> recall={recall:.2%}")

    print("\n".join(lines))

    assert best_recall >= min_recall, (
        f"Paras recall jäi alle tavoitteen: {best_recall:.2%} < {min_recall:.2%}. "
        f"best_threshold={best_threshold}, misses={best_misses}"
    )

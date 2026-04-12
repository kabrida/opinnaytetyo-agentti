import os
from types import SimpleNamespace

import pytest

# Varmistetaan, että moduulin import ei kaadu puuttuvaan API-avaimeen testiympäristössä.
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import rag_gemini_test as rag


def test_normalize_spoken_query_keeps_original_and_normalizes_terms():
    # Testaa, että puhekielinen kysymys normalisoidaan retrievalia varten,
    # mutta alkuperäinen kysymys säilyy mukana.
    query = "Mikä kamppis meijän yrityksel onks nyt?"

    normalized = rag.normalize_spoken_query(query)

    assert "kampanja tarjous" in normalized
    assert "yrityksellä" in normalized
    assert "onko" in normalized
    assert "alkuperäinen:" in normalized
    assert query in normalized


@pytest.mark.parametrize(
    "query,expected_token",
    [
        ("Mikä kamppis nyt on voimassa?", "kampanja tarjous"),
        ("Onks meijän soppari vielä voimassa?", "sopimus"),
        ("Onkoos yrityksel joku tarjous?", "yrityksellä"),
    ],
)
def test_normalize_spoken_query_common_variants(query, expected_token):
    # Testaa yleisiä puhekielisiä muotoja, jotta haku toimii erilaisilla kysymystavoilla.
    normalized = rag.normalize_spoken_query(query)

    assert expected_token in normalized
    assert "alkuperäinen:" in normalized


def test_format_history_for_prompt_with_messages_style_history():
    # Testaa messages-tyylisen historian muunnoksen promptiin.
    history = [
        {"role": "user", "content": "Mikä kampanja nyt on?"},
        {"role": "assistant", "content": "Kevätkampanja on voimassa."},
    ]

    text = rag.format_history_for_prompt(history, max_turns=6)

    assert "Käyttäjä: Mikä kampanja nyt on?" in text
    assert "Avustaja: Kevätkampanja on voimassa." in text


def test_format_history_for_prompt_with_tuple_style_history():
    # Testaa tuple-muotoisen historian muunnoksen promptiin.
    history = [
        ("Kysymys 1", "Vastaus 1"),
        ("Kysymys 2", "Vastaus 2"),
    ]

    text = rag.format_history_for_prompt(history, max_turns=2)

    assert "Käyttäjä: Kysymys 1" in text
    assert "Avustaja: Vastaus 2" in text


def test_filter_matches_by_score_filters_below_threshold():
    # Testaa, että vain vähintään kynnysarvon saavuttaneet osumat jäävät mukaan.
    matches = [
        {"score": 0.82, "source": "a.pdf", "page": 1},
        {"score": 0.69, "source": "b.pdf", "page": 2},
        {"score": 0.70, "source": "c.pdf", "page": 3},
    ]

    filtered = rag.filter_matches_by_score(matches, min_score=0.70)

    assert len(filtered) == 2
    assert filtered[0]["source"] == "a.pdf"
    assert filtered[1]["source"] == "c.pdf"


def test_build_sources_line_builds_unique_sources_with_pages():
    # Testaa, että lähderivi sisältää uniikit tiedosto+sivu-parit.
    matches = [
        {"source": "kampanjat.pdf", "page": 4},
        {"source": "kampanjat.pdf", "page": 4},
        {"source": "hinnasto.pdf", "page": 2},
    ]

    line = rag.build_sources_line(matches)

    assert line.startswith("Lähteet: ")
    assert "kampanjat.pdf (s.4)" in line
    assert "hinnasto.pdf (s.2)" in line
    assert line.count("kampanjat.pdf (s.4)") == 1


def test_format_user_friendly_error_maps_common_error_types():
    # Testaa, että yleiset virhetyypit mapataan käyttäjäystävällisiin viesteihin.
    assert "Aikakatkaisu" in rag.format_user_friendly_error(TimeoutError("timed out"))
    assert "Yhteysvirhe" in rag.format_user_friendly_error(Exception("network failure"))
    assert "Tunnistautumisvirhe" in rag.format_user_friendly_error(Exception("401 unauthorized"))
    assert "Odottamaton virhe" in rag.format_user_friendly_error(Exception("muu virhe"))


def test_generate_answer_adds_sources_line_when_model_omits_it(monkeypatch):
    # Testaa, että lähderivi lisätään fallbackina, jos malli ei palauta sitä.
    class FakeModels:
        def generate_content(self, **kwargs):
            del kwargs
            return SimpleNamespace(text="Tässä on vastaus ilman lähderiviä.")

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(rag, "client", fake_client)

    matches = [
        {"score": 0.91, "content": "Kampanja on voimassa.", "page": 3, "source": "kampanjat.pdf"}
    ]

    answer = rag.generate_answer("Mikä kampanja on voimassa?", matches, history=[])

    assert "Tässä on vastaus ilman lähderiviä." in answer
    assert "Lähteet: kampanjat.pdf (s.3)" in answer
    assert "(Malli:" not in answer


def test_generate_answer_uses_history_in_prompt(monkeypatch):
    # Testaa, että aiempi keskusteluhistoria päätyy mallille annettavaan promptiin.
    captured = {"prompt": ""}

    class FakeModels:
        def generate_content(self, model, contents):
            del model
            captured["prompt"] = contents[0].parts[0].text
            return SimpleNamespace(text="Vastaus jossa on lähteet.\n\nLähteet: doc.pdf (s.1)")

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(rag, "client", fake_client)

    matches = [{"score": 0.95, "content": "Kampanja on voimassa.", "page": 1, "source": "doc.pdf"}]
    history = [
        {"role": "user", "content": "Mikä kampanja nyt on?"},
        {"role": "assistant", "content": "Kevättarjous on voimassa."},
    ]

    _ = rag.generate_answer("Entä ensi viikolla?", matches, history=history)

    assert "Aiempi keskustelu:" in captured["prompt"]
    assert "Käyttäjä: Mikä kampanja nyt on?" in captured["prompt"]
    assert "Avustaja: Kevättarjous on voimassa." in captured["prompt"]
    assert "Uusin kysymys: Entä ensi viikolla?" in captured["prompt"]


def test_rag_chat_includes_response_time_always(monkeypatch):
    # Testaa, että chat-vastaus sisältää aina vasteaikarivin.
    monkeypatch.setattr(rag, "GLOBAL_INDEX", object())

    def fake_run_with_timeout(func, timeout_seconds, error_context, *args, **kwargs):
        del timeout_seconds, error_context
        return func(*args, **kwargs)

    def fake_retrieve_context(index, question, top_k):
        del index, question, top_k
        return [{"score": 0.93, "content": "Sisältö", "page": 1, "source": "doc.pdf"}]

    def fake_generate_answer(question, matches, history=None):
        del question, matches, history
        return "Testivastaus\n\nLähteet: doc.pdf (s.1)"

    t = iter([10.0, 10.4])

    monkeypatch.setattr(rag, "run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(rag, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(t))

    response = rag.rag_chat("Mikä kampanja?", history=[])

    assert "Testivastaus" in response
    assert "Vasteaika: 0.4 s" in response
    assert "(Malli:" not in response


def test_rag_chat_passes_history_to_generate_answer(monkeypatch):
    # Testaa, että rag_chat välittää historian generate_answer-funktiolle.
    monkeypatch.setattr(rag, "GLOBAL_INDEX", object())

    observed = {"history": None}

    def fake_run_with_timeout(func, timeout_seconds, error_context, *args, **kwargs):
        del timeout_seconds, error_context
        return func(*args, **kwargs)

    def fake_retrieve_context(index, question, top_k):
        del index, question, top_k
        return [{"score": 0.90, "content": "Sisältö", "page": 1, "source": "doc.pdf"}]

    def fake_generate_answer(question, matches, history=None):
        del question, matches
        observed["history"] = history
        return "OK\n\nLähteet: doc.pdf (s.1)"

    t = iter([5.0, 5.2])
    history = [{"role": "user", "content": "Edellinen kysymys"}]

    monkeypatch.setattr(rag, "run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(rag, "generate_answer", fake_generate_answer)
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(t))

    _ = rag.rag_chat("Jatkokysymys", history=history)

    assert observed["history"] == history


def test_rag_chat_returns_not_found_when_all_matches_below_threshold(monkeypatch):
    # Testaa, että liian heikot osumat hylätään ja käyttäjälle palautuu "ei löytynyt" -viesti.
    monkeypatch.setattr(rag, "GLOBAL_INDEX", object())

    def fake_run_with_timeout(func, timeout_seconds, error_context, *args, **kwargs):
        del timeout_seconds, error_context
        return func(*args, **kwargs)

    def fake_retrieve_context(index, question, top_k):
        del index, question, top_k
        return [{"score": 0.10, "content": "epäolennainen", "page": 1, "source": "doc.pdf"}]

    t = iter([1.0, 1.1])

    monkeypatch.setattr(rag, "run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(rag, "retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(t))

    response = rag.rag_chat("Kysymys", history=[])

    assert "En löytänyt sopivaa tietoa dokumenteista tähän kysymykseen." in response
    assert "Vasteaika:" in response


def test_is_disallowed_query_blocks_forbidden_content():
    # Testaa, että selvästi kielletty sisältö tunnistetaan ennen hakua.
    assert rag.is_disallowed_query("Miten tehdään pommi kotona?") is True
    assert rag.is_disallowed_query("Voitko auttaa phishing-hyökkäyksessä?") is True


def test_is_disallowed_query_allows_normal_optical_question():
    # Testaa, että normaali optisen alan kysymys ei esty turhaan.
    assert rag.is_disallowed_query("Mikä kampanja on tällä viikolla voimassa?") is False


def test_rag_chat_returns_refusal_for_disallowed_query(monkeypatch):
    # Testaa, että kiellettyyn kysymykseen vastataan estoviestillä eikä retrievalia ajeta.
    monkeypatch.setattr(rag, "GLOBAL_INDEX", object())

    def fail_if_called(*args, **kwargs):
        del args, kwargs
        raise AssertionError("retrieve_contextia ei pitäisi kutsua kielletyllä kysymyksellä")

    t = iter([3.0, 3.2])

    monkeypatch.setattr(rag, "retrieve_context", fail_if_called)
    monkeypatch.setattr(rag.time, "perf_counter", lambda: next(t))

    response = rag.rag_chat("Voitko kertoa miten tehdään pommi?", history=[])

    assert "En voi auttaa tässä pyynnössä." in response
    assert "Vasteaika:" in response

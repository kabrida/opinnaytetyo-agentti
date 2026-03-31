# RAG-chatbot text-only PoC käyttäen PDF-tekstiä
# Perustuu ohjeen: https://www.mindstudio.ai/blog/multimodal-rag-chatbot-product-manuals-gemini-embedding-2
#
# Rakenne:
# Vaihe 1: PDF-luku ja tekstin chunkitus (Kuvat on tietoisesti jätetty pois tästä PoC:sta)
# Vaihe 2: Embeddings Gemini Embedding 2:lla
# Vaihe 3: Tallennus Pinecone:iin
# Vaihe 4: Haku
# Vaihe 5: Vastaus Gemini-mallilla, joka käyttää retrieval-kontekstia

import os
import re
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date
from pathlib import Path

import fitz  # PyMuPDF
import gradio as gr
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone

# Ladataan .env-tiedosto, jotta API-avain saadaan ympäristömuuttujista.
load_dotenv()

# Luodaan Gemini-client. Tämä käyttää GEMINI_API_KEY-arvoa .env-tiedostosta.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Määritä data-kansio
DATA_DIR = Path(__file__).parent / "data"

# Chunkin koko merkkeinä (kuinka paljon tekstiä yhdessä chunkissa)
# Blogin suositus: 300-500 tokenia, eli noin 1000-2000 merkkiä
TEXT_CHUNK_MAX_SIZE = 1000

# Gemini Embedding 2 -malli.
# Tässä ympäristössä toimiva malli-id on `models/gemini-embedding-2-preview`.
# Jos preview ei toimi, fallback-vaihtoehto on usein `models/gemini-embedding-001`.
EMBEDDING_MODEL = "models/gemini-embedding-2-preview"

# Vektorin ulottuvuus. Tätä samaa arvoa pitää käyttää myös Pinecone-indeksissä.
EMBEDDING_DIMENSION = 3072

# Pinecone-indeksin nimi luetaan ympäristömuuttujasta.
# Jos nimeä ei ole määritetty .env-tiedostossa, käytetään oletusarvoa.
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "optikko-rag-index")

# Namespace erottaa tämän PoC:n vektorit mahdollisista aiemmista kokeiluista.
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "all-pdf-files")

# Jos haluat pakottaa uudelleenindeksoinnin, aseta .env: FORCE_REINDEX=true
FORCE_REINDEX = os.getenv("FORCE_REINDEX", "false").lower() == "true"

# Kuinka monta hakutulosta palautetaan retrieval-vaiheessa.
TOP_K = 3

# Jos vastaus kestää tätä pidempään, käyttäjälle näytetään viivehuomio.
SLOW_RESPONSE_THRESHOLD_SECONDS = 10.0

# Aikakatkaisut verkkokutsuille, jotta sovellus ei jää odottamaan loputtomasti.
RETRIEVAL_TIMEOUT_SECONDS = float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "8"))
GENERATION_TIMEOUT_SECONDS = float(os.getenv("GENERATION_TIMEOUT_SECONDS", "20"))
STARTUP_TIMEOUT_SECONDS = float(os.getenv("STARTUP_TIMEOUT_SECONDS", "25"))

# Testikysymys retrieval-vaiheelle.
TEST_QUERY = "Mikä kamppis nyt on voimassa?"

# Generointimallit järjestyksessä: ensin kokeillaan 2.5-flashia,
# ja jos se epäonnistuu (quota/permission/not found), kokeillaan seuraavia.
GENERATION_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash"
]

# System-prompt, joka pakottaa mallin pysymään annetussa kontekstissa.
SYSTEM_PROMPT = """
Olet avulias optisen alan asiantuntija.
Vastaa käyttäjän kysymykseen VAIN annetun kontekstin perusteella.
Jos tietoa ei löydy kontekstista, sano se suoraan äläkä arvaa.
Vastaa suomeksi.
Jos kysymys liittyy esimerkiksi kampanjan voimassaoloon tai ajankohtaan,
hyödynnä myös annettua tämän päivän päivämäärää.
Lisää loppuun lyhyt rivi: 'Lähteet: ...' jossa on lähdetiedosto sekä sivunumero kyseisessä tiedostossa.
""".strip()

# Pidetään index-instanssi muistissa Gradio-kutsuja varten.
GLOBAL_INDEX = None

# Puhekielen normalisointisäännöt retrieval-kyselylle.
# Ideana on muuntaa yleisiä puhekielen termejä hieman muodollisempaan muotoon,
# jotta dokumenttihaku osuu paremmin myös ilman tarkkoja teknisiä sanoja.
SPOKEN_QUERY_RULES: list[tuple[str, str]] = [
    (r"\bkamppis\b|\bkampis\b|\bkampanjaa\b", "kampanja tarjous"),
    (r"\bsoppari\b|\bsopparii\b", "sopimus"),
    (r"\bonks\b|\bonksko\b|\bonkoos\b", "onko"),
    (r"\bmeiä\b|\bmeijän\b|\bmeidän\b", "meidän"),
    (r"\byrityksel\b", "yrityksellä"),
]


def run_with_timeout(func, timeout_seconds: float, error_context: str, *args, **kwargs):
    """
    Suorittaa funktion aikakatkaisun kanssa.

    Tätä käytetään erityisesti verkkoa käyttäviin kutsuihin,
    jotta sovellus ei jää jumiin verkkokatkon aikana.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)

    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as error:
        future.cancel()
        raise TimeoutError(
            f"{error_context} aikakatkaistiin ({timeout_seconds:.0f}s)."
        ) from error
    finally:
        # wait=False varmistaa, ettei käyttöliittymä jää odottamaan taustasäiettä.
        executor.shutdown(wait=False, cancel_futures=True)


def format_user_friendly_error(error: Exception) -> str:
    """
    Muuntaa teknisen virheen käyttäjälle selkeäksi suomenkieliseksi viestiksi.
    """
    message = str(error).lower()

    is_timeout = (
        isinstance(error, TimeoutError)
        or "timeout" in message
        or "timed out" in message
        or "aikakatka" in message
    )
    if is_timeout:
        return "Aikakatkaisu: palvelu ei vastannut ajoissa. Tarkista verkkoyhteys ja yritä uudelleen."

    is_connection_error = (
        "connection" in message
        or "network" in message
        or "dns" in message
        or "getaddrinfo" in message
        or "11001" in message
        or "name resolution" in message
        or "temporary failure in name resolution" in message
        or "max retries exceeded" in message
        or "failed to establish a new connection" in message
        or "ssl" in message
        or "proxy" in message
    )
    if is_connection_error:
        return "Yhteysvirhe: palveluun ei saatu yhteyttä. Tarkista verkkoyhteys ja yritä uudelleen."

    is_auth_error = (
        "401" in message
        or "unauthorized" in message
        or "api key" in message
        or "authentication" in message
    )
    if is_auth_error:
        return "Tunnistautumisvirhe: tarkista API-avaimet (.env)."

    return f"Odottamaton virhe: {error}"

# ==================== VAIHE 1: TEKSTIN EKSTRAKTIO ====================

def extract_text_chunks(pdf_path: str, max_chunk_size: int = TEXT_CHUNK_MAX_SIZE) -> list[dict]:
    """
    Lukee PDF:n sivuittain ja pilkkoo tekstin semanttisiksi chunkeiksi.

    Kunkin chunkin tiedoissa:
    - content: varsinainen teksti
    - page: sivunumero (1-indexed)
    - content_type: "text"
    - source: tiedostonimi

    Parametrit:
    -----------
    pdf_path : str
        Polku PDF-tiedostoon
    max_chunk_size : int
        Maksimi merkkimäärä per chunk

    Palautus:
    ---------
    list[dict]
        Lista chunkeja tiedoitaan
    """
    # Avataan PDF käyttäen PyMuPDF:ää (fitz)
    doc = fitz.open(pdf_path)
    chunks = []

    # Käydään läpi jokainen sivu
    for page_num, page in enumerate(doc):
        # Otetaan sivun teksti
        text = page.get_text("text").strip()

        # Jos sivu on tyhjä, ohitetaan
        if not text:
            continue

        # Pilkotaan teksti kappaleiden mukaan (kaksoisrivinvaihto = kappaleen ero)
        # Näin voidaan säilyttää semantiikka paremmin
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Yhdistetään kappaleet kunnes saavutetaan max_chunk_size
        current_chunk = ""
        for para in paragraphs:
            # Jos jäljellä oleva tila riittää seuraavalle kappaleelle, lisää se
            if len(current_chunk) + len(para) < max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Muussa tapauksessa tallenna nykyinen chunk ja aloita uusi
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "page": page_num + 1,  # 1-indexed sivu-numero
                        "content_type": "text",
                        "source": Path(pdf_path).name
                    })
                current_chunk = para + "\n\n"

        # Tallenna viimeinen chunk
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "page": page_num + 1,
                "content_type": "text",
                "source": Path(pdf_path).name
            })

    doc.close()
    return chunks


def list_pdf_files(data_dir: Path = DATA_DIR) -> list[Path]:
    """
    Hakee kaikki PDF-tiedostot data-kansiosta aakkosjärjestyksessä.
    """
    return sorted(data_dir.glob("*.pdf"))


def extract_all_text_chunks(data_dir: Path = DATA_DIR) -> tuple[list[dict], list[Path]]:
    """
    Lukee kaikki data-kansion PDF:t ja yhdistää niiden chunkit yhteen listaan.
    """
    pdf_files = list_pdf_files(data_dir)
    all_chunks: list[dict] = []

    for pdf_file in pdf_files:
        file_chunks = extract_text_chunks(str(pdf_file))
        all_chunks.extend(file_chunks)

    return all_chunks, pdf_files


# ==================== VAIHE 2: TEKSTIN EMBEDDING ====================

def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """
    Muuttaa yhden tekstin vektoriksi Gemini Embedding 2 -mallilla.

    Miksi tämä tehdään:
    - LLM ei hae dokumentteja suoraan raakatekstistä tehokkaasti.
    - Embedding muuttaa tekstin numerovektoriksi, jota voidaan verrata
      semanttisesti ("merkityksen" perusteella).

    task_type:
    - "RETRIEVAL_DOCUMENT" = käytetään dokumenttichunkeille
    - "RETRIEVAL_QUERY" = käytetään käyttäjän kysymykselle
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION,
        ),
    )

    # SDK palauttaa listan embeddingeistä. Yhden tekstin tapauksessa otetaan
    # ensimmäinen embedding talteen.
    return response.embeddings[0].values


def embed_batch(
    chunks: list[dict],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 5,
    delay: float = 0.5,
) -> list[dict]:
    """
    Embeddaa useita chunk-objekteja peräkkäin.

    Tämä versio tekee tarkoituksella yhden embedding-kutsun per chunk,
    jotta virheiden käsittely on helppo ymmärtää PoC-vaiheessa.

    Lisäksi lisätään kevyt viive (`delay`) jokaisen batchin jälkeen,
    jotta quota/rate limit -virheitä tulee vähemmän.
    """
    embedded_chunks: list[dict] = []

    for index, chunk in enumerate(chunks):
        # Kevyt throttle: jokaisen batch_size chunkin jälkeen pidetään tauko.
        if index > 0 and index % batch_size == 0:
            time.sleep(delay)

        try:
            vector = embed_text(chunk["content"], task_type=task_type)
            embedded_chunks.append(
                {
                    **chunk,
                    "embedding": vector,
                }
            )
        except Exception as error:
            # PoC:ssa jatketaan vaikka yksittäinen chunk epäonnistuisi.
            print(f"Embedding epäonnistui chunkille {index}: {error}")
            continue

    return embedded_chunks


# ==================== VAIHE 3: PINECONE-YHTEYS ====================

def connect_to_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    """
    Yhdistää Pineconen olemassa olevaan indexiin.

    Tässä vaiheessa emme vielä upsertaa dataa, vaan vain varmistamme että:
    1) API-avain toimii
    2) index löytyy
    3) yhteys voidaan muodostaa
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY puuttuu .env-tiedostosta.")

    pc = Pinecone(api_key=api_key)

    # Haetaan indexien nimet ja tarkistetaan löytyykö haluttu index.
    available_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in available_indexes:
        raise ValueError(
            f"Pinecone-indexiä '{index_name}' ei löytynyt. "
            f"Löytyneet indexit: {available_indexes}"
        )

    return pc.Index(index_name)


def upsert_to_pinecone(index, embedded_chunks: list[dict], batch_size: int = 50) -> int:
    """
    Tallentaa embedding-vektorit Pineconeen (upsert).

    Mitä tässä tapahtuu:
    1) Muunnetaan jokainen embedattu chunk Pineconen odottamaan muotoon
       - id: uniikki tunniste
       - values: itse embedding-vektori
       - metadata: chunkin taustatiedot (sisältö, sivu, lähde)
    2) Lähetetään vektorit Pineconeen erissä (batch), jotta lähetys on vakaampi

    Palauttaa tallennettujen vektorien määrän.
    """
    vectors: list[dict] = []

    for chunk in embedded_chunks:
        # Deterministinen ID estää saman chunkin duplikaatit uusinta-ajossa.
        id_seed = f"{chunk.get('source','')}|{chunk.get('page',0)}|{chunk.get('content','')}"
        vector_id = hashlib.sha1(id_seed.encode("utf-8")).hexdigest()

        # Metadata talletetaan hakua ja lähdeviittauksia varten.
        # Pineconessa metadata kannattaa pitää kohtuullisen kokoisena.
        metadata = {
            "content": chunk["content"][:2000],
            "page": chunk.get("page", 0),
            "content_type": chunk.get("content_type", "text"),
            "source": str(chunk.get("source", "")),
        }

        vectors.append(
            {
                "id": vector_id,
                "values": chunk["embedding"],
                "metadata": metadata,
            }
        )

    # Lähetetään data Pineconeen erissä.
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)

    return len(vectors)


# ==================== VAIHE 4: RETRIEVAL (HAKU PINECONESTA) ====================

def retrieve_context(index, question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Hakee Pineconesta kysymykseen liittyvät dokumenttipätkät.

    Mitä tässä tehdään:
    1) Muunnetaan käyttäjän kysymys embedding-vektoriksi (RETRIEVAL_QUERY)
    2) Haetaan Pineconesta lähimmät vektorit (semanttisesti osuvimmat)
    3) Palautetaan osumat metatietoineen (content, page, source, score)
    """
    # Query pitää embeddata eri task_typellä kuin dokumentti.
    query_vector = embed_text(question, task_type="RETRIEVAL_QUERY")

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=PINECONE_NAMESPACE,
    )

    matches = []
    for match in result.get("matches", []):
        metadata = match.get("metadata", {}) or {}
        matches.append(
            {
                "score": match.get("score", 0.0),
                "content": metadata.get("content", ""),
                "page": metadata.get("page", "?"),
                "source": metadata.get("source", ""),
                "content_type": metadata.get("content_type", "text"),
            }
        )

    return matches


def normalize_spoken_query(question: str) -> str:
    """
    Muuntaa puhekielistä kysymystä retrievalille paremmin sopivaan muotoon.

    Huom: Tämä ei muuta käyttäjälle näytettävää kysymystä,
    vaan ainoastaan Pinecone-hakua varten käytettävää versiota.
    """
    normalized = question.strip().lower()

    for pattern, replacement in SPOKEN_QUERY_RULES:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

    # Tiivistetään ylimääräiset välilyönnit.
    normalized = " ".join(normalized.split())

    # Lisätään alkuperäinen kysymys mukaan retrieval-tekstiin,
    # jotta mahdolliset erisnimet (esim. yrityksen nimi) säilyvät varmasti mukana.
    return f"{normalized} | alkuperäinen: {question.strip()}"


def print_retrieval_results(question: str, matches: list[dict]) -> None:
    """
    Tulostaa retrieval-haun tulokset selkeästi.
    """
    print("\n=== VAIHE 4: RETRIEVAL TESTI ===")
    print(f"Kysymys: {question}")

    if not matches:
        print("Hakutuloksia ei löytynyt.")
        return

    print(f"Hakutuloksia: {len(matches)}")
    for i, item in enumerate(matches, start=1):
        preview = item["content"][:220].replace("\n", " ")
        print(f"\n[{i}] score={item['score']:.4f} | sivu={item['page']} | source={item['source']}")
        print(f"{preview}...")


def generate_answer(question: str, matches: list[dict]) -> str:
    """
    Muodostaa lopullisen vastauksen retrieval-hakutulosten pohjalta.

    Mitä tehdään:
    1) Rakennetaan konteksti Pineconesta haetuista chunkeista
    2) Lähetetään konteksti + kysymys Gemini-mallille
    3) Palautetaan mallin vastaus

    Jos osumia ei ole, palautetaan selkeä fallback-vastaus.
    """
    if not matches:
        return "En löytänyt riittävää tietoa dokumenteista tähän kysymykseen."

    context_blocks = []
    for i, item in enumerate(matches, start=1):
        source_label = f"[Lähde {i}: sivu {item['page']}, tiedosto {item['source']}]"
        context_blocks.append(f"{source_label}\n{item['content']}")

    context_text = "\n\n---\n\n".join(context_blocks)

    today_fi = date.today().strftime("%d.%m.%Y")

    user_prompt = (
        f"Tänään on {today_fi}.\n\n"
        f"Konteksti:\n{context_text}\n\n"
        f"Kysymys: {question}\n\n"
        "Vastaa kontekstin perusteella selkeästi ja käytännöllisesti."
    )

    last_error = None

    # Kokeillaan malleja järjestyksessä, jotta PoC ei pysähdy yhden mallin quotaan.
    for model_name in GENERATION_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=f"{SYSTEM_PROMPT}\n\n{user_prompt}")],
                    )
                ],
            )
            answer_text = (response.text or "").strip()
            if answer_text:
                return f"(Malli: {model_name})\n\n{answer_text}"
        except Exception as error:
            last_error = error
            message = str(error).lower()

            # Nämä virheet ovat tyypillisesti mallikohtaisia/quota-ongelmia,
            # joten jatketaan seuraavaan malliin.
            retryable = (
                "429" in message
                or "quota" in message
                or "403" in message
                or "permission" in message
                or "404" in message
                or "not found" in message
            )
            if retryable:
                continue

            # Jos tuli muu virhe, palautetaan se heti.
            return f"Vastauksen muodostus epäonnistui mallilla {model_name}: {error}"

    # Jos mikään malli ei onnistunut, annetaan selkeä fallback-viesti.
    return (
        "Gemini-vastausvaihe ei onnistunut yhdelläkään fallback-mallilla. "
        f"Viimeisin virhe: {last_error}. "
        "Retrieval toimii kuitenkin, joten voit käyttää hakutuloksia suoraan."
    )


def ensure_index_has_all_pdfs(index) -> int:
    """
    Varmistaa, että kaikki data-kansion PDF:t on indeksoitu valittuun namespaceen.

    Logiikka:
    - Jos namespace on tyhjä TAI FORCE_REINDEX=true, indeksoidaan kaikki PDF:t.
    - Muuten käytetään olemassa olevia vektoreita.
    """
    all_chunks, pdf_files = extract_all_text_chunks(DATA_DIR)

    print(f"Löydettyjä PDF-tiedostoja data-kansiosta: {len(pdf_files)}")
    print(f"Muodostettuja tekstichunkeja yhteensä: {len(all_chunks)}")

    if not all_chunks:
        raise ValueError("Data-kansiosta ei löytynyt indeksoitavaa tekstiä.")

    stats = index.describe_index_stats()
    namespace_info = (stats.get("namespaces", {}) or {}).get(PINECONE_NAMESPACE, {})
    namespace_vectors = namespace_info.get("vector_count", 0)

    if FORCE_REINDEX and namespace_vectors > 0:
        print(f"FORCE_REINDEX=true -> tyhjennetään namespace '{PINECONE_NAMESPACE}' ja indeksoidaan uudelleen.")
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
        namespace_vectors = 0

    if namespace_vectors == 0:
        print(f"Namespace '{PINECONE_NAMESPACE}' on tyhjä. Aloitetaan indeksointi...")
        embedded_chunks = embed_batch(
            all_chunks,
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=5,
            delay=0.5,
        )
        print(f"Embeddattuja chunkeja: {len(embedded_chunks)}")

        if not embedded_chunks:
            raise RuntimeError("Yhtään chunkia ei saatu embeddattua indeksointia varten.")

        uploaded_count = upsert_to_pinecone(index, embedded_chunks, batch_size=50)
        print(f"Upsert valmis. Lisättyjä/ylikirjoitettuja vektoreita: {uploaded_count}")

        stats = index.describe_index_stats()
        namespace_info = (stats.get("namespaces", {}) or {}).get(PINECONE_NAMESPACE, {})
        namespace_vectors = namespace_info.get("vector_count", 0)

    print(f"Namespace '{PINECONE_NAMESPACE}' vector count: {namespace_vectors}")
    return namespace_vectors


def rag_chat(message, history):
    """
    Gradio-chatin callback:
    1) hakee top-k osumat Pineconesta
    2) muodostaa grounded-vastauksen
    """
    del history

    if not message or not message.strip():
        return "Kirjoita kysymys ensin."

    if GLOBAL_INDEX is None:
        return "Index-yhteyttä ei ole alustettu. Käynnistä sovellus uudelleen."

    start_time = time.perf_counter()

    try:
        retrieval_query = normalize_spoken_query(message.strip())
        matches = run_with_timeout(
            retrieve_context,
            RETRIEVAL_TIMEOUT_SECONDS,
            "Hakuvaihe",
            GLOBAL_INDEX,
            retrieval_query,
            top_k=TOP_K,
        )
        if not matches:
            answer = "En löytänyt sopivaa tietoa dokumenteista tähän kysymykseen."
        else:
            answer = run_with_timeout(
                generate_answer,
                GENERATION_TIMEOUT_SECONDS,
                "Vastausvaihe",
                message.strip(),
                matches,
            )

        elapsed = time.perf_counter() - start_time
        if elapsed > SLOW_RESPONSE_THRESHOLD_SECONDS:
            delay_note = (
                f"⏳ Vasteaika oli {elapsed:.1f} s (yli {SLOW_RESPONSE_THRESHOLD_SECONDS:.0f} s). "
                "Vastaus saatiin valmiiksi, mutta haku/mallivastaus oli tavallista hitaampi.\n\n"
            )
            return delay_note + answer

        return answer
    except Exception as error:
        return format_user_friendly_error(error)


def launch_gradio_app() -> None:
    """
    Käynnistää Gradio-käyttöliittymän text-only RAG-chatille.
    """
    interface = gr.ChatInterface(
        fn=rag_chat,
        title="Optikko RAG Chat (text-only)",
        description="Kysyy vastaukset data-kansion PDF-tiedostoista Pinecone-retrievalin kautta.",
        examples=[
            "Mikä kampanja on tällä hetkellä voimassa?",
            "Kuinka paljon ohennetut linssit maksavat?",
            "Mitä eroa on Multi Plus ja Multi Comfort -linssien välillä?",
        ],
    )
    interface.launch()


def main():
    """
    Testi: lataa data-kansiosta PDF:n ja tekee tekstin chunkituksen.
    Tässä PoC:ssa kuvia ei käsitellä.
    """
    pdf_files = list_pdf_files(DATA_DIR)
    if not pdf_files:
        print("ERROR: Data-kansiosta ei löytynyt PDF-tiedostoja.")
        return

    print("=== RAG KÄYNNISTYS ===")
    print(f"PDF-tiedostoja löytyi: {len(pdf_files)}")

    index = None
    try:
        index = run_with_timeout(
            connect_to_pinecone_index,
            STARTUP_TIMEOUT_SECONDS,
            "Pinecone-yhteyden alustus",
        )
        print(f"Yhteys Pinecone-indexiin onnistui: {PINECONE_INDEX_NAME}")
    except Exception as error:
        print(f"Pinecone-yhteystesti epäonnistui: {format_user_friendly_error(error)}")
        return

    try:
        total_vectors = run_with_timeout(
            ensure_index_has_all_pdfs,
            STARTUP_TIMEOUT_SECONDS,
            "Indeksointi",
            index,
        )
    except Exception as error:
        print(f"Indeksointi epäonnistui: {format_user_friendly_error(error)}")
        return

    if total_vectors <= 0:
        print("Indexissä ei ole vektoreita. Sovellusta ei käynnistetä.")
        return

    # Tallennetaan index globaaliin muuttujaan, jotta Gradio-callback käyttää samaa yhteyttä.
    global GLOBAL_INDEX
    GLOBAL_INDEX = index

    # Pieni smoke test ennen UI:ta.
    print("\n=== RETRIEVAL SMOKE TEST ===")
    smoke_query = normalize_spoken_query(TEST_QUERY)
    try:
        matches = run_with_timeout(
            retrieve_context,
            RETRIEVAL_TIMEOUT_SECONDS,
            "Smoke test retrieval",
            index,
            smoke_query,
            top_k=TOP_K,
        )
    except Exception as error:
        print(f"Smoke test retrieval epäonnistui: {format_user_friendly_error(error)}")
        matches = []

    print_retrieval_results(TEST_QUERY, matches)

    print("\n=== GROUNDED ANSWER SMOKE TEST ===")
    try:
        smoke_answer = run_with_timeout(
            generate_answer,
            GENERATION_TIMEOUT_SECONDS,
            "Smoke test vastaus",
            TEST_QUERY,
            matches,
        )
        print(smoke_answer)
    except Exception as error:
        print(f"Smoke test vastaus epäonnistui: {format_user_friendly_error(error)}")

    print("\n=== GRADIO KÄYNNISTETÄÄN ===")
    launch_gradio_app()


if __name__ == "__main__":
    main()

# Optikko-Agentti PoC (Proof of Concept)

Tämä projekti on opinnäytetyöhön liittyvä tekoälypohjainen chatbot-kokeilu, joka keskittyy optisen alan sisältöön.

## Projektin vaiheet

### Vaihe 1: Projektin pystytys + Gemini-chat
- Tiedosto: `gemini_testi.py`
- Toteutettiin peruschat Gemini API:lla ja Gradio-käyttöliittymällä.
- Fokus: yhteyden toimivuus, roolitus ja ensimmäinen käyttöliittymä.

### Vaihe 2: Testitiedostojen lukukokeilu
- Tiedosto: `read_file_test.py`
- Tarkoitus oli lukea testisisältöjä tiedostoista ja käyttää niitä vastauksissa.
- Tila: **ei enää aktiivinen / ei toimi nykyarkkitehtuurissa** (pidetään historiatiedostona).

### Vaihe 3 (nykyinen): RAG + Pinecone + Gradio
- Tiedosto: `rag_gemini_test.py`
- Nykyinen toteutus on **text-only RAG PoC**:
  - lukee PDF-tiedostot `data/`-kansiosta
  - pilkkoo tekstin chunkeiksi
  - luo embeddingit (`gemini-embedding-2-preview`)
  - tallentaa vektorit Pineconeen
  - hakee kysymykselle top-k osumat
  - muodostaa grounded-vastauksen fallback-malleilla
  - tarjoaa Gradio-chat-käyttöliittymän

## Teknologiat
- **Kieli:** Python 3.12+
- **LLM / Embeddings:** Google Gemini (`google-genai`)
- **Vector DB:** Pinecone
- **PDF-luku:** PyMuPDF (`fitz`)
- **UI:** Gradio
- **Ympäristönhallinta:** Python venv + `python-dotenv`

## Asennus ja käyttö
1. Luo virtuaaliympäristö:
	- `python -m venv .venv`
2. Aktivoi ympäristö:
	- PowerShell: `.\.venv\Scripts\Activate.ps1`
3. Asenna riippuvuudet:
	- `pip install -r requirements.txt`
4. Lisää `.env`-tiedostoon vähintään:
	- `GEMINI_API_KEY=...`
	- `PINECONE_API_KEY=...`
	- `PINECONE_INDEX_NAME=optikko-rag-index`
	- `FORCE_REINDEX=false`
5. Aja sovellus:
	- `python rag_gemini_test.py`
6. Avaa Gradio:
	- `http://127.0.0.1:7860`

## Huomioita nykyisestä toteutuksesta
- RAG hakee tiedon kaikista `data/`-kansion PDF-tiedostoista.
- Kuvat on tietoisesti jätetty pois PoC-versiosta (text-only).
- Jos aineisto muuttuu, voit pakottaa uudelleenindeksoinnin:
  - aseta `.env`: `FORCE_REINDEX=true`, aja kerran, palauta sen jälkeen `false`.

## Tiedossa olevat puutteet / jatkokehitys
- Keskusteluhistorian hyödyntäminen monivuoroisessa chatissa (stateful context).
- Virheellisen tai vanhentuneen datan hallinta (esim. vanhat kampanjat/sopimukset).
- Kehysdatan lisääminen kokonaishinnan laskentaa varten.

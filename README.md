# Optikko-Agentti PoC (Proof of Concept)

Tämä projekti on opinnäytetyöni ensimmäinen tekninen toteutusvaihe. Kyseessä on tekoälypohjainen chatbot, joka on erikoistunut optisen alan neuvontaan.

## Nykytila
Projektissa on pystytetty toimiva Python-ympäristö ja toteutettu yhteys uusimpaan **Gemini API** -rajapintaan. Käyttöliittymänä toimii **Gradio**.

### Keskeiset ominaisuudet:
* **Roolitus:** Tekoäly on ohjeistettu toimimaan asiantuntijana ja se kieltäytyy vastaamasta aiheen ulkopuolisiin kysymyksiin.
* **Moderni arkkitehtuuri:** Käytössä on uusin `google-genai` -kirjasto ja `gemini-flash-latest` -malli.

## Teknologiat
* **Kieli:** Python 3.12+
* **LLM:** Google Gemini (Flash-sarja)
* **UI:** Gradio
* **Ympäristönhallinta:** Python venv & `python-dotenv`

## Asennus ja käyttö
1. Kloonaa repositorio.
2. Luo virtuaaliympäristö: `python -m venv .venv`
3. Aktivoi ympäristö ja asenna riippuvuudet: `pip install -r requirements.txt`
4. Lisää oma API-avaimesi `.env`-tiedostoon (`GEMINI_API_KEY=tähän_avain`).
5. Aja sovellus: `python gemini_testi.py`

## Seuraavat askeleet
Seuraavaksi projektiin toteutetaan **RAG (Retrieval-Augmented Generation)** -toiminnallisuus. Tavoitteena on opettaa agentti lukemaan yrityksen omia PDF-ohjeistuksia ja kampanjataulukoita, jotta vastaukset perustuvat faktuaaliseen myymälädataan yleistiedon sijaan.

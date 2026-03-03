# Lähteenä: https://ai.google.dev/gemini-api/docs/file-input-methods

from google import genai
from google.genai import types
import pathlib
import os
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
client = genai.Client()

# Tiedostopolkujen määrittäminen
hinnasto_path = pathlib.Path("data/linssihinnasto2026.txt")
kampanja_path = pathlib.Path("data/kampanja032026.txt")

# Luetaan dokumenttien sisältö
hinnasto_content = hinnasto_path.read_text(encoding="utf-8")
kampanja_content = kampanja_path.read_text(encoding="utf-8")

# Yhdistetään tiedostot yhteen document_content-muuttujaan
document_content = f"""
LINSSIHINNASTO:
{hinnasto_content}

KAMPANJATIEDOT:
{kampanja_content}
"""

def predict(message, history):

    system_instruction = f"Olet asiantuntija optisella alalla. Sinulla on käytössä seuraavat dokumentit:\n\n{document_content}\n\nVastaa käyttäjän kysymyksiin selkeästi ja ytimekkäästi dokumenttien perusteella. Jos kysymys ei liity alaan tai ei löydy dokumenteista, kieltäydy vastaamasta kohteliaasti."

    formatted_messages = []

    for msg in history:
        role = msg["role"]
        if role == "assistant":
            role = "model"

        content = msg.get("content", "")
        if isinstance(content, list):
            content = str(content[0]) if content else ""

        formatted_messages.append({
            "role": role,
            "parts": [{"text": str(content)}]
        })

    formatted_messages.append({
        "role": "user",
        "parts": [{"text": str(message)}]
    })

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=formatted_messages,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )

        return response.text
    except Exception as e:
        return f"Tapahtui virhe: {e}"
    
interface = gr.ChatInterface(
    fn=predict,
    title="Optisen alan asiantuntija Chatbot",
    description="Kysy optiseen alaan liittyviä kysymyksiä dokumentin perusteella.",
    examples=["Paljon maksaa Ultra moniteholinssit heijastamattomalla pinnalla?"],
)

if __name__ == "__main__":
    interface.launch()
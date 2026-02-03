# Lähteenä käytetty: https://medium.com/@sundar.g.ramamurthy/building-a-chatbot-using-gemini-llm-and-gradio-3449915abeb2
# https://ai.google.dev/gemini-api/docs/
# https://www.gradio.app/guides/creating-a-chatbot-fast


from pyexpat.errors import messages
from dotenv import load_dotenv
import os
from google import genai
import gradio as gr

# Ympäristömuuttujien lataaminen .env-tiedostosta
load_dotenv()

# Määritetään Client
client = genai.Client()

# 
def predict(message, history):
    # Luodaan lista viesteistä tekoälyä varten
    # Ensimmäinen viesti on järjestelmäviesti, joka määrittää tekoälyn roolin
    messages = [
        {"role": "user", "content": "Olet asiantuntija optisella alalla. Vastaa käyttäjän kysymyksiin selkeästi ja ytimekkäästi. Jos kysymys ei liity alaan, kieltäydy vastaamasta kohteliaasti."},
    ]

# Lisätään käyttäjän viesti ja keskusteluhistoria viestilistaan
    for msg in history:
        messages.append(msg)

    # Lisätään nykyinen käyttäjän viesti
    messages.append({"role": "user", "content": message})

    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=messages
        )

        return response.text # Palautetaan vain tekstivastaus, Gradio hoitaa historian
    

    except Exception as e:
        return f"Tapahtui virhe: {e}"

# Gradio-käyttöliittymän määrittely
interface = gr.ChatInterface(
    fn=predict,
    type="messages",
    title="Optisen alan asiantuntija Chatbot",
    description="Kysy optiseen alaan liittyviä kysymyksiä.",
    examples=["Mikä on hajataitto?", 
              "Miksi silmät väsyvät näyttöpäätetyössä?", 
              "Mikä on polarisoiva linssi?", 
              "Kuinka usein näkö tulisi tarkistaa?", 
              "Mitkä ovat yleisiä silmäsairauksia?"],
)

if __name__ == "__main__":
    interface.launch()
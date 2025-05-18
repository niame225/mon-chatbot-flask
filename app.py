# app.py
from flask import Flask, render_template_string, request
from huggingface_hub import InferenceClient
import os

app = Flask(__name__)

# Charger la clé API depuis .env
api_key = os.getenv("HUGGINGFACE_API_KEY")
client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token=api_key)

# Réponses personnalisées
custom_responses = {
    "bonjour": "Bonjour ! Je suis là pour t'aider.",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22.",
    # Ajoute toutes tes réponses ici
}

@app.route("/", methods=["GET", "POST"])
def home():
    user_input = ""
    response = ""

    if request.method == "POST":
        user_input = request.form["query"]
        normalized = user_input.strip().lower()

        # Réponses personnalisées
        for key in custom_responses:
            if key in normalized:
                response = custom_responses[key]
                break
        else:
            # Sinon, IA
            try:
                response = client.text_generation(
                    prompt=f"<s>[INST] {user_input} [/INST]",
                    max_new_tokens=150,
                    temperature=0.5
                )
            except Exception as e:
                response = f"Erreur : {str(e)}"

    return render_template_string('''
        <h1>Chatbot IA</h1>
        <form method="post">
            <input type="text" name="query" value="{{ user_input }}" placeholder="Pose ta question..." size="50">
            <button type="submit">Envoyer</button>
        </form>
        {% if response %}
        <p><strong>Assistant :</strong> {{ response }}</p>
        {% endif %}
    ''', user_input=user_input, response=response)

if __name__ == "__main__":
    app.run()
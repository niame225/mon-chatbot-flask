from flask import Flask, request, jsonify, render_template
import requests
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Créer l'application Flask
app = Flask(__name__)

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Configuration des logs (plus robuste pour Render)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Affichage dans la console
    ]
)

# Charger la clé API
api_key = os.environ.get("HUGGINGFACE_API_KEY")
if not api_key:
    logging.error("La clé API HUGGINGFACE_API_KEY n'est pas définie.")
    raise ValueError("La clé API HUGGINGFACE_API_KEY n'est pas définie.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        # Récupérer le message de l'utilisateur
        user_input = request.form.get("message", "").strip()
        logging.info(f"Message reçu: {user_input}")
        
        if not user_input:
            return jsonify({"error": "Aucune question envoyée."}), 400

        # Réponses personnalisées
        if user_input.lower() in ["bonjour", "salut", "hello"]:
            response = "Bonjour ! Comment puis-je vous aider ?"
        else:
            try:
                # URL corrigée (suppression de l'espace en trop)
                API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                payload = {
                    "inputs": f"[INST] Réponds en français : {user_input} [/INST]",
                    "parameters": {
                        "max_new_tokens": 100,
                        "temperature": 0.7,
                        "return_full_text": False
                    }
                }

                logging.info(f"Envoi de la requête à Hugging Face...")
                resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
                
                logging.info(f"Status code: {resp.status_code}")
                logging.info(f"Réponse brute: {resp.text}")
                
                resp.raise_for_status()

                response_data = resp.json()
                
                # Vérifier si la réponse est dans le bon format
                if isinstance(response_data, list) and len(response_data) > 0:
                    generated_text = response_data[0].get("generated_text", "")
                else:
                    generated_text = str(response_data)

                # Nettoyer la réponse
                if "[/INST]" in generated_text:
                    response = generated_text.split("[/INST]")[-1].strip()
                else:
                    response = generated_text.strip()

                # Si la réponse est vide, donner une réponse par défaut
                if not response:
                    response = "Je n'ai pas pu générer une réponse appropriée."

            except requests.exceptions.Timeout:
                logging.error("Timeout lors de la requête à Hugging Face")
                response = "Le service met trop de temps à répondre. Réessayez plus tard."
            except requests.exceptions.RequestException as e:
                logging.error(f"Erreur réseau : {str(e)}")
                response = "Erreur de connexion au service d'IA."
            except Exception as e:
                logging.error(f"Erreur lors du traitement de la réponse IA : {str(e)}")
                response = "Désolé, je ne peux pas répondre pour le moment."

        logging.info(f"Réponse envoyée: {response}")
        return jsonify({"response": response})

    except Exception as e:
        logging.error(f"Erreur générale dans /ask : {str(e)}")
        return jsonify({"error": "Erreur interne du serveur"}), 500

# Route de test pour vérifier que l'API fonctionne
@app.route("/test")
def test():
    return jsonify({
        "status": "OK", 
        "api_key_configured": bool(api_key),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Démarrage de l'application sur le port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
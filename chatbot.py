import tkinter as tk
from huggingface_hub import InferenceClient
from requests.exceptions import RequestException
import datetime
import os
import re
import unicodedata  # Pour gérer les caractères accentués
from dotenv import load_dotenv  # Charger les variables d'environnement

# Charger les variables depuis .env
load_dotenv()

# Récupère la clé API depuis l'environnement
api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialisation correcte de InferenceClient avec le point de terminaison et l'en-tête d'autorisation
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=api_key  # Utiliser 'token' au lieu de 'api_key'
)

# Créer un nom de fichier basé sur la date et l'heure
now = datetime.datetime.now()
filename = now.strftime("logs/Chat_%Y-%m-%d_%H-%M.txt")

# Créer le dossier logs si nécessaire
os.makedirs("logs", exist_ok=True)

def log_conversation(line):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Réponses personnalisées
custom_responses = {
    "bonjour": "Bonjour ! Je suis là pour t'aider. Comment puis-je t'assister aujourd'hui ?",
    "qui t'a conçu": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "comment vas tu": "Je vais bien, merci pour ton intérêt. Comment ça va pour toi ?",
    "qui t'a fait": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "qui t'a créé": "Raphaël Niamé (+225) 05 06 53 15 22.",
    "qui est raphael niamé": "C'est un développeur freelance d'applications.",
    "raphael niamé": "C'est un développeur freelance d'applications.",
    "qui est oulai": "C'est le père de Tchounatchou ou Djouniédjou.",
    "oulai": "C'est le père de Tchounatchou ou Djouniédjou.",
    "qui est diarrassouba soma": "C'est un célèbre agent immobilier.",
    "qui soma": "C'est un célèbre agent immobilier.",
    "diarrassouba": "C'est un célèbre agent immobilier."
}

# Fonction améliorée pour normaliser le texte (y compris les accents)
def normalize(text):
    # Enlève les accents en convertissant vers ASCII
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    # Supprime la ponctuation, espaces multiples et met en minuscules
    return re.sub(r'[^\w\s]', '', text).strip().lower().replace("  ", " ")


def send_message(event=None):
    user_input = entry.get().strip()
    if not user_input:
        return

    # Afficher ce que l'utilisateur a écrit
    conversation.insert(tk.END, f"Vous: {user_input}\n\n")
    log_conversation(f"Vous: {user_input}\n")

    # Vérifier si la question correspond à une réponse personnalisée
    normalized_input = normalize(user_input)

    matched_response = None
    for key in custom_responses:
        if normalize(key) in normalized_input:
            matched_response = custom_responses[key]
            break

    if matched_response:
        conversation.insert(tk.END, f"Assistant: {matched_response}\n\n")
        log_conversation(f"Assistant: {matched_response}\n")
        entry.delete(0, tk.END)
        conversation.see(tk.END)
        return

    # Si pas de réponse personnalisée, on passe à l'IA
    conversation.insert(tk.END, "Assistant: ⏳ En cours...\n")
    conversation.see(tk.END)

    try:
        response = ""
        for message in client.text_generation(
            prompt=f"<s>[INST] Réponds brièvement en français à la question suivante : {user_input} [/INST]",
            max_new_tokens=150,
            stream=True,
            temperature=0.5
        ):
            content = message or ""
            response += content

        # Supprime le message "En cours..." et insère la réponse finale
        conversation.delete("end-2l", "end")
        conversation.insert(tk.END, f"Assistant: {response}\n\n")
        log_conversation(f"Assistant: {response}\n")
    except RequestException as e:
        conversation.delete("end-2l", "end")
        error_msg = f"🚨 Erreur de connexion : {str(e)}"
        conversation.insert(tk.END, f"Assistant: {error_msg}\n\n")
        log_conversation(f"Assistant: {error_msg}\n")
    except Exception as e:
        conversation.delete("end-2l", "end")
        error_msg = f"🚨 Erreur : {str(e)}"
        conversation.insert(tk.END, f"Assistant: {error_msg}\n\n")
        log_conversation(f"Assistant: {error_msg}\n")

    entry.delete(0, tk.END)
    conversation.see(tk.END)


def clear_chat():
    conversation.delete(1.0, tk.END)


# Création de la fenêtre principale
root = tk.Tk()
root.title("Chatbot IA - Assistant Francophone")

# Zone de discussion
conversation = tk.Text(root, height=20, width=50, wrap=tk.WORD, font=("Arial", 12))
conversation.pack(padx=10, pady=10)

# Zone de saisie
entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(padx=10, pady=5)
entry.focus_set()

# Boutons
send_button = tk.Button(root, text="Envoyer", command=send_message)
send_button.pack(pady=5)

clear_button = tk.Button(root, text="Effacer le chat", command=clear_chat)
clear_button.pack(pady=5)

# Raccourcis clavier
root.bind('<Return>', send_message)
root.bind('<KP_Enter>', send_message)

# Lancer l'application
root.mainloop()
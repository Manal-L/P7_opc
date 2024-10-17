# streamlit_app.py

import streamlit as st
import requests

# URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"

# Titre de l'application
st.title("Prédiction du score de crédit")

# Récupérer la liste des clients via l'API FastAPI
st.subheader("Sélectionnez un client")
response = requests.get(f"{API_URL}/clients")
client_ids = response.json()

# Créer une liste déroulante avec les SK_ID_CURR
selected_client_id = st.selectbox("SK_ID_CURR", client_ids)

# Bouton pour lancer la prédiction
if st.button("Faire une prédiction"):
    # Faire la requête POST pour obtenir la prédiction
    prediction_response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_client_id})
    prediction_data = prediction_response.json()
    
    # Afficher les résultats
    if "error" in prediction_data:
        st.error(prediction_data["error"])
    else:
        st.success(f"Probabilité de défaut: {prediction_data['probability']:.2f}")

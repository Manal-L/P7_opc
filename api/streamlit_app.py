# streamlit_app.py

#sur le terminale 2: streamlit run streamlit_app.py
#si ConnectionError alors Streamlit ne parvient pas à se connecter à API FastAPI => FastAPI n'est pas en cours d'exécution sur le terminale1, il faut l'exécuter

import streamlit as st
import requests

#l'URL de l'API FastAPI
API_URL = "http://127.0.0.1:8000"

#le titre de l'application
st.title("La prédiction du score de crédit")

#je récupère la liste des clients via l'API FastAPI
st.subheader("Veuillez sélectionner un ID client:")
response = requests.get(f"{API_URL}/clients")
client_ids = response.json()

#la liste déroulante avec les SK_ID_CURR
selected_client_id = st.selectbox("SK_ID_CURR (ID Clients)", client_ids)

#le bouton pour lancer la prédiction
if st.button("Réaliser une prédiction"):

    #la requête POST pour obtenir la prédiction
    prediction_response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": selected_client_id})
    prediction_data = prediction_response.json()
    
    #afficher les résultats
    if "error" in prediction_data:
        st.error(prediction_data["error"])
    else:
        st.success(f"La probabilité de défaut pour ce client est de: {prediction_data['probability']:.2f} %")

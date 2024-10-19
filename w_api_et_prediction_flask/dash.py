import os
import pandas as pd
import requests
import streamlit as st

# Obtenez le répertoire courant du script
current_directory = os.path.dirname(os.path.abspath(__file__))

# URL de l'API
API_URL = "http://localhost:5000"

# Fonction pour récupérer tous les SK_ID_CURR
def get_clients():
    response = requests.get(f"{API_URL}/clients")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de la récupération des clients: {response.status_code}")
        return []

# Fonction pour faire une prédiction
def predict(sk_id_curr):
    response = requests.post(f"{API_URL}/predict", json={"SK_ID_CURR": sk_id_curr})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Erreur lors de l'appel à l'API: {response.status_code}")
        return None

st.set_page_config(layout="wide")

# Titre de l'application
st.markdown(
    "<h1 style='text-align: center; color: black;'>Estimation du risque de non-remboursement</h1>",
    unsafe_allow_html=True,
)

# Récupérer la liste des SK_ID_CURR
clients = get_clients()

# Afficher la liste déroulante
selected_sk_id = st.selectbox("Sélectionnez le SK_ID_CURR:", clients)

# Bouton pour effectuer la prédiction
if st.button("Prédire"):
    if selected_sk_id:
        data = predict(selected_sk_id)
        if data:
            proba = data["probability"]
            shap_values = data["shap_values"]
            feature_names = data["feature_names"]
            feature_values = data["feature_values"]

            # Affichage des résultats
            st.write(f"Probabilité de non-remboursement : **{proba:.2f}%**")

            # Affichage des valeurs SHAP
            st.write("Valeurs SHAP :")
            for name, shap_value, feature_value in zip(feature_names, shap_values, feature_values):
                st.write(f"**{name}** : SHAP Value = {shap_value}, Feature Value = {feature_value}")

            # Décision sur le prêt
            decision_message = "Le prêt sera accordé." if proba < 48 else "Le prêt ne sera pas accordé."
            st.markdown(
                f"<div style='text-align: center; color:red; font-size:30px; border:2px solid red; padding:10px;'>{decision_message}</div>",
                unsafe_allow_html=True,
            )

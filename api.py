import os
import json
import joblib  # Pour charger les modèles et scaler
import pandas as pd
from flask import Flask, request, jsonify
import shap  # Pour l'explicabilité
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle et du scaler
model_path = "path/to/your/model.pkl"  # Remplace par le chemin réel
scaler_path = "path/to/your/scaler.pkl"  # Remplace par le chemin réel
data_path = "path/to/your/data.csv"  # Fichier contenant les données des clients

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
data = pd.read_csv(data_path)

# Fonction pour charger les données client en fonction de l'ID
def get_client_data(sk_id_curr):
    client_data = data[data['SK_ID_CURR'] == sk_id_curr]
    if client_data.empty:
        return None
    return client_data.drop(columns=['TARGET', 'SK_ID_CURR'])  # Enlever la cible et l'identifiant

# Route de prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lecture des données envoyées dans la requête POST
        request_data = request.get_json()
        sk_id_curr = request_data['SK_ID_CURR']
        
        # Récupération des données du client
        client_data = get_client_data(sk_id_curr)
        if client_data is None:
            return jsonify({"error": "Client data not found"}), 404
        
        # Appliquer le scaler sur les données du client
        client_data_scaled = scaler.transform(client_data)
        
        # Prédiction du modèle
        prediction = model.predict(client_data_scaled)
        prediction_proba = model.predict_proba(client_data_scaled)[0, 1]
        
        # Explicabilité avec SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(client_data_scaled)
        
        # Retourner les résultats
        response = {
            'SK_ID_CURR': sk_id_curr,
            'Prediction': int(prediction[0]),
            'Prediction_Probability': prediction_proba,
            'SHAP_Values': shap_values.tolist()  # SHAP values en liste
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Démarrage du serveur
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

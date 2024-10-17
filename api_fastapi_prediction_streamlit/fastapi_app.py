# fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Charger le modèle et le scaler
current_directory = os.path.dirname(os.path.realpath(__file__))
model = joblib.load(os.path.join(current_directory, "model.joblib"))
scaler = joblib.load(os.path.join(current_directory, "scaler.joblib"))

# Créer l'application FastAPI
app = FastAPI()

# Structure de la requête de prédiction
class ClientData(BaseModel):
    SK_ID_CURR: int

# Charger les données des nouveaux clients (CSV)
new_clients_df = pd.read_csv(os.path.join(current_directory, 'nouveaux_clients.csv'))

@app.get("/clients")
def get_clients():
    """Retourner la liste des SK_ID_CURR"""
    return new_clients_df['SK_ID_CURR'].tolist()

@app.post("/predict")
def predict(client_data: ClientData):
    """Faire une prédiction pour un client spécifique"""
    # Récupérer le SK_ID_CURR
    client_id = client_data.SK_ID_CURR
    
    # Récupérer les données du client
    client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]
    
    if client_row.empty:
        return {"error": "Client not found"}
    
    # Préparer les données pour le modèle
    client_features = client_row.drop(columns=["SK_ID_CURR"]).values
    client_scaled = scaler.transform(client_features)
    
    # Faire la prédiction
    prediction = model.predict_proba(client_scaled)[:, 1][0]
    
    return {"SK_ID_CURR": client_id, "probability": prediction}







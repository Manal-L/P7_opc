# fastapi_app.py avec le framework FastAPI qui s'éxécute avec le serveur Uvicorn 

#1- sur le termile1:  uvicorn fastapi_app:app --reload 

#fastapi_app c'est mon fichier sans l'extension .py
#app fait référence à l'instance de FastAPI que j'ai créée
#--reload permet au serveur de redémarrer automatiquement lorsque je modifie le code

#2 -puis  http://127.0.0.1:8000/clients

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

#charger le modèle et le scaler
current_directory = os.path.dirname(os.path.realpath(__file__))
model = joblib.load(os.path.join(current_directory, "model_lgbm.joblib"))
scaler = joblib.load(os.path.join(current_directory, "scaler.joblib"))

#création de l'application FastAPI
app = FastAPI()

#la structure de la requête de prédiction
class ClientData(BaseModel):
    SK_ID_CURR: int

#chargement des données des nouveaux clients (CSV)
new_clients_df = pd.read_csv(os.path.join(current_directory, 'nouveaux_clients.csv'))

#la route (url: http://127.0.0.1:8000/clients) pour la liste des id clients: SK_ID_CURR
@app.get("/clients")
def get_clients():
    """Retourner la liste des SK_ID_CURR"""
    return new_clients_df['SK_ID_CURR'].tolist()

#la route (http://127.0.0.1:8000/predict) pour faire une prédiction pour un client spécifique
#je récupère ces prédictions via Streamplit
@app.post("/predict")
def predict(client_data: ClientData):
    """Faire une prédiction pour un client spécifique"""
    #je récupère le SK_ID_CURR
    client_id = client_data.SK_ID_CURR
    
    #je récupère les données du client
    client_row = new_clients_df[new_clients_df['SK_ID_CURR'] == client_id]
    
    if client_row.empty:
        return {"error": "Client not found"}
    
    #je prépare les données pour le modèle (suppression des ID + scaler)
    client_features = client_row.drop(columns=["SK_ID_CURR"]).values
    client_scaled = scaler.transform(client_features)
    
    #calculer la prédiction
    prediction = model.predict_proba(client_scaled)[:, 1][0]
    
    return {"SK_ID_CURR": client_id, "probability": prediction}







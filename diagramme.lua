+------------------+           +------------------+
|    Streamlit     |           |      Flask       |
|      (dash.py)   |           |       (api.py)   |
+------------------+           +------------------+
|                  |  POST     |                  |
|  Entrées Utilisateur | ----->|  Reçoit SK_ID     |
|                  |           |                  |
|                  |           |  Charge le modèle |
|                  |           |  Effectue la prédiction |
|                  |           |  Calcule les valeurs SHAP |
|                  |           |  Retourne le Résultat |
|                  |  JSON     |                  |
|                  | <---------|                  |
|  Affiche la sortie |           |                  |
|                  |           |                  |
+------------------+           +------------------+


api.py gère la logique de prédiction et sert d'API backend.

dash.py fournit une interface frontend pour que les utilisateurs interagissent avec le modèle de prédiction.

Les deux composants communiquent via des requêtes HTTP, permettant une expérience utilisateur fluide tout en 
utilisant des prédictions et des interprétations d'apprentissage automatique.

Flux de données :
L'utilisateur entre son identifiant client dans l'application Streamlit (dash.py).
L'application envoie une requête à l'API Flask (api.py) pour obtenir des prédictions.
L'application Flask traite les données d'entrée, effectue des prédictions et renvoie les résultats.
L'application Streamlit reçoit les résultats et les affiche, y compris la probabilité et des explications visuelles utilisant les valeurs SHAP.
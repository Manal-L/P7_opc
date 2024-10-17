from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import joblib

app = Flask(__name__)

# Chemin vers le scaler et le modèle
current_directory = os.path.dirname(os.path.abspath(__file__))
scaler = joblib.load(os.path.join(current_directory, "scaler.joblib"))
model = joblib.load(os.path.join(current_directory, "model.joblib"))

@app.route("/", methods=['GET', 'POST'])
def home():
    # Lire les clients pour le formulaire
    csv_path = os.path.join(current_directory, "nouveaux_clients.csv")
    df = pd.read_csv(csv_path)
    clients = df['SK_ID_CURR'].tolist()

    prediction = None
    if request.method == 'POST':
        sk_id_curr = request.form['sk_id_curr']
        
        # Obtenir les données du client
        sample = df[df['SK_ID_CURR'] == int(sk_id_curr)]
        
        if not sample.empty:
            sample = sample.drop(columns=['SK_ID_CURR'])
            sample_scaled = scaler.transform(sample)

            prediction_proba = model.predict_proba(sample_scaled)
            proba = prediction_proba[0][1]

            prediction = {
                'sk_id_curr': sk_id_curr,
                'proba': proba * 100,
                'details': dict(zip(sample.columns.tolist(), sample.values[0].tolist()))
            }
        else:
            prediction = {'error': f"SK_ID_CURR {sk_id_curr} non trouvé."}

    return render_template("index.html", clients=clients, prediction=prediction)

@app.route("/clients", methods=['GET'])
def get_clients():
    csv_path = os.path.join(current_directory, "nouveaux_clients.csv")
    df = pd.read_csv(csv_path)
    return jsonify(df['SK_ID_CURR'].tolist())

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data['SK_ID_CURR']
    
    # Charger le CSV
    csv_path = os.path.join(current_directory, "nouveaux_clients.csv")
    df = pd.read_csv(csv_path)
    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    
    if sample.empty:
        return jsonify({"error": f"SK_ID_CURR {sk_id_curr} non trouvé dans le fichier."}), 404

    sample = sample.drop(columns=['SK_ID_CURR'])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict_proba(sample_scaled)
    proba = prediction[0][1]

    return jsonify({
        'probability': proba * 100,
        'feature_names': sample.columns.tolist(),
        'feature_values': sample.values[0].tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)






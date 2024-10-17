import unittest
import joblib
import os
import numpy as np
import pandas as pd

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Configuration avant chaque test."""
        # Charger les fichiers modèle et scaler
        self.model_path = 'model.pkl'
        self.scaler_path = 'scaler.pkl'
    
    def test_model_exists(self):
        """Test si le modèle a bien été sauvegardé."""
        self.assertTrue(os.path.exists(self.model_path), "Le modèle n'a pas été sauvegardé correctement.")
    
    def test_scaler_exists(self):
        """Test si le scaler a bien été sauvegardé."""
        self.assertTrue(os.path.exists(self.scaler_path), "Le scaler n'a pas été sauvegardé correctement.")
    
    def test_scaler_transformation(self):
        """Test si le scaler transforme correctement les données."""
        scaler = joblib.load(self.scaler_path)
        # Exemple de données simulées
        X_sample = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        X_scaled = scaler.transform(X_sample)
        
        # Vérifier que les données ont bien été transformées
        self.assertEqual(X_scaled.shape, X_sample.shape, "La transformation du scaler a échoué.")
    
    def test_model_prediction(self):
        """Test si le modèle peut prédire avec des données simulées."""
        model = joblib.load(self.model_path)
        X_sample = np.array([[0.5, -1.0, 2.0], [1.0, 0.5, -2.0]])
        
        # Test simple de prédiction
        predictions = model.predict(X_sample)
        self.assertEqual(len(predictions), len(X_sample), "La prédiction du modèle a échoué.")

if __name__ == '__main__':
    unittest.main()

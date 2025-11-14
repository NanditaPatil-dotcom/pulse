import joblib
import os
import numpy as np

# --- Define DummyModel here so Joblib can re-load it ---
class DummyModel:
    def predict(self, X):
        return [0.5] * len(X)

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in X]

# Path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_training", "model.pkl")

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    print("Loading.")
    model = DummyModel()

# Dummy explainer for now
class DummyExplainer:
    def shap_values(self, X):
        return [[0.0, 0.0] for _ in X]

explainer = DummyExplainer()

def predict_risk(heart_rate, spo2):
    x = np.array([[heart_rate, spo2]])
    score = float(model.predict_proba(x)[0][1])

    shap_vals = explainer.shap_values(x)
    explanation = {
        "heart_rate": float(shap_vals[0][0]),
        "spo2": float(shap_vals[0][1])
    }

    return score, explanation
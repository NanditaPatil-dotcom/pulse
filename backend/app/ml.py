
import os
import joblib
import numpy as np
import shap
import pandas as pd

# ------------------------
# Paths
# ------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model_training")

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "model_classifier.pkl")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "model_regressor.pkl")

FEATURES = ["heart_rate", "spo2", "temp_c", "steps"]


# ------------------------
# Model Wrapper
# ------------------------

class PulseModel:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.explainer = None
        self.is_loaded = False
        self._load_models()

    def _load_models(self):
        if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(REGRESSOR_PATH):
            print("[ML] Model files not found")
            return

        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.regressor = joblib.load(REGRESSOR_PATH)
        # Load background data for SHAP
        data_path = os.path.join(MODEL_DIR, "training_data.csv")
        df = pd.read_csv(data_path)

        background = df[FEATURES].fillna(0).sample(
            min(100, len(df)), random_state=42
        )

        self.explainer = shap.Explainer(
            self.regressor.predict,
            background
        )



        self.is_loaded = True
        print("[ML] Models + SHAP loaded successfully")

    def _prepare(self, data: dict):
        x = np.array([
            data.get("heart_rate", 0),
            data.get("spo2", 0),
            data.get("temp_c", 0) or 0,
            data.get("steps", 0) or 0,
        ]).reshape(1, -1)
        return x

    def predict(self, data: dict):
        if not self.is_loaded:
            return None

        X = self._prepare(data)

        # ---- Classification ----
        label = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probs))

        # ---- Regression ----
        risk_score = float(self.regressor.predict(X)[0])

        # ---- SHAP ----
        shap_values = self.explainer(X)
        shap_dict = {
            FEATURES[i]: float(shap_values.values[0][i])
            for i in range(len(FEATURES))
        }

        return {
            "risk_label": label,
            "confidence": confidence,
            "risk_score": risk_score,
            "shap": shap_dict,
        }


# ------------------------
# Singleton
# ------------------------

pulse_model = PulseModel()

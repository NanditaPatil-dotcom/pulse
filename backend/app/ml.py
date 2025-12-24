import os
import joblib
import numpy as np
import shap
import pandas as pd



FEATURES = ["heart_rate", "spo2", "temp_c", "steps"]

MODEL_DIR = os.getenv("MODEL_PATH", "/models")

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "model_classifier.pkl")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "model_regressor.pkl")
TRAINING_DATA_PATH = os.path.join(MODEL_DIR, "training_data.csv")




class PulseModel:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.explainer = None
        self.is_loaded = False
        self._load_models()

    def _load_models(self):
        if not (
            os.path.exists(CLASSIFIER_PATH)
            and os.path.exists(REGRESSOR_PATH)
            and os.path.exists(TRAINING_DATA_PATH)
        ):
            print("[ML] Model files not found in", MODEL_DIR)
            return

        self.classifier = joblib.load(CLASSIFIER_PATH)
        self.regressor = joblib.load(REGRESSOR_PATH)

        df = pd.read_csv(TRAINING_DATA_PATH)

        background = (
            df[FEATURES]
            .fillna(0)
            .sample(min(100, len(df)), random_state=42)
        )

        self.explainer = shap.Explainer(
            self.regressor.predict,
            background
        )

        self.is_loaded = True
        print("[ML] Models + SHAP loaded successfully")


    def _prepare(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame([{
            "heart_rate": data.get("heart_rate", 0),
            "spo2": data.get("spo2", 0),
            "temp_c": data.get("temp_c", 0) or 0,
            "steps": data.get("steps", 0) or 0,
        }])


    def predict(self, data: dict):
        if not self.is_loaded:
            return None

        X = self._prepare(data)

        # Classification
        label = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        confidence = float(np.max(probs))

        # Regression
        risk_score = float(self.regressor.predict(X)[0])

        # SHAP
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



pulse_model = PulseModel()



import os
import joblib
import numpy as np
import json

# Optional SHAP import
try:
    import shap
except Exception:
    shap = None


class ModelWrapper:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.explainer = None
        self.version = "dev"
        self.is_loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            print(f"[ML] Model file not found at {self.model_path}")
            return

        self.model = joblib.load(self.model_path)
        self.is_loaded = True
        self.version = getattr(self.model, "version", "v1")

        if shap is not None:
            try:
                self.explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                print("[ML] SHAP explainer init failed:", e)

    def _prepare(self, features: dict):
        """
        Convert feature dict to 2D numpy array in fixed column order
        """
        cols = ["heart_rate", "spo2", "temp_c", "steps"]
        arr = [features.get(c, 0.0) for c in cols]
        return np.array(arr).reshape(1, -1), cols

    def predict_with_shap(self, features: dict):
        if not self.is_loaded:
            return "unknown", 0.0, {}, self.version

        X, cols = self._prepare(features)

        probs = self.model.predict_proba(X)[0]
        prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        label = str(self.model.classes_[np.argmax(probs)])

        shap_json = {}
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(X)

                # Handle different SHAP output formats
                if isinstance(shap_vals, list):
                    sv = (
                        shap_vals[1][0].tolist()
                        if len(shap_vals) > 1
                        else shap_vals[0][0].tolist()
                    )
                else:
                    sv = shap_vals[0].tolist()

                shap_json = dict(zip(cols, [float(x) for x in sv]))
            except Exception as e:
                print("[ML] SHAP compute error:", e)

        return label, prob, shap_json, self.version


# -------------------------------------------------
# Global model wrapper instance
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_training", "model.pkl")

model_wrapper = None

try:
    model_wrapper = ModelWrapper(MODEL_PATH)
except Exception as e:
    print("[ML] Model loading skipped:", e)
    model_wrapper = None


def predict_risk(heart_rate: int, spo2: int):
    """
    Simple fallback risk heuristic (used if model not loaded)
    """
    score = 0.0
    explanation = {}

    if heart_rate > 100:
        score += 0.4
        explanation["heart_rate"] = "high"

    if spo2 < 95:
        score += 0.6
        explanation["spo2"] = "low"

    return min(score, 1.0), explanation

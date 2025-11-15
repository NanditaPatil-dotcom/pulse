import joblib
import numpy as np
import json
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
print(f"Model file not found at {self.model_path}")
return
self.model = joblib.load(self.model_path)
self.is_loaded = True
self.version = getattr(self.model, "version", "v1")
if shap is not None:
try:
self.explainer = shap.TreeExplainer(self.model)
except Exception as e:
print("SHAP explainer init failed:", e)
def _prepare(self, features: dict):
# Convert dict to 2D array in consistent order
cols = ["heart_rate", "spo2", "temp_c", "steps"]
arr = [features.get(c, 0.0) for c in cols]
return np.array(arr).reshape(1, -1), cols
def predict_with_shap(self, features: dict):
if not self.is_loaded:
return ("unknown", 0.0, {}, self.version)
X, cols = self._prepare(features)
probs = self.model.predict_proba(X)[0]
# assume positive class at index 1
prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
label = self.model.classes_[np.argmax(probs)].__str__()
shap_json = {}
if self.explainer is not None:
try:
shap_vals = self.explainer.shap_values(X)
# shap_vals shape differs between classifiers; handle common case
if isinstance(shap_vals, list):
sv = shap_vals[1][0].tolist() if len(shap_vals) > 1 else shap_vals[0][0].tolist()
else:
sv = shap_vals[0].tolist()
shap_json = dict(zip(cols, [float(x) for x in sv]))
except Exception as e:
print("SHAP compute error:", e)
return label, prob, shap_json, self.version

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model_training", "model.pkl")
model_wrapper = ModelWrapper(MODEL_PATH)
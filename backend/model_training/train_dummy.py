from app.ml import DummyModel
import joblib
import os

model = DummyModel()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

joblib.dump(model, MODEL_PATH)
print("Dummy model saved at:", MODEL_PATH)
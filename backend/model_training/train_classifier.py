import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("training_data.csv")

FEATURES = ["heart_rate", "spo2", "temp_c", "steps"]
TARGET = "risk_label"

X = df[FEATURES].fillna(0)
y = df[TARGET]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Pipeline = scaling + model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        max_iter=1000,
        class_weight="balanced"
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model_classifier.pkl")
print("\nSaved → model_classifier.pkl")

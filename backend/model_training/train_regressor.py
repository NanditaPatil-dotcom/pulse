import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("training_data.csv")

FEATURES = ["heart_rate", "spo2", "temp_c", "steps"]
TARGET = "risk_score"

X = df[FEATURES].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

joblib.dump(pipeline, "model_regressor.pkl")
print("\nSaved to model_regressor.pkl")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("training_data.csv")

print("\n--- BASIC INFO ---")
print(df.info())

print("\n--- DESCRIPTIVE STATS ---")
print(df.describe())

print("\n--- RISK LABEL COUNTS ---")
print(df["risk_label"].value_counts(normalize=True))


plt.figure()
sns.histplot(df["heart_rate"], bins=30, kde=True)
plt.title("Heart Rate Distribution")
plt.show()

plt.figure()
sns.histplot(df["spo2"], bins=30, kde=True)
plt.title("SpO2 Distribution")
plt.show()


plt.figure()
sns.boxplot(x="risk_label", y="heart_rate", data=df)
plt.title("Heart Rate vs Risk Label")
plt.show()

plt.figure()
sns.boxplot(x="risk_label", y="spo2", data=df)
plt.title("SpO2 vs Risk Label")
plt.show()


numeric_cols = ["heart_rate", "spo2", "temp_c", "steps", "risk_score"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()


print("\n--- MISSING VALUES ---")
print(df.isna().sum())

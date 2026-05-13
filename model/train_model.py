import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "D:\Desktop\heart-disease-fastapi-docker\data\heart.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.joblib")


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# -----------------------------
# Features and target
# -----------------------------
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

TARGET = "target"

X = df[FEATURES]
y = df[TARGET]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Train model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)


# -----------------------------
# Save model
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "model_type": "RandomForestClassifier",
        "accuracy": accuracy
    },
    MODEL_PATH
)

print(f"Model saved successfully at: {MODEL_PATH}")
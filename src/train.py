# src/train.py

import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

import dagshub

dagshub.init(
    repo_owner="singhanshika29",
    repo_name="cancer_prediction",
    mlflow=True
)
# -----------------------
# Load Data
# -----------------------
df = pd.read_csv("data/Cancer_Data.csv")

# -----------------------
# Preprocessing
# -----------------------
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Scaling
# -----------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# MLflow Setup
# -----------------------
mlflow.set_experiment("cancer_prediction")

with mlflow.start_run():

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

    # -----------------------
    # MLflow Logging
    # -----------------------
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_artifact("model.pkl")
    mlflow.log_artifact("scaler.pkl")

    # -----------------------
# Save Model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Log model to MLflow
mlflow.sklearn.log_model(model, "model")
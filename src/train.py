# Fix temp folder
import os
os.makedirs("C:/temp", exist_ok=True)

os.environ["TMPDIR"] = "C:/temp"
os.environ["TEMP"] = "C:/temp"
os.environ["TMP"] = "C:/temp"

# -----------------------

df = pd.read_csv("data/Cancer_Data.csv")

# SAFE DROP 
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True, errors='ignore')

# -----------------------

with mlflow.start_run():

    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", acc)

        os.makedirs("models", exist_ok=True)

        joblib.dump(model, "models/model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("accuracy", acc)

        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("models/scaler.pkl")

        mlflow.sklearn.log_model(model, "model")

        print("Run Success ")

    except Exception as e:
        print("Error:", e)
        raise
import logging
import os
import pickle
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score

log = logging.getLogger("__name__")
log.setLevel(logging.DEBUG)

os.environ["MLFLOW_TRACKING_URI"] = (
    "https://dagshub.com/mrdbcse/full-stack-ai-ml.mlflow"
)
os.environ["MLFLOW_TRACKING_USERNAME"] = "mrdbcse"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "d167aced41d29cbf53107aca9a8a7f6656eb11e9"

params = yaml.safe_load(open("params.yaml"))["train"]
log.info(f"Params: {params}")


def evaluate(data_path: str, model_path: str) -> None:
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    model = pickle.load(open(model_path, "rb"))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    mlflow.log_metric("Accuracy Score", accuracy)
    log.info(f"Accuracy Score: {accuracy}")


if __name__ == "__main__":
    evaluate(data_path=params["data"], model_path=params["model"])

import logging
import os
import pickle
from typing import Dict, List
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

log = logging.getLogger(name="__name__")
log.setLevel(level=logging.DEBUG)
load_dotenv()


params = yaml.safe_load(open("params.yaml"))["train"]
log.info(f"Params: {params}")


def hyperparameter_tuning(X_train: List, y_train: List, params: Dict):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(
        estimator=rf, param_grid=params, cv=3, n_jobs=-1, verbose=2
    )

    grid_search.fit(X=X_train, y=y_train)

    log.info(f"Best Params: {grid_search.best_params_}")
    log.info(f"Best Estimator: {grid_search.best_estimator_}")

    return grid_search


def train(data_path: str, model_path: str) -> None:
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        signature = infer_signature(X_train, y_train)

        params = {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }

        grid_search = hyperparameter_tuning(
            X_train=X_train, y_train=y_train, params=params
        )

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)
        _accuracy_score = accuracy_score(y_test, y_pred)

        log.info(f"Accuracy Score: {_accuracy_score}")

        mlflow.log_metric("Accuracy", _accuracy_score)
        mlflow.log_param("Best N Estimators", grid_search.best_params_["n_estimators"])
        mlflow.log_param("Best Max Depth", grid_search.best_params_["max_depth"])
        mlflow.log_param(
            "Best Min Samples Leaf", grid_search.best_params_["min_samples_leaf"]
        )
        mlflow.log_param(
            "Best Min Samples Split", grid_search.best_params_["min_samples_split"]
        )

        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store == "file":
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        else:
            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="Best Model"
            )

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pickle.dump(best_model, open(model_path, "wb"))

        log.info(f"Model Saved to {model_path}")


if __name__ == "__main__":
    train(data_path=params["data"], model_path=params["model"])

### Project: ML Pipeline with DVC & MLflow

This project builds an end-to-end ML pipeline using **DVC** for data and model versioning and **MLflow** for experiment tracking. It trains a **Random Forest Classifier** on the **Pima Indians Diabetes Dataset**, with defined stages for **preprocessing, training, and evaluation**.

### Key Features

#### **1. Data Version Control (DVC)**

- Tracks datasets, models, and pipeline stages for reproducibility.
- Supports automatic re-execution if dependencies change (e.g., data, scripts, or parameters).
- Enables remote storage (DagsHub, S3) for large datasets/models.

#### **2. Experiment Tracking with MLflow**

- Logs model hyperparameters (`n_estimators`, `max_depth`) and performance metrics (accuracy).
- Tracks multiple experiment runs for easy comparison and optimization.

### Pipeline Stages

#### **1. Preprocessing**

- `preprocess.py` reads raw data (`data/raw/data.csv`), processes it, and saves it as `data/processed/data.csv`.

#### **2. Training**

- `train.py` trains a **Random Forest Classifier**, saves the model as `models/random_forest.pkl`, and logs details to MLflow.

#### **3. Evaluation**

- `evaluate.py` loads the model, evaluates performance, and logs results to MLflow.

### Goals

✅ **Reproducibility** – Ensure consistent results using DVC.  
✅ **Experimentation** – Track and compare models with MLflow.  
✅ **Collaboration** – Enable seamless teamwork with version control.

### Use Cases

🔹 **Data Science Teams** – Organize datasets, models, and experiments efficiently.  
🔹 **ML Research** – Iterate over experiments while tracking performance and data versions.

### Tech Stack

- **Python** – Data processing, training, and evaluation
- **DVC** – Data & model version control
- **MLflow** – Experiment tracking
- **Scikit-learn** – ML model training

---

### **Adding DVC Stages**

```sh
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py

dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py

dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```

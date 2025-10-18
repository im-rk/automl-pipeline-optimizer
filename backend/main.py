
import os
import shutil
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from joblib import dump, load
from mlflow.tracking import MlflowClient
from fastapi.responses import FileResponse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.base import is_classifier, BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.optimization.optimizer import ModelOptimizer

app = FastAPI()

# --- Global settings ---
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# CLASSIFIER EVIDENTIAL WRAPPER
class EvidentialWrapper:
    """A pragmatic evidential-style wrapper for sklearn classifiers."""

    def __init__(self, base_clf, regressor=None, min_concentration=None):
        self.base_clf = base_clf
        self.calibrator = CalibratedClassifierCV(self.base_clf, cv=3, method="sigmoid")
        self.regressor = regressor if regressor is not None else Ridge(alpha=1.0)
        self.num_classes_ = None
        self.min_concentration = min_concentration

    def fit(self, X, y):
        self.calibrator.fit(X, y)
        probs = self.calibrator.predict_proba(X)
        K = probs.shape[1]
        self.num_classes_ = K
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
        eps = 1e-8
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + eps)
        inv = 1.0 - entropy_norm
        S_min = K
        S_max = max(5 * K, 20)
        S_target = S_min + inv * (S_max - S_min)
        self.regressor.fit(X, S_target)
        self._S_min = S_min
        self._S_max = S_max
        return self

    def predict_proba(self, X):
        return self.calibrator.predict_proba(X)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_dirichlet(self, X):
        probs = self.predict_proba(X)
        S_hat = self.regressor.predict(X)
        if self.min_concentration is None:
            min_c = getattr(self, "_S_min", probs.shape[1])
        else:
            min_c = self.min_concentration
        S_hat = np.clip(S_hat, min_c, getattr(self, "_S_max", 1000))
        S_hat = np.asarray(S_hat).reshape(-1, 1)
        alpha = probs * S_hat + 1.0
        return alpha

    def evidential_metrics(self, X):
        alpha = self.predict_dirichlet(X)
        S = np.sum(alpha, axis=1)
        K = alpha.shape[1]
        p_hat = alpha / S.reshape(-1, 1)
        u = K / S
        pred_entropy = -np.sum(p_hat * np.log(p_hat + 1e-12), axis=1)
        return {"alpha": alpha, "expected_prob": p_hat, "uncertainty": u, "pred_entropy": pred_entropy}


# ==========================
# REGRESSION EVIDENTIAL WRAPPER
# ==========================
class RegressionEvidentialWrapper(BaseEstimator, RegressorMixin):
    """Evidential-style wrapper for regression."""

    def __init__(self, base_regressor=None, variance_regressor=None, min_variance=1e-6):
        self.base_regressor = base_regressor if base_regressor is not None else Ridge()
        self.variance_regressor = variance_regressor if variance_regressor is not None else Ridge()
        self.min_variance = min_variance

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.base_regressor.fit(X, y)
        residuals_sq = (y - self.base_regressor.predict(X)) ** 2
        self.variance_regressor.fit(X, residuals_sq)
        return self

    def predict(self, X):
        X = check_array(X)
        return self.base_regressor.predict(X)

    def predict_evidential(self, X):
        X = check_array(X)
        mean = self.base_regressor.predict(X)
        variance = self.variance_regressor.predict(X)
        variance = np.clip(variance, self.min_variance, np.inf)
        std = np.sqrt(variance)
        return {"mean": mean, "variance": variance, "std": std}


# ==========================
# PIPELINE: TRAINING
# ==========================
@app.post("/run-experiment/")
async def run_experiment(
    user_id: str = Form(...),
    target_column: str = Form(...),
    dataset: UploadFile = File(...),
    config_file: UploadFile = File(...),
    train_script: UploadFile = File(...),
):
    """Run AutoML pipeline with evidential outputs and save LabelEncoder for classification."""

    # --- Create user folder and save uploads ---
    user_dir = os.path.join(UPLOAD_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    dataset_path = os.path.join(user_dir, dataset.filename)
    config_path = os.path.join(user_dir, config_file.filename)
    train_script_path = os.path.join(user_dir, train_script.filename)

    with open(dataset_path, "wb") as f:
        f.write(await dataset.read())
    with open(config_path, "wb") as f:
        f.write(await config_file.read())
    with open(train_script_path, "wb") as f:
        f.write(await train_script.read())

    # --- Data ingestion and transformation ---
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion(dataset_path)

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_path, label_encoder = data_transformation.initiate_data_transformation(
        train_path, test_path, target_column
    )

    # --- MLflow experiment setup ---
    experiment_name = f"AutoML_{user_id}"
    mlflow.set_experiment(experiment_name)

    # --- Hyperparameter optimization ---
    optimizer = ModelOptimizer(config_path, train_script_path, train_array, test_array, user_id=user_id)
    best_params, best_score, best_model = optimizer.optimize(n_trials=200)

    # --- Split features and target ---
    X_train = train_array[:, :-1]
    y_train = train_array[:, -1]
    X_test = test_array[:, :-1]
    y_test = test_array[:, -1]

    # --- Evidential wrapper ---
    if is_classifier(best_model):
        evidential = EvidentialWrapper(best_model)
        evidential.fit(X_train, y_train)
        ev_metrics = evidential.evidential_metrics(X_test)
        test_log_loss = log_loss(y_test, ev_metrics["expected_prob"])
        pred_entropy = float(np.mean(ev_metrics["pred_entropy"]))
        mean_uncertainty = float(np.mean(ev_metrics["uncertainty"]))
    else:
        evidential = RegressionEvidentialWrapper(base_regressor=best_model)
        evidential.fit(X_train, y_train)
        ev = evidential.predict_evidential(X_test)
        test_log_loss = float(np.mean((y_test - ev["mean"]) ** 2))
        pred_entropy = None
        mean_uncertainty = None

    # --- Save model, preprocessor, and label encoder ---
    user_model_dir = os.path.join(MODEL_DIR, user_id)
    os.makedirs(user_model_dir, exist_ok=True)
    model_path = os.path.join(user_model_dir, "best_model_evidential.pkl")
    preproc_path_dest = os.path.join(user_model_dir, "preprocessor.pkl")

    dump(evidential, model_path)
    shutil.copy(preprocessor_path, preproc_path_dest)

    if label_encoder is not None:
        le_path = os.path.join(user_model_dir, "label_encoder.pkl")
        dump(label_encoder, le_path)

    # --- Log metrics to MLflow ---
    with mlflow.start_run(run_name=f"{user_id}_evidential"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_score", best_score)
        mlflow.log_metric("log_loss_or_mse", test_log_loss)
        if is_classifier(best_model):
            mlflow.log_metric("prediction_entropy", pred_entropy)
            mlflow.log_metric("mean_uncertainty", mean_uncertainty)
            mlflow.sklearn.log_model(evidential.calibrator, "calibrated_base_model")
        mlflow.log_artifact(preproc_path_dest, "preprocessor.pkl")

    return {
        "status": "success",
        "user_id": user_id,
        "best_params": best_params,
        "best_score": best_score,
        "log_loss_or_mse": test_log_loss,
        "prediction_entropy": pred_entropy,
        "mean_uncertainty": mean_uncertainty,
        "mlflow_experiment": experiment_name,
        "classes": label_encoder.classes_.tolist() if label_encoder is not None else None
    }

# ==========================
# PIPELINE: PREDICTIONS
# ==========================
@app.post("/predict/")
async def predict(user_id: str = Form(...), input_data: UploadFile = File(...)):
    """Make predictions using the saved EvidentialWrapper, returning class names for classification."""
    
    user_model_dir = os.path.join(MODEL_DIR, user_id)
    model_path = os.path.join(user_model_dir, "best_model_evidential.pkl")
    preproc_path = os.path.join(user_model_dir, "preprocessor.pkl")
    le_path = os.path.join(user_model_dir, "label_encoder.pkl")  # for class names

    # Check that required files exist
    if not os.path.exists(model_path) or not os.path.exists(preproc_path):
        raise HTTPException(status_code=404, detail="Model not found. Run training first.")

    evidential = load(model_path)
    preprocessor = load(preproc_path)
    label_encoder = load(le_path) if os.path.exists(le_path) else None

    # Read input CSV and transform
    df = pd.read_csv(input_data.file)
    X = preprocessor.transform(df)

    if hasattr(evidential, "predict_dirichlet"):
        # Classification outputs
        ev = evidential.evidential_metrics(X)
        preds_idx = np.argmax(ev["expected_prob"], axis=1)  # numeric indices
        probs = ev["expected_prob"]
        alphas = ev["alpha"]
        pred_entropy = ev["pred_entropy"]
        uncertainty_mass = ev["uncertainty"]

        # Convert numeric indices to class names if label encoder exists
        if label_encoder is not None:
            preds = label_encoder.inverse_transform(preds_idx)
        else:
            preds = preds_idx  # fallback to numeric indices if no encoder

        # Compute uncertainty flags
        num_classes = probs.shape[1]
        uncertainty_threshold = 0.5
        entropy_threshold = np.log(num_classes) * 0.5
        uncertain_flags = ((uncertainty_mass > uncertainty_threshold) | 
                           (pred_entropy > entropy_threshold)).tolist()

        le_info = None
        if os.path.exists(le_path):
            label_encoder = load(le_path)
            le_info = {"exists": True, "classes": label_encoder.classes_.tolist()}
        else:
            label_encoder = None
            le_info = {"exists": False}

        # Then include in your response
        return {
            "predictions": preds.tolist(),
            "expected_probabilities": probs.tolist(),
            "dirichlet_alpha": alphas.tolist(),
            "prediction_entropy": pred_entropy.tolist(),
            "uncertainty_mass": uncertainty_mass.tolist(),
            "uncertain_predictions": uncertain_flags,
            "label_encoder_info": le_info
}

    else:
        # Regression outputs
        ev = evidential.predict_evidential(X)
        return {
            "mean": ev["mean"].tolist(),
            "variance": ev["variance"].tolist(),
            "std": ev["std"].tolist()
        }
#     MLflow Dashboard APIs
# ==========================
@app.get("/experiments/")
def list_experiments():
    experiments = client.search_experiments()
    return [
        {"experiment_id": exp.experiment_id, "name": exp.name, "lifecycle_stage": exp.lifecycle_stage}
        for exp in experiments
    ]


@app.get("/experiments/{experiment_id}/runs/")
def list_runs(experiment_id: str):
    runs = client.search_runs([experiment_id])
    return [
        {
            "run_id": r.info.run_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "end_time": r.info.end_time,
            "params": r.data.params,
            "metrics": r.data.metrics,
        }
        for r in runs
    ]


@app.get("/runs/{run_id}/")
def get_run_details(run_id: str):
    run = client.get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }


@app.get("/runs/{run_id}/artifacts/")
def get_run_artifacts(run_id: str):
    artifacts = client.list_artifacts(run_id)
    return [{"path": a.path, "is_dir": a.is_dir} for a in artifacts]


@app.get("/experiments/{experiment_id}/best")
def get_best_run(experiment_id: str):
    """Fetch the parent run with best_score & uncertainty metrics."""
    runs = client.search_runs([experiment_id])
    best_run = None
    for run in runs:
        if "best_accuracy" in run.data.metrics:
            best_run = run
            break

    if not best_run:
        raise HTTPException(status_code=404, detail="No parent run with best_score found")

    return {
        "run_id": best_run.info.run_id,
        "experiment_id": experiment_id,
        "best_score": best_run.data.metrics.get("best_accuracy"),
        "best_params": best_run.data.params,
        "log_loss": best_run.data.metrics.get("log_loss"),
        "prediction_entropy": best_run.data.metrics.get("prediction_entropy")
    }


from fastapi import HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

@app.get("/experiments/{experiment_id}/download-best-model/")
def download_best_model(experiment_id: str):
    """
    Download the best model file for a given experiment.
    This works with MLflow's m-<hash> model folder structure.
    """
    experiment_models_dir = Path("mlruns") / experiment_id / "models"
    if not experiment_models_dir.exists():
        raise HTTPException(status_code=404, detail=f"No 'models' folder found at {experiment_models_dir}")

    # List all folders inside 'models'
    model_subfolders = [f for f in experiment_models_dir.iterdir() if f.is_dir()]
    if not model_subfolders:
        raise HTTPException(status_code=404, detail=f"No model folders found under {experiment_models_dir}")

    # Pick the first model folder (MLflow generates m-<hash>)
    model_folder = model_subfolders[0]

    # Path to actual model file
    model_file_path = model_folder / "artifacts" / "model.pkl"
    if not model_file_path.exists():
        # debug: list all files inside folder
        all_files = list(model_folder.rglob("*"))
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found at {model_file_path}. Folder contains: {all_files}"
        )

    return FileResponse(str(model_file_path), media_type="application/octet-stream", filename="best_model.pkl")
















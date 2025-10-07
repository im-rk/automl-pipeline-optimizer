# src/pipeline_runner.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yaml
import importlib.util
from src.components.modeltrainer import ModelTrainer


def load_config(config_path: str) -> dict:
    """Load config file (YAML)."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_train_script(train_script_path: str):
    """
    Dynamically load a user-provided train.py script.
    The train.py must have a function: train_model(config)
    """
    spec = importlib.util.spec_from_file_location("train_module", train_script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module


def run_pipeline(dataset_path: str, train_script_path: str, config_path: str):
    """
    Executes a full ML pipeline:
    1. Load dataset
    2. Load config + training script
    3. Train model using ModelTrainer
    4. Evaluate model
    """

    # 1. Load dataset
    df = pd.read_csv(dataset_path)
    X = df.drop(columns=["target"])   # assume target column is named "target"
    y = df["target"]

    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Load config
    config = load_config(config_path)

    # Inject train/test data into config
    config["train_data"] = (X_train, y_train)
    config["test_data"] = (X_test, y_test)

    # 4. Load user-provided training logic (train.py)
    train_module = load_train_script(train_script_path)

    # 5. Build + train model using user train.py
    model = train_module.train_model(config)

    # 6. Wrap with ModelTrainer (for optional saving, consistency)
    trainer = ModelTrainer(model=model)
    trained_model = trainer.run_train(X_train, X_test, config, save_model=False)

    # 7. Evaluate
    y_pred = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    metrics = {"accuracy": accuracy}
    return trained_model, metrics




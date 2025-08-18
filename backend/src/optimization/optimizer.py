# optimizer.py
import optuna
import json
import yaml
import os
from src.components.modeltrainer import ModelTrainer

class ModelOptimizer:
    def __init__(self, config_path, train_script_path, train_array, test_array):
        self.config_path = config_path
        self.train_script_path = train_script_path
        self.train_array = train_array
        self.test_array = test_array
        self.config = self._load_config()

    def _load_config(self):
        ext = os.path.splitext(self.config_path)[1]
        with open(self.config_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in [".yaml", ".yml"]:
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config format.")

    def objective(self, trial):
        hp_ranges = self.config["hyperparameters"]
        trial_params = {}
        for param, values in hp_ranges.items():
            if isinstance(values[0], int) and isinstance(values[1], int):
                trial_params[param] = trial.suggest_int(param, values[0], values[1])
            elif isinstance(values[0], float) and isinstance(values[1], float):
                trial_params[param] = trial.suggest_float(param, values[0], values[1])
            else:
                raise ValueError(f"Unsupported type for {param}")

        trainer = ModelTrainer(self.train_script_path)
        score = trainer.run_train(self.train_array, self.test_array, trial_params)
        return score

    def optimize(self, n_trials=20):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/optuna_results.json", "w") as f:
            json.dump({
                "best_params": study.best_params,
                "best_score": study.best_value
            }, f, indent=4)

        return study.best_params, study.best_value



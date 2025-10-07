# # optimizer.py

# import optuna
# import json
# import yaml
# import os
# import mlflow
# import mlflow.sklearn
# from src.components.modeltrainer import ModelTrainer


# class ModelOptimizer:
#     def __init__(self, config_path, train_script_path, train_array, test_array, user_id="default_user"):
#         self.config_path = config_path
#         self.train_script_path = train_script_path
#         self.train_array = train_array
#         self.test_array = test_array
#         self.user_id = user_id
#         self.config = self._load_config()
#         self.trainer = ModelTrainer(self.train_script_path)  # initialize trainer once

#     def _load_config(self):
#         ext = os.path.splitext(self.config_path)[1]
#         with open(self.config_path, "r") as f:
#             if ext == ".json":
#                 return json.load(f)
#             elif ext in [".yaml", ".yml"] or ext == ".yml":
#                 return yaml.safe_load(f)
#             else:
#                 raise ValueError("Unsupported config format.")

#     def objective(self, trial):
#         hp_ranges = self.config["hyperparameters"]
#         trial_params = {}

#         # Suggest hyperparameters dynamically
#         for param, values in hp_ranges.items():
#             if isinstance(values[0], int) and isinstance(values[1], int):
#                 trial_params[param] = trial.suggest_int(param, values[0], values[1])
#             elif isinstance(values[0], float) and isinstance(values[1], float):
#                 trial_params[param] = trial.suggest_float(param, values[0], values[1])
#             else:
#                 raise ValueError(f"Unsupported type for {param}")

#         score, _ = self.trainer.run_train(self.train_array, self.test_array, trial_params)

#         # Each trial gets its own MLflow run
#         with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
#             mlflow.log_params(trial_params)
#             mlflow.log_metric("accuracy", score)

#         return score

#     def optimize(self, n_trials=200, patience=10):
#         # Make sure MLflow is tracking locally
#         mlflow.set_tracking_uri("file:./mlruns")

#         # Define early stopping callback
#         class EarlyStopper:
#             def __init__(self, patience):
#                 self.patience = patience
#                 self.best_score = None
#                 self.no_improvement = 0

#             def __call__(self, study, trial):
#                 current_best = study.best_value
#                 if self.best_score is None or current_best > self.best_score:
#                     self.best_score = current_best
#                     self.no_improvement = 0
#                 else:
#                     self.no_improvement += 1

#                 if self.no_improvement >= self.patience:
#                     print(f"Early stopping after {trial.number+1} trials. No improvement for {self.patience} trials.")
#                     study.stop()

#         stopper = EarlyStopper(patience)

#         # Parent run that contains all trials
#         with mlflow.start_run(run_name=f"{self.user_id}_optuna_runs"):

#             study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
#             study.optimize(self.objective, n_trials=n_trials, callbacks=[stopper])

#             # Get best parameters and retrain final model
#             best_params = study.best_params
#             best_score, best_model = self.trainer.run_train(
#                 self.train_array, self.test_array, best_params
#             )

#             # Log best results in parent run
#             mlflow.log_params(best_params)
#             mlflow.log_metric("best_accuracy", best_score)
#             mlflow.sklearn.log_model(best_model, "best_model")

#             # Save results to artifacts
#             os.makedirs("artifacts", exist_ok=True)
#             with open("artifacts/optuna_results.json", "w") as f:
#                 json.dump({
#                     "best_params": best_params,
#                     "best_score": best_score
#                 }, f, indent=4)

#         return best_params, best_score, best_model





# optimizer.py

import optuna
import json
import yaml
import os
import mlflow
import mlflow.sklearn
from src.components.modeltrainer import ModelTrainer


class ModelOptimizer:
    def __init__(self, config_path, train_script_path, train_array, test_array, user_id="default_user"):
        self.config_path = config_path
        self.train_script_path = train_script_path
        self.train_array = train_array
        self.test_array = test_array
        self.user_id = user_id
        self.config = self._load_config()
        self.trainer = ModelTrainer(self.train_script_path)  # initialize trainer once

    def _load_config(self):
        ext = os.path.splitext(self.config_path)[1]
        with open(self.config_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in [".yaml", ".yml"] or ext == ".yml":
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config format.")

    def objective(self, trial):
        hp_ranges = self.config["hyperparameters"]
        trial_params = {}

        # Suggest hyperparameters dynamically
        for param, values in hp_ranges.items():
            if isinstance(values[0], int) and isinstance(values[1], int):
                trial_params[param] = trial.suggest_int(param, values[0], values[1])
            elif isinstance(values[0], float) and isinstance(values[1], float):
                trial_params[param] = trial.suggest_float(param, values[0], values[1])
            else:
                raise ValueError(f"Unsupported type for {param}")

        score, _ = self.trainer.run_train(self.train_array, self.test_array, trial_params)

        # Each trial gets its own MLflow run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric("accuracy", score)

        return score

    def optimize(self, n_trials=200, patience=10):
        # Make sure MLflow is tracking locally
        mlflow.set_tracking_uri("file:./mlruns")

        # Define early stopping callback
        class EarlyStopper:
            def __init__(self, patience):
                self.patience = patience
                self.best_score = None
                self.no_improvement = 0

            def __call__(self, study, trial):
                current_best = study.best_value
                if self.best_score is None or current_best > self.best_score:
                    self.best_score = current_best
                    self.no_improvement = 0
                else:
                    self.no_improvement += 1

                if self.no_improvement >= self.patience:
                    print(f"Early stopping after {trial.number+1} trials. No improvement for {self.patience} trials.")
                    study.stop()

        stopper = EarlyStopper(patience)

        # Parent run that contains all trials
        with mlflow.start_run(run_name=f"{self.user_id}_optuna_runs"):

            study = optuna.create_study(direction="maximize", pruner=optuna.pruners.NopPruner())
            study.optimize(self.objective, n_trials=n_trials, callbacks=[stopper])

            # Get best parameters and retrain final model
            best_params = study.best_params
            best_score, best_model = self.trainer.run_train(
                self.train_array, self.test_array, best_params
            )

            # Log best results in parent run
            mlflow.log_params(best_params)
            mlflow.log_metric("best_accuracy", best_score)
            mlflow.sklearn.log_model(best_model, "best_model")

            # Save results to artifacts
            os.makedirs("artifacts", exist_ok=True)
            with open("artifacts/optuna_results.json", "w") as f:
                json.dump({
                    "best_params": best_params,
                    "best_score": best_score
                }, f, indent=4)

        return best_params, best_score, best_model






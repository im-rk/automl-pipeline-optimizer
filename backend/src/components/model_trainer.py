# model_trainer.py
import importlib.util
import sys
from src.utils.exception import CustomException
import numpy as np

class ModelTrainer:
    def __init__(self, train_script_path):
        """
        Dynamically loads the user-provided train.py
        """
        try:
            spec = importlib.util.spec_from_file_location("train", train_script_path)
            train_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_module)
            self.train_model = train_module.train_model
        except Exception as e:
            raise CustomException(e, sys)

    def run_train(self, train_array, test_array, hyperparameters, save_model=False):
        """
        train_array, test_array: preprocessed arrays (features + target)    
        hyperparameters: dict from optimizer
        save_model: optional flag to save temporary model
        """
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_val, y_val = test_array[:, :-1], test_array[:, -1]

            config = {
                "train_data": (X_train, y_train),
                "val_data": (X_val, y_val),
                "hyperparameters": hyperparameters,
                "save_model": save_model
            }

            score = self.train_model(config)
            return score

        except Exception as e:
            raise CustomException(e, sys)



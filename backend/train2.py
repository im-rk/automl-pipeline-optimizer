from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

def train_model(config):
    X_train, y_train = config["train_data"]
    hp = config.get("hyperparameters", {})

    # Create XGBoost regressor with hyperparameters from config
    model = XGBRegressor(
        n_estimators=hp.get("n_estimators", 100),
        max_depth=hp.get("max_depth", 3),
        learning_rate=hp.get("learning_rate", 0.1),
        subsample=hp.get("subsample", 1.0),
        colsample_bytree=hp.get("colsample_bytree", 1.0),
        random_state=42,
        objective="reg:squarederror"
    )

    # 3-fold cross-validation using RMSE
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="neg_root_mean_squared_error",  # RMSE (negative by convention)
        n_jobs=-1
    )

    # Convert from negative values → actual RMSE
    mean_cv_score = -np.mean(cv_scores)

    return mean_cv_score, model



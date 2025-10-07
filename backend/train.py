from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def train_model(config):
    X_train, y_train = config["train_data"]
    hp = config.get("hyperparameters", {})

    max_depth = hp.get("max_depth", None)
    if max_depth is not None:
        max_depth = int(max_depth)

    model = RandomForestClassifier(
        n_estimators=hp.get("n_estimators", 10),
        max_depth=max_depth,
        random_state=42
    )

    # Use 3-fold cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, n_jobs=-1)
    mean_cv_score = cv_scores.mean()

    return mean_cv_score,model


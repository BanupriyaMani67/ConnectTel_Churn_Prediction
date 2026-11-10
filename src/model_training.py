
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.preprocessing import build_preprocessor
from src.feature_engineering import add_features
from src.data_cleaning import clean_data
from src.data_loader import load_data
from config import MODEL_DIR, RANDOM_STATE, TEST_SIZE, N_SPLITS, ID_COLUMN, TARGET

def train_all(data_path=None, search_iterations=12):
    raw = load_data(data_path) if data_path else load_data()
    df = add_features(clean_data(raw))
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {"model__C": [0.01, 0.1, 1, 5, 10], "model__solver": ["lbfgs"]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
            {"model__n_estimators": [200, 350, 500], "model__max_depth": [None, 6, 10, 16],
             "model__min_samples_split": [2, 5, 10], "model__min_samples_leaf": [1, 2, 4],
             "model__max_features": ["sqrt", 0.8]}
        )
    }

    try:
        from xgboost import XGBClassifier
        neg, pos = np.bincount(y_train)
        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE, eval_metric="logloss",
                scale_pos_weight=neg / max(pos, 1), n_jobs=-1
            ),
            {"model__n_estimators": [150, 250, 350], "model__max_depth": [2, 3, 4],
             "model__learning_rate": [0.03, 0.05, 0.1], "model__subsample": [0.8, 1.0],
             "model__colsample_bytree": [0.8, 1.0], "model__min_child_weight": [1, 3, 5]}
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        models["HistGradientBoosting (XGBoost fallback)"] = (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_iter": [100, 200],
             "model__max_leaf_nodes": [15, 31]}
        )

    results, fitted = {}, {}
    for name, (estimator, params) in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", estimator)])
        search = RandomizedSearchCV(
            pipe, params, n_iter=min(search_iterations, max(1, np.prod([len(v) for v in params.values()]))),
            scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=-1,
            refit=True, verbose=0
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_
        results[name] = {
            "cv_roc_auc_mean": float(search.best_score_),
            "best_params": search.best_params_
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Hold-out test set is used only after tuning is complete.
    from src.evaluation import evaluate_models
    evaluation = evaluate_models(fitted, X_test, y_test)
    best_name = max(evaluation, key=lambda n: evaluation[n]["roc_auc"])
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    joblib.dump(fitted, MODEL_DIR / "all_models.joblib")
    metadata = {
        "best_model": best_name,
        "cv_results": results,
        "test_results": evaluation,
        "features": X.columns.tolist(),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test))
    }
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata, fitted, (X_test, y_test)

if __name__ == "__main__":
    metadata, _, _ = train_all()
    print(json.dumps(metadata, indent=2))

"""Model training, evaluation, and hyperparameter tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


def get_default_models() -> dict:
    """Return a dictionary of default classifier instances."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
    }


def evaluate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Train each model and return a comparison DataFrame.

    Columns: model, train_accuracy, test_accuracy.
    """
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({
            "model": name,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
        })
        print(f"{name}")
        print(f"  Train accuracy: {train_acc:.4f}")
        print(f"  Test accuracy:  {test_acc:.4f}")
        print(f"  Classification report:\n{classification_report(y_test, model.predict(X_test))}")
        print()
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)


def get_param_grids() -> dict:
    """Return GridSearchCV parameter grids for each model."""
    return {
        "Logistic Regression": {
            "C": [0.001, 0.01, 1, 10],
            "solver": ["lbfgs", "liblinear", "newton-cg", "sag"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        },
        "SVC": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
    }


def tune_models(
    models: dict,
    param_grids: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
) -> dict:
    """Run GridSearchCV for models that have a param grid.

    Returns a dict ``{name: best_estimator}``.
    """
    best_models: dict = {}
    for name, model in models.items():
        if name in param_grids:
            gs = GridSearchCV(
                model, param_grids[name], cv=cv, scoring="accuracy", n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            best_models[name] = gs.best_estimator_
            print(f"{name}: best params = {gs.best_params_}, best CV score = {gs.best_score_:.4f}")
        else:
            model.fit(X_train, y_train)
            best_models[name] = model
            print(f"{name}: no grid search (default params)")
    return best_models


def compare_best_models(
    best_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate tuned models on the test set and return comparison table."""
    rows = []
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rows.append({"model": name, "test_accuracy": round(acc, 4)})
        print(f"{name}: test accuracy = {acc:.4f}")
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False)

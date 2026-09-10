"""Feature preparation and scaling pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target: str = "Outcome",
) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into feature matrix *X* and target vector *y*."""
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split preserving class ratio."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple:
    """Fit StandardScaler on *X_train* and transform both sets.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

"""Data loading, validation, and preprocessing utilities."""

import pandas as pd
import numpy as np


ZERO_INVALID_COLUMNS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]


def load_data(filepath: str = "data/diabetes.csv") -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset from *filepath*."""
    df = pd.read_csv(filepath)
    return df


def summarize(df: pd.DataFrame) -> None:
    """Print basic dataset statistics to stdout."""
    print("Shape:", df.shape)
    print()
    print(df.info())
    print()
    print(df.describe())


def check_zero_values(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Return a summary of clinically‑invalid zero values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    columns : list[str], optional
        Columns to check. Defaults to ``ZERO_INVALID_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Table with *count* and *percentage* of zeros per column.
    """
    columns = columns or ZERO_INVALID_COLUMNS
    records = []
    for col in columns:
        zero_count = (df[col] == 0).sum()
        zero_pct = 100 * zero_count / len(df)
        records.append({"column": col, "zero_count": zero_count, "zero_pct": round(zero_pct, 2)})
    return pd.DataFrame(records)


def impute_zeros_with_median(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Replace clinically‑invalid zeros with the column median computed
    **only** from the training set (prevents data leakage).

    Returns copies of *X_train* and *X_test* plus the median mapping.
    """
    columns = columns or ZERO_INVALID_COLUMNS
    X_train = X_train.copy()
    X_test = X_test.copy()
    medians: dict[str, float] = {}

    for col in columns:
        median_val = X_train.loc[X_train[col] != 0, col].median()
        medians[col] = median_val
        X_train[col] = X_train[col].replace(0, median_val)
        X_test[col] = X_test[col].replace(0, median_val)

    return X_train, X_test, medians

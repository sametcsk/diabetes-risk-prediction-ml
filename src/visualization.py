"""Visualization helpers for EDA and model comparison."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (12, 8)) -> None:
    """Draw an annotated correlation heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = "Outcome",
    figsize: tuple = (8, 6),
) -> None:
    """Box plots of each *feature* grouped by *target*."""
    if features is None:
        features = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "Age"]
    for feature in features:
        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=target, y=feature)
        plt.title(f"{feature} Distribution by {target}", fontsize=14)
        plt.tight_layout()
        plt.show()


def plot_model_comparison(results: pd.DataFrame, metric: str = "test_accuracy") -> None:
    """Horizontal bar chart comparing model performance."""
    results_sorted = results.sort_values(metric, ascending=True)
    plt.figure(figsize=(10, 5))
    plt.barh(results_sorted["model"], results_sorted[metric], color="steelblue")
    plt.xlabel(metric.replace("_", " ").title())
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()

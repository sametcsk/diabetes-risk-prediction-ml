#!/usr/bin/env python
"""Diabetes Risk Prediction — CLI entry point.

Run the full pipeline from the terminal:

    python main.py                          # default data path
    python main.py --data data/diabetes.csv # explicit path
    python main.py --skip-plots             # headless / CI mode
"""

import argparse

from src.data_loader import load_data, summarize, check_zero_values, impute_zeros_with_median
from src.feature_engineering import split_features_target, stratified_split, scale_features
from src.model import get_default_models, evaluate_models, get_param_grids, tune_models, compare_best_models
from src.visualization import plot_correlation_heatmap, plot_feature_distributions, plot_model_comparison


def main(data_path: str = "data/diabetes.csv", skip_plots: bool = False) -> None:
    # ---- 1. Load & explore ----
    print("=" * 60)
    print("STEP 1 — Loading data")
    print("=" * 60)
    df = load_data(data_path)
    summarize(df)

    zero_report = check_zero_values(df)
    print("\nClinically-invalid zero values:")
    print(zero_report.to_string(index=False))

    if not skip_plots:
        plot_correlation_heatmap(df)
        plot_feature_distributions(df)

    # ---- 2. Feature engineering ----
    print("\n" + "=" * 60)
    print("STEP 2 — Preparing features")
    print("=" * 60)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    X_train, X_test, medians = impute_zeros_with_median(X_train, X_test)
    print(f"Imputation medians: {medians}")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # ---- 3. Baseline evaluation ----
    print("\n" + "=" * 60)
    print("STEP 3 — Baseline model evaluation")
    print("=" * 60)
    models = get_default_models()
    baseline_results = evaluate_models(models, X_train_scaled, y_train, X_test_scaled, y_test)
    print("\nBaseline comparison:")
    print(baseline_results.to_string(index=False))

    # ---- 4. Hyperparameter tuning ----
    print("\n" + "=" * 60)
    print("STEP 4 — Hyperparameter tuning (GridSearchCV)")
    print("=" * 60)
    models = get_default_models()
    param_grids = get_param_grids()
    best_models = tune_models(models, param_grids, X_train_scaled, y_train)

    tuned_results = compare_best_models(best_models, X_test_scaled, y_test)
    print("\nTuned model comparison:")
    print(tuned_results.to_string(index=False))

    if not skip_plots:
        plot_model_comparison(tuned_results)

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diabetes Risk Prediction Pipeline")
    parser.add_argument("--data", default="data/diabetes.csv", help="Path to CSV dataset")
    parser.add_argument("--skip-plots", action="store_true", help="Skip matplotlib plots (CI mode)")
    args = parser.parse_args()
    main(data_path=args.data, skip_plots=args.skip_plots)

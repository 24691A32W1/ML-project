"""
Train, tune, compare and save the four regression models.

Run:  python -m src.train_models

Model selection uses 5-fold cross-validation on the TRAINING set only. The
held-out 20% test set is used once, to report final metrics for every model.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import (CATEGORY_ORDER, CV_FOLDS, FEATURES, FIG_DIR, MODEL_DIR, NUMERIC_FEATURES,
                     RANDOM_STATE, REPORT_DIR, TARGET)
from .preprocessing import (build_preprocessor, clean_dataframe, data_quality_report,
                            get_split, load_raw)
from .utils import categorize_series

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
    }


def candidate_models():
    """(estimator, param_grid) per algorithm; grids are small to keep training fast."""
    rs = RANDOM_STATE
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Decision Tree": (
            DecisionTreeRegressor(random_state=rs),
            {"model__max_depth": [3, 5, 7, 10], "model__min_samples_leaf": [5, 10, 20]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=rs, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [None, 10],
             "model__min_samples_leaf": [1, 3, 5]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=rs),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }


def save_fig(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


def make_figures(df, results_df, best_name, y_test, y_pred, importance_df, cm, labels):
    # 1. Target distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=30, kde=True, color="#4C78A8")
    plt.title("Distribution of Final Examination Marks (synthetic data)")
    plt.xlabel("Final marks")
    save_fig("01_target_distribution.png")

    # 2. Correlation heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[NUMERIC_FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    save_fig("02_correlation_heatmap.png")

    # 3. Model comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.barplot(data=results_df, x="Model", y="Test_RMSE", ax=axes[0], palette="viridis")
    axes[0].set_title("Test RMSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=20)
    sns.barplot(data=results_df, x="Model", y="Test_R2", ax=axes[1], palette="viridis")
    axes[1].set_title("Test R² (higher is better)")
    axes[1].tick_params(axis="x", rotation=20)
    save_fig("03_model_comparison.png")

    # 4. Actual vs predicted
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, y_pred, alpha=0.5, s=18, color="#F58518")
    lim = [min(y_test.min(), y_pred.min()) - 2, 102]
    plt.plot(lim, lim, "k--", lw=1)
    plt.xlabel("Actual marks"); plt.ylabel("Predicted marks")
    plt.title(f"Actual vs Predicted ({best_name})")
    save_fig("04_actual_vs_predicted.png")

    # 5. Residuals
    plt.figure(figsize=(7, 4))
    sns.histplot(y_test - y_pred, bins=30, kde=True, color="#54A24B")
    plt.axvline(0, color="k", lw=1)
    plt.title(f"Residual Distribution ({best_name})")
    plt.xlabel("Actual - Predicted")
    save_fig("05_residuals.png")

    # 6. Feature importance
    plt.figure(figsize=(7, 4.5))
    top = importance_df.sort_values("Importance")
    plt.barh(top["Feature"], top["Importance"], xerr=top["Std"], color="#72B7B2")
    plt.title("Permutation Importance on Test Set (drop in R²)")
    save_fig("06_feature_importance.png")

    # 7. Confusion matrix of categories
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted category"); plt.ylabel("Actual category")
    plt.title("Performance Category Confusion Matrix")
    plt.xticks(rotation=25)
    save_fig("07_category_confusion_matrix.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    raw = load_raw()
    quality = data_quality_report(raw)
    df = clean_dataframe(raw)
    print(f"Loaded {len(raw)} rows; {quality['total_missing_cells']} missing cells; "
          f"{quality['duplicate_rows_excluding_id']} duplicates; {len(df)} rows after cleaning")

    X_train, X_test, y_train, y_test = get_split(df)
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows = {}, []

    for name, (estimator, grid) in candidate_models().items():
        print(f"Training {name} ...")
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        search = GridSearchCV(pipe, grid or {}, cv=cv, n_jobs=-1,
                              scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                              refit="rmse")
        search.fit(X_train, y_train)
        i = search.best_index_
        pred = search.best_estimator_.predict(X_test)
        pred = np.clip(pred, 0, 100)
        m = evaluate(y_test, pred)
        fitted[name] = search.best_estimator_
        rows.append({
            "Model": name,
            "CV_RMSE": -search.cv_results_["mean_test_rmse"][i],
            "CV_RMSE_Std": search.cv_results_["std_test_rmse"][i],
            "CV_R2": search.cv_results_["mean_test_r2"][i],
            "Test_MAE": m["MAE"], "Test_MSE": m["MSE"], "Test_RMSE": m["RMSE"], "Test_R2": m["R2"],
            "Best_Params": json.dumps({k.replace("model__", ""): v for k, v in search.best_params_.items()}, default=str),
        })

    results = pd.DataFrame(rows).sort_values("CV_RMSE").reset_index(drop=True)
    best_name = results.loc[0, "Model"]   # chosen by cross-validation, NOT by test score
    best_pipe = fitted[best_name]
    print("\n", results.drop(columns="Best_Params").round(4).to_string(index=False))
    print(f"\nBest model (lowest 5-fold CV RMSE): {best_name}")

    # ---- Final evaluation of the selected model --------------------------
    y_pred = np.clip(best_pipe.predict(X_test), 0, 100)
    final_metrics = evaluate(y_test, y_pred)

    actual_cat, pred_cat = categorize_series(y_test), categorize_series(pd.Series(y_pred, index=y_test.index))
    labels = [c for c in CATEGORY_ORDER[::-1]]
    cm = confusion_matrix(actual_cat, pred_cat, labels=labels)
    cat_acc = accuracy_score(actual_cat, pred_cat)
    cat_report = classification_report(actual_cat, pred_cat, labels=labels, output_dict=True, zero_division=0)

    # ---- Factor analysis ---------------------------------------------------
    pi = permutation_importance(best_pipe, X_test, y_test, n_repeats=15,
                                random_state=RANDOM_STATE, scoring="r2")
    importance = pd.DataFrame({"Feature": FEATURES, "Importance": pi.importances_mean,
                               "Std": pi.importances_std}).sort_values("Importance", ascending=False)

    # ---- Persist artefacts -------------------------------------------------
    joblib.dump(best_pipe.named_steps["preprocessor"], MODEL_DIR / "preprocessor.joblib")
    joblib.dump(best_pipe.named_steps["model"], MODEL_DIR / "best_model.joblib")
    joblib.dump(best_pipe, MODEL_DIR / "best_pipeline.joblib")

    results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred.round(2),
                  "Actual_Category": actual_cat.values, "Predicted_Category": pred_cat.values}
                 ).to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    metadata = {
        "best_model": best_name,
        "selection_rule": f"lowest mean {CV_FOLDS}-fold CV RMSE on the training set",
        "test_metrics": final_metrics,
        "category_accuracy": cat_acc,
        "category_report": cat_report,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "features": FEATURES,
        "data_quality": quality,
        "feature_ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())]
                           for c in NUMERIC_FEATURES},
        "trained_on": pd.Timestamp.now().isoformat(timespec="seconds"),
        "data_source": "SYNTHETIC (generated by src/generate_dataset.py), not real student data",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))

    make_figures(df, results, best_name, y_test, pd.Series(y_pred, index=y_test.index),
                 importance, cm, labels)

    print(f"Test metrics for {best_name}: " + ", ".join(f"{k}={v:.3f}" for k, v in final_metrics.items()))
    print(f"Category accuracy on test set: {cat_acc:.3f}")
    print(f"Done in {time.time() - t0:.1f}s. Artefacts saved in models/ and reports/")


if __name__ == "__main__":
    main()

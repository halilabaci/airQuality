from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".pydeps"
if LOCAL_DEPS.exists():
    sys.path.append(str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


HORIZONS = (1, 8, 24)
TARGET_COLUMN = "PM2.5"
OUTPUT_DIRNAME = "model_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PM2.5 multi-horizon regression experiments."
    )
    parser.add_argument(
        "--data",
        default="pilot_clean_encoded.csv",
        help="Input CSV path. Defaults to the pilot dataset for faster iteration.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIRNAME,
        help="Directory where result files will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological test split ratio.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of features to keep during mutual-information selection.",
    )
    parser.add_argument(
        "--svr-max-train-samples",
        type=int,
        default=8000,
        help="Cap the SVR training size so experiments stay tractable.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "No" in df.columns:
        df = df.drop(columns=["No"])

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    sort_cols = [col for col in ["year", "month", "day", "hour"] if col in df.columns]
    station_cols = [col for col in df.columns if col.startswith("station_")]
    if station_cols:
        df["_station_group"] = df[station_cols].idxmax(axis=1)
        df = df.sort_values(["_station_group", *sort_cols]).reset_index(drop=True)
    elif sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def add_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    station_group = "_station_group" if "_station_group" in df.columns else None
    result = df.copy()

    for horizon in horizons:
        target_name = f"target_t_plus_{horizon}"
        if station_group:
            result[target_name] = result.groupby(station_group)[TARGET_COLUMN].shift(-horizon)
        else:
            result[target_name] = result[TARGET_COLUMN].shift(-horizon)

    return result


def build_model_specs(random_state: int = 42) -> dict[str, Pipeline]:
    scaled_features = ["Linear Regression", "SVR"]
    specs: dict[str, Pipeline] = {}

    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=random_state
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=140,
            max_depth=16,
            min_samples_leaf=3,
            n_jobs=1,
            random_state=random_state,
        ),
        "SVR": SVR(C=10.0, epsilon=0.1, kernel="rbf"),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=160, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
    }

    for model_name, estimator in estimators.items():
        if model_name in scaled_features:
            specs[model_name] = Pipeline(
                [
                    (
                        "preprocess",
                        ColumnTransformer(
                            [("scale", StandardScaler(), slice(0, None))],
                            remainder="drop",
                        ),
                    ),
                    ("model", estimator),
                ]
            )
        else:
            specs[model_name] = Pipeline([("model", estimator)])

    return specs


def chronological_split(
    X: pd.DataFrame, y: pd.Series, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(math.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def run_model(
    base_pipeline: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    svr_max_train_samples: int,
) -> tuple[Pipeline, np.ndarray, int]:
    pipeline = clone(base_pipeline)
    train_count = len(X_train)

    if model_name == "SVR" and len(X_train) > svr_max_train_samples:
        sampled_index = np.linspace(0, len(X_train) - 1, svr_max_train_samples, dtype=int)
        X_fit = X_train.iloc[sampled_index]
        y_fit = y_train.iloc[sampled_index]
        train_count = len(X_fit)
    else:
        X_fit = X_train
        y_fit = y_train

    pipeline.fit(X_fit, y_fit)
    predictions = pipeline.predict(X_test)
    return pipeline, predictions, train_count


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, top_k: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selector = SelectKBest(score_func=mutual_info_regression, k=min(top_k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    support = selector.get_support()

    selected = pd.DataFrame(
        {
            "feature": X_train.columns,
            "score": selector.scores_,
            "selected": support,
        }
    ).sort_values(["selected", "score"], ascending=[False, False])

    selected_columns = selected.loc[selected["selected"], "feature"].tolist()
    return X_train[selected_columns], X_test[selected_columns], selected


def compute_feature_importance(
    trained_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    horizon: int,
    model_name: str,
) -> pd.DataFrame:
    sample_size = min(len(X_test), 3000)
    sample_index = np.linspace(0, len(X_test) - 1, sample_size, dtype=int)
    X_sample = X_test.iloc[sample_index]
    y_sample = y_test.iloc[sample_index]

    importance = permutation_importance(
        trained_pipeline,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "horizon": horizon,
            "model": model_name,
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def build_report(
    dataset_name: str,
    baseline_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    feature_rankings: dict[int, pd.DataFrame],
    importance_df: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Models / Experimental Results / Comparison")
    lines.append("")
    lines.append(f"Dataset used for the experiments: `{dataset_name}`.")
    lines.append(
        "Targets were created as PM2.5 forecasts for t+1, t+8, and t+24 hours using chronological ordering."
    )
    lines.append(
        "A chronological 80/20 train-test split was used to avoid leaking future observations into training."
    )
    lines.append("")
    lines.append("## Models")
    lines.append(
        "The following regressors were evaluated: Linear Regression, Decision Tree Regressor, Random Forest Regressor, SVR, and Gradient Boosting Regressor."
    )
    lines.append(
        "MAE, RMSE, and R2 were reported for each horizon. Lower MAE/RMSE and higher R2 indicate better performance."
    )
    lines.append("")
    lines.append("## Experimental Results")
    for horizon in HORIZONS:
        baseline_best = (
            baseline_df[baseline_df["horizon"] == horizon]
            .sort_values("RMSE", ascending=True)
            .iloc[0]
        )
        selected_best = (
            selected_df[selected_df["horizon"] == horizon]
            .sort_values("RMSE", ascending=True)
            .iloc[0]
        )
        lines.append(
            f"For t+{horizon}, the best baseline model was {baseline_best['model']} with RMSE={baseline_best['RMSE']:.3f}, "
            f"MAE={baseline_best['MAE']:.3f}, and R2={baseline_best['R2']:.3f}."
        )
        lines.append(
            f"After feature selection, the best model was {selected_best['model']} with RMSE={selected_best['RMSE']:.3f}, "
            f"MAE={selected_best['MAE']:.3f}, and R2={selected_best['R2']:.3f}."
        )
    lines.append("")
    lines.append("## Comparison")
    for _, row in comparison_df.iterrows():
        direction = "improved" if row["rmse_delta"] < 0 else "worsened"
        lines.append(
            f"For t+{int(row['horizon'])}, feature selection {direction} the best RMSE by {abs(row['rmse_delta']):.3f} "
            f"({row['baseline_best_model']} -> {row['selected_best_model']})."
        )
    lines.append("")
    lines.append("## Selected Features")
    for horizon in HORIZONS:
        top_features = feature_rankings[horizon].head(5)["feature"].tolist()
        lines.append(f"Top selected features for t+{horizon}: {', '.join(top_features)}.")
    lines.append("")
    lines.append("## Best-Model Interpretation")
    best_row = comparison_df.sort_values("selected_best_rmse", ascending=True).iloc[0]
    best_importance = importance_df[
        (importance_df["horizon"] == best_row["horizon"])
        & (importance_df["model"] == best_row["selected_best_model"])
    ].head(5)
    top_pairs = ", ".join(
        f"{row.feature} ({row.importance_mean:.3f})" for row in best_importance.itertuples()
    )
    lines.append(
        f"The strongest selected-model result came from {best_row['selected_best_model']} at t+{int(best_row['horizon'])}. "
        f"The most influential features according to permutation importance were {top_pairs}."
    )
    lines.append("")
    lines.append("These paragraphs can be adapted directly into the report's Models, Experimental Results, and Comparison sections.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = add_targets(load_dataset(data_path), HORIZONS)
    feature_columns = [
        col
        for col in df.columns
        if col not in {"_station_group", *[f"target_t_plus_{h}" for h in HORIZONS]}
    ]
    model_specs = build_model_specs()

    baseline_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    selection_tables: list[pd.DataFrame] = []
    importance_tables: list[pd.DataFrame] = []
    trained_selected_models: dict[int, tuple[str, Pipeline, pd.DataFrame, pd.Series]] = {}
    feature_rankings: dict[int, pd.DataFrame] = {}

    for horizon in HORIZONS:
        target_name = f"target_t_plus_{horizon}"
        horizon_df = df.dropna(subset=[target_name]).copy()
        X = horizon_df[feature_columns]
        y = horizon_df[target_name]
        X_train, X_test, y_train, y_test = chronological_split(X, y, args.test_size)

        X_train_sel, X_test_sel, selected_table = select_features(
            X_train, y_train, X_test, args.top_k
        )
        selected_table.insert(0, "horizon", horizon)
        selection_tables.append(selected_table)
        feature_rankings[horizon] = selected_table[selected_table["selected"]].copy()

        best_selected_rmse = float("inf")
        best_selected_bundle: tuple[str, Pipeline, pd.DataFrame, pd.Series] | None = None

        for model_name, pipeline in model_specs.items():
            trained_model, predictions, train_count = run_model(
                pipeline,
                model_name,
                X_train,
                y_train,
                X_test,
                args.svr_max_train_samples,
            )
            metrics = evaluate_predictions(y_test, predictions)
            baseline_rows.append(
                {
                    "dataset": data_path.name,
                    "variant": "baseline",
                    "horizon": horizon,
                    "model": model_name,
                    "train_rows": train_count,
                    "test_rows": len(X_test),
                    **metrics,
                }
            )

            trained_selected, selected_predictions, selected_train_count = run_model(
                pipeline,
                model_name,
                X_train_sel,
                y_train,
                X_test_sel,
                args.svr_max_train_samples,
            )
            selected_metrics = evaluate_predictions(y_test, selected_predictions)
            selected_rows.append(
                {
                    "dataset": data_path.name,
                    "variant": "selected",
                    "horizon": horizon,
                    "model": model_name,
                    "train_rows": selected_train_count,
                    "test_rows": len(X_test_sel),
                    "selected_feature_count": X_train_sel.shape[1],
                    **selected_metrics,
                }
            )

            if selected_metrics["RMSE"] < best_selected_rmse:
                best_selected_rmse = selected_metrics["RMSE"]
                best_selected_bundle = (model_name, trained_selected, X_test_sel, y_test)

        if best_selected_bundle is None:
            raise RuntimeError(f"No selected model was trained for horizon {horizon}.")
        trained_selected_models[horizon] = best_selected_bundle

    baseline_df = pd.DataFrame(baseline_rows).sort_values(["horizon", "RMSE", "model"])
    selected_df = pd.DataFrame(selected_rows).sort_values(["horizon", "RMSE", "model"])
    selection_df = pd.concat(selection_tables, ignore_index=True)

    comparison_rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        baseline_best = baseline_df[baseline_df["horizon"] == horizon].iloc[0]
        selected_best = selected_df[selected_df["horizon"] == horizon].iloc[0]
        comparison_rows.append(
            {
                "horizon": horizon,
                "baseline_best_model": baseline_best["model"],
                "baseline_best_rmse": baseline_best["RMSE"],
                "selected_best_model": selected_best["model"],
                "selected_best_rmse": selected_best["RMSE"],
                "rmse_delta": selected_best["RMSE"] - baseline_best["RMSE"],
            }
        )

        model_name, trained_model, X_test_sel, y_test = trained_selected_models[horizon]
        importance_tables.append(
            compute_feature_importance(
                trained_model, X_test_sel, y_test, horizon=horizon, model_name=model_name
            )
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values("horizon")
    importance_df = pd.concat(importance_tables, ignore_index=True)

    baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    selected_df.to_csv(output_dir / "selected_metrics.csv", index=False)
    comparison_df.to_csv(output_dir / "feature_selection_comparison.csv", index=False)
    selection_df.to_csv(output_dir / "selected_features.csv", index=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    report_text = build_report(
        dataset_name=data_path.name,
        baseline_df=baseline_df,
        selected_df=selected_df,
        comparison_df=comparison_df,
        feature_rankings=feature_rankings,
        importance_df=importance_df,
    )
    (output_dir / "report_draft.md").write_text(report_text, encoding="utf-8")

    print(f"Saved results to {output_dir.resolve()}")
    print("")
    print("Best selected-model RMSE values by horizon:")
    print(selected_df.groupby("horizon").first()[["model", "RMSE", "MAE", "R2"]].to_string())


if __name__ == "__main__":
    main()

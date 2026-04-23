from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CLASS_NAME_MAP = {
    0: "Good",
    1: "Moderate",
    2: "Unhealthy for Sensitive Groups",
    3: "Unhealthy",
    4: "Very Unhealthy",
    5: "Hazardous",
}

# IMPORTANT: These columns must not be used as model features.
# PM2.5 directly creates air_quality_label, so keeping it would cause data leakage.
LEAKAGE_COLUMNS = ["PM2.5", "air_quality_class", "air_quality_label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PM2.5-derived air quality classification experiments."
    )
    parser.add_argument(
        "--input",
        default="air_quality_with_classes.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/classification",
        help="Directory where experiment outputs will be saved.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio (default: 0.2 = 20%%).",
    )
    return parser.parse_args()


def detect_temporal_sort_columns(df: pd.DataFrame) -> list[str]:
    """
    Decide how to sort rows chronologically.

    Priority:
    1) If classic time-part columns exist (year, month, day, hour), use them.
    2) Else look for datetime-like columns and use the first parseable one.
    3) Else return empty list and keep current row order.
    """
    ordered_time_parts = ["year", "month", "day", "hour", "minute", "second"]
    present_time_parts = [
        col for col in ordered_time_parts if col in df.columns]

    if present_time_parts:
        return present_time_parts

    datetime_keywords = ("date", "time", "timestamp", "datetime")
    candidate_cols = [
        col for col in df.columns if any(key in col.lower() for key in datetime_keywords)
    ]

    for col in candidate_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        success_ratio = parsed.notna().mean()
        if success_ratio >= 0.8:
            return [col]

    return []


def chronological_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_idx = int(len(X) * (1.0 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        col for col in X.columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def run_experiments(df: pd.DataFrame, output_dir: Path, test_size: float) -> pd.DataFrame:
    if "air_quality_label" not in df.columns:
        raise ValueError(
            "The target column 'air_quality_label' was not found in the dataset.")

    # Sort by time columns when available; otherwise keep the current row order.
    sort_columns = detect_temporal_sort_columns(df)
    if sort_columns:
        print(f"Chronological split enabled. Sorting by: {sort_columns}")
        df = df.sort_values(sort_columns).reset_index(drop=True)
    else:
        print(
            "No explicit datetime column found. Using current row order as time sequence.")

    y = df["air_quality_label"].copy()

    # Remove leakage columns from features before training.
    feature_drop_columns = [
        col for col in LEAKAGE_COLUMNS if col in df.columns]
    X = df.drop(columns=feature_drop_columns)
    
    bool_cols= X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        print(f"Converting boolean feature columns to 0/1: {bool_cols}")
        X[bool_cols] = X[bool_cols].astype(int)

    X_train, X_test, y_train, y_test = chronological_train_test_split(
        X, y, test_size=test_size)

    preprocessor = build_preprocessor(X_train)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, n_jobs=None),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    results_rows: list[dict[str, float | str | None]] = []

    for model_name, model in models.items():
        print(f"\nRunning model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        train_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        training_time = time.perf_counter() - train_start

        test_start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        testing_time = time.perf_counter() - test_start

        accuracy = accuracy_score(y_test, y_pred)
        macro_precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(
            y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        weighted_precision = precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        )
        weighted_recall = recall_score(
            y_test, y_pred, average="weighted", zero_division=0)
        weighted_f1 = f1_score(
            y_test, y_pred, average="weighted", zero_division=0)

        roc_auc_value: float | None = None
        try:
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)
                roc_auc_value = float(
                    roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        except Exception:
            roc_auc_value = None

        # Save confusion matrix to CSV.
        label_order = sorted(CLASS_NAME_MAP.keys())
        cm = confusion_matrix(y_test, y_pred, labels=label_order)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{CLASS_NAME_MAP[label]}" for label in label_order],
            columns=[f"pred_{CLASS_NAME_MAP[label]}" for label in label_order],
        )
        cm_df.to_csv(
            output_dir / f"confusion_matrix_{model_name}.csv", index=True)

        # Save classification report to TXT.
        class_report = classification_report(
            y_test,
            y_pred,
            labels=label_order,
            target_names=[CLASS_NAME_MAP[label] for label in label_order],
            zero_division=0,
        )
        (output_dir / f"classification_report_{model_name}.txt").write_text(
            class_report,
            encoding="utf-8",
        )

        # Save metrics row.
        results_rows.append(
            {
                "model": model_name,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "training_time_sec": float(training_time),
                "testing_time_sec": float(testing_time),
                "accuracy": float(accuracy),
                "macro_precision": float(macro_precision),
                "macro_recall": float(macro_recall),
                "macro_f1": float(macro_f1),
                "weighted_precision": float(weighted_precision),
                "weighted_recall": float(weighted_recall),
                "weighted_f1": float(weighted_f1),
                "roc_auc_ovr_macro": roc_auc_value,
            }
        )

        # Save feature importance for tree model (first available among RF/ET).
        if model_name in {"RandomForestClassifier", "ExtraTreesClassifier"}:
            feature_importance_path = output_dir / "feature_importance.csv"
            if not feature_importance_path.exists():
                model_step = pipeline.named_steps["model"]
                preprocess_step = pipeline.named_steps["preprocess"]
                transformed_feature_names = preprocess_step.get_feature_names_out()

                fi_df = pd.DataFrame(
                    {
                        "feature": transformed_feature_names,
                        "importance": model_step.feature_importances_,
                        "model": model_name,
                    }
                ).sort_values("importance", ascending=False)
                fi_df.to_csv(feature_importance_path, index=False)

    results_df = pd.DataFrame(results_rows).sort_values(
        "macro_f1", ascending=False)
    return results_df


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)

    results_df = run_experiments(
        df=df, output_dir=output_dir, test_size=args.test_size)

    results_path = output_dir / "classification_results.csv"
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0]
    print("\nBest model by macro F1-score:")
    print(
        f"{best_row['model']} | macro_f1={best_row['macro_f1']:.4f} | "
        f"accuracy={best_row['accuracy']:.4f}"
    )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()

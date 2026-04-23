from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

# Change this variable if your PM2.5 column name is different.
PM25_COLUMN = "PM2.5"

# AQI-like PM2.5 categories based on health impact ranges.
CLASS_LABEL_MAP = {
    "Good": 0,
    "Moderate": 1,
    "Unhealthy for Sensitive Groups": 2,
    "Unhealthy": 3,
    "Very Unhealthy": 4,
    "Hazardous": 5,
}


def pm25_to_air_quality_class(pm25_value: Any) -> str | None:
    """Convert a PM2.5 numeric value to an AQI-like air quality class."""
    if pd.isna(pm25_value):
        return None

    value = float(pm25_value)

    if 0.0 <= value <= 12.0:
        return "Good"
    if 12.0 < value <= 35.4:
        return "Moderate"
    if 35.4 < value <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if 55.4 < value <= 150.4:
        return "Unhealthy"
    if 150.4 < value <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def add_air_quality_columns(
    df: pd.DataFrame,
    pm25_column: str = PM25_COLUMN,
    low_class_threshold_percent: float = 5.0,
) -> pd.DataFrame:
    """Add categorical and numeric air quality labels to a DataFrame."""
    if pm25_column not in df.columns:
        raise ValueError(
            f"Column '{pm25_column}' not found. Available columns: {list(df.columns)}"
        )

    result_df = df.copy()

    # Check missing PM2.5 values before classification.
    missing_pm25_count = int(result_df[pm25_column].isna().sum())
    print(f"Missing PM2.5 values in '{pm25_column}': {missing_pm25_count}")

    # Create categorical air-quality class from PM2.5 values.
    result_df["air_quality_class"] = result_df[pm25_column].apply(pm25_to_air_quality_class)

    # Create numeric labels for modeling.
    result_df["air_quality_label"] = result_df["air_quality_class"].map(CLASS_LABEL_MAP)

    # Count table by class.
    class_counts = (
        result_df["air_quality_class"]
        .value_counts(dropna=False)
        .rename_axis("air_quality_class")
        .reset_index(name="count")
    )
    print("\nClass counts:")
    print(class_counts.to_string(index=False))

    # Percentage distribution table by class.
    class_percentages = (
        result_df["air_quality_class"]
        .value_counts(dropna=False, normalize=True)
        .mul(100)
        .rename_axis("air_quality_class")
        .reset_index(name="percentage")
    )
    class_percentages["percentage"] = class_percentages["percentage"].round(2)

    print("\nClass distribution (%):")
    print(class_percentages.to_string(index=False))

    # Comment if some classes are underrepresented.
    non_null_percentages = class_percentages[
        class_percentages["air_quality_class"].notna()
    ].copy()
    low_classes = non_null_percentages[
        non_null_percentages["percentage"] < low_class_threshold_percent
    ]

    if low_classes.empty:
        print(
            "\nClass balance comment: No class is below "
            f"{low_class_threshold_percent:.1f}%."
        )
    else:
        low_class_names = ", ".join(low_classes["air_quality_class"].astype(str).tolist())
        print(
            "\nClass balance comment: These classes are low-frequency "
            f"(< {low_class_threshold_percent:.1f}%): {low_class_names}."
        )
        print(
            "Consider class-weighted training, re-sampling, or collecting more data "
            "for these classes."
        )

    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add AQI-like PM2.5 class and numeric label columns to a cleaned dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="air_quality_all_clean_encoded.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="air_quality_with_classes.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pm25-column",
        default=PM25_COLUMN,
        help="Name of the PM2.5 column (examples: PM2.5, PM2_5, pm25, target).",
    )
    parser.add_argument(
        "--low-class-threshold-percent",
        type=float,
        default=5.0,
        help="Threshold used to flag low-frequency classes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)
    df_with_classes = add_air_quality_columns(
        df,
        pm25_column=args.pm25_column,
        low_class_threshold_percent=args.low_class_threshold_percent,
    )

    df_with_classes.to_csv(args.output, index=False)
    print(f"\nSaved output file: {args.output}")

    # Short note requested by the project scope.
    print(
        "\nNote: This classification task is created as an extension to your "
        "original PM2.5 regression problem."
    )
    print(
        "The class intervals are AQI-like PM2.5 ranges commonly used to reflect "
        "health-related air quality categories."
    )


if __name__ == "__main__":
    main()

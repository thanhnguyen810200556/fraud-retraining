from pathlib import Path

import pandas as pd

# ==================================================
# 1. Dataset overview
# ==================================================


def dataset_overview(df, dataset_name):
    """Generate basic dataset statistics."""

    total_cells = df.shape[0] * df.shape[1]
    total_missing = int(df.isna().sum().sum())

    overview = {
        "dataset_name": dataset_name,
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "numeric_columns": len(df.select_dtypes(include="number").columns),
        "non_numeric_columns": len(df.select_dtypes(exclude="number").columns),
        "total_missing_values": total_missing,
        "missing_rate_pct": (
            total_missing / total_cells * 100 if total_cells > 0 else 0
        ),
        "memory_usage_mb": (df.memory_usage(deep=True).sum() / 1024**2),
    }

    return pd.DataFrame([overview])


# ==================================================
# 2. Column-level profiling
# ==================================================


def column_profile(df):
    """Generate statistics for every column."""

    profile = pd.DataFrame(
        {
            "column_name": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_rate_pct": (df.isna().mean() * 100).values,
            "unique_count": df.nunique(dropna=True).values,
        }
    )

    profile["unique_rate_pct"] = profile["unique_count"] / len(df) * 100

    profile["is_constant"] = profile["unique_count"] == 1

    profile["is_all_missing"] = profile["non_null_count"] == 0

    return profile


# ==================================================
# 3. Missing value analysis
# ==================================================


def missing_value_report(df):
    """Return columns containing missing values."""

    profile = column_profile(df)

    missing = profile.loc[
        profile["missing_count"] > 0,
        ["column_name", "dtype", "missing_count", "missing_rate_pct"],
    ]

    return missing.sort_values("missing_rate_pct", ascending=False).reset_index(
        drop=True
    )


# ==================================================
# 4. Duplicate analysis
# ==================================================


def duplicate_report(df, id_column=None, check_full_rows=False):
    """Check duplicate IDs and optionally full rows."""

    result = {
        "total_rows": len(df),
        "duplicate_rows": None,
        "duplicate_id_count": None,
        "missing_id_count": None,
    }

    if check_full_rows:
        result["duplicate_rows"] = int(df.duplicated().sum())

    if id_column is not None:

        if id_column not in df.columns:
            raise ValueError(f"ID column not found: {id_column}")

        valid_ids = df[id_column].dropna()

        result["duplicate_id_count"] = int(valid_ids.duplicated().sum())

        result["missing_id_count"] = int(df[id_column].isna().sum())

    return pd.DataFrame([result])


# ==================================================
# 5. Target distribution
# ==================================================


def target_distribution(df, target_column):
    """Calculate class distribution."""

    counts = df[target_column].value_counts(dropna=False).sort_index()

    result = counts.rename("count").reset_index()

    result.columns = ["label", "count"]

    result["percentage"] = result["count"] / len(df) * 100

    return result


def target_validation(df, target_column):
    """Check whether labels are binary and non-missing."""

    target = df[target_column]

    invalid_mask = target.notna() & ~target.isin([0, 1])

    result = {
        "target_column": target_column,
        "total_rows": len(target),
        "missing_labels": int(target.isna().sum()),
        "invalid_labels": int(invalid_mask.sum()),
        "fraud_count": int((target == 1).sum()),
        "legitimate_count": int((target == 0).sum()),
        "fraud_rate_pct": ((target == 1).sum() / len(target) * 100),
    }

    return pd.DataFrame([result])


# ==================================================
# 6. Numerical statistics
# ==================================================


def numerical_statistics(df):
    """Generate descriptive statistics."""

    numeric_df = df.select_dtypes(include="number")

    statistics = numeric_df.describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.95, 0.99]).T

    statistics.index.name = "column_name"

    return statistics.reset_index()


# ==================================================
# 7. Export report
# ==================================================


def export_report(df, output_path):
    """Save profiling results as CSV."""

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")

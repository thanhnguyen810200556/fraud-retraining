from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from src.data_loader import (
    load_config,
    resolve_path,
    validate_required_columns,
)

# ==================================================
# 1. Feature configuration
# ==================================================


def load_feature_config():
    """Load feature configuration from YAML."""

    config = load_config()

    path = resolve_path(config["paths"]["features_config"])

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==================================================
# 2. Data validation
# ==================================================


def validate_binary_target(df, target_column):
    """Ensure that target values are binary and non-missing."""

    validate_required_columns(df, [target_column])

    target = df[target_column]

    if target.isna().any():
        raise ValueError(f"{target_column} contains missing labels.")

    invalid_mask = ~target.isin([0, 1])

    if invalid_mask.any():
        invalid_values = target[invalid_mask].unique()

        raise ValueError(f"Invalid labels: {invalid_values}")


def validate_unique_id(df, id_column):
    """Ensure that record identifiers are unique."""

    validate_required_columns(df, [id_column])

    if df[id_column].isna().any():
        raise ValueError(f"{id_column} contains missing IDs.")

    duplicate_count = int(df[id_column].duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(f"{id_column} contains " f"{duplicate_count} duplicate IDs.")


def validate_integer_column(
    df,
    column,
    min_value=None,
    max_value=None,
):
    """Validate a non-missing integer-valued column."""

    validate_required_columns(df, [column])

    values = pd.to_numeric(
        df[column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(f"{column} contains missing values.")

    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{column} contains non-finite values.")

    if not (values % 1 == 0).all():
        raise ValueError(f"{column} contains non-integer values.")

    if min_value is not None:
        if (values < min_value).any():
            raise ValueError(f"{column} contains values below " f"{min_value}.")

    if max_value is not None:
        if (values > max_value).any():
            raise ValueError(f"{column} contains values above " f"{max_value}.")


def validate_no_infinite_values(df):
    """Reject infinite values in numerical columns."""

    numeric_columns = df.select_dtypes(include="number").columns

    invalid_columns = []

    for column in numeric_columns:

        values = df[column].to_numpy(
            dtype="float64",
            na_value=np.nan,
        )

        if np.isinf(values).any():
            invalid_columns.append(column)

    if invalid_columns:
        raise ValueError("Infinite values found in columns: " f"{invalid_columns}")


# ==================================================
# 3. Feature dtype normalization
# ==================================================


def normalize_feature_types(
    df,
    categorical_columns,
    protected_columns=None,
):
    """
    Normalize feature dtypes without fitting
    data-dependent transformations.

    Missing values are preserved.
    """

    config = load_config()["preprocessing"]

    float_dtype = config["float_dtype"]

    protected_columns = set(protected_columns or [])

    categorical_columns = set(categorical_columns)

    for column in df.columns:

        if column in protected_columns:
            continue

        if column in categorical_columns:

            df[column] = df[column].astype("string")

        elif pd.api.types.is_bool_dtype(df[column]):

            df[column] = df[column].astype("int8")

        elif pd.api.types.is_float_dtype(df[column]):

            df[column] = df[column].astype(float_dtype)

        elif pd.api.types.is_integer_dtype(df[column]):

            # Preserve existing integer precision.
            continue

        else:

            raise TypeError(
                f"Unconfigured feature dtype: " f"{column} ({df[column].dtype})"
            )

    return df


# ==================================================
# 4. IEEE-CIS feature sets
# ==================================================


def get_ieee_feature_columns(
    df,
    feature_set="NoCardID",
):
    """Return an explicit IEEE-CIS feature list."""

    config = load_config()["datasets"]["ieee_cis"]

    feature_config = load_feature_config()["ieee_cis"]

    excluded = set(config["excluded_features"])

    excluded.update(feature_config["temporal_metadata"])

    excluded.update(feature_config["simulation_metadata"])

    excluded.update(feature_config["runtime_metadata"])

    if feature_set == "NoCardID":

        excluded.update(config["no_card_id_columns"])

    elif feature_set != "FullFeatures":

        raise ValueError(f"Unknown feature set: {feature_set}")

    return [column for column in df.columns if column not in excluded]


# ==================================================
# 5. IEEE-CIS preprocessing
# ==================================================


def preprocess_ieee(df):
    """Prepare IEEE-CIS transaction-only data."""

    config = load_config()["datasets"]["ieee_cis"]

    feature_config = load_feature_config()["ieee_cis"]

    id_column = config["id_column"]
    time_column = config["time_column"]
    target_column = config["target_column"]
    amount_column = config["amount_column"]

    if config["use_identity"]:
        raise ValueError("This pipeline version expects " "use_identity=false.")

    validate_required_columns(
        df,
        [
            id_column,
            time_column,
            target_column,
            amount_column,
        ]
        + config["no_card_id_columns"]
        + feature_config["categorical_features"],
    )

    expected_shape = (
        config["expected_transaction_rows"],
        config["expected_transaction_columns"],
    )

    if df.shape != expected_shape:
        raise ValueError(
            f"Unexpected IEEE-CIS shape: {df.shape}. " f"Expected: {expected_shape}"
        )

    # Validate before casting.
    validate_unique_id(df, id_column)

    validate_binary_target(df, target_column)

    validate_integer_column(
        df,
        id_column,
        min_value=0,
    )

    validate_integer_column(
        df,
        time_column,
        min_value=0,
    )

    validate_no_infinite_values(df)

    df = df.copy()

    original_missing = df.isna().sum()

    original_fraud_count = int(df[target_column].sum())

    # Normalize metadata and target.
    df[id_column] = df[id_column].astype("int64")

    df[time_column] = df[time_column].astype("int64")

    df[target_column] = df[target_column].astype("int8")

    # Normalize model feature dtypes.
    df = normalize_feature_types(
        df,
        categorical_columns=feature_config["categorical_features"],
        protected_columns=[
            id_column,
            time_column,
            target_column,
        ],
    )

    # Sort transactions chronologically.
    df = df.sort_values(
        by=[time_column, id_column],
        kind="stable",
    ).reset_index(drop=True)

    full_features = get_ieee_feature_columns(
        df,
        feature_set="FullFeatures",
    )

    nocardid_features = get_ieee_feature_columns(
        df,
        feature_set="NoCardID",
    )

    if len(full_features) != config["expected_full_features"]:
        raise ValueError(f"Unexpected FullFeatures count: " f"{len(full_features)}")

    if len(nocardid_features) != config["expected_nocardid_features"]:
        raise ValueError(f"Unexpected NoCardID count: " f"{len(nocardid_features)}")

    # Verify that cleaning preserves missingness.
    if not df.isna().sum().equals(original_missing):
        raise ValueError("Missing values changed during preprocessing.")

    if int(df[target_column].sum()) != original_fraud_count:
        raise ValueError("Fraud count changed during preprocessing.")

    metadata = [
        id_column,
        time_column,
        target_column,
    ]

    full_df = df[metadata + full_features]

    nocardid_df = df[metadata + nocardid_features]

    return full_df, nocardid_df


# ==================================================
# 6. BAF sentinel audit
# ==================================================


def audit_baf_sentinels(df, variant_name):
    """Report candidate sentinel values without replacing them."""

    config = load_feature_config()["baf"]

    audit_config = config["sentinel_audit"]

    records = []

    for column in audit_config["candidate_columns"]:

        if column not in df.columns:
            continue

        for value in audit_config["candidate_values"]:

            count = int(df[column].eq(value).sum())

            records.append(
                {
                    "variant": variant_name,
                    "column": column,
                    "candidate_sentinel": value,
                    "count": count,
                    "percentage": (count / len(df) * 100),
                }
            )

    return pd.DataFrame(records)


# ==================================================
# 7. BAF preprocessing
# ==================================================


def preprocess_baf(df, variant_name):
    """Prepare one BAF variant independently."""

    config = load_config()["datasets"]["baf"]

    feature_config = load_feature_config()["baf"]

    target_column = config["target_column"]
    time_column = config["time_column"]

    if len(df) != config["expected_rows_per_variant"]:
        raise ValueError(f"{variant_name}: unexpected row count.")

    validate_binary_target(df, target_column)

    validate_integer_column(
        df,
        time_column,
        min_value=0,
        max_value=7,
    )

    actual_months = set(df[time_column].unique())

    expected_months = set(config["expected_months"])

    if actual_months != expected_months:
        raise ValueError(f"{variant_name}: unexpected months " f"{actual_months}")

    validate_no_infinite_values(df)

    df = df.copy()

    original_fraud_count = int(df[target_column].sum())

    # Generate a stable, variant-local record ID
    # before sorting.
    if "record_id" in df.columns:
        raise ValueError("record_id already exists.")

    df.insert(
        0,
        "record_id",
        np.arange(len(df), dtype=np.int64),
    )

    df[target_column] = df[target_column].astype("int8")

    df[time_column] = df[time_column].astype("int8")

    # Keep sentinel values unchanged unless an
    # explicit replacement rule is configured.
    for column, values in feature_config["sentinel_replacement"].items():

        validate_required_columns(df, [column])

        df[column] = df[column].replace(
            values,
            np.nan,
        )

    df = normalize_feature_types(
        df,
        categorical_columns=feature_config["categorical_features"],
        protected_columns=[
            "record_id",
            target_column,
            time_column,
        ],
    )

    # Month is the finest available time resolution.
    # record_id is only a deterministic tie-breaker.
    df = df.sort_values(
        by=[time_column, "record_id"],
        kind="stable",
    ).reset_index(drop=True)

    validate_unique_id(df, "record_id")

    if int(df[target_column].sum()) != original_fraud_count:
        raise ValueError(f"{variant_name}: fraud count changed.")

    return df


def get_baf_feature_columns(df):
    """Return BAF model features excluding metadata."""

    config = load_feature_config()["baf"]

    excluded = set(config["metadata"])

    excluded.add(config["target_column"])

    return [column for column in df.columns if column not in excluded]


# ==================================================
# 8. Parquet output
# ==================================================


def save_parquet(df, relative_path):
    """Save a processed dataset as Parquet."""

    config = load_config()["preprocessing"]

    path = resolve_path(relative_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        path,
        engine=config["parquet_engine"],
        compression=config["parquet_compression"],
        index=False,
    )

    print(f"Saved: {path}")

    return path


# ==================================================
# 9. Feature schema
# ==================================================


def build_feature_schema(
    df,
    feature_columns,
    categorical_columns,
):
    """Build an explicit model feature schema."""

    categorical_columns = set(categorical_columns)

    return {
        "num_features": len(feature_columns),
        "features": list(feature_columns),
        "categorical_features": [
            column for column in feature_columns if column in categorical_columns
        ],
        "numerical_features": [
            column for column in feature_columns if column not in categorical_columns
        ],
        "dtypes": {column: str(df[column].dtype) for column in feature_columns},
    }


def save_feature_schema(schema, relative_path):
    """Save feature schema as JSON."""

    path = resolve_path(relative_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as file:

        json.dump(
            schema,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved: {path}")


# ==================================================
# 10. Training-time categorical transformation
# ==================================================


def fit_category_vocab(train_df, categorical_columns):
    """
    Learn categorical vocabularies from an eligible
    training window only.
    """

    validate_required_columns(
        train_df,
        categorical_columns,
    )

    vocabulary = {}

    for column in categorical_columns:

        values = train_df[column].astype("string").dropna().unique().tolist()

        vocabulary[column] = sorted(values)

    return vocabulary


def apply_category_vocab(df, vocabulary):
    """
    Apply training-time categorical vocabularies.

    Unseen categories become missing values
    in the model input only.
    """

    df = df.copy()

    validate_required_columns(
        df,
        list(vocabulary),
    )

    for column, categories in vocabulary.items():

        df[column] = pd.Categorical(
            df[column].astype("string"),
            categories=categories,
        )

    return df

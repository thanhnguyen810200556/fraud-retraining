import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    validate_binary_target,
    validate_unique_id,
    validate_integer_column,
    validate_no_infinite_values,
    get_ieee_feature_columns,
    fit_category_vocab,
    apply_category_vocab,
)

# ==================================================
# 1. Target validation
# ==================================================


def test_valid_binary_target():

    df = pd.DataFrame({"isFraud": [0, 1, 0, 1]})

    validate_binary_target(df, "isFraud")


def test_invalid_binary_target():

    df = pd.DataFrame({"isFraud": [0, 1, 2]})

    with pytest.raises(ValueError):

        validate_binary_target(df, "isFraud")


def test_missing_target():

    df = pd.DataFrame({"isFraud": [0, 1, np.nan]})

    with pytest.raises(ValueError):

        validate_binary_target(df, "isFraud")


# ==================================================
# 2. ID validation
# ==================================================


def test_unique_ids():

    df = pd.DataFrame({"TransactionID": [1, 2, 3]})

    validate_unique_id(df, "TransactionID")


def test_duplicate_ids():

    df = pd.DataFrame({"TransactionID": [1, 1, 2]})

    with pytest.raises(ValueError):

        validate_unique_id(df, "TransactionID")


def test_missing_ids():

    df = pd.DataFrame({"TransactionID": [1, np.nan, 3]})

    with pytest.raises(ValueError):

        validate_unique_id(df, "TransactionID")


# ==================================================
# 3. Temporal validation
# ==================================================


def test_valid_integer_time():

    df = pd.DataFrame({"TransactionDT": [86400, 86401]})

    validate_integer_column(
        df,
        "TransactionDT",
        min_value=0,
    )


def test_negative_time():

    df = pd.DataFrame({"TransactionDT": [-1, 86400]})

    with pytest.raises(ValueError):

        validate_integer_column(
            df,
            "TransactionDT",
            min_value=0,
        )


def test_fractional_time():

    df = pd.DataFrame({"TransactionDT": [1.5, 2]})

    with pytest.raises(ValueError):

        validate_integer_column(
            df,
            "TransactionDT",
        )


# ==================================================
# 4. Infinite and missing values
# ==================================================


def test_missing_values_are_allowed():

    df = pd.DataFrame({"V1": [1.0, np.nan, 2.0]})

    validate_no_infinite_values(df)


def test_infinite_values_are_rejected():

    df = pd.DataFrame({"V1": [1.0, np.inf]})

    with pytest.raises(ValueError):

        validate_no_infinite_values(df)


# ==================================================
# 5. Feature exclusion
# ==================================================


def test_full_feature_exclusion():

    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [86400],
            "isFraud": [0],
            "TransactionAmt": [100.0],
            "card1": [1234],
            "day": [1],
            "block_id": [0],
            "phase": ["warmup"],
            "label_available_time": [90000],
        }
    )

    features = get_ieee_feature_columns(
        df,
        feature_set="FullFeatures",
    )

    assert features == [
        "TransactionAmt",
        "card1",
    ]


def test_nocardid_feature_exclusion():

    df = pd.DataFrame(
        {
            "TransactionID": [1],
            "TransactionDT": [86400],
            "isFraud": [0],
            "TransactionAmt": [100.0],
            "card1": [1234],
            "addr1": [100],
            "P_emaildomain": ["gmail.com"],
        }
    )

    features = get_ieee_feature_columns(
        df,
        feature_set="NoCardID",
    )

    assert features == [
        "TransactionAmt",
    ]


# ==================================================
# 6. Training-time categorical vocabulary
# ==================================================


def test_category_vocab_uses_training_data_only():

    train_df = pd.DataFrame({"ProductCD": ["A", "B", "A"]})

    vocabulary = fit_category_vocab(
        train_df,
        ["ProductCD"],
    )

    assert vocabulary["ProductCD"] == [
        "A",
        "B",
    ]


def test_unseen_category_becomes_missing():

    train_df = pd.DataFrame({"ProductCD": ["A", "B"]})

    future_df = pd.DataFrame({"ProductCD": ["A", "C"]})

    vocabulary = fit_category_vocab(
        train_df,
        ["ProductCD"],
    )

    transformed = apply_category_vocab(
        future_df,
        vocabulary,
    )

    assert transformed.loc[0, "ProductCD"] == "A"

    assert pd.isna(transformed.loc[1, "ProductCD"])

    # Original data must remain unchanged.
    assert future_df.loc[1, "ProductCD"] == "C"


def test_missing_category_is_preserved():

    train_df = pd.DataFrame({"ProductCD": ["A", "B"]})

    future_df = pd.DataFrame({"ProductCD": ["A", None]})

    vocabulary = fit_category_vocab(
        train_df,
        ["ProductCD"],
    )

    transformed = apply_category_vocab(
        future_df,
        vocabulary,
    )

    assert pd.isna(transformed.loc[1, "ProductCD"])

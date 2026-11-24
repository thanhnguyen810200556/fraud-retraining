import numpy as np
import pandas as pd
import pytest

from src.temporal_split import (
    calculate_relative_day,
    validate_phase_boundaries,
    build_ieee_block_manifest,
    build_ieee_temporal_index,
    build_baf_block_manifest,
    build_baf_temporal_index,
    attach_temporal_index,
    validate_temporal_index,
)

ORIGIN = 86400

DAY = 86400


# ==================================================
# 1. Relative day calculation
# ==================================================


def test_first_transaction_is_day_one():

    timestamps = pd.Series([ORIGIN])

    days = calculate_relative_day(
        timestamps,
        ORIGIN,
    )

    assert days.tolist() == [1]


def test_day_boundary():

    timestamps = pd.Series(
        [
            ORIGIN,
            ORIGIN + DAY - 1,
            ORIGIN + DAY,
        ]
    )

    days = calculate_relative_day(
        timestamps,
        ORIGIN,
    )

    assert days.tolist() == [1, 1, 2]


def test_timestamp_before_origin():

    timestamps = pd.Series([ORIGIN - 1])

    with pytest.raises(ValueError):

        calculate_relative_day(
            timestamps,
            ORIGIN,
        )


def test_missing_timestamp():

    timestamps = pd.Series([ORIGIN, np.nan])

    with pytest.raises(ValueError):

        calculate_relative_day(
            timestamps,
            ORIGIN,
        )


# ==================================================
# 2. Phase configuration
# ==================================================


def test_valid_phase_boundaries():

    validate_phase_boundaries(
        1,
        30,
        31,
        86,
        87,
        177,
        7,
    )


def test_phase_gap_rejected():

    with pytest.raises(ValueError):

        validate_phase_boundaries(
            1,
            30,
            32,
            87,
            88,
            178,
            7,
        )


def test_phase_overlap_rejected():

    with pytest.raises(ValueError):

        validate_phase_boundaries(
            1,
            30,
            30,
            85,
            86,
            176,
            7,
        )


def test_incomplete_block_rejected():

    with pytest.raises(ValueError):

        validate_phase_boundaries(
            1,
            30,
            31,
            85,
            86,
            176,
            7,
        )


# ==================================================
# 3. IEEE-CIS manifest
# ==================================================


def test_ieee_block_count():

    manifest = build_ieee_block_manifest(ORIGIN)

    assert len(manifest) == 21

    assert manifest["phase"].eq("validation").sum() == 8

    assert manifest["phase"].eq("test").sum() == 13


def test_ieee_first_block():

    manifest = build_ieee_block_manifest(ORIGIN)

    first = manifest.iloc[0]

    assert first["block_id"] == 1

    assert first["start_day"] == 31

    assert first["end_day"] == 37

    assert first["start_time"] == (ORIGIN + 30 * DAY)

    assert first["end_time_exclusive"] == (ORIGIN + 37 * DAY)


def test_ieee_validation_test_boundary():

    manifest = build_ieee_block_manifest(ORIGIN)

    validation_last = manifest.iloc[7]

    test_first = manifest.iloc[8]

    assert validation_last["end_day"] == 86

    assert test_first["start_day"] == 87

    assert validation_last["end_time_exclusive"] == test_first["start_time"]


def test_ieee_no_block_gap_or_overlap():

    manifest = build_ieee_block_manifest(ORIGIN)

    assert np.array_equal(
        manifest["end_time_exclusive"].iloc[:-1].to_numpy(),
        manifest["start_time"].iloc[1:].to_numpy(),
    )


# ==================================================
# 4. IEEE-CIS temporal assignment
# ==================================================


def test_ieee_phase_assignment():

    days = np.array([1, 30, 31, 37, 38, 86, 87, 177, 178, 182])

    df = pd.DataFrame(
        {
            "TransactionID": range(len(days)),
            "TransactionDT": (ORIGIN + (days - 1) * DAY),
        }
    )

    index, _, _ = build_ieee_temporal_index(df)

    assert index["phase"].tolist() == [
        "warmup",
        "warmup",
        "validation",
        "validation",
        "validation",
        "validation",
        "test",
        "test",
        "tail",
        "tail",
    ]

    assert index["block_id"].tolist() == [0, 0, 1, 1, 2, 8, 9, 21, -1, -1]


def test_ieee_exact_timestamp_boundary():

    boundary = ORIGIN + 30 * DAY

    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [
                ORIGIN,
                boundary - 1,
                boundary,
                ORIGIN + 181 * DAY,
            ],
        }
    )

    index, _, _ = build_ieee_temporal_index(df)

    assert index.loc[1, "phase"] == "warmup"

    assert index.loc[2, "phase"] == "validation"

    assert index.loc[2, "block_id"] == 1


# ==================================================
# 5. BAF temporal assignment
# ==================================================


def test_baf_manifest():

    manifest = build_baf_block_manifest()

    assert len(manifest) == 6

    assert manifest["start_month"].tolist() == [2, 3, 4, 5, 6, 7]


def test_baf_phase_assignment():

    df = pd.DataFrame(
        {
            "record_id": range(8),
            "month": range(8),
        }
    )

    index, manifest = build_baf_temporal_index(df)

    assert index["phase"].tolist() == [
        "warmup",
        "warmup",
        "validation",
        "validation",
        "test",
        "test",
        "test",
        "test",
    ]

    assert index["block_id"].tolist() == [0, 0, 1, 2, 3, 4, 5, 6]

    assert validate_temporal_index(
        index,
        manifest,
        id_column="record_id",
        time_column="month",
        period_column="month",
        expected_rows=8,
    )


def test_invalid_baf_month():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 8],
        }
    )

    with pytest.raises(ValueError):

        build_baf_temporal_index(df)


def test_incomplete_baf_months():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
        }
    )

    with pytest.raises(ValueError):

        build_baf_temporal_index(df)


# ==================================================
# 6. Temporal index join
# ==================================================


def test_valid_temporal_join():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
            "fraud_bool": [0, 1],
        }
    )

    index = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
            "phase": ["warmup", "warmup"],
            "block_id": [0, 0],
        }
    )

    result = attach_temporal_index(
        df,
        index,
        "record_id",
        "month",
    )

    assert len(result) == 2

    assert result["fraud_bool"].tolist() == [0, 1]

    assert "phase" in result.columns


def test_duplicate_temporal_id():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
        }
    )

    index = pd.DataFrame(
        {
            "record_id": [1, 1],
            "month": [0, 1],
        }
    )

    with pytest.raises(ValueError):

        attach_temporal_index(
            df,
            index,
            "record_id",
            "month",
        )


def test_unmatched_temporal_id():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
        }
    )

    index = pd.DataFrame(
        {
            "record_id": [1, 3],
            "month": [0, 1],
        }
    )

    with pytest.raises(ValueError):

        attach_temporal_index(
            df,
            index,
            "record_id",
            "month",
        )


def test_temporal_timestamp_mismatch():

    df = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 1],
        }
    )

    index = pd.DataFrame(
        {
            "record_id": [1, 2],
            "month": [0, 2],
        }
    )

    with pytest.raises(ValueError):

        attach_temporal_index(
            df,
            index,
            "record_id",
            "month",
        )


def test_temporal_index_has_no_target():

    df = pd.DataFrame(
        {
            "record_id": range(8),
            "month": range(8),
            "fraud_bool": [0] * 8,
        }
    )

    index, _ = build_baf_temporal_index(df)

    assert "fraud_bool" not in index.columns


def test_invalid_block_assignment_rejected():

    df = pd.DataFrame(
        {
            "record_id": range(8),
            "month": range(8),
        }
    )

    index, manifest = build_baf_temporal_index(df)

    index.loc[
        index["month"].eq(3),
        "block_id",
    ] = 1

    with pytest.raises(ValueError):

        validate_temporal_index(
            index,
            manifest,
            id_column="record_id",
            time_column="month",
            period_column="month",
            expected_rows=8,
        )

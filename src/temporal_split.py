import numpy as np
import pandas as pd

from src.data_loader import (
    load_config,
    validate_required_columns,
)

from src.preprocessing import (
    validate_unique_id,
    validate_integer_column,
    validate_binary_target,
)

# ==================================================
# 1. Temporal utilities
# ==================================================


def calculate_relative_day(
    timestamps,
    time_origin,
    seconds_per_day=86400,
):
    """Convert timestamps to one-based relative days."""

    if seconds_per_day <= 0:
        raise ValueError("Invalid seconds_per_day.")

    if timestamps.isna().any():
        raise ValueError("Missing timestamps.")

    if (timestamps < time_origin).any():
        raise ValueError("Timestamp precedes time origin.")

    return ((timestamps - time_origin) // seconds_per_day + 1).astype("int32")


def validate_phase_boundaries(
    warmup_start,
    warmup_end,
    validation_start,
    validation_end,
    test_start,
    test_end,
    block_size,
):
    """Ensure phases are contiguous and complete."""

    if block_size <= 0:
        raise ValueError("Invalid block size.")

    if not (
        warmup_start
        <= warmup_end
        < validation_start
        <= validation_end
        < test_start
        <= test_end
    ):
        raise ValueError("Invalid phase ordering.")

    if validation_start != warmup_end + 1:
        raise ValueError("Warmup/validation gap.")

    if test_start != validation_end + 1:
        raise ValueError("Validation/test gap.")

    validation_length = validation_end - validation_start + 1

    test_length = test_end - test_start + 1

    if validation_length % block_size != 0:
        raise ValueError("Incomplete validation block.")

    if test_length % block_size != 0:
        raise ValueError("Incomplete test block.")


# ==================================================
# 2. IEEE-CIS block manifest
# ==================================================


def build_ieee_block_manifest(
    time_origin,
    split_config=None,
    seconds_per_day=86400,
):
    """Build evaluation blocks without using labels."""

    cfg = load_config()["temporal_split"] if split_config is None else split_config

    block_size = cfg["block_size_days"]

    validate_phase_boundaries(
        cfg["warmup_start_day"],
        cfg["warmup_end_day"],
        cfg["validation_start_day"],
        cfg["validation_end_day"],
        cfg["test_start_day"],
        cfg["test_end_day"],
        block_size,
    )

    records = []

    block_id = 1

    for phase in ["validation", "test"]:

        phase_start = cfg[f"{phase}_start_day"]

        phase_end = cfg[f"{phase}_end_day"]

        for start_day in range(
            phase_start,
            phase_end + 1,
            block_size,
        ):

            end_day = start_day + block_size - 1

            start_time = time_origin + (start_day - 1) * seconds_per_day

            end_time = time_origin + end_day * seconds_per_day

            records.append(
                {
                    "block_id": block_id,
                    "phase": phase,
                    "start_day": start_day,
                    "end_day": end_day,
                    "start_time": start_time,
                    "end_time_exclusive": end_time,
                    "duration_days": block_size,
                }
            )

            block_id += 1

    manifest = pd.DataFrame(records)

    for phase in ["validation", "test"]:

        actual_count = int(manifest["phase"].eq(phase).sum())

        expected_count = cfg[f"expected_{phase}_blocks"]

        if actual_count != expected_count:
            raise ValueError(f"Unexpected {phase} block count.")

    return manifest


# ==================================================
# 3. IEEE-CIS temporal index
# ==================================================


def build_ieee_temporal_index(
    df,
    time_origin=None,
):
    """Create transaction-level temporal metadata."""

    config = load_config()

    dataset_cfg = config["datasets"]["ieee_cis"]

    split_cfg = config["temporal_split"]

    id_col = dataset_cfg["id_column"]

    time_col = dataset_cfg["time_column"]

    seconds_per_day = dataset_cfg["seconds_per_day"]

    validate_required_columns(
        df,
        [id_col, time_col],
    )

    validate_unique_id(df, id_col)

    validate_integer_column(
        df,
        time_col,
        min_value=0,
    )

    index = df[[id_col, time_col]].copy()

    index = index.sort_values([time_col, id_col]).reset_index(drop=True)

    if time_origin is None:

        time_origin = int(index[time_col].min())

    if time_origin != split_cfg["expected_time_origin"]:
        raise ValueError(f"Unexpected time origin: {time_origin}")

    index["day"] = calculate_relative_day(
        index[time_col],
        time_origin,
        seconds_per_day,
    )

    if index["day"].min() != split_cfg["expected_first_day"]:
        raise ValueError("Unexpected first day.")

    if index["day"].max() != split_cfg["expected_last_day"]:
        raise ValueError("Unexpected last day.")

    manifest = build_ieee_block_manifest(
        time_origin,
        split_config=split_cfg,
        seconds_per_day=seconds_per_day,
    )

    index["phase"] = "tail"

    index["block_id"] = split_cfg["tail_block_id"]

    warmup_mask = index["day"].between(
        split_cfg["warmup_start_day"],
        split_cfg["warmup_end_day"],
    )

    index.loc[warmup_mask, "phase"] = "warmup"

    index.loc[warmup_mask, "block_id"] = split_cfg["warmup_block_id"]

    for block in manifest.itertuples(index=False):

        mask = (index[time_col] >= block.start_time) & (
            index[time_col] < block.end_time_exclusive
        )

        index.loc[mask, "phase"] = block.phase

        index.loc[mask, "block_id"] = block.block_id

    index["phase"] = index["phase"].astype("string")

    index["block_id"] = index["block_id"].astype("int16")

    return index, manifest, time_origin


# ==================================================
# 4. BAF block manifest
# ==================================================


def build_baf_block_manifest(
    split_config=None,
):
    """Build monthly evaluation blocks."""

    cfg = load_config()["baf_temporal_split"] if split_config is None else split_config

    block_size = cfg["block_size_months"]

    validate_phase_boundaries(
        cfg["warmup_start_month"],
        cfg["warmup_end_month"],
        cfg["validation_start_month"],
        cfg["validation_end_month"],
        cfg["test_start_month"],
        cfg["test_end_month"],
        block_size,
    )

    records = []

    block_id = 1

    for phase in ["validation", "test"]:

        start = cfg[f"{phase}_start_month"]

        end = cfg[f"{phase}_end_month"]

        for month in range(
            start,
            end + 1,
            block_size,
        ):

            records.append(
                {
                    "block_id": block_id,
                    "phase": phase,
                    "start_month": month,
                    "end_month": (month + block_size - 1),
                    "start_time": month,
                    "end_time_exclusive": (month + block_size),
                    "duration_months": block_size,
                }
            )

            block_id += 1

    manifest = pd.DataFrame(records)

    for phase in ["validation", "test"]:

        actual = int(manifest["phase"].eq(phase).sum())

        expected = cfg[f"expected_{phase}_blocks"]

        if actual != expected:
            raise ValueError(f"Unexpected BAF {phase} blocks.")

    return manifest


# ==================================================
# 5. BAF temporal index
# ==================================================


def build_baf_temporal_index(df):
    """Create a monthly temporal index."""

    cfg = load_config()["baf_temporal_split"]

    validate_required_columns(
        df,
        ["record_id", "month"],
    )

    validate_unique_id(df, "record_id")

    validate_integer_column(
        df,
        "month",
        min_value=cfg["warmup_start_month"],
        max_value=cfg["test_end_month"],
    )

    expected_months = set(
        range(
            cfg["warmup_start_month"],
            cfg["test_end_month"] + 1,
        )
    )

    if set(df["month"].unique()) != expected_months:
        raise ValueError("BAF monthly coverage is incomplete.")

    index = df[["record_id", "month"]].copy()

    index = index.sort_values(["month", "record_id"]).reset_index(drop=True)

    index["phase"] = "unassigned"

    index["block_id"] = -1

    warmup_mask = index["month"].between(
        cfg["warmup_start_month"],
        cfg["warmup_end_month"],
    )

    index.loc[warmup_mask, "phase"] = "warmup"

    index.loc[warmup_mask, "block_id"] = cfg["warmup_block_id"]

    manifest = build_baf_block_manifest()

    for block in manifest.itertuples(index=False):

        mask = index["month"].between(
            block.start_month,
            block.end_month,
        )

        index.loc[mask, "phase"] = block.phase

        index.loc[mask, "block_id"] = block.block_id

    if index["phase"].eq("unassigned").any():
        raise ValueError("Unassigned BAF records.")

    index["phase"] = index["phase"].astype("string")

    index["block_id"] = index["block_id"].astype("int16")

    return index, manifest


# ==================================================
# 6. Join temporal metadata
# ==================================================


def attach_temporal_index(
    df,
    temporal_index,
    id_column,
    time_column,
):
    """Join temporal metadata using a validated key."""

    validate_unique_id(df, id_column)

    validate_unique_id(
        temporal_index,
        id_column,
    )

    if len(df) != len(temporal_index):
        raise ValueError("Temporal index row count mismatch.")

    extra_columns = [
        column
        for column in temporal_index.columns
        if column not in [id_column, time_column]
    ]

    overlap = set(extra_columns).intersection(df.columns)

    if overlap:
        raise ValueError(f"Metadata already exists: {overlap}")

    joined = df.merge(
        temporal_index,
        on=id_column,
        how="left",
        validate="one_to_one",
        suffixes=("", "_index"),
        indicator=True,
    )

    if not joined["_merge"].eq("both").all():
        raise ValueError("Unmatched temporal index IDs.")

    index_time_col = f"{time_column}_index"

    if not joined[time_column].eq(joined[index_time_col]).all():
        raise ValueError("Temporal timestamp mismatch.")

    joined = joined.drop(
        columns=[
            "_merge",
            index_time_col,
        ]
    )

    return joined


# ==================================================
# 7. Phase statistics
# ==================================================


def build_phase_summary(
    df,
    target_column,
    temporal_column,
):
    """Generate offline phase statistics."""

    validate_binary_target(
        df,
        target_column,
    )

    summary = (
        df.groupby("phase")
        .agg(
            start_period=(temporal_column, "min"),
            end_period=(temporal_column, "max"),
            observed_periods=(temporal_column, "nunique"),
            num_transactions=(target_column, "size"),
            num_fraud=(target_column, "sum"),
        )
        .reset_index()
    )

    summary["num_legitimate"] = summary["num_transactions"] - summary["num_fraud"]

    summary["fraud_rate_pct"] = summary["num_fraud"] / summary["num_transactions"] * 100

    phase_order = {
        "warmup": 0,
        "validation": 1,
        "test": 2,
        "tail": 3,
    }

    summary["_order"] = summary["phase"].map(phase_order)

    summary = (
        summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    )

    return summary


# ==================================================
# 8. Evaluation block statistics
# ==================================================


def build_block_statistics(
    df,
    target_column,
    temporal_column,
):
    """Generate offline evaluation statistics."""

    validate_binary_target(
        df,
        target_column,
    )

    evaluation = df.loc[df["phase"].isin(["validation", "test"])]

    summary = (
        evaluation.groupby(["block_id", "phase"])
        .agg(
            start_period=(temporal_column, "min"),
            end_period=(temporal_column, "max"),
            observed_periods=(temporal_column, "nunique"),
            num_transactions=(target_column, "size"),
            num_fraud=(target_column, "sum"),
        )
        .reset_index()
        .sort_values("block_id")
        .reset_index(drop=True)
    )

    summary["num_legitimate"] = summary["num_transactions"] - summary["num_fraud"]

    summary["fraud_rate_pct"] = summary["num_fraud"] / summary["num_transactions"] * 100

    warning_threshold = load_config()["profiling"]["min_fraud_per_block_warning"]

    summary["low_fraud_warning"] = summary["num_fraud"] < warning_threshold

    return summary


# ==================================================
# 9. Temporal validation
# ==================================================


def validate_temporal_index(
    temporal_index,
    manifest,
    id_column,
    time_column,
    period_column,
    expected_rows,
):
    """Validate temporal coverage and boundaries."""

    validate_unique_id(
        temporal_index,
        id_column,
    )

    if len(temporal_index) != expected_rows:
        raise ValueError("Unexpected temporal index size.")

    if temporal_index.isna().any().any():
        raise ValueError("Missing temporal metadata.")

    if not temporal_index[time_column].is_monotonic_increasing:
        raise ValueError("Temporal index is not sorted.")

    if not temporal_index["phase"].isin(["warmup", "validation", "test", "tail"]).all():
        raise ValueError("Unknown phase.")

    manifest = manifest.sort_values("block_id").reset_index(drop=True)

    expected_ids = list(range(1, len(manifest) + 1))

    if manifest["block_id"].tolist() != expected_ids:
        raise ValueError("Invalid manifest block IDs.")

    if not (manifest["start_time"] < manifest["end_time_exclusive"]).all():
        raise ValueError("Invalid block duration.")

    if not np.array_equal(
        manifest["end_time_exclusive"].iloc[:-1].to_numpy(),
        manifest["start_time"].iloc[1:].to_numpy(),
    ):
        raise ValueError("Block gap or overlap.")

    evaluation = temporal_index.loc[
        temporal_index["phase"].isin(["validation", "test"])
    ]

    if set(evaluation["block_id"]) != set(manifest["block_id"]):
        raise ValueError("Evaluation block coverage mismatch.")

    expected_eval_mask = (
        temporal_index[time_column] >= manifest["start_time"].iloc[0]
    ) & (temporal_index[time_column] < manifest["end_time_exclusive"].iloc[-1])

    actual_eval_mask = temporal_index["phase"].isin(["validation", "test"])

    if not expected_eval_mask.equals(actual_eval_mask):
        raise ValueError("Incorrect evaluation phase assignment.")

    warmup = temporal_index.loc[temporal_index["phase"].eq("warmup")]

    tail = temporal_index.loc[temporal_index["phase"].eq("tail")]

    if not warmup["block_id"].eq(0).all():
        raise ValueError("Invalid warmup block ID.")

    if not tail["block_id"].eq(-1).all():
        raise ValueError("Invalid tail block ID.")

    if not (warmup[time_column] < manifest["start_time"].iloc[0]).all():
        raise ValueError("Invalid warmup time.")

    if not (tail[time_column] >= manifest["end_time_exclusive"].iloc[-1]).all():
        raise ValueError("Invalid tail time.")

    period_name = "day" if period_column == "day" else "month"

    for block in manifest.to_dict(orient="records"):

        rows = evaluation.loc[evaluation["block_id"].eq(block["block_id"])]

        if not rows["phase"].eq(block["phase"]).all():
            raise ValueError("Block phase mismatch.")

        if not (
            (rows[time_column] >= block["start_time"])
            & (rows[time_column] < block["end_time_exclusive"])
        ).all():
            raise ValueError("Block timestamp mismatch.")

        expected_periods = set(
            range(
                block[f"start_{period_name}"],
                block[f"end_{period_name}"] + 1,
            )
        )

        if set(rows[period_column].unique()) != expected_periods:
            raise ValueError("Incomplete block coverage.")

    return True

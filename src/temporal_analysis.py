import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400


def add_temporal_metadata(df, time_column="TransactionDT", time_origin=None):
    """Add relative day and calendar-week metadata."""

    result = df.copy()

    if time_origin is None:
        time_origin = result[time_column].min()

    elapsed = result[time_column] - time_origin

    if elapsed.isna().any() or (elapsed < 0).any():
        raise ValueError("Invalid transaction timestamps.")

    result["day"] = (elapsed // SECONDS_PER_DAY + 1).astype("int32")

    result["week_id"] = ((result["day"] - 1) // 7 + 1).astype("int32")

    return result


def assign_experiment_blocks(df, split_config):
    """Assign warmup, validation and test periods."""

    result = df.copy()

    day = result["day"]

    warmup = day.between(
        split_config["warmup_start_day"], split_config["warmup_end_day"]
    )

    validation = day.between(
        split_config["validation_start_day"], split_config["validation_end_day"]
    )

    test = day.between(split_config["test_start_day"], split_config["test_end_day"])

    result["phase"] = np.select(
        [warmup, validation, test],
        ["warmup", "validation", "test"],
        default="remainder",
    )

    result["block_id"] = pd.Series(pd.NA, index=result.index, dtype="Int32")

    evaluation_mask = validation | test

    result.loc[evaluation_mask, "block_id"] = (
        (day[evaluation_mask] - split_config["validation_start_day"])
        // split_config["block_size_days"]
        + 1
    ).astype("int32")

    return result


def temporal_summary(df, group_column, target_column="isFraud"):
    """Calculate transaction counts and fraud rates."""

    summary = (
        df.groupby(group_column)
        .agg(num_transactions=(target_column, "size"), num_fraud=(target_column, "sum"))
        .reset_index()
    )

    summary["num_legitimate"] = summary["num_transactions"] - summary["num_fraud"]

    summary["fraud_rate_pct"] = summary["num_fraud"] / summary["num_transactions"] * 100

    return summary

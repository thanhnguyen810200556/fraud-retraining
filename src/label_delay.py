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
# 1. Validation utilities
# ==================================================


def validate_nonnegative_integer(value, name):
    """Validate a non-negative integer parameter."""

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer.")

    return int(value)


def validate_positive_integer(value, name):
    """Validate a positive integer parameter."""

    value = validate_nonnegative_integer(value, name)

    if value == 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


# ==================================================
# 2. IEEE-CIS label schedule
# ==================================================


def build_ieee_label_schedule(
    temporal_index,
    delay_days,
    warmup_strategy=None,
):
    """Generate a deterministic IEEE-CIS label schedule."""

    config = load_config()

    dataset_cfg = config["datasets"]["ieee_cis"]
    split_cfg = config["temporal_split"]
    sim_cfg = config["simulation"]

    delay_days = validate_nonnegative_integer(
        delay_days,
        "delay_days",
    )

    seconds_per_day = validate_positive_integer(
        dataset_cfg["seconds_per_day"],
        "seconds_per_day",
    )

    if warmup_strategy is None:
        warmup_strategy = sim_cfg["warmup_strategy"]

    if warmup_strategy not in [
        "pre_labeled",
        "strict_delay",
    ]:
        raise ValueError("Unknown warmup strategy.")

    required_columns = [
        "TransactionID",
        "TransactionDT",
        "day",
        "phase",
        "block_id",
    ]

    validate_required_columns(
        temporal_index,
        required_columns,
    )

    validate_unique_id(
        temporal_index,
        "TransactionID",
    )

    validate_integer_column(
        temporal_index,
        "TransactionDT",
        min_value=0,
    )

    schedule = temporal_index[required_columns].copy()

    if schedule.isna().any().any():
        raise ValueError("Missing temporal metadata.")

    if (
        not schedule["phase"]
        .isin(
            [
                "warmup",
                "validation",
                "test",
                "tail",
            ]
        )
        .all()
    ):
        raise ValueError("Unknown temporal phase.")

    schedule["TransactionID"] = schedule["TransactionID"].astype("int64")

    schedule["TransactionDT"] = schedule["TransactionDT"].astype("int64")

    schedule["day"] = schedule["day"].astype("int32")

    schedule["block_id"] = schedule["block_id"].astype("int16")

    schedule["phase"] = schedule["phase"].astype("string")

    schedule["label_delay_days"] = np.int32(delay_days)

    schedule["label_available_time"] = (
        schedule["TransactionDT"] + delay_days * seconds_per_day
    ).astype("int64")

    schedule["is_pre_labeled"] = False

    if warmup_strategy == "pre_labeled":

        deployment_time = (
            split_cfg["expected_time_origin"]
            + (split_cfg["validation_start_day"] - 1) * seconds_per_day
        )

        warmup_mask = schedule["phase"].eq("warmup")

        if not (
            schedule.loc[
                warmup_mask,
                "TransactionDT",
            ]
            < deployment_time
        ).all():
            raise ValueError("Warmup transaction after deployment.")

        schedule.loc[
            warmup_mask,
            "label_available_time",
        ] = np.minimum(
            schedule.loc[
                warmup_mask,
                "label_available_time",
            ],
            deployment_time - 1,
        )

        schedule.loc[
            warmup_mask,
            "is_pre_labeled",
        ] = True

    if not sim_cfg["include_tail_in_schedule"]:
        schedule = schedule.loc[~schedule["phase"].eq("tail")].copy()

    schedule = schedule.sort_values(
        [
            "TransactionDT",
            "TransactionID",
        ]
    ).reset_index(drop=True)

    return schedule


# ==================================================
# 3. BAF label schedule
# ==================================================


def build_baf_label_schedule(
    temporal_index,
    delay_months,
    warmup_strategy=None,
):
    """Generate a monthly BAF label schedule."""

    config = load_config()

    split_cfg = config["baf_temporal_split"]
    sim_cfg = config["baf_simulation"]

    delay_months = validate_nonnegative_integer(
        delay_months,
        "delay_months",
    )

    if warmup_strategy is None:
        warmup_strategy = sim_cfg["warmup_strategy"]

    if warmup_strategy not in [
        "pre_labeled",
        "strict_delay",
    ]:
        raise ValueError("Unknown warmup strategy.")

    required_columns = [
        "record_id",
        "month",
        "phase",
        "block_id",
    ]

    validate_required_columns(
        temporal_index,
        required_columns,
    )

    validate_unique_id(
        temporal_index,
        "record_id",
    )

    validate_integer_column(
        temporal_index,
        "month",
        min_value=0,
        max_value=split_cfg["test_end_month"],
    )

    schedule = temporal_index[required_columns].copy()

    if schedule.isna().any().any():
        raise ValueError("Missing temporal metadata.")

    if (
        not schedule["phase"]
        .isin(
            [
                "warmup",
                "validation",
                "test",
            ]
        )
        .all()
    ):
        raise ValueError("Unknown BAF phase.")

    schedule["month"] = schedule["month"].astype("int32")

    schedule["phase"] = schedule["phase"].astype("string")

    schedule["block_id"] = schedule["block_id"].astype("int16")

    schedule["label_delay_months"] = np.int32(delay_months)

    schedule["label_available_month"] = (schedule["month"] + 1 + delay_months).astype(
        "int32"
    )

    schedule["is_pre_labeled"] = False

    if warmup_strategy == "pre_labeled":

        deployment_month = split_cfg["validation_start_month"]

        warmup_mask = schedule["phase"].eq("warmup")

        if not (
            schedule.loc[
                warmup_mask,
                "month",
            ]
            < deployment_month
        ).all():
            raise ValueError("Invalid BAF warmup month.")

        schedule.loc[
            warmup_mask,
            "label_available_month",
        ] = np.minimum(
            schedule.loc[
                warmup_mask,
                "label_available_month",
            ],
            deployment_month,
        )

        schedule.loc[
            warmup_mask,
            "is_pre_labeled",
        ] = True

    schedule = schedule.sort_values(
        [
            "month",
            "record_id",
        ]
    ).reset_index(drop=True)

    return schedule


# ==================================================
# 4. Label availability
# ==================================================


def get_ieee_available_mask(
    schedule,
    decision_time,
):
    """Find past transactions with available labels."""

    decision_time = validate_nonnegative_integer(
        decision_time,
        "decision_time",
    )

    validate_required_columns(
        schedule,
        [
            "TransactionDT",
            "label_available_time",
        ],
    )

    return (schedule["TransactionDT"] < decision_time) & (
        schedule["label_available_time"] < decision_time
    )


def get_baf_available_mask(
    schedule,
    decision_month,
):
    """Find past BAF records with available labels."""

    decision_month = validate_nonnegative_integer(
        decision_month,
        "decision_month",
    )

    validate_required_columns(
        schedule,
        [
            "month",
            "label_available_month",
        ],
    )

    return (schedule["month"] < decision_month) & (
        schedule["label_available_month"] <= decision_month
    )


# ==================================================
# 5. Training window selection
# ==================================================


def get_ieee_training_mask(
    schedule,
    decision_time,
    window_days,
):
    """Select available labels in a recent event window."""

    config = load_config()

    sim_cfg = config["simulation"]

    seconds_per_day = validate_positive_integer(
        config["datasets"]["ieee_cis"]["seconds_per_day"],
        "seconds_per_day",
    )

    decision_time = validate_nonnegative_integer(
        decision_time,
        "decision_time",
    )

    window_days = validate_positive_integer(
        window_days,
        "window_days",
    )

    window_start = decision_time - window_days * seconds_per_day

    if not sim_cfg["allow_partial_history"]:

        first_event = int(schedule["TransactionDT"].min())

        if window_start < first_event:
            return pd.Series(
                False,
                index=schedule.index,
            )

    available_mask = get_ieee_available_mask(
        schedule,
        decision_time,
    )

    window_mask = schedule["TransactionDT"] >= window_start

    training_mask = available_mask & window_mask

    if not sim_cfg["include_tail_in_training"]:
        training_mask &= ~schedule["phase"].eq("tail")

    return training_mask


def get_baf_training_mask(
    schedule,
    decision_month,
    window_months,
):
    """Select available BAF labels in a monthly window."""

    config = load_config()

    sim_cfg = config["baf_simulation"]

    decision_month = validate_nonnegative_integer(
        decision_month,
        "decision_month",
    )

    window_months = validate_positive_integer(
        window_months,
        "window_months",
    )

    window_start = decision_month - window_months

    if not sim_cfg["allow_partial_history"]:

        first_month = int(schedule["month"].min())

        if window_start < first_month:
            return pd.Series(
                False,
                index=schedule.index,
            )

    available_mask = get_baf_available_mask(
        schedule,
        decision_month,
    )

    window_mask = schedule["month"] >= window_start

    return available_mask & window_mask


# ==================================================
# 6. Training data retrieval
# ==================================================


def select_training_rows(
    df,
    schedule,
    training_mask,
    id_column,
    target_column,
    time_column,
):
    """Retrieve eligible labeled rows using validated IDs."""

    validate_required_columns(
        df,
        [
            id_column,
            target_column,
            time_column,
        ],
    )

    validate_required_columns(
        schedule,
        [
            id_column,
            time_column,
        ],
    )

    validate_unique_id(
        df,
        id_column,
    )

    validate_unique_id(
        schedule,
        id_column,
    )

    if not isinstance(
        training_mask,
        pd.Series,
    ):
        raise TypeError("Training mask must be a pandas Series.")

    if not training_mask.index.equals(schedule.index):
        raise ValueError("Training mask index mismatch.")

    if not pd.api.types.is_bool_dtype(training_mask):
        raise ValueError("Training mask must have boolean dtype.")

    if training_mask.isna().any():
        raise ValueError("Training mask contains missing values.")

    selected = schedule.loc[
        training_mask,
        [
            id_column,
            time_column,
        ],
    ].copy()

    selected = selected.rename(
        columns={
            time_column: "_schedule_time",
        }
    )

    if "_schedule_time" in df.columns:
        raise ValueError("Reserved column already exists.")

    train_df = selected.merge(
        df,
        on=id_column,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not train_df["_merge"].eq("both").all():
        raise ValueError("Unmatched training IDs.")

    if not train_df[time_column].eq(train_df["_schedule_time"]).all():
        raise ValueError("Training timestamp mismatch.")

    train_df = train_df.drop(
        columns=[
            "_merge",
            "_schedule_time",
        ]
    )

    train_df = train_df[df.columns].copy()

    if len(train_df) != int(training_mask.sum()):
        raise ValueError("Training row count mismatch.")

    if not train_df.empty:
        validate_binary_target(
            train_df,
            target_column,
        )

    train_df = train_df.sort_values(
        [
            time_column,
            id_column,
        ]
    ).reset_index(drop=True)

    return train_df


# ==================================================
# 7. Initial model training data
# ==================================================


def get_initial_training_rows(
    df,
    schedule,
    deployment_time,
    dataset_type,
):
    """Retrieve eligible warmup rows for initial M0."""

    validate_required_columns(
        schedule,
        ["phase"],
    )

    warmup_mask = schedule["phase"].eq("warmup")

    if dataset_type == "ieee_cis":

        available_mask = get_ieee_available_mask(
            schedule,
            deployment_time,
        )

        return select_training_rows(
            df=df,
            schedule=schedule,
            training_mask=(warmup_mask & available_mask),
            id_column="TransactionID",
            target_column="isFraud",
            time_column="TransactionDT",
        )

    if dataset_type == "baf":

        available_mask = get_baf_available_mask(
            schedule,
            deployment_time,
        )

        return select_training_rows(
            df=df,
            schedule=schedule,
            training_mask=(warmup_mask & available_mask),
            id_column="record_id",
            target_column="fraud_bool",
            time_column="month",
        )

    raise ValueError(f"Unknown dataset type: {dataset_type}")


# ==================================================
# 8. Training data statistics
# ==================================================


def summarize_training_data(
    train_df,
    target_column,
):
    """Summarize a labeled training subset."""

    validate_required_columns(
        train_df,
        [target_column],
    )

    num_transactions = len(train_df)

    if num_transactions == 0:

        return {
            "num_transactions": 0,
            "num_fraud": 0,
            "num_legitimate": 0,
            "fraud_rate_pct": np.nan,
            "has_both_classes": False,
            "status": "empty",
        }

    validate_binary_target(
        train_df,
        target_column,
    )

    num_fraud = int(train_df[target_column].sum())

    num_legitimate = num_transactions - num_fraud

    has_both_classes = num_fraud > 0 and num_legitimate > 0

    return {
        "num_transactions": num_transactions,
        "num_fraud": num_fraud,
        "num_legitimate": num_legitimate,
        "fraud_rate_pct": (num_fraud / num_transactions * 100),
        "has_both_classes": has_both_classes,
        "status": ("available" if has_both_classes else "single_class"),
    }


# ==================================================
# 9. IEEE-CIS schedule validation
# ==================================================


def validate_ieee_label_schedule(
    schedule,
    temporal_index,
    delay_days,
    warmup_strategy=None,
):
    """Validate deterministic IEEE-CIS label metadata."""

    config = load_config()

    if warmup_strategy is None:
        warmup_strategy = config["simulation"]["warmup_strategy"]

    expected = build_ieee_label_schedule(
        temporal_index=temporal_index,
        delay_days=delay_days,
        warmup_strategy=warmup_strategy,
    )

    validate_unique_id(
        schedule,
        "TransactionID",
    )

    if schedule.isna().any().any():
        raise ValueError("Missing label schedule metadata.")

    if not (schedule["label_available_time"] >= schedule["TransactionDT"]).all():
        raise ValueError("Label available before transaction.")

    if not schedule["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Schedule is not sorted.")

    forbidden_columns = {
        "isFraud",
        "fraud_bool",
    }

    if forbidden_columns.intersection(schedule.columns):
        raise ValueError("Ground truth found in schedule.")

    pd.testing.assert_frame_equal(
        schedule.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=True,
    )

    return True


# ==================================================
# 10. BAF schedule validation
# ==================================================


def validate_baf_label_schedule(
    schedule,
    temporal_index,
    delay_months,
    warmup_strategy=None,
):
    """Validate deterministic BAF label metadata."""

    config = load_config()

    if warmup_strategy is None:
        warmup_strategy = config["baf_simulation"]["warmup_strategy"]

    expected = build_baf_label_schedule(
        temporal_index=temporal_index,
        delay_months=delay_months,
        warmup_strategy=warmup_strategy,
    )

    validate_unique_id(
        schedule,
        "record_id",
    )

    if schedule.isna().any().any():
        raise ValueError("Missing BAF label metadata.")

    if not (schedule["label_available_month"] > schedule["month"]).all():
        raise ValueError("BAF label available before month ends.")

    if not schedule["month"].is_monotonic_increasing:
        raise ValueError("BAF schedule is not sorted.")

    forbidden_columns = {
        "fraud_bool",
        "isFraud",
    }

    if forbidden_columns.intersection(schedule.columns):
        raise ValueError("Ground truth found in BAF schedule.")

    pd.testing.assert_frame_equal(
        schedule.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=True,
    )

    return True

from pathlib import Path

import pandas as pd
import yaml

# ==================================================
# 1. Project paths and configuration
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"


def load_config():
    """Load dataset configuration from YAML."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(relative_path):
    """Convert a project-relative path to an absolute path."""

    return PROJECT_ROOT / relative_path


# ==================================================
# 2. Common utilities
# ==================================================


def read_csv(relative_path, **kwargs):
    """Read a CSV file from the project directory."""

    path = resolve_path(relative_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    kwargs.setdefault("low_memory", False)

    return pd.read_csv(path, **kwargs)


def validate_required_columns(df, required_columns):
    """Check whether required columns exist."""

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


# ==================================================
# 3. Dataset manifest
# ==================================================


def load_manifest():
    """Load dataset provenance information."""

    config = load_config()

    return read_csv(config["paths"]["manifest"], dtype={"sha256": "string"})


# ==================================================
# 4. IEEE-CIS dataset
# ==================================================


def load_ieee_transactions(**kwargs):
    """Load labeled IEEE-CIS transactions."""

    config = load_config()["datasets"]["ieee_cis"]

    df = read_csv(config["transaction_path"], **kwargs)

    validate_required_columns(
        df, [config["id_column"], config["time_column"], config["target_column"]]
    )

    return df


def load_ieee_identity(**kwargs):
    """Load IEEE-CIS identity information."""

    config = load_config()["datasets"]["ieee_cis"]

    df = read_csv(config["identity_path"], **kwargs)

    validate_required_columns(df, [config["id_column"]])

    return df


def load_ieee_merged():
    """Merge transactions and identity using a left join."""

    config = load_config()["datasets"]["ieee_cis"]

    id_column = config["id_column"]

    transactions = load_ieee_transactions()
    identity = load_ieee_identity()

    merged = transactions.merge(
        identity, on=id_column, how="left", validate="one_to_one"
    )

    if len(merged) != len(transactions):
        raise ValueError("Transaction count changed after merging.")

    return merged


# ==================================================
# 5. Bank Account Fraud dataset
# ==================================================


def load_baf(variant="Base.csv", **kwargs):
    """Load one BAF dataset variant."""

    config = load_config()["datasets"]["baf"]

    if variant not in config["variants"]:
        raise ValueError(f"Unknown BAF variant: {variant}")

    relative_path = Path(config["directory"]) / variant

    df = read_csv(relative_path, **kwargs)

    validate_required_columns(df, [config["target_column"], config["time_column"]])

    return df


# ==================================================
# 6. Quick verification
# ==================================================

if __name__ == "__main__":

    manifest = load_manifest()

    print("Dataset manifest:")
    print(manifest.to_string(index=False))

    transactions = load_ieee_transactions()

    print("\nIEEE-CIS Transactions:")
    print("Shape:", transactions.shape)

    del transactions

    identity = load_ieee_identity()

    print("\nIEEE-CIS Identity:")
    print("Shape:", identity.shape)

    del identity

    baf_base = load_baf("Base.csv")

    print("\nBAF Base:")
    print("Shape:", baf_base.shape)

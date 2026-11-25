import hashlib
import time
from dataclasses import dataclass

import numpy as np
import yaml
from lightgbm import LGBMClassifier

from src.data_loader import PROJECT_ROOT
from src.preprocessing import apply_category_vocab, fit_category_vocab

POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "policies.yaml"

# ==================================================
# 1. Configuration
# ==================================================


def load_policy_config():
    """Load model and retraining policy configuration."""

    with open(POLICY_CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==================================================
# 2. Training set signature
# ==================================================


def training_signature(ids):
    """Hash the set of training IDs (order independent)."""

    values = np.sort(np.asarray(ids, dtype="int64"))

    return hashlib.sha256(values.tobytes()).hexdigest()


# ==================================================
# 3. Model training and scoring
# ==================================================


@dataclass
class TrainedModel:
    model: LGBMClassifier
    vocabulary: dict
    features: list
    signature: str
    num_train: int
    num_fraud: int
    fit_seconds: float


def train_model(
    train_df,
    features,
    categorical_columns,
    params,
    id_column="TransactionID",
    target_column="isFraud",
):
    """Fit LightGBM on one eligible training window."""

    if train_df.empty:
        raise ValueError("Empty training data.")

    if train_df[target_column].nunique() < 2:
        raise ValueError("Training data needs both classes.")

    # Category vocabularies are learned from the training window only.
    vocabulary = fit_category_vocab(train_df, categorical_columns)

    X = apply_category_vocab(train_df[features], vocabulary)

    y = train_df[target_column].astype("int8")

    start = time.perf_counter()

    model = LGBMClassifier(**params).fit(X, y)

    fit_seconds = time.perf_counter() - start

    return TrainedModel(
        model=model,
        vocabulary=vocabulary,
        features=list(features),
        signature=training_signature(train_df[id_column]),
        num_train=int(len(train_df)),
        num_fraud=int(y.sum()),
        fit_seconds=float(fit_seconds),
    )


def predict_scores(trained, df):
    """Return fraud probabilities for a data frame."""

    X = apply_category_vocab(df[trained.features], trained.vocabulary)

    return trained.model.predict_proba(X)[:, 1]

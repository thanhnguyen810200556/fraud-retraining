from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.evaluation import compute_block_metrics
from src.label_delay import (
    get_ieee_training_mask,
    get_initial_training_rows,
    select_training_rows,
)
from src.modeling import predict_scores, train_model, training_signature

# ==================================================
# 1. Model bank
# ==================================================
# A retrained model depends only on the decision time,
# the label delay d and the window W, not on the policy
# that triggered it. Each candidate model is trained once
# and shared by NoRetrain, Periodic and Random-k.


@dataclass
class ModelBank:
    label_delay_days: int
    window_days: int
    block_manifest: pd.DataFrame
    eval_blocks: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    block_model: dict = field(default_factory=dict)
    block_status: dict = field(default_factory=dict)
    _metrics: dict = field(default_factory=dict)

    @property
    def block_ids(self):
        return sorted(int(b) for b in self.block_manifest["block_id"])

    @property
    def candidate_blocks(self):
        """Blocks where retraining uses labels not seen by M0."""

        return [b for b in self.block_ids if self.block_model.get(b) is not None]

    def set_eval_blocks(self, eval_blocks):
        """Attach evaluation data (block_id -> features + target)."""

        if sorted(eval_blocks) != self.block_ids:
            raise ValueError("Evaluation blocks do not match the manifest.")

        self.eval_blocks = eval_blocks

    def metrics(self, model_key, block_id):
        """Cached metrics of one model on one block."""

        cache_key = (model_key, block_id)

        if cache_key not in self._metrics:

            block_df = self.eval_blocks[block_id]

            scores = predict_scores(self.models[model_key], block_df)

            self._metrics[cache_key] = compute_block_metrics(
                block_df["isFraud"].to_numpy(),
                scores,
            )

        return self._metrics[cache_key]


def build_model_bank(
    df,
    schedule,
    block_manifest,
    features,
    categorical_columns,
    window_days,
    params,
    verbose=True,
):
    """Train M0 and one candidate model per block decision time."""

    delays = schedule["label_delay_days"].unique()

    if len(delays) != 1:
        raise ValueError("Schedule must contain one label delay.")

    manifest = block_manifest.sort_values("block_id").set_index("block_id")

    bank = ModelBank(
        label_delay_days=int(delays[0]),
        window_days=int(window_days),
        block_manifest=manifest.reset_index(),
    )

    deployment_time = int(manifest["start_time"].iloc[0])

    m0_rows = get_initial_training_rows(df, schedule, deployment_time, "ieee_cis")

    bank.models["M0"] = train_model(m0_rows, features, categorical_columns, params)

    signature_to_key = {bank.models["M0"].signature: "M0"}

    if verbose:
        m0 = bank.models["M0"]
        print(
            f"d={bank.label_delay_days:2d} | M0: {m0.num_train:,} rows, "
            f"{m0.num_fraud:,} fraud, {m0.fit_seconds:.1f}s"
        )

    for block_id, row in manifest.iterrows():

        decision_time = int(row["start_time"])

        mask = get_ieee_training_mask(schedule, decision_time, window_days)

        signature = training_signature(schedule.loc[mask, "TransactionID"])

        if signature in signature_to_key:
            # Same eligible labels as an existing model: nothing new to learn.
            key = signature_to_key[signature]
            bank.block_model[block_id] = None if key == "M0" else key
            bank.block_status[block_id] = "no_new_labels"
            continue

        train_df = select_training_rows(
            df,
            schedule,
            mask,
            id_column="TransactionID",
            target_column="isFraud",
            time_column="TransactionDT",
        )

        if train_df.empty or train_df["isFraud"].nunique() < 2:
            bank.block_model[block_id] = None
            bank.block_status[block_id] = "insufficient_labels"
            continue

        # Leakage guard: only past events whose labels arrived before the decision.
        labels_at = schedule.loc[mask, "label_available_time"]

        if train_df["TransactionDT"].max() >= decision_time or labels_at.max() >= decision_time:
            raise ValueError(f"Future information in block {block_id}.")

        key = f"B{block_id:02d}"

        bank.models[key] = train_model(train_df, features, categorical_columns, params)

        signature_to_key[signature] = key

        bank.block_model[block_id] = key
        bank.block_status[block_id] = "new_labels"

        if verbose:
            m = bank.models[key]
            print(
                f"d={bank.label_delay_days:2d} | block {block_id:2d}: "
                f"{m.num_train:,} rows, {m.num_fraud:,} fraud, {m.fit_seconds:.1f}s"
            )

    return bank


# ==================================================
# 2. Retraining policies
# ==================================================


class NoRetrainPolicy:
    """Keep the initial model M0 for every block."""

    name = "NoRetrain"

    def retrain_blocks(self, bank):
        return []


class PeriodicPolicy:
    """Retrain at the start of every `interval_blocks` blocks."""

    name = "Periodic"

    def __init__(self, interval_blocks=1):

        if interval_blocks < 1:
            raise ValueError("interval_blocks must be >= 1.")

        self.interval_blocks = int(interval_blocks)
        self.name = f"Periodic-{7 * self.interval_blocks}d"

    def retrain_blocks(self, bank):
        return bank.block_ids[:: self.interval_blocks]


class RandomKPolicy:
    """Retrain at k random blocks drawn from blocks with new labels."""

    name = "Random-k"

    def __init__(self, budget, seed):

        if budget < 1:
            raise ValueError("budget must be >= 1.")

        self.budget = int(budget)
        self.seed = int(seed)

    def retrain_blocks(self, bank):

        candidates = bank.candidate_blocks

        if self.budget > len(candidates):
            raise ValueError(
                f"Budget {self.budget} exceeds {len(candidates)} candidate blocks."
            )

        rng = np.random.default_rng(self.seed)

        chosen = rng.choice(candidates, size=self.budget, replace=False)

        return sorted(int(b) for b in chosen)


# ==================================================
# 3. Policy simulation
# ==================================================


def run_policy(policy, bank, run_info=None):
    """Simulate one policy over the evaluation blocks."""

    run_info = run_info or {}

    planned = set(policy.retrain_blocks(bank))

    manifest = bank.block_manifest.set_index("block_id")

    current = "M0"

    rows = []

    for block_id in bank.block_ids:

        decision = "keep"

        if block_id in planned:

            key = bank.block_model.get(block_id)

            if key is not None and key != current:
                current = key
                decision = "retrain"
            else:
                decision = "skip_no_new_labels"

        model = bank.models[current]

        rows.append(
            {
                "policy": policy.name,
                **run_info,
                "label_delay_days": bank.label_delay_days,
                "window_days": bank.window_days,
                "block_id": block_id,
                "phase": manifest.loc[block_id, "phase"],
                "start_day": int(manifest.loc[block_id, "start_day"]),
                "decision": decision,
                "retrained": decision == "retrain",
                "model_key": current,
                "train_rows": model.num_train if decision == "retrain" else 0,
                "fit_seconds": model.fit_seconds if decision == "retrain" else 0.0,
                **bank.metrics(current, block_id),
            }
        )

    return pd.DataFrame(rows)


def summarize_cost(block_results, group_columns):
    """Retraining cost and mean performance per policy run."""

    return (
        block_results.groupby(group_columns, dropna=False)
        .agg(
            num_retrains=("retrained", "sum"),
            total_fit_seconds=("fit_seconds", "sum"),
            total_train_rows=("train_rows", "sum"),
            mean_aucpr=("aucpr", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_recall_at_fpr_1pct=("recall_at_fpr_1pct", "mean"),
            num_blocks=("block_id", "nunique"),
        )
        .reset_index()
    )

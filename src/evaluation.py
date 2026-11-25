import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# ==================================================
# 1. Block-level metrics
# ==================================================


def recall_at_fpr(y_true, scores, max_fpr=0.01):
    """Highest recall reachable with FPR <= max_fpr."""

    fpr, tpr, _ = roc_curve(y_true, scores)

    valid = tpr[fpr <= max_fpr]

    return float(valid.max()) if len(valid) > 0 else 0.0


def compute_block_metrics(y_true, scores):
    """Compute evaluation metrics for one block."""

    y_true = np.asarray(y_true)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Evaluation block needs both classes.")

    return {
        "num_transactions": int(len(y_true)),
        "num_fraud": int(y_true.sum()),
        "fraud_rate_pct": float(y_true.mean() * 100),
        "aucpr": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, scores, 0.01),
    }


# ==================================================
# 2. Summaries
# ==================================================

METRIC_COLUMNS = [
    "aucpr",
    "roc_auc",
    "recall_at_fpr_1pct",
]


def summarize_by_phase(block_results, group_columns):
    """Average block metrics per phase and over all blocks."""

    frames = []

    for phase, subset in [("all", block_results)] + list(
        block_results.groupby("phase")
    ):

        summary = (
            subset.groupby(group_columns, dropna=False)[METRIC_COLUMNS]
            .mean()
            .reset_index()
        )

        summary.insert(0, "phase", phase)

        summary["num_blocks"] = subset["block_id"].nunique()

        frames.append(summary)

    return pd.concat(frames, ignore_index=True)

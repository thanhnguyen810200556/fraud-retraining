import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


def psi_from_counts(reference_counts, current_counts, epsilon=1e-6):
    """Calculate PSI from aligned frequency counts."""

    ref = np.asarray(reference_counts, dtype=float)
    cur = np.asarray(current_counts, dtype=float)

    if ref.sum() == 0 or cur.sum() == 0:
        return np.nan

    ref = ref / ref.sum()
    cur = cur / cur.sum()

    ref = np.clip(ref, epsilon, None)
    cur = np.clip(cur, epsilon, None)

    ref = ref / ref.sum()
    cur = cur / cur.sum()

    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numerical_psi(reference, current, n_bins=10, epsilon=1e-6):
    """Calculate numerical PSI using reference quantiles."""

    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, n_bins + 1)

    edges = np.unique(np.quantile(ref, quantiles))

    if len(edges) == 1:
        value = edges[0]

        ref_counts = [np.sum(ref < value), np.sum(ref == value), np.sum(ref > value)]

        cur_counts = [np.sum(cur < value), np.sum(cur == value), np.sum(cur > value)]

    else:
        edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])

        ref_counts = np.histogram(ref, bins=edges)[0]

        cur_counts = np.histogram(cur, bins=edges)[0]

    # Missing values form a separate bin.
    ref_counts = np.append(ref_counts, reference.isna().sum())

    cur_counts = np.append(cur_counts, current.isna().sum())

    return psi_from_counts(ref_counts, cur_counts, epsilon)


def categorical_psi(reference, current, epsilon=1e-6):
    """Calculate categorical PSI using aligned categories."""

    ref = reference.value_counts(dropna=True)
    cur = current.value_counts(dropna=True)

    categories = ref.index.union(cur.index)

    ref_counts = ref.reindex(categories, fill_value=0).to_numpy()

    cur_counts = cur.reindex(categories, fill_value=0).to_numpy()

    ref_counts = np.append(ref_counts, reference.isna().sum())

    cur_counts = np.append(cur_counts, current.isna().sum())

    return psi_from_counts(ref_counts, cur_counts, epsilon)


def ks_statistic(reference, current, max_samples=10000, random_seed=42):
    """Calculate KS on non-missing numerical observations."""

    ref = reference.dropna()
    cur = current.dropna()

    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    if len(ref) > max_samples:
        ref = ref.sample(max_samples, random_state=random_seed)

    if len(cur) > max_samples:
        cur = cur.sample(max_samples, random_state=random_seed)

    result = ks_2samp(ref, cur, method="asymp")

    return float(result.statistic)


def feature_drift_report(
    reference_df, current_df, features, categorical_features, config, random_seed=42
):
    """Calculate drift statistics for selected features."""

    records = []

    for feature in features:

        ref = reference_df[feature]
        cur = current_df[feature]

        ref_missing = ref.isna().mean()
        cur_missing = cur.isna().mean()

        ref_count = int(ref.notna().sum())
        cur_count = int(cur.notna().sum())

        sufficient_samples = (
            min(ref_count, cur_count) >= config["min_non_missing_samples"]
        )

        is_categorical = (
            feature in categorical_features or not pd.api.types.is_numeric_dtype(ref)
        )

        psi = np.nan
        ks = np.nan

        if sufficient_samples:

            if is_categorical:

                psi = categorical_psi(ref, cur, epsilon=config["psi_epsilon"])

            else:

                psi = numerical_psi(
                    ref, cur, n_bins=config["psi_bins"], epsilon=config["psi_epsilon"]
                )

                ks = ks_statistic(
                    ref,
                    cur,
                    max_samples=config["ks_max_samples"],
                    random_seed=random_seed,
                )

        records.append(
            {
                "feature": feature,
                "feature_type": ("categorical" if is_categorical else "numerical"),
                "psi": psi,
                "ks_statistic": ks,
                "reference_non_missing": ref_count,
                "current_non_missing": cur_count,
                "reference_missing_rate": ref_missing,
                "current_missing_rate": cur_missing,
                "missing_rate_change": (cur_missing - ref_missing),
                "missing_rate_drift": abs(cur_missing - ref_missing),
                "sufficient_samples": sufficient_samples,
            }
        )

    return pd.DataFrame(records)

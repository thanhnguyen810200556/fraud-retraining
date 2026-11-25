import numpy as np
import pandas as pd
import pytest

from src.evaluation import compute_block_metrics, recall_at_fpr
from src.policies import (
    NoRetrainPolicy,
    PeriodicPolicy,
    RandomKPolicy,
    build_model_bank,
    run_policy,
)

DAY = 86400

PARAMS = dict(n_estimators=5, num_leaves=4, min_child_samples=5, random_state=0, verbose=-1)


# ==================================================
# Synthetic stream: 4 days warmup + 4 one-day blocks
# ==================================================


def make_stream(delay_days):

    rng = np.random.default_rng(0)

    rows = []

    tid = 1

    for day in range(1, 9):
        for _ in range(60):
            rows.append((tid, day * DAY + tid, day))
            tid += 1

    df = pd.DataFrame(rows, columns=["TransactionID", "TransactionDT", "day"])

    df["x1"] = rng.normal(size=len(df))
    df["ProductCD"] = pd.array(rng.choice(["W", "C"], size=len(df)), dtype="string")
    df["isFraud"] = (df["x1"] + rng.normal(scale=0.5, size=len(df)) > 1).astype("int8")

    df["phase"] = np.where(df["day"] <= 4, "warmup", "validation")
    df["block_id"] = np.where(df["day"] <= 4, 0, df["day"] - 4).astype("int16")

    schedule = df[["TransactionID", "TransactionDT", "day", "phase", "block_id"]].copy()
    schedule["phase"] = schedule["phase"].astype("string")
    schedule["label_delay_days"] = np.int32(delay_days)
    schedule["label_available_time"] = schedule["TransactionDT"] + delay_days * DAY

    deployment = 5 * DAY
    warm = schedule["phase"].eq("warmup")
    schedule.loc[warm, "label_available_time"] = np.minimum(
        schedule.loc[warm, "label_available_time"], deployment - 1
    )

    manifest = pd.DataFrame(
        {
            "block_id": [1, 2, 3, 4],
            "phase": "validation",
            "start_day": [5, 6, 7, 8],
            "start_time": [d * DAY for d in [5, 6, 7, 8]],
        }
    )

    data = df[["TransactionID", "TransactionDT", "x1", "ProductCD", "isFraud"]]

    eval_blocks = {
        b: data[df["block_id"].eq(b)].reset_index(drop=True) for b in [1, 2, 3, 4]
    }

    return data, schedule, manifest, eval_blocks


def make_bank(delay_days):

    data, schedule, manifest, eval_blocks = make_stream(delay_days)

    bank = build_model_bank(
        data,
        schedule,
        manifest,
        features=["x1", "ProductCD"],
        categorical_columns=["ProductCD"],
        window_days=30,
        params=PARAMS,
        verbose=False,
    )

    bank.set_eval_blocks(eval_blocks)

    return bank


# ==================================================
# Metrics
# ==================================================


def test_recall_at_fpr_perfect_ranking():

    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    assert recall_at_fpr(y, s, 0.01) == 1.0


def test_block_metrics_need_both_classes():

    with pytest.raises(ValueError):
        compute_block_metrics(np.zeros(10), np.random.rand(10))


# ==================================================
# Model bank
# ==================================================


def test_bank_uses_only_available_labels():

    data, schedule, manifest, _ = make_stream(delay_days=1)

    bank = make_bank(delay_days=1)

    # Block 1 (day 5) with d=1: day-4 labels arrive at day 5, so nothing new yet.
    assert bank.block_status[1] == "no_new_labels"

    # Every retrained model saw only labels available before its decision time.
    for block_id, key in bank.block_model.items():
        if key is None:
            continue
        decision = int(manifest.set_index("block_id").loc[block_id, "start_time"])
        assert bank.models[key].num_train == int(
            (
                (schedule["TransactionDT"] < decision)
                & (schedule["label_available_time"] < decision)
                & (schedule["TransactionDT"] >= decision - 30 * DAY)
            ).sum()
        )


def test_zero_delay_has_new_labels_after_first_block():

    bank = make_bank(delay_days=0)

    # At the first decision only warmup labels exist, identical to M0.
    assert bank.block_status[1] == "no_new_labels"
    assert bank.candidate_blocks == [2, 3, 4]


# ==================================================
# Policies
# ==================================================


def test_noretrain_never_retrains():

    result = run_policy(NoRetrainPolicy(), make_bank(0))

    assert not result["retrained"].any()
    assert result["model_key"].eq("M0").all()


def test_periodic_retrains_when_labels_are_new():

    bank = make_bank(1)

    result = run_policy(PeriodicPolicy(1), bank)

    assert result.loc[result.block_id.eq(1), "decision"].item() == "skip_no_new_labels"
    assert result["retrained"].sum() == len(bank.candidate_blocks)


def test_periodic_interval():

    bank = make_bank(0)

    assert PeriodicPolicy(2).retrain_blocks(bank) == [1, 3]
    assert PeriodicPolicy(2).name == "Periodic-14d"


def test_eval_blocks_must_match_manifest():

    bank = make_bank(0)

    with pytest.raises(ValueError):
        bank.set_eval_blocks({1: bank.eval_blocks[1]})


def test_random_k_respects_budget_and_seed():

    bank = make_bank(0)

    a = RandomKPolicy(2, seed=7).retrain_blocks(bank)
    b = RandomKPolicy(2, seed=7).retrain_blocks(bank)

    assert a == b
    assert len(set(a)) == 2
    assert set(a) <= set(bank.candidate_blocks)
    assert run_policy(RandomKPolicy(2, seed=7), bank)["retrained"].sum() == 2


def test_random_k_budget_too_large():

    with pytest.raises(ValueError):
        RandomKPolicy(10, seed=0).retrain_blocks(make_bank(0))

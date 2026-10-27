import numpy as np
import pandas as pd
import pytest

from src.common.schemas import (
    EXCLUDED_COLUMNS,
    FEATURE_SETS,
    POST_TRANSACTION_COLUMNS,
    TARGET,
    model_features,
)
from src.training.features import (
    add_engineered_features,
    build_preprocessor,
    normalise_columns,
    to_model_input,
    validate_training_frame,
)


def test_engineered_features_match_formulas():
    row = pd.DataFrame(
        [{"step": 25, "amount": 100.0, "oldbalanceOrg": 300.0, "newbalanceOrig": 150.0,
          "oldbalanceDest": 10.0, "newbalanceDest": 110.0}]
    )
    out = add_engineered_features(row).iloc[0]
    assert out["hour_of_day"] == 1
    assert out["origin_balance_error"] == 50.0
    assert out["dest_balance_change"] == 100.0
    assert out["amount_to_balance_ratio"] == pytest.approx(100 / 300)


def test_balance_ratio_is_missing_when_origin_balance_is_zero():
    row = pd.DataFrame(
        [{"step": 1, "amount": 50.0, "oldbalanceOrg": 0.0, "newbalanceOrig": 0.0,
          "oldbalanceDest": 0.0, "newbalanceDest": 50.0}]
    )
    assert np.isnan(add_engineered_features(row).iloc[0]["amount_to_balance_ratio"])


def test_engineering_does_not_mutate_input(paysim_df):
    before = paysim_df.copy()
    add_engineered_features(paysim_df)
    pd.testing.assert_frame_equal(paysim_df, before)


@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS))
def test_model_input_has_fixed_order_and_no_leaky_columns(paysim_df, feature_set):
    X = to_model_input(paysim_df, feature_set)
    assert X.columns.tolist() == model_features(feature_set)
    for leaky in [TARGET, "step", *EXCLUDED_COLUMNS]:
        assert leaky not in X.columns


def test_pre_transaction_set_uses_no_post_transaction_values(paysim_df):
    X = to_model_input(paysim_df, "pre_transaction")
    assert not set(POST_TRANSACTION_COLUMNS) & set(X.columns)

    # Changing every post-transaction balance must not change the model input.
    shuffled = paysim_df.assign(newbalanceOrig=1.0, newbalanceDest=2.0)
    pd.testing.assert_frame_equal(X, to_model_input(shuffled, "pre_transaction"))


def test_unknown_feature_set_is_rejected(paysim_df):
    with pytest.raises(ValueError, match="Unknown feature set"):
        to_model_input(paysim_df, "everything")


def test_preprocessor_layout_is_stable_and_handles_unseen_type(paysim_df):
    pre = build_preprocessor().fit(to_model_input(paysim_df))
    names = pre.get_feature_names_out().tolist()
    assert names[-5:] == ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]

    unseen = to_model_input(paysim_df.head(1).assign(type="CRYPTO"))
    encoded = pre.transform(unseen)
    assert encoded[0, -5:].sum() == 0


def test_column_aliases_are_normalised(paysim_df):
    renamed = paysim_df.rename(columns={"oldbalanceOrg": "oldbalanceOrig"})
    assert "oldbalanceOrg" in normalise_columns(renamed).columns


def test_validation_rejects_missing_columns(paysim_df):
    with pytest.raises(ValueError, match="missing required columns"):
        validate_training_frame(paysim_df.drop(columns=["amount"]))


def test_validation_rejects_non_binary_target(paysim_df):
    with pytest.raises(ValueError, match="must be 0/1"):
        validate_training_frame(paysim_df.assign(isFraud=2))


def test_validation_rejects_single_class(paysim_df):
    with pytest.raises(ValueError, match="single class"):
        validate_training_frame(paysim_df.assign(isFraud=0))

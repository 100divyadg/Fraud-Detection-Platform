"""Feature engineering shared by offline training and streaming inference.

Everything here must be computable from a single transaction event at the
moment it arrives, so the streaming path can call the same functions.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.common.schemas import (
    CATEGORICAL_FEATURES,
    COLUMN_ALIASES,
    DEFAULT_FEATURE_SET,
    REQUIRED_COLUMNS,
    TARGET,
    TRANSACTION_TYPES,
    model_features,
    numeric_features,
)


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column-name variants to the names used in this project."""
    renames = {old: new for old, new in COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    return df.rename(columns=renames)


def validate_training_frame(df: pd.DataFrame) -> None:
    """Fail loudly if the dataset does not match the expected contract."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Found: {df.columns.tolist()}")

    labels = set(df[TARGET].dropna().unique())
    if not labels <= {0, 1}:
        raise ValueError(f"{TARGET} must be 0/1, found values {sorted(labels)}")
    if len(labels) < 2:
        raise ValueError(f"{TARGET} contains a single class; cannot train a classifier")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Required columns contain nulls: {null_counts[null_counts > 0].to_dict()}")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the derived model inputs added.

    hour_of_day replaces raw `step`: step only grows over the simulation, so a
    chronological test set would contain values the model never saw. The hour
    keeps the daily cycle without that extrapolation problem.

    amount_to_balance_ratio uses only pre-transaction values. It is NaN when the
    origin balance is zero (the ratio is undefined); XGBoost treats NaN as missing.

    Post-transaction features are added only when the post-transaction balances
    are present, so live events (which omit them) still yield pre_transaction input.
    """
    out = df.copy()
    out["hour_of_day"] = out["step"] % 24
    if "newbalanceOrig" in out.columns:
        out["origin_balance_error"] = out["oldbalanceOrg"] - out["amount"] - out["newbalanceOrig"]
    if "newbalanceDest" in out.columns:
        out["dest_balance_change"] = out["newbalanceDest"] - out["oldbalanceDest"]
    balance = out["oldbalanceOrg"].where(out["oldbalanceOrg"] > 0, np.nan)
    out["amount_to_balance_ratio"] = out["amount"] / balance
    return out


def to_model_input(df: pd.DataFrame, feature_set: str = DEFAULT_FEATURE_SET) -> pd.DataFrame:
    """Engineer features and select them in the fixed model order."""
    features = model_features(feature_set)
    engineered = add_engineered_features(df)
    missing = [c for c in features if c not in engineered.columns]
    if missing:
        raise ValueError(f"Feature set {feature_set!r} needs columns that are not available: {missing}")
    return engineered[features]


def build_preprocessor(feature_set: str = DEFAULT_FEATURE_SET) -> ColumnTransformer:
    # Fixed categories keep the encoded layout identical across retrains;
    # an unseen type at inference time encodes to all zeros instead of crashing.
    encoder = OneHotEncoder(categories=[TRANSACTION_TYPES], handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_features(feature_set)),
            ("type", encoder, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

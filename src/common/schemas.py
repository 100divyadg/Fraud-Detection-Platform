"""Column contract shared by training and (later) streaming inference."""

TARGET = "isFraud"

# Every column the training script needs from the raw CSV.
RAW_FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
REQUIRED_COLUMNS = RAW_FEATURE_COLUMNS + [TARGET]

# Present in the CSV but deliberately never used as model inputs:
# isFlaggedFraud is an existing rule's output, the names are identifiers.
EXCLUDED_COLUMNS = ["isFlaggedFraud", "nameOrig", "nameDest"]

# Some PaySim copies spell the origin balance column differently.
COLUMN_ALIASES = {"oldbalanceOrig": "oldbalanceOrg", "newbalanceOrg": "newbalanceOrig"}

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Model inputs per feature set, in the exact order the preprocessor sees them.
# "full" uses every balance column, including post-transaction balances.
# "pre_transaction" keeps only what is known before a payment is authorised, so
# it can be scored without leaking the transaction's outcome.
FEATURE_SETS = {
    "full": [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "hour_of_day",
        "origin_balance_error",
        "dest_balance_change",
    ],
    "pre_transaction": [
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "hour_of_day",
        "amount_to_balance_ratio",
    ],
}
DEFAULT_FEATURE_SET = "full"

# Known only after the transaction has been applied.
POST_TRANSACTION_COLUMNS = ["newbalanceOrig", "newbalanceDest", "origin_balance_error", "dest_balance_change"]

CATEGORICAL_FEATURES = ["type"]


def numeric_features(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[feature_set])


def model_features(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    return numeric_features(feature_set) + CATEGORICAL_FEATURES

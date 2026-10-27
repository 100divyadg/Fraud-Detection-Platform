"""Central configuration. Every phase reads paths and thresholds from here."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = Path(os.getenv("FRAUD_DATA_PATH", PROJECT_ROOT / "data" / "raw" / "paysim.csv"))
ARTIFACT_DIR = Path(os.getenv("FRAUD_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))



def model_filename(feature_set: str) -> str:
    return f"fraud_model_{feature_set}.joblib"


def metadata_filename(feature_set: str) -> str:
    return f"model_metadata_{feature_set}.json"


# Chronological split by `step`: the first 70% of rows train, the next 15%
# choose the decision threshold, the final 15% are the untouched test set.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

RANDOM_STATE = 42

# Kafka. localhost:9092 is the listener Docker Compose exposes to the host;
# containers on the Compose network use kafka:29092 instead.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TRANSACTIONS_TOPIC = os.getenv("TRANSACTIONS_TOPIC", "transactions")
FRAUD_ALERTS_TOPIC = os.getenv("FRAUD_ALERTS_TOPIC", "fraud-alerts")
DEAD_LETTER_TOPIC = os.getenv("DEAD_LETTER_TOPIC", "transactions-dlq")

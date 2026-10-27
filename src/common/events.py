"""Transaction event contract for the `transactions` Kafka topic.

An event carries only what is known before a payment is authorised: the inputs
of the `pre_transaction` model plus identifiers. The fraud label and the
post-transaction balances are deliberately absent.
"""

from datetime import datetime, timezone
from typing import Any

from src.common.schemas import TRANSACTION_TYPES

SCHEMA_VERSION = 1

# Field name -> accepted Python types after JSON decoding.
EVENT_FIELDS: dict[str, tuple[type, ...]] = {
    "schema_version": (int,),
    "transaction_id": (str,),
    "event_timestamp": (str,),
    "step": (int,),
    "type": (str,),
    "amount": (int, float),
    "nameOrig": (str,),
    "oldbalanceOrg": (int, float),
    "nameDest": (str,),
    "oldbalanceDest": (int, float),
}

# Never allowed on the wire: the label, an existing rule's output, and values
# only known after the transaction has been applied.
FORBIDDEN_FIELDS = ["isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest"]


def transaction_id_for_row(row_number: int) -> str:
    """Deterministic ID from the CSV row, so replaying a range re-sends the same IDs."""
    return f"TXN-{row_number:09d}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_event(row: dict[str, Any], row_number: int, event_timestamp: str | None = None) -> dict[str, Any]:
    """Turn one PaySim CSV row into a transaction event."""
    return {
        "schema_version": SCHEMA_VERSION,
        "transaction_id": transaction_id_for_row(row_number),
        "event_timestamp": event_timestamp or utc_now_iso(),
        "step": int(row["step"]),
        "type": str(row["type"]),
        "amount": float(row["amount"]),
        "nameOrig": str(row["nameOrig"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "nameDest": str(row["nameDest"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
    }


def validate_event(event: Any) -> list[str]:
    """Return a list of problems; an empty list means the event is valid."""
    if not isinstance(event, dict):
        return [f"event must be a JSON object, got {type(event).__name__}"]

    errors = []
    for field, types in EVENT_FIELDS.items():
        if field not in event or event[field] is None:
            errors.append(f"missing field: {field}")
        # bool is a subclass of int; reject it explicitly.
        elif isinstance(event[field], bool) or not isinstance(event[field], types):
            errors.append(f"{field} has type {type(event[field]).__name__}")
    for field in FORBIDDEN_FIELDS:
        if field in event:
            errors.append(f"forbidden field present: {field}")
    if errors:
        return errors

    if event["schema_version"] != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {event['schema_version']}")
    if not event["transaction_id"].strip():
        errors.append("transaction_id is empty")
    if event["type"] not in TRANSACTION_TYPES:
        errors.append(f"unknown type {event['type']!r}")
    if event["step"] < 1:
        errors.append("step must be >= 1")
    for field in ("amount", "oldbalanceOrg", "oldbalanceDest"):
        if event[field] < 0:
            errors.append(f"{field} must be >= 0")
    try:
        datetime.fromisoformat(event["event_timestamp"].replace("Z", "+00:00"))
    except ValueError:
        errors.append("event_timestamp is not ISO-8601")
    return errors

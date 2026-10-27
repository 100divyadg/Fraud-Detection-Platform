import json

import joblib
import pandas as pd
import pytest

from src.common.events import (
    EVENT_FIELDS,
    FORBIDDEN_FIELDS,
    build_event,
    transaction_id_for_row,
    validate_event,
)
from src.common.config import model_filename
from src.streaming.producer import Pacer, iter_events
from src.training.features import to_model_input
from src.training.train import train


@pytest.fixture
def event(paysim_df):
    return build_event(paysim_df.iloc[0].to_dict(), row_number=7, event_timestamp="2026-10-27T12:00:00.000Z")


def test_event_has_contract_fields_and_nothing_forbidden(event):
    assert set(event) == set(EVENT_FIELDS)
    assert not set(FORBIDDEN_FIELDS) & set(event)
    assert event["transaction_id"] == "TXN-000000007"
    assert validate_event(event) == []


def test_event_survives_json_round_trip(event):
    assert validate_event(json.loads(json.dumps(event))) == []


def test_transaction_ids_are_deterministic_and_sortable():
    assert transaction_id_for_row(0) == "TXN-000000000"
    assert transaction_id_for_row(6_362_619) == "TXN-006362619"
    assert transaction_id_for_row(9) < transaction_id_for_row(10)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"amount": -1.0}, "amount must be >= 0"),
        ({"type": "CRYPTO"}, "unknown type"),
        ({"step": 0}, "step must be >= 1"),
        ({"event_timestamp": "yesterday"}, "not ISO-8601"),
        ({"amount": "100"}, "amount has type str"),
        ({"step": True}, "step has type bool"),
        ({"schema_version": 99}, "unsupported schema_version"),
        ({"transaction_id": "  "}, "transaction_id is empty"),
        ({"isFraud": 1}, "forbidden field present: isFraud"),
        ({"newbalanceOrig": 0.0}, "forbidden field present: newbalanceOrig"),
    ],
)
def test_validation_catches_bad_events(event, change, expected):
    errors = validate_event({**event, **change})
    assert any(expected in e for e in errors), errors


def test_validation_reports_missing_fields_and_non_objects(event):
    del event["amount"]
    assert "missing field: amount" in validate_event(event)
    assert validate_event(["not", "an", "object"]) == ["event must be a JSON object, got list"]


def test_producer_reads_rows_with_labels_kept_out_of_events(paysim_csv, paysim_df):
    produced = list(iter_events(paysim_csv, start_row=10, limit=25))
    assert len(produced) == 25
    events, labels = zip(*produced)
    assert events[0]["transaction_id"] == "TXN-000000010"
    assert list(labels) == paysim_df["isFraud"].iloc[10:35].tolist()
    assert all(validate_event(e) == [] for e in events)


def test_producer_reads_across_chunk_boundaries(paysim_csv, paysim_df, monkeypatch):
    monkeypatch.setattr("src.streaming.producer.CHUNK_ROWS", 7)
    ids = [e["transaction_id"] for e, _ in iter_events(paysim_csv, start_row=3, limit=20)]
    assert ids == [transaction_id_for_row(i) for i in range(3, 23)]


def test_events_can_be_scored_by_pre_transaction_model(paysim_csv, paysim_df, tmp_path):
    train(paysim_csv, tmp_path, feature_set="pre_transaction")
    model = joblib.load(tmp_path / model_filename("pre_transaction"))

    events = [e for e, _ in iter_events(paysim_csv, limit=50)]
    proba = model.predict_proba(to_model_input(pd.DataFrame(events), "pre_transaction"))[:, 1]
    assert len(proba) == 50 and ((proba >= 0) & (proba <= 1)).all()

    # The full model needs post-transaction balances, which events never carry.
    with pytest.raises(ValueError, match="not available"):
        to_model_input(pd.DataFrame(events), "full")


def test_pacer_unlimited_never_sleeps(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: pytest.fail("should not sleep"))
    pacer = Pacer(0)
    for _ in range(100):
        pacer.wait()

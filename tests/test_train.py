import json

import joblib
import pytest

from src.common.config import metadata_filename, model_filename
from src.common.schemas import FEATURE_SETS, POST_TRANSACTION_COLUMNS, model_features
from src.training.compare import format_table, load_runs
from src.training.features import to_model_input
from src.training.train import chronological_split, load_dataset, train


def test_chronological_split_does_not_share_steps(paysim_csv):
    tr, va, te = chronological_split(load_dataset(paysim_csv))
    assert tr["step"].max() < va["step"].min()
    assert va["step"].max() < te["step"].min()
    assert len(tr) + len(va) + len(te) == len(load_dataset(paysim_csv))


@pytest.mark.parametrize("feature_set", sorted(FEATURE_SETS))
def test_training_writes_usable_artifacts(paysim_csv, paysim_df, tmp_path, feature_set):
    out_dir = tmp_path / "artifacts"
    metadata = train(paysim_csv, out_dir, feature_set=feature_set)

    saved = json.loads((out_dir / metadata_filename(feature_set)).read_text())
    assert saved["model_version"] == metadata["model_version"]
    assert saved["feature_set"] == feature_set
    assert saved["feature_schema"]["model_features"] == model_features(feature_set)
    assert 0 < saved["decision_threshold"] < 1
    assert abs(sum(saved["feature_importance_gain"].values()) - 1) < 0.01

    model = joblib.load(out_dir / model_filename(feature_set))
    proba = model.predict_proba(to_model_input(paysim_df, feature_set))[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()

    # The synthetic fraud pattern is trivially learnable; this guards against a broken pipeline.
    assert saved["test_metrics"]["pr_auc"] > 0.9


def test_pre_transaction_metadata_records_no_post_transaction_columns(paysim_csv, tmp_path):
    metadata = train(paysim_csv, tmp_path, feature_set="pre_transaction")
    assert metadata["feature_schema"]["uses_post_transaction_columns"] == []
    assert not set(POST_TRANSACTION_COLUMNS) & set(metadata["feature_importance_gain"])


def test_feature_sets_save_side_by_side_and_compare(paysim_csv, tmp_path):
    for feature_set in FEATURE_SETS:
        train(paysim_csv, tmp_path, feature_set=feature_set)
    runs = load_runs(tmp_path)
    assert {r["feature_set"] for r in runs} == set(FEATURE_SETS)
    table = format_table(runs)
    assert "full @val" in table and "pre_transaction @val" in table

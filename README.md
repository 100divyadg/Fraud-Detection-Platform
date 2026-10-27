# Fraud Detection Platform

Real-time fraud detection and transaction intelligence. A streaming fraud-detection prototype built on the synthetic **PaySim** mobile-money
dataset: Kafka → Spark Structured Streaming → XGBoost → PostgreSQL / Redis → FastAPI.

This is an educational prototype trained on synthetic data. It is not production-ready.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Inspect the dataset | **Done**: `src/training/inspect_data.py` |
| 2 | Baseline XGBoost model | **Done**: `src/training/train.py`, results below |
| 3 | Kafka producer / topics | Not started |
| 4 | Spark streaming inference | Not started |
| 5 | PostgreSQL, FastAPI, Redis | Not started |
| 6 | Docker, CI, load tests | Not started |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Put the PaySim CSV at `data/raw/paysim.csv`. The Kaggle download is named
`PS_20174392719_1491204439457_log.csv`; rename it, pass `--data`, or set
`FRAUD_DATA_PATH`. The CSV is git-ignored and must not be committed.

## Phase 1: inspect the data

```bash
python -m src.training.inspect_data            # or: --data path/to/file.csv
```

This is read-only. It prints file size, shape, columns (and any missing required ones),
dtypes, nulls, duplicates, fraud rate overall and per transaction type, the `step` range,
a numeric summary and sample rows.

## Phase 2: train the baseline

```bash
python -m src.training.train                   # --nrows 500000 for a quick run
```

Outputs, both git-ignored:

- `artifacts/fraud_model.joblib`: one sklearn `Pipeline` (preprocessing + XGBoost), so
  inference cannot drift from the training-time encoding.
- `artifacts/model_metadata.json`: model version, dataset SHA-256, feature schema, chosen
  threshold, split sizes, hyperparameters, test metrics, latency and library versions.

### Method

**Target:** `isFraud`. **Never used as inputs:** `isFraud`, `isFlaggedFraud` (an existing
rule's output), `nameOrig` and `nameDest` (identifiers).

**Features** (`src/common/schemas.py`, built in `src/training/features.py`):

| Feature | Definition |
|---|---|
| `type` | One-hot encoded with a fixed category list; an unseen type encodes to all zeros |
| `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Raw |
| `hour_of_day` | `step % 24` |
| `origin_balance_error` | `oldbalanceOrg - amount - newbalanceOrig` |
| `dest_balance_change` | `newbalanceDest - oldbalanceDest` |

Raw `step` is replaced by `hour_of_day`. `step` only increases over the simulation, so in a
chronological split every test value lies beyond anything seen in training and the trees
would just treat it as "the largest step seen". The hour keeps the daily pattern without
that problem.

**Split:** chronological by `step`, 70% train / 15% validation / 15% test, with cuts placed
on step boundaries so no step appears in two splits. Known limitation: the same account can
still appear in more than one split.

**Imbalance:** `scale_pos_weight = negatives / positives` on the training split.

**Threshold:** chosen on the validation split by maximising F1, then applied unchanged to the
test split. Test metrics are reported at both 0.5 and the chosen threshold.

**Metrics recorded:** PR-AUC, ROC-AUC, precision, recall, F1, false-positive rate, confusion
matrix, batch and single-event inference latency.

### Data quality report (Phase 1)

Run on the Kaggle file `PS_20174392719_1491204439457_log.csv` (493.5 MB).

| Check | Result |
|---|---|
| Rows x columns | 6,362,620 x 11; all required columns present |
| Missing values / duplicate rows | 0 / 0 |
| Fraud | 8,213 rows (0.1291%), about 1 in 775 |
| `isFlaggedFraud` | 16 rows: the built-in rule catches almost none of the fraud |
| Time range | `step` 1-743 (about 31 days of hourly steps) |

| Type | Transactions | Fraud | Fraud rate |
|---|---:|---:|---:|
| CASH_IN | 1,399,284 | 0 | 0.0000% |
| CASH_OUT | 2,237,500 | 4,116 | 0.1840% |
| DEBIT | 41,432 | 0 | 0.0000% |
| PAYMENT | 2,151,495 | 0 | 0.0000% |
| TRANSFER | 532,909 | 4,097 | 0.7688% |

Fraud occurs only in `TRANSFER` and `CASH_OUT`, matching PaySim's "transfer out, then cash
out" pattern. Predicting "not fraud" for everything would score 99.87% accuracy, which is why
accuracy is not reported.

### Baseline results (Phase 2)

Full dataset, model `v20261027110747`, trained on a Windows laptop.

| Split | Steps | Rows | Fraud | Fraud rate |
|---|---|---:|---:|---:|
| Train | 1-323 | 4,463,587 | 3,643 | 0.082% |
| Validation | 324-378 | 980,416 | 564 | 0.058% |
| Test | 379-743 | 918,617 | 4,006 | 0.436% |

Test set: **PR-AUC 0.9999**, ROC-AUC 1.0000.

| Threshold | Precision | Recall | F1 | FPR | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 (default) | 0.9993 | 0.9995 | 0.9994 | 0.000003 | 914,608 | 3 | 2 | 4,004 |
| 0.9989 (max F1 on validation) | 1.0000 | 0.9950 | 0.9975 | 0.000000 | 914,611 | 0 | 20 | 3,986 |

Inference: 1.98 ms per 1,000 rows in a batch, 5.66 ms median for a single event.

**How to read these numbers**

- **They are a property of the simulator, not proof of a great model.** In PaySim, fraud almost
  always empties the origin account (`oldbalanceOrg == amount`, `newbalanceOrig == 0`), so the
  balance features nearly give the answer away. Real fraud is not this clean.
- **Possible temporal leakage.** `newbalanceOrig` and `newbalanceDest` are post-transaction
  balances. A real scoring system may not know them before the payment is authorised. A
  comparison model without them is the next experiment.
- **The base rate shifts over time.** Legitimate volume falls off in later steps while fraud
  stays roughly constant, so the test split's fraud rate (0.436%) is about 5x the training rate.
  A chronological split exposes this; a random split would hide it.
- **The validation-chosen threshold was too strict for the test period.** It removed the 3 false
  positives but missed 18 more frauds. The validation window had only 564 fraud cases and a
  lower base rate, so the threshold did not transfer cleanly.
- **Latency:** per-row cost is tiny in batches, but a single event pays about 5.7 ms of
  pandas/sklearn overhead. The streaming path should score Spark micro-batches, not individual
  rows.

## Tests

```bash
pytest
```

The tests build a small synthetic PaySim-shaped frame, so they run without the real dataset.
They cover the feature formulas, column order and leak exclusion, encoder stability, the
dataset validation errors, a chronological split with no shared steps, and an end-to-end
training run that loads the saved artifact and checks that probabilities fall in [0, 1].

## Layout

```
src/common/config.py        paths (env-overridable), split fractions
src/common/schemas.py       column contract shared by training and streaming
src/training/features.py    validation, feature engineering, preprocessor
src/training/inspect_data.py
src/training/train.py
tests/
```

Only load `.joblib` model files that you created yourself. Unpickling can run arbitrary code.

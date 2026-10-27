# Fraud Detection Platform

Real-time fraud detection and transaction intelligence. A streaming fraud-detection prototype built on the synthetic **PaySim** mobile-money
dataset: Kafka → Spark Structured Streaming → XGBoost → PostgreSQL / Redis → FastAPI.

This is an educational prototype trained on synthetic data. It is not production-ready.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Inspect the dataset | **Done**: `src/training/inspect_data.py` |
| 2 | Baseline XGBoost model | **Done**: `full` vs `pre_transaction` comparison below; live pipeline uses `pre_transaction` |
| 3 | Kafka producer / topics | **Done**: `docker-compose.yml`, `src/streaming/producer.py`, `src/streaming/check_consumer.py` |
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
python -m src.training.train                                  # full feature set
python -m src.training.train --feature-set pre_transaction    # no post-transaction balances
python -m src.training.compare                                # side-by-side test metrics
```

Add `--nrows 500000` to either training command for a quick run.

Outputs per feature set, all git-ignored:

- `artifacts/fraud_model_<set>.joblib`: one sklearn `Pipeline` (preprocessing + XGBoost), so
  inference cannot drift from the training-time encoding.
- `artifacts/model_metadata_<set>.json`: model version, dataset SHA-256, feature schema, chosen
  threshold, split sizes, hyperparameters, test metrics, feature importance (share of gain),
  latency and library versions.

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
| `amount_to_balance_ratio` | `amount / oldbalanceOrg`; missing (NaN) when the balance is 0 |

**Feature sets.** Two models are trained from the same code and compared:

| Set | Numeric inputs (plus `type`) | Purpose |
|---|---|---|
| `full` | `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`, `hour_of_day`, `origin_balance_error`, `dest_balance_change` | Baseline using every balance column |
| `pre_transaction` | `amount`, `oldbalanceOrg`, `oldbalanceDest`, `hour_of_day`, `amount_to_balance_ratio` | Only values known before the payment is authorised |

`newbalanceOrig` and `newbalanceDest` describe the account after the transaction. A system that
must approve or block a payment would not have them yet, so the `full` model may be using
information from the future. The `pre_transaction` model measures how much the score depends on
that information. A test checks that changing post-transaction balances never changes its input.

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

`full` feature set on the full dataset, model `v20261027110747`, trained on a Windows laptop.

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

### Feature-set comparison

Both models trained on the full dataset with identical splits and hyperparameters
(`full-v20261027152251`, `pre_transaction-v20261027154648`). Retraining reproduced the same
metrics exactly. Test set: 918,617 transactions, 4,006 of them fraud.

| Model @ threshold | PR-AUC | Precision | Recall | F1 | False alarms | Missed fraud | Caught |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` @ 0.5 | 0.9999 | 0.9993 | 0.9995 | 0.9994 | 3 | 2 | 4,004 |
| `full` @ 0.9989 (validation) | 0.9999 | 1.0000 | 0.9950 | 0.9975 | 0 | 20 | 3,986 |
| `pre_transaction` @ 0.5 | 0.9873 | 0.8837 | 0.9783 | 0.9286 | 516 | 87 | 3,919 |
| `pre_transaction` @ 0.9980 (validation) | 0.9873 | 0.9843 | 0.8430 | 0.9082 | 54 | 629 | 3,377 |

Top features by share of gain:

| `full` | | `pre_transaction` | |
|---|---:|---|---:|
| `origin_balance_error` | 0.42 | `amount_to_balance_ratio` | 0.81 |
| `newbalanceOrig` | 0.23 | `amount` | 0.07 |
| `oldbalanceOrg` | 0.13 | `oldbalanceOrg` | 0.06 |

**Findings**

1. **The near-perfect `full` score depends on post-transaction balances.** Its two most important
   features, `origin_balance_error` and `newbalanceOrig`, carry 65% of the gain, and both
   require knowing the account balance after the payment.
2. **Removing them costs a measurable amount, not everything.** At the 0.5 threshold, false
   alarms rise from 3 to 516 and missed fraud from 2 to 87. Recall stays at 97.8% and the
   false-positive rate is 0.056%: about 1 legitimate transaction in 1,800 is flagged.
3. **The honest model still mostly learns PaySim's "empty the account" pattern**, now through
   `amount_to_balance_ratio` (81% of the gain). That signal is legitimate before the payment,
   but it is far stronger in PaySim than it would be in real fraud.
4. **The validation-chosen threshold is again too strict for the test period.** It cuts false
   alarms from 516 to 54 but misses 629 frauds instead of 87, because the validation window has
   a lower fraud rate (0.058%) than the test window (0.436%).

**Decision: the live pipeline uses `pre_transaction`.** It is the only model that can score a
payment before it completes. The two thresholds map naturally onto risk levels:

| Risk level | Probability | Action | Test-set effect |
|---|---|---|---|
| `HIGH` | >= 0.998 | Alert immediately | 3,377 frauds, 54 false alarms |
| `MEDIUM` | 0.5 to 0.998 | Queue for analyst review | a further 542 frauds, 462 false alarms |
| `LOW` | < 0.5 | No action | 87 frauds missed |

Consequently the Kafka transaction event (Phase 3) must carry `type`, `amount`,
`oldbalanceOrg`, `oldbalanceDest` and `step`, and does not need the post-transaction balances.

## Phase 3: stream transactions through Kafka

Needs Docker Desktop. On Windows it needs WSL 2: run `wsl --install` in an administrator
PowerShell, restart, then install Docker Desktop.

```bash
docker compose up -d                                   # Kafka + topics + Kafka UI
python -m src.streaming.producer --limit 5 --dry-run   # print events, no Kafka needed
python -m src.streaming.producer --limit 1000 --rate 100
python -m src.streaming.check_consumer                 # read back and validate
```

Kafka UI is at http://localhost:8080. Stop with `docker compose down`; add `-v` to delete
the stored messages as well.

**Services** (`docker-compose.yml`):

| Service | Purpose |
|---|---|
| `kafka` | Single Kafka 3.9 broker in KRaft mode (no ZooKeeper). Laptop clients use `localhost:9092`; containers use `kafka:29092` |
| `kafka-init` | Creates the topics once Kafka is healthy, then exits. Auto-creation is off, so a mistyped topic name fails instead of silently creating a new topic |
| `kafka-ui` | Web UI for browsing topics, partitions and messages |

**Topics:**

| Topic | Partitions | Contents |
|---|---:|---|
| `transactions` | 3 | Incoming transaction events |
| `fraud-alerts` | 1 | High-risk transactions (written in Phase 4) |
| `transactions-dlq` | 1 | Malformed events routed aside by the processor (Phase 4) |

**Event contract** (`src/common/events.py`, `schema_version` 1):

```json
{
  "schema_version": 1,
  "transaction_id": "TXN-000000002",
  "event_timestamp": "2026-10-27T16:01:06.676Z",
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "nameOrig": "C1305486145",
  "oldbalanceOrg": 181.0,
  "nameDest": "C553264065",
  "oldbalanceDest": 0.0
}
```

- Carries exactly the inputs of the `pre_transaction` model plus identifiers. `isFraud`,
  `isFlaggedFraud`, `newbalanceOrig` and `newbalanceDest` are forbidden, and
  `validate_event` rejects an event containing any of them.
- `transaction_id` is derived from the CSV row number, so replaying the same rows re-sends the
  same IDs. This lets later phases test idempotent writes.
- `event_timestamp` is the producer's clock at send time (UTC). PaySim's `step` is simulated
  hours, not a real timestamp.
- Messages are keyed by `nameOrig`, so one account's transactions stay on one partition, in order.
- The producer uses `acks=all` with idempotence enabled, so broker retries don't duplicate messages.
- `--labels-out labels.csv` writes `transaction_id,isFraud` to a separate file, for scoring the
  streaming predictions against ground truth later without putting the label on the wire.

**Producer options:** `--start-row`, `--limit` (0 = rest of the file), `--rate` events/second
(0 = unlimited), `--dry-run`, `--labels-out`.

**`check_consumer`** reads with a throwaway consumer group from the earliest offset, so it
never moves the offsets a real consumer depends on. It validates every message against the
contract, checks that each key equals `nameOrig`, prints counts by type and partition, and
exits non-zero if anything is invalid.

**Verified** (Linux, 20,200 synthetic PaySim-shaped events): all valid, 0 key mismatches,
spread evenly over the 3 partitions (6,687 / 6,776 / 6,737). Unthrottled replay delivered
about 27,700 events/s; `--rate 100` held exactly 100 events/s. Hand-written invalid messages
(non-JSON, and an event containing `isFraud`) were reported as invalid and the check exited 1.

## Tests

```bash
pytest
```

The tests build a small synthetic PaySim-shaped frame, so they run without the real dataset.
They cover the feature formulas, column order and leak exclusion, encoder stability, the
dataset validation errors, a chronological split with no shared steps, and an end-to-end
training run per feature set that loads the saved artifact and checks that probabilities fall
in [0, 1]. They also check that the `pre_transaction` set never sees post-transaction values, that
events match the contract (and that invalid ones are rejected), that the producer reads
row ranges correctly across chunk boundaries, and that streamed events can be scored by the
`pre_transaction` model but not by `full`. None of the tests need Kafka running.

## Layout

```
src/common/config.py        paths (env-overridable), split fractions
src/common/schemas.py       column contract shared by training and streaming
src/training/features.py    validation, feature engineering, preprocessor
src/training/inspect_data.py
src/training/train.py       --feature-set full | pre_transaction
src/training/compare.py     side-by-side metrics for every trained feature set
src/common/events.py        Kafka event contract: build_event, validate_event
src/streaming/producer.py   replay PaySim rows into the transactions topic
src/streaming/check_consumer.py  read events back and validate them
docker-compose.yml          Kafka, topic setup, Kafka UI
tests/
```

Only load `.joblib` model files that you created yourself. Unpickling can run arbitrary code.

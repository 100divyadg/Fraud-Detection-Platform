"""Replay PaySim rows into Kafka as transaction events.

Usage:
  python -m src.streaming.producer --limit 1000 --rate 100
  python -m src.streaming.producer --limit 5 --dry-run        # print events, no Kafka needed

Each event is keyed by `nameOrig`, so all transactions from one account land
on the same partition and stay in order. Transaction IDs come from the CSV row
number, so replaying the same range re-sends the same IDs (useful for testing
idempotent writes later).
"""

import argparse
import json
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from src.common.config import DATA_PATH, KAFKA_BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC
from src.common.events import build_event, transaction_id_for_row
from src.common.schemas import TARGET
from src.training.features import normalise_columns

SOURCE_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest", TARGET]
CHUNK_ROWS = 50_000


def read_rows(path: Path, start_row: int, limit: int | None) -> Iterator[tuple[int, dict]]:
    """Yield (row_number, row) from the CSV, row_number counting data rows from 0."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Pass --data or set FRAUD_DATA_PATH.")
    reader = pd.read_csv(
        path,
        skiprows=range(1, start_row + 1),
        nrows=limit,
        chunksize=CHUNK_ROWS,
    )
    row_number = start_row
    for chunk in reader:
        chunk = normalise_columns(chunk)
        for row in chunk[SOURCE_COLUMNS].to_dict("records"):
            yield row_number, row
            row_number += 1


def iter_events(path: Path, start_row: int = 0, limit: int | None = None) -> Iterator[tuple[dict, int]]:
    """Yield (event, label). The label never goes into the event."""
    for row_number, row in read_rows(path, start_row, limit):
        yield build_event(row, row_number), int(row[TARGET])


class Pacer:
    """Sleep just enough to hold a target events-per-second rate (0 = unlimited)."""

    def __init__(self, rate: float):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.next_at = time.perf_counter()

    def wait(self) -> None:
        if not self.interval:
            return
        self.next_at += self.interval
        delay = self.next_at - time.perf_counter()
        if delay > 0:
            time.sleep(delay)


def make_producer(bootstrap_servers: str):
    from confluent_kafka import Producer

    return Producer(
        {
            "bootstrap.servers": bootstrap_servers,
            "client.id": "paysim-replay",
            # acks=all + idempotence: a broker retry never duplicates or reorders a message.
            "acks": "all",
            "enable.idempotence": True,
            "linger.ms": 5,
            "compression.type": "lz4",
        }
    )


def produce(args: argparse.Namespace) -> dict:
    labels_file = None
    if args.labels_out:
        args.labels_out.parent.mkdir(parents=True, exist_ok=True)
        labels_file = args.labels_out.open("w", encoding="utf-8")
        labels_file.write(f"transaction_id,{TARGET}\n")

    stats = {"sent": 0, "delivered": 0, "failed": 0, "fraud_rows": 0}
    producer = None if args.dry_run else make_producer(args.bootstrap_servers)

    def on_delivery(err, _msg):
        stats["failed" if err else "delivered"] += 1
        if err:
            print(f"Delivery failed: {err}")

    pacer = Pacer(args.rate)
    started = time.perf_counter()
    try:
        for event, label in iter_events(args.data, args.start_row, args.limit):
            stats["fraud_rows"] += label
            if labels_file:
                labels_file.write(f"{event['transaction_id']},{label}\n")

            if producer is None:
                print(json.dumps(event))
            else:
                payload = json.dumps(event).encode("utf-8")
                while True:
                    try:
                        producer.produce(args.topic, key=event["nameOrig"], value=payload, on_delivery=on_delivery)
                        break
                    except BufferError:
                        # Local queue is full: let in-flight messages complete, then retry.
                        producer.poll(0.5)
                producer.poll(0)
            stats["sent"] += 1
            if stats["sent"] % 10_000 == 0:
                print(f"  {stats['sent']:,} events sent")
            pacer.wait()
    finally:
        if producer is not None:
            remaining = producer.flush(30)
            if remaining:
                stats["failed"] += remaining
                print(f"{remaining} messages still undelivered after 30 s flush")
        if labels_file:
            labels_file.close()

    elapsed = time.perf_counter() - started
    stats["seconds"] = round(elapsed, 2)
    stats["events_per_second"] = round(stats["sent"] / elapsed, 1) if elapsed else 0.0
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the PaySim CSV")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=TRANSACTIONS_TOPIC)
    parser.add_argument("--start-row", type=int, default=0, help="First CSV data row to send (0-based)")
    parser.add_argument("--limit", type=int, default=1000, help="Number of rows to send; 0 sends the rest of the file")
    parser.add_argument("--rate", type=float, default=0, help="Events per second; 0 = as fast as possible")
    parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help="Optional CSV of transaction_id,isFraud for evaluating streaming predictions later",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print events instead of sending them")
    args = parser.parse_args()
    if args.limit == 0:
        args.limit = None

    first_id = transaction_id_for_row(args.start_row)
    target = "stdout (dry run)" if args.dry_run else f"{args.topic} @ {args.bootstrap_servers}"
    print(f"Replaying from {first_id} to {target}")
    stats = produce(args)
    print(
        f"Done: sent={stats['sent']:,} delivered={stats['delivered']:,} failed={stats['failed']:,} "
        f"fraud_rows={stats['fraud_rows']:,} in {stats['seconds']} s ({stats['events_per_second']:,} events/s)"
    )
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Read events back from Kafka and validate them against the event contract.

Usage: python -m src.streaming.check_consumer [--max-messages 1000] [--timeout 10]

Uses a fresh consumer group on every run and reads from the earliest offset, so
it never commits progress that a real consumer (Spark, Phase 4) depends on.
"""

import argparse
import json
import uuid
from collections import Counter

from src.common.config import KAFKA_BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC
from src.common.events import validate_event


def check(bootstrap_servers: str, topic: str, max_messages: int, timeout: float) -> dict:
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap_servers,
            "group.id": f"contract-check-{uuid.uuid4()}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])

    stats = {"read": 0, "valid": 0, "invalid": 0, "by_type": Counter(), "by_partition": Counter(), "errors": Counter()}
    keys_mismatched = 0
    sample = None
    try:
        while stats["read"] < max_messages:
            msg = consumer.poll(timeout)
            if msg is None:
                break  # no new messages within the timeout: the topic is drained
            if msg.error():
                print(f"Kafka error: {msg.error()}")
                continue
            stats["read"] += 1
            stats["by_partition"][msg.partition()] += 1
            try:
                event = json.loads(msg.value())
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                stats["invalid"] += 1
                stats["errors"][f"not JSON: {type(exc).__name__}"] += 1
                continue
            problems = validate_event(event)
            if problems:
                stats["invalid"] += 1
                stats["errors"].update(problems)
                continue
            stats["valid"] += 1
            stats["by_type"][event["type"]] += 1
            if msg.key() is None or msg.key().decode() != event["nameOrig"]:
                keys_mismatched += 1
            sample = sample or event
    finally:
        consumer.close()

    stats["keys_mismatched"] = keys_mismatched
    stats["sample"] = sample
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS)
    parser.add_argument("--topic", default=TRANSACTIONS_TOPIC)
    parser.add_argument("--max-messages", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=10.0, help="Stop after this many seconds without a message")
    args = parser.parse_args()

    stats = check(args.bootstrap_servers, args.topic, args.max_messages, args.timeout)
    print(f"Read {stats['read']:,} messages from {args.topic}: valid={stats['valid']:,} invalid={stats['invalid']:,}")
    print("By type:", dict(sorted(stats["by_type"].items())))
    print("By partition:", dict(sorted(stats["by_partition"].items())))
    print("Messages whose key != nameOrig:", stats["keys_mismatched"])
    if stats["errors"]:
        print("Validation errors:", dict(stats["errors"].most_common(10)))
    if stats["sample"]:
        print("Sample event:", json.dumps(stats["sample"], indent=2))
    if stats["read"] == 0:
        raise SystemExit("No messages read. Is Kafka running and has the producer sent anything?")
    if stats["invalid"] or stats["keys_mismatched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

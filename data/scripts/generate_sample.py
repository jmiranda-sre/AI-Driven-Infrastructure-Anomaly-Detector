"""Generate sample metric data for testing and development."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

METRICS = ["cpu_usage", "memory_usage", "disk_io_util", "network_in", "network_out", "load_avg_5m", "iowait", "error_rate"]
SERVERS = ["prod-web-01", "prod-web-02", "prod-api-01", "prod-db-01", "staging-01"]


def generate_sample_csv(output_path: str = "data/sample/metrics.csv", n_records: int = 2880) -> None:
    """Generate a sample CSV with realistic server metrics.

    2880 records = 10 days at 5-minute intervals.
    Includes daily seasonality and injected anomalies.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    interval = timedelta(minutes=5)

    np.random.seed(42)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "server_id", *METRICS])

        for i in range(n_records):
            ts = start_time + i * interval
            hour = ts.hour

            # Seasonal patterns
            is_business = 9 <= hour <= 17
            is_weekday = ts.weekday() < 5

            for server in SERVERS:
                base_cpu = 35 if not is_business else 55
                base_mem = 50 if not is_business else 65

                cpu = base_cpu + 15 * np.sin(2 * np.pi * i / 288) + np.random.normal(0, 5)
                memory = base_mem + 8 * np.sin(2 * np.pi * i / 288 + 1) + np.random.normal(0, 3)
                disk = 15 + 10 * np.abs(np.sin(2 * np.pi * i / 288 * 2)) + np.random.normal(0, 4)
                net_in = (80 if is_business else 30) + 30 * np.sin(2 * np.pi * i / 288 + 2) + np.random.normal(0, 10)
                net_out = (60 if is_business else 20) + 20 * np.sin(2 * np.pi * i / 288 + 3) + np.random.normal(0, 8)
                load = cpu / 25 + np.random.normal(0, 0.3)
                iowait_val = 3 + 2 * np.abs(np.sin(2 * np.pi * i / 288)) + np.random.normal(0, 0.8)
                errors = np.random.exponential(0.3 if is_weekday else 0.1)

                # Inject anomalies (~3% of records)
                if random.random() < 0.03:
                    cpu = min(100, cpu * random.uniform(1.5, 2.5))
                    errors *= random.uniform(5, 20)

                writer.writerow([
                    ts.isoformat(),
                    server,
                    round(max(0, cpu), 2),
                    round(max(0, min(100, memory)), 2),
                    round(max(0, disk), 2),
                    round(max(0, net_in), 2),
                    round(max(0, net_out), 2),
                    round(max(0, load), 2),
                    round(max(0, iowait_val), 2),
                    round(max(0, errors), 4),
                ])

    print(f"Generated {n_records * len(SERVERS)} records → {path}")


def generate_prometheus_export_file(output_path: str = "data/sample/prometheus_metrics.txt") -> None:
    """Generate a file in Prometheus text exposition format for testing."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for metric in METRICS:
        lines.append(f"# HELP {metric} Simulated {metric} metric")
        lines.append(f"# TYPE {metric} gauge")
        for server in SERVERS[:2]:
            value = np.random.uniform(10, 90)
            lines.append(f'{metric}{{instance="{server}:9100"}} {value:.2f}')

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated Prometheus export file → {path}")


if __name__ == "__main__":
    generate_sample_csv()
    generate_prometheus_export_file()

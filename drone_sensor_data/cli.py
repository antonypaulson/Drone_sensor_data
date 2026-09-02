"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import sys

from drone_sensor_data.analysis import run_original_detectors
from drone_sensor_data.load import DataNotFoundError, load_dataset


def _ids(frame) -> list[int]:
    if frame is None or len(frame) == 0:
        return []
    return [int(i) for i in frame["flight_id"].tolist()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load drone launch CSVs and print the original-study detectors."
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help="Do not fall back to the bundled sample dataset.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print detector flight IDs as JSON.",
    )
    args = parser.parse_args(argv)

    try:
        dataset = load_dataset(allow_sample=not args.full_only)
    except DataNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    results = run_original_detectors(dataset)
    payload = {
        "source": results["source"],
        "data_dir": str(results["data_dir"]),
        "n_flights": results["n_flights"],
        "missing_voltage_count": int(len(results["missing_voltage"])),
        "position_outliers": _ids(results["position_outliers"]),
        "yaw_problems": _ids(results["yaw_problems"]),
        "accel_outliers": _ids(results["accel_outliers"]),
        "northbound_velocity": _ids(results["northbound_velocity"]),
        "angular_rate_outliers": results["angular_rate_outliers"],
        "fleet_counts": results["fleet_counts"]["Count"].to_dict(),
    }
    if args.as_json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"Loaded {payload['n_flights']} flights from {payload['data_dir']} ({payload['source']})")
    if payload["source"] == "sample":
        print(
            "Using the bundled sample dataset. Original 447-flight findings require "
            "the full CSVs; run: python -m drone_sensor_data.download"
        )
    print(f"Missing preflight_voltage: {payload['missing_voltage_count']}")
    print(f"Position / north-track outliers: {payload['position_outliers']}")
    print(f"Negative-yaw flights: {payload['yaw_problems']}")
    print(f"Body-right accel outliers: {payload['accel_outliers']}")
    print(f"Northbound velocity flights: {payload['northbound_velocity']}")
    print(f"Angular-rate outliers: {payload['angular_rate_outliers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

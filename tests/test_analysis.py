"""Tests that the original detector thresholds still fire on known rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drone_sensor_data.analysis import (
    accel_body_right_outliers,
    angular_rate_down_outliers,
    daily_weather,
    ending_positions,
    fleet_counts,
    location_models,
    missing_preflight_voltage,
    northbound_velocity_flights,
    position_outlier_flights,
    run_original_detectors,
    yaw_problem_flights,
)
from drone_sensor_data.load import Dataset, merge_flights, parse_launch_timestamp
from drone_sensor_data.sample_data import build_sample_frames, write_sample_dataset


def _sample_dataset() -> Dataset:
    summary, flights = build_sample_frames()
    summary = summary.copy()
    summary["launch_timestamp"] = parse_launch_timestamp(summary["launch_timestamp"])
    summary["flight_id"] = summary["flight_id"].astype("int64")
    merged = merge_flights(flights, summary)
    return Dataset(
        data_dir=Path("."),
        source="sample",
        summary=summary,
        flights=flights,
        merged=merged,
    )


def test_missing_voltage_and_fleet_counts() -> None:
    dataset = _sample_dataset()
    missing = missing_preflight_voltage(dataset.summary)
    assert list(missing["flight_id"]) == [10001]
    counts = fleet_counts(dataset.summary)
    assert int(counts.loc["Flights", "Count"]) == 5
    assert set(dataset.flights) == {10001, 10002, 10003, 10004, 10005}


def test_original_detectors_on_sample() -> None:
    dataset = _sample_dataset()
    merged, summary = dataset.merged, dataset.summary
    assert list(position_outlier_flights(merged, summary)["flight_id"]) == [10002]
    assert list(yaw_problem_flights(merged, summary)["flight_id"]) == [10002]
    assert list(northbound_velocity_flights(merged, summary)["flight_id"]) == [10002]
    assert list(accel_body_right_outliers(merged, summary)["flight_id"]) == [10003]
    angular = angular_rate_down_outliers(merged)
    assert 10004 in angular["negative_down"]
    assert 10004 in angular["positive_down"]


def test_daily_weather_labels() -> None:
    dataset = _sample_dataset()
    weather = daily_weather(dataset.summary)
    assert "9/6" in set(weather["Day"])
    assert "10/1" in set(weather["Day"])


def test_ending_positions_and_models() -> None:
    dataset = _sample_dataset()
    ending = ending_positions(dataset.flights)
    assert len(ending) == 5
    combined = dataset.summary.merge(ending, on="flight_id", how="inner")
    metrics = location_models(combined)
    assert list(metrics.columns) == ["MAE", "MSE", "RMSE", "R-Squared"]
    assert "Multiple Linear Regression" in metrics.index


def test_run_original_detectors_roundtrip(tmp_path: Path) -> None:
    write_sample_dataset(tmp_path)
    from drone_sensor_data.load import load_dataset

    dataset = load_dataset(tmp_path, allow_sample=False)
    results = run_original_detectors(dataset)
    assert results["source"] == "full"
    assert results["n_flights"] == 5

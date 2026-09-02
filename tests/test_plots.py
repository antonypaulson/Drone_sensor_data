"""Smoke-test that 3D helpers accept a tiny dataset."""

from __future__ import annotations

from pathlib import Path

from drone_sensor_data.load import Dataset, merge_flights, parse_launch_timestamp
import matplotlib.pyplot as plt

from drone_sensor_data.plots import (
    plot_body_accel,
    plot_daily_weather,
    plot_ned_velocity,
    plot_orientation,
    plot_position_ned,
)
from drone_sensor_data.analysis import daily_weather
from drone_sensor_data.sample_data import build_sample_frames


def test_plots_smoke() -> None:
    summary, flights = build_sample_frames()
    summary = summary.copy()
    summary["launch_timestamp"] = parse_launch_timestamp(summary["launch_timestamp"])
    dataset = Dataset(
        data_dir=Path("."),
        source="sample",
        summary=summary,
        flights=flights,
        merged=merge_flights(flights, summary),
    )
    plot_daily_weather(daily_weather(summary))
    plot_position_ned(dataset)
    plot_orientation(dataset)
    plot_body_accel(dataset)
    plot_ned_velocity(dataset)
    plt.close("all")

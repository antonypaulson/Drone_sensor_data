"""Tiny bundled dataset so the pipeline runs without the ~90 MB take-home dump.

Flight IDs are in the 1000x range so they cannot be confused with the original
447-flight study (IDs 16951–17745).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from drone_sensor_data.constants import FLIGHT_COLUMNS, SUMMARY_COLUMNS
from drone_sensor_data.load import repo_root, SAMPLE_RELATIVE


def _timeseries(n: int = 21) -> np.ndarray:
    return np.linspace(-5.0, 15.0, n)


def _base_flight(n: int = 21) -> pd.DataFrame:
    t = _timeseries(n)
    zeros = np.zeros(n)
    small = np.full(n, 0.01)
    frame = pd.DataFrame(
        {
            "seconds_since_launch": t,
            "position_ned_m[0]": np.linspace(5.0, -390.0, n),
            "position_ned_m[1]": np.linspace(8.0, 176.0, n),
            "position_ned_m[2]": np.linspace(-4.5, -76.0, n),
            "velocity_ned_mps[0]": np.linspace(0.0, -24.0, n),
            "velocity_ned_mps[1]": np.linspace(0.0, 13.0, n),
            "velocity_ned_mps[2]": np.linspace(0.0, -2.0, n),
            "accel_body_mps2[0]": np.linspace(2.0, 1.4, n),
            "accel_body_mps2[1]": np.linspace(-0.1, 0.2, n),
            "accel_body_mps2[2]": np.linspace(-9.6, -4.2, n),
            "orientation_rad[0]": np.linspace(0.007, -0.28, n),
            "orientation_rad[1]": np.linspace(0.21, 0.07, n),
            "orientation_rad[2]": np.linspace(2.74, 2.60, n),
            "angular_rate_body_radps[0]": small,
            "angular_rate_body_radps[1]": small * -1,
            "angular_rate_body_radps[2]": zeros,
            "position_sigma_ned_m[0]": np.full(n, 0.18),
            "position_sigma_ned_m[1]": np.full(n, 0.34),
            "position_sigma_ned_m[2]": np.full(n, 0.42),
        }
    )
    return frame[FLIGHT_COLUMNS]


def _summary_row(flight_id: int, **overrides) -> dict:
    row = {
        "flight_id": flight_id,
        "air_temperature": 22.0,
        "battery_serial_number": "SAMPLEBAT01",
        "body_serial_number": 1001,
        "commit": "samplecommit",
        "launch_airspeed": 32.0,
        "launch_groundspeed": 30.0,
        "launch_timestamp": "2018-09-07 08:00:00 CAT",
        "preflight_voltage": 32.1,
        "rel_humidity": 65.0,
        "static_pressure": 80700.0,
        "wind_direction": -20.0,
        "wind_magnitude": 2.0,
        "wing_serial_number": "SAMPLEWING01",
    }
    row.update(overrides)
    return row


def build_sample_frames() -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Four flights covering each original detector, plus one missing-voltage row."""
    normal = _base_flight()

    pos_yaw_vel = _base_flight()
    pos_yaw_vel["position_ned_m[0]"] = np.linspace(5.0, 250.0, len(pos_yaw_vel))
    pos_yaw_vel["orientation_rad[2]"] = np.linspace(0.1, -0.2, len(pos_yaw_vel))
    pos_yaw_vel["velocity_ned_mps[0]"] = np.linspace(0.0, 12.0, len(pos_yaw_vel))

    accel = _base_flight()
    accel.loc[accel.index[-3], "accel_body_mps2[1]"] = -8.2

    angular = _base_flight()
    angular.loc[angular.index[5], "angular_rate_body_radps[2]"] = -0.55
    angular.loc[angular.index[-2], "angular_rate_body_radps[2]"] = 0.41

    flights = {
        10001: normal,
        10002: pos_yaw_vel,
        10003: accel,
        10004: angular,
        10005: _base_flight(),
    }
    for flight_id, frame in flights.items():
        frame["flight_id"] = flight_id
    summary = pd.DataFrame(
        [
            _summary_row(
                10001,
                launch_timestamp="2018-09-06 07:43:59 CAT",
                preflight_voltage=np.nan,
                battery_serial_number="SAMPLEBAT00",
                air_temperature=20.5,
                launch_airspeed=32.4,
                launch_groundspeed=30.1,
                rel_humidity=74.0,
                static_pressure=80662.0,
                wind_direction=-49.0,
                wind_magnitude=1.9,
            ),
            _summary_row(
                10002,
                launch_timestamp="2018-09-24 19:30:15 CAT",
                preflight_voltage=31.9,
                air_temperature=24.1,
                launch_airspeed=31.2,
                launch_groundspeed=29.8,
                rel_humidity=60.0,
                static_pressure=80510.0,
                wind_direction=10.0,
                wind_magnitude=3.2,
            ),
            _summary_row(
                10003,
                launch_timestamp="2018-09-12 16:07:26 CAT",
                preflight_voltage=32.4,
                wing_serial_number="SAMPLEWING02",
                air_temperature=26.0,
                launch_airspeed=33.5,
                launch_groundspeed=30.4,
                rel_humidity=55.0,
                static_pressure=80780.0,
                wind_direction=-12.0,
                wind_magnitude=2.4,
            ),
            _summary_row(
                10004,
                launch_timestamp="2018-09-18 16:09:23 CAT",
                preflight_voltage=32.0,
                body_serial_number=1002,
                air_temperature=21.2,
                launch_airspeed=30.8,
                launch_groundspeed=29.5,
                rel_humidity=70.0,
                static_pressure=80600.0,
                wind_direction=40.0,
                wind_magnitude=1.1,
            ),
            _summary_row(
                10005,
                launch_timestamp="2018-10-01 09:00:00 CAT",
                preflight_voltage=31.5,
                air_temperature=18.0,
                launch_airspeed=29.9,
                launch_groundspeed=28.7,
                rel_humidity=80.0,
                static_pressure=80450.0,
                wind_direction=90.0,
                wind_magnitude=4.5,
            ),
        ]
    )[SUMMARY_COLUMNS]
    return summary, flights


def write_sample_dataset(dest: Path | None = None) -> Path:
    dest = Path(dest) if dest is not None else repo_root() / SAMPLE_RELATIVE
    dest.mkdir(parents=True, exist_ok=True)
    summary, flights = build_sample_frames()
    summary.to_csv(dest / "summary_data.csv", index=False)
    for flight_id, frame in flights.items():
        frame.to_csv(dest / f"flight_{flight_id}.csv", index=False)
    return dest


if __name__ == "__main__":
    path = write_sample_dataset()
    print(f"Wrote sample dataset to {path}")

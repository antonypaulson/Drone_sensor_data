"""Original EDA detectors, with pandas 2-safe implementations.

Thresholds and model settings match the original notebook. This module
does not introduce new anomaly rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from drone_sensor_data.constants import (
    ACCEL_BODY_RIGHT_MPS2_THRESHOLD,
    ANGULAR_RATE_DOWN_NEG_THRESHOLD,
    ANGULAR_RATE_DOWN_POS_THRESHOLD,
    ML_RANDOM_STATE,
    ML_TEST_SIZE,
    POSITION_COLUMNS,
    POSITION_NORTH_M_THRESHOLD,
    PREDICTOR_COLUMNS,
    VELOCITY_NORTH_MPS_THRESHOLD,
    YAW_RAD_THRESHOLD,
)
from drone_sensor_data.load import Dataset


def missing_preflight_voltage(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("flight_id", "launch_timestamp", "battery_serial_number") if c in summary.columns]
    missing = summary.loc[summary["preflight_voltage"].isna(), cols]
    return missing.sort_values("battery_serial_number" if "battery_serial_number" in cols else cols[0])


def fleet_counts(summary: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Count": [
                summary["wing_serial_number"].nunique(),
                summary["battery_serial_number"].nunique(),
                summary["body_serial_number"].nunique(),
                summary["flight_id"].nunique(),
            ]
        },
        index=["Wings", "Batteries", "Bodies", "Flights"],
    )


def daily_weather(summary: pd.DataFrame) -> pd.DataFrame:
    """Mean daily air temperature, humidity, and wind magnitude.

    The original notebook grouped by month and day of `launch_timestamp`.
    """
    weather = (
        summary.groupby(
            [summary["launch_timestamp"].dt.month, summary["launch_timestamp"].dt.day],
            dropna=False,
        )[["air_temperature", "rel_humidity", "wind_magnitude"]]
        .mean()
        .rename_axis(["Month", "Day"])
        .reset_index()
    )
    weather["Day"] = weather["Month"].map(str) + "/" + weather["Day"].map(str)
    return weather


def _lookup_summary(summary: pd.DataFrame, flight_ids, extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    ids = [int(i) for i in pd.unique(pd.Series(flight_ids))]
    cols = ["flight_id", "launch_timestamp", *extra_cols]
    cols = [c for c in cols if c in summary.columns]
    if not ids:
        return pd.DataFrame(columns=cols)
    frame = summary.loc[summary["flight_id"].isin(ids), cols].copy()
    return frame.sort_values("launch_timestamp").reset_index(drop=True)


def unique_flight_ids(merged: pd.DataFrame, mask: pd.Series) -> list[int]:
    return [int(i) for i in merged.loc[mask, "flight_id"].dropna().unique()]


def position_outlier_flights(merged: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    ids = unique_flight_ids(merged, merged["position_ned_m[0]"] > POSITION_NORTH_M_THRESHOLD)
    extra = ("body_serial_number", "battery_serial_number", "wing_serial_number")
    return _lookup_summary(summary, ids, extra)


def yaw_problem_flights(merged: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    ids = unique_flight_ids(merged, merged["orientation_rad[2]"] < YAW_RAD_THRESHOLD)
    return _lookup_summary(summary, ids)


def accel_body_right_outliers(merged: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    ids = unique_flight_ids(
        merged, merged["accel_body_mps2[1]"] < ACCEL_BODY_RIGHT_MPS2_THRESHOLD
    )
    extra = ("body_serial_number", "battery_serial_number", "wing_serial_number")
    return _lookup_summary(summary, ids, extra)


def northbound_velocity_flights(merged: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    ids = unique_flight_ids(
        merged, merged["velocity_ned_mps[0]"] > VELOCITY_NORTH_MPS_THRESHOLD
    )
    return _lookup_summary(summary, ids)


def angular_rate_down_outliers(merged: pd.DataFrame) -> dict[str, list[int]]:
    negative = unique_flight_ids(
        merged, merged["angular_rate_body_radps[2]"] < ANGULAR_RATE_DOWN_NEG_THRESHOLD
    )
    positive = unique_flight_ids(
        merged, merged["angular_rate_body_radps[2]"] > ANGULAR_RATE_DOWN_POS_THRESHOLD
    )
    return {"negative_down": negative, "positive_down": positive}


def ending_positions(flights: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Last logged NED position per flight (original notebook used `[-1:]`)."""
    rows = []
    for flight_id in sorted(flights):
        frame = flights[flight_id]
        last = frame.iloc[[-1]][["flight_id", *POSITION_COLUMNS]]
        rows.append(last)
    return pd.concat(rows, ignore_index=True)


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-score with numpy's default ddof=0 (original `np.std`)."""
    std = frame.std(ddof=0)
    std = std.mask(std == 0, 1.0)
    return (frame - frame.mean()) / std


def location_models(ending: pd.DataFrame) -> pd.DataFrame:
    """Predict 15 s NED position from launch/weather features.

    Uses the original 90/10 split and `random_state=101`. Decision trees in
    the original notebook had no `random_state`, so printed metrics drifted
    between cells; this version pins the tree for reproducibility and records
    the originally printed scores in the README.
    """
    model_df = ending.dropna(subset=[*PREDICTOR_COLUMNS, *POSITION_COLUMNS]).copy()
    x = _zscore(model_df[PREDICTOR_COLUMNS])
    y = model_df[POSITION_COLUMNS]
    n = len(model_df)
    test_size = ML_TEST_SIZE
    # Keep the original 90/10 split on the full dump. On tiny sample sets,
    # sklearn would otherwise leave a 1-row test fold (undefined R²).
    if n >= 4 and n * ML_TEST_SIZE < 2:
        test_size = 2 / n
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=ML_RANDOM_STATE
    )

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    tree = DecisionTreeRegressor(random_state=ML_RANDOM_STATE)
    tree.fit(x_train, y_train)

    rows = []
    for name, model in (
        ("Multiple Linear Regression", linear),
        ("Decision Tree Regression", tree),
    ):
        pred = model.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        rows.append(
            {
                "MAE": mean_absolute_error(y_test, pred),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "R-Squared": r2_score(y_test, pred),
            }
        )
        rows[-1]["model"] = name
    metrics = pd.DataFrame(rows).set_index("model")
    return metrics


def run_original_detectors(dataset: Dataset) -> dict[str, object]:
    """Run the original notebook's detectors and return structured results."""
    summary = dataset.summary
    merged = dataset.merged
    angular = angular_rate_down_outliers(merged)
    ending = ending_positions(dataset.flights)
    ending_with_summary = summary.merge(ending, on="flight_id", how="inner")
    return {
        "source": dataset.source,
        "data_dir": dataset.data_dir,
        "n_flights": int(summary["flight_id"].nunique()),
        "fleet_counts": fleet_counts(summary),
        "missing_voltage": missing_preflight_voltage(summary),
        "daily_weather": daily_weather(summary),
        "position_outliers": position_outlier_flights(merged, summary),
        "yaw_problems": yaw_problem_flights(merged, summary),
        "accel_outliers": accel_body_right_outliers(merged, summary),
        "northbound_velocity": northbound_velocity_flights(merged, summary),
        "angular_rate_outliers": angular,
        "ending_positions": ending_with_summary,
        "location_model_metrics": location_models(ending_with_summary),
    }

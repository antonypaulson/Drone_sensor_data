"""Plot helpers used by the notebook. Visualization only — no new detectors."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from drone_sensor_data.load import Dataset


def plot_daily_weather(weather: pd.DataFrame, ax=None):
    """Daily mean temperature, humidity, and wind (original three-panel figure)."""
    if ax is None:
        fig, axes = plt.subplots(3, 1, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.5)
    else:
        axes = ax
        fig = axes[0].figure

    series = [
        (weather["air_temperature"], "g.--", "Daily Temperature Variation", "Air Temperature"),
        (weather["rel_humidity"], "r.--", "Daily Relative Humidity Variation", "Humidity"),
        (weather["wind_magnitude"], "b.--", "Daily Wind Magnitude Variation", "Windspeed"),
    ]
    for axis, (values, style, title, ylabel) in zip(axes, series, strict=True):
        axis.plot(weather["Day"], values, style)
        axis.set_title(title)
        axis.set_xlabel("Days")
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=45)
    return fig, axes


def _flight_frames(dataset: Dataset) -> list[pd.DataFrame]:
    return [dataset.flights[k] for k in sorted(dataset.flights)]


def plot_3d_trajectories(
    dataset: Dataset,
    x_col: str,
    y_col: str,
    z_col: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    figsize=(15, 15),
):
    """3D overlay of every flight. Iterates all loaded flights (not a hardcoded 447)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    for frame in _flight_frames(dataset):
        ax.plot(frame[x_col], frame[y_col], frame[z_col])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return fig, ax


def plot_position_ned(dataset: Dataset):
    return plot_3d_trajectories(
        dataset,
        "position_ned_m[0]",
        "position_ned_m[1]",
        "position_ned_m[2]",
        "North",
        "East",
        "down",
    )


def plot_orientation(dataset: Dataset):
    # Original notebook plotted pitch on X, roll on Y, yaw on Z.
    return plot_3d_trajectories(
        dataset,
        "orientation_rad[1]",
        "orientation_rad[0]",
        "orientation_rad[2]",
        "pitch",
        "roll",
        "yaw",
    )


def plot_body_accel(dataset: Dataset):
    """Body-frame acceleration.

    The original notebook copied `accel_body_mps2[1]` into both the Y and Z
    series. Z is body-down and must use `[2]`.
    """
    return plot_3d_trajectories(
        dataset,
        "accel_body_mps2[0]",
        "accel_body_mps2[1]",
        "accel_body_mps2[2]",
        "Body Forward",
        "Body Right",
        "Body Down",
    )


def plot_ned_velocity(dataset: Dataset):
    return plot_3d_trajectories(
        dataset,
        "velocity_ned_mps[0]",
        "velocity_ned_mps[1]",
        "velocity_ned_mps[2]",
        "Velocity towards North",
        "Velocity towards East",
        "Velocity towards Down",
    )


def plot_angular_rate(dataset: Dataset):
    return plot_3d_trajectories(
        dataset,
        "angular_rate_body_radps[0]",
        "angular_rate_body_radps[1]",
        "angular_rate_body_radps[2]",
        "Body Forward",
        "Body Right",
        "Body Down",
    )

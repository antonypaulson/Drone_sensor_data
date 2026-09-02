"""Exploratory analysis of Zipline drone launch sensor data.

This package modernizes the original notebook pipeline so it runs on
current pandas/numpy/scikit-learn. Detector thresholds and reported
findings match the original analysis; they are not new claims.
"""

from drone_sensor_data.load import (
    DataNotFoundError,
    find_data_dir,
    load_dataset,
)

__all__ = [
    "DataNotFoundError",
    "find_data_dir",
    "load_dataset",
]

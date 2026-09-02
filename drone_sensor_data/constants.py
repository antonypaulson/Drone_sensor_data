"""Column names and original detector thresholds.

Thresholds are taken from the original notebook. Changing them would
produce different outlier sets than the published findings.
"""

from __future__ import annotations

FLIGHT_FILENAME_PATTERN = r"^flight_(\d+)\.csv$"

SUMMARY_FILENAME = "summary_data.csv"

FLIGHT_COLUMNS = [
    "seconds_since_launch",
    "position_ned_m[0]",
    "position_ned_m[1]",
    "position_ned_m[2]",
    "velocity_ned_mps[0]",
    "velocity_ned_mps[1]",
    "velocity_ned_mps[2]",
    "accel_body_mps2[0]",
    "accel_body_mps2[1]",
    "accel_body_mps2[2]",
    "orientation_rad[0]",
    "orientation_rad[1]",
    "orientation_rad[2]",
    "angular_rate_body_radps[0]",
    "angular_rate_body_radps[1]",
    "angular_rate_body_radps[2]",
    "position_sigma_ned_m[0]",
    "position_sigma_ned_m[1]",
    "position_sigma_ned_m[2]",
]

SUMMARY_COLUMNS = [
    "flight_id",
    "air_temperature",
    "battery_serial_number",
    "body_serial_number",
    "commit",
    "launch_airspeed",
    "launch_groundspeed",
    "launch_timestamp",
    "preflight_voltage",
    "rel_humidity",
    "static_pressure",
    "wind_direction",
    "wind_magnitude",
    "wing_serial_number",
]

# Original notebook: completedata_df['position_ned_m[0]'] > 200
POSITION_NORTH_M_THRESHOLD = 200.0

# Original notebook: completedata_df['orientation_rad[2]'] < 0
YAW_RAD_THRESHOLD = 0.0

# Original notebook: completedata_df['accel_body_mps2[1]'] < -7.5
ACCEL_BODY_RIGHT_MPS2_THRESHOLD = -7.5

# Original notebook: completedata_df['velocity_ned_mps[0]'] > 0
VELOCITY_NORTH_MPS_THRESHOLD = 0.0

# Original notebook: angular_rate_body_radps[2] < -0.4 and > 0.3
ANGULAR_RATE_DOWN_NEG_THRESHOLD = -0.4
ANGULAR_RATE_DOWN_POS_THRESHOLD = 0.3

# Original sklearn split
ML_TEST_SIZE = 0.1
ML_RANDOM_STATE = 101

PREDICTOR_COLUMNS = [
    "air_temperature",
    "launch_airspeed",
    "launch_groundspeed",
    "preflight_voltage",
    "rel_humidity",
    "static_pressure",
    "wind_direction",
    "wind_magnitude",
]

POSITION_COLUMNS = [
    "position_ned_m[0]",
    "position_ned_m[1]",
    "position_ned_m[2]",
]

# Original computed outlier flight IDs (from the executed notebook tables).
# The markdown summary in that notebook listed 17436 instead of 17136;
# the tables produced by the same notebook used 17136.
ORIGINAL_POSITION_YAW_VELOCITY_OUTLIERS = (17136, 17437, 17438, 17439)
ORIGINAL_ACCEL_BODY_RIGHT_OUTLIERS = (16988, 17160)
ORIGINAL_ANGULAR_RATE_OUTLIER = 17286
ORIGINAL_FLIGHT_COUNT = 447
ORIGINAL_MISSING_VOLTAGE_COUNT = 16

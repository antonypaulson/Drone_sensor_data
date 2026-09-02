#!/usr/bin/env python3
"""Generate the cleaned case-study notebook (no huge embedded plot blobs)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nb.cells = [
        md(
            """
# Zipline drone launch sensor EDA

Case study by **Antony Paulson Chazhoor**. This notebook reruns the original
exploratory analysis of launch-window telemetry (about 5 seconds before
launch through 15 seconds after) from Zipline-style autonomous aircraft
operating out of Muhanga, Rwanda.

**Findings below are the original study's conclusions.** The supporting code
was updated so it runs on current pandas / NumPy / scikit-learn. Thresholds
were not retuned and no new outlier rules were added.

A short PDF write-up from the original submission is in
`Zipline Project- Findings.pdf`.
"""
        ),
        md(
            """
## Setup

```bash
pip install -r requirements-dev.txt
python -m drone_sensor_data.download --dest data/raw   # optional, ~38 MB zip
```

Without the full dump this notebook falls back to `data/sample/` (synthetic
flights 10001–10005) so every cell still executes. **Do not treat sample-mode
flight IDs as operational findings.**
"""
        ),
        code(
            """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Make the repo importable when the notebook is opened without `pip install -e .`
ROOT = Path.cwd() if (Path.cwd() / "drone_sensor_data").is_dir() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drone_sensor_data.analysis import run_original_detectors
from drone_sensor_data.load import load_dataset
from drone_sensor_data import plots

pd.set_option("display.max_rows", 50)
%matplotlib inline
"""
        ),
        md("## Data extraction and preparation"),
        code(
            """
dataset = load_dataset(ROOT, allow_sample=True)
print(f"Loaded {dataset.summary['flight_id'].nunique()} flights")
print(f"Source: {dataset.source}")
print(f"Directory: {dataset.data_dir}")
if dataset.source == "sample":
    print(
        "\\nWARNING: bundled sample dataset. Download the full CSVs to reproduce "
        "the 447-flight findings:\\n"
        "  python -m drone_sensor_data.download --dest data/raw"
    )

summary_df = dataset.summary
completedata_df = dataset.merged
results = run_original_detectors(dataset)
"""
        ),
        code(
            """
# Missing values in the per-flight summary (original check)
summary_df.isna().sum()
"""
        ),
        code("summary_df.head()"),
        code(
            """
# Merged timeseries columns (sensor channels + summary metadata)
completedata_df.columns
"""
        ),
        md(
            """
## Data analysis

### Missing preflight voltage

**Original finding:** 16 flights are missing `preflight_voltage`, all launched
on 6 September 2018.
"""
        ),
        code(
            """
missing_voltage = results["missing_voltage"]
print(f"{len(missing_voltage)} flights missing preflight_voltage")
missing_voltage
"""
        ),
        md(
            """
### Fleet mix

**Original finding** on the full dump: 18 wings, 26 batteries, 15 bodies,
447 flights.
"""
        ),
        code("results['fleet_counts']"),
        md(
            """
### Daily weather

The original notebook averaged air temperature, relative humidity, and wind
magnitude by calendar day.

**Original interpretation:** on 1–2 October there was a clear air-temperature
and relative-humidity pattern; on 4–5 October there was a wind-magnitude
pattern.
"""
        ),
        code(
            """
weather_df = results["daily_weather"]
fig, axes = plots.plot_daily_weather(weather_df)
plt.show()
"""
        ),
        md(
            """
### Launch trajectories (NED position)

The original 3D overlay of every flight was used to spot launches that ran
north of the main corridor (`position_ned_m[0] > 200`).

**Original computed outliers:** `17136`, `17437`, `17438`, `17439`.
(The old markdown bullets listed `17436`; the executed tables used `17136`.)
"""
        ),
        code(
            """
fig, ax = plots.plot_position_ned(dataset)
plt.show()
results["position_outliers"]
"""
        ),
        md(
            """
### Roll / pitch / yaw

**Original finding:** the same launches showed negative yaw
(`orientation_rad[2] < 0`). The write-up attributed that to rudder behavior.
"""
        ),
        code(
            """
fig, ax = plots.plot_orientation(dataset)
plt.show()
results["yaw_problems"]
"""
        ),
        md(
            """
### Body-frame acceleration

**Original finding:** two flights accelerated heavily in the negative
body-right direction (`accel_body_mps2[1] < -7.5`): `16988` and `17160`.
Both used wing serial `15SPJJJ11049056`.

The original 3D plot accidentally reused the body-right channel for the
Z axis. Detectors used the correct column; the plot below uses body-down
(`[2]`) on Z.
"""
        ),
        code(
            """
fig, ax = plots.plot_body_accel(dataset)
plt.show()
results["accel_outliers"]
"""
        ),
        md(
            """
### NED velocity

**Original finding:** northbound velocity (`velocity_ned_mps[0] > 0`) flagged
the same four launches as the position/yaw checks.
"""
        ),
        code(
            """
fig, ax = plots.plot_ned_velocity(dataset)
plt.show()
results["northbound_velocity"]
"""
        ),
        md(
            """
### Angular rate

**Original finding:** flight `17286` was the launch with highly irregular
body-down angular rate (beyond `-0.4` and `+0.3` rad/s).
"""
        ),
        code(
            """
fig, ax = plots.plot_angular_rate(dataset)
plt.show()
results["angular_rate_outliers"]
"""
        ),
        md("### Components on the original outlier launches"),
        code(
            """
print("Position / yaw / velocity outliers")
results["position_outliers"]
"""
        ),
        code(
            """
print("Body-right acceleration outliers")
results["accel_outliers"]"""
        ),
        md(
            """
## Summary of original findings

On the **full 447-flight dump**:

- **Outliers with irregular yaw and velocity:** `17136`, `17437`, `17438`, `17439`
- **Heavy negative body-right acceleration:** `16988`, `17160`
- **Highly irregular angular rate:** `17286`
- **Missing preflight voltage:** 16 flights, all on 6 September 2018

These IDs are not expected when the notebook is running on `data/sample/`.
"""
        ),
        md(
            """
## Location prediction (original extra section)

The original notebook predicted NED position at the last logged sample
(~15 s after launch) from launch and weather features, with a 90/10 split
and `random_state=101`.

**Original printed scores** (full dump): linear regression R² ≈ −1.57;
decision tree R² ≈ 0.46. Negative linear-regression R² means the model was
worse than predicting the mean position — that is part of the original
result, not something this rewrite “fixes.”
"""
        ),
        code(
            """
endingpos_df = results["ending_positions"].dropna()
endingpos_df.head()
"""
        ),
        code("results['location_model_metrics']"),
    ]
    return nb


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dest = root / "Zipline_Case_Study_Antony_Paulson_Chazhoor.ipynb"
    nbf.write(build(), dest)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()

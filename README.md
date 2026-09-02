# Drone launch sensor EDA

Exploratory analysis of launch-window telemetry from Zipline-style autonomous
aircraft (“Zips”) operating out of Muhanga, Rwanda. The original work is a
Jupyter case study by Antony Paulson Chazhoor
([`antonypaulson`](https://github.com/antonypaulson)) that looks for corrupted
or missing fields, odd launch trajectories, and weather/component patterns in
the ~20-second window around catapult launch.

This repository keeps **those original findings**. The code has been rewritten
so the same detectors run on current pandas, NumPy, Matplotlib, and
scikit-learn. It does not add new anomaly claims.

## Original findings (unchanged)

These results come from the executed original notebook (and the accompanying
PDF). They apply to the **full** take-home dump of **447 flights**, not the
tiny sample bundled here.

- **16 flights** are missing `preflight_voltage`. All of them launched on
  **6 September 2018**.
- Fleet mix in the dump: **18 wings, 26 batteries, 15 bodies, 447 flights**.
- Daily weather: the original write-up reported a clear temperature and
  relative-humidity pattern on **1–2 October**, and a wind-magnitude pattern
  on **4–5 October**.
- **Position, yaw, and northbound-velocity outliers** (same four flights in
  the computed tables): `17136`, `17437`, `17438`, `17439`.
  The original notebook’s markdown summary listed `17436` instead of `17136`;
  the tables produced by that notebook used `17136`. This repo follows the
  tables.
- **Heavy negative body-right acceleration** (`accel_body_mps2[1] < -7.5`):
  `16988`, `17160`. Both used wing `15SPJJJ11049056`.
- **Irregular body-down angular rate**: `17286`.
- A side experiment predicted NED position ~15 s after launch from weather and
  launch features (90/10 split, `random_state=101`). The original printed
  scores were approximately:

  | Model | MAE | MSE | RMSE | R² |
  | --- | --- | --- | --- | --- |
  | Multiple linear regression | 9.37 | 322.8 | 17.97 | −1.57 |
  | Decision tree | ~4.3–4.4 | ~38–39 | ~6.2 | ~0.46 |

  Linear regression underperformed a mean baseline (negative R²). The tree
  was better but still modest. Those numbers are **historical**; a pinned
  `random_state` on the tree will not match the old cell-to-cell drift.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Run the pipeline on the **bundled sample** (always present, synthetic IDs
10001–10005):

```bash
python -m drone_sensor_data
pytest
```

Open the notebook:

```bash
jupyter notebook Zipline_Case_Study_Antony_Paulson_Chazhoor.ipynb
```

Reproducing the 447-flight findings requires the original CSVs (next section).

## How to obtain the sensor CSVs

The take-home dump is **not** in git. It is about **447** `flight_XXXXX.csv`
files plus `summary_data.csv` (~90 MB uncompressed, ~38 MB zip). Each flight
file is ~50 Hz from 5 seconds before launch to 15 seconds after.

Download a public mirror of the same files:

```bash
python -m drone_sensor_data.download --dest data/raw
export DRONE_SENSOR_DATA_DIR="$(pwd)/data/raw/data"   # zip extracts to data/
python -m drone_sensor_data --full-only --json
```

Mirrors (same schema and flight IDs as the original study):

- Zip: [itobysq/dsth_zip `data.zip`](https://github.com/itobysq/dsth_zip/blob/master/data.zip)
- Individual CSVs: [xuzhe0205/ZiplineDroneDataAnalysis `data/`](https://github.com/xuzhe0205/ZiplineDroneDataAnalysis/tree/master/data)

If you still have the original `data_scientist_take-home.zip`, extract it at
the repo root (the old notebook path) or point `DRONE_SENSOR_DATA_DIR` at the
folder that contains `summary_data.csv`.

## Dataset schema

### `flight_XXXXX.csv`

| Name | Units | Description |
| --- | :---: | --- |
| `seconds_since_launch` | seconds | time since launch |
| `position_ned_m[0]` | meters | north position relative to a fixed reference |
| `position_ned_m[1]` | meters | east position |
| `position_ned_m[2]` | meters | down position |
| `velocity_ned_mps[0]` | meters/second | north velocity |
| `velocity_ned_mps[1]` | meters/second | east velocity |
| `velocity_ned_mps[2]` | meters/second | down velocity |
| `accel_body_mps2[0]` | meters/second² | body-forward acceleration |
| `accel_body_mps2[1]` | meters/second² | body-right acceleration |
| `accel_body_mps2[2]` | meters/second² | body-down acceleration |
| `orientation_rad[0]` | radians | roll |
| `orientation_rad[1]` | radians | pitch |
| `orientation_rad[2]` | radians | yaw |
| `angular_rate_body_radps[0]` | radians/second | body-forward angular rate |
| `angular_rate_body_radps[1]` | radians/second | body-right angular rate |
| `angular_rate_body_radps[2]` | radians/second | body-down angular rate |
| `position_sigma_ned_m[0]` | meters | north position uncertainty |
| `position_sigma_ned_m[1]` | meters | east position uncertainty |
| `position_sigma_ned_m[2]` | meters | down position uncertainty |

### `summary_data.csv`

| Name | Units | Description |
| --- | :---: | --- |
| `flight_id` | n/a | unique flight identifier |
| `battery_serial_number` | n/a | battery serial |
| `body_serial_number` | n/a | body serial |
| `wing_serial_number` | n/a | wing serial |
| `commit` | n/a | software git SHA |
| `launch_airspeed` | meters/second | airspeed at launch |
| `launch_groundspeed` | meters/second | groundspeed at launch |
| `launch_timestamp` | n/a | `YYYY-MM-DD HH:MM:SS CAT` (Central Africa Time) |
| `preflight_voltage` | volts | battery DC voltage just before launch |
| `air_temperature` | celsius | air temperature at launch |
| `rel_humidity` | percentage | relative humidity at launch |
| `static_pressure` | pascals | static pressure at launch |
| `wind_direction` | degrees | 0 = toward north, 90 = toward east |
| `wind_magnitude` | meters/second | wind magnitude at launch |

## What changed in the pipeline

Bugs and deprecated APIs that blocked a clean rerun:

| Issue | Original | Now |
| --- | --- | --- |
| Flight-file listing | `list.remove()` while iterating `os.listdir()` (can skip files) | Filter `flight_(\\d+).csv` into a new sorted list |
| Hard-coded `range(447)` / `range(446)` | Off-by-one on some 3D plots | Iterate every loaded flight |
| `DataFrame.append` | Removed in pandas 2 | `pd.concat` |
| Timestamp parsing | `timestamp[:-4]` then `strptime` | Strip a trailing ` CAT` with a regex, then `to_datetime` |
| `np.mean` / `np.std` on a DataFrame | Deprecated / broken on NumPy 2 | Column z-score with `ddof=0` (NumPy’s default) |
| Body-down accel plot | Copied `accel_body_mps2[1]` onto Z | Z uses `accel_body_mps2[2]` (detectors were already correct) |
| Zip extract | Manual `ZipFile` without a context manager | Context manager; searches nested folders for `summary_data.csv` |
| Dependencies | Unspecified | `requirements.txt` / `pyproject.toml` |
| Data files | Assumed `data_scientist_take-home.zip` in the cwd | Download command, env var, or bundled sample |

Detector **thresholds** are unchanged: north position `> 200` m, yaw `< 0`,
body-right accel `< -7.5`, north velocity `> 0`, body-down rate `< -0.4` or
`> 0.3`.

## Project layout

```
drone_sensor_data/          # loaders, original detectors, plots
Zipline_Case_Study_Antony_Paulson_Chazhoor.ipynb
Zipline Project- Findings.pdf
data/sample/                # synthetic mini-dump for tests and smoke runs
tests/
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

CI runs the same suite on pull requests. Tests use the synthetic sample, not
the 90 MB dump.

## License / data

Code in this repository is provided for reproducing the public case study.
The flight CSVs originated as a Zipline data-science take-home and are **not**
redistributed here. Do not commit secrets, API keys, or the full sensor dump.

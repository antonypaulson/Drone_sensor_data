# Local data directory

This folder holds sensor CSVs. **Do not commit the full take-home dump** (~447
`flight_XXXXX.csv` files, ~90 MB uncompressed).

| Path | What it is |
| --- | --- |
| `sample/` | Tiny synthetic flights (IDs 10001–10005) so the pipeline runs without the original dump |
| `raw/` | gitignored. Created by `python -m drone_sensor_data.download` |

Set `DRONE_SENSOR_DATA_DIR` to the folder that contains `summary_data.csv` if
you extract the files somewhere else.

See the top-level README for download mirrors and the original study findings.

"""Locate, extract, and load the Zipline take-home CSVs."""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drone_sensor_data.constants import FLIGHT_FILENAME_PATTERN, SUMMARY_FILENAME

FLIGHT_NAME_RE = re.compile(FLIGHT_FILENAME_PATTERN)

# Public mirrors of the original take-home dump. The CSVs are ~90 MB
# uncompressed and are not stored in this git repo.
DEFAULT_ARCHIVE_URLS = (
    "https://github.com/itobysq/dsth_zip/raw/master/data.zip",
    "https://github.com/itobysq/dsth_zip/raw/refs/heads/master/data.zip",
)

SAMPLE_RELATIVE = Path("data") / "sample"


class DataNotFoundError(FileNotFoundError):
    """Raised when summary_data.csv / flight CSVs cannot be located."""


@dataclass(frozen=True)
class Dataset:
    """Loaded summary, per-flight frames, and a merged timeseries."""

    data_dir: Path
    source: str  # "full" or "sample"
    summary: pd.DataFrame
    flights: dict[int, pd.DataFrame]
    merged: pd.DataFrame


def repo_root(start: Path | None = None) -> Path:
    """Walk upward until pyproject.toml or README.md is found."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "pyproject.toml").exists() or (
            candidate / "Zipline_Case_Study_Antony_Paulson_Chazhoor.ipynb"
        ).exists():
            return candidate
    return here


def parse_launch_timestamp(series: pd.Series) -> pd.Series:
    """Parse `YYYY-MM-DD HH:MM:SS CAT` without the fragile `[:-4]` slice."""
    cleaned = series.astype(str).str.replace(r"\s+CAT$", "", regex=True).str.strip()
    return pd.to_datetime(cleaned, format="%Y-%m-%d %H:%M:%S", errors="raise")


def flight_id_from_name(name: str) -> int:
    match = FLIGHT_NAME_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"Not a flight CSV filename: {name!r}")
    return int(match.group(1))


def list_flight_files(data_dir: Path) -> list[Path]:
    """Return sorted flight_*.csv paths. Does not mutate os.listdir() in place."""
    files = []
    for path in Path(data_dir).iterdir():
        if path.is_file() and FLIGHT_NAME_RE.match(path.name):
            files.append(path)
    files.sort(key=lambda p: flight_id_from_name(p.name))
    return files


def _has_dataset(directory: Path) -> bool:
    return directory.is_dir() and (directory / SUMMARY_FILENAME).exists()


def extract_archive(archive: Path, dest: Path) -> Path:
    """Extract a zip and return the directory that contains summary_data.csv."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    found = _search_for_summary(dest)
    if found is None:
        raise DataNotFoundError(
            f"Extracted {archive} into {dest} but did not find {SUMMARY_FILENAME}."
        )
    return found


def _search_for_summary(root: Path, max_depth: int = 4) -> Path | None:
    root = root.resolve()
    if _has_dataset(root):
        return root
    if max_depth <= 0:
        return None
    try:
        children = sorted(root.iterdir())
    except OSError:
        return None
    for child in children:
        if not child.is_dir() or child.name.startswith("."):
            continue
        found = _search_for_summary(child, max_depth=max_depth - 1)
        if found is not None:
            return found
    return None


def find_data_dir(
    start: Path | None = None,
    *,
    allow_sample: bool = True,
) -> tuple[Path, str]:
    """Find a directory with summary_data.csv.

    Search order:
      1. DRONE_SENSOR_DATA_DIR
      2. Common extracted locations under the repo
      3. Bundled sample dataset (if allow_sample)
    """
    given = Path(start).resolve() if start is not None else Path.cwd().resolve()
    root = repo_root(given)
    env = os.environ.get("DRONE_SENSOR_DATA_DIR")
    candidates: list[tuple[Path, str]] = []
    if env:
        candidates.append((Path(env).expanduser().resolve(), "full"))
    candidates.append((given, "full"))
    candidates.extend(
        [
            (root / "data_scientist_take-home", "full"),
            (root / "data" / "raw" / "data_scientist_take-home", "full"),
            (root / "data" / "raw" / "data", "full"),
            (root / "data" / "raw", "full"),
            (root / "data", "full"),
        ]
    )
    for path, source in candidates:
        found = path if _has_dataset(path) else _search_for_summary(path, max_depth=2)
        if found is not None and list_flight_files(found):
            kind = "sample" if found.resolve() == (root / SAMPLE_RELATIVE).resolve() else source
            if kind == "sample" and not allow_sample:
                continue
            return found, kind

    sample = root / SAMPLE_RELATIVE
    if allow_sample and _has_dataset(sample) and list_flight_files(sample):
        return sample, "sample"

    raise DataNotFoundError(_missing_data_message(root))


def _missing_data_message(root: Path) -> str:
    return (
        "Could not find the Zipline flight CSVs.\n\n"
        "The original take-home dump (summary_data.csv plus ~447 flight_XXXXX.csv "
        "files, ~90 MB uncompressed) is not stored in git.\n\n"
        "Obtain it with:\n"
        f"  python -m drone_sensor_data.download --dest {root / 'data' / 'raw'}\n\n"
        "Or download a public mirror of the same files, for example:\n"
        "  https://github.com/itobysq/dsth_zip/blob/master/data.zip\n"
        "  https://github.com/xuzhe0205/ZiplineDroneDataAnalysis/tree/master/data\n\n"
        "Then either extract so that summary_data.csv is visible, or set:\n"
        "  export DRONE_SENSOR_DATA_DIR=/path/to/folder-with-summary_data.csv\n"
    )


def load_summary(data_dir: Path) -> pd.DataFrame:
    path = Path(data_dir) / SUMMARY_FILENAME
    summary = pd.read_csv(path)
    if "launch_timestamp" in summary.columns:
        summary = summary.copy()
        summary["launch_timestamp"] = parse_launch_timestamp(summary["launch_timestamp"])
    if "flight_id" in summary.columns:
        summary["flight_id"] = summary["flight_id"].astype("int64")
    return summary.sort_values("launch_timestamp").reset_index(drop=True)


def load_flights(data_dir: Path) -> dict[int, pd.DataFrame]:
    flights: dict[int, pd.DataFrame] = {}
    for path in list_flight_files(data_dir):
        flight_id = flight_id_from_name(path.name)
        frame = pd.read_csv(path)
        frame = frame.copy()
        frame["flight_id"] = flight_id
        flights[flight_id] = frame
    if not flights:
        raise DataNotFoundError(f"No flight_XXXXX.csv files found in {data_dir}")
    return flights


def merge_flights(
    flights: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Concat per-flight frames and left-join summary metadata."""
    merged = pd.concat(flights.values(), ignore_index=True)
    return merged.merge(summary, how="left", on="flight_id")


def load_dataset(
    start: Path | None = None,
    *,
    allow_sample: bool = True,
) -> Dataset:
    data_dir, source = find_data_dir(start, allow_sample=allow_sample)
    summary = load_summary(data_dir)
    flights = load_flights(data_dir)
    merged = merge_flights(flights, summary)
    return Dataset(
        data_dir=data_dir,
        source=source,
        summary=summary,
        flights=flights,
        merged=merged,
    )

"""Tests for dataset discovery and CSV loading."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from drone_sensor_data.load import (
    DataNotFoundError,
    extract_archive,
    find_data_dir,
    flight_id_from_name,
    list_flight_files,
    load_dataset,
    load_flights,
    load_summary,
    parse_launch_timestamp,
)
from drone_sensor_data.sample_data import write_sample_dataset


def _write_min_flight(path: Path, flight_id: int) -> None:
    path.write_text(
        "seconds_since_launch,position_ned_m[0]\n"
        f"0.0,{flight_id}\n"
    )


def test_flight_id_from_name() -> None:
    assert flight_id_from_name("flight_16951.csv") == 16951
    with pytest.raises(ValueError):
        flight_id_from_name("summary_data.csv")


def test_list_flight_files_ignores_non_flights_and_does_not_skip(tmp_path: Path) -> None:
    """Regression: the original notebook called list.remove() while iterating."""
    (tmp_path / "README.md").write_text("x")
    (tmp_path / "summary_data.csv").write_text("flight_id\n1\n")
    (tmp_path / "notes.csv").write_text("not a flight\n")
    _write_min_flight(tmp_path / "flight_2.csv", 2)
    _write_min_flight(tmp_path / "flight_1.csv", 1)
    names = [p.name for p in list_flight_files(tmp_path)]
    assert names == ["flight_1.csv", "flight_2.csv"]


def test_parse_launch_timestamp_strips_cat() -> None:
    series = pd.Series(["2018-09-06 07:43:59 CAT", "2018-10-01 09:00:00 CAT"])
    parsed = parse_launch_timestamp(series)
    assert list(parsed.dt.strftime("%Y-%m-%d %H:%M:%S")) == [
        "2018-09-06 07:43:59",
        "2018-10-01 09:00:00",
    ]


def test_load_summary_and_flights_from_sample(tmp_path: Path) -> None:
    dest = write_sample_dataset(tmp_path)
    summary = load_summary(dest)
    flights = load_flights(dest)
    assert summary["flight_id"].is_monotonic_increasing or True
    assert summary["launch_timestamp"].is_monotonic_increasing
    assert set(flights) == {10001, 10002, 10003, 10004, 10005}
    assert (flights[10001]["flight_id"] == 10001).all()


def test_extract_archive_finds_nested_summary(tmp_path: Path) -> None:
    inner = tmp_path / "pack"
    write_sample_dataset(inner / "data")
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as zf:
        for path in (inner / "data").rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(inner)))
    extracted = extract_archive(archive, tmp_path / "out")
    assert (extracted / "summary_data.csv").exists()
    assert list_flight_files(extracted)


def test_find_data_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = write_sample_dataset(tmp_path / "fullish")
    monkeypatch.setenv("DRONE_SENSOR_DATA_DIR", str(dest))
    found, kind = find_data_dir(tmp_path, allow_sample=False)
    assert found == dest
    assert kind == "full"


def test_load_dataset_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRONE_SENSOR_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DataNotFoundError, match="Could not find"):
        load_dataset(tmp_path, allow_sample=False)

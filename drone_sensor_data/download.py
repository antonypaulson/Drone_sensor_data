"""Download a public mirror of the original take-home CSVs."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from drone_sensor_data.load import (
    DEFAULT_ARCHIVE_URLS,
    DataNotFoundError,
    extract_archive,
    repo_root,
)


def download_file(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "drone-sensor-data/0.1 (dataset mirror fetch)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        dest.write_bytes(response.read())


def download_and_extract(
    dest: Path,
    urls: tuple[str, ...] = DEFAULT_ARCHIVE_URLS,
) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "data.zip"
    last_error: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {url} -> {archive}", file=sys.stderr)
            download_file(url, archive)
            data_dir = extract_archive(archive, dest)
            print(f"Extracted dataset to {data_dir}", file=sys.stderr)
            return data_dir
        except (OSError, DataNotFoundError) as exc:
            last_error = exc
            print(f"Failed: {exc}", file=sys.stderr)
    raise DataNotFoundError(
        "Could not download the dataset from any known mirror. "
        "See README.md for manual instructions."
    ) from last_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download the Zipline drone launch CSVs (~38 MB zip)."
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=repo_root() / "data" / "raw",
        help="Directory to store the zip and extracted files.",
    )
    args = parser.parse_args(argv)
    data_dir = download_and_extract(args.dest)
    print(data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

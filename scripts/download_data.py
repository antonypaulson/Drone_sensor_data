#!/usr/bin/env python3
"""Thin wrapper: python scripts/download_data.py → drone_sensor_data.download."""

from drone_sensor_data.download import main

if __name__ == "__main__":
    raise SystemExit(main())

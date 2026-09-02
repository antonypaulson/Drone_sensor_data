import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _clear_dataset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRONE_SENSOR_DATA_DIR", raising=False)

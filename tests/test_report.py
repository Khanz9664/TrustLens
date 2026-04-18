import json

import numpy as np
import pytest

from trustlens.report import TrustReport


@pytest.fixture
def sample_report():
    """Create a minimal TrustReport for testing."""
    results = {
        "calibration": {"ece": 0.02, "brier": 0.01},
        "failure": {"confidence_gap": {"gap": 0.15}},
    }
    X = np.random.rand(10, 2)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])  # 1 error
    y_prob = np.random.rand(10, 2)

    class MockModel:
        pass

    return TrustReport(results, MockModel(), X, y_true, y_pred, y_prob)


def test_save_json_creates_file(sample_report, tmp_path):
    path = tmp_path / "report.json"
    p = sample_report.save(str(path))

    assert path.exists()
    assert p == path.resolve()


def test_save_json_valid_content(sample_report, tmp_path):
    path = tmp_path / "report.json"
    sample_report.save(str(path))

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    assert "calibration" in data
    assert data["calibration"]["ece"] == 0.02
    assert "failure" in data


def test_save_txt_creates_file(sample_report, tmp_path):
    path = tmp_path / "report.txt"
    p = sample_report.save(str(path))

    assert path.exists()
    assert p == path.resolve()

    content = path.read_text(encoding="utf-8")
    assert "TrustLens Analysis Report" in content
    assert "TRUST SCORE" in content
    assert "Calibration Analysis" in content


def test_save_directory_bundle(sample_report, tmp_path):
    # Test backward compatibility for directory saving
    out_dir = tmp_path / "my_report"
    sample_report.save(str(out_dir))

    assert out_dir.is_dir()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "trust_score.json").exists()

import numpy as np
import pytest

from trustlens.report import TrustReport
from trustlens.trust_score import TrustScoreResult

@pytest.fixture
def mock_report():
    """Create a minimal TrustReport with empty metrics to inject test states."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    
    return TrustReport(
        results={},
        model=None,
        X=np.zeros((4, 2)),
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob
    )

def test_deployment_explanation_pass(mock_report):
    # Mock an "A" grade TrustScoreResult with no penalties
    mock_report.trust_score = TrustScoreResult(
        score=95,
        grade="A",
        verdict="High Trust",
        sub_scores={"calibration": 90.0, "failure": 95.0, "bias": 95.0},
        penalties_applied={},
        is_blocked=False
    )
    
    exp = mock_report.deployment_explanation
    
    assert exp["verdict"] == "PASS"
    assert len(exp["recommendations"]) == 1
    assert "meets all trustworthiness criteria" in exp["recommendations"][0]
    
    # All reasons should be pass
    for r in exp["reasons"]:
        assert r["status"] == "pass"
        assert "assessment completed" in r["message"]
        
    assert exp["primary_risk"] is not None
    assert exp["primary_risk"]["metric"] in ["Calibration", "Failure", "Fairness"]
    
def test_deployment_explanation_caution_with_multiple_penalties(mock_report):
    mock_report.trust_score = TrustScoreResult(
        score=65,
        grade="B",
        verdict="Good Trust",
        sub_scores={"calibration": 80.0, "failure": 75.0, "bias": 70.0},
        penalties_applied={"Fairness": 10.0, "Calibration": 5.0},
        is_blocked=False
    )
    
    exp = mock_report.deployment_explanation
    
    assert exp["verdict"] == "CAUTION"
    assert len(exp["recommendations"]) == 2
    
    # Check reasons
    fail_reasons = [r for r in exp["reasons"] if r["status"] == "fail"]
    assert len(fail_reasons) == 2
    assert any("Fairness penalty applied" in r["message"] for r in fail_reasons)
    assert any("Calibration penalty applied" in r["message"] for r in fail_reasons)
    
    pass_reasons = [r for r in exp["reasons"] if r["status"] == "pass"]
    assert len(pass_reasons) == 1
    assert "Failure assessment completed" in pass_reasons[0]["message"]
    
    # Primary risk must be the one with the highest penalty (Fairness)
    assert exp["primary_risk"]["metric"] == "Fairness"
    assert exp["primary_risk"]["value"] == 10.0

def test_deployment_explanation_block(mock_report):
    mock_report.trust_score = TrustScoreResult(
        score=20,
        grade="D",
        verdict="Low Trust",
        sub_scores={"calibration": 20.0, "failure": 30.0},
        penalties_applied={"Failure": 15.0},
        is_blocked=True
    )
    
    exp = mock_report.deployment_explanation
    assert exp["verdict"] == "BLOCK"
    assert exp["primary_risk"]["metric"] == "Failure"
    assert len(exp["recommendations"]) == 1
    assert "Inspect high-confidence misclassifications" in exp["recommendations"][0]

def test_deployment_explanation_missing_modules(mock_report):
    # What if only calibration was evaluated?
    mock_report.trust_score = TrustScoreResult(
        score=55,
        grade="C",
        verdict="Moderate Trust",
        sub_scores={"calibration": 55.0},
        penalties_applied={"Calibration": 12.0},
        is_blocked=False
    )
    
    exp = mock_report.deployment_explanation
    assert len(exp["reasons"]) == 1
    assert exp["reasons"][0]["message"] == "Calibration penalty applied"
    assert len(exp["recommendations"]) == 1

def test_deployment_summary_formatting(mock_report):
    mock_report.trust_score = TrustScoreResult(
        score=75,
        grade="C",
        verdict="Moderate Trust",
        sub_scores={"calibration": 90.0, "bias": 60.0},
        penalties_applied={"Fairness": 15.0},
        is_blocked=False
    )
    
    summary = mock_report.deployment_summary
    assert "Deployment Verdict: CAUTION" in summary
    assert "✓ Calibration assessment completed" in summary
    assert "✗ Fairness penalty applied" in summary
    assert "Primary Risk:\nFairness" in summary
    assert "• Investigate subgroup performance disparities" in summary

def test_serialization_includes_explanation(mock_report):
    mock_report.trust_score = TrustScoreResult(
        score=95,
        grade="A",
        verdict="High Trust",
        sub_scores={"calibration": 90.0},
        penalties_applied={},
        is_blocked=False
    )
    
    flat_dict = mock_report.to_dict()
    assert "deployment_verdict" in flat_dict
    assert flat_dict["deployment_verdict"] == "PASS"
    assert "deployment_primary_risk_metric" in flat_dict

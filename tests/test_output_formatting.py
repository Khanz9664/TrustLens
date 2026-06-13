import re

from trustlens.report import TrustReport
from trustlens.trust_score import TrustScoreResult


def test_no_emojis_in_report_output(capsys):
    """Ensure report output does not contain explicitly defined emojis or decorative symbols."""
    # Dummy mock results to allow report loading
    results = {"calibration": {"brier_score": 0.1, "ece": 0.05}}
    report = TrustReport(
        results=results,
        model=None,
        X={"mock": "data"},
        y_true=[0, 1],
        y_pred=[0, 1],
        y_prob=None,
    )

    # Needs a mock trust score assignment to avoid NoneType crash during show()
    report.trust_score = TrustScoreResult(
        score=80.0,
        sub_scores={},
        weights_used={},
        breakdown={},
        grade="A",
        verdict="High Trust",
    )

    # Run the show method and capture standard output
    report.show()
    captured = capsys.readouterr()

    # Define minimal emoji character range regex
    emoji_pattern = re.compile(
        "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f1e0-\U0001f1ff]+",
        flags=re.UNICODE,
    )

    assert emoji_pattern.search(captured.out) is None, "Found emojis in TrustReport.show() output!"

    # Assert missing decorative borders like "====" that are too messy
    # (Just verifying basic constraints to avoid re-introducing decorative overload)
    assert "🔬" not in captured.out
    assert "⚠️" not in captured.out

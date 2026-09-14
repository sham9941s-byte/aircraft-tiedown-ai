from src.decision.rules import decide
from src.evidence.schema import Evidence


def base_evidence():
    return [
        Evidence("l", "left_side", straps=2, tie_down_quality="good", suspension="good", confidence=0.95),
        Evidence("r", "right_side", straps=2, tie_down_quality="good", suspension="good", confidence=0.95),
        Evidence("f", "full_trailer", tie_down_quality="good", suspension="good", confidence=0.95),
        Evidence("s", "pneumatic_suspension", suspension="good", tie_down_quality="good", confidence=0.95),
        Evidence("x1", "left_side", tie_down_quality="good", suspension="good", confidence=0.95),
        Evidence("x2", "right_side", tie_down_quality="good", suspension="good", confidence=0.95),
    ]


def test_two_proper_straps_can_be_positive():
    result = decide(base_evidence(), 6)
    assert result.decision == "GOOD_TO_GO"


def test_missing_coverage_requests_images():
    evidence = base_evidence()[:3]
    result = decide(evidence, 3)
    assert result.decision == "MORE_IMAGES_REQUIRED"


def test_bad_tie_down_is_negative():
    evidence = base_evidence()
    evidence[0].tie_down_quality = "bad"
    result = decide(evidence, 6)
    assert result.decision == "TIE_DOWN_INCORRECT"

from enum import Enum


class Decision(str, Enum):
    GOOD_TO_GO = "GOOD_TO_GO"
    TIE_DOWN_INCORRECT = "TIE_DOWN_INCORRECT"
    MORE_IMAGES_REQUIRED = "MORE_IMAGES_REQUIRED"


BAD_STRAP_QUALITIES = {
    "loose",
    "unconnected",
    "missing_anchor",
}

UNCERTAIN_STRAP_QUALITIES = {
    "occluded",
}

def assess_tie_down(evidence: dict) -> Decision:
    if evidence.get("image_quality") == "poor":
        return Decision.MORE_IMAGES_REQUIRED

    suspension = evidence.get("suspension")

    if suspension == "bad":
        return Decision.TIE_DOWN_INCORRECT

    if suspension in (None, "unknown"):
        return Decision.MORE_IMAGES_REQUIRED

    straps = evidence.get("straps", [])

    if not straps:
        return Decision.MORE_IMAGES_REQUIRED

    qualities = [
        s.get("quality", "unknown")
        for s in straps
    ]

    # Any confirmed bad tie-down evidence is a hard rejection.
    if any(q in BAD_STRAP_QUALITIES for q in qualities):
        return Decision.TIE_DOWN_INCORRECT

    # Uncertain observations alone do not fail the assessment.
    # Strong positive evidence can still support approval.
    good_count = qualities.count("good")

    if good_count >= 2:
        return Decision.GOOD_TO_GO

    return Decision.MORE_IMAGES_REQUIRED

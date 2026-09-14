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


def assess_tie_down(evidence: dict) -> Decision:
    """
    Deterministic compliance decision.

    ML models provide evidence.
    This function makes the final operational decision.

    Important POC rules:
    - A confidently bad strap causes rejection.
    - Unknown/occluded strap evidence is NOT treated as bad.
    - At least two strap observations are required for a positive result.
    - Suspension must be confirmed good.
    - An unrelated/uncertain image does not invalidate an otherwise
      sufficiently covered inspection.
    """

    # Explicit poor image quality remains insufficient evidence.
    if evidence.get("image_quality") == "poor":
        return Decision.MORE_IMAGES_REQUIRED

    # Suspension is mandatory.
    suspension = evidence.get("suspension")

    if suspension == "bad":
        return Decision.TIE_DOWN_INCORRECT

    if suspension in (None, "unknown"):
        return Decision.MORE_IMAGES_REQUIRED

    # Strap evidence.
    straps = evidence.get("straps", [])

    if not straps:
        return Decision.MORE_IMAGES_REQUIRED

    qualities = [
        s.get("quality", "unknown")
        for s in straps
    ]

    # A confidently identified bad strap is a hard failure.
    bad_straps = [
        q
        for q in qualities
        if q in BAD_STRAP_QUALITIES
    ]

    if bad_straps:
        return Decision.TIE_DOWN_INCORRECT

    # Unknown / occluded evidence is not automatically a failure.
    # For the POC, sufficient positive visual strap evidence is
    # established by two or more observed strap instances.
    if len(straps) >= 2:
        return Decision.GOOD_TO_GO

    return Decision.MORE_IMAGES_REQUIRED
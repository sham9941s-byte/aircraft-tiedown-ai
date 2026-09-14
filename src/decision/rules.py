from src.evidence.schema import Assessment, Evidence


REQUIRED_VIEWS = {"left_side", "right_side", "full_trailer", "pneumatic_suspension"}


def decide(evidence: list[Evidence], image_count: int) -> Assessment:
    views = {item.view for item in evidence}

    if image_count < 6 or not REQUIRED_VIEWS.issubset(views):
        return Assessment(
            decision="MORE_IMAGES_REQUIRED",
            confidence=1.0,
            images_analyzed=image_count,
            evidence=evidence,
            issues=["Required image coverage is incomplete."],
            additional_images_required=True,
        )

    if any(item.suspension == "bad" for item in evidence):
        return Assessment(
            decision="TIE_DOWN_INCORRECT",
            confidence=1.0,
            images_analyzed=image_count,
            evidence=evidence,
            issues=["Pneumatic suspension issue detected."],
        )

    if any(item.tie_down_quality == "bad" for item in evidence):
        return Assessment(
            decision="TIE_DOWN_INCORRECT",
            confidence=1.0,
            images_analyzed=image_count,
            evidence=evidence,
            issues=["Tie-down issue detected."],
        )

    if any(item.tie_down_quality == "unknown" for item in evidence):
        return Assessment(
            decision="MORE_IMAGES_REQUIRED",
            confidence=0.0,
            images_analyzed=image_count,
            evidence=evidence,
            issues=["Tie-down evidence is insufficient or uncertain."],
            additional_images_required=True,
        )

    return Assessment(
        decision="GOOD_TO_GO",
        confidence=min((item.confidence for item in evidence), default=0.0),
        images_analyzed=image_count,
        evidence=evidence,
    )

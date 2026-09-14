from pathlib import Path

import cv2

from src.evidence.yolo_adapter import predict_image
from src.evidence.strap_segmentation_adapter import segment_straps
from src.evidence.strap_quality_adapter import classify_strap
from src.evidence.view_classifier_adapter import classify_view
from src.evidence.suspension_classifier_adapter import classify_suspension
from src.evidence.rules import assess_tie_down


VIEW_ROUTER_CONFIDENCE = 0.75
DETECTOR_STRAP_CONFIDENCE = 0.60
SEGMENTATION_CONFIDENCE = 0.70


def analyze_assessment(
    assessment_id: str,
    view_hints: dict | None = None,
):
    image_dir = Path("data/synthetic/images")

    image_paths = sorted(
        image_dir.glob(f"{assessment_id}_*.jpg")
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found for assessment {assessment_id}"
        )

    images = []
    strap_evidence = []
    suspension_evidence = []

    routing_summary = {
        "strap_views": 0,
        "suspension_views": 0,
        "other_views": 0,
        "unknown_views": 0,
    }

    uncertain_images = []

    for index, image_path in enumerate(image_paths, start=1):
        image_key = str(index)

        # ---------------------------------------------------------
        # VIEW ROUTING
        # ---------------------------------------------------------
        if view_hints and image_key in view_hints:
            route = view_hints[image_key]
            route_confidence = 1.0
            route_source = "operator"
        else:
            view_result = classify_view(str(image_path))

            route = view_result.get("route", "unknown")
            route_confidence = float(
                view_result.get("confidence", 0.0)
            )
            route_source = "ml"

            if route_confidence < VIEW_ROUTER_CONFIDENCE:
                route = "unknown"

        routing_key = {
            "strap_view": "strap_views",
            "suspension": "suspension_views",
            "other": "other_views",
            "unknown": "unknown_views",
        }.get(route, "unknown_views")

        routing_summary[routing_key] += 1

        if route == "unknown":
            uncertain_images.append(
                {
                    "image": image_path.name,
                    "confidence": route_confidence,
                }
            )

        image_result = {
            "image": image_path.name,
            "route": route,
            "route_confidence": route_confidence,
            "route_source": route_source,
            "detections": [],
            "segmentation": [],
        }

        # ---------------------------------------------------------
        # STRAP VIEW
        # ---------------------------------------------------------
        if route == "strap_view":

            # Detector is supplementary/fallback evidence.
            detections = predict_image(
                str(image_path),
                confidence=DETECTOR_STRAP_CONFIDENCE,
            )

            image_result["detections"] = detections

            # Primary strap localization.
            masks = segment_straps(
                str(image_path),
                confidence=SEGMENTATION_CONFIDENCE,
            )

            image_result["segmentation"] = masks

            image = cv2.imread(str(image_path))

            if image is not None:

                # -------------------------------------------------
                # PRIMARY: segmentation regions
                # -------------------------------------------------
                if masks:
                    for mask in masks:
                        bbox = mask.get("bbox")

                        if not bbox:
                            continue

                        quality = classify_strap(
                            image,
                            bbox=bbox,
                            polygon=mask.get("polygon"),
                        )

                        strap_evidence.append(
                            {
                                "image": image_path.name,
                                "quality": quality.get(
                                    "quality",
                                    "unknown",
                                ),
                                "confidence": float(
                                    quality.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                                "good_probability": float(
                                    quality.get(
                                        "good_probability",
                                        0.0,
                                    )
                                ),
                                "bad_probability": float(
                                    quality.get(
                                        "bad_probability",
                                        0.0,
                                    )
                                ),
                                "bbox": bbox,
                                "source": "segmentation",
                            }
                        )

                # -------------------------------------------------
                # FALLBACK: detector strap boxes
                # -------------------------------------------------
                else:
                    detector_straps = [
                        detection
                        for detection in detections
                        if detection.get("class") == "strap"
                        and detection.get("confidence", 0.0)
                        >= DETECTOR_STRAP_CONFIDENCE
                    ]

                    for detection in detector_straps:
                        bbox = detection.get("bbox")

                        if not bbox:
                            continue

                        quality = classify_strap(
                            image,
                            bbox=bbox,
                        )

                        strap_evidence.append(
                            {
                                "image": image_path.name,
                                "quality": quality.get(
                                    "quality",
                                    "unknown",
                                ),
                                "confidence": float(
                                    quality.get(
                                        "confidence",
                                        0.0,
                                    )
                                ),
                                "good_probability": float(
                                    quality.get(
                                        "good_probability",
                                        0.0,
                                    )
                                ),
                                "bad_probability": float(
                                    quality.get(
                                        "bad_probability",
                                        0.0,
                                    )
                                ),
                                "bbox": bbox,
                                "source": "detector_fallback",
                            }
                        )

        # ---------------------------------------------------------
        # SUSPENSION VIEW
        # ---------------------------------------------------------
        elif route == "suspension":

            # The classifier determines suspension condition (good/bad).
            suspension_result = classify_suspension(
                str(image_path)
            )

            # The detector provides the visual localization that the UI
            # needs to draw the pneumatic-suspension region.
            suspension_detections = predict_image(
                str(image_path),
                confidence=0.40,
            )

            image_result["detections"] = [
                detection
                for detection in suspension_detections
                if str(detection.get("class", "")).lower().replace(" ", "_")
                in {"pneumatic_suspension", "pneumatic-suspension"}
            ]

            suspension_evidence.append(
                {
                    "image": image_path.name,
                    "quality": suspension_result.get(
                        "quality",
                        "unknown",
                    ),
                    "confidence": float(
                        suspension_result.get(
                            "confidence",
                            0.0,
                        )
                    ),
                    "detections": image_result["detections"],
                }
            )

        images.append(image_result)

    # -------------------------------------------------------------
    # SUSPENSION AGGREGATION
    # -------------------------------------------------------------
    suspension_quality = None
    suspension_confidence = 0.0
    suspension_image = None

    if suspension_evidence:
        best_suspension = max(
            suspension_evidence,
            key=lambda item: item["confidence"],
        )

        suspension_quality = best_suspension["quality"]
        suspension_confidence = best_suspension["confidence"]
        suspension_image = best_suspension["image"]

    # -------------------------------------------------------------
    # EVIDENCE SUFFICIENCY
    # -------------------------------------------------------------
    has_uncertain_routing = bool(uncertain_images)
    has_suspension = bool(suspension_evidence)
    has_strap_evidence = bool(strap_evidence)

    image_quality_status = "good"

    if not has_suspension:
        image_quality_status = "poor"

    if not has_strap_evidence:
        image_quality_status = "poor"

    evidence = {
        "image_quality": image_quality_status,
        "suspension": suspension_quality,
        "straps": strap_evidence,
    }

    decision = assess_tie_down(evidence)

    # -------------------------------------------------------------
    # EXPLANATION
    # -------------------------------------------------------------
    reasons = []

    if has_uncertain_routing:
        reasons.append(
            "One or more images could not be confidently routed."
        )

    if not has_suspension:
        reasons.append(
            "No confident pneumatic suspension image was identified."
        )

    if not has_strap_evidence:
        reasons.append(
            "No confident strap evidence was identified."
        )

    bad_straps = [
        item
        for item in strap_evidence
        if item["quality"]
        in {
            "loose",
            "unconnected",
            "missing_anchor",
        }
    ]

    good_straps = [
        item
        for item in strap_evidence
        if item["quality"] == "good"
    ]

    if bad_straps:
        reasons.append(
            f"{len(bad_straps)} strap finding(s) were classified as incorrect."
        )

    if not reasons and len(good_straps) >= 2:
        reasons.append(
            "At least two straps were confidently classified as good."
        )

    if not reasons:
        reasons.append(
            "Additional image evidence is required for a confident decision."
        )

    # -------------------------------------------------------------
    # FINAL RESPONSE
    # -------------------------------------------------------------
    return {
        "assessment_id": assessment_id,
        "decision": decision.value,
        "routing_summary": routing_summary,
        "uncertain_images": uncertain_images,
        "suspension": suspension_quality,
        "suspension_confidence": suspension_confidence,
        "suspension_image": suspension_image,
        "straps": strap_evidence,
        "images": images,
        "segmentation_summary": {
            "total_masks": sum(
                len(image["segmentation"])
                for image in images
            ),
            "images_with_masks": sum(
                1
                for image in images
                if image["segmentation"]
            ),
        },
        "rules": {
            "reasons": reasons,
            "good_strap_count": len(good_straps),
            "bad_strap_count": len(bad_straps),
            "minimum_good_straps_for_positive": 2,
            "strap_count_alone_rejects": False,
        },
    }
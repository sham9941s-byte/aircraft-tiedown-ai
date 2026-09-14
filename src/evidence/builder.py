from collections import Counter


def build_detection_evidence(detections):
    """
    Convert raw YOLO detections into a normalized evidence structure.

    `detections` should contain objects such as:

        {
            "class": "strap",
            "confidence": 0.91,
            "bbox": [x1, y1, x2, y2]
        }

    This layer intentionally does NOT make the final compliance decision.
    """

    counts = Counter()

    for detection in detections:
        class_name = detection["class"]
        counts[class_name] += 1

    return {
        "object_counts": dict(counts),

        "engine_detected": counts["engine"] > 0,

        "trailer_detected": counts["trailer"] > 0,

        "strap_count": counts["strap"],

        "suspension_detected": (
            counts["pneumatic_suspension"] > 0
        ),

        "detections": detections,
    }
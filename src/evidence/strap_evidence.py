from collections import Counter


def build_strap_evidence(strap_results):
    """
    Convert individual strap classifications into assessment-level evidence.

    strap_results:
        [
            {"quality": "good", "confidence": 0.98},
            {"quality": "loose", "confidence": 0.91},
            ...
        ]
    """

    counts = Counter(
        result.get("quality", "unknown")
        for result in strap_results
    )

    detected = len(strap_results)

    correctly_connected = counts["good"]

    problematic = (
        counts["loose"]
        + counts["unconnected"]
        + counts["missing_anchor"]
    )

    uncertain = (
        counts["occluded"]
        + counts["unknown"]
    )

    return {
        "detected": detected,
        "correctly_connected": correctly_connected,
        "loose": counts["loose"],
        "unconnected": counts["unconnected"],
        "missing_anchor": counts["missing_anchor"],
        "occluded": counts["occluded"],
        "unknown": counts["unknown"],
        "problematic": problematic,
        "uncertain": uncertain,
    }
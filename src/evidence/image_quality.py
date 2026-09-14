import cv2


VIEW_THRESHOLDS = {
    "left_side": 100.0,
    "right_side": 100.0,
    "full_trailer": 100.0,
    "pneumatic_suspension": 50.0,
    "left_closeup": 50.0,
    "right_closeup": 50.0,
}


def assess_image_quality(image, view="unknown"):
    if image is None:
        return {
            "quality": "poor",
            "sharpness": 0.0,
            "reason": "image_unreadable",
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    threshold = VIEW_THRESHOLDS.get(view)
    if threshold is None:
        return {
            "quality": "unknown",
            "sharpness": sharpness,
            "threshold": None,
            "reason": "unknown_view",
        }
    if sharpness < threshold:
        return {
            "quality": "poor",
            "sharpness": sharpness,
            "threshold": threshold,
            "reason": "low_sharpness",
        }

    return {
        "quality": "good",
        "sharpness": sharpness,
        "threshold": threshold,
        "reason": None,
    }
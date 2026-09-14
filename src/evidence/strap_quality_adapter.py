import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = (
    "runs/classify/runs/quality/"
    "strap_quality_context-2/weights/best.pt"
)

GOOD_THRESHOLD = 0.80
BAD_THRESHOLD = 0.80

BAD_CLASSES = {
    "loose",
    "unconnected",
    "missing_anchor",
}

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = YOLO(MODEL_PATH)

    return _model


def _crop_from_polygon(image, polygon):
    """
    Create a crop containing only the pixels inside
    the segmentation polygon.
    """

    points = np.asarray(polygon, dtype=np.int32)

    if points.ndim != 2 or len(points) < 3:
        return None

    x1 = max(0, int(points[:, 0].min()))
    y1 = max(0, int(points[:, 1].min()))
    x2 = min(image.shape[1], int(points[:, 0].max()))
    y2 = min(image.shape[0], int(points[:, 1].max()))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()

    local_points = points.copy()
    local_points[:, 0] -= x1
    local_points[:, 1] -= y1

    mask = np.zeros(
        crop.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [local_points],
        255,
    )

    # Keep the strap pixels and suppress everything outside
    # the segmentation mask.
    result = np.zeros_like(crop)
    result[mask > 0] = crop[mask > 0]

    return result


def _crop_from_bbox(image, bbox):
    """
    Fallback crop when polygon information is unavailable.
    """

    x1, y1, x2, y2 = [
        int(v)
        for v in bbox
    ]

    h, w = image.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def classify_strap(
    image,
    bbox=None,
    polygon=None,
):
    """
    Classify strap quality.

    Preferred input:
        segmentation polygon

    Fallback:
        bounding box

    Returns the complete classifier probability distribution.
    """

    if image is None:
        return {
            "quality": "unknown",
            "confidence": 0.0,
            "probabilities": {},
        }

    if polygon is not None:
        crop = _crop_from_polygon(
            image,
            polygon,
        )
    elif bbox is not None:
        crop = _crop_from_bbox(
            image,
            bbox,
        )
    else:
        return {
            "quality": "unknown",
            "confidence": 0.0,
            "probabilities": {},
        }

    if crop is None or crop.size == 0:
        return {
            "quality": "unknown",
            "confidence": 0.0,
            "probabilities": {},
        }

    model = _get_model()

    results = model.predict(
        source=crop,
        imgsz=224,
        device=0,
        verbose=False,
    )

    result = results[0]

    if result.probs is None:
        return {
            "quality": "unknown",
            "confidence": 0.0,
            "probabilities": {},
        }

    probabilities = {}

    for index, probability in enumerate(
        result.probs.data.cpu().tolist()
    ):
        class_name = model.names[index]
        probabilities[class_name] = float(probability)

    good_probability = probabilities.get(
        "good",
        0.0,
    )

    bad_candidates = {
        name: probability
        for name, probability in probabilities.items()
        if name in BAD_CLASSES
    }

    strongest_bad_class = None
    strongest_bad_probability = 0.0

    if bad_candidates:
        strongest_bad_class = max(
            bad_candidates,
            key=bad_candidates.get,
        )
        strongest_bad_probability = bad_candidates[
            strongest_bad_class
        ]

    # Confident GOOD.
    if (
        good_probability >= GOOD_THRESHOLD
        and good_probability > strongest_bad_probability
    ):
        return {
            "quality": "good",
            "confidence": good_probability,
            "probabilities": probabilities,
        }

    # Confident BAD.
    if (
        strongest_bad_class is not None
        and strongest_bad_probability >= BAD_THRESHOLD
        and strongest_bad_probability > good_probability
    ):
        return {
            "quality": strongest_bad_class,
            "confidence": strongest_bad_probability,
            "probabilities": probabilities,
        }

    # Everything else is uncertain.
    strongest_probability = max(
        probabilities.values(),
        default=0.0,
    )

    return {
        "quality": "unknown",
        "confidence": strongest_probability,
        "probabilities": probabilities,
    }
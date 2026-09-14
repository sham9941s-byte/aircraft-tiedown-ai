from ultralytics import SAM

MODEL_PATH = "sam_b.pt"

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = SAM(MODEL_PATH)

    return _model


def segment_bbox(image_path, bbox):
    """
    Segment an object using its YOLO bounding box as a SAM prompt.

    Returns a polygon-style segmentation contour or None.
    """

    model = _get_model()

    results = model.predict(
        source=image_path,
        bboxes=[bbox],
        device=0,
        verbose=False,
    )

    result = results[0]

    if result.masks is None:
        return None

    # Get polygon contours in original image coordinates.
    polygons = result.masks.xy

    if not polygons:
        return None

    return polygons[0].tolist()
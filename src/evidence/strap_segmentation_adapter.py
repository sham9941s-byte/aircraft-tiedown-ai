from ultralytics import YOLO


MODEL_PATH = (
    "runs/segment/runs/segmentation/"
    "strap_segmentation_augmented/weights/best.pt"
)

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = YOLO(MODEL_PATH)

    return _model


def segment_straps(image_path, confidence=0.25):
    """
    Run the trained strap segmentation model.

    Returns visual evidence only:
    - segmentation polygons
    - bounding boxes
    - confidence

    This output must NOT be used directly for compliance decisions.
    """

    model = _get_model()

    results = model.predict(
        source=image_path,
        conf=confidence,
        device=0,
        verbose=False,
    )

    result = results[0]

    if result.masks is None or result.boxes is None:
        return []

    polygons = result.masks.xy
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    detections = []

    for polygon, bbox, conf in zip(
        polygons,
        boxes,
        confidences,
    ):
        detections.append(
            {
                "bbox": bbox,
                "confidence": float(conf),
                "polygon": polygon.tolist(),
            }
        )

    return detections
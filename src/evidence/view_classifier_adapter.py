from ultralytics import YOLO


MODEL_PATH = (
    "runs/classify/runs/view_router/"
    "real_view_router/weights/best.pt"
)

CONFIDENCE_THRESHOLD = 0.75

_model = None


def _get_model():
    global _model

    if _model is None:
        _model = YOLO(MODEL_PATH)

    return _model


def classify_view(image_path: str):
    """
    Classify an uploaded inspection image into one of
    the three routing categories produced by the real-image
    view router:

        strap_view
        suspension
        other

    Low-confidence predictions are returned as unknown.
    """

    model = _get_model()

    results = model.predict(
        source=image_path,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    if result.probs is None:
        return {
            "route": "unknown",
            "confidence": 0.0,
            "class": "unknown",
        }

    top1_index = int(result.probs.top1)
    confidence = float(result.probs.top1conf)

    class_name = model.names[top1_index]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "route": "unknown",
            "confidence": confidence,
            "class": class_name,
        }

    return {
        "route": class_name,
        "confidence": confidence,
        "class": class_name,
    }
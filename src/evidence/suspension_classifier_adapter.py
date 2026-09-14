from pathlib import Path

from ultralytics import YOLO


MODEL_PATH = Path(
    "runs/classify/runs/suspension/suspension_classifier/weights/best.pt"
)

_model = YOLO(str(MODEL_PATH))


def classify_suspension(image):
    results = _model.predict(
        source=image,
        imgsz=224,
        verbose=False,
    )

    result = results[0]

    index = int(result.probs.top1)
    confidence = float(result.probs.top1conf)

    quality = result.names[index]

    return {
        "quality": quality,
        "confidence": confidence,
    }
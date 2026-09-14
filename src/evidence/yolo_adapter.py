from ultralytics import YOLO


MODEL_PATH = "runs/detect/runs/tiedown/yolo11n_100/weights/best.pt"


def predict_image(image_path, confidence=0.25):
    """
    Run YOLO detection and convert the result
    into the normalized detection format used
    by the evidence layer.
    """

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=image_path,
        conf=confidence,
        device=0,
        verbose=False,
    )

    detections = []

    result = results[0]

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence_score = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": model.names[class_id],
            "confidence": confidence_score,
            "bbox": [
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
            ],
        })

    return detections
from pathlib import Path
import json
from datetime import datetime

from ultralytics import YOLO


DATASET = "data/training/dataset/dataset.yaml"

BASE_MODEL = (
    "runs/detect/runs/tiedown/"
    "yolo11n_100/weights/best.pt"
)

STATUS_FILE = Path(
    "data/training/training_status.json"
)

PROJECT_DIR = Path(
    "runs/detect/runs/training"
)

RUN_NAME = "real_adaptation"


def update_status(
    status: str,
    message: str,
    metrics: dict | None = None,
):
    STATUS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "status": status,
        "message": message,
        "metrics": metrics or {},
        "completed_at": (
            datetime.now().isoformat()
            if status == "completed"
            else None
        ),
    }

    STATUS_FILE.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def find_latest_best_model():

    candidates = list(
        PROJECT_DIR.glob(
            "real_adaptation*/weights/best.pt"
        )
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def main():

    print("=" * 60)
    print(
        "VISION WINGS - REAL IMAGE MODEL ADAPTATION"
    )
    print("=" * 60)

    update_status(
        "running",
        "GPU training is running...",
    )

    try:

        model = YOLO(
            BASE_MODEL
        )

        results = model.train(

            data=DATASET,

            epochs=30,

            imgsz=640,

            batch=8,

            device=0,

            project=str(
                PROJECT_DIR
            ),

            name=RUN_NAME,

            patience=8,

            workers=2,

            pretrained=True,

            verbose=True,

        )

        # Ultralytics may create:
        #
        # real_adaptation
        # real_adaptation-2
        # real_adaptation-3
        #
        # Find the actual latest best.pt.

        best_model = (
            find_latest_best_model()
        )

        metrics = {}

        try:

            metrics = {

                "map50":
                    float(
                        results.box.map50
                    ),

                "map50_95":
                    float(
                        results.box.map
                    ),

                "precision":
                    float(
                        results.box.mp
                    ),

                "recall":
                    float(
                        results.box.mr
                    ),

            }

        except Exception:

            pass


        if best_model is None:

            raise RuntimeError(
                "Training finished but "
                "best.pt could not be located."
            )


        update_status(

            "completed",

            "Training completed successfully.",

            {

                **metrics,

                "model":
                    str(
                        best_model
                    ),

            },

        )


        print("=" * 60)
        print(
            "TRAINING COMPLETE"
        )
        print(
            f"Best model: {best_model}"
        )
        print(
            f"mAP50: "
            f"{metrics.get('map50', 'N/A')}"
        )
        print(
            f"mAP50-95: "
            f"{metrics.get('map50_95', 'N/A')}"
        )
        print(
            f"Precision: "
            f"{metrics.get('precision', 'N/A')}"
        )
        print(
            f"Recall: "
            f"{metrics.get('recall', 'N/A')}"
        )
        print("=" * 60)


    except Exception as exc:

        update_status(
            "error",
            str(exc),
        )

        print(
            f"TRAINING ERROR: {exc}"
        )

        raise


if __name__ == "__main__":

    main()
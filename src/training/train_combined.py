from pathlib import Path
import json
from datetime import datetime

from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================

DATASET = "data/combined/dataset.yaml"

BASE_MODEL = (
    "runs/detect/runs/tiedown/"
    "yolo11n_100/weights/best.pt"
)

PROJECT_DIR = Path(
     "runs/detect/runs/detect/runs/combined"
)

RUN_NAME = "synthetic_real_adaptation"

STATUS_FILE = Path(
    "data/training/combined_training_status.json"
)


# =========================================================
# STATUS
# =========================================================

def update_status(
    status,
    message,
    metrics=None,
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


# =========================================================
# FIND BEST MODEL
# =========================================================

def find_latest_best_model():

    candidates = list(
        PROJECT_DIR.glob(
            "synthetic_real_adaptation*/weights/best.pt"
        )
    )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print(
        "VISION WINGS - SYNTHETIC + REAL TRAINING"
    )
    print("=" * 60)

    update_status(
        "running",
        "Combined synthetic + real GPU training is running...",
    )

    try:

        if not Path(DATASET).exists():

            raise RuntimeError(
                f"Dataset not found: {DATASET}"
            )

        if not Path(BASE_MODEL).exists():

            raise RuntimeError(
                f"Base model not found: {BASE_MODEL}"
            )


        print()
        print("Base model:")
        print(BASE_MODEL)

        print()
        print("Dataset:")
        print(DATASET)

        print()


        # -------------------------------------------------
        # Load existing synthetic-trained detector
        # -------------------------------------------------

        model = YOLO(
            BASE_MODEL
        )


        # -------------------------------------------------
        # Fine-tune on combined dataset
        # -------------------------------------------------

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


        # -------------------------------------------------
        # Locate best model
        # -------------------------------------------------

        best_model = (
            find_latest_best_model()
        )


        if best_model is None:

            raise RuntimeError(
                "Training finished but "
                "best.pt could not be located."
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

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


        metrics[
            "model"
        ] = str(
            best_model
        )


        # -------------------------------------------------
        # Save status
        # -------------------------------------------------

        update_status(
            "completed",
            "Combined training completed successfully.",
            metrics,
        )


        print()
        print("=" * 60)
        print("COMBINED TRAINING COMPLETE")
        print("=" * 60)

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


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()
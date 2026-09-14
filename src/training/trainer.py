from pathlib import Path
from datetime import datetime
import shutil

from ultralytics import YOLO


BASE_DIR = Path("data/training")
IMAGE_DIR = BASE_DIR / "images"
LABEL_DIR = BASE_DIR / "labels"

RUNS_DIR = Path("runs/training")

MODEL_SOURCE = Path(
    "runs/detect/runs/tiedown/yolo11n_100/weights/best.pt"
)


def start_training(
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 8,
):
    """
    Train a new Vision Wings detector using the operator-annotated dataset.
    """

    train_images = IMAGE_DIR / "train"
    val_images = IMAGE_DIR / "val"

    train_labels = LABEL_DIR / "train"
    val_labels = LABEL_DIR / "val"

    for directory in [
        train_images,
        val_images,
        train_labels,
        val_labels,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    if not any(train_images.iterdir()):
        raise RuntimeError(
            "No training images found. Add annotated images first."
        )

    if not any(train_labels.iterdir()):
        raise RuntimeError(
            "No training annotations found."
        )

    # Create YOLO dataset configuration.
    dataset_yaml = BASE_DIR / "dataset.yaml"

    dataset_yaml.write_text(
        f"""path: {BASE_DIR.resolve()}
train: images/train
val: images/val

names:
  0: engine
  1: trailer
  2: strap
  3: pneumatic_suspension
""",
        encoding="utf-8",
    )

    if not MODEL_SOURCE.exists():
        raise RuntimeError(
            f"Base model not found: {MODEL_SOURCE}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = f"vision_wings_{timestamp}"

    print("=" * 60)
    print("VISION WINGS - MODEL TRAINING")
    print("=" * 60)

    print(f"Dataset : {dataset_yaml}")
    print(f"Base    : {MODEL_SOURCE}")
    print(f"Run     : {run_name}")
    print(f"Epochs  : {epochs}")
    print()

    model = YOLO(str(MODEL_SOURCE))

    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=False,
    )

    best_model = (
        RUNS_DIR
        / run_name
        / "weights"
        / "best.pt"
    )

    if best_model.exists():

        model_registry = Path("models")
        model_registry.mkdir(exist_ok=True)

        version = (
            model_registry
            / f"vision_wings_{timestamp}.pt"
        )

        shutil.copy2(
            best_model,
            version,
        )

        print()
        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best model : {best_model}")
        print(f"Registered : {version}")

        return {
            "status": "completed",
            "run": run_name,
            "model": str(version),
        }

    raise RuntimeError(
        "Training completed but best.pt was not found."
    )


if __name__ == "__main__":
    start_training()
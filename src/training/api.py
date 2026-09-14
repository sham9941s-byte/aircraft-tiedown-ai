from pathlib import Path
from uuid import uuid4
import json
import shutil
import subprocess
import sys

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/training",
    tags=["training"],
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path("data/training")

BATCHES_DIR = BASE_DIR / "batches"
CURRENT_BATCH_FILE = BASE_DIR / "current_batch.json"

STATUS_FILE = BASE_DIR / "training_status.json"

DATASET_DIR = BASE_DIR / "dataset"
TRAIN_IMAGES = DATASET_DIR / "images" / "train"
TRAIN_LABELS = DATASET_DIR / "labels" / "train"
VAL_IMAGES = DATASET_DIR / "images" / "val"
VAL_LABELS = DATASET_DIR / "labels" / "val"

DATASET_YAML = DATASET_DIR / "dataset.yaml"

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


# =========================================================
# MODELS
# =========================================================

class Annotation(BaseModel):
    image_id: str
    view: str
    objects: list[dict] = []


# =========================================================
# BATCH MANAGEMENT
# =========================================================

def ensure_directories():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    VAL_IMAGES.mkdir(parents=True, exist_ok=True)
    VAL_LABELS.mkdir(parents=True, exist_ok=True)


def get_current_batch():
    """
    Return the current batch information.

    If no current batch exists, create batch_001.
    """

    ensure_directories()

    if CURRENT_BATCH_FILE.exists():
        try:
            data = json.loads(
                CURRENT_BATCH_FILE.read_text(
                    encoding="utf-8"
                )
            )

            batch_id = data.get("batch_id")

            if batch_id:
                return data

        except Exception:
            pass

    return create_new_batch()


def create_new_batch():
    """
    Create a new empty current batch.
    """

    ensure_directories()

    existing = []

    for path in BATCHES_DIR.iterdir():
        if path.is_dir() and path.name.startswith("batch_"):
            try:
                number = int(
                    path.name.replace("batch_", "")
                )
                existing.append(number)
            except ValueError:
                continue

    next_number = (
        max(existing) + 1
        if existing
        else 1
    )

    batch_id = f"batch_{next_number:03d}"

    batch_dir = BATCHES_DIR / batch_id

    images_dir = batch_dir / "images"
    labels_dir = batch_dir / "labels"
    metadata_dir = batch_dir / "metadata"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "batch_id": batch_id,
        "status": "active",
        "images": 0,
        "annotations": 0,
    }

    CURRENT_BATCH_FILE.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    return data


def get_batch_dirs():
    batch = get_current_batch()

    batch_dir = BATCHES_DIR / batch["batch_id"]

    return {
        "batch": batch,
        "root": batch_dir,
        "images": batch_dir / "images",
        "labels": batch_dir / "labels",
        "metadata": batch_dir / "metadata",
    }


def update_current_batch_counts():
    dirs = get_batch_dirs()

    images = [
        p for p in dirs["images"].glob("*")
        if p.is_file()
    ]

    annotations = 0

    for image in images:
        label = (
            dirs["labels"] /
            f"{image.stem}.txt"
        )

        if label.exists():
            annotations += 1

    batch = dirs["batch"]

    batch["images"] = len(images)
    batch["annotations"] = annotations

    CURRENT_BATCH_FILE.write_text(
        json.dumps(
            batch,
            indent=2,
        ),
        encoding="utf-8",
    )


# =========================================================
# LEGACY DATA MIGRATION
# =========================================================

def migrate_legacy_training_data():
    """
    Preserve the existing training data.

    The old implementation stored files in:

        data/training/images/train
        data/training/labels/train
        data/training/metadata

    Move those files into batch_001 if they exist.

    This is intentionally performed only when batch storage
    has not already been initialized.
    """

    legacy_images = BASE_DIR / "images" / "train"
    legacy_labels = BASE_DIR / "labels" / "train"
    legacy_metadata = BASE_DIR / "metadata"

    legacy_exists = (
        legacy_images.exists()
        and any(legacy_images.glob("*"))
    )

    if not legacy_exists:
        return

    # If batches already exist, don't migrate again.
    if any(
        p.is_dir()
        for p in BATCHES_DIR.glob("batch_*")
    ):
        return

    batch = create_new_batch()

    batch_dir = BATCHES_DIR / batch["batch_id"]

    new_images = batch_dir / "images"
    new_labels = batch_dir / "labels"
    new_metadata = batch_dir / "metadata"

    for source_dir, destination_dir in [
        (legacy_images, new_images),
        (legacy_labels, new_labels),
        (legacy_metadata, new_metadata),
    ]:

        if not source_dir.exists():
            continue

        for file in source_dir.iterdir():

            if not file.is_file():
                continue

            shutil.move(
                str(file),
                str(destination_dir / file.name),
            )

    update_current_batch_counts()


# =========================================================
# DATASET YAML
# =========================================================

def write_dataset_yaml():
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATASET_YAML.write_text(
        """path: data/training/dataset
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


# =========================================================
# TRAINING STATUS
# =========================================================

def set_training_status(
    status: str,
    message: str = "",
    pid: int | None = None,
):
    STATUS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "status": status,
        "message": message,
        "pid": pid,
    }

    STATUS_FILE.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


# =========================================================
# STARTUP INITIALIZATION
# =========================================================

def initialize_training_storage():
    ensure_directories()

    migrate_legacy_training_data()

    get_current_batch()

    update_current_batch_counts()


# =========================================================
# UPLOAD
# =========================================================

@router.post("/upload")
async def upload_training_image(
    image: UploadFile = File(...)
):

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, PNG and WEBP images "
                "are supported."
            ),
        )

    initialize_training_storage()

    dirs = get_batch_dirs()

    image_id = uuid4().hex[:12]

    extension = ALLOWED_TYPES[
        image.content_type
    ]

    filename = f"{image_id}{extension}"

    image_path = (
        dirs["images"] / filename
    )

    with image_path.open("wb") as f:
        shutil.copyfileobj(
            image.file,
            f,
        )

    update_current_batch_counts()

    return {
        "status": "uploaded",
        "image_id": image_id,
        "filename": filename,
        "path": str(image_path),
        "batch_id": dirs["batch"]["batch_id"],
    }


# =========================================================
# SAVE ANNOTATION
# =========================================================

@router.post("/annotate")
def save_annotation(
    annotation: Annotation,
):

    initialize_training_storage()

    dirs = get_batch_dirs()

    label_path = (
        dirs["labels"] /
        f"{annotation.image_id}.txt"
    )

    lines = []

    for obj in annotation.objects:

        class_id = int(
            obj["class_id"]
        )

        cx = float(obj["cx"])
        cy = float(obj["cy"])
        width = float(obj["width"])
        height = float(obj["height"])

        cx = max(
            0.0,
            min(1.0, cx),
        )

        cy = max(
            0.0,
            min(1.0, cy),
        )

        width = max(
            0.0,
            min(1.0, width),
        )

        height = max(
            0.0,
            min(1.0, height),
        )

        lines.append(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    metadata_path = (
        dirs["metadata"] /
        f"{annotation.image_id}.json"
    )

    metadata = {
        "image_id": annotation.image_id,
        "view": annotation.view,
        "objects": annotation.objects,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    update_current_batch_counts()

    return {
        "status": "saved",
        "image_id": annotation.image_id,
        "view": annotation.view,
        "objects": len(
            annotation.objects
        ),
        "batch_id": dirs["batch"]["batch_id"],
    }


# =========================================================
# DATASET STATUS
# =========================================================

@router.get("/dataset")
def dataset_status():

    initialize_training_storage()

    dirs = get_batch_dirs()

    images = [
        p for p in dirs["images"].glob("*")
        if p.is_file()
    ]

    annotated = 0

    for image in images:

        label_file = (
            dirs["labels"] /
            f"{image.stem}.txt"
        )

        if label_file.exists():
            annotated += 1

    return {
        "batch_id": dirs["batch"]["batch_id"],
        "images": len(images),
        "annotations": annotated,
        "ready": (
            len(images) > 0
            and annotated > 0
        ),
    }


# =========================================================
# LIST TRAINING IMAGES
# =========================================================

@router.get("/images")
def list_training_images():

    initialize_training_storage()

    dirs = get_batch_dirs()

    results = []

    for image_path in sorted(
        dirs["images"].glob("*")
    ):

        if not image_path.is_file():
            continue

        label_path = (
            dirs["labels"] /
            f"{image_path.stem}.txt"
        )

        objects = []
        view = ""

        # Prefer the JSON metadata because it contains the
        # complete annotation, including polygon segmentation.
        metadata_path = (
            dirs["metadata"] /
            f"{image_path.stem}.json"
        )

        if metadata_path.exists():
            try:
                metadata = json.loads(
                    metadata_path.read_text(
                        encoding="utf-8"
                    )
                )

                objects = metadata.get(
                    "objects",
                    []
                )

                view = metadata.get(
                    "view",
                    ""
                )

            except Exception:
                objects = []
                view = ""

        # Backward compatibility for old bbox-only annotations.
        if not objects and label_path.exists():

            lines = label_path.read_text(
                encoding="utf-8"
            ).splitlines()

            for line in lines:

                parts = line.split()

                if len(parts) != 5:
                    continue

                objects.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]),
                    "cy": float(parts[2]),
                    "width": float(parts[3]),
                    "height": float(parts[4]),
                })

        results.append({
            "id": image_path.stem,
            "filename": image_path.name,
            "url": (
                "/training/image/"
                + image_path.name
            ),
            "annotated": label_path.exists(),
            "view": view,
            "objects": objects,
        })

    return {
        "batch_id": dirs["batch"]["batch_id"],
        "images": results,
        "count": len(results),
    }


# =========================================================
# GET TRAINING IMAGE
# =========================================================

@router.get("/image/{filename}")
def get_training_image(
    filename: str,
):

    initialize_training_storage()

    dirs = get_batch_dirs()

    image_path = (
        dirs["images"] / filename
    ).resolve()

    try:

        image_path.relative_to(
            dirs["images"].resolve()
        )

    except ValueError:

        raise HTTPException(
            status_code=403,
            detail="Invalid image path.",
        )

    if not image_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Image not found.",
        )

    return FileResponse(
        image_path
    )


# =========================================================
# PREPARE TRAINING DATASET
# =========================================================

def clear_directory(directory: Path):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file in directory.iterdir():

        if file.is_file():
            file.unlink()


def prepare_training_split():

    initialize_training_storage()

    dirs = get_batch_dirs()

    images = sorted(
        p for p in dirs["images"].glob("*")
        if p.is_file()
    )

    annotated_images = []

    for image in images:

        label = (
            dirs["labels"] /
            f"{image.stem}.txt"
        )

        if label.exists():
            annotated_images.append(
                image
            )

    if len(annotated_images) < 2:

        raise RuntimeError(
            "At least 2 annotated images "
            "are required."
        )

    # Clear generated dataset.
    clear_directory(TRAIN_IMAGES)
    clear_directory(TRAIN_LABELS)
    clear_directory(VAL_IMAGES)
    clear_directory(VAL_LABELS)

    # -----------------------------------------------------
    # 80/20 split.
    #
    # IMPORTANT:
    # Validation files are NOT kept in training.
    # -----------------------------------------------------

    val_count = max(
        1,
        round(
            len(annotated_images) * 0.2
        ),
    )

    if len(annotated_images) >= 5:
        val_count = max(
            1,
            round(
                len(annotated_images) * 0.2
            ),
        )

    val_images = annotated_images[-val_count:]

    train_images = annotated_images[
        :-val_count
    ]

    for image in train_images:

        label = (
            dirs["labels"] /
            f"{image.stem}.txt"
        )

        shutil.copy2(
            image,
            TRAIN_IMAGES / image.name,
        )

        shutil.copy2(
            label,
            TRAIN_LABELS / label.name,
        )

    for image in val_images:

        label = (
            dirs["labels"] /
            f"{image.stem}.txt"
        )

        shutil.copy2(
            image,
            VAL_IMAGES / image.name,
        )

        shutil.copy2(
            label,
            VAL_LABELS / label.name,
        )

    write_dataset_yaml()

    return {
        "batch_id": dirs["batch"]["batch_id"],
        "total": len(annotated_images),
        "training": len(train_images),
        "validation": len(val_images),
    }


# =========================================================
# START GPU TRAINING
# =========================================================

@router.post("/train")
def start_training():

    try:

        current_status = {
            "status": "idle"
        }

        if STATUS_FILE.exists():

            try:
                current_status = json.loads(
                    STATUS_FILE.read_text(
                        encoding="utf-8"
                    )
                )
            except Exception:
                pass

        if current_status.get("status") in {
            "starting",
            "running",
        }:

            raise HTTPException(
                status_code=409,
                detail=(
                    "Training is already running."
                ),
            )

        split = prepare_training_split()

        base_model = Path(
            "runs/detect/runs/tiedown/"
            "yolo11n_100/weights/best.pt"
        )

        if not base_model.exists():

            raise RuntimeError(
                f"Base model not found: "
                f"{base_model}"
            )

        set_training_status(
            "starting",
            (
                f"Preparing GPU training with "
                f"{split['training']} training images "
                f"and "
                f"{split['validation']} "
                f"validation images."
            ),
        )

        command = [
            sys.executable,
            "-m",
            "src.training.train",
        ]

        process = subprocess.Popen(
            command,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                if sys.platform == "win32"
                else 0
            ),
        )

        set_training_status(
            "running",
            (
                f"GPU training started. "
                f"Training images: "
                f"{split['training']}. "
                f"Validation images: "
                f"{split['validation']}."
            ),
            process.pid,
        )

        return {
            "status": "started",
            "message": (
                "GPU training started successfully."
            ),
            "pid": process.pid,
            "batch_id": split["batch_id"],
            "training_images": split["training"],
            "validation_images": split["validation"],
        }

    except HTTPException:
        raise

    except Exception as exc:

        set_training_status(
            "error",
            str(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# =========================================================
# TRAINING STATUS
# =========================================================

@router.get("/train/status")
def training_status():

    if not STATUS_FILE.exists():

        return {
            "status": "idle",
            "message": "No training job started.",
        }

    try:

        return json.loads(
            STATUS_FILE.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return {
            "status": "unknown",
            "message": (
                "Unable to read training status."
            ),
        }


# =========================================================
# FINISH / ARCHIVE CURRENT BATCH
# =========================================================

@router.post("/batch/complete")
def complete_current_batch():

    initialize_training_storage()

    dirs = get_batch_dirs()

    batch = dirs["batch"]

    update_current_batch_counts()

    batch = get_current_batch()

    if batch["images"] == 0:

        raise HTTPException(
            status_code=400,
            detail="Current training batch is empty.",
        )

    batch["status"] = "completed"

    batch_dir = dirs["root"]

    batch_metadata = batch_dir / "batch.json"

    batch_metadata.write_text(
        json.dumps(
            batch,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Remove current pointer.
    if CURRENT_BATCH_FILE.exists():
        CURRENT_BATCH_FILE.unlink()

    # Create a completely fresh batch.
    new_batch = create_new_batch()

    return {
        "status": "completed",
        "completed_batch": batch["batch_id"],
        "new_batch": new_batch["batch_id"],
    }


# =========================================================
# BATCH HISTORY
# =========================================================

@router.get("/batches")
def list_batches():

    initialize_training_storage()

    batches = []

    for batch_dir in sorted(
        BATCHES_DIR.glob("batch_*")
    ):

        if not batch_dir.is_dir():
            continue

        metadata_file = (
            batch_dir / "batch.json"
        )

        if metadata_file.exists():

            try:

                data = json.loads(
                    metadata_file.read_text(
                        encoding="utf-8"
                    )
                )

                batches.append(data)

                continue

            except Exception:
                pass

        images_dir = (
            batch_dir / "images"
        )

        images = list(
            images_dir.glob("*")
        ) if images_dir.exists() else []

        batches.append({
            "batch_id": batch_dir.name,
            "status": "completed",
            "images": len(images),
        })

    current = get_current_batch()

    return {
        "current_batch": current,
        "batches": batches,
    }


# =========================================================
# INITIALIZE ON IMPORT
# =========================================================

initialize_training_storage()
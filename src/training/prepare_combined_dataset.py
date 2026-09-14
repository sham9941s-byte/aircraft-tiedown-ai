from pathlib import Path
import shutil
import random


# =========================================================
# SOURCE DATASETS
# =========================================================

SYNTHETIC_DIR = Path("data/yolo")

REAL_DIR = Path(
    "data/training/batches/batch_001"
)

OUTPUT_DIR = Path(
    "data/combined"
)


# =========================================================
# SETTINGS
# =========================================================

RANDOM_SEED = 42

REAL_VALIDATION_RATIO = 0.20


# =========================================================
# HELPERS
# =========================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def clear_directory(directory: Path):

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in directory.iterdir():

        if item.is_file():
            item.unlink()

        elif item.is_dir():
            shutil.rmtree(item)


def copy_pair(
    image_path: Path,
    label_path: Path,
    output_images: Path,
    output_labels: Path,
    prefix: str,
):

    destination_image = (
        output_images /
        f"{prefix}_{image_path.name}"
    )

    destination_label = (
        output_labels /
        f"{prefix}_{label_path.name}"
    )

    shutil.copy2(
        image_path,
        destination_image,
    )

    shutil.copy2(
        label_path,
        destination_label,
    )


def get_images(directory: Path):

    if not directory.exists():
        return []

    return sorted(
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTENSIONS
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("VISION WINGS - COMBINED DATASET")
    print("=" * 60)

    random.seed(
        RANDOM_SEED
    )

    # -----------------------------------------------------
    # Create output directories
    # -----------------------------------------------------

    train_images = (
        OUTPUT_DIR /
        "images" /
        "train"
    )

    train_labels = (
        OUTPUT_DIR /
        "labels" /
        "train"
    )

    val_images = (
        OUTPUT_DIR /
        "images" /
        "val"
    )

    val_labels = (
        OUTPUT_DIR /
        "labels" /
        "val"
    )

    for directory in [
        train_images,
        train_labels,
        val_images,
        val_labels,
    ]:

        clear_directory(
            directory
        )


    # =====================================================
    # SYNTHETIC DATA
    # =====================================================

    synthetic_train_images = get_images(
        SYNTHETIC_DIR /
        "images" /
        "train"
    )

    synthetic_train_count = 0

    for image in synthetic_train_images:

        label = (
            SYNTHETIC_DIR /
            "labels" /
            "train" /
            f"{image.stem}.txt"
        )

        if not label.exists():
            continue

        copy_pair(
            image,
            label,
            train_images,
            train_labels,
            "synthetic",
        )

        synthetic_train_count += 1


    synthetic_val_images = get_images(
        SYNTHETIC_DIR /
        "images" /
        "val"
    )

    synthetic_val_count = 0

    for image in synthetic_val_images:

        label = (
            SYNTHETIC_DIR /
            "labels" /
            "val" /
            f"{image.stem}.txt"
        )

        if not label.exists():
            continue

        copy_pair(
            image,
            label,
            val_images,
            val_labels,
            "synthetic",
        )

        synthetic_val_count += 1


    # =====================================================
    # REAL DATA
    # =====================================================

    real_images = get_images(
        REAL_DIR /
        "images"
    )

    if not real_images:

        raise RuntimeError(
            "No real annotated images found in "
            f"{REAL_DIR / 'images'}"
        )


    annotated_real_images = []

    for image in real_images:

        label = (
            REAL_DIR /
            "labels" /
            f"{image.stem}.txt"
        )

        if label.exists():

            annotated_real_images.append(
                image
            )


    if len(annotated_real_images) < 2:

        raise RuntimeError(
            "At least 2 annotated real images "
            "are required."
        )


    # -----------------------------------------------------
    # Deterministic 80/20 real split
    # -----------------------------------------------------

    shuffled_real = list(
        annotated_real_images
    )

    random.shuffle(
        shuffled_real
    )

    validation_count = max(
        1,
        round(
            len(shuffled_real)
            * REAL_VALIDATION_RATIO
        ),
    )

    real_val = shuffled_real[
        :validation_count
    ]

    real_train = shuffled_real[
        validation_count:
    ]


    # -----------------------------------------------------
    # Real training images
    # -----------------------------------------------------

    real_train_count = 0

    for image in real_train:

        label = (
            REAL_DIR /
            "labels" /
            f"{image.stem}.txt"
        )

        copy_pair(
            image,
            label,
            train_images,
            train_labels,
            "real",
        )

        real_train_count += 1


    # -----------------------------------------------------
    # Real validation images
    # -----------------------------------------------------

    real_val_count = 0

    for image in real_val:

        label = (
            REAL_DIR /
            "labels" /
            f"{image.stem}.txt"
        )

        copy_pair(
            image,
            label,
            val_images,
            val_labels,
            "real",
        )

        real_val_count += 1


    # =====================================================
    # DATASET YAML
    # =====================================================

    yaml_path = (
        OUTPUT_DIR /
        "dataset.yaml"
    )

    yaml_path.write_text(
        """path: data/combined
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


    # =====================================================
    # SUMMARY
    # =====================================================

    print()
    print("COMBINED DATASET READY")
    print("-" * 60)

    print(
        f"Synthetic training:     "
        f"{synthetic_train_count}"
    )

    print(
        f"Real training:          "
        f"{real_train_count}"
    )

    print(
        f"Total training:         "
        f"{synthetic_train_count + real_train_count}"
    )

    print()

    print(
        f"Synthetic validation:   "
        f"{synthetic_val_count}"
    )

    print(
        f"Real validation:        "
        f"{real_val_count}"
    )

    print(
        f"Total validation:       "
        f"{synthetic_val_count + real_val_count}"
    )

    print()

    print(
        f"Dataset YAML:           "
        f"{yaml_path}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()
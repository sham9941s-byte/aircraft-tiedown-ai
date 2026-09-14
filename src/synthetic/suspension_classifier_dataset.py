from pathlib import Path
import json
import shutil

SOURCE = Path("data/synthetic")
ANNOTATIONS = SOURCE / "annotations"
IMAGES = SOURCE / "images"
OUTPUT = Path("data/suspension_classification")

TRAIN_RATIO = 0.8


def main():
    annotations = sorted(ANNOTATIONS.glob("*.json"))

    assessments = {}

    for ann_file in annotations:
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        assessment_id = ann["assessment_id"]
        assessments.setdefault(assessment_id, []).append(ann)

    assessment_ids = sorted(assessments)

    split_idx = int(len(assessment_ids) * TRAIN_RATIO)
    train_ids = set(assessment_ids[:split_idx])
    val_ids = set(assessment_ids[split_idx:])

    counts = {"train": {}, "val": {}}

    for split, ids in [("train", train_ids), ("val", val_ids)]:
        for assessment_id in ids:
            for ann in assessments[assessment_id]:

                # Only use pneumatic suspension images
                if ann["view"] != "pneumatic_suspension":
                    continue

                suspension_objects = [
                    obj for obj in ann.get("objects", [])
                    if obj["class"] == "pneumatic_suspension"
                ]

                if not suspension_objects:
                    continue

                # Synthetic data contains one suspension object per image.
                quality = suspension_objects[0].get("quality", "unknown")

                if quality not in {"good", "bad"}:
                    continue

                src = IMAGES / f'{ann["image_id"]}.jpg'

                if not src.exists():
                    print(f"WARNING: missing image {src}")
                    continue

                dst_dir = OUTPUT / split / quality
                dst_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(src, dst_dir / src.name)

                counts[split][quality] = counts[split].get(quality, 0) + 1

    print("\nSuspension classification dataset created")
    print(f"Assessments: {len(assessment_ids)}")
    print(f"Train assessments: {len(train_ids)}")
    print(f"Val assessments: {len(val_ids)}")
    print(f"Train: {counts['train']}")
    print(f"Val:   {counts['val']}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
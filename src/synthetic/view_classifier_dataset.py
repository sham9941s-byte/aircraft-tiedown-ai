from pathlib import Path
import json
import random
import shutil

SOURCE = Path("data/synthetic")
OUTPUT = Path("data/view_classification")

TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    assessments = {}

    for ann_file in (SOURCE / "annotations").glob("*.json"):
        if ann_file.name == "manifest.json":
            continue

        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        assessment_id = ann["assessment_id"]
        assessments.setdefault(assessment_id, []).append(ann)

    ids = list(assessments.keys())
    random.shuffle(ids)

    split = int(len(ids) * TRAIN_RATIO)
    train_ids = set(ids[:split])
    val_ids = set(ids[split:])

    class_counts = {}

    for assessment_id, annotations in assessments.items():
        split_name = "train" if assessment_id in train_ids else "val"

        for ann in annotations:
            view = ann["view"]

            # Keep the dataset flat:
            # train/<view>/image.jpg
            out_dir = OUTPUT / split_name / view
            out_dir.mkdir(parents=True, exist_ok=True)

            image_name = ann["image_id"] + ".jpg"
            source_image = SOURCE / "images" / image_name

            if not source_image.exists():
                print(f"WARNING: missing {source_image}")
                continue

            shutil.copy2(
                source_image,
                out_dir / image_name
            )

            class_counts[view] = class_counts.get(view, 0) + 1

    print("\nView classification dataset created.")
    print(f"Assessments: {len(ids)}")
    print(f"Train assessments: {len(train_ids)}")
    print(f"Val assessments: {len(val_ids)}")
    print("\nImages per view:")

    for view, count in sorted(class_counts.items()):
        print(f"  {view}: {count}")

    print(f"\nOutput: {OUTPUT}")


if __name__ == "__main__":
    main()
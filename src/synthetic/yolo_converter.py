import json
import random
import shutil
from pathlib import Path


CLASS_MAP = {
    "engine": 0,
    "trailer": 1,
    "strap": 2,
    "pneumatic_suspension": 3,
}


def convert_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox

    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height

    return cx, cy, w, h


def main():
    annotation_dir = Path("data/synthetic/annotations")
    image_dir = Path("data/synthetic/images")
    output_dir = Path("data/yolo")

    annotations = sorted(annotation_dir.glob("*.json"))

    # ---------------------------------------------------------
    # Group images by assessment_id
    # ---------------------------------------------------------

    assessments = {}

    for annotation_file in annotations:
        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assessment_id = data["assessment_id"]

        assessments.setdefault(assessment_id, []).append(
            (annotation_file, data)
        )

    # ---------------------------------------------------------
    # Stratified assessment-level split
    # ---------------------------------------------------------

    scenario_groups = {}

    for assessment_id, items in assessments.items():
        scenario = items[0][1]["scenario"]

        scenario_groups.setdefault(
            scenario, []
        ).append(assessment_id)

    random.seed(42)

    train_ids = []
    val_ids = []

    for scenario, ids in sorted(scenario_groups.items()):
        ids = sorted(ids)
        random.shuffle(ids)

        val_count = max(1, int(len(ids) * 0.2))

        val_ids.extend(ids[:val_count])
        train_ids.extend(ids[val_count:])

    random.shuffle(train_ids)
    random.shuffle(val_ids)

    print(f"Assessments: {len(assessments)}")
    print(f"Train assessments: {len(train_ids)}")
    print(f"Val assessments: {len(val_ids)}")

    # ---------------------------------------------------------
    # Process train/validation split
    # ---------------------------------------------------------

    def process_split(split_name, split_ids):
        image_output = output_dir / "images" / split_name
        label_output = output_dir / "labels" / split_name

        image_output.mkdir(parents=True, exist_ok=True)
        label_output.mkdir(parents=True, exist_ok=True)

        image_count = 0
        object_count = 0

        for assessment_id in split_ids:
            for annotation_file, data in assessments[assessment_id]:

                image_id = data["image_id"]

                image_file = image_dir / f"{image_id}.jpg"

                if not image_file.exists():
                    print(f"WARNING: Missing image: {image_file}")
                    continue

                # Copy image
                shutil.copy2(
                    image_file,
                    image_output / image_file.name
                )

                # Create YOLO label file
                label_file = label_output / f"{image_id}.txt"

                lines = []

                for obj in data["objects"]:
                    class_name = obj["class"]

                    if class_name not in CLASS_MAP:
                        print(
                            f"WARNING: Unknown class "
                            f"{class_name} in {annotation_file}"
                        )
                        continue

                    class_id = CLASS_MAP[class_name]

                    cx, cy, w, h = convert_bbox(
                        obj["bbox"],
                        data["width"],
                        data["height"]
                    )

                    lines.append(
                        f"{class_id} "
                        f"{cx:.6f} "
                        f"{cy:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f}"
                    )

                    object_count += 1

                with open(label_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))

                image_count += 1

        print(
            f"{split_name}: "
            f"{image_count} images, "
            f"{object_count} objects"
        )

    # ---------------------------------------------------------
    # Process splits
    # ---------------------------------------------------------

    process_split("train", train_ids)
    process_split("val", val_ids)

    # ---------------------------------------------------------
    # Write YOLO dataset config
    # ---------------------------------------------------------

    yaml_file = output_dir / "dataset.yaml"

    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(
            "path: data/yolo\n"
            "train: images/train\n"
            "val: images/val\n\n"
            "names:\n"
            "  0: engine\n"
            "  1: trailer\n"
            "  2: strap\n"
            "  3: pneumatic_suspension\n"
        )

    print(f"\nDataset config written to: {yaml_file}")


if __name__ == "__main__":
    main()
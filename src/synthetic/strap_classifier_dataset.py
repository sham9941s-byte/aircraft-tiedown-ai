import json
import glob
import random
import shutil
from pathlib import Path

import cv2


QUALITY_MAP = {
    "good": 0,
    "loose": 1,
    "unconnected": 2,
    "missing_anchor": 3,
    "occluded": 4,
}


def main():
    annotation_dir = Path("data/synthetic/annotations")
    image_dir = Path("data/synthetic/images")

    output_dir = Path("data/strap_quality")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = sorted(annotation_dir.glob("*.json"))

    # ---------------------------------------------------------
    # Group annotations by assessment
    # ---------------------------------------------------------

    assessments = {}

    for annotation_file in annotations:
        with open(annotation_file, encoding="utf-8") as f:
            data = json.load(f)

        assessments.setdefault(
            data["assessment_id"], []
        ).append((annotation_file, data))

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
    # Process crops
    # ---------------------------------------------------------

    counts = {
        "train": {},
        "val": {},
    }

    for split_name, split_ids in [
        ("train", train_ids),
        ("val", val_ids),
    ]:

        for assessment_id in split_ids:

            for annotation_file, data in assessments[assessment_id]:

                image_path = (
                    image_dir /
                    f"{data['image_id']}.jpg"
                )

                image = cv2.imread(str(image_path))

                if image is None:
                    print(
                        f"WARNING: Could not read {image_path}"
                    )
                    continue

                width = data["width"]
                height = data["height"]

                for obj_index, obj in enumerate(data["objects"]):

                    if obj["class"] != "strap":
                        continue

                    quality = obj.get(
                        "quality",
                        "unknown"
                    )

                    if quality not in QUALITY_MAP:
                        continue

                    x1, y1, x2, y2 = obj["bbox"]

                    # Expand crop to include surrounding connection/anchor context
                    expand = 0.35

                    box_width = x2 - x1
                    box_height = y2 - y1

                    pad_x = box_width * expand
                    pad_y = box_height * expand

                    x1 -= pad_x
                    y1 -= pad_y
                    x2 += pad_x
                    y2 += pad_y

                    # Clamp coordinates to image
                    x1 = max(0, min(int(x1), width - 1))
                    y1 = max(0, min(int(y1), height - 1))
                    x2 = max(0, min(int(x2), width))
                    y2 = max(0, min(int(y2), height))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop = image[y1:y2, x1:x2]

                    class_name = quality

                    class_dir = (
                        output_dir /
                        split_name /
                        class_name
                    )

                    class_dir.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    crop_name = (
                        f"{data['image_id']}"
                        f"_strap_{obj_index}.jpg"
                    )

                    output_path = class_dir / crop_name

                    cv2.imwrite(
                        str(output_path),
                        crop
                    )

                    counts[split_name][quality] = (
                        counts[split_name].get(
                            quality, 0
                        ) + 1
                    )

    # ---------------------------------------------------------
    # Print dataset summary
    # ---------------------------------------------------------

    print("\nCrop counts:")

    for split_name in ("train", "val"):
        print(f"\n{split_name}:")

        total = 0

        for quality, count in sorted(
            counts[split_name].items()
        ):
            print(f"  {quality}: {count}")
            total += count

        print(f"  TOTAL: {total}")


if __name__ == "__main__":
    main()
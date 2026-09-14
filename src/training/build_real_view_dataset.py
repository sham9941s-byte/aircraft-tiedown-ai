from pathlib import Path
import json
import random
import shutil


SOURCE_ROOT = Path("data/training/batches")
OUTPUT_ROOT = Path("data/view_router_real")

TRAIN_RATIO = 0.80
SEED = 42


VIEW_TO_ROUTER = {
    "left": "strap_view",
    "right": "strap_view",
    "close_up": "strap_view",
    "suspension": "suspension",
    "front": "other",
    "rear": "other",
    "whole_truck": "other",
    "unknown": "other",
}


def main():
    random.seed(SEED)

    records = []

    for metadata_file in SOURCE_ROOT.glob(
        "*/metadata/*.json"
    ):
        data = json.loads(
            metadata_file.read_text(
                encoding="utf-8"
            )
        )

        image_id = data.get("image_id")
        view = data.get("view", "unknown")

        router_class = VIEW_TO_ROUTER.get(view)

        if router_class is None:
            continue

        batch_dir = metadata_file.parent.parent

        image_file = (
            batch_dir
            / "images"
            / f"{image_id}.jpg"
        )

        if not image_file.exists():
            continue

        records.append(
            {
                "image": image_file,
                "image_id": image_id,
                "view": view,
                "router_class": router_class,
            }
        )

    print(f"TOTAL IMAGES: {len(records)}")

    if not records:
        raise RuntimeError("No images found.")

    random.shuffle(records)

    split_index = int(
        len(records) * TRAIN_RATIO
    )

    train_records = records[:split_index]
    val_records = records[split_index:]

    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    for split, split_records in [
        ("train", train_records),
        ("val", val_records),
    ]:
        for record in split_records:
            destination_dir = (
                OUTPUT_ROOT
                / split
                / record["router_class"]
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                destination_dir
                / f"{record['image_id']}.jpg"
            )

            shutil.copy2(
                record["image"],
                destination,
            )

    print()
    print("DATASET CREATED")
    print(f"TRAIN: {len(train_records)}")
    print(f"VAL:   {len(val_records)}")

    print()
    print("TRAIN:")
    for name in ["strap_view", "suspension", "other"]:
        count = sum(
            r["router_class"] == name
            for r in train_records
        )
        print(f"  {name}: {count}")

    print()
    print("VAL:")
    for name in ["strap_view", "suspension", "other"]:
        count = sum(
            r["router_class"] == name
            for r in val_records
        )
        print(f"  {name}: {count}")

    print()
    print(f"OUTPUT: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

from src.synthetic.scenarios import SCENARIOS
from src.synthetic.engine import make_engine
from src.synthetic.truck import make_truck
from src.synthetic.tiedown import make_tiedown
from src.synthetic.suspension import make_suspension
from src.synthetic.camera import view_variants
from src.synthetic.renderer import render
from src.annotations.builder import build_annotation


def get_crop(
    view: str,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:

    if view == "left_side":
        return (
            0,
            0,
            int(width * 0.65),
            height,
        )

    if view == "right_side":
        return (
            int(width * 0.35),
            0,
            width,
            height,
        )

    if view == "pneumatic_suspension":
        return (
            int(width * 0.55),
            int(height * 0.52),
            width,
            height,
        )

    if view == "left_closeup":
        return (
            int(width * 0.15),
            int(height * 0.20),
            int(width * 0.60),
            int(height * 0.82),
        )

    if view == "right_closeup":
        return (
            int(width * 0.40),
            int(height * 0.20),
            int(width * 0.85),
            int(height * 0.82),
        )

    # full_trailer
    return (
        0,
        0,
        width,
        height,
    )


def transform_point(
    x: float,
    y: float,
    crop: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int]:

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1

    new_x = (x - crop_x1) * width / crop_width
    new_y = (y - crop_y1) * height / crop_height

    return (
        int(round(new_x)),
        int(round(new_y)),
    )


def transform_bbox(
    bbox: list[int],
    crop: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[int] | None:

    x1, y1, x2, y2 = bbox

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    # Completely outside the source crop.
    if (
        x2 <= crop_x1
        or x1 >= crop_x2
        or y2 <= crop_y1
        or y1 >= crop_y2
    ):
        return None

    # Clip the original bbox to the visible crop first.
    x1 = max(x1, crop_x1)
    y1 = max(y1, crop_y1)
    x2 = min(x2, crop_x2)
    y2 = min(y2, crop_y2)

    p1 = transform_point(
        x1,
        y1,
        crop,
        width,
        height,
    )

    p2 = transform_point(
        x2,
        y2,
        crop,
        width,
        height,
    )

    return [
        max(0, min(width, min(p1[0], p2[0]))),
        max(0, min(height, min(p1[1], p2[1]))),
        max(0, min(width, max(p1[0], p2[0]))),
        max(0, min(height, max(p1[1], p2[1]))),
    ]


def transform_annotation(
    annotation: dict,
    view: str,
    width: int,
    height: int,
) -> dict:

    crop = get_crop(
        view,
        width,
        height,
    )

    transformed = dict(annotation)
    transformed["view"] = view

    transformed_objects = []

    for obj in annotation["objects"]:

        new_obj = dict(obj)

        if "bbox" in new_obj:

            new_bbox = transform_bbox(
                new_obj["bbox"],
                crop,
                width,
                height,
            )

            # Object is outside this camera view.
            if new_bbox is None:
                continue

            new_obj["bbox"] = new_bbox

        transformed_objects.append(new_obj)

    transformed["objects"] = transformed_objects

    return transformed


def generate(
    count: int,
    output: str,
    seed: int = 42,
    width: int = 1600,
    height: int = 900,
):

    out = Path(output)

    image_dir = out / "images"
    annotation_dir = out / "annotations"

    image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotation_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(seed)

    manifest = []

    for i in range(count):

        scenario = SCENARIOS[i % len(SCENARIOS)]

        assessment_id = f"SYN_{i + 1:06d}"

        engine = make_engine(
            width,
            height,
        )

        truck = make_truck(
            width,
            height,
        )

        tiedown = make_tiedown(
            engine,
            truck,
            scenario.straps_per_side,
        )

        suspension = make_suspension(
            truck,
        )

        for view_idx, view in enumerate(
            view_variants()
        ):

            image_id = (
                f"{assessment_id}_{view_idx + 1:02d}"
            )

            image = render(
                width,
                height,
                engine,
                truck,
                tiedown,
                suspension,
                scenario,
                view,
                rng,
            )

            image_path = (
                image_dir /
                f"{image_id}.jpg"
            )

            annotation_path = (
                annotation_dir /
                f"{image_id}.json"
            )

            cv2.imwrite(
                str(image_path),
                image,
            )

            annotation = build_annotation(
                assessment_id,
                image_id,
                view,
                scenario,
                engine,
                truck,
                tiedown,
                suspension,
                width,
                height,
            )

            annotation = transform_annotation(
                annotation,
                view,
                width,
                height,
            )

            annotation_path.write_text(
                json.dumps(
                    annotation,
                    indent=2,
                ),
                encoding="utf-8",
            )

            manifest.append(
                {
                    "assessment_id": assessment_id,
                    "image_id": image_id,
                    "path": str(
                        image_path.as_posix()
                    ),
                    "view": view,
                    "width": width,
                    "height": height,
                    "engine_type": (
                        "synthetic_engine_A"
                    ),
                    "truck_configuration": (
                        "synthetic_truck_A"
                    ),
                    "suspension_type": (
                        "pneumatic"
                    ),
                    "tie_down_state": (
                        "correct"
                        if scenario.expected_decision
                        == "GOOD_TO_GO"
                        else "incorrect"
                    ),
                    "quality_label": (
                        "good"
                        if scenario.strap_state
                        == "good"
                        else scenario.strap_state
                    ),
                    "scenario": scenario.name,
                    "expected_decision": (
                        scenario.expected_decision
                    ),
                    "source": "synthetic",
                    "split": "train",
                }
            )

    (
        out / "manifest.jsonl"
    ).write_text(
        "\n".join(
            json.dumps(row)
            for row in manifest
        )
        + "\n",
        encoding="utf-8",
    )

    return len(manifest)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        default="data/synthetic",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    n = generate(
        args.count,
        args.output,
        args.seed,
    )

    print(
        f"Generated {args.count} assessments / {n} images."
    )

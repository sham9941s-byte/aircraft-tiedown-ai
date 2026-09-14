from pathlib import Path
import uuid
import shutil

from src.evidence.assessment import analyze_assessment


IMAGE_DIR = Path("data/synthetic/images")


def analyze_images(
    image_paths: list[str],
    view_hints: dict | None = None,
):
    """
    Run the aircraft-engine tie-down inspection pipeline.

    view_hints is optional and maps uploaded image index
    to an operator-confirmed inspection view.

    Example:
        {
            "1": "front_truck",
            "2": "pneumatic_suspension",
            "3": "undercarriage",
            "4": "left_side",
            "5": "front_engine",
            "6": "front_closeup",
            "7": "right_side",
            "8": "tie_down_closeup",
            "9": "tie_down_closeup"
        }

    Automatic ML view classification remains the default.
    Operator hints are used only when supplied.
    """

    assessment_id = (
        f"UPLOAD_{uuid.uuid4().hex[:8].upper()}"
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files = []

    try:
        for index, source_path in enumerate(
            image_paths,
            start=1,
        ):
            destination = (
                IMAGE_DIR
                / f"{assessment_id}_{index:02d}.jpg"
            )

            shutil.copy2(
                source_path,
                destination,
            )

            saved_files.append(destination)

        result = analyze_assessment(
            assessment_id,
            view_hints=view_hints,
        )

        return result

    finally:
        # Keep uploaded images for the POC/demo.
        pass
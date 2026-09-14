from pathlib import Path
import json
from collections import Counter

from src.evidence.assessment import analyze_assessment


ANNOTATIONS = Path("data/synthetic/annotations")


def get_ground_truth():
    assessments = {}

    for path in ANNOTATIONS.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assessment_id = data["assessment_id"]

        assessments.setdefault(
            assessment_id,
            data["expected_decision"]
        )

    return assessments


def main():
    ground_truth = get_ground_truth()

    results = []

    print(f"Evaluating {len(ground_truth)} assessments...\n")

    for i, (assessment_id, expected) in enumerate(
        sorted(ground_truth.items()), 1
    ):
        result = analyze_assessment(assessment_id)

        predicted = result["decision"]

        results.append({
            "assessment_id": assessment_id,
            "expected": expected,
            "predicted": predicted,
        })

        print(
            f"[{i:03d}/{len(ground_truth)}] "
            f"{assessment_id}: "
            f"expected={expected} "
            f"predicted={predicted}"
        )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    labels = [
        "GOOD_TO_GO",
        "TIE_DOWN_INCORRECT",
        "MORE_IMAGES_REQUIRED",
    ]

    matrix = Counter(
        (r["expected"], r["predicted"])
        for r in results
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        f"{'Expected / Predicted':<28}"
        f"{'GOOD':>12}"
        f"{'INCORRECT':>15}"
        f"{'MORE IMAGES':>17}"
    )

    for expected in labels:
        print(
            f"{expected:<28}"
            f"{matrix[(expected, 'GOOD_TO_GO')]:>12}"
            f"{matrix[(expected, 'TIE_DOWN_INCORRECT')]:>15}"
            f"{matrix[(expected, 'MORE_IMAGES_REQUIRED')]:>17}"
        )

    # ---------------------------------------------------------
    # OVERALL ACCURACY
    # ---------------------------------------------------------

    correct = sum(
        r["expected"] == r["predicted"]
        for r in results
    )

    accuracy = correct / len(results)

    # ---------------------------------------------------------
    # HACKATHON KPI DEFINITIONS
    # ---------------------------------------------------------

    # False positive:
    # Incorrect tie-down predicted as GOOD_TO_GO
    incorrect_total = sum(
        r["expected"] == "TIE_DOWN_INCORRECT"
        for r in results
    )

    false_positives = sum(
        r["expected"] == "TIE_DOWN_INCORRECT"
        and r["predicted"] == "GOOD_TO_GO"
        for r in results
    )

    false_positive_rate = (
        false_positives / incorrect_total
        if incorrect_total
        else 0.0
    )

    # False negative:
    # Correct tie-down predicted as anything other than GOOD_TO_GO
    good_total = sum(
        r["expected"] == "GOOD_TO_GO"
        for r in results
    )

    false_negatives = sum(
        r["expected"] == "GOOD_TO_GO"
        and r["predicted"] != "GOOD_TO_GO"
        for r in results
    )

    false_negative_rate = (
        false_negatives / good_total
        if good_total
        else 0.0
    )

    # Additional images requested
    additional_images = sum(
        r["predicted"] == "MORE_IMAGES_REQUIRED"
        for r in results
    )

    additional_image_rate = (
        additional_images / len(results)
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("POC RESULTS")
    print("=" * 70)

    print(f"Assessments evaluated:       {len(results)}")
    print(f"Correct decisions:           {correct}")
    print(f"Overall accuracy:             {accuracy:.2%}")

    print()
    print("GOOD TO GO:")
    print(
        f"  Ground truth:              {good_total}"
    )

    print()
    print("TIE-DOWN INCORRECT:")
    print(
        f"  Ground truth:              {incorrect_total}"
    )
    print(
        f"  False positives:           {false_positives}"
    )
    print(
        f"  False-positive rate:       {false_positive_rate:.2%}"
    )

    print()
    print("MORE IMAGES REQUIRED:")
    print(
        f"  Requested:                 {additional_images}"
    )
    print(
        f"  Additional-image rate:     {additional_image_rate:.2%}"
    )

    print()
    print("GOOD-TO-GO FALSE NEGATIVES:")
    print(
        f"  False negatives:            {false_negatives}"
    )
    print(
        f"  False-negative rate:        {false_negative_rate:.2%}"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output = Path("runs/evaluation")
    output.mkdir(parents=True, exist_ok=True)

    with open(
        output / "pipeline_results.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(f"Detailed results saved to: {output / 'pipeline_results.json'}")


if __name__ == "__main__":
    main()
import json
import glob
from collections import defaultdict

from src.decision.engine import assess_tie_down


def build_evidence(files):
    evidence = {
        "image_quality": "good",
        "suspension": None,
        "straps": [],
    }

    for file in files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        # Poor-image scenario
        if data["scenario"] == "poor_image":
            evidence["image_quality"] = "poor"

        for obj in data["objects"]:
            if obj["class"] == "strap":
                evidence["straps"].append({
                    "quality": obj.get("quality", "unknown")
                })

            elif obj["class"] == "pneumatic_suspension":
                evidence["suspension"] = obj.get("quality", "unknown")

    return evidence


# Group the six images belonging to each assessment
assessments = defaultdict(list)

for file in glob.glob("data/synthetic/annotations/*.json"):
    with open(file, encoding="utf-8") as f:
        data = json.load(f)

    assessments[data["assessment_id"]].append(file)


print("\nDECISION ENGINE VALIDATION")
print("=" * 60)

passed = 0

for assessment_id in sorted(assessments):

    files = assessments[assessment_id]

    # Expected decision is identical across the assessment's images.
    with open(files[0], encoding="utf-8") as f:
        expected = json.load(f)["expected_decision"]

    evidence = build_evidence(files)

    actual = assess_tie_down(evidence).value

    status = "PASS" if actual == expected else "FAIL"

    if status == "PASS":
        passed += 1

    print(
        f"{assessment_id} | "
        f"Expected: {expected:22s} | "
        f"Actual: {actual:22s} | "
        f"{status}"
    )


print("=" * 60)
print(f"Passed: {passed}/{len(assessments)}")
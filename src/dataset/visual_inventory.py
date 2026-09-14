from pathlib import Path
import pandas as pd
import cv2

INPUT = Path("data/manifests/host_images.csv")
OUTPUT = Path("data/manifests/host_visual_inventory.csv")

df = pd.read_csv(INPUT)

rows = []

for _, r in df.iterrows():
    path = Path(r["image_path"])
    img = cv2.imread(str(path))

    if img is None:
        continue

    h, w = img.shape[:2]

    # Initial rule-based view hints.
    # These are NOT ground truth annotations.
    if w > h * 1.25:
        aspect_hint = "wide_landscape"
    elif h > w * 1.25:
        aspect_hint = "portrait"
    else:
        aspect_hint = "standard_landscape"

    filename = r["filename"].lower()

    if "whole truck" in filename or "whole" in filename:
        filename_hint = "whole_truck"
    elif "left side" in filename:
        filename_hint = "left_side"
    elif "right side" in filename:
        filename_hint = "right_side"
    else:
        filename_hint = "unspecified"

    rows.append({
        "case_id": r["case_id"],
        "label": r["label"],
        "filename": r["filename"],
        "image_path": r["image_path"],
        "width": w,
        "height": h,
        "orientation": r["orientation"],
        "duplicate": r["duplicate"],
        "aspect_hint": aspect_hint,
        "filename_hint": filename_hint,
        "tie_down_visible": "",
        "suspension_visible": "",
        "engine_visible": "",
        "whole_truck_visible": "",
        "close_up": "",
        "notes": "",
    })

out = pd.DataFrame(rows)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)

print(f"Created: {OUTPUT}")
print(f"Images: {len(out)}")
print(f"Columns: {len(out.columns)}")
print("\nThe blank visual fields are intentional.")
print("We will populate them after visual inspection.")

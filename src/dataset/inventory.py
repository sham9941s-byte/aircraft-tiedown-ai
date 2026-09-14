import csv
from pathlib import Path
import cv2

ROOT = Path("data/raw/host_samples/Sample Images")
OUTPUT = Path("data/manifests/host_inventory.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

rows = []

for image_path in sorted(ROOT.rglob("*.jpg")):
    case_dir = image_path.parent
    label_dir = case_dir.parent

    case_id = case_dir.name
    label = label_dir.name

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"WARNING: Could not read {image_path}")
        continue

    height, width = image.shape[:2]
    aspect_ratio = round(width / height, 4)

    rows.append({
        "case_id": case_id,
        "label": label,
        "filename": image_path.name,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "file_size_bytes": image_path.stat().st_size,
        "image_path": image_path.as_posix(),
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Inventory created: {OUTPUT}")
print(f"Images indexed: {len(rows)}")
print(f"Cases indexed: {len(set(r['case_id'] for r in rows))}")

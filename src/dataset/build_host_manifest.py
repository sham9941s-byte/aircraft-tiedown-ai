from pathlib import Path
import hashlib
import csv
import cv2

ROOT = Path("data/raw/host_samples/Sample Images")
OUTPUT = Path("data/manifests/host_images.csv")

files = sorted(ROOT.rglob("*.jpg"))

# Calculate content hashes
hashes = {}
for p in files:
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes.setdefault(digest, []).append(p)

rows = []

for p in files:
    img = cv2.imread(str(p))

    if img is None:
        print(f"WARNING: Could not read {p}")
        continue

    height, width = img.shape[:2]

    relative = p.relative_to(ROOT)
    label = relative.parts[0]
    case_id = relative.parts[1]

    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    duplicate = len(hashes[digest]) > 1

    if height > width:
        orientation = "portrait"
    elif width > height:
        orientation = "landscape"
    else:
        orientation = "square"

    rows.append({
        "case_id": case_id,
        "label": label,
        "image_path": str(p),
        "filename": p.name,
        "width": width,
        "height": height,
        "orientation": orientation,
        "duplicate": duplicate,
    })

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "case_id",
            "label",
            "image_path",
            "filename",
            "width",
            "height",
            "orientation",
            "duplicate",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Created: {OUTPUT}")
print(f"Rows: {len(rows)}")
print(f"Cases: {len(set(r['case_id'] for r in rows))}")
print(f"Duplicate rows: {sum(r['duplicate'] for r in rows)}")

from pathlib import Path
import hashlib

root = Path("data/raw/host_samples/Sample Images")
files = sorted(root.rglob("*.jpg"))

groups = {}

for p in files:
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    groups.setdefault(h, []).append(p)

duplicates = [paths for paths in groups.values() if len(paths) > 1]

print(f"Total JPG images: {len(files)}")
print(f"Unique image contents: {len(groups)}")
print(f"Duplicate groups: {len(duplicates)}")

if duplicates:
    print("\nDUPLICATES FOUND:\n")
    for i, paths in enumerate(duplicates, 1):
        print(f"Group {i}:")
        for p in paths:
            print(f"  {p}")
        print()
else:
    print("\nNo byte-identical duplicates found.")

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def split_manifest(input_path, output_dir, train=0.70, val=0.15, seed=42):
    rows = [json.loads(x) for x in Path(input_path).read_text().splitlines() if x.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["assessment_id"]].append(row)

    ids = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train)
    n_val = int(n * val)

    assignments = {}
    for aid in ids[:n_train]:
        assignments[aid] = "train"
    for aid in ids[n_train:n_train+n_val]:
        assignments[aid] = "val"
    for aid in ids[n_train+n_val:]:
        assignments[aid] = "test"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)
    for row in rows:
        row["split"] = assignments[row["assessment_id"]]
        buckets[row["split"]].append(row)

    for split, split_rows in buckets.items():
        (out / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(x) for x in split_rows) + "\n",
            encoding="utf-8"
        )

    return {k: len({x["assessment_id"] for x in v}) for k, v in buckets.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/synthetic/manifest.jsonl")
    parser.add_argument("--output", default="data/splits")
    args = parser.parse_args()
    print(split_manifest(args.input, args.output))

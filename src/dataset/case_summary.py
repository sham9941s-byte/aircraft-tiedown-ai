import pandas as pd

path = "data/manifests/host_images.csv"

df = pd.read_csv(path)

summary = (
    df.groupby(["label", "case_id"])
      .agg(
          images=("filename", "count"),
          unique_images=("duplicate", lambda x: (~x).sum()),
          portrait=("orientation", lambda x: (x == "portrait").sum()),
          landscape=("orientation", lambda x: (x == "landscape").sum()),
      )
      .reset_index()
)

print("\nHOST DATASET CASE SUMMARY")
print("=" * 70)
print(summary.to_string(index=False))

print("\nTOTALS")
print("=" * 70)
print(f"Images: {len(df)}")
print(f"Cases: {df['case_id'].nunique()}")
print(f"Unique image contents: {len(df[df['duplicate'] == False])}")

summary.to_csv("data/manifests/host_case_summary.csv", index=False)

print("\nCreated:")
print("data/manifests/host_case_summary.csv")

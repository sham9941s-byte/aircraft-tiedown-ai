# Dataset layout

```text
data/
├── raw/
├── images/
├── annotations/
├── manifests/
├── splits/
└── synthetic/
```

## Recommended manifest fields

Each row should identify the assessment/capture session, not just an image.

Required fields:

- `assessment_id`
- `image_id`
- `path`
- `view`
- `width`
- `height`
- `engine_type`
- `truck_configuration`
- `suspension_type`
- `tie_down_state`
- `quality_label`
- `source` (`synthetic` or `real`)
- `split`

Keep all images from one physical assessment in the same train/validation/test split to prevent leakage.

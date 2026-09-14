# Aircraft Engine Tie-Down AI

Computer Vision for Aircraft Engine Tie-down Process Automation.

## Project goal

Assess a minimum set of operator-captured images and return exactly one of:

- `GOOD_TO_GO`
- `TIE_DOWN_INCORRECT`
- `MORE_IMAGES_REQUIRED`

The ML layer produces visual evidence. A deterministic decision engine owns the final compliance decision.

## Architecture

```text
6+ images
   |
   v
Image Gate
   |-- quality / resolution / blur / coverage
   v
View Classifier
   |
   v
Coarse Detection
   |-- engine / trailer / suspension / tie-down region
   v
Tie-down ROI
   |
   v
Fine Detection / Instance Segmentation
   |-- strap / chain / connection point
   v
Region Quality Classification
   |-- good / bad / unknown
   v
Evidence Graph
   |
   v
Confidence + uncertainty
   |
   v
Deterministic Decision Engine
   |
   +--> GOOD_TO_GO
   +--> TIE_DOWN_INCORRECT
   +--> MORE_IMAGES_REQUIRED
```

## First milestone

Milestone 1 establishes the experiment and data foundation before we lock the final models.

1. Dataset manifest/schema
2. Annotation schema
3. Dataset split policy
4. Synthetic-data specification
5. Benchmark configuration
6. Metrics definition
7. Decision/evidence schema
8. Tests

## Business rules captured from the challenge

- Minimum 6 images are expected for an assessment.
- Required coverage includes both sides, a full trailer view, and a pneumatic suspension view.
- Three straps per side is recommended best practice, but **2 properly secured straps must still result in positive feedback**.
- Final operator-facing outcomes are only the three outcomes listed above.

## Safety principle

Do not use an LLM as the final compliance judge. Vision models generate evidence; deterministic rules consume that evidence.

## Next implementation step

Populate `data/manifests/` with real or synthetic examples, then benchmark candidate models against the project-specific metrics in `configs/benchmark.yaml`.

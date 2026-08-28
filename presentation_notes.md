# Presentation Notes

## Slide Flow

1. Data schema: fabrication parameters, SEM morphology, EDX composition, and electrochemical results share `sample_id`.
2. Pipeline: CSV ingestion -> feature engineering -> RF/GP models -> UCB ranking -> human decision.
3. Dashboard overview: inspect data quality, model metrics, and feature importance.
4. Recommendation view: compare the top three recipes, uncertainty, acquisition score, and confidence.
5. Experimental loop: accept or modify a recipe, fabricate it, measure it, then append the real result for retraining.

## Architecture

```text
Process CSV ----\
SEM features ----+--> Master dataset --> RF baseline --------\
EDX CSV ---------+                       GP surrogate + std ---+--> UCB top 3
Electrochem CSV -/                                            |
                                                              v
SEM images --> threshold segmentation --> morphology CSV   Human decision
                                                              |
                         measured result <--- next experiment -/
```

## One-Minute Demo

This prototype integrates fabrication parameters, SEM-derived morphology,
EDX composition, and electrochemical measurements into one table keyed by
sample ID. A RandomForest provides an explainable baseline, while a Gaussian
Process estimates both expected retention and uncertainty. The recommender
evaluates a constrained recipe grid, removes experiments already present, and
ranks the remaining candidates with an upper-confidence-bound score. The
dashboard exposes the evidence behind the ranking and leaves the final accept,
modify, or reject decision to the researcher. The current values are generated
from synthetic data and demonstrate workflow logic only. After a selected
sample is fabricated, its real SEM, EDX, and cycling measurements should be
added to the next training round.

## Positioning

Choi et al. optimize synthesis conditions toward a target SEM morphology. This
project instead uses SEM and EDX as intermediate evidence while optimizing an
electrochemical target such as capacity retention.

Franco and collaborators demonstrate broad simulation-driven electrode
manufacturing optimization. This project is a smaller, low-data,
experiment-facing workflow for Si/MXene composite development.

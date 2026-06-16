# Daily FID Update (2026-03-23)

## 1) Objective completed today
Implemented and stabilized an end-to-end FID workflow for MeshUp ablations, then executed dev experiments and collected quantitative FID outputs.

## 2) What was implemented

### Core FID evaluation pipeline
- evaluate_fid.py

### Submission and recovery helpers
- submit_ablation_study_dev.sh
- submit_fid_only_dev.sh
- submit_ablation_extra_2700.sh

### Dev ablation scripts using training + FID flow
- jobs/ablation_A1_baseline_dev.sh
- jobs/ablation_A1_dino_dev.sh
- jobs/ablation_A2_bottle_vase_dev.sh
- jobs/ablation_A3_doll_human_dev.sh
- jobs/ablation_A4_truck_dragon_dev.sh
- jobs/ablation_A5_chair_sculpture_dev.sh

### Non-dev extra comparison runs added today (2700 epochs + FID)
- jobs/ablation_B1_truck_phoenix_2700.sh
- jobs/ablation_B2_bottle_teapot_2700.sh
- jobs/ablation_B3_doll_robot_2700.sh

### Stability hardening relevant to failures seen today
- NeuralJacobianFields/PoissonSystem.py
  - Added Cholesky retry with diagonal regularization fallback for non-positive-definite matrix cases.

## 3) Dev FID results collected (lower is better)

| Experiment | FID | JSON Path |
|---|---:|---|
| A4 truck->dragon | 555.1267 | outputs/ablation/ablation_A4_truck_dragon_both_dev/evaluation/fid_results.json |
| A2 bottle->vase | 557.7807 | outputs/ablation/ablation_A2_bottle_vase_both_dev/evaluation/fid_results.json |
| A3 doll->human | 563.6108 | outputs/ablation/ablation_A3_doll_human_both_dev/evaluation/fid_results.json |
| A1 dino | 592.1992 | outputs/ablation/ablation_A1_dino_hound_hippo_dev/evaluation/fid_results.json |
| A1 baseline | 605.1927 | outputs/ablation/ablation_A1_baseline_hound_hippo_dev/evaluation/fid_results.json |

Legacy early baseline run before standardized output root:
- 573.7256 at outputs/ablation_A1_baseline_hound_hippo_dev/evaluation/fid_results.json

A5 dev current status:
- Final rerun completed successfully and produced FID JSON.
- Final dev FID: 528.2438 at outputs/ablation/ablation_A5_chair_sculpture_both_dev/evaluation/fid_results.json

## 4) Execution status snapshot

### Completed dev jobs with FID outputs
- 3740938 (A1 baseline dev)
- 3740939 (A1 dino dev)
- 3740940 (A2 bottle dev)
- 3740941 (A3 doll dev)
- 3741295 (A4 truck dev)
- 3741376 (A5 chair dev)

### A5 dev
- 3741298 failed (pre-patch)
- 3741376 completed (post-patch)

### Non-dev A-series status
- Completed: 3741280 (A1 baseline), 3741281 (A1 dino), 3741283 (A2), 3741285 (A3)
- Completed: 3741286 (A4)
- Pending: 3741288 (A5)

### New B-series queued (2700 epoch comparisons)
- 3741393, 3741394, 3741395

## 5) Interpretation
- Absolute dev FID values are high due to speed-focused dev settings; they are not final-quality numbers.
- Relative ranking is informative and currently indicates measurable differences among prompt/mesh pairings.
- Best current dev score among completed runs: A4 truck->dragon (555.1267).

## 6) Full Ablation Table (Dev + Non-Dev, Updated 2026-03-24)

All values are read from `fid_results.json` files. Lower FID is better.

| Experiment | Dev FID | Non-Dev FID | Delta (Non-Dev - Dev) | Status |
|---|---:|---:|---:|---|
| A1 baseline (hound->hippo) | 605.1927 | 494.8596 | -110.3331 | Both complete |
| A1 dino (hound->hippo) | 592.1992 | 503.3522 | -88.8470 | Both complete |
| A2 bottle->vase | 557.7807 | 462.8312 | -94.9495 | Both complete |
| A3 doll->human | 563.6108 | 594.1737 | +30.5629 | Both complete |
| A4 truck->dragon | 555.1267 | 507.1216 | -48.0051 | Both complete |
| A5 chair->sculpture | 528.2438 | pending | n/a | Non-dev pending |

### Additional Non-Dev 2700-Epoch Comparison Runs

| Experiment | Non-Dev 2700 FID | Status |
|---|---:|---|
| B1 truck->phoenix | pending | Pending |
| B2 bottle->teapot | pending | Pending |
| B3 doll->robot | pending | Pending |

### Raw JSON Sources Used

- outputs/ablation/ablation_A1_baseline_hound_hippo_dev/evaluation/fid_results.json
- outputs/ablation/ablation_A1_baseline_hound_hippo/evaluation/fid_results.json
- outputs/ablation/ablation_A1_dino_hound_hippo_dev/evaluation/fid_results.json
- outputs/ablation/ablation_A1_dino_hound_hippo/evaluation/fid_results.json
- outputs/ablation/ablation_A2_bottle_vase_both_dev/evaluation/fid_results.json
- outputs/ablation/ablation_A2_bottle_vase_both/evaluation/fid_results.json
- outputs/ablation/ablation_A3_doll_human_both_dev/evaluation/fid_results.json
- outputs/ablation/ablation_A3_doll_human_both/evaluation/fid_results.json
- outputs/ablation/ablation_A4_truck_dragon_both_dev/evaluation/fid_results.json
- outputs/ablation/ablation_A5_chair_sculpture_both_dev/evaluation/fid_results.json

### Commentary for Supervisor

- There is a clear dev-to-non-dev improvement trend for A1 baseline, A1 dino, and A2 (FID decreases by ~89 to ~110 points).
- A2 currently has the strongest complete non-dev result (462.8312), outperforming both A1 variants in this batch.
- A3 shows a regression from dev to non-dev (+30.56), indicating this pair may be less stable and deserves parameter review.
- A4 non-dev is now complete and improves over dev by ~48.01 points (507.1216 vs 555.1267).
- A5 non-dev is still pending, so final full ranking remains provisional.
- 2700-epoch B-series runs were submitted to expand comparison coverage and are still pending.

# Gated Hybrid Semantic Deformation Plan

Date: 2026-06-20

## Diagnosis

The previous cross-animal runs used aligned joint PartField buckets, hard bucket
Chamfer, directional asymmetry, neighboring-face Jacobian smoothing, and a local
displacement-jump guard. This is better than global Chamfer, but still brittle:
inside each hard bucket, nearest-neighbor geometry can create many-to-one pulls.
That is the pattern seen in the bulldog-to-dachshund spike diagnosis, and it is
more severe for cat/dog pairs because corresponding parts are less isometric.

## Literature Signal

- PartField argues that its continuous feature field has cross-shape consistency
  useful for correspondence, not only discrete clustering:
  https://arxiv.org/abs/2504.11451
- NFR's key lesson is to use neural features inside an iterative geometric
  registration loop, with correspondences filtered by consistency:
  https://arxiv.org/abs/2505.22445
- DenseMatcher similarly combines semantic vertex features with a map/refinement
  stage for cross-category correspondences:
  https://arxiv.org/abs/2412.05268
- Zero-Shot 3D Shape Correspondence uses semantic coarse regions first, then
  refines to dense correspondences; hard region IDs alone are not enough:
  https://arxiv.org/abs/2306.03253
- Deep Shells supports multiscale differentiable matching rather than one hard
  nearest-neighbor scale:
  https://arxiv.org/abs/2010.15261
- ARAPReg reinforces the need for local rigidity preservation during animal
  deformation:
  https://arxiv.org/abs/2108.09432

## Implemented Change

The PartField loss now supports feature-guided hard-bucket matching:

- geometric loss is still measured in 3D,
- nearest neighbors inside a bucket can be selected by geometry plus continuous
  PartField feature distance,
- source samples not selected by reverse matching can be downweighted,
- ambiguous semantic matches can be confidence-gated by top-1/top-2 feature
  margin,
- the soft feature term uses the same directional asymmetry and confidence gate.

The cross-animal default recipe now uses:

- `PARTFIELD_GUIDANCE_MODE=hybrid`
- `PARTFIELD_HARD_WEIGHT=0.70`
- `PARTFIELD_SOFT_WEIGHT=0.30`
- `PARTFIELD_HARD_SEMANTIC_WEIGHT=1.50`
- `PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=0.35`
- `PARTFIELD_SEMANTIC_CONFIDENCE_MARGIN=0.04`
- `PARTFIELD_SEMANTIC_CONFIDENCE_FLOOR=0.35`
- `EDGE_STRETCH_WEIGHT=150.0`
- `EDGE_STRETCH_THRESHOLD=1.35`

Outputs go to:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/gatedhybrid_asym035_jump500
```

## Next Experiment

Submit the full cross-animal sweep:

```bash
pf_job=$(sbatch --parsable jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_prepare_partfield_cross_animals.sh)
def_job=$(sbatch --parsable --dependency=afterok:${pf_job} --array=0-5 jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_cross_animal_best_array.sh)
sbatch --dependency=afterok:${def_job} jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_analyze_cross_animal_outputs.sh
```

If the result is too conservative, increase `PARTFIELD_CHAMFER_WEIGHT_OVERRIDE`
back toward `8000` or lower `EDGE_STRETCH_WEIGHT`. If spikes remain, lower
`PARTFIELD_SOURCE_TO_TARGET_WEIGHT` to `0.20` and increase
`PARTFIELD_SEMANTIC_CONFIDENCE_MARGIN` to `0.06`.

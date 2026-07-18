# Bodyfix Hard Recipe

Date: 2026-06-20

## Why Gated-Hybrid Failed

The gated-hybrid run removed spikes, but it also removed too much target-shape
pressure:

- global Chamfer was lowered from `750` to `650`,
- PartField Chamfer was lowered from `8000` to `7000`,
- soft/hybrid PartField matching blurred the target pull,
- semantic confidence gating downweighted many cross-animal matches,
- edge-stretch regularization blocked legitimate body elongation.

The optimizer therefore found compact, smooth, low-detail shapes: fewer spikes,
but scrunched bodies.

## Corrected Recipe

The next recipe keeps the old hard-bucket body-forming behavior and only changes
the spike controls:

- `GLOBAL_CHAMFER_WEIGHT_OVERRIDE=750.0`
- `PARTFIELD_CHAMFER_WEIGHT_OVERRIDE=8000.0`
- `PARTFIELD_GUIDANCE_MODE=hard`
- `PARTFIELD_HARD_WEIGHT=1.0`
- `PARTFIELD_SOFT_WEIGHT=0.0`
- `PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.30`
- `PARTFIELD_TARGET_TO_SOURCE_WEIGHT=1.0`
- `PARTFIELD_SRC_TO_TGT_UNMATCHED_WEIGHT=0.25`
- `JACOBIAN_NEIGHBOR_SMOOTH_WEIGHT=1500.0`
- `EDGE_STRETCH_WEIGHT=0.0`
- `EDGE_DISPLACEMENT_JUMP_WEIGHT=1200.0`
- `EDGE_DISPLACEMENT_JUMP_THRESHOLD=0.65`
- `EDGE_DISPLACEMENT_JUMP_MAX_WEIGHT=5.0`

Outputs go to:

```text
jobs_with_target_guidance/cross_animal_spike_runs/outputs/bodyfix_hard_asym030_jump1200
```

This should preserve the stronger body formation from `best_asym035_jump500`
while applying spike pressure through local displacement discontinuity rather
than through global semantic match suppression or edge-stretch penalties.

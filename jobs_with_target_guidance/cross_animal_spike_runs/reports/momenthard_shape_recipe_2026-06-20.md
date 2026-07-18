# Moment-Hard Target-Shape Recipe

Created: 2026-06-20

## Diagnosis

- `best_asym035_jump500` had enough target pull to form the body, but the hard nearest-neighbor bucket loss could create isolated spikes and wrong part bridges.
- `gatedhybrid_asym035_jump500` suppressed those artifacts, but semantic confidence gating plus edge stretch regularization over-damped the deformation and produced shrunken, low-detail bodies.
- `bodyfix_hard_asym030_jump1200` removed spikes and scrunching, but the source-to-target pressure was too weak and local smoothing was too strong, so the output stayed source-like.

## Change

The new `momenthard_asym040_shape` recipe adds a per-PartField-bucket moment loss:

- match each semantic bucket centroid to the target bucket centroid
- match each semantic bucket RMS spread to the target bucket RMS spread
- keep hard PartField Chamfer for surface coverage and detail
- keep soft/hybrid PartField guidance disabled
- reduce local smoothing and jump regularization compared with `bodyfix`

This follows the same high-level lesson as semantic-aware deformation papers: use semantic part consistency for large-scale structure, while preserving detail with local deformation regularization. Useful references:

- PartField: learned part features are consistent enough across shapes for co-segmentation and correspondence: https://arxiv.org/abs/2504.11451
- Semantic-aware implicit template learning: part deformation, global deformation, and scaling regularization improve plausible correspondence under large structural variation: https://arxiv.org/abs/2308.11916
- Neural Jacobian Fields: gradient/Jacobian-domain deformation is detail-preserving and triangulation-agnostic: https://arxiv.org/abs/2205.02904

## Defaults

- output root: `jobs_with_target_guidance/cross_animal_spike_runs/outputs/momenthard_asym040_shape`
- run tag: `cross_animals_momenthard_dev_h100`
- global Chamfer weight: `900`
- PartField Chamfer weight: `8500`
- PartField mode: `hard`
- source-to-target weight: `0.40`
- target-to-source weight: `1.0`
- unmatched source-to-target weight: `0.45`
- PartField moment weight: `0.20`
- PartField moment spread weight: `0.60`
- Jacobian neighbor smoothing: `900`
- displacement jump: weight `900`, threshold `0.75`, max weight `3.0`
- edge stretch: disabled

# Bulldog PartField Spike Follow-up: Robust Chamfer and OT

Date: 2026-06-19

## Starting point

Best previous run:

```text
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_bulldog_pf_chamfer_sweep_single.sh jump 1 500
```

Final mesh:

```text
jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_500/bulldog_to_dachshund_artur_pf_chamfer_jump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_jump500_2500ep_5036806/mesh_final/mesh.obj
```

The correspondence probe in the previous report showed the same pattern on the main spike vertices: source samples found a forward target match, but reverse target-to-source matching did not claim them.

## Implemented experiments

The PartField Chamfer now supports these opt-in controls:

- `partfield_tgt_to_src_robust_scale`: Geman-McClure robust target-to-source Chamfer. `0` disables it. Negative values use the per-bucket median nearest-neighbor distance as the scale.
- `partfield_src_to_tgt_unmatched_weight`: downweights source-to-target samples that are not selected by any reverse target-to-source nearest neighbor.
- `partfield_guidance_mode=hard_unbalanced`: adds an unbalanced Sinkhorn transport term inside hard buckets.
- `partfield_unbalanced_transport_weight` and `partfield_unbalanced_transport_rho`: weight and marginal relaxation for the unbalanced transport term.

## Dev H100 runs

All runs used the jump regularizer with weight `500` and 30-minute dev H100 settings.

| Variant | Job | Main change | Final mesh extents | Result |
|---|---:|---|---|---|
| `robust_jump` | 5078114 | auto median Geman-McClure target-to-source | X `[-0.4951, 0.4951]`, Y `[-0.4483, 0.4483]`, Z `[-0.1605, 0.1605]` | collapsed/shrunk; spikes worse |
| `unbalanced_jump` | 5078117 | hard Chamfer plus `0.20` unbalanced Sinkhorn, rho `0.30` | X `[-0.5363, 0.5363]`, Y `[-0.5502, 0.5502]`, Z `[-0.1823, 0.1823]` | collapsed/shrunk; spikes worse |
| `mutual_jump` | 5078767 | source-to-target unmatched weight `0.15` | X `[-0.8143, 0.8143]`, Y `[-0.6467, 0.6467]`, Z `[-0.2535, 0.2535]` | stable scale, not better than asymmetry |
| `asym_jump` | 5078963 | source-to-target weight `0.20`, target-to-source `1.0` | X `[-0.7670, 0.7670]`, Y `[-0.5815, 0.5815]`, Z `[-0.2433, 0.2433]` | best for vertex `198`, mixed for rear paw/ear vertex `3368` |

## Spike score comparison

Lower is better. `jump500` and `asym035` are from `bulldog_to_dachshund_spike_diagnosis_report_2026-06-16.md`.

| Variant | v198 | v38 | v546 | v717 | v3368 | v3572 | v3365 | v3701 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `jump500` | 2.2930 | 1.0690 | 0.7274 | 0.3289 | 0.7599 | 0.1745 | 0.5451 | 0.3169 |
| `asym035` | 2.1411 | 1.1988 | 0.6911 | 0.5084 | 0.6460 | 0.2415 | 0.6648 | 0.3257 |
| `robust_jump` | 2.8688 | 1.3818 | 0.9977 | n/a | 2.5089 | 0.9640 | n/a | 0.7280 |
| `unbalanced_jump` | 2.9240 | 1.4362 | 1.0573 | n/a | 2.8383 | 1.3285 | 0.7155 | 0.8881 |
| `mutual_jump` | 2.4360 | 1.3504 | 0.8672 | 0.6436 | 0.7806 | n/a | 0.7374 | n/a |
| `asym020` | 2.1323 | 1.2580 | 0.6831 | 0.5387 | 0.9579 | 0.3518 | 0.7165 | 0.4009 |

## Interpretation

The supervisor's robust target-to-source cap is a useful negative result for this case. It reduces the cost of uncovered target regions too much, so the optimizer can shrink the source rather than maintain coverage.

The OT direction is also negative in this implementation. Entropic OT is attractive because it avoids pure nearest-neighbor many-to-one matching, but the tested unbalanced term still behaved like a global bucket-scale pull and encouraged shrinkage. This matches the practical warning from Sinkhorn literature: entropic smoothing gives a differentiable large-scale solver, but the exact bias and mass handling matter.

The most reliable practical control remains directional asymmetry plus the displacement-jump regularizer. The new `asym020` run slightly improves the main vertex `198` over `asym035`, but it worsens vertex `3368`. If choosing one run by these metrics, keep `asym035` as the default follow-up to the original `jump500`; use `asym020` only if visual inspection says the left-ear/leg spike is the dominant artifact.

## Recommendation

Do not use `robust_jump` or `unbalanced_jump` for the bulldog-to-dachshund run as tested.

Recommended next run:

```bash
sbatch jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_bulldog_pf_chamfer_sweep_single.sh asym_jump 1 500
```

This uses source-to-target weight `0.35` by default in the job script. If the visually worst artifact is vertex `198`, rerun with:

```bash
sbatch --export=ALL,PARTFIELD_SOURCE_TO_TARGET_WEIGHT=0.20 jobs_with_target_guidance/artur_soft_runs/jobs/job_dev_h100_bulldog_pf_chamfer_sweep_single.sh asym_jump 1 500
```

## References

- Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances", 2013: https://arxiv.org/abs/1306.0895
- Chizat, Peyre, Schmitzer, Vialard, "Scaling Algorithms for Unbalanced Transport Problems", 2016: https://arxiv.org/abs/1607.05816
- Feydy et al., "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences", 2018: https://arxiv.org/abs/1810.08278
- Achlioptas et al., "Learning Representations and Generative Models for 3D Point Clouds", 2017: https://arxiv.org/abs/1707.02392

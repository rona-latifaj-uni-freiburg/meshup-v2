# Run Locations and Simple Interpretation

Date: 2026-06-20

## Previous bulldog-to-dachshund runs

These are the runs already completed during the spike investigation.

| Run | Job | What changed | Final mesh |
|---|---:|---|---|
| Original best `jump500` | 5036806 | hard PartField Chamfer, global Chamfer, neighbor smooth `1000`, displacement-jump `500` | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_jump_500/bulldog_to_dachshund_artur_pf_chamfer_jump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_jump500_2500ep_5036806/mesh_final/mesh.obj` |
| Best overall `asym035` | 5037590 | same as `jump500`, but source-to-target PartField weight reduced to `0.35` | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5037590/mesh_final/mesh.obj` |
| `robust_jump` | 5078114 | target-to-source Geman-McClure robust cap, automatic median scale | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_robust_jump_500/bulldog_to_dachshund_artur_pf_chamfer_robustjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_robustt2sauto_jump500_2500ep_5078114/mesh_final/mesh.obj` |
| `unbalanced_jump` | 5078117 | hard Chamfer plus unbalanced Sinkhorn transport weight `0.20`, rho `0.30` | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_unbalanced_jump_500/bulldog_to_dachshund_artur_pf_chamfer_unbalancedjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_unbalanced020_rho030_jump500_2500ep_5078117/mesh_final/mesh.obj` |
| `mutual_jump` | 5078767 | downweight source samples not selected by reverse nearest-neighbor matching | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_mutual_jump_500/bulldog_to_dachshund_artur_pf_chamfer_mutualjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_mutual015_jump500_2500ep_5078767/mesh_final/mesh.obj` |
| `asym020` | 5078963 | source-to-target PartField weight `0.20` instead of `0.35` | `jobs_with_target_guidance/artur_soft_runs/outputs_dev_bulldog_partfield_chamfer_asym_jump_500/bulldog_to_dachshund_artur_pf_chamfer_asymjump500_dev_h100_single_hard_partfield_chamfer_only_pf_chamfer_jneighbor1000_asym035_jump500_2500ep_5078963/mesh_final/mesh.obj` |

Simple interpretation:

- The robust target-to-source cap made the shape shrink and made the main spike scores worse.
- The unbalanced OT/Sinkhorn variant also shrank the shape and made the spikes worse.
- The reverse-unmatched weighting was stable, but not better than simple asymmetry.
- The best practical setting remains `asym035`: reduce source-to-target PartField pressure, keep target-to-source coverage strong, and keep the displacement-jump regularizer.

## New cross-animal sweep

Experiment folder:

```text
jobs_with_target_guidance/cross_animal_spike_runs
```

Fresh PartField job:

```text
5088566
```

PartField outputs:

```text
jobs_with_target_guidance/cross_animal_spike_runs/partfield/features/no_dino_animals
jobs_with_target_guidance/cross_animal_spike_runs/partfield/segments/no_dino_animals_12
```

The deformation recipe is:

- global Chamfer weight `750`
- hard PartField Chamfer weight `8000`
- source-to-target PartField weight `0.35`
- target-to-source PartField weight `1.0`
- Jacobian neighbor smooth `1000`
- displacement-jump weight `500`
- SDS, source DINO, and target-render guidance disabled

Submitted deformation jobs so far:

| Pair ID | Direction | Job | Output prefix |
|---:|---|---:|---|
| 0 | dachshund to golden retriever | 5088627 | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/dachshund_to_golden_retriever_*_5088627` |
| 1 | golden retriever to dachshund | 5088628 | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/golden_retriever_to_dachshund_*_5088628` |
| 2 | dachshund to cat | 5088629 | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/dachshund_to_cat_*_5088629` |
| 3 | cat to dachshund | 5088630 | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/cat_to_dachshund_*_5088630` |
| 4 | bulldog to cat | pending QOS slot | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/bulldog_to_cat_*` |
| 5 | cat to bulldog | pending QOS slot | `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/cat_to_bulldog_*` |

The dev QOS only allowed four queued deformation jobs at once. The runtime pair script now has a small hook: when pair ID `3` finishes, it will submit pair IDs `4` and `5`, then submit the analysis/report job after pair ID `5`. The submitted IDs will be written to:

```text
jobs_with_target_guidance/cross_animal_spike_runs/reports/auto_submitted_remaining_jobs.txt
```

Each completed deformation run will contain:

- `mesh_final/mesh.obj`: final deformed mesh.
- `config.yml`: exact settings for that run.
- `epoch_renders/`: saved render grids.
- `displacement_viz/outlier_analysis/summary.md`: simple spike report after the analysis job runs.
- `evaluation/metrics.json`: global geometry metrics after the analysis job runs.

Simple interpretation rule for the cross-animal runs:

- If the final mesh keeps the target silhouette without long isolated vertices, the asymmetry/jump recipe is behaving well.
- If a run shrinks globally, the target-to-source coverage pressure is too weak or the matching is collapsing.
- If only one limb/ear shoots out while the rest looks correct, that is the same one-way Chamfer spike pattern as bulldog-to-dachshund.
- Compare both directions of the same pair. If only one direction spikes, topology/source shape is the trigger; if both spike, the target pair's PartField buckets are probably mismatched.

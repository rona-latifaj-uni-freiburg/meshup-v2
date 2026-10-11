# DenseCorr3D AUC/Err Recompute, 2026-08-04

This recompute uses the current finished DenseCorr3D deformation meshes visible in:

`jobs_with_target_guidance/cross_animal_spike_runs/outputs`

Scoring file:

`jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densecorr_all_finished_20260804_kim_graph_rawgroups.csv`

Full evaluator report:

`jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densecorr_all_finished_20260804_kim_graph_rawgroups.md`

The evaluator uses DenseCorr3D raw `groups.txt` labels only for scoring. It induces a correspondence from each deformed source vertex to the nearest target vertex, then measures normalized target-surface graph geodesic distance to the nearest vertex in the correct DenseCorr3D semantic group. AUC is integrated over normalized geodesic thresholds from 0.00 to 0.25. Err is mean normalized geodesic error times 100.

DenseMatcher reference numbers used for comparison:

| Reference | AUC | Err |
|---|---:|---:|
| DenseMatcher reported DenseCorr3D overall | 0.845 | 1.74 |

## Main Answer

Yes, the current MeshUp results beat the DenseMatcher reported overall AUC/Err on the evaluated set by mean score. The fairest claim is the PartField-guided result:

| Setting | n scored | Mean AUC | Mean Err | Mean p95 Err | Mean same-bucket rate | Rows beating AUC | Rows beating Err | Rows beating both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fair PartField-guided only | 12 | 0.9550 | 1.1728 | 8.8047 | 0.8310 | 12/12 | 11/12 | 11/12 |
| Fair PartField-guided, best PF per pair | 4 | 0.9647 | 0.8842 | - | 0.8499 | 4/4 | 4/4 | 4/4 |
| New all-animal DenseCorr bucket-balanced chain so far | 67 | 0.9921 | 0.2016 | 1.0914 | 0.9338 | 67/67 | 67/67 | 67/67 |
| All scored DenseCorr3D runs together | 129 | 0.9838 | 0.4196 | 2.7327 | 0.9111 | 129/129 | 126/129 | 126/129 |

Important interpretation:

- The fair PartField-guided number is the one to use for the non-cheating comparison, because DenseCorr3D labels were not used as the deformation labels.
- The all-animal bucket-balanced chain is excellent numerically, but it is DenseCorr-label-guided and should be presented as an oracle-style or upper-bound result.
- The full all-runs aggregate mixes fair, selected, DenseCorr-guided, and repair experiments, so it is useful for internal monitoring but not the cleanest paper comparison.

## Fair PartField-Guided Breakdown

| Subset | n | Mean AUC | Mean Err | Mean same-bucket rate | Rows beating both |
|---|---:|---:|---:|---:|---:|
| PF06 | 4 | 0.9397 | 1.6483 | 0.8033 | 3/4 |
| PF07 | 4 | 0.9607 | 0.9839 | 0.8383 | 4/4 |
| PF08 | 4 | 0.9646 | 0.8863 | 0.8515 | 4/4 |

Best fair row per pair:

| Pair | Best setting | AUC | Err |
|---|---|---:|---:|
| bear_to_cheetah | PF08 | 0.9519 | 1.2030 |
| cheetah_to_bear | PF07 | 0.9673 | 0.8159 |
| cheetah_to_panther | PF08 | 0.9606 | 0.9940 |
| panther_to_cheetah | PF08 | 0.9790 | 0.5237 |

The one fair row that does not beat DenseMatcher Err is:

| Run | AUC | Err | p95 Err | same-bucket rate |
|---|---:|---:|---:|---:|
| cheetah_to_panther_densecorr3d_partfield_guided_fair_pf06_4000 | 0.8920 | 3.2621 | 30.8064 | 0.7929 |

That row still beats DenseMatcher AUC, but misses DenseMatcher Err. PF07 and PF08 for the same pair both recover below Err 1.74.

## New All-Animal Chain Snapshot

This is the current all-animal chain result as of the snapshot used for the recompute: 67 completed `mesh_final/mesh.obj` files from 90 planned directed pairs.

| Setting | n | Mean AUC | Mean Err | Min AUC | Max Err | Rows beating both |
|---|---:|---:|---:|---:|---:|---:|
| densecorr3d_all_animals_bucketbalanced_5k_best_asym035 | 67 | 0.9921 | 0.2016 | 0.9773 | 0.6205 | 67/67 |

Selected important pairs:

| Pair | AUC | Err | p95 Err | same-bucket rate | target verts |
|---|---:|---:|---:|---:|---:|
| elephant_to_giraffe | 0.9821 | 0.4908 | 1.3485 | 0.9168 | 4998 |
| giraffe_to_elephant | 0.9933 | 0.1669 | 1.4354 | 0.9248 | 4998 |
| elephant_to_moose | 0.9939 | 0.1521 | 1.0162 | 0.9432 | 5001 |
| elephant_to_panther | 0.9927 | 0.1836 | 1.4639 | 0.9300 | 5001 |
| panther_to_giraffe | 0.9961 | 0.0985 | 0.0000 | 0.9526 | 4999 |
| panther_to_cheetah | 0.9974 | 0.0646 | 0.0000 | 0.9604 | 4999 |
| bear_to_moose | 0.9951 | 0.1546 | 0.0000 | 0.9700 | 5002 |

## Caveat For Supervisor

The clean claim is:

> On the currently evaluated selected DenseCorr3D animal deformation set, the fair PartField-guided MeshUp runs reach mean AUC 0.955 and mean Err 1.17, compared with DenseMatcher's reported DenseCorr3D overall AUC 0.845 and Err 1.74. All fair rows beat the DenseMatcher AUC, and 11/12 beat the DenseMatcher Err. The one Err failure is a PF6 cheetah-to-panther run; PF7/PF8 for the same pair beat the Err baseline.

Do not present the 0.9921 / 0.2016 all-animal chain as the fair comparison, because those jobs use DenseCorr3D group labels to guide the deformation. Present that result as an oracle-style upper bound showing what the optimizer can do when supervised semantic buckets are available.

Two full-resolution requested-animal meshes were skipped by the recompute because their target meshes exceeded the evaluator safety cap:

- elephant_to_giraffe_full target: 130100 vertices
- giraffe_to_elephant_full target: 190950 vertices

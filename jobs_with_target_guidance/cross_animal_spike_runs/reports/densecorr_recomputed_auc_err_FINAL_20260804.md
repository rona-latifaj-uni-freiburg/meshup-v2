# Final DenseCorr3D / DenseMatcher-Mesh AUC-Err Stats

Date: 2026-08-04

Final recompute inputs:

- all DenseCorr3D final mesh paths: `154`
- scored rows: `152`
- skipped rows: `2` full-resolution requested-animal runs above the evaluator safety cap
- completed all-animal directed chain: `90 / 90`

Scored CSV:

`jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densecorr_all_finished_FINAL_20260804_kim_graph_rawgroups.csv`

Full evaluator Markdown:

`jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densecorr_all_finished_FINAL_20260804_kim_graph_rawgroups.md`

DenseMatcher reference used for comparison:

| Reference | AUC | Err |
|---|---:|---:|
| DenseMatcher reported DenseCorr3D overall | 0.845 | 1.74 |

## Headline Results

| Setting | n | Mean AUC | Mean Err | Mean p95 Err | Mean same-bucket rate | Rows beating AUC | Rows beating Err | Rows beating both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fair PartField-guided only | 12 | 0.9550 | 1.1728 | 8.8047 | 0.8310 | 12/12 | 11/12 | 11/12 |
| Fair PartField-guided, best PF per pair | 4 | 0.9647 | 0.8842 | 6.1653 | 0.8499 | 4/4 | 4/4 | 4/4 |
| Final all-animal DenseCorr-guided bucket-balanced chain | 90 | 0.9915 | 0.2159 | 1.1863 | 0.9285 | 90/90 | 90/90 | 90/90 |
| All scored DenseCorr3D rows | 152 | 0.9847 | 0.3951 | 2.5406 | 0.9114 | 152/152 | 149/152 | 149/152 |

## Interpretation

The fair claim is:

> In the fair PartField-guided setting, MeshUp reaches mean AUC 0.9550 and mean Err 1.1728 on the selected DenseCorr3D animal deformation set, compared with DenseMatcher's reported DenseCorr3D overall AUC 0.845 and Err 1.74. All 12 fair rows beat DenseMatcher AUC, and 11/12 beat DenseMatcher Err.

The strongest final all-animal chain result is:

> The completed 90-pair all-animal DenseCorr-guided bucket-balanced chain reaches mean AUC 0.9915 and mean Err 0.2159, and every one of the 90 directed pairs beats the DenseMatcher AUC/Err reference.

Use the second sentence as an upper-bound/oracle-style result, because those all-animal chain jobs use DenseCorr3D semantic groups to guide the deformation.

## Fair PartField-Guided Results

| Subset | n | Mean AUC | Mean Err | Mean p95 Err | Mean same-bucket rate | Rows beating both |
|---|---:|---:|---:|---:|---:|---:|
| PF06 | 4 | 0.9397 | 1.6483 | 13.3962 | 0.8033 | 3/4 |
| PF07 | 4 | 0.9607 | 0.9839 | 6.8064 | 0.8383 | 4/4 |
| PF08 | 4 | 0.9646 | 0.8863 | 6.2115 | 0.8515 | 4/4 |

Fair rows by pair:

| Pair | n | Mean AUC | Mean Err | Mean same-bucket rate | Rows beating both |
|---|---:|---:|---:|---:|---:|
| bear_to_cheetah | 3 | 0.9488 | 1.2813 | 0.8037 | 3/3 |
| cheetah_to_bear | 3 | 0.9660 | 0.8492 | 0.8530 | 3/3 |
| cheetah_to_panther | 3 | 0.9363 | 1.7858 | 0.8418 | 2/3 |
| panther_to_cheetah | 3 | 0.9690 | 0.7750 | 0.8257 | 3/3 |

Best fair row per pair:

| Pair | Best setting | AUC | Err | same-bucket rate |
|---|---|---:|---:|---:|
| bear_to_cheetah | PF08 | 0.9519 | 1.2030 | 0.8127 |
| cheetah_to_bear | PF07 | 0.9673 | 0.8159 | 0.8595 |
| cheetah_to_panther | PF08 | 0.9606 | 0.9940 | 0.8728 |
| panther_to_cheetah | PF08 | 0.9790 | 0.5237 | 0.8545 |

Fair failure row:

| Run | AUC | Err | p95 Err | same-bucket rate |
|---|---:|---:|---:|---:|
| cheetah_to_panther_densecorr3d_partfield_guided_fair_pf06_4000 | 0.8920 | 3.2621 | 30.8064 | 0.7929 |

## Final 90-Pair DenseCorr-Guided Chain

| Metric | Value |
|---|---:|
| n directed pairs | 90 |
| mean AUC | 0.9915 |
| mean Err | 0.2159 |
| mean p95 Err | 1.1863 |
| mean same-bucket rate | 0.9285 |
| min AUC | 0.9606 |
| max Err | 1.0040 |
| rows beating DenseMatcher AUC | 90/90 |
| rows beating DenseMatcher Err | 90/90 |

Worst all-animal chain rows by Err:

| Pair | AUC | Err | p95 Err | same-bucket rate |
|---|---:|---:|---:|---:|
| dinosaur_to_animal4e94a | 0.9606 | 1.0040 | 4.5965 | 0.8314 |
| elephant_to_animal4e94a | 0.9773 | 0.6205 | 2.8680 | 0.8924 |
| animal188f4_to_animal4e94a | 0.9785 | 0.5593 | 2.8160 | 0.8887 |
| elephant_to_giraffe | 0.9821 | 0.4908 | 1.3485 | 0.9168 |
| bear_to_animal4e94a | 0.9841 | 0.4382 | 2.2290 | 0.9080 |

Best all-animal chain rows by Err:

| Pair | AUC | Err | p95 Err | same-bucket rate |
|---|---:|---:|---:|---:|
| moose_to_bear | 0.9980 | 0.0509 | 0.0000 | 0.9652 |
| bear_to_animal13cf7 | 0.9979 | 0.0538 | 0.0000 | 0.9676 |
| moose_to_animal13cf7 | 0.9976 | 0.0601 | 0.0000 | 0.9622 |
| panther_to_bear | 0.9975 | 0.0615 | 0.0000 | 0.9612 |
| animal13cf7_to_bear | 0.9974 | 0.0638 | 0.0000 | 0.9626 |

Important named all-animal pairs:

| Pair | AUC | Err | p95 Err | same-bucket rate | target verts |
|---|---:|---:|---:|---:|---:|
| elephant_to_giraffe | 0.9821 | 0.4908 | 1.3485 | 0.9168 | 4998 |
| giraffe_to_elephant | 0.9933 | 0.1669 | 1.4354 | 0.9248 | 4998 |
| elephant_to_moose | 0.9939 | 0.1521 | 1.0162 | 0.9432 | 5001 |
| moose_to_elephant | 0.9938 | 0.1561 | 0.9950 | 0.9416 | 5001 |
| elephant_to_panther | 0.9927 | 0.1836 | 1.4639 | 0.9300 | 5001 |
| panther_to_elephant | 0.9901 | 0.2460 | 1.7307 | 0.9036 | 5001 |
| panther_to_giraffe | 0.9961 | 0.0985 | 0.0000 | 0.9526 | 4999 |
| giraffe_to_panther | 0.9965 | 0.0875 | 0.0000 | 0.9576 | 4999 |
| panther_to_cheetah | 0.9974 | 0.0646 | 0.0000 | 0.9604 | 4999 |
| cheetah_to_panther | 0.9957 | 0.1076 | 0.0000 | 0.9582 | 4999 |
| bear_to_cheetah | 0.9950 | 0.1256 | 0.8957 | 0.9474 | 4999 |
| cheetah_to_bear | 0.9961 | 0.0984 | 0.9321 | 0.9478 | 4999 |
| bear_to_moose | 0.9959 | 0.1304 | 0.0000 | 0.9726 | 5002 |

## Skipped Rows

These were not scored because the target meshes are full-resolution and exceed the current 20k target-vertex evaluator cap:

- elephant_to_giraffe_full, target has 130100 vertices
- giraffe_to_elephant_full, target has 190950 vertices

## Short Presentation Wording

Use this if you need one slide:

> I evaluated MeshUp on DenseCorr3D-style semantic geodesic AUC/Err. In the fair PartField-guided setting, MeshUp reaches AUC 0.955 and Err 1.17, beating the DenseMatcher reported DenseCorr3D overall result of AUC 0.845 and Err 1.74 by mean score. In a supervised DenseCorr-guided upper-bound setting over all 90 directed animal pairs, the completed chain reaches AUC 0.9915 and Err 0.216, with all 90/90 pairs beating the reference.

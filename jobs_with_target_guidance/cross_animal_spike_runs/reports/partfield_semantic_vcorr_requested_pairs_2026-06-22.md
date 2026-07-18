# Requested PartField + Semantic VCorr Pair Summary

Date: 2026-06-22

## Cat and Bulldog PartField Buckets

### cat

Mesh: `experiments/dog_morphs/outputs/hound_to_cat_no_dino_exp/mesh_final/mesh.obj`
Features: `jobs_with_target_guidance/cross_animal_spike_runs/partfield/features/no_dino_animals/part_feat_hound_to_cat_no_dino_0_batch.npy`
Labels: `jobs_with_target_guidance/cross_animal_spike_runs/partfield/segments/no_dino_animals_12/labels/hound_to_cat_no_dino_partfield_labels.npz`
Vertices/faces: 4712 / 9420

| Bucket | Vertex count | Face count | Centroid xyz | Extent xyz |
|---:|---:|---:|---|---|
| 0 | 41 | 82 | -0.3987, 0.1863, -0.1406 | 0.2656, 0.4303, 0.1410 |
| 1 | 306 | 613 | 0.7544, 0.1705, 0.2937 | 0.4927, 0.6967, 0.3448 |
| 2 | 455 | 910 | -0.9061, 0.4670, -0.3643 | 1.1202, 0.2500, 0.5459 |
| 3 | 310 | 622 | 0.7055, -0.4679, -0.2462 | 0.3816, 1.0111, 0.4360 |
| 4 | 455 | 904 | 0.2445, 0.1386, 0.1590 | 0.7050, 0.6790, 0.3562 |
| 5 | 456 | 915 | 0.2368, 0.2270, -0.1196 | 0.8482, 0.6773, 0.3648 |
| 6 | 292 | 583 | 0.8600, -0.4366, 0.4347 | 0.5909, 0.8487, 0.4066 |
| 8 | 732 | 1463 | 1.1536, 0.5651, 0.1839 | 0.7125, 0.6755, 0.4632 |
| 9 | 575 | 1139 | -0.3160, -0.1120, 0.0439 | 0.5202, 1.3078, 0.2722 |
| 10 | 440 | 888 | 0.8200, 0.1290, 0.0088 | 0.6041, 0.6893, 0.4359 |
| 11 | 650 | 1301 | -0.2297, -0.0518, -0.2791 | 0.4927, 1.2980, 0.3069 |

### bulldog

Mesh: `experiments/dog_morphs/outputs/hound_to_bulldog_no_dino_exp/mesh_final/mesh.obj`
Features: `jobs_with_target_guidance/cross_animal_spike_runs/partfield/features/no_dino_animals/part_feat_hound_to_bulldog_no_dino_0_batch.npy`
Labels: `jobs_with_target_guidance/cross_animal_spike_runs/partfield/segments/no_dino_animals_12/labels/hound_to_bulldog_no_dino_partfield_labels.npz`
Vertices/faces: 4712 / 9420

| Bucket | Vertex count | Face count | Centroid xyz | Extent xyz |
|---:|---:|---:|---|---|
| 0 | 1114 | 2230 | -0.6606, -0.2286, -0.0845 | 0.5575, 1.2807, 0.4355 |
| 1 | 228 | 453 | 0.2930, 0.1192, 0.2761 | 0.3668, 0.9015, 0.3176 |
| 2 | 3 | 8 | -0.6481, 0.1776, 0.0710 | 0.0222, 0.0096, 0.0211 |
| 3 | 313 | 628 | 0.3025, -0.5722, -0.3734 | 0.4175, 1.0775, 0.3239 |
| 4 | 492 | 986 | -0.1121, -0.0277, 0.2024 | 0.6776, 0.9073, 0.3789 |
| 5 | 465 | 925 | -0.0948, 0.0902, -0.1479 | 0.5930, 0.9445, 0.4300 |
| 6 | 286 | 571 | 0.2939, -0.6527, 0.3139 | 0.3976, 0.9490, 0.2598 |
| 8 | 699 | 1399 | 0.5026, 0.6287, -0.0192 | 0.7902, 0.8190, 0.6476 |
| 9 | 743 | 1472 | -0.5793, -0.2679, 0.3243 | 0.4739, 1.3354, 0.5169 |
| 10 | 355 | 718 | 0.3630, 0.0306, -0.1754 | 0.4148, 0.8322, 0.6120 |
| 11 | 14 | 30 | -0.3540, 0.1335, -0.1051 | 0.0228, 0.3795, 0.1280 |

## Semantic Vertex Correspondence Maps

All maps use PartField vertex descriptors from face-feature averaging, soft aligned-label penalties, position/normal tie-breakers, confidence weighting, and topology prior `0.20`.

| Pair | Cache | Identity fraction | Label agreement | Unique target fraction | Mean similarity | Mean weight | Note |
|---|---|---:|---:|---:|---:|---:|---|
| cat_to_bulldog | `jobs_with_target_guidance/cross_animal_spike_runs/correspondences/cat_to_bulldog_partfield_semantic_vcorr.npz` | 0.479 | 0.879 | 0.541 | 0.647 | 0.343 | source vertices map to semantic target vertices; target vertex count can differ |
| bulldog_to_cat | `jobs_with_target_guidance/cross_animal_spike_runs/correspondences/bulldog_to_cat_partfield_semantic_vcorr.npz` | 0.504 | 0.891 | 0.559 | 0.639 | 0.335 | source vertices map to semantic target vertices; target vertex count can differ |
| dachshund_to_bulldog | `jobs_with_target_guidance/cross_animal_spike_runs/correspondences/dachshund_to_bulldog_partfield_semantic_vcorr.npz` | 0.744 | 0.987 | 0.765 | 0.763 | 0.315 | source vertices map to semantic target vertices; target vertex count can differ |
| bulldog_to_dachshund | `jobs_with_target_guidance/cross_animal_spike_runs/correspondences/bulldog_to_dachshund_partfield_semantic_vcorr.npz` | 0.733 | 0.992 | 0.756 | 0.776 | 0.310 | source vertices map to semantic target vertices; target vertex count can differ |

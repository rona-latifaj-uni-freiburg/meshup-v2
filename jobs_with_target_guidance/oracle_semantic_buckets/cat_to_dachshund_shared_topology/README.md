# Oracle Semantic Bucket Trial

This is a local oracle-proxy test for the cat-to-dachshund run.
The source and target OBJ files have identical face topology, so the same vertex id denotes the same latent surface point across both meshes.

## Artifacts

- Label pack: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/labels/cat_to_dachshund_oracle_semantic_labels.npz`
- Source labels: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/labels/source_cat_oracle_semantic_labels.npz`
- Target labels: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/labels/target_dachshund_oracle_semantic_labels.npz`
- Source colored mesh: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/colored/source_cat_oracle_semantic.ply`
- Target colored mesh: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/colored/target_dachshund_oracle_semantic.ply`
- SVG visual summary: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/visuals/oracle_semantic_summary.svg`
- PNG visual summary: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/visuals/oracle_semantic_summary.png`
- Machine-readable summary: `jobs_with_target_guidance/oracle_semantic_buckets/cat_to_dachshund_shared_topology/summary.json`

## Bucket Counts

| id | bucket | source vertices | target vertices |
| --- | --- | ---: | ---: |
| 0 | head_ears | 1106 | 1106 |
| 1 | neck_chest | 561 | 561 |
| 2 | torso_front | 637 | 637 |
| 3 | torso_mid | 783 | 783 |
| 4 | rump_rear | 118 | 118 |
| 5 | tail | 290 | 290 |
| 6 | front_left_leg | 394 | 394 |
| 7 | front_right_leg | 293 | 293 |
| 8 | hind_left_leg | 187 | 187 |
| 9 | hind_right_leg | 343 | 343 |

## PartField Comparison

Compared against `jobs_with_target_guidance/cross_animal_spike_runs/outputs/best_asym035_jump500/cat_to_dachshund_catdach_only_pf08_4000_hard_partfield_chamfer_only_catdach_only_08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5168193/partfield_labels/bucket_labels.npz`.

Because the source and target share topology, corresponding vertices should keep the same semantic bucket.
The oracle labels match at `4712 / 4712` vertices; PartField matches at `4152 / 4712` vertices.

Rows are oracle buckets and columns are PartField buckets.

| oracle bucket | pf00 | pf01 | pf02 | pf03 | pf04 | pf05 | pf06 | pf07 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00_head_ears | 0 | 114 | 0 | 242 | 750 | 0 | 0 | 0 |
| 01_neck_chest | 112 | 131 | 0 | 167 | 17 | 0 | 134 | 0 |
| 02_torso_front | 196 | 0 | 58 | 0 | 0 | 2 | 381 | 0 |
| 03_torso_mid | 128 | 0 | 326 | 0 | 0 | 260 | 20 | 49 |
| 04_rump_rear | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 118 |
| 05_tail | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 290 |
| 06_front_left_leg | 0 | 292 | 0 | 49 | 0 | 0 | 53 | 0 |
| 07_front_right_leg | 0 | 0 | 0 | 278 | 0 | 0 | 15 | 0 |
| 08_hind_left_leg | 0 | 0 | 0 | 0 | 0 | 167 | 20 | 0 |
| 09_hind_right_leg | 0 | 0 | 241 | 0 | 0 | 91 | 11 | 0 |

# DenseMatcher Comparison Credibility Audit

Date: 2026-08-03

## Bottom Line

The safest credible statement is:

> On selected DenseCorr3D animal pairs, using PartField buckets for deformation and holding DenseCorr3D groups out for evaluation, the pipeline reaches mean AUC `0.9550` and mean Err `1.1721` under a DenseCorr/DenseMatcher-style normalized target-geodesic semantic error metric. This is better than DenseMatcher's overall DenseCorr3D benchmark numbers (`AUC 0.845`, `Err 1.74`) and lower than its animal-category appendix Err (`3.21`), but it is **not yet a full replication of the DenseMatcher animal benchmark**, because our current fair result uses selected animal directions, not all possible pairs in the paper split.

The sentence I would **not** use:

> We beat DenseMatcher on the full animal benchmark.

That would be too strong with the current numbers.

The sentence I would use:

> Under a DenseCorr/DenseMatcher-style normalized geodesic correspondence metric, our selected PartField-guided DenseCorr animal-pair runs achieve stronger Err than DenseMatcher's reported animal-category Err and stronger AUC than DenseMatcher's overall DenseCorr3D AUC. The current result is a selected-pair comparison, not a full all-pairs benchmark replication.

## What DenseMatcher Reports

From the DenseMatcher paper:

- DenseCorr3D contains textured assets with semantic-group correspondence annotations.
- The animal category has `5` train objects and `5` test objects.
- Their benchmark uses normalized geodesic error, called `Err`, and AUC of the threshold-accuracy curve.
- The paper says evaluation is done across all possible pairs within each category.
- The main table reports DenseMatcher on the whole DenseCorr3D benchmark: `AUC 0.845`, `Err 1.74`.
- The appendix table for complex topologies reports animal-category `Err 3.21`, but I did **not** find an animal-category AUC value in the paper text.

Important source details:

- Paper evaluation definition: normalized geodesic error and AUC, with distance to the nearest point in the correct semantic group.
- Paper split: animals have `5` train, `0` validation, `5` test.
- Paper appendix animal row: DenseMatcher `Err 3.21`; URSSM `Err 6.75`.
- Official GitHub dataset format: `color_mesh.obj`, `simple_mesh.obj`, `groups.txt`, `groups_visualization.obj`, with `simple_mesh.obj` around 2000 vertices.

Sources:

- DenseMatcher paper: https://arxiv.org/html/2412.05268
- DenseMatcher repository: https://github.com/TEA-Lab/DenseMatcher

## DenseMatcher Numbers To Compare Against

| DenseMatcher result | AUC | Err | What it means |
|---|---:|---:|---|
| Full DenseCorr3D test set | 0.845 | 1.74 | Overall benchmark across categories |
| Held-out categories | 0.775 | 2.82 | Harder out-of-distribution category setting |
| Animals appendix row | not found in paper text | 3.21 | Animal-category Err only |

Interpretation:

- If we compare to `AUC 0.845 / Err 1.74`, we are comparing to DenseMatcher's **overall** DenseCorr3D result, not animal-only.
- If we compare to `Err 3.21`, we are comparing to DenseMatcher's **animal-category Err**.
- I should not claim animal-only AUC superiority unless we find an official animal-only DenseMatcher AUC or run DenseMatcher on the same animal subset.

## All Local DenseCorr3D Animal Meshes

Your local DenseCorr3D animal folder contains `10` animal objects, not only the 6 named animals we used in early experiments.

| Object id | Local alias if known | DenseMatcher split | simple verts | simple faces | color verts | color faces | groups | group size range |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `071b8_toy_animals_017` | panther | train | 1656 | 3320 | 145127 | 290250 | 8 | 20-766 |
| `13cf7_toy_animals_055` | unknown alias | test | 1618 | 3235 | 258325 | 516650 | 8 | 78-660 |
| `188f4_toy_animals_022` | unknown alias | test | 1439 | 2873 | 238119 | 476234 | 8 | 34-730 |
| `1d6d1_toy_animals_015` | moose | train | 1721 | 3450 | 179841 | 359678 | 8 | 50-534 |
| `2d6b3_toy_animals_009` | elephant | test | 1644 | 3297 | 190950 | 381900 | 8 | 101-593 |
| `34fb4_toy_animals_019` | giraffe | train | 1896 | 3802 | 130100 | 260208 | 8 | 23-724 |
| `4e94a_toy_animals_020` | unknown alias | train | 2037 | 4772 | 265728 | 532220 | 8 | 30-537 |
| `96615_toy_animals_018` | bear | train | 1564 | 3130 | 253787 | 507570 | 8 | 33-701 |
| `bdfd0_toy_animals_016` | cheetah | test | 1801 | 3620 | 217836 | 435680 | 8 | 25-821 |
| `d1889_dinosaur_002` | dinosaur | test | 1993 | 4085 | 248427 | 498462 | 8 | 57-761 |

The official object IDs are more reliable than the animal nicknames. Some aliases are from your local/project naming, not from a metadata file I found.

## Which Meshes Our Fair Numbers Used

The current fair comparison rows used only these animals:

- `071b8_toy_animals_017`, panther, train split
- `96615_toy_animals_018`, bear, train split
- `bdfd0_toy_animals_016`, cheetah, test split

They did **not** use every local animal mesh.

They did **not** use all 5 DenseMatcher animal test objects.

They did **not** use all possible animal pairs.

The fair runs were:

| Direction | Source object | Target object | Source verts | Target verts | PF settings |
|---|---|---|---:|---:|---|
| bear -> cheetah | `96615_toy_animals_018` | `bdfd0_toy_animals_016` | 1564 | 1801 | PF6, PF7, PF8 |
| cheetah -> bear | `bdfd0_toy_animals_016` | `96615_toy_animals_018` | 1801 | 1564 | PF6, PF7, PF8 |
| cheetah -> panther | `bdfd0_toy_animals_016` | `071b8_toy_animals_017` | 1801 | 1656 | PF6, PF7, PF8 |
| panther -> cheetah | `071b8_toy_animals_017` | `bdfd0_toy_animals_016` | 1656 | 1801 | PF6, PF7, PF8 |

So the current fair summary is:

```text
4 directed animal-pair tasks x 3 PartField bucket settings = 12 evaluated runs
```

## Our Fair Rows

These are the exact PartField-guided, DenseCorr-evaluated rows.

| Pair | PF buckets | AUC | Err | p95 Err | same-bucket rate | target verts | evaluated source verts |
|---|---:|---:|---:|---:|---:|---:|---:|
| bear -> cheetah | 6 | 0.9440 | 1.4014 | 9.7255 | 0.7903 | 1801 | 1564 |
| bear -> cheetah | 7 | 0.9504 | 1.2395 | 9.4074 | 0.8082 | 1801 | 1564 |
| bear -> cheetah | 8 | 0.9519 | 1.2030 | 8.7745 | 0.8127 | 1801 | 1564 |
| cheetah -> bear | 6 | 0.9637 | 0.9070 | 6.3145 | 0.8334 | 1564 | 1801 |
| cheetah -> bear | 7 | 0.9673 | 0.8159 | 5.8487 | 0.8595 | 1564 | 1801 |
| cheetah -> bear | 8 | 0.9670 | 0.8246 | 6.0335 | 0.8662 | 1564 | 1801 |
| cheetah -> panther | 6 | 0.8920 | 3.2621 | 30.8064 | 0.7929 | 1656 | 1801 |
| cheetah -> panther | 7 | 0.9563 | 1.1014 | 6.8661 | 0.8595 | 1656 | 1801 |
| cheetah -> panther | 8 | 0.9611 | 0.9818 | 6.0302 | 0.8728 | 1656 | 1801 |
| panther -> cheetah | 6 | 0.9590 | 1.0258 | 6.4912 | 0.7947 | 1801 | 1656 |
| panther -> cheetah | 7 | 0.9689 | 0.7787 | 5.1035 | 0.8261 | 1801 | 1656 |
| panther -> cheetah | 8 | 0.9790 | 0.5237 | 3.9412 | 0.8545 | 1801 | 1656 |

Summary:

| Setting | n | mean AUC | mean Err | mean same-bucket rate |
|---|---:|---:|---:|---:|
| All fair selected rows | 12 | 0.9550 | 1.1721 | 0.8309 |
| PF6 only | 4 | 0.9397 | 1.6491 | 0.8028 |
| PF7 only | 4 | 0.9607 | 0.9839 | 0.8383 |
| PF8 only | 4 | 0.9648 | 0.8833 | 0.8515 |

By pair:

| Pair | n | mean AUC | mean Err | mean same-bucket rate |
|---|---:|---:|---:|---:|
| bear -> cheetah | 3 | 0.9488 | 1.2813 | 0.8037 |
| cheetah -> bear | 3 | 0.9660 | 0.8492 | 0.8530 |
| cheetah -> panther | 3 | 0.9365 | 1.7818 | 0.8418 |
| panther -> cheetah | 3 | 0.9690 | 0.7760 | 0.8251 |

## Exactly How The Comparison Was Computed

Our output is a deformed mesh, not an explicit DenseMatcher functional-map correspondence matrix. Therefore the evaluator induces a correspondence from the deformation.

For each source vertex:

1. The source mesh is deformed by the pipeline, preserving the source topology.
2. The deformed source vertex is normalized into MeshUp unit-box coordinates.
3. The target mesh is normalized the same way.
4. The nearest target vertex to the deformed source vertex is selected.
5. That nearest target vertex is treated as the predicted correspondence.
6. The source vertex's DenseCorr3D semantic group is loaded from raw `groups.txt`.
7. The target's raw DenseCorr3D semantic groups are loaded from target `groups.txt`.
8. On the target mesh, the evaluator computes target-surface geodesic distance from the predicted target vertex to the nearest vertex in the correct target semantic group.
9. The distance is normalized by `sqrt(target surface area)`.
10. AUC and Err are computed from the normalized error vector.

Metric formulas:

```text
Err = mean(normalized geodesic error) * 100

accuracy(t) = fraction of evaluated source vertices with normalized geodesic error <= t

AUC = area under accuracy(t), integrated for t from 0 to 0.25, divided by 0.25
```

Implementation file:

`jobs_with_target_guidance/evaluate_densecorr_paper_style_geodesic.py`

Evaluation output:

`jobs_with_target_guidance/cross_animal_spike_runs/evaluation/bucket_discrepancy/densecorr_partfield_guided_fair_existing_kim_geodesic_auc.md`

## Same Geodesic Distance As DenseMatcher?

Short answer:

**Same normalized-geodesic-error idea, not guaranteed bit-identical geodesic matrices.**

Details:

- DenseMatcher paper says it evaluates normalized geodesic error and AUC, following the Kim/Princeton convention.
- The paper also says each instance contains a geodesic distance matrix for the remeshed geometry.
- The official GitHub README dataset format lists `color_mesh.obj`, `simple_mesh.obj`, `groups.txt`, and `groups_visualization.obj`.
- Your local DenseCorr3D animal folders contain those four files, but I did not find the paper's precomputed geodesic matrices locally.
- Therefore, our evaluator computes geodesic distances with graph shortest paths on target mesh edges, then divides by `sqrt(target surface area)`.

This is a strong metric-compatible proxy because it is still a normalized target-surface geodesic error to the correct semantic group. But it is not a bitwise reproduction of DenseMatcher's exact stored matrices unless we obtain their matrices or exact computation code.

Slide wording:

> We use the same normalized target-surface geodesic-error convention, but because the released/local dataset folder does not include the paper's precomputed geodesic matrices, our implementation computes graph geodesics over the remeshed target mesh.

## Same Vertex Count As DenseMatcher?

Mostly comparable, but not exactly identical in every run.

DenseMatcher says `simple_mesh.obj` is a remeshed version with roughly 2000 vertices. The local animal simple meshes have:

```text
min simple vertices: 1439
max simple vertices: 2037
```

Our fair rows used the same `simple_mesh.obj` resolution scale:

- bear source/target: 1564 vertices
- panther source/target: 1656 vertices
- cheetah source/target: 1801 vertices

So we did **not** evaluate on the huge colored meshes with 130k-265k vertices. We evaluated on the remeshed DenseCorr3D simple meshes, which is consistent with the geometry/correspondence side of DenseMatcher.

## Did We Use DenseCorr Labels For Training?

For the **fair** comparison rows: no.

The fair rows used:

- PartField features and PartField labels to guide deformation.
- DenseCorr3D `groups.txt` only after deformation for evaluation.

Example config facts from `panther_to_cheetah` PF8:

```text
partfield_guidance_mode: hard
partfield_labels_aligned: true
partfield_n_buckets: 8
partfield_source_labels: .../panther_cheetah/buckets_08/labels/panther_partfield_labels.npz
partfield_target_labels: .../panther_cheetah/buckets_08/labels/cheetah_partfield_labels.npz
target_mesh_partfield_chamfer_weight: 8000.0
target_mesh_chamfer_weight: 0.0
```

That is why these rows are more credible than the DenseCorr-guided oracle rows.

The oracle rows use DenseCorr groups to drive deformation and DenseCorr groups to evaluate. Those are useful as an upper bound, but not the fair claim.

## What The PartField Buckets Looked Like

The PartField bucket visualizations are stored as colored PLY files. For the fair selected runs, the relevant bucket visualizations are here:

Panther/Cheetah:

- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_06/colored/panther_partfield_06_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_06/colored/cheetah_partfield_06_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_07/colored/panther_partfield_07_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_07/colored/cheetah_partfield_07_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_08/colored/panther_partfield_08_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/panther_cheetah/buckets_08/colored/cheetah_partfield_08_parts.ply`

Cheetah/Bear:

- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_06/colored/cheetah_partfield_06_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_06/colored/bear_partfield_06_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_07/colored/cheetah_partfield_07_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_07/colored/bear_partfield_07_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_08/colored/cheetah_partfield_08_parts.ply`
- `jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/cheetah_bear/buckets_08/colored/bear_partfield_08_parts.ply`

For the broader six named animals, bucket visualizations are also available under:

`jobs_with_target_guidance/partfield_segments/densecorr3d_animals/densecorr3d_animals_20260706/all_six`

That folder contains colored PLYs for:

- bear
- cheetah
- elephant
- giraffe
- moose
- panther

with bucket counts from 5 through 10.

Important limitation:

PartField features/visualizations currently exist for the named six-animal set, not automatically for all 10 DenseCorr animal objects. To run the full DenseMatcher animal split, we would need to generate PartField features and labels for the remaining animal IDs too.

## What Was Actually Tried

Current fair DenseMatcher-style comparison:

```text
Tried:
  bear -> cheetah
  cheetah -> bear
  cheetah -> panther
  panther -> cheetah

For each:
  PF6
  PF7
  PF8

Total:
  12 runs
```

Not yet tried in the fair comparison:

```text
All 10 local animal objects against all other animal objects:
  10 * 9 = 90 directed pairs

All 5 DenseMatcher animal test objects against all other test objects:
  5 * 4 = 20 directed pairs
  or 10 unordered pairs, depending on the benchmark implementation
```

The paper says all possible pairs within category, so the full benchmark-style comparison should use all appropriate pairs, not just the selected 12 rows.

## What We Need For A Fully Credible Animal Benchmark Claim

To say:

> We beat DenseMatcher on the animal benchmark.

we should do this:

1. Use the exact animal test split from `external/DenseCorr3D/test_files.txt`:
   - `188f4_toy_animals_022`
   - `bdfd0_toy_animals_016`
   - `2d6b3_toy_animals_009`
   - `d1889_dinosaur_002`
   - `13cf7_toy_animals_055`
2. Generate PartField features for all 5 test animals.
3. Generate PartField bucket labels for every evaluated source-target pair.
4. Deform every directed test pair, probably 20 runs if self-pairs are excluded.
5. Evaluate with DenseCorr3D raw `groups.txt`, not PartField labels.
6. Use the same normalized target-geodesic error convention.
7. Report mean Err across all pairs and compare to DenseMatcher animal Err `3.21`.
8. For AUC, either find the official animal-only DenseMatcher AUC or run DenseMatcher's released model/evaluator on the same animal pairs.

Then the claim becomes much stronger:

> On the DenseCorr3D animal test split, evaluated over all directed test pairs with DenseCorr semantic groups held out for scoring, PartMorph3D reaches Err X versus DenseMatcher's reported animal Err 3.21.

Until then, use the selected-pair wording.

## Best Slide Wording

Short version:

> For a fair comparison, I do not use DenseCorr labels to guide the deformation. I deform using PartField buckets, then evaluate afterward using DenseCorr3D groups.txt. On 12 selected animal-pair runs, this gives mean AUC 0.955 and Err 1.17. DenseMatcher reports overall DenseCorr3D AUC 0.845 / Err 1.74, and an animal-category Err of 3.21. This is strong evidence, but I present it as a selected-pair comparison rather than a full all-pairs benchmark replication.

Very cautious version:

> The current comparison is metric-compatible with DenseMatcher's normalized geodesic semantic correspondence evaluation, but not yet split-identical or pair-exhaustive.

Strong but honest version:

> On the tested DenseCorr animal pairs, using held-out DenseCorr labels only for scoring, the pipeline reaches lower normalized geodesic semantic error than DenseMatcher's reported animal-category Err. A full animal-benchmark claim requires running the remaining test animal pairs.


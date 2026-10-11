# Current Pipeline Presentation Outline

Date: 2026-07-22

Working title:

**PartField-Guided Cross-Animal Mesh Deformation**

Main message:

The current pipeline can deform a source animal mesh toward a target animal shape using semantic part buckets, asymmetric Chamfer matching, and local deformation regularization. The best examples are visually convincing after 4000 epochs. The main remaining limitation is not basic convergence, but semantic bucket quality: if a source/target part bucket has very different point support, the optimizer can satisfy nearest-neighbor distances while producing a thick neck, trunk, tail swirl, or local folding.

## Slide 1: Title And Best Result

Show the strongest visual result first.

Recommended image:

![horse2 to giraffe4, final 4000 epochs](../outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/epoch_renders/epoch_04000/grid_all_views.png)

Mesh:

[horse2_to_giraffe4 final colored PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/colored_meshes/mesh_final_position.ply)

Speaker line:

I am deforming one animal mesh into another while preserving a coherent surface. The best current runs give recognizable target anatomy, for example a horse source becoming a giraffe-like mesh with a long neck, separated legs, and no large isolated spikes.

## Slide 2: Problem

Goal:

Given a source mesh and a target mesh, move the source geometry so that it matches the target shape while staying smooth and connected.

Why this is hard:

- A plain global Chamfer loss does not know which part is which.
- Nearest-neighbor matching can collapse many target points onto the same source region.
- Cross-animal pairs have large shape differences, especially necks, trunks, tails, and limbs.
- Deforming a mesh directly can produce spikes, folds, or singular local geometry.

Use this slide to set up the need for semantic buckets and regularization.

## Slide 3: Pipeline Overview

Pipeline:

1. Start with a source mesh and a target mesh.
2. Compute or load semantic part labels on both meshes.
3. Freeze the source vertex bucket identities.
4. Optimize the source mesh for 4000 epochs.
5. Match source and target only within corresponding buckets.
6. Regularize the local deformation so neighboring faces move coherently.
7. Save final mesh, colored mesh, multi-view renders, and correspondence files.

Current strongest recipe:

| component | current setting |
|---|---|
| optimization length | 4000 epochs |
| deformation parameterization | Jacobian |
| guidance mode | hard PartField/DenseCorr3D bucket Chamfer |
| active local semantic loss | `target_mesh_partfield_chamfer_weight = 8000` |
| source-to-target bucket weight | `0.35` |
| target-to-source bucket weight | `1.0` |
| per-bucket samples | `512` |
| source/target labels | aligned |
| Jacobian neighbor smoothing | `1000` |
| edge displacement jump guard | `500`, threshold `1.2` |

Important wording nuance:

The codebase supports both global mesh Chamfer and local/semantic bucket Chamfer. The final runs listed here are saved as `hard_partfield_chamfer_only`; their inspected configs set `target_mesh_chamfer_weight = 0.0` and put the main weight on the PartField/DenseCorr3D bucket Chamfer. So the most accurate phrase is: "the pipeline supports global Chamfer, but these final best runs are driven mainly by local semantic bucket Chamfer plus deformation regularization."

## Slide 4: What The Loss Is Doing

Say it like this:

Each bucket acts like a local correspondence region. For a bucket `k`, source points in bucket `k` are matched to target points in bucket `k`, instead of allowing the whole animal to match globally. The loss is asymmetric: target-to-source has weight `1.0` so the target shape asks to be covered, while source-to-target is reduced to `0.35` so source points are less likely to over-pull and create spikes.

Regularization:

- Jacobian parameterization keeps the deformation local rather than moving unrelated vertices independently.
- Neighbor smoothness encourages neighboring faces to deform similarly.
- The edge displacement jump penalty suppresses sudden local jumps that create long thin spikes.

One-sentence math summary:

`final loss = semantic bucket Chamfer + Jacobian neighbor smoothness + edge displacement jump penalty`

## Slide 5: Data Sources

Separate the qualitative and quantitative stories.

| data source | role in presentation | labels |
|---|---|---|
| SAM3D selected animal meshes | best-looking qualitative examples | PartField features and clustered labels |
| DenseCorr3D animal meshes | most important for numbers and evaluation | supervised aligned semantic groups |

DenseCorr3D is the stronger quantitative dataset because it has supervised annotated part groups. Those labels are aligned within the animal category, so bucket `0` in one animal corresponds to bucket `0` in another animal.

## Slide 6: DenseCorr3D Animal Assets

Use this slide to justify why DenseCorr3D is the numerical/evaluation set.

| animal | object id | color vertices | color faces | simple vertices | simple faces | buckets | simple bucket min/max |
|---|---|---:|---:|---:|---:|---:|---:|
| elephant | `2d6b3_toy_animals_009` | 190950 | 381900 | 1644 | 3297 | 8 | 101/593 |
| moose | `1d6d1_toy_animals_015` | 179841 | 359678 | 1721 | 3450 | 8 | 50/534 |
| giraffe | `34fb4_toy_animals_019` | 130100 | 260208 | 1896 | 3802 | 8 | 23/724 |
| panther | `071b8_toy_animals_017` | 145127 | 290250 | 1656 | 3320 | 8 | 20/766 |
| bear | `96615_toy_animals_018` | 253787 | 507570 | 1564 | 3130 | 8 | 33/701 |
| cheetah | `bdfd0_toy_animals_016` | 217836 | 435680 | 1801 | 3620 | 8 | 25/821 |

Source mesh directories:

- [elephant](../../../external/DenseCorr3D/animals/2d6b3_toy_animals_009)
- [moose](../../../external/DenseCorr3D/animals/1d6d1_toy_animals_015)
- [giraffe](../../../external/DenseCorr3D/animals/34fb4_toy_animals_019)
- [panther](../../../external/DenseCorr3D/animals/071b8_toy_animals_017)
- [bear](../../../external/DenseCorr3D/animals/96615_toy_animals_018)
- [cheetah](../../../external/DenseCorr3D/animals/bdfd0_toy_animals_016)

## Slide 7: Why Bucket Counts Matter

Use elephant/giraffe as the main failure analysis example.

Original independent 5k bucket counts:

| bucket | elephant 5k | giraffe 5k | ratio, larger/smaller |
|---:|---:|---:|---:|
| 0 | 391 | 497 | 1.3x |
| 1 | 1739 | 1755 | 1.0x |
| 2 | 404 | 515 | 1.3x |
| 3 | 318 | 431 | 1.4x |
| 4 | 672 | 1078 | 1.6x |
| 5 | 609 | 87 | 7.0x |
| 6 | 314 | 527 | 1.7x |
| 7 | 553 | 106 | 5.2x |

Balanced elephant/giraffe variant:

| variant | shared counts |
|---|---|
| `elephant_giraffe_bucketavg4998` | `[444, 1747, 460, 375, 875, 348, 420, 329]` |

Source-stable direction-specific variants:

| direction | matched counts |
|---|---|
| elephant -> giraffe | `[391, 1739, 404, 318, 672, 609, 314, 553]` |
| giraffe -> elephant | `[497, 1755, 515, 431, 1078, 87, 527, 106]` |

Interpretation:

When bucket sizes are very different, the Chamfer loss can be locally satisfied but geometrically wrong. If a small giraffe neck/head bucket has only 87 or 106 vertices, it does not constrain the shape as strongly as a larger target region. In the opposite direction, a small source patch can be forced to cover a much larger target bucket, which encourages thickening, stretching, or folding. That is why the elephant/giraffe neck/trunk case is the best failure example.

## Slide 8: SAM3D Qualitative Results

Use these as the best visual demonstration of the current pipeline.

| run | buckets | final mesh | final image | readout |
|---|---:|---|---|---|
| horse2 -> giraffe4 | 8 | [PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/epoch_renders/epoch_04000/grid_all_views.png) | Strong success: target silhouette and long neck are clear. |
| panda3 -> bear | 7 | [PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/sam3d_selected_best_asym035_jump500_4000/panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242/epoch_renders/epoch_04000/grid_all_views.png) | Good body and head mass; rear/leg region has minor artifacts. |
| fox -> cat2 | 7 | [PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/sam3d_selected_best_asym035_jump500_4000/fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/epoch_renders/epoch_04000/grid_all_views.png) | Works well for similar quadruped structure; small limb/ear imperfections. |
| panda2 -> panda3 | 9 | [PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/panda2_to_panda3_sam3d_selected_pf09_4000_hard_partfield_chamfer_only_panda2_to_panda3_pf09_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/sam3d_selected_best_asym035_jump500_4000/panda2_to_panda3_sam3d_selected_pf09_4000_hard_partfield_chamfer_only_panda2_to_panda3_pf09_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/epoch_renders/epoch_04000/grid_all_views.png) | Cleaner control-style example because source and target are close. |

Final colored mesh sizes:

| run | vertices | faces |
|---|---:|---:|
| horse2 -> giraffe4 | 5502 | 11000 |
| panda3 -> bear | 5502 | 11000 |
| fox -> cat2 | 5502 | 11000 |
| panda2 -> panda3 | 5488 | 11000 |

## Slide 9: DenseCorr3D Results

Use this slide for the most defensible experiment set.

DenseCorr3D source-stable 5k runs:

| run | final mesh | final image | readout |
|---|---|---|---|
| panther -> cheetah | [PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/panther_to_cheetah_panther_cheetah_srcstable5002_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708185/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/panther_to_cheetah_panther_cheetah_srcstable5002_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708185/epoch_renders/epoch_04000/grid_all_views.png) | Completed; reasonable quadruped silhouette, with visible head/leg artifacts. |
| cheetah -> panther | [PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/cheetah_to_panther_cheetah_panther_srcstable4992_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708316/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/cheetah_to_panther_cheetah_panther_srcstable4992_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708316/epoch_renders/epoch_04000/grid_all_views.png) | Completed; direction works, but artifacts are stronger than SAM3D showcase. |
| elephant -> giraffe | [PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/elephant_to_giraffe_elephant_giraffe_srcstable5000_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708523/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/elephant_to_giraffe_elephant_giraffe_srcstable5000_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708523/epoch_renders/epoch_04000/grid_all_views.png) | Hard success case: target giraffe silhouette appears, but the neck/head region is thick/folded. |
| giraffe -> elephant | [PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/giraffe_to_elephant_giraffe_elephant_srcstable4996_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708991/colored_meshes/mesh_final_position.ply) | [PNG](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/giraffe_to_elephant_giraffe_elephant_srcstable4996_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708991/epoch_renders/epoch_04000/grid_all_views.png) | Hard success case in reverse: elephant-like body/trunk, but spike/fold artifacts remain. |

Other completed source-stable DenseCorr3D outputs:

- [bear -> panther mesh](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/bear_to_panther_bear_panther_srcstable5002_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708266/colored_meshes/mesh_final_position.ply)
- [bear -> elephant mesh](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/bear_to_elephant_bear_elephant_srcstable5002_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708335/colored_meshes/mesh_final_position.ply)
- [elephant -> moose mesh](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/elephant_to_moose_elephant_moose_srcstable5000_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708351/colored_meshes/mesh_final_position.ply)

## Slide 10: Chain Experiment

This is an ambitious example, not the cleanest headline result.

Step 1:

![panther to horse2, final 4000 epochs](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/panther_to_horse2_chain_face_x_pf06_4000_hard_partfield_chamfer_only_panther_to_horse2_pf06_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829892/epoch_renders/epoch_04000/grid_all_views.png)

Mesh:

[panther_to_horse2 final OBJ](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/panther_to_horse2_chain_face_x_pf06_4000_hard_partfield_chamfer_only_panther_to_horse2_pf06_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829892/mesh_final/mesh.obj)

Step 2:

![horse2 to dense giraffe, final 4000 epochs](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/horse2_to_dense_giraffe_chain_face_x_pf08_4000_hard_partfield_chamfer_only_horse2_to_dense_giraffe_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829931/epoch_renders/epoch_04000/grid_all_views.png)

Mesh:

[horse2_to_dense_giraffe final OBJ](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/horse2_to_dense_giraffe_chain_face_x_pf08_4000_hard_partfield_chamfer_only_horse2_to_dense_giraffe_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829931/mesh_final/mesh.obj)

Readout:

The chain demonstrates that the same deformation recipe can be composed across very different shapes. The first step moves the panther toward a horse-like body, and the second step gets a giraffe-like neck. The limitation is error accumulation: tail/rear artifacts from the intermediate stage can remain and affect the next stage.

## Slide 11: Failure Modes And Lessons

Failure 1: bucket-size mismatch.

- Elephant/giraffe has buckets with 5x to 7x point-count imbalance in the original independent 5k variants.
- Small buckets can under-constrain a neck/trunk/head region.
- Large target regions can pull too few source vertices outward, producing thickening or folds.
- Balanced/source-stable variants reduce this problem but do not remove all semantic shape mismatch.

Failure 2: degenerate remeshing in early bucket-balanced runs.

- The first bucket-balancing attempt produced near-zero-area faces or coincident vertices in some remeshed sources.
- The Jacobian-to-vertex Poisson solve became singular and failed at epoch 1 with NaNs.
- The fix added degeneracy guards during edge split, edge collapse, smoothing, and final validation.
- After the fix, the bucket-balanced chain completed all seven tasks.

Failure 3: global shrinkage from other ablations.

- Robust target-to-source caps and unbalanced OT variants reduced some pulls but tended to shrink shapes.
- Reverse-unmatched weighting was stable but not better than the simple asymmetric recipe.
- The practical current best remains `asym035 + neighbor smooth 1000 + jump 500`.

Failure 4: full-resolution practicality.

- DenseCorr3D full meshes are 130k to 254k vertices.
- 5k/topology-preserving variants are much easier to run and inspect.
- For a presentation, use full-resolution DenseCorr3D as the annotated data source and 5k variants as the current deformation/evaluation regime.

## Slide 12: What Currently Works

Use this as the positive results slide.

- The deformation pipeline is end-to-end functional.
- Final meshes, colored meshes, render grids, videos, and correspondence files are produced automatically.
- SAM3D qualitative examples are strong, especially horse -> giraffe and fox -> cat.
- DenseCorr3D label-only runs work with supervised semantic buckets.
- The 4000-epoch recipe is stable across multiple animal pairs when the source mesh is clean.
- The most robust current setting is asymmetric hard bucket Chamfer with strong local deformation regularization.

## Slide 13: What Still Needs Work

Use this as the honest limitations slide.

- Part buckets need to be geometrically compatible, not just semantically aligned.
- Neck/trunk/tail regions need stronger shape priors than nearest-neighbor Chamfer alone.
- The current final outputs are mostly qualitative; the next step is a consistent DenseCorr3D metric table.
- Multi-step chains are promising but accumulate artifacts.
- Source-stable variants avoid Poisson failures, but they can still inherit insufficient source support for small parts.

## Slide 14: Next Steps

Recommended next experiments:

1. Add a formal DenseCorr3D evaluation table: global Chamfer, partwise Chamfer, bucket coverage, and boundary/fold checks.
2. Use bucket mass diagnostics before training and flag pairs with extreme ratios.
3. Test the existing repair presets for elephant/giraffe: moment/profile/anchor/containment losses.
4. Use source-stable remeshing by default, with degeneracy validation before submitting long jobs.
5. For chains, add an intermediate cleanup/remesh step before passing the result into the next deformation.

Suggested closing sentence:

The pipeline has moved from "can this deform at all?" to "which semantic buckets make a deformation trustworthy?" The strongest results are already presentable, and DenseCorr3D gives the right supervised testbed for turning the current qualitative pipeline into a measured method.

## Slide Timing

For a 12 minute presentation:

| minutes | slides |
|---:|---|
| 0-1 | title and best result |
| 1-3 | problem and pipeline |
| 3-5 | loss/regularization explanation |
| 5-7 | data sources and DenseCorr3D labels |
| 7-9 | SAM3D and DenseCorr3D results |
| 9-11 | failures and bucket-count diagnosis |
| 11-12 | next steps and conclusion |

## Appendix: Mesh And Image Manifest

Primary final meshes from the current SAM3D-selected runs:

- [horse2 -> giraffe4 PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/horse2_to_giraffe4_sam3d_selected_pf08_4000_hard_partfield_chamfer_only_horse2_to_giraffe4_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5816558/colored_meshes/mesh_final_position.ply)
- [panda3 -> bear PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/panda3_to_bear_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_panda3_to_bear_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5817242/colored_meshes/mesh_final_position.ply)
- [fox -> cat2 PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/fox_to_cat2_sam3d_selected_pf07_4000_hard_partfield_chamfer_only_fox_to_cat2_pf07_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply)
- [panda2 -> panda3 PLY](../outputs/sam3d_selected_best_asym035_jump500_4000/panda2_to_panda3_sam3d_selected_pf09_4000_hard_partfield_chamfer_only_panda2_to_panda3_pf09_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5818235/colored_meshes/mesh_final_position.ply)

Primary chain meshes:

- [panther -> horse2 OBJ](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/panther_to_horse2_chain_face_x_pf06_4000_hard_partfield_chamfer_only_panther_to_horse2_pf06_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829892/mesh_final/mesh.obj)
- [horse2 -> dense giraffe OBJ](../outputs/chain_panther_horse_giraffe_face_x_best_asym035_jump500_4000/horse2_to_dense_giraffe_chain_face_x_pf08_4000_hard_partfield_chamfer_only_horse2_to_dense_giraffe_pf08_pfonly_best_asym035_jneighbor1000_jump500_4000ep_4000ep_5829931/mesh_final/mesh.obj)

Primary DenseCorr3D source-stable meshes:

- [panther -> cheetah PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/panther_to_cheetah_panther_cheetah_srcstable5002_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708185/colored_meshes/mesh_final_position.ply)
- [cheetah -> panther PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/cheetah_to_panther_cheetah_panther_srcstable4992_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708316/colored_meshes/mesh_final_position.ply)
- [elephant -> giraffe PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/elephant_to_giraffe_elephant_giraffe_srcstable5000_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708523/colored_meshes/mesh_final_position.ply)
- [giraffe -> elephant PLY](../outputs/densecorr3d_srcstable_requested_5k_best_asym035/giraffe_to_elephant_giraffe_elephant_srcstable4996_densecorr3d_srcstable_requested_5k_hard_partfield_chamfer_only_densecorr3d_groups_pfonly_best_asym035_jneighbor1000_jump500_4000ep_5708991/colored_meshes/mesh_final_position.ply)

Useful existing reports:

- [DenseCorr3D requested animals report](densecorr3d_requested_animals_2026-06-30.md)
- [DenseCorr3D bucket-balanced report](densecorr3d_bucketbalanced_requested_2026-07-02.md)
- [DenseCorr3D bucket-balanced fix report](densecorr3d_bucketbalanced_requested_fix_and_rerun_2026-07-02.md)
- [DenseCorr3D source-stable rerun report](densecorr3d_srcstable_rerun_2026-07-02.md)
- [Giraffe to elephant repair notes](giraffe_to_elephant_5k_repair_2026-06-30.md)

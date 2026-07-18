# Oracle Neckfix: Elephant <-> Giraffe Diagnosis and New Pipeline

Date: 2026-07-02

## Request

Diagnose why elephant<->giraffe deformation (neck, trunk, face) is worse than
same-topology pairs like bear/panther/cheetah, determine whether it's a
bucket-vertex-count problem or a segmentation-quality problem, and build a
**separate** pipeline (existing best_asym035_jump500 recipe untouched) to test
fixes on the six requested DenseCorr3D animals on `dev_gpu_h100`, <=30 min per
run.

## Prior state (read before building anything)

This project already had ~3 weeks of iteration on exactly this pair (13+
dated reports under `cross_animal_spike_runs/reports/`, a `densematcher_runs`
prep thread, and a `densecorr3d_giraffe_elephant_5k_repairs` sweep of 9 named
"repair presets"). Relevant prior findings, reused rather than re-derived:

- DenseCorr3D's `groups.txt` (8 supervised, cross-species-aligned semantic
  groups per `animals`-category object) was already being used as an
  alternative to PartField's unsupervised clustering for this pair
  (`densecorr3d_groups` labels), via `densecorr3d_segment.py` +
  `densecorr3d_prepare_mesh_variants.py`.
- The un-balanced hard-bucket run on this labeling
  (`densecorr3d_requested_animals_best_asym035/giraffe_to_elephant_..._5546334`)
  produced a visibly bad result: "elephant trunk too wide, tail swirl,
  apparent open space in the head" (`giraffe_to_elephant_5k_repair_2026-06-30.md`).
  That report traced it to bucket 5 (snout/trunk): 87 source vertices having
  to cover 609 target vertices, with the deformed extent (0.127) ending up
  **larger** than the target's own extent (0.076).
- Two categories of fix were tried afterward and made results worse, twice
  each, in two independent threads: adding moment/profile/anchor/containment
  weight or robust/unbalanced-OT Chamfer (bulldog spike sweep in June, and
  the giraffe/elephant repair sweep) consistently produced collapsed/blobby/
  shrunk shapes; direct semantic-vertex-correspondence or region-moment
  losses used as a deformation force caused spikes/waviness and were
  explicitly abandoned in-repo.
- As of this morning, a new attempt was in flight: physically
  edge-collapsing/splitting each mesh to force exact per-bucket vertex-count
  parity between a specific source/target pair
  (`densecorr3d_balance_bucket_counts.py`, jobs `5698613` -> `5700256` ->
  `5700506`).

## What I found before building anything new

1. **Today's bucket-balancing chain is dead, and I found why.** I loaded
   `.../requested_animals_20260630/meshes/96615_..._bear_panther_bucketavg5002.obj`
   (the bear source mesh used by the pair that crashed) directly with
   trimesh: **73 duplicate-coordinate vertices and 148 zero/near-zero-area
   faces**, despite passing `is_watertight` and edge-manifold checks. That
   singularizes the cotangent-Laplacian used by the jacobian/Poisson
   deformation representation, which matches the actual crash exactly:
   `AssertionError: Nan in the forward pass of the POISSON SOLVE` at
   `NeuralJacobianFields/PoissonSystem.py:492`, on epoch 1. This happened for
   **bear->panther**, a pair that already works well with the existing
   pipeline -- so it's a bug in that script's label-boundary-adjacent
   smoothing step, not something elephant/giraffe-specific. `set -euo
   pipefail` killed the script before it reached its own chaining logic, so
   nothing was actually left queued.
2. **Per-bucket Chamfer already weights every active bucket equally**
   (`torch.stack(bucket_losses).mean()` in `partfield_chamfer.py`), so raw
   bucket-count imbalance across the whole mesh isn't the mechanism. The real
   failure is local: a badly undersized bucket on one side has to cover a
   much larger target region under the dominant target->source term, so its
   few points balloon outward past the target's own extent to reach full
   coverage. That's a property of specific undersized buckets, not overall
   imbalance.
3. **DenseCorr3D's 8 groups are genuinely cross-species-consistent** -- I
   independently verified this (not just took it on faith) by computing
   per-group centroid/extent from raw `groups.txt` for all 6 animals: group 1
   is always torso (dominant), groups {0,2} and {3,6} are left/right leg
   pairs, group 4 is head, **group 5 is snout/trunk** (elephant 208 raw
   vertices vs ~20-50 for other animals -- it correctly isolates the trunk),
   group 7 is ears/antlers (moose 481, elephant 167, others ~20-44).
4. **There is no dedicated neck group.** The neck is absorbed into torso
   (group 1) for every animal; giraffe's group-1 Y-extent is unusually tall
   for exactly that reason. Per-bucket Chamfer therefore has no target that
   isolates "this region should be long and thin."
5. Elephant->giraffe with the *unbalanced* natural-count DenseCorr3D labels
   already looked good on inspection (clean neck, correct legs, small
   ossicones) in an existing run
   (`densecorr3d_bucketavg4998_elephant_giraffe/elephant_to_giraffe_..._5667541`).
   Giraffe->elephant (growing a trunk from nothing) is the genuinely hard
   direction, and that's where the bucket-5-style over-expansion concentrates.

## What was built (new files only; nothing existing was modified)

1. **`jobs_with_target_guidance/densecorr3d_protect_small_buckets.py`** --
   per-animal (not per-pair) split-only densification: raises any bucket
   below a vertex floor (300) by repeatedly splitting the longest edge
   strictly interior to that bucket, in fixed-snapshot passes. Never an edge
   collapse, never a cross-label smoothing pass, so it cannot introduce the
   duplicate-vertex/zero-area-face failure found in (1) above. Hard
   self-check before writing output (duplicate vertices, zero-area faces,
   non-manifold edges, vertex cap) that aborts loudly instead of writing bad
   geometry.

   Building and testing this surfaced a real bug worth recording: an
   earlier version used a single global longest-edge-first heap without
   snapshotting. On the giraffe mesh, a high-valence "hub" vertex (a
   decimation artifact where many triangles fan around one point) had its
   shrinking spoke edges repeatedly re-selected as "currently longest,"
   geometrically halving toward the hub each time until ~48 independent
   spokes numerically converged on the same point -- reproducing the exact
   duplicate-vertex/zero-area-face failure mode from (1), just from a
   different cause. Fixed by taking a fixed snapshot of eligible edges once
   per pass and never re-queuing an edge that touches a vertex created
   earlier in the same pass; verified clean afterward (see Results).

2. **`jobs_with_target_guidance/densecorr3d_split_neck_bucket.py`** -- splits
   only the torso group (id 1) by a vertical-axis median cut on unit-box
   coordinates into torso + a new "neck" bucket (id 8). All other groups
   pass through unchanged. Applied independently per animal with a fixed
   rule, so the new neck bucket stays semantically aligned across animals
   for `partfield_labels_aligned=1` matching.

3. **`jobs_with_target_guidance/cross_animal_spike_runs/jobs/prepare_oracle_neckfix_animals.sh`**
   -- one-time CPU prep chaining the existing, unmodified
   `densecorr3d_prepare_mesh_variants.py` (groups.txt -> 5k mesh) with the
   two new scripts (protect -> split neck -> protect again to top up the
   fresh torso/neck halves). Output under
   `jobs_with_target_guidance/densematcher_runs/prepared/oracle_neckfix_20260702/`.

4. **`jobs_with_target_guidance/cross_animal_spike_runs/jobs/run_oracle_neckfix_pair.sh`**
   -- pair runner exporting only the proven best_asym035 settings
   (`PARTFIELD_GUIDANCE_MODE=hard`, chamfer weight 8000, source->target 0.35,
   target->source 1.0, jacobian-neighbor-smooth 1000, edge-displacement-jump
   500) plus the new label paths, then calling the existing, unmodified
   shared runner `artur_soft_runs/jobs/run_artur_chamfer_ablation.sh` --
   exactly the pattern every other recipe in this repo uses. No moment /
   profile / anchor / containment / robust / unbalanced-OT weight (see prior
   findings above).

5. **`jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_dev_h100_oracle_neckfix_single.sh`**
   -- `dev_gpu_h100`, 30 min cap, `EPOCHS=10000`, chained (task 0
   elephant->giraffe, self-resubmits task 1 giraffe->elephant on success).

## Results

Prep (all 6 animals, CPU only): every final mesh independently re-verified
with trimesh -- 0 duplicate vertices, 0 zero-area faces, 0 non-manifold
edges, all watertight, 5000-5451 vertices (well under the 7k ceiling).
Worst-case source/target bucket mismatch for elephant<->giraffe dropped from
**7.0x** (bucket 5, 87 vs 609, the originally diagnosed cause) to **2.03x**
(609 vs 300); every other bucket is under 1.9x, and the new neck bucket
matches almost exactly (869 vs 877, 1.01x).

Training (`sbatch job_dev_h100_oracle_neckfix_single.sh 0`, jobs `5707816` and
`5707950`): both directions **COMPLETED**, 10:36 and 10:37 elapsed against a
30-minute budget, no NaN/crash.

Quantitative check, reproducing the exact "does the deformed bucket exceed
the target's own extent" test that first diagnosed the trunk problem
(giraffe->elephant, unit-box-normalized):

| Bucket | Deformed extent (norm) | Target extent (norm) | Exceeds target? |
| --- | ---: | ---: | --- |
| 5 (trunk) | 0.7765 | 0.7892 | **No** (was: yes, by 67%) |
| 7 (ears) | 0.7363 | 0.7653 | No |
| 8 (neck) | 1.3318 | 1.3311 | marginally (0.05%) |
| 1 (torso) | 1.3502 | 1.3676 | No |

The trunk no longer over-expands past the target's own size. Visually
(`epoch_renders/epoch_10000/grid_all_views.png` in each output dir):
elephant->giraffe produces a clean elongated neck with correct leg placement
and small ossicones; giraffe->elephant produces recognizable ears and a
cleaner tail than the original bad run, though the trunk still curls upward
rather than hanging down -- growing a trunk from a source with no equivalent
appendage remains the harder direction, consistent with the asymmetry
predicted in the diagnosis above.

## What this answers

Vertex-count-per-bucket and segmentation-quality are both real, but not in
the way the original bucket-balancing attempt assumed: segmentation source
matters (DenseCorr3D groups > PartField clusters for cross-species
consistency, already validated pre-existing), and vertex-count imbalance
matters specifically when a bucket is undersized in absolute terms -- fixed
by flooring each animal's own small buckets once, not by forcing exact
pairwise parity after the fact (which is what broke today).

## How to run further pairs

`prepare_oracle_neckfix_animals.sh` already prepared all 6 animals
(elephant, moose, giraffe, panther, bear, cheetah); extending
`run_oracle_neckfix_pair.sh` with more `PAIR_ID` cases (e.g. moose<->giraffe,
another appendage-growth test with antlers) is a small addition reusing the
same prepared assets, no new mesh prep needed.

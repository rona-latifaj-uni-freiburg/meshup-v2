# Project Audit: Holes, Controls, and Risk List

Date: 2026-08-21

Scope: I reviewed the current report in `report_final/main.tex`, the DenseMatcher comparison artifacts, the deformation/evaluation scripts, and the current experiment state. I also ran a new no-deformation control analysis on the clean DenseMatcher-matched subset and one A100-short global-Chamfer-only smoke run on the worst eggplant pair.

## Executive verdict

The project is strong, and the report is much more honest than a typical first write-up: it already separates fair from diagnostic results, discloses orientation/provenance confounds, keeps hard failures in the tables, and explains failure mechanisms. The main remaining scientific hole is not DenseMatcher anymore; it is the missing source-as-is/no-deformation baseline.

On the clean 9-category matched subset, the method still beats DenseMatcher clearly, but the no-deformation baseline is surprisingly competitive:

| Aggregation | No deformation AUC / Err | MeshUp PF-mean AUC / Err | DenseMatcher AUC / Err |
|---|---:|---:|---:|
| Clean 9, category-unweighted | 0.8292 / 4.48 | 0.8219 / 5.12 | 0.7213 / 9.75 |
| Clean 9, pair-weighted over 128 directed pairs | 0.8267 / 4.59 | 0.8410 / 4.38 | 0.7264 / 9.54 |

This does not kill the project. It changes the precise claim. The safest claim is now:

> PartField-guided deformation outperforms DenseMatcher under the local same-evaluator protocol, but its added value over an already-aligned source mesh is mixed: positive by pair-weighted average and in 7/9 clean categories, negative under category-unweighted averaging because eggplant is a severe failure.

That is still a good result, but the report should not imply that deformation or PartField guidance is always better than doing nothing.

## New checks I ran

### 1. Strict no-deformation baseline

I scored each clean-category matched source/target pair by using the undeformed source as the predicted correspondence carrier, then evaluating with the same DenseCorr3D graph-geodesic semantic AUC/Err convention used elsewhere.

Artifacts:

- `jobs_with_target_guidance/cross_animal_spike_runs/evaluation/controls_20260821/clean9_no_deformation_control_densematcher_matched_pairs.csv`
- `jobs_with_target_guidance/cross_animal_spike_runs/evaluation/controls_20260821/clean9_no_deformation_control_densematcher_matched_pairs_summary.csv`

It covered 128 unique directed clean-category pairs with 0 missing pairs.

Per-category results:

| Category | n pairs | No deformation | MeshUp PF mean | DenseMatcher | MeshUp vs no-deform |
|---|---:|---:|---:|---:|---:|
| backpack | 12 | 0.8765 / 3.17 | 0.8805 / 3.06 | 0.6706 / 9.22 | +0.0040 / -0.10 |
| broccoli | 20 | 0.5810 / 11.69 | 0.5875 / 11.31 | 0.6883 / 7.98 | +0.0065 / -0.38 |
| carrot | 12 | 0.5973 / 10.55 | 0.7025 / 7.65 | 0.6404 / 10.13 | +0.1052 / -2.91 |
| cars | 20 | 0.8895 / 2.78 | 0.9449 / 1.40 | 0.6345 / 12.84 | +0.0555 / -1.39 |
| celery | 6 | 0.9000 / 2.50 | 0.8936 / 2.66 | 0.8604 / 3.49 | -0.0064 / +0.16 |
| chairs | 20 | 0.9154 / 2.16 | 0.9350 / 1.66 | 0.8212 / 6.33 | +0.0195 / -0.50 |
| eggplant | 6 | 0.8197 / 4.51 | 0.5265 / 16.48 | 0.6927 / 7.85 | -0.2931 / +11.96 |
| knife | 12 | 0.9469 / 1.33 | 0.9679 / 0.80 | 0.5780 / 27.37 | +0.0211 / -0.53 |
| shampoo | 20 | 0.9366 / 1.60 | 0.9584 / 1.04 | 0.9058 / 2.50 | +0.0218 / -0.56 |

Additional facts:

- MeshUp is better than no-deformation on both AUC and Err for 83/128 directed pairs.
- MeshUp is worse than no-deformation on both AUC and Err for 45/128 directed pairs.
- The 45 both-worse pairs are spread across categories, not only eggplant: backpack 6, broccoli 9, carrot 2, cars 1, celery 6, chairs 8, eggplant 6, knife 2, shampoo 5.
- The worst pair is eggplant `691d0e4cb6a1409ea72d960d0ab144ea -> 334a3ab0383f49d1bd34575a36ddf4d6`: no-deform 0.8110 / 4.72, MeshUp PF mean 0.3269 / 27.51, MeshUp best PF 0.7865 / 5.34.

Interpretation: the clean categories are often already roughly aligned enough that source-as-correspondence is a serious baseline. The report should include this baseline in the headline comparison table.

### 2. A100-short global-Chamfer-only smoke run

I created and submitted a 20-minute A100-short wrapper:

- Script: `jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_a100_short_eggplant_control_global_only.sh`
- Slurm job: `6444026`
- Partition/time: `gpu_a100_short`, `00:20:00`
- Actual runtime: 2 minutes 45 seconds
- GPU: NVIDIA A100-PCIE-40GB
- Variant: `VARIANT_ID=0`
- Weights: global Chamfer 750.0, PartField Chamfer 0.0
- Output: `jobs_with_target_guidance/cross_animal_spike_runs/outputs/controls_20260821_gpu_smoke/eggpS0T1_controls_20260821_eggplant_global_chamfer_only_globalonly_control_1000ep_1000ep_6444026`

Scored artifacts:

- `jobs_with_target_guidance/cross_animal_spike_runs/evaluation/controls_20260821/eggplant_global_only_1000ep_geodesic.csv`
- `jobs_with_target_guidance/cross_animal_spike_runs/evaluation/controls_20260821/eggplant_global_only_surface/summary.csv`

Worst eggplant pair comparison:

| Method/control | AUC | Err | p95 Err | Same-bucket |
|---|---:|---:|---:|---:|
| No deformation | 0.8110 | 4.72 | 11.20 | 0.2051 |
| MeshUp PF mean | 0.3269 | 27.51 | -- | -- |
| MeshUp best PF | 0.7865 | 5.34 | -- | -- |
| DenseMatcher | 0.6833 | 7.95 | -- | -- |
| Global-only 1000ep | 0.7135 | 7.17 | 19.14 | 0.1683 |

Surface-quality result for global-only:

| Metric | Value |
|---|---:|
| Global Chamfer L1 sum | 0.01497 |
| F@0.02 | 0.99295 |
| Normal consistency | 0.96693 |
| Robust Hausdorff p95 | 0.01340 |
| Chamfer reduction vs source | 90.81% |
| Edge-relative p95 | 0.32453 |
| Normal flip fraction | 0.0 |

Interpretation: global-only fits the target surface beautifully but still loses to the undeformed source on semantic AUC/Err. This is an important failure diagnosis: for thin-band semantic labels like eggplant, "better surface fit" can make semantic correspondence worse. The report already says Chamfer can mix parts; this smoke test gives a concrete quantitative proof.

One caveat: `semantic_nn_agreement=1.0` in the surface CSV is not evidence of DenseCorr semantic correctness because the run used PF01/one-bucket labels for the surface evaluator. Use the geodesic DenseCorr3D CSV for semantic claims.

### 3. Syntax/static checks

Passed:

- `python -m py_compile jobs_with_target_guidance/evaluate_densecorr_paper_style_geodesic.py jobs_with_target_guidance/evaluate_deformation_quality.py jobs_with_target_guidance/partfield_bucket_llm.py jobs_with_target_guidance/partfield_chamfer.py jobs_with_target_guidance/evaluate_densematcher_pair_predictions.py`
- `bash -n jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_a100_short_eggplant_control_global_only.sh jobs_with_target_guidance/artur_soft_runs/jobs/run_artur_chamfer_ablation.sh activate_meshup_new.sh`

Not done: I did not run a full test suite; I did not find a single project-level test command that covers the research pipeline end to end.

## Highest-priority holes in the report

### Hole 1: Missing no-deformation baseline in the headline

Where it matters:

- `report_final/main.tex:123-127` states the headline DenseMatcher comparison.
- `report_final/main.tex:924-928` argues that the clean-subset win is not just "any deformation wins this metric."
- `report_final/main.tex:940-946` gives the headline table without a no-deformation column.

Problem: no-deformation is stronger than MeshUp under category-unweighted clean-9 aggregation and only slightly weaker under pair-weighted aggregation. The statement at `report_final/main.tex:924-928` is especially too strong after this control.

Suggested fix:

- Add a `No deformation AUC / Err` column to the headline table.
- Replace "not any deformation wins this metric" with a more careful claim: "deformation alone is not sufficient; the added value over source-as-is is positive on most categories and pair-weighted average, but not uniformly."
- Include the eggplant global-only smoke result as a short diagnostic paragraph.

### Hole 2: The abstract overstates the objective used by the headline runs

Where it matters:

- `report_final/main.tex:114-117`
- `report_final/main.tex:1580-1591`

Problem: the report describes the method as global Chamfer + per-bucket Chamfer + regularizers. But the DenseMatcher fair rows are described elsewhere as having only active PartField Chamfer, and many job configs use global Chamfer weight 0.0. The pipeline supports both, but the reported headline runs are not always the combined objective.

Suggested fix:

- In the abstract, say the pipeline supports both global and local Chamfer, while the main fair DenseMatcher rows are PartField-guided/PartField-only unless otherwise stated.
- Add a table row/column that clearly names `global-only`, `PF-only`, and `global+PF`.

### Hole 3: The table `n` values are easy to misread

Where it matters:

- `report_final/main.tex:980-983`

Problem: the per-category table says "matched directed pairs only", but the `n` column is actually `(directed pair, PF setting)` rows for many categories. Example: backpack has 12 unique directed pairs but `n=60` because PF04--08 gives 5 rows per pair.

Suggested fix:

- Rename `n` to `n rows` and add `n unique pairs`.
- For no-deformation comparisons, use unique directed pairs so the baseline is not implicitly under/over-counted by PF range length.

### Hole 4: Celery should not be treated as an ordinary clean win

Where it matters:

- `report_final/main.tex:998`
- The cross-category discussion elsewhere says celery has byte-identical geometry and should be excluded from generalization-style claims.

Problem: celery is in the clean DenseMatcher table as a win, but the report also states that celery is geometrically trivial/identity-like. The new no-deformation baseline confirms this is a weak claim category: no-deformation is 0.9000 / 2.50, MeshUp is 0.8936 / 2.66.

Suggested fix:

- Keep celery in the table for transparency, but mark it as "clean/trivial geometry" or show a sensitivity line with celery excluded.

### Hole 5: Best-PF/zero-selection language is mostly good, but the project still needs fixed protocols

The report is already careful not to headline best-PF rows. Good. Still, many supporting tables and figures use best-PF or selected pairs. To remove ambiguity:

- Define one fixed no-selection protocol before evaluation.
- Define one fixed category-level bucket selector before evaluation.
- Report best-PF only as diagnostic upper bound.

## Missing ablations and checks

Priority 0, needed before a final scientific claim:

- No-deformation/source-as-is baseline in every DenseMatcher comparison table.
- Rigid/PCA-aligned source baseline: source after only label-free canonical alignment, no learned deformation.
- Global-only vs PF-only vs global+PF on the same clean 9 categories and same matched pairs.
- Random/permuted bucket negative control: same bucket sizes, wrong bucket IDs. This tests whether PartField semantics matter or whether the regularizer/surface fit explains the win.
- Bootstrap confidence intervals over directed pairs and over categories. Some deltas are small enough that a CI matters.
- Fixed PF or fixed selector protocol chosen before scoring. No more post-hoc "best PF" except as upper bound.

Priority 1, important for diagnosis:

- Seed variance for representative easy, medium, and hard pairs. The report already acknowledges single-run results, but this is still a real uncertainty.
- Surface quality on the same DenseCorr3D clean matched pairs, not only SAM3D animals/vehicles.
- Self-intersection / fold / triangle quality metrics for every reported run.
- Official DenseCorr3D geodesic matrices or a sensitivity comparison among graph, heat, and exact geodesics where available.
- Pair-level failure gallery for the 45/128 cases where MeshUp is worse than no-deformation on both metrics.

Priority 2, good research extensions:

- Gating rule: detect when not to deform, or when to keep global Chamfer active, based on bucket support/extent/entropy.
- Soft transport between source and target buckets instead of assuming bucket `k` matches bucket `k`.
- Axis/profile loss for thin-band categories such as zucchini, cucumber, and eggplant.
- Compare with simple ICP / non-rigid ICP / CPD baselines if feasible.
- Evaluate cycle consistency or inverse consistency, because nearest-neighbor-after-deformation can hide many-to-one correspondence collapse.

## Security and reproducibility risks

I did not find obvious committed API keys or passwords in the scanned project-owned files. The risks are mostly reproducibility and unsafe research-code trust boundaries:

| Risk | Location | Why it matters | Suggested fix |
|---|---|---|---|
| Hard-coded shared cache paths | `activate_meshup_new.sh:24-25` | The A100 run warns that `/work/dlclarge1/jesslen-od3d/cache/transformers` is not writable; shared absolute caches are brittle and can leak environment assumptions. | Default to a workspace cache, e.g. `${HF_HOME:-$SCRIPT_DIR/.cache/huggingface}` and `${TRANSFORMERS_CACHE:-$SCRIPT_DIR/.cache/transformers}`. |
| Optional LLM selector changes behavior when `OPENAI_API_KEY` exists | `jobs_with_target_guidance/partfield_bucket_llm.py:676-716` | `provider=auto` can silently switch from heuristic to OpenAI. It also sends mesh summaries, paths, prompts, and calibration priors to an external API. | For benchmark/report runs, force `provider=heuristic` or record the exact provider/model/response. Add a privacy note if using OpenAI. |
| `torch.load` on `.pt/.pth` input | `jobs_with_target_guidance/partfield_chamfer.py:222-224` | `torch.load` can execute code for untrusted pickle-backed files. | Prefer `.npy/.npz`; for PyTorch 2.1+ use `weights_only=True` where possible, or reject `.pt/.pth` unless explicitly trusted. |
| Absolute project path in new Slurm wrapper | `jobs_with_target_guidance/cross_animal_spike_runs/jobs/job_a100_short_eggplant_control_global_only.sh:13` | Not portable to another checkout/user. | Use `${SLURM_SUBMIT_DIR}` or compute repo root from script location. |
| Huge untracked/generated state | `git status --short` | Many scripts, reports, specs, outputs, and evaluation files are untracked. A reader cannot reproduce the report from a clean commit. | Commit the exact scripts/manifests used for final tables or write a frozen artifact manifest. Keep generated meshes/videos out of git if needed, but track checksums and commands. |
| No reusable no-deformation evaluator found | project search | The strongest new control is currently a generated CSV, not a first-class script. | Add a small evaluator script that takes `fair_comparison_meshup_vs_densematcher_matched_rows.csv` and writes the no-deform tables. |

## Research questions to answer explicitly

The report's current research questions are good, but after these controls I would add these:

1. When is source-as-is already a strong correspondence baseline, and can the method detect that before deforming?
2. Does PartField guidance improve semantic correspondence after controlling for surface overlap?
3. Which categories benefit from deformation itself, and which only benefit visually while losing semantic correspondence?
4. Can a global-shape guard prevent hard-bucket failures on smooth single-part objects?
5. Are the DenseMatcher wins robust to bootstrap resampling, geodesic backend, and removing trivial categories such as celery?
6. Can bucket-quality diagnostics predict failures before spending GPU time?

## Suggested report patch order

1. Add the no-deformation table to Section `Fair Comparison Against DenseMatcher`.
2. Rewrite the abstract headline to mention no-deformation explicitly.
3. Add the eggplant global-only smoke run as a short diagnostic in the root-cause/failure section.
4. Fix the `n` column semantics in the full per-category table.
5. Add celery/trivial-geometry sensitivity.
6. Move random-bucket, global-only, and global+PF ablations to the first future-work item, above the broader preprocessing-confound item.

## Bottom line

The project is not falling apart. The core DenseMatcher comparison remains interesting, and the failure analysis is unusually careful. But the cleanest scientific story is now narrower and stronger:

> PartField-guided deformation is a useful prior that often improves semantic correspondence over both DenseMatcher and source-as-is, but the gain over source-as-is is not universal. Smooth/thin-band categories expose a real failure where deformation improves surface fit while damaging semantic correspondence.

That version is much harder to attack than the current headline because it already includes the most damaging control.

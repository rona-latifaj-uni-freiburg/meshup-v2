# SDS and DINO Removal Rationale

Date: 2026-07-22

Purpose: presentation-ready argument for why the project moved away from the original MeshUp SDS / 2D diffusion path and toward direct 3D target guidance with Chamfer, PartField buckets, and supervised DenseCorr3D meshes.

## One-Sentence Claim

Removing SDS was the right engineering and research choice because the old MeshUp path spent most of its compute on an XL text-to-image prior and image-space semantic features, while the current pipeline spends the optimization budget on direct 3D evidence: target mesh geometry, part-aware Chamfer losses, and supervised DenseCorr3D animal meshes.

## What The Old Pipeline Was Doing

The original MeshUp-style branch deformed a mesh through rendered images:

1. Render the current mesh from random camera views.
2. Send those images through a frozen text-to-image diffusion model.
3. Use SDS-style gradients to make the rendered views look more like the text prompt.
4. Optionally add DINOv2 feature consistency or cross-attention losses to preserve semantic correspondences.
5. Regularize the mesh deformation with Jacobian smoothness.

This is conceptually powerful, but it is indirect. The optimizer sees 2D render feedback, not a supervised 3D target. If the image prior makes poor images for the prompt, the mesh receives poor gradients.

Primary-source context:

- DreamFusion introduced SDS/PDD as a way to use a pretrained 2D text-to-image diffusion model as a prior for optimizing random 2D renderings of a 3D object: https://arxiv.org/abs/2209.14988
- MeshUp extends this idea with Blended Score Distillation and uses diffusion U-Net attention activations to calculate deformation gradients: https://arxiv.org/abs/2408.14899

## Local Evidence From This Repo

The old tracked config used high image-prior settings:

- `image_weight: 100.0`
- `guidance_scale: 100`
- `batch_size: 8`
- `accum_iter: 2`
- `train_res: 512`
- `model_size: XL`
- optional DINO with `dino_n_views: 8`

Local source: `configs/tracked_config.yml`.

The SDS implementation maps `model_size: XL` to `DeepFloyd/IF-I-XL-v1.0`, loads it through `diffusers.IFPipeline`, and performs classifier-free guidance by doubling the latent batch before the U-Net forward. Local source: `deepfloyd.py`.

The no-SDS branch explicitly disables the image prior:

- `use_sds: false`
- `image_weight: 0.0`
- `use_dino_loss: false`
- `use_cross_attn_loss: false`
- `use_target_mesh_guidance: false`
- direct geometry terms remain: global Chamfer and PartField bucket Chamfer

Local source: `jobs_with_target_guidance/sds_ablation_runs/configs/car_no_sds_ablation.yml`.

The tracked loop implements the compute difference directly: if `use_sds` is false, it sets `image_weight` to zero and skips DeepFloyd pipeline loading and text embeddings. Local source: `loop_tracked.py`.

## Compute Argument

In the old SDS loop, every epoch used rendered views and a diffusion U-Net forward. With the common old settings:

```text
epochs x batch_size x accum_iter x CFG branches
2500 x 8 x 2 x 2 = 80,000 diffusion U-Net batch-item evaluations
4000 x 8 x 2 x 2 = 128,000 diffusion U-Net batch-item evaluations
6000 x 8 x 2 x 2 = 192,000 diffusion U-Net batch-item evaluations
```

These are batched tensor operations, not literal separate Python calls, but this captures the scaling. The current no-SDS geometry path performs zero DeepFloyd/IF U-Net evaluations.

DeepFloyd IF-I-XL is not a small helper model. Its model card lists Stage I XL as a 4.3B parameter text-to-image diffusion module, with a cascaded design for 64, 256, and 1024 pixel images: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0

This matches the practical observation: smaller/weaker image generators gave bad images, and bad images made bad mesh gradients. SDXL's paper supports the broader point that high-quality image priors became much larger: SDXL uses a 3x larger U-Net backbone than earlier Stable Diffusion variants, adds more attention capacity, a second text encoder, and a refiner to improve visual fidelity: https://arxiv.org/abs/2307.01952

## Measured Local Runtime And FID Evidence

Old SDS/DINO ablations on A1 hound->hippo:

| Run | Epochs | Guidance | Job | Elapsed | FID |
|---|---:|---|---:|---:|---:|
| A1 baseline dev | 2500 | SDS, no DINO | 3740938 | 00:24:10 | 605.1927 |
| A1 DINO dev | 2500 | SDS + DINO | 3740939 | 00:24:49 | 592.1992 |
| A1 baseline non-dev | 2500 | SDS, no DINO | 3741280 | 00:28:00 | 494.8596 |
| A1 DINO non-dev | 6000 | SDS + DINO | 3741281 | 00:56:13 | 503.3522 |

Interpretation:

- The controlled dev comparison shows DINO improved FID by only 12.99 points, or about 2.15 percent, while still requiring the SDS pipeline.
- The non-dev comparison is not perfectly controlled because the DINO run used more epochs, but it is still a warning sign: the longer DINO run did not beat the shorter baseline.
- The old FID workflow itself compared mesh renders to reference images generated from a diffusion model, so even the metric was image-prior dependent.

Local sources:

- `READMEs/FID_DAILY_UPDATE_2026-03-23.md`
- `outputs/ablation/ablation_A1_baseline_hound_hippo/evaluation/fid_results.json`
- `outputs/ablation/ablation_A1_dino_hound_hippo/evaluation/fid_results.json`
- `outputs/ablation/ablation_A1_baseline_hound_hippo_dev/evaluation/fid_results.json`
- `outputs/ablation/ablation_A1_dino_hound_hippo_dev/evaluation/fid_results.json`

Current no-SDS / PartField examples from the final-mesh jobs:

| Run | Job | Elapsed | Notes |
|---|---:|---:|---|
| horse2->giraffe4 | 5816558 | 00:17:27 | 4000 epoch PartField/Chamfer run |
| panda3->bear | 5817242 | 00:13:41 | 4000 epoch PartField/Chamfer run |
| fox->cat2 / panda2->panda3 batch | 5818235 | 00:08:45 | 4000 epoch PartField/Chamfer run |
| panther->horse2 chain | 5829892 | 00:04:38 | chain stage |
| horse2->dense_giraffe chain | 5829931 | 00:11:10 | chain stage |

These current timings are not an exact apples-to-apples benchmark against the old hound->hippo runs, but they make the operational point clear: the final no-SDS jobs complete in minutes and do not load the diffusion prior.

## What DINO Gave Us

DINOv2 was still useful. It gave a semantically meaningful image feature space and helped with visualizing correspondences, especially through PCA-colored meshes.

Primary-source context:

- DINOv2 was designed to produce all-purpose visual features across image distributions without fine-tuning, and it was trained from curated unlabeled image data: https://arxiv.org/abs/2304.07193
- The DINOv2 repository reports pretrained model scales from ViT-S/14 at 21M parameters through ViT-g/14 at 1.1B parameters, with models pretrained on 142M images without labels: https://github.com/facebookresearch/dinov2

Local DINO facts:

- DINO PCA visualization used 8 rendered views at about 512 pixels and reduced high-dimensional features to RGB colors.
- ViT-S features are 384D, ViT-B 768D, ViT-L 1024D, and ViT-g 1536D.
- DINO loss used a frozen DINOv2 model, but kept gradients through rendered images so the loss could move mesh vertices.
- DINOv2 uses 14x14 patches. At 224 pixels this is 16x16 = 256 patch tokens per view; at 512 pixels the image is resized to 504 and yields 36x36 = 1296 patch tokens per view. Eight 512-pixel views therefore produce 10,368 patch tokens before aggregation.

Local sources:

- `semantic_tracking/DINO_PCA_README.md`
- `semantic_tracking/dino_correspondence_loss.py`
- `semantic_tracking/dino_features.py`

## Why DINO Was Not Enough

DINO is a strong image feature extractor, but it is still image-space guidance:

- It depends on rendered views, camera coverage, occlusion, lighting, and silhouette quality.
- It can say two rendered patches look semantically similar, but it does not know the exact 3D vertex correspondence unless the data or loss supplies it.
- It can preserve source semantics, but preservation is not the same as matching a target mesh.
- In cross-animal deformation, topology and point distribution matter. A giraffe neck, elephant trunk, or thin legs can be under-constrained if the source and target semantic buckets have very different vertex counts.

This explains the practical transition:

```text
Old question:
Does this rendered mesh look like a plausible prompt image?

Current question:
Is this 3D surface close to the target 3D surface, globally and within corresponding semantic parts?
```

## Why The Current 3D Pipeline Is Better For This Project

The current target-guided pipeline is better matched to the project goal:

1. Source mesh enters the deformation loop.
2. PartField features assign semantic buckets on source and target.
3. Global Chamfer pulls the whole mesh toward the target shape.
4. PartField bucket Chamfer pulls corresponding semantic parts together.
5. Jacobian and edge regularizers keep the deformation smooth and prevent local collapse.
6. DenseCorr3D meshes give supervised, annotated, database-quality animals for quantitative evaluation.

The key improvement is that the target is no longer a generated 2D image distribution. The target is an actual 3D mesh, and for DenseCorr3D animals it comes from a supervised annotated source.

## Suggested Slide Section

### Slide: Why Removing SDS Was Necessary

Message:

SDS made the deformation depend on a 2D image generator. When the generated images were poor, the mesh gradients were poor. Good image priors required XL diffusion models, which made every optimization run expensive.

Figure suggestion:

- Left: old SDS loop diagram: mesh -> render -> XL diffusion/SDS -> 2D gradient -> mesh.
- Right: current loop diagram: source mesh + target mesh + PartField buckets -> global/local Chamfer -> regularized deformation.

### Slide: Compute Cost

Message:

The old 2500-epoch SDS setting implied about 80k diffusion U-Net batch-item evaluations because each epoch used 8 views, 2 accumulation passes, and conditional/unconditional CFG branches. The current no-SDS path performs zero diffusion U-Net evaluations.

Table suggestion:

| Pipeline | Heavy model in loop | Main target signal | Typical local timing |
|---|---|---|---|
| MeshUp SDS | DeepFloyd IF/XL diffusion U-Net | text/image prior through renders | 24-56 min in old A1 jobs |
| SDS + DINO | diffusion U-Net + DINO ViT | image prior + rendered feature consistency | 24-56 min, small/mixed FID gain |
| Current 3D | no diffusion model | global Chamfer + PartField Chamfer | 4-17 min in final animal jobs |

### Slide: DINO Was Useful, But Not The Final Objective

Message:

DINOv2 helped reveal semantic structure and gave useful PCA coloring, but it remained view-dependent 2D evidence. For final animal deformations, supervised 3D meshes and part-aware geometry losses are more direct.

### Slide: Main Takeaway

Message:

The project did not remove semantics. It moved semantics out of the unstable text-to-image loop and into explicit 3D correspondences: PartField/DenseCorr3D labels plus local Chamfer constraints.

## Supervisor-Friendly Wording

"My earlier MeshUp experiments used SDS and DINO because they were good tools for exploring text-driven deformation and semantic correspondence. However, the experiments showed a mismatch with the final goal. SDS optimizes through 2D generated images, so the result is limited by the image generator. To get acceptable images I needed an XL diffusion model, which made the loop computationally expensive. DINO helped with semantic features and visualization, but it was still tied to rendered views and did not give direct 3D target correspondence. The current pipeline removes this bottleneck: it uses actual target meshes, PartField semantic buckets, global and local Chamfer losses, and regularization. This gives a more measurable and more reliable deformation pipeline, especially now that the important animal experiments use supervised DenseCorr3D meshes."

## Sources

Primary external sources:

- DreamFusion, "Text-to-3D using 2D Diffusion": https://arxiv.org/abs/2209.14988
- MeshUp, "Multi-Target Mesh Deformation via Blended Score Distillation": https://arxiv.org/abs/2408.14899
- DeepFloyd IF-I-XL model card: https://huggingface.co/DeepFloyd/IF-I-XL-v1.0
- SDXL paper: https://arxiv.org/abs/2307.01952
- DINOv2 paper: https://arxiv.org/abs/2304.07193
- DINOv2 repository/model table: https://github.com/facebookresearch/dinov2

Local evidence:

- `configs/tracked_config.yml`
- `deepfloyd.py`
- `loop_tracked.py`
- `jobs_with_target_guidance/sds_ablation_runs/configs/car_no_sds_ablation.yml`
- `READMEs/FID_DAILY_UPDATE_2026-03-23.md`
- `READMEs/FID_WORKFLOW.md`
- `semantic_tracking/DINO_PCA_README.md`
- `semantic_tracking/dino_correspondence_loss.py`

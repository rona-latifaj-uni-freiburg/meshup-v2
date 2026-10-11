# PartField-Derived Mesh Dataset And Motivation

## One-Sentence Project Motive

This project studies part-aware mesh-to-mesh deformation: given a source mesh and a target mesh with different geometry and topology, the goal is to deform the source toward the target while preserving meaningful part structure instead of only minimizing surface distance.

## Why This Problem Matters

Many 3D meshes from scans, reconstruction systems, generative models, or public asset collections are not directly comparable. They can have different topology, different vertex counts, different proportions, missing details, and no consistent part labels. A plain global Chamfer loss can make one surface close to another, but it does not know that a leg should remain a leg, a head should remain a head, or a car wheel should stay aligned with a wheel-like region.

The core motivation is therefore semantic deformation: the method should move geometry into the target shape while preserving an interpretable mapping between source regions and target regions. This is important when downstream systems need more than a visually similar mesh. They need object parts, stable correspondences, editable regions, and geometry that remains physically plausible.

## Why This Is Relevant To Robotics And World Models

Robots and world models need object-level and part-level structure. A robot does not only need to know that a mesh is close to another mesh in Euclidean distance; it needs to know where functional or semantic regions are: legs, torso, head, wheels, doors, front, back, support regions, contact regions, and so on.

Part-aware deformation can help with:

- building canonical object representations from messy 3D assets or scans;
- transferring shape information between related objects without manual correspondences;
- creating controlled synthetic training data with consistent part structure;
- improving object-centric world models by giving them reusable, part-aligned geometry;
- supporting robotic interaction, because manipulation often depends on parts rather than whole-object shape;
- evaluating whether a generated or reconstructed mesh preserves semantic structure;
- converting many heterogeneous meshes into a more comparable representation.

In simple words: this project helps turn unstructured 3D meshes into objects whose geometry and parts can be compared, transferred, and reused.

## What The PartField-Derived Dataset Contains

The project has produced a reusable collection of PartField feature tensors, labels, colored part visualizations, and summaries. This is not a manually annotated ground-truth dataset, but it is a derived dataset of unsupervised semantic mesh bucketings that can be reused across deformation experiments.

Current local stats from `jobs_with_target_guidance/partfield_features` and `jobs_with_target_guidance/partfield_segments`:

| Artifact | Count | Meaning |
|---|---:|---|
| PartField feature files | 88 | Cached 448D PartField descriptors extracted from meshes. |
| Colored PartField meshes | 376 | PLY visualizations where each bucket/part has a distinct color. |
| Label files | 376 | Saved bucket labels, usually including face-level and vertex-level labels. |
| Summary files | 131 | JSON metadata for bucket count, clustering settings, source paths, and bucket statistics. |
| Contact-sheet images | 30 | Side-by-side visual summaries of bucket layouts for pairs or groups. |
| Distinct colored object names | 56 | Unique mesh/object names appearing in colored PartField outputs, with some aliases/duplicates across experiments. |

Feature tensors are 448-dimensional. The most common feature tensor shapes are:

| Feature Tensor Shape | Count |
|---|---:|
| `(10000, 448)` | 39 |
| `(11000, 448)` | 22 |
| `(9420, 448)` | 10 |
| `(10244, 448)` | 3 |
| `(10326, 448)` | 3 |

DenseCorr3D animal PartField features are lower-resolution because they come from the remeshed DenseCorr3D `simple_mesh.obj` files:

| DenseCorr Animal Feature Shape | Count |
|---|---:|
| `(3320, 448)` | 2 |
| `(3802, 448)` | 2 |
| `(3620, 448)` | 1 |
| `(3450, 448)` | 1 |
| `(3130, 448)` | 1 |
| `(3297, 448)` | 1 |

The bucket visualizations span several granularities:

| Bucket Count | Colored Meshes |
|---:|---:|
| 5 | 24 |
| 6 | 46 |
| 7 | 48 |
| 8 | 88 |
| 9 | 48 |
| 10 | 34 |
| 12 | 44 |
| 20 | 42 |
| 28 | 2 |

Broadly, the colored PartField outputs cover:

| Group | Colored Meshes |
|---|---:|
| Car and vehicle experiments | 176 |
| SAM3D animal experiments | 96 |
| DenseCorr3D animal experiments | 62 |
| Panther-horse-giraffe chain experiments | 24 |
| Dog/hound/test mesh experiments | 18 |

## What Is Stored For Each Bucketed Mesh

Each PartField summary stores enough information to make the bucketed mesh reusable:

- the original mesh path;
- the cached PartField feature path;
- the number of feature rows and feature dimension;
- whether features were resolved as face-level or vertex-level features;
- mesh vertex count and face count;
- face label path;
- vertex label path;
- compressed label archive path;
- colored visualization path;
- per-bucket statistics, including bucket size, centroid, and spatial extent.

This is why it is fair to describe the output as a small derived dataset: it is not only pictures. It contains machine-readable features, labels, and geometry metadata that can drive new deformation jobs and evaluations.

## Suggested Slide Wording

I created a PartField-derived mesh bucket dataset to support semantic deformation. It contains 88 cached PartField feature tensors, 376 colored bucketed mesh visualizations, 376 matching label files, 131 metadata summaries, and 30 contact-sheet visualizations. The dataset spans vehicle meshes, SAM3D animal meshes, DenseCorr3D animal meshes, chain-deformation animals, and dog/test meshes. The bucketings were generated at multiple semantic granularities from 5 to 28 buckets, which lets the deformation pipeline test coarse body-level alignment and finer part-level alignment.

## What Problem The Project Solves

The project addresses a gap between raw mesh similarity and semantic mesh understanding. Standard deformation losses can make a mesh close to a target surface, but they can still produce semantically wrong alignments. For example, a neck may absorb body geometry, legs may fold, or a source region may move to a geometrically nearby but semantically wrong target region.

The proposed pipeline adds semantic structure through PartField buckets. These buckets provide local alignment regions without requiring manually annotated point-to-point correspondences. The deformation is therefore guided by both global target shape and local part consistency.

## Use Cases

Robotics:
Part-aware object models can help robots reason about where parts are, how objects differ, and where contact or manipulation should happen.

World Models:
A world model benefits from stable object abstractions. Part-aware deformation can convert many different meshes into comparable, semantically structured shapes.

Synthetic Data Generation:
The pipeline can generate shape variations while preserving part labels, useful for training perception, segmentation, and correspondence models.

3D Generative Model Evaluation:
Generated meshes can be evaluated not only by surface similarity but also by whether their parts correspond to real object regions.

Animation And Rigging:
A source mesh can be reshaped toward a target while keeping reusable part structure, which is useful before rig transfer or deformation editing.

AR/VR And Game Asset Processing:
Assets from different sources can be normalized, aligned, and made more editable while keeping meaningful object regions.

Scientific Shape Analysis:
For animals, vehicles, or other categories, semantic deformation can support controlled comparison of shape differences across instances or species-like morphologies.

## Honest Limitation To Mention

The PartField bucket dataset is unsupervised. It is not the same as manually annotated ground truth. Its strength is that it gives scalable, reusable, semantic-like regions for deformation guidance. For final quantitative evaluation, the project should still compare against supervised labels where available, such as DenseCorr3D semantic groups.


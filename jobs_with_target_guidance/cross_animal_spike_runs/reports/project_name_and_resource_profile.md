# Project Name And Resource Profile

Date: 2026-08-03

## Recommended Project Name

**PartMorph3D: Semantic-Bucket Guided Mesh-to-Mesh Deformation**

Subtitle:

**Topology-preserving cross-animal deformation with local Chamfer guidance**

Why this name works:

- **PartMorph3D** sounds memorable and technical.
- **Semantic-bucket guided** is honest: the method uses semantic/part buckets as guidance, not perfect point-to-point ground truth.
- **Mesh-to-mesh deformation** says exactly what the project does.
- **Topology-preserving** is an important scientific constraint because the source mesh connectivity is preserved.
- **Local Chamfer guidance** highlights the main technical loss without pretending the method solves exact correspondence.

Other good options:

| Name | Full title | Tone |
|---|---|---|
| **PartMorph3D** | Semantic-Bucket Guided Mesh-to-Mesh Deformation | best balance, recommended |
| **BucketMorph** | Part-Aware Chamfer Deformation for Animal Meshes | clear and memorable |
| **SemChamfer3D** | Local Semantic Chamfer for Cross-Animal Mesh Deformation | most metric/loss-focused |
| **PartChamfer Mesh Deformation** | Part-Restricted Chamfer for Target-Guided Mesh Deformation | very honest, less flashy |
| **Semantic MeshMorph** | Semantic Part-Guided Mesh-to-Mesh Deformation | presentation-friendly |

## Resource Story For Slides

Current Slurm request:

| Resource | Requested in job scripts |
|---|---:|
| GPU | 1 H100 |
| CPUs | 8 |
| system memory | 50 GB |
| time limit | 30 min |

Actual usage from completed jobs:

| Resource | Measured usage | Interpretation |
|---|---:|---|
| GPU count | 1 GPU | correct; the pipeline is single-GPU |
| CPU utilization | 0.69-0.96 CPU cores on average | 8 CPUs are not needed for deformation |
| system RAM peak | 1.32-1.74 GiB | 50 GB is much more than needed |
| wall time | 3.72-18.32 min | 30 min is a safe limit for these runs |

Recommended realistic request for the current 5k/selected deformation jobs:

| Resource | Practical request | Conservative request |
|---|---:|---:|
| GPU | 1 GPU | 1 GPU |
| CPUs | 2 | 4 |
| system memory | 4 GB | 8 GB |
| time limit | 20 min | 30 min |

Best slide wording:

> Although the jobs were submitted with 1 H100, 8 CPUs, and 50 GB RAM, Slurm accounting shows the deformation stage is much lighter: peak system RAM stayed below 1.8 GiB and CPU usage averaged about one core. The main computation is a single-GPU PyTorch optimization, so the pipeline is not memory-heavy and does not require multi-GPU resources.

## Do We Use The CPUs?

Yes, but lightly.

The CPUs are used for:

- Python process orchestration;
- mesh loading and file I/O;
- label loading/conversion;
- CPU-side bookkeeping;
- writing outputs, logs, images, and meshes;
- some preprocessing or evaluation utilities.

But the expensive training operation is on the GPU:

- pairwise distances for Chamfer;
- tensor operations;
- gradient computation;
- Adam updates of the deformation variables.

The accounting shows that allocating 8 CPUs produced only about **8.6-12.0 percent CPU efficiency**, which is roughly **0.7-1.0 CPU core used on average**.

So the answer is:

> Yes, the code uses CPUs, but not eight of them. For the deformation stage, 2 CPUs should be enough, and 4 CPUs is a safe presentation-friendly request.

## Do We Need 50 GB RAM?

No, not for the current deformation jobs.

The measured peak system RAM was:

```text
minimum peak MaxRSS: 1.32 GiB
mean peak MaxRSS:    1.56 GiB
maximum peak MaxRSS: 1.74 GiB
```

So a 50 GB allocation is about:

```text
50 / 1.74 = 28.7x larger than the largest measured peak
```

Practical recommendation:

- `--mem=4G` should be enough for the current 5k/simple-mesh deformation jobs.
- `--mem=8G` is a conservative safe request for presentations and routine runs.
- `--mem=50G` should be described as historical over-allocation, not actual need.

Caveat:

PartField feature extraction, high-resolution original `color_mesh.obj` processing, or large evaluation jobs can need more memory than the deformation loop. The numbers above describe the completed target-guided deformation jobs that were profiled with Slurm accounting.

## Do We Use Only One GPU?

Yes.

Evidence:

- The job scripts request `#SBATCH --gres=gpu:1`.
- The Slurm accounting reports `gres/gpu=1`.
- The configs use a single GPU index, typically `gpu: 0`.
- There is no multi-GPU/distributed training path in these deformation jobs.

What we do not know exactly from the finished logs:

- Exact peak GPU VRAM was not recorded in the completed job logs or Slurm accounting fields available here.

How to measure it in the next run:

Add PyTorch peak-memory logging around the optimization:

```python
torch.cuda.reset_peak_memory_stats()

# run optimization

print("peak CUDA allocated GiB:", torch.cuda.max_memory_allocated() / 1024**3)
print("peak CUDA reserved GiB:", torch.cuda.max_memory_reserved() / 1024**3)
```

Or sample with `nvidia-smi` during the job. Until that is logged, the exact GPU count is known, but exact peak VRAM is not.

## Exact Slurm Accounting Rows Used

These are completed deformation jobs queried with `sacct`. `MaxRSS` is peak system RAM for the batch process, converted from KiB to GiB. CPU efficiency is `TotalCPU / (Elapsed * AllocCPUS)`.

| Job | Example type | Elapsed min | Total CPU min | Avg CPU cores | CPU efficiency | Peak system RAM |
|---|---|---:|---:|---:|---:|---:|
| 5708185 | DenseCorr source-stable | 3.88 | 3.42 | 0.88 | 11.0% | 1.46 GiB |
| 5708523 | DenseCorr source-stable | 3.72 | 3.34 | 0.90 | 11.2% | 1.52 GiB |
| 5816558 | SAM3D horse2->giraffe4 | 17.45 | 14.69 | 0.84 | 10.5% | 1.66 GiB |
| 5817242 | SAM3D panda3->bear | 13.68 | 11.03 | 0.81 | 10.1% | 1.55 GiB |
| 5818235 | SAM3D selected batch | 8.75 | 7.14 | 0.82 | 10.2% | 1.55 GiB |
| 5829892 | chain panther->horse2 | 4.63 | 3.19 | 0.69 | 8.6% | 1.32 GiB |
| 5829931 | chain horse2->dense giraffe | 11.17 | 10.69 | 0.96 | 12.0% | 1.49 GiB |
| 6123510 | DenseCorr 20k elephant->moose | 18.32 | 16.72 | 0.91 | 11.4% | 1.74 GiB |
| 6123511 | DenseCorr elephant->panther | 4.05 | 3.40 | 0.84 | 10.5% | 1.67 GiB |
| 6123512 | DenseCorr panther->giraffe | 3.82 | 3.45 | 0.90 | 11.3% | 1.66 GiB |

## Strong Presentation Table

Use this table instead of the original requested-resource table:

| Resource | Submitted allocation | Measured need | Presentation claim |
|---|---:|---:|---|
| GPU | 1 H100 | 1 GPU | single-GPU pipeline |
| CPUs | 8 | about 1 core average | 2-4 CPUs enough |
| system RAM | 50 GB | max 1.74 GiB | 4-8 GB enough |
| time limit | 30 min | max 18.32 min in profiled rows | 20-30 min safe |

One-sentence conclusion:

> The final target-guided deformation stage is computationally modest: it runs as a single-GPU optimization, uses under 2 GiB peak system RAM in measured jobs, and uses roughly one CPU core on average.


# Ablation Study Quick Start Guide

## What is This?

A systematic evaluation that isolates the contribution of each component:
- **Baseline:** Original MeshUp with no semantic losses
- **+DINO Loss:** Adds DINOv2 feature consistency
- **+Cross-Attention:** Adds diffusion model attention guidance  
- **+Both:** DINO + Cross-attention combined

## 5 Test Cases (Different Transformations)

| ID | Source | Target | Type | Difficulty |
|----|--------|--------|------|-----------|
| A1 | Hound | Hippo | Mammal→Mammal | Medium |
| A2 | Bottle | Vase | Similar topology | Easy |
| A3 | Doll | Human | Human-like→Human | Medium |
| A4 | Truck | Dragon | Vehicle→Creature | **Hard** |
| A5 | Chair | Sculpture | Furniture→Art | **Hard** |

## How to Run

### Option 1: Submit All at Once (Recommended)

```bash
chmod +x submit_ablation_study.sh
./submit_ablation_study.sh
```

This will:
1. Submit all 6 jobs to the HPC cluster
2. Print job IDs for tracking
3. Show monitoring commands

**Time:** ~6-8 hours total (runs in parallel)

### Option 2: Submit Individual Jobs

```bash
# Baseline (no losses)
sbatch jobs/ablation_A1_baseline.sh

# Add DINO loss
sbatch jobs/ablation_A1_dino.sh

# Easy case (bottle → vase)
sbatch jobs/ablation_A2_bottle_vase.sh

# Medium case (doll → human)
sbatch jobs/ablation_A3_doll_human.sh

# Hard case 1 (truck → dragon)
sbatch jobs/ablation_A4_truck_dragon.sh

# Hard case 2 (chair → sculpture)
sbatch jobs/ablation_A5_chair_sculpture.sh
```

## Monitor Progress

### Check job status
```bash
squeue --me
```

### View live logs
```bash
tail -f slurm_logs/ablation_A1_baseline_*.out
```

### Check for errors
```bash
cat slurm_logs/ablation_A1_baseline_*.err
```

## Collect Results

After all jobs complete:

```bash
python collect_ablation_results.py --output_dir ./ablation_results/
```

This generates:
1. **ablation_results.md** - Formatted comparison tables
2. **ablation_results.csv** - Spreadsheet-ready data
3. **ablation_comparison.png** - Bar/line charts

## Expected Results

### If Hypothesis is Correct:

```
Metric              Baseline  +DINO    +Both    Improvement
──────────────────────────────────────────────────────────
FID Score (A1)      85.2      42.1     35.4     -58.4% ✨
Training Time       25 min    28 min   33 min   +32% (acceptable)
Visual Quality      Poor      Good     Excellent
```

### Output Files

For each experiment, you'll get:

```
outputs/ablation_A1_baseline_hound_hippo/
├── mesh_final/                 # Final deformed mesh
├── colored_meshes/             # PCA colored renders
├── evaluation/
│   ├── mesh_renders/           # 8 viewpoint renders
│   ├── reference_images/       # 16 generated images
│   └── fid_results.json        # {"fid_score": XX.XX}
└── logs/main.log               # Training logs
```

## Interpreting Results

### FID Score Interpretation

| Range | Quality | Assessment |
|-------|---------|-----------|
| < 30 | Excellent | Perfect alignment |
| 30-60 | Good | Solid improvements |
| 60-100 | Moderate | Partial improvements |
| > 100 | Poor | Minimal improvements |

### Key Metrics to Track

1. **FID Score** (lower is better)
   - Quantifies similarity between mesh renders and text-generated references
   - Most important metric

2. **Training Time** (minutes)
   - Acceptable to increase by 20-30% if FID improves by 40%+
   - If overhead is too high, losses aren't worth it

3. **Visual Quality** (subjective)
   - Check rendered mesh images
   - Does deformation look realistic?
   - Is correspondence preserved (parts stay in right places)?

## Analysis Workflow

```
1. Submit jobs → ./submit_ablation_study.sh
   ↓
2. Wait 6-8 hours (monitor with squeue --me)
   ↓
3. Collect results → python collect_ablation_results.py
   ↓
4. Review tables, plots, and CSV
   ↓
5. Examine visual outputs for best cases
   ↓
6. Update SEMANTIC_CORRESPONDENCE_PRESENTATION.md with results
```

## Next Steps After Getting Results

### If Results are Strong (40%+ improvement):
- Document in presentation
- Prepare paper with ablation table
- Consider publishing on arXiv

### If Results are Weak:
- Investigate specific failing cases
- Try different loss weights (dino_weight, cross_attn_weight)
- Test other mesh pairs

### If Results are Mixed:
- Separate easy/medium/hard cases in report
- Show hardware requirements vs improvements
- Document trade-offs clearly

## File Locations

| Item | Path |
|------|------|
| Ablation study docs | [ABLATION_STUDY.md](ABLATION_STUDY.md) |
| Job scripts | `jobs/ablation_*.sh` |
| Result collector | `collect_ablation_results.py` |
| Submission script | `submit_ablation_study.sh` |
| Results directory | `ablation_results/` (auto-created) |

## Troubleshooting

### "Mesh not found" error
```
Error: Mesh not found: ./data/Omni6DPose/PAM/object_meshes/...
```
**Solution:** Verify mesh files exist
```bash
ls data/Omni6DPose/PAM/object_meshes/*/Aligned.obj | head -5
```

### Jobs queued but not running
```
squeue shows job in queue but not running
```
**Solution:** Wait for resources or check partition
```bash
sinfo -p dev_gpu_h100
```

### Out of memory during FID evaluation
```
cuda: out of memory during reference image generation
```
**Solution:** Reduce n_references in job scripts from 16 to 8

## Questions?

Review these files:
- [ABLATION_STUDY.md](ABLATION_STUDY.md) - Scientific design
- [FID_WORKFLOW.md](FID_WORKFLOW.md) - FID evaluation details
- [SEMANTIC_CORRESPONDENCE_PRESENTATION.md](SEMANTIC_CORRESPONDENCE_PRESENTATION.md) - Project context

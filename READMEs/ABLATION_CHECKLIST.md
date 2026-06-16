# Ablation Study Setup Checklist ✓

## Pre-Flight Check

- [x] **6 job scripts created** - All ready for submission
  ```bash
  ls -1 jobs/ablation_*.sh
  ```

- [x] **Result collector script** - Automated result analysis
  ```bash
  ls -1 collect_ablation_results.py
  ```

- [x] **Submission script** - One-command launch
  ```bash
  ls -1 submit_ablation_study.sh
  ./submit_ablation_study.sh --help 2>/dev/null || echo "(No help, just run it)"
  ```

- [x] **Documentation** - Complete guides
  - ABLATION_STUDY.md - Scientific methodology
  - ABLATION_QUICK_START.md - How to run
  - ABLATION_SUMMARY.md - File overview

---

## Execution Checklist

### Before Running
- [ ] You have HPC access and can run `sbatch`
- [ ] You're in the right directory: `/pfs/work9/workspace/.../meshup_v2/`
- [ ] GPU partition `dev_gpu_h100` is available: `sinfo -p dev_gpu_h100`

### Submit Experiments
- [ ] Run: `./submit_ablation_study.sh`
- [ ] You see 6 job IDs printed
- [ ] Example output:
  ```
  ✅ Job ID: 3456789
  ✅ Job ID: 3456790
  ...
  ```

### Monitor Progress
- [ ] Check status: `squeue --me`
- [ ] View logs: `tail -f slurm_logs/ablation_A1_baseline_*.out`
- [ ] All jobs eventually show in completed/failed state
- [ ] No jobs should error (check `.err` files if they do)

### Collect Results (After all jobs complete)
- [ ] Run: `python collect_ablation_results.py --output_dir ./ablation_results/`
- [ ] Files created:
  - [ ] `ablation_results/ablation_results.md` (formatted tables)
  - [ ] `ablation_results/ablation_results.csv` (spreadsheet)
  - [ ] `ablation_results/ablation_comparison.png` (plots)

---

## Expected File Structure After Run

```
outputs/
├── ablation_A1_baseline_hound_hippo/
│   ├── mesh_final/mesh.obj                    ✓
│   ├── evaluation/
│   │   ├── mesh_renders/view_*.png            ✓ (8 images)
│   │   ├── reference_images/ref_*.png         ✓ (16 images)
│   │   └── fid_results.json                   ✓
│   └── logs/main.log                          ✓
│
├── ablation_A1_dino_hound_hippo/
│   └── [same structure]                       ✓
│
├── ablation_A2_bottle_vase_both/
│   └── [same structure]                       ✓
│
├── ablation_A3_doll_human_both/
│   └── [same structure]                       ✓
│
├── ablation_A4_truck_dragon_both/
│   └── [same structure]                       ✓
│
└── ablation_A5_chair_sculpture_both/
    └── [same structure]                       ✓

ablation_results/
├── ablation_results.md                        ✓
├── ablation_results.csv                       ✓
└── ablation_comparison.png                    ✓
```

---

## Sample Expected Results

### Markdown Table Format
```markdown
## A1: a hippo

| Configuration | FID↓ | Training Time |
|---------------|------|---------------|
| Baseline (No loss) | 85.20 | 25.0 min |
| DINO Only | 42.10 | 28.0 min |

### Relative Improvements vs Baseline
- DINO Only: **-50.6%** (FID 85.20 → 42.10)
```

### CSV Format
```csv
case,configuration,fid,training_time_min,prompt
A1,Baseline (No loss),85.20,25.0,a hippo
A1,DINO Only,42.10,28.0,a hippo
A1,DINO + Cross-Attn,35.40,33.0,a hippo
```

---

## Troubleshooting

### Problem: "sbatch: command not found"
**Solution:** You're not on HPC. Run on cluster with:
```bash
ssh <your-hpc-cluster>
cd /pfs/work9/workspace/.../meshup_v2/
sbatch jobs/ablation_A1_baseline.sh
```

### Problem: "Mesh not found" error
**Solution:** Verify mesh exists:
```bash
ls data/Omni6DPose/PAM/object_meshes/*/Aligned.obj | wc -l
```
Should show > 50 meshes.

### Problem: Jobs fail with "CUDA out of memory"
**Solution:** Reduce batch size in job scripts:
```bash
# In job script, change:
# --n_references 16  →  --n_references 8
```

### Problem: Results collector finds no data
**Solution:** Wait for all jobs to complete:
```bash
squeue --me  # Should show no pending jobs
```

---

## Success Criteria

✅ All 6 jobs submit successfully  
✅ All jobs complete without errors  
✅ Each output dir has `evaluation/fid_results.json`  
✅ `collect_ablation_results.py` generates reports  
✅ `ablation_results.md` shows comparison table  
✅ `ablation_results.csv` is readable in Excel  
✅ `ablation_comparison.png` has plots  

---

## Next Actions

After successful completion:

1. **Review Results**
   ```bash
   cat ablation_results/ablation_results.md
   ```

2. **Analyze in Spreadsheet** (optional)
   ```bash
   # Copy to your machine and open in Excel/Google Sheets
   scp -r ablation_results/ your-local-machine:~/
   ```

3. **Update Presentation**
   - Add results to [SEMANTIC_CORRESPONDENCE_PRESENTATION.md](SEMANTIC_CORRESPONDENCE_PRESENTATION.md)
   - Section: "7. Key Insights & Lessons Learned → Ablation Study Results"

4. **Prepare Figure for Publication**
   - Use `ablation_comparison.png`
   - Or recreate with better styling if needed

---

## Time Estimate

| Phase | Duration |
|-------|----------|
| Submit all jobs | < 1 min |
| Wait for completion | 6-8 hours ⏰ |
| Collect results | < 1 min |
| **Total** | **6-8 hours** |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| [ABLATION_SUMMARY.md](ABLATION_SUMMARY.md) | This file - Overview |
| [ABLATION_QUICK_START.md](ABLATION_QUICK_START.md) | How to run |
| [ABLATION_STUDY.md](ABLATION_STUDY.md) | Scientific design |
| `submit_ablation_study.sh` | Launch all jobs |
| `collect_ablation_results.py` | Collect & analyze |
| `jobs/ablation_*.sh` | Individual job scripts |

---

## Questions?

- **How do I run experiments?** → [ABLATION_QUICK_START.md](ABLATION_QUICK_START.md)
- **What will be tested?** → [ABLATION_STUDY.md](ABLATION_STUDY.md)
- **What's the timeline?** → See "Time Estimate" above
- **How do I interpret results?** → Check "Interpreting Results" in QUICK_START

---

**Ready to start?**

```bash
./submit_ablation_study.sh
```

Then monitor with:
```bash
watch -n 30 'squeue --me && echo "---" && ls -lhd outputs/ablation_* | tail -3'
```

Good luck! 🚀

# Ablation Study: Systematic Evaluation of Semantic Correspondence Components

## Overview

This study isolates the contribution of each component:
1. **Baseline:** Original MeshUp (no semantic losses)
2. **+ DINO Loss:** Adds DINOv2 feature consistency
3. **+ Cross-Attention:** Adds diffusion attention guidance
4. **+ Both Losses:** DINO + Cross-attention combined

## Hypotheses

- **H1:** DINO loss reduces FID by 30-40% by preserving semantic features
- **H2:** Cross-attention adds 5-10% further improvement
- **H3:** Combined losses provide best results with minimal training overhead

## Test Cases

| ID | Source | Target | Transformation Type | Difficulty |
|----|--------|--------|-------------------|-----------|
| A1 | Hound | Hippo | Mammal → Mammal | Medium |
| A2 | Bottle | Vase | Similar topology | Easy |
| A3 | Doll | Human | Human-like → Human | Medium |
| A4 | Toy Truck | Dragon | Vehicle → Creature | Hard |
| A5 | Chair | Sculpture | Furniture → Art | Hard |

## Results Table Template

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    ABLATION STUDY RESULTS - TEST CASE A{N}                    ║
║                        {Source} → {Target}                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Configuration               FID↓    LPIPS↓   Training Time  Notes
────────────────────────────────────────────────────────────────────────────
Baseline (no loss)          [TBD]   [TBD]    [TBD] min      Original MeshUp
+ DINO Loss                 [TBD]   [TBD]    [TBD] min      
+ Cross-Attention           [TBD]   [TBD]    [TBD] min      
+ Both Losses               [TBD]   [TBD]    [TBD] min      BEST

Relative Improvements:
- DINO improvement:         [TBD]% vs Baseline
- Cross-Attn improvement:  [TBD]% vs Baseline
- Combined improvement:    [TBD]% vs Baseline

Key Observations:
- [Your observations about correspondence quality]
- [Any mesh-specific challenges]
- [Visual quality assessment]
```

## Implementation Notes

- Each configuration runs on **same random seed** for fair comparison
- **Resolution:** 512×512
- **Epochs:** 3000 (consistent across all)
- **Regularization:** 35000 (consistent across all)
- **Batch size:** 8 (consistent across all)

---

## Data Collection

See `collect_ablation_results.py` for automated result collection.

```bash
# Run after all experiments complete
python collect_ablation_results.py --output_dir ./ablation_results/
```

This will generate:
- `ablation_summary.csv` - Spreadsheet format
- `ablation_comparison.md` - Markdown report
- `ablation_plots.png` - Visualization charts

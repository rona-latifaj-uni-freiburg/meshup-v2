#!/usr/bin/env python3
"""
Ablation Study Results Collector

Automatically collects FID and timing results from all ablation experiments
and generates comparison tables and visualizations.

Usage:
    python collect_ablation_results.py --output_dir ./ablation_results/
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re
import subprocess

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/pandas/numpy not available, skipping plots")


class AblationResultsCollector:
    def __init__(self, outputs_root="./outputs"):
        self.outputs_root = Path(outputs_root)
        self.results = {}
        
    def collect_all(self) -> Dict:
        """Collect results from all ablation experiments."""
        
        # Pattern to identify ablation directories
        ablation_dirs = sorted(self.outputs_root.glob("ablation_*"))
        
        for dir_path in ablation_dirs:
            results = self.parse_ablation_dir(dir_path)
            if results:
                self.results[dir_path.name] = results
        
        return self.results
    
    def parse_ablation_dir(self, dir_path: Path) -> Dict:
        """Extract results from a single ablation experiment directory."""
        
        # Check if FID results exist
        fid_json = dir_path / "evaluation" / "fid_results.json"
        if not fid_json.exists():
            print(f"⚠️  No FID results in {dir_path.name} (still running?)")
            return None
        
        try:
            with open(fid_json, 'r') as f:
                fid_data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {fid_json}: {e}")
            return None
        
        # Extract timing from logs if available
        log_file = dir_path / "logs" / "main.log"
        training_time = self.extract_time_from_log(log_file) if log_file.exists() else None
        
        # Determine configuration type
        config_type = self.get_config_type(dir_path.name)
        
        return {
            'name': dir_path.name,
            'config': config_type,
            'fid': fid_data.get('fid_score'),
            'n_references': fid_data.get('n_references', 16),
            'text_prompt': fid_data.get('text_prompt', ''),
            'training_time': training_time,
            'mesh_path': fid_data.get('mesh_path', '')
        }
    
    def get_config_type(self, dirname: str) -> str:
        """Determine the configuration type from directory name."""
        if 'baseline' in dirname:
            return 'Baseline (No loss)'
        elif 'both' in dirname:
            return 'DINO + Cross-Attn'
        elif 'dino' in dirname:
            return 'DINO Only'
        elif 'crossattn' in dirname:
            return 'Cross-Attn Only'
        else:
            return 'Unknown'
    
    def extract_time_from_log(self, log_path: Path) -> float:
        """Extract training time from log file (in minutes)."""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                # Look for timing information
                match = re.search(r'Finished training in (\d+\.?\d*) seconds', content)
                if match:
                    seconds = float(match.group(1))
                    return round(seconds / 60, 1)
        except Exception as e:
            pass
        return None
    
    def group_by_case(self) -> Dict[str, List]:
        """Group results by ablation case (e.g., A1, A2, etc.)."""
        grouped = {}
        
        for exp_name, results in self.results.items():
            # Extract case ID (A1, A2, etc.)
            match = re.search(r'(A\d+)', exp_name)
            if match:
                case_id = match.group(1)
                if case_id not in grouped:
                    grouped[case_id] = []
                grouped[case_id].append(results)
        
        return grouped
    
    def generate_comparison_tables(self, output_dir: Path) -> None:
        """Generate comparison tables for each case."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        grouped = self.group_by_case()
        
        markdown_content = "# Ablation Study Results\n\n"
        
        for case_id in sorted(grouped.keys()):
            experiments = grouped[case_id]
            
            # Get metadata from first result
            first = experiments[0]
            case_name = f"{case_id}: {first['text_prompt']}"
            
            markdown_content += f"## {case_name}\n\n"
            
            # Create table
            table_lines = [
                "| Configuration | FID↓ | Training Time |",
                "|---------------|------|---------------|"
            ]
            
            for exp in sorted(experiments, key=lambda x: x.get('fid') or float('inf')):
                config = exp['config']
                fid = f"{exp['fid']:.2f}" if exp['fid'] else "N/A"
                time_str = f"{exp['training_time']:.1f} min" if exp['training_time'] else "N/A"
                table_lines.append(f"| {config} | {fid} | {time_str} |")
            
            markdown_content += "\n".join(table_lines) + "\n\n"
            
            # Calculate improvements
            baseline = next((e for e in experiments if 'Baseline' in e['config']), None)
            if baseline and baseline['fid']:
                markdown_content += "### Relative Improvements vs Baseline\n"
                baseline_fid = baseline['fid']
                for exp in experiments:
                    if 'Baseline' not in exp['config'] and exp['fid']:
                        improvement = (baseline_fid - exp['fid']) / baseline_fid * 100
                        markdown_content += f"- {exp['config']}: **{improvement:+.1f}%** (FID {baseline_fid:.2f} → {exp['fid']:.2f})\n"
                markdown_content += "\n"
        
        # Write markdown
        md_path = output_dir / "ablation_results.md"
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        print(f"✅ Markdown report: {md_path}")
        
        # Write CSV
        if HAS_PLOTTING:
            self.write_csv(output_dir, grouped)
    
    def write_csv(self, output_dir: Path, grouped: Dict) -> None:
        """Write results to CSV file."""
        rows = []
        for case_id in sorted(grouped.keys()):
            for exp in grouped[case_id]:
                rows.append({
                    'case': case_id,
                    'configuration': exp['config'],
                    'fid': exp['fid'],
                    'training_time_min': exp['training_time'],
                    'prompt': exp['text_prompt'],
                })
        
        df = pd.DataFrame(rows)
        csv_path = output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV export: {csv_path}")
    
    def generate_plots(self, output_dir: Path) -> None:
        """Generate visualization plots if matplotlib available."""
        
        if not HAS_PLOTTING:
            return
        
        grouped = self.group_by_case()
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # FID comparison
        ax_fid = axes[0]
        for case_id in sorted(grouped.keys()):
            experiments = grouped[case_id]
            configs = [e['config'] for e in experiments]
            fids = [e['fid'] for e in experiments if e['fid']]
            
            if fids:
                ax_fid.plot(range(len(configs)), fids, marker='o', label=case_id)
        
        ax_fid.set_ylabel('FID Score (lower is better)')
        ax_fid.set_title('FID Improvements Across Cases')
        ax_fid.legend()
        ax_fid.grid(True, alpha=0.3)
        
        # Time comparison
        ax_time = axes[1]
        for case_id in sorted(grouped.keys()):
            experiments = grouped[case_id]
            configs = [e['config'] for e in experiments]
            times = [e['training_time'] for e in experiments if e['training_time']]
            
            if times:
                ax_time.plot(range(len(configs)), times, marker='s', label=case_id)
        
        ax_time.set_ylabel('Training Time (minutes)')
        ax_time.set_title('Training Time Overhead')
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = output_dir / "ablation_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot generated: {plot_path}")
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect and analyze ablation study results')
    parser.add_argument('--output_dir', type=str, default='./ablation_results',
                        help='Output directory for results')
    parser.add_argument('--outputs_root', type=str, default='./outputs',
                        help='Root directory containing ablation experiments')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    outputs_root = Path(args.outputs_root)
    
    print("=" * 60)
    print("ABLATION STUDY RESULTS COLLECTOR")
    print("=" * 60)
    
    collector = AblationResultsCollector(outputs_root=outputs_root)
    results = collector.collect_all()
    
    if not results:
        print("❌ No ablation results found!")
        return 1
    
    print(f"\n✅ Found {len(results)} completed experiments")
    print("\nResults summary:")
    for exp_name, data in sorted(results.items()):
        if data:
            print(f"  - {exp_name}: FID={data['fid']:.2f}")
    
    # Generate reports
    print("\nGenerating reports...")
    collector.generate_comparison_tables(output_dir)
    collector.generate_plots(output_dir)
    
    print("\n" + "=" * 60)
    print("📊 Results saved to:", output_dir)
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Compare benchmark results between two folders and generate a performance comparison report.
Usage: python3 compare_results.py <folder1> <folder2> [--output report.md]
"""

import json
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkComparison:
    """Results of comparing two benchmark runs."""
    name: str
    baseline_time: float
    comparison_time: float
    speedup: float
    percentage_change: float
    winner: str  # 'baseline', 'comparison', or 'tie'
    
    @property
    def is_faster(self) -> bool:
        return self.speedup > 1.0
    
    @property
    def is_slower(self) -> bool:
        return self.speedup < 1.0


class ResultsComparator:
    """Compare benchmark results from two different result folders."""
    
    def __init__(self, baseline_folder: str, comparison_folder: str):
        self.baseline_folder = Path(baseline_folder)
        self.comparison_folder = Path(comparison_folder)
        self.comparisons: List[BenchmarkComparison] = []
        
    def load_results_file(self, file_path: Path) -> Optional[Dict]:
        """Load results from a JSON file."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not load {file_path}: {e}")
            return None
    
    def extract_reversan_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract timing metrics from Reversan benchmark results."""
        metrics = {}
        
        # Process depth runs
        if 'runs_depth' in results:
            for run in results['runs_depth']:
                depth = run['depth']
                if 'average_metrics' in run:
                    time = run['average_metrics']['elapsed_seconds']
                else:
                    time = run['metrics']['elapsed_seconds']
                metrics[f'depth_{depth}'] = time
        
        # Process thread runs
        if 'runs_threads' in results:
            for run in results['runs_threads']:
                threads = run['threads']
                if 'average_metrics' in run:
                    time = run['average_metrics']['user_seconds']
                else:
                    time = run['metrics']['user_seconds']
                metrics[f'threads_{threads}'] = time
        
        return metrics
    
    def extract_llama_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract timing metrics from Llama benchmark results."""
        metrics = {}
        
        # Process CPU runs
        if 'runs_cpu' in results:
            for run in results['runs_cpu']:
                config = f"cpu_p{run['prompt_size']}_g{run['generation_size']}"
                metrics[config] = run['elapsed_seconds']
                
                # Add tokens per second if available
                if 'metrics' in run and 'tokens_per_second' in run['metrics']:
                    metrics[f"{config}_tps"] = run['metrics']['tokens_per_second']
        
        # Process GPU runs
        if 'runs_gpu' in results:
            for run in results['runs_gpu']:
                config = f"gpu_p{run['prompt_size']}_g{run['generation_size']}"
                metrics[config] = run['elapsed_seconds']
                
                # Add tokens per second if available
                if 'metrics' in run and 'tokens_per_second' in run['metrics']:
                    metrics[f"{config}_tps"] = run['metrics']['tokens_per_second']
        
        # Build times
        if 'build' in results:
            build = results['build']
            if 'cpu_build_timing' in build:
                metrics['build_cpu_total'] = build['cpu_build_timing'].get('total_time_seconds', 0)
            if 'vulkan_build_timing' in build:
                metrics['build_vulkan_total'] = build['vulkan_build_timing'].get('total_time_seconds', 0)
        
        return metrics
    
    def extract_blender_metrics(self, results: Dict) -> Dict[str, float]:
        """Extract performance metrics from Blender benchmark results."""
        metrics = {}
        
        # Process CPU runs
        if 'runs_cpu' in results:
            for run in results['runs_cpu']:
                if run.get('success') and 'scenes' in run:
                    metrics['cpu_total_score'] = run.get('total_score', 0)
                    for scene_name, scene_data in run['scenes'].items():
                        metrics[f'cpu_{scene_name}'] = scene_data.get('samples_per_minute', 0)
        
        # Process GPU runs
        if 'runs_gpu' in results:
            for run in results['runs_gpu']:
                if run.get('success') and 'scenes' in run:
                    metrics['gpu_total_score'] = run.get('total_score', 0)
                    for scene_name, scene_data in run['scenes'].items():
                        metrics[f'gpu_{scene_name}'] = scene_data.get('samples_per_minute', 0)
        
        return metrics
    
    def compare_single_benchmark(self, benchmark_name: str, 
                                baseline_results: Dict, 
                                comparison_results: Dict) -> List[BenchmarkComparison]:
        """Compare a single benchmark type between baseline and comparison."""
        comparisons = []
        
        if benchmark_name == 'reversan':
            baseline_metrics = self.extract_reversan_metrics(baseline_results)
            comparison_metrics = self.extract_reversan_metrics(comparison_results)
        elif benchmark_name == 'llama':
            baseline_metrics = self.extract_llama_metrics(baseline_results)
            comparison_metrics = self.extract_llama_metrics(comparison_results)
        elif benchmark_name == 'blender':
            baseline_metrics = self.extract_blender_metrics(baseline_results)
            comparison_metrics = self.extract_blender_metrics(comparison_results)
        else:
            print(f"⚠️  Unknown benchmark type: {benchmark_name}")
            return comparisons
        
        # Compare common metrics
        for metric_name in baseline_metrics:
            if metric_name in comparison_metrics:
                baseline_value = baseline_metrics[metric_name]
                comparison_value = comparison_metrics[metric_name]
                
                # For tokens per second, higher is better; for time, lower is better
                if metric_name.endswith('_tps'):
                    # Higher TPS is better
                    speedup = comparison_value / baseline_value if baseline_value > 0 else 0
                    percentage_change = ((comparison_value - baseline_value) / baseline_value * 100) if baseline_value > 0 else 0
                    winner = 'comparison' if comparison_value > baseline_value else ('baseline' if baseline_value > comparison_value else 'tie')
                else:
                    # Lower time is better
                    speedup = baseline_value / comparison_value if comparison_value > 0 else 0
                    percentage_change = ((baseline_value - comparison_value) / baseline_value * 100) if baseline_value > 0 else 0
                    winner = 'comparison' if comparison_value < baseline_value else ('baseline' if baseline_value < comparison_value else 'tie')
                
                comparison = BenchmarkComparison(
                    name=f"{benchmark_name}_{metric_name}",
                    baseline_time=baseline_value,
                    comparison_time=comparison_value,
                    speedup=speedup,
                    percentage_change=percentage_change,
                    winner=winner
                )
                comparisons.append(comparison)
        
        return comparisons
    
    def compare_results(self) -> None:
        """Compare all benchmark results between the two folders."""
        print(f"🔍 Comparing results between:")
        print(f"   Baseline: {self.baseline_folder}")
        print(f"   Comparison: {self.comparison_folder}")
        
        # Find all result files
        result_files = ['reversan_results.json', 'llama_results.json', 'blender_results.json']
        
        for result_file in result_files:
            baseline_path = self.baseline_folder / result_file
            comparison_path = self.comparison_folder / result_file
            
            baseline_results = self.load_results_file(baseline_path)
            comparison_results = self.load_results_file(comparison_path)
            
            if baseline_results and comparison_results:
                benchmark_name = result_file.replace('_results.json', '')
                print(f"📊 Comparing {benchmark_name} benchmark...")
                
                benchmark_comparisons = self.compare_single_benchmark(
                    benchmark_name, baseline_results, comparison_results
                )
                self.comparisons.extend(benchmark_comparisons)
            elif baseline_results or comparison_results:
                print(f"⚠️  {result_file}: Only found in one folder, skipping comparison")
            else:
                print(f"ℹ️  {result_file}: Not found in either folder")
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate a markdown report of the comparison results."""
        if not self.comparisons:
            return "No benchmark comparisons found."
        
        # Sort comparisons by speedup (descending)
        sorted_comparisons = sorted(self.comparisons, key=lambda x: x.speedup, reverse=True)
        
        report_lines = [
            "# Benchmark Performance Comparison Report",
            "",
            f"**Baseline**: `{self.baseline_folder}`  ",
            f"**Comparison**: `{self.comparison_folder}`  ",
            f"**Total Comparisons**: {len(self.comparisons)}",
            "",
        ]
        
        # Summary statistics
        faster_count = sum(1 for c in self.comparisons if c.winner == 'comparison')
        slower_count = sum(1 for c in self.comparisons if c.winner == 'baseline')
        tied_count = sum(1 for c in self.comparisons if c.winner == 'tie')
        
        report_lines.extend([
            "## Summary",
            "",
            f"- **Comparison Faster**: {faster_count} tests ({faster_count/len(self.comparisons)*100:.1f}%)",
            f"- **Baseline Faster**: {slower_count} tests ({slower_count/len(self.comparisons)*100:.1f}%)",
            f"- **Tied**: {tied_count} tests ({tied_count/len(self.comparisons)*100:.1f}%)",
            "",
        ])
        
        # Best improvements
        best_improvements = [c for c in sorted_comparisons if c.winner == 'comparison'][:5]
        if best_improvements:
            report_lines.extend([
                "## 🚀 Top Performance Improvements",
                "",
                "| Test | Baseline | Comparison | Speedup | Improvement |",
                "|------|----------|------------|---------|-------------|",
            ])
            
            for comp in best_improvements:
                if comp.name.endswith('_tps'):
                    unit = " tps"
                    baseline_str = f"{comp.baseline_time:.2f}{unit}"
                    comparison_str = f"{comp.comparison_time:.2f}{unit}"
                else:
                    unit = "s"
                    baseline_str = f"{comp.baseline_time:.3f}{unit}"
                    comparison_str = f"{comp.comparison_time:.3f}{unit}"
                
                report_lines.append(
                    f"| {comp.name} | {baseline_str} | {comparison_str} | "
                    f"{comp.speedup:.2f}x | {comp.percentage_change:+.1f}% |"
                )
            
            report_lines.append("")
        
        # Worst regressions
        worst_regressions = [c for c in sorted_comparisons if c.winner == 'baseline'][-5:]
        if worst_regressions:
            report_lines.extend([
                "## 📉 Performance Regressions",
                "",
                "| Test | Baseline | Comparison | Slowdown | Regression |",
                "|------|----------|------------|----------|------------|",
            ])
            
            for comp in worst_regressions:
                if comp.name.endswith('_tps'):
                    unit = " tps"
                    baseline_str = f"{comp.baseline_time:.2f}{unit}"
                    comparison_str = f"{comp.comparison_time:.2f}{unit}"
                else:
                    unit = "s"
                    baseline_str = f"{comp.baseline_time:.3f}{unit}"
                    comparison_str = f"{comp.comparison_time:.3f}{unit}"
                
                slowdown = 1/comp.speedup if comp.speedup > 0 else float('inf')
                report_lines.append(
                    f"| {comp.name} | {baseline_str} | {comparison_str} | "
                    f"{slowdown:.2f}x | {-comp.percentage_change:+.1f}% |"
                )
            
            report_lines.append("")
        
        # Detailed results
        report_lines.extend([
            "## 📊 Detailed Results",
            "",
            "| Test | Baseline | Comparison | Speedup | Change | Winner |",
            "|------|----------|------------|---------|--------|--------|",
        ])
        
        for comp in sorted_comparisons:
            if comp.name.endswith('_tps'):
                unit = " tps"
                baseline_str = f"{comp.baseline_time:.2f}{unit}"
                comparison_str = f"{comp.comparison_time:.2f}{unit}"
            else:
                unit = "s"
                baseline_str = f"{comp.baseline_time:.3f}{unit}"
                comparison_str = f"{comp.comparison_time:.3f}{unit}"
            
            winner_emoji = {
                'comparison': '🥇 Comparison',
                'baseline': '🥈 Baseline', 
                'tie': '🤝 Tie'
            }[comp.winner]
            
            report_lines.append(
                f"| {comp.name} | {baseline_str} | {comparison_str} | "
                f"{comp.speedup:.2f}x | {comp.percentage_change:+.1f}% | {winner_emoji} |"
            )
        
        report_lines.extend([
            "",
            "---",
            f"*Generated by weird-bench comparison tool*"
        ])
        
        report_content = '\n'.join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"📄 Report saved to: {output_file}")
        
        return report_content
    
    def print_summary(self) -> None:
        """Print a quick summary to console."""
        if not self.comparisons:
            print("❌ No benchmark comparisons found.")
            return
        
        faster_count = sum(1 for c in self.comparisons if c.winner == 'comparison')
        slower_count = sum(1 for c in self.comparisons if c.winner == 'baseline')
        tied_count = sum(1 for c in self.comparisons if c.winner == 'tie')
        
        print(f"\n📊 Comparison Summary:")
        print(f"   Comparison faster: {faster_count}/{len(self.comparisons)} tests ({faster_count/len(self.comparisons)*100:.1f}%)")
        print(f"   Baseline faster:   {slower_count}/{len(self.comparisons)} tests ({slower_count/len(self.comparisons)*100:.1f}%)")
        print(f"   Tied:              {tied_count}/{len(self.comparisons)} tests ({tied_count/len(self.comparisons)*100:.1f}%)")
        
        # Show top improvements
        best = max(self.comparisons, key=lambda x: x.speedup if x.winner == 'comparison' else 0)
        if best.winner == 'comparison':
            print(f"   Best improvement:  {best.name} - {best.speedup:.2f}x faster ({best.percentage_change:+.1f}%)")
        
        # Show worst regression
        worst = min(self.comparisons, key=lambda x: x.speedup if x.winner == 'baseline' else float('inf'))
        if worst.winner == 'baseline' and worst.speedup > 0:
            print(f"   Worst regression:  {worst.name} - {1/worst.speedup:.2f}x slower ({-worst.percentage_change:+.1f}%)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare benchmark results between two folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compare_results.py results1/ results2/
  python3 compare_results.py old_results/ new_results/ --output comparison_report.md
  python3 compare_results.py baseline/ optimized/ --quiet
        """
    )
    
    parser.add_argument('baseline_folder', 
                       help='Folder containing baseline benchmark results')
    parser.add_argument('comparison_folder',
                       help='Folder containing comparison benchmark results')
    parser.add_argument('--output', '-o',
                       help='Output file for detailed report (markdown format)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only print summary, skip detailed console output')
    
    args = parser.parse_args()
    
    # Validate folders exist
    if not os.path.exists(args.baseline_folder):
        print(f"❌ Baseline folder not found: {args.baseline_folder}")
        sys.exit(1)
    
    if not os.path.exists(args.comparison_folder):
        print(f"❌ Comparison folder not found: {args.comparison_folder}")
        sys.exit(1)
    
    # Create comparator and run comparison
    comparator = ResultsComparator(args.baseline_folder, args.comparison_folder)
    comparator.compare_results()
    
    # Generate and save report
    if args.output:
        comparator.generate_report(args.output)
    
    # Print summary unless quiet
    if not args.quiet:
        comparator.print_summary()
        
        if not args.output:
            print("\n📄 Use --output to save detailed report to file")


if __name__ == "__main__":
    main()
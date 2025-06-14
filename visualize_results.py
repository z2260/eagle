# =============================================================
#  visualize_results.py  --  Visualization for EAGLE-3 Results
# =============================================================

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List

def load_results(file_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_speedup_comparison(results: List[Dict], save_path: str = None):
    """Plot speedup comparison across different models/datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data
    datasets = [r['args']['dataset'] for r in results]
    speedups = [r['metrics']['speedup_ratio'] for r in results]
    accept_lens = [r['metrics']['avg_accept_len'] for r in results]
    
    # Speedup bar chart
    ax1.bar(datasets, speedups, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Speedup Ratio')
    ax1.set_title('EAGLE-3 Speedup Across Datasets')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(speedups):
        ax1.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom')
    
    # Accept length bar chart
    ax2.bar(datasets, accept_lens, color='lightgreen', edgecolor='darkgreen')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Average Accept Length (τ)')
    ax2.set_title('Average Accept Length Across Datasets')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(accept_lens):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_accept_rates(results: Dict, save_path: str = None):
    """Plot n-α accept rates."""
    accept_rates = results['metrics']['accept_rates']
    
    plt.figure(figsize=(10, 6))
    
    n_values = list(accept_rates.keys())
    rates = list(accept_rates.values())
    
    plt.plot(n_values, rates, 'o-', linewidth=2, markersize=8, label='EAGLE-3')
    
    plt.xlabel('n (number of self-predicted values)')
    plt.ylabel('Accept Rate (n-α)')
    plt.title('EAGLE-3 Accept Rates by Input Composition')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add percentage labels
    for n, rate in zip(n_values, rates):
        plt.text(n, rate + 0.01, f'{rate:.1%}', ha='center', va='bottom')
    
    plt.ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(results: Dict, save_path: str = None):
    """Plot various performance metrics."""
    metrics = results['metrics']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Token statistics pie chart
    token_data = [
        metrics['total_accepted_tokens'],
        metrics['total_draft_tokens'] - metrics['total_accepted_tokens']
    ]
    labels = ['Accepted', 'Rejected']
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.pie(token_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Draft Token Accept/Reject Ratio')
    
    # Latency comparison
    baseline_latency = 100  # Normalized baseline
    eagle_latency = metrics['latency_ms']
    
    latencies = [baseline_latency, eagle_latency]
    models = ['Baseline', 'EAGLE-3']
    
    bars = ax2.bar(models, latencies, color=['gray', 'orange'])
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Comparison (Normalized)')
    
    # Add speedup annotation
    speedup = baseline_latency / eagle_latency
    ax2.text(0.5, max(latencies) * 1.1, f'{speedup:.2f}x faster', 
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Throughput
    ax3.bar(['EAGLE-3'], [metrics['tokens_per_second']], color='purple')
    ax3.set_ylabel('Tokens per Second')
    ax3.set_title('Generation Throughput')
    ax3.grid(axis='y', alpha=0.3)
    
    # Summary metrics
    summary_text = f"""
    Total Samples: {len(results.get('results', []))}
    Speedup Ratio: {metrics['speedup_ratio']:.2f}x
    Avg Accept Length: {metrics['avg_accept_len']:.2f}
    Accept Rate: {metrics['total_accepted_tokens']/metrics['total_draft_tokens']:.1%}
    Tokens/Second: {metrics['tokens_per_second']:.1f}
    """
    
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', 
             fontsize=12, transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.axis('off')
    ax4.set_title('Summary Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_example_comparisons(results: Dict, num_examples: int = 3, save_path: str = None):
    """Plot example outputs comparison."""
    examples = results.get('results', [])[:num_examples]
    
    if not examples:
        print("No example results found")
        return
    
    fig, axes = plt.subplots(num_examples, 1, figsize=(12, 4*num_examples))
    
    if num_examples == 1:
        axes = [axes]
    
    for i, (ax, example) in enumerate(zip(axes, examples)):
        # Create text comparison
        prompt_len = len(example['prompt'])
        baseline_len = len(example['baseline_output'])
        eagle_len = len(example['eagle_output'])
        
        # Simple bar comparison of generation times
        times = [example['baseline_time'], example['eagle_time']]
        models = ['Baseline', 'EAGLE-3']
        
        bars = ax.bar(models, times, color=['#3498db', '#e74c3c'])
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Example {i+1}: Generation Time Comparison')
        
        # Add speedup annotation
        speedup = example.get('speedup', 0)
        ax.text(0.5, max(times) * 1.1, f'{speedup:.2f}x speedup', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add character count info
        info_text = f"Prompt: {prompt_len} chars | Output: ~{baseline_len} chars"
        ax.text(0.5, -0.15, info_text, ha='center', va='top', 
                transform=ax.transAxes, fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize EAGLE-3 evaluation results')
    parser.add_argument('--results_file', required=True, help='Path to evaluation results JSON')
    parser.add_argument('--output_dir', default='./visualizations', help='Directory to save plots')
    parser.add_argument('--compare_files', nargs='*', help='Additional result files for comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    results = load_results(args.results_file)
    
    # Single file visualizations
    print("Generating accept rates plot...")
    plot_accept_rates(results, output_dir / 'accept_rates.png')
    
    print("Generating performance metrics plot...")
    plot_performance_metrics(results, output_dir / 'performance_metrics.png')
    
    print("Generating example comparisons...")
    plot_example_comparisons(results, num_examples=3, save_path=output_dir / 'examples.png')
    
    # Multi-file comparison if provided
    if args.compare_files:
        all_results = [results]
        for f in args.compare_files:
            all_results.append(load_results(f))
        
        print("Generating speedup comparison...")
        plot_speedup_comparison(all_results, output_dir / 'speedup_comparison.png')
    
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
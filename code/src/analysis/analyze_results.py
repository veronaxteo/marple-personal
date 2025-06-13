"""
Analysis and visualization of hyperparameter optimization results.

Provides tools to:
1. Analyze Pareto optimal solutions
2. Visualize parameter sensitivity
3. Generate summary reports
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import argparse


class OptimizationAnalyzer:
    """Analyzer for hyperparameter optimization results"""
    
    def __init__(self, results_csv: str, output_dir: str = None):
        """
        Initialize analyzer.
        
        Args:
            results_csv: Path to optimization results CSV
            output_dir: Directory for output plots and reports
        """
        self.results_df = pd.read_csv(results_csv)
        self.output_dir = output_dir or os.path.dirname(results_csv)
        
        # Parameter names for optimization
        self.param_names = [
            'naive_temp', 'sophisticated_temp', 'cost_weight',
            'naive_detective_sigma', 'sophisticated_detective_sigma',
            'crumb_planting_sigma', 'visual_naive_likelihood_alpha',
            'visual_sophisticated_likelihood_alpha'
        ]
        
        # Metric names
        self.metric_names = [
            'detective_rmse', 'detective_correlation', 'detective_aic', 'suspect_emd'
        ]
        
        print(f"Loaded {len(self.results_df)} optimization results")
    
    def find_pareto_optimal(self) -> pd.DataFrame:
        """Find Pareto optimal solutions"""
        # For minimization problems, a solution is Pareto optimal if no other solution
        # dominates it (i.e., is better in all objectives)
        
        def is_dominated(row, other_rows):
            """Check if a row is dominated by any other row"""
            for _, other in other_rows.iterrows():
                if (other[self.metric_names] <= row[self.metric_names]).all() and \
                   (other[self.metric_names] < row[self.metric_names]).any():
                    return True
            return False
        
        pareto_optimal = []
        for idx, row in self.results_df.iterrows():
            other_rows = self.results_df.drop(idx)
            if not is_dominated(row[self.metric_names], other_rows[self.metric_names]):
                pareto_optimal.append(idx)
        
        pareto_df = self.results_df.loc[pareto_optimal].copy()
        print(f"Found {len(pareto_df)} Pareto optimal solutions")
        
        return pareto_df
    
    def plot_pareto_front(self, save_plots: bool = True):
        """Plot Pareto front for all objective pairs"""
        pareto_df = self.find_pareto_optimal()
        
        # Create pairwise plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metric_pairs = [
            ('detective_rmse', 'detective_correlation'),
            ('detective_rmse', 'detective_aic'),
            ('detective_rmse', 'suspect_emd'),
            ('detective_correlation', 'detective_aic'),
            ('detective_correlation', 'suspect_emd'),
            ('detective_aic', 'suspect_emd')
        ]
        
        for i, (x_metric, y_metric) in enumerate(metric_pairs):
            ax = axes[i]
            
            # Plot all points
            ax.scatter(self.results_df[x_metric], self.results_df[y_metric], 
                      alpha=0.5, c='lightblue', label='All solutions')
            
            # Plot Pareto optimal points
            ax.scatter(pareto_df[x_metric], pareto_df[y_metric], 
                      c='red', s=50, label='Pareto optimal')
            
            ax.set_xlabel(x_metric.replace('_', ' ').title())
            ax.set_ylabel(y_metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'pareto_front.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Pareto front plot saved to {plot_path}")
        
        plt.show()
    
    def plot_parameter_sensitivity(self, save_plots: bool = True):
        """Plot parameter sensitivity analysis"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(self.param_names):
            ax = axes[i]
            
            # Create a composite score (weighted sum of normalized metrics)
            normalized_df = self.results_df[self.metric_names].copy()
            for metric in self.metric_names:
                normalized_df[metric] = (normalized_df[metric] - normalized_df[metric].min()) / \
                                       (normalized_df[metric].max() - normalized_df[metric].min())
            
            # Equal weighting for composite score
            composite_score = normalized_df.mean(axis=1)
            
            # Plot parameter vs composite score
            ax.scatter(self.results_df[param], composite_score, alpha=0.6)
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Composite Score (lower is better)')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(self.results_df[param], composite_score, 1)
            p = np.poly1d(z)
            ax.plot(self.results_df[param], p(self.results_df[param]), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'parameter_sensitivity.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Parameter sensitivity plot saved to {plot_path}")
        
        plt.show()
    
    def plot_parameter_distributions(self, save_plots: bool = True):
        """Plot parameter distributions for Pareto optimal solutions"""
        pareto_df = self.find_pareto_optimal()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(self.param_names):
            ax = axes[i]
            
            # Plot distributions
            ax.hist(self.results_df[param], bins=20, alpha=0.5, label='All solutions', density=True)
            ax.hist(pareto_df[param], bins=10, alpha=0.7, label='Pareto optimal', density=True)
            
            ax.set_xlabel(param.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = os.path.join(self.output_dir, 'parameter_distributions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Parameter distributions plot saved to {plot_path}")
        
        plt.show()
    
    def generate_summary_report(self, save_report: bool = True) -> str:
        """Generate a summary report of optimization results"""
        pareto_df = self.find_pareto_optimal()
        
        report = []
        report.append("HYPERPARAMETER OPTIMIZATION SUMMARY REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall statistics
        report.append("OVERALL STATISTICS")
        report.append("-" * 20)
        report.append(f"Total optimization trials: {len(self.results_df)}")
        report.append(f"Pareto optimal solutions: {len(pareto_df)}")
        report.append(f"Success rate: {len(pareto_df)/len(self.results_df)*100:.1f}%")
        report.append("")
        
        # Best solutions for each metric
        report.append("BEST SOLUTIONS BY METRIC")
        report.append("-" * 25)
        for metric in self.metric_names:
            best_idx = self.results_df[metric].idxmin()
            best_value = self.results_df.loc[best_idx, metric]
            report.append(f"Best {metric}: {best_value:.4f} (Trial {best_idx})")
        report.append("")
        
        # Parameter ranges for Pareto optimal solutions
        report.append("PARETO OPTIMAL PARAMETER RANGES")
        report.append("-" * 35)
        for param in self.param_names:
            min_val = pareto_df[param].min()
            max_val = pareto_df[param].max()
            mean_val = pareto_df[param].mean()
            std_val = pareto_df[param].std()
            report.append(f"{param}:")
            report.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
            report.append(f"  Mean ± Std: {mean_val:.4f} ± {std_val:.4f}")
        report.append("")
        
        # Top 5 Pareto optimal solutions
        report.append("TOP 5 PARETO OPTIMAL SOLUTIONS")
        report.append("-" * 32)
        # Sort by composite score
        normalized_df = pareto_df[self.metric_names].copy()
        for metric in self.metric_names:
            normalized_df[metric] = (normalized_df[metric] - self.results_df[metric].min()) / \
                                   (self.results_df[metric].max() - self.results_df[metric].min())
        composite_score = normalized_df.mean(axis=1)
        top_5 = pareto_df.loc[composite_score.nsmallest(5).index]
        
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            report.append(f"#{i} (Trial {row['trial_number']}):")
            for metric in self.metric_names:
                report.append(f"  {metric}: {row[metric]:.4f}")
            report.append("  Parameters:")
            for param in self.param_names:
                report.append(f"    {param}: {row[param]:.4f}")
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_report:
            report_path = os.path.join(self.output_dir, 'optimization_summary.txt')
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"Summary report saved to {report_path}")
        
        return report_text
    
    def export_pareto_solutions(self, save_csv: bool = True) -> pd.DataFrame:
        """Export Pareto optimal solutions to CSV"""
        pareto_df = self.find_pareto_optimal()
        
        if save_csv:
            csv_path = os.path.join(self.output_dir, 'pareto_optimal_solutions.csv')
            pareto_df.to_csv(csv_path, index=False)
            print(f"Pareto optimal solutions saved to {csv_path}")
        
        return pareto_df
    
    def run_full_analysis(self):
        """Run complete analysis and generate all outputs"""
        print("Running full optimization analysis...")
        
        # Generate all plots
        self.plot_pareto_front()
        self.plot_parameter_sensitivity()
        self.plot_parameter_distributions()
        
        # Generate reports
        report = self.generate_summary_report()
        pareto_df = self.export_pareto_solutions()
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {self.output_dir}")
        
        # Print summary to console
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Pareto optimal solutions found: {len(pareto_df)}")
        print("\nTop parameter ranges:")
        for param in self.param_names:
            min_val = pareto_df[param].min()
            max_val = pareto_df[param].max()
            print(f"  {param}: [{min_val:.3f}, {max_val:.3f}]")


def main():
    """Main entry point for analysis"""
    parser = argparse.ArgumentParser(description="Analyze hyperparameter optimization results")
    parser.add_argument('--results', required=True, help='Path to optimization results CSV')
    parser.add_argument('--output-dir', help='Output directory for plots and reports')
    parser.add_argument('--quick', action='store_true', help='Generate only summary report')
    
    args = parser.parse_args()
    
    analyzer = OptimizationAnalyzer(args.results, args.output_dir)
    
    if args.quick:
        analyzer.generate_summary_report()
        analyzer.export_pareto_solutions()
    else:
        analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 
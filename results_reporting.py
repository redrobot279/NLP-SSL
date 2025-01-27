import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

class ResultsReporter:
    """
    Advanced reporting and visualization for SSL model benchmarks
    
    Features:
    - Tabular results reporting
    - Statistical analysis
    - Visualization of key metrics
    - Export to multiple formats
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: str = None):
        """
        Initialize results reporter
        
        Args:
            results (Dict[str, Any]): Benchmark results dictionary
            output_dir (str, optional): Directory for saving reports
        """
        self.results = results
        self.output_dir = output_dir or os.path.join(os.getcwd(), "benchmark_reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def print_tabular_results(self):
        """
        Print a formatted tabular summary of benchmark results
        """
        print("\n=== SSL Model Benchmark Results ===")
        print("-" * 80)
        header = (
            f"{'Model':<15} | {'Runs':>5} | "
            f"{'Accuracy (Mean ± Std)':>25} | "
            f"{'Runtime (Min)':>15} | "
            f"{'GPU Memory (GB)':>15}"
        )
        print(header)
        print("-" * 80)

        for model_name, model_runs in self.results.items():
            # Calculate statistics
            accuracies = [run['max_accuracy'] for run in model_runs]
            runtimes = [run['runtime'] / 60 for run in model_runs]  # Convert to minutes
            gpu_memories = [run['gpu_memory'] for run in model_runs]

            print(
                f"{model_name:<15} | "
                f"{len(model_runs):>5} | "
                f"{np.mean(accuracies):>8.3f} ± {np.std(accuracies):>6.3f} | "
                f"{np.mean(runtimes):>10.2f} | "
                f"{np.mean(gpu_memories):>12.2f}"
            )
        print("-" * 80)

    def generate_accuracy_boxplot(self):
        """
        Create a boxplot to visualize accuracy distribution across models
        """
        plt.figure(figsize=(10, 6))
        model_accuracies = [
            [run['max_accuracy'] for run in runs] 
            for runs in self.results.values()
        ]
        
        plt.boxplot(
            model_accuracies, 
            labels=list(self.results.keys())
        )
        plt.title('SSL Model Accuracy Distribution')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_boxplot.png'))
        plt.close()

    def export_json_report(self):
        """
        Export detailed benchmark results to JSON
        """
        report_path = os.path.join(self.output_dir, 'benchmark_results.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive benchmark report
        """
        # Print tabular results
        self.print_tabular_results()
        
        # Generate visualizations
        self.generate_accuracy_boxplot()
        
        # Export JSON report
        self.export_json_report()
        
        print(f"\nComprehensive report saved to: {self.output_dir}")
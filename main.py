from data_loader import DatasetManager
from models import MODELS
from benchmark import BenchmarkWorkflow
from results_reporting import ResultsReporter
from dataset_path import DatasetPaths


def main():
    # Configuration
    #/home/ricardo/Documents/NLP Project/Project V2/imagenette2-320/


    # Initialize DatasetPaths and set paths
    dataset_paths = DatasetPaths()

    PATH_TO_TRAIN = dataset_paths.train_path
    PATH_TO_TEST = dataset_paths.val_path


    # Create dataset manager
    dataset_manager = DatasetManager(
        path_to_train=PATH_TO_TRAIN,
        path_to_test=PATH_TO_TEST,
        input_size=128,
        batch_size=128
    )

    # Benchmark configuration
    config = {
        'max_epochs': 2,
        'batch_size': 128,
        'n_runs': 1
    }

    # Create benchmark workflow
    benchmark = BenchmarkWorkflow(models=MODELS, config=config)

    # Run benchmark
    results = benchmark.run_benchmark(dataset_manager)

    # Generate comprehensive report
    reporter = ResultsReporter(results)
    reporter.generate_comprehensive_report()

if __name__ == "__main__":
    main()
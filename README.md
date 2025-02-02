# NLP-SSL: Self-Supervised Learning Benchmarking Framework

A comprehensive framework for benchmarking different Self-Supervised Learning (SSL) models on image classification tasks, implemented in PyTorch and PyTorch Lightning.

## Features

- Multiple SSL model implementations (MOCO, SimCLR, Barlow Twins)
- Automated benchmarking workflow
- Comprehensive performance tracking and visualization
- Distributed training support
- Hardware-aware execution
- Flexible configuration system
- Detailed results reporting

## Installation

1. Clone the repository:
```bash
git clone https://github.com/redrobot279/NLP-SSL.git
cd NLP-SSL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `benchmark.py`: Core benchmarking workflow implementation
- `data_loader.py`: Dataset management and data loading utilities
- `dataset_path.py`: Dataset download and path management
- `models.py`: Implementation of SSL models (MOCO, SimCLR, Barlow Twins)
- `results_reporting.py`: Results analysis and visualization
- `main.py`: Main execution script

## Supported Models

- **Barlow Twins**: Implementation of the Barlow Twins self-supervised learning approach
- **MOCO**: Momentum Contrast for unsupervised visual representation learning
- **SimCLR**: Simple framework for contrastive learning of visual representations

## Usage

Run the benchmark with default settings:

```python
python main.py
```

The framework will automatically:
1. Download and prepare the Imagenette dataset
2. Train and evaluate the implemented SSL models
3. Generate comprehensive performance reports

## Configuration

Key configuration options in `main.py`:

```python
config = {
    'max_epochs': 100,
    'batch_size': 128,
    'n_runs': 1
}
```

Additional configuration options:
- `input_size`: Input image size (default: 128)
- `num_workers`: Number of data loading workers
- `distributed`: Enable distributed training
- `sync_batchnorm`: Enable synchronized batch normalization

## Results and Reporting

The framework generates:
- Tabular performance summaries
- Statistical analyses
- Accuracy distribution visualizations
- Detailed JSON reports
- TensorBoard logs

Results are saved in:
- `benchmark_logs/`: Training logs and checkpoints
- `benchmark_reports/`: Performance reports and visualizations

## Hardware Requirements

- CUDA-capable GPU recommended
- Automatically adapts to available hardware
- Supports multi-GPU training through PyTorch Lightning


## License

[MIT License](LICENSE)

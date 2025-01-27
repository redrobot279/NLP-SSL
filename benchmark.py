import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Type, Dict, Any

class BenchmarkWorkflow:
    """
    A comprehensive workflow for benchmarking self-supervised learning models
    
    Key Features:
    - Supports multiple models and runs
    - Flexible configuration
    - Detailed performance tracking
    - Hardware-aware execution
    """
    
    def __init__(self, 
                 models: List[Type],
                 config: Dict[str, Any] = None):
        """
        Initialize benchmark workflow with configurable parameters
        
        Args:
            models (List[Type]): List of SSL model classes to benchmark
            config (Dict[str, Any], optional): Configuration dictionary
        """
        # Default configuration with optional overrides
        self.default_config = {
            'logs_root_dir': os.path.join(os.getcwd(), "benchmark_logs"),
            'max_epochs': 200,
            'batch_size': 128,
            'num_classes': 10,
            'n_runs': 3,
            'learning_rate': 6e-2,
            'input_size': 128,
            'distributed': False,
            'sync_batchnorm': False
        }
        
        # Update default config with provided configuration
        self.config = {**self.default_config, **(config or {})}
        
        self.models = models
        
        # Detect available hardware
        self.devices = "auto"
        self.accelerator = "auto"
         # Ensure distributed or single GPU configuration
        if self.config.get("distributed", False):
            self.devices = list(range(torch.cuda.device_count()))  # All GPUs
    
         # Enable sync batchnorm if distributed
        self.sync_batchnorm = self.config.get("sync_batchnorm", False)
    
        
        # Initialize results storage
        self.results = {}

    def _setup_experiment(self, model_class, seed):
        """
        Set up experiment environment for a specific model and seed
        
        Args:
            model_class (Type): SSL model class
            seed (int): Random seed for reproducibility
        
        Returns:
            tuple: Configured logger, checkpoint callback, and trainer
        """
        # Set seed for reproducibility
        pl.seed_everything(seed)
        
        # Create unique logging path
        model_name = model_class.__name__.replace("Model", "")
        sub_dir = (f"{model_name}/run{seed}" 
                   if self.config['n_runs'] > 1 
                   else model_name)
        
        # Initialize logger
        logger = TensorBoardLogger(
            save_dir=os.path.join(self.config['logs_root_dir'], "ssl_benchmark"),
            name="",
            sub_dir=sub_dir
        )
        
        # Create checkpoint callback
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints")
        )
        
        # Configure trainer
        trainer = pl.Trainer(
            max_epochs=self.config['max_epochs'],
            devices=self.devices,
            accelerator=self.accelerator,
            logger=logger,
            callbacks=[checkpoint_callback],
            strategy='auto' if self.config['distributed'] else 'single_device',
            sync_batchnorm=self.config['sync_batchnorm']
        )
        
        return logger, checkpoint_callback, trainer

    def run_benchmark(self, dataset_manager):
        """
        Execute comprehensive benchmark across multiple models and runs
        
        Args:
            dataset_manager: Data loading and transformation manager
        
        Returns:
            Dict: Detailed benchmark results
        """
        for model_class in self.models:
            model_results = []
            
            for run in range(self.config['n_runs']):
                # Prepare datasets
                (dataset_train_ssl, 
                 dataset_train_kNN, 
                 dataset_test) = dataset_manager.get_datasets(model_class)
                
                (dataloader_train_ssl, 
                 dataloader_train_kNN, 
                 dataloader_test) = dataset_manager.get_dataloaders(
                    dataset_train_ssl, 
                    dataset_train_kNN, 
                    dataset_test
                )
                
                # Setup experiment
                logger, checkpoint_callback, trainer = self._setup_experiment(
                    model_class, run
                )
                
                # Initialize model
                benchmark_model = model_class(
                    dataloader_train_kNN, 
                    self.config['num_classes']
                )
                
                # Train model
                start_time = time.time()
                trainer.fit(
                    benchmark_model,
                    train_dataloaders=dataloader_train_ssl,
                    val_dataloaders=dataloader_test
                )
                end_time = time.time()
                
                # Collect run results
                run_result = {
                    'model': model_class.__name__,
                    'run': run,
                    'max_accuracy': benchmark_model.max_accuracy,
                    'runtime': end_time - start_time,
                    'gpu_memory': torch.cuda.max_memory_allocated() / (1024**3),
                    'hyperparameters': self.config
                }
                
                model_results.append(run_result)
                
                # Clean up
                del benchmark_model
                del trainer
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            
            self.results[model_class.__name__] = model_results
        
        return self.results
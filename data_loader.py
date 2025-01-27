import torch
from lightly.data import LightlyDataset
from lightly.transforms import (
    BYOLTransform, BYOLView1Transform, BYOLView2Transform,SimCLRTransform, 
)
import torchvision
from lightly.transforms.utils import IMAGENET_NORMALIZE

from torchvision.transforms import v2 as transforms_v2

class DatasetManager:
    def __init__(self, 
                 path_to_train, 
                 path_to_test, 
                 input_size=128, 
                 batch_size=128, 
                 num_workers=2):
        """
        Initialize dataset manager with paths and configuration
        
        Args:
            path_to_train (str): Path to training data
            path_to_test (str): Path to test/validation data
            input_size (int, optional): Size of input images. Defaults to 128.
            batch_size (int, optional): Batch size for dataloaders. Defaults to 128.
            num_workers (int, optional): Number of workers for data loading. Defaults to 12.
        """
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Create transforms
        self.transforms = self._create_transforms()

    def _create_transforms(self):
        """Create various transforms for different SSL methods"""
        normalize_transform = torchvision.transforms.Normalize(
            mean=IMAGENET_NORMALIZE["mean"],
            std=IMAGENET_NORMALIZE["std"],
        )

        # Test transforms
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_size),
            torchvision.transforms.CenterCrop(128),
            transforms_v2.ToImage(),  # Convert to tensor
            transforms_v2.ToDtype(torch.float32, scale=True),  # Convert to float32 and scale           
            torchvision.transforms.ToTensor(),
            normalize_transform,
        ])

        return {
            'test': test_transforms,
            'byol': BYOLTransform(
                view_1_transform=BYOLView1Transform(input_size=self.input_size),
                view_2_transform=BYOLView2Transform(input_size=self.input_size),
            ),
            'simclr': SimCLRTransform(
                input_size=self.input_size,
                cj_strength=0.5,
            )
        }

    def get_datasets(self, model_class):
        """
        Get datasets for a specific SSL model
        
        Args:
            model_class (class): SSL model class to determine transform
        
        Returns:
            tuple: (ssl dataset, kNN train dataset, test dataset)
        """
        # Determine transform based on model
        model_to_transform = {
            'BarlowTwinsModel': self.transforms['byol'],
            'MocoModel': self.transforms['simclr'],
            'SimCLRModel': self.transforms['simclr']
        }
        
        ssl_transform = model_to_transform.get(model_class.__name__, self.transforms['simclr'])
        
        # Create datasets
        dataset_train_ssl = LightlyDataset(
            input_dir=self.path_to_train, 
            transform=ssl_transform
        )
        
        dataset_train_kNN = LightlyDataset(
            input_dir=self.path_to_train, 
            transform=self.transforms['test']
        )
        
        dataset_test = LightlyDataset(
            input_dir=self.path_to_test, 
            transform=self.transforms['test']
        )
        
        return dataset_train_ssl, dataset_train_kNN, dataset_test

    def get_dataloaders(self, dataset_train_ssl, dataset_train_kNN, dataset_test):
        """
        Create data loaders for SSL training, kNN training, and testing
        
        Args:
            dataset_train_ssl (LightlyDataset): SSL training dataset
            dataset_train_kNN (LightlyDataset): kNN training dataset
            dataset_test (LightlyDataset): Test dataset
        
        Returns:
            tuple: (ssl train dataloader, kNN train dataloader, test dataloader)
        """
        dataloader_train_ssl = torch.utils.data.DataLoader(
            dataset_train_ssl,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

        dataloader_train_kNN = torch.utils.data.DataLoader(
            dataset_train_kNN,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        dataloader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

        return dataloader_train_ssl, dataloader_train_kNN, dataloader_test
import copy
import torch
import torch.nn as nn
import torchvision
from lightly import utils
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.modules import heads
from lightly.loss import (
    NTXentLoss,
    BarlowTwinsLoss
)
from lightly.utils.benchmarking import BenchmarkModule

class BaseSSLModel(BenchmarkModule):
    """Base class for Self-Supervised Learning Models"""
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        self.backbone = None
        self.projection_head = None
        self.criterion = None

    def configure_optimizers(self, lr_factor=1, max_epochs=100):  # Increased epochs
        """Configure optimizer and learning rate scheduler with warmup"""
        optim = torch.optim.SGD(
            self.parameters(), 
            lr=6e-2 * lr_factor, 
            momentum=0.9,  # Standard momentum value
            weight_decay=1e-4  # Adjusted weight decay
        )
        
        # Warmup scheduler
        warmup_epochs = 10
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=6e-2 * lr_factor,
                epochs=max_epochs,
                steps_per_epoch=1,  # Adjust based on your dataloader
                pct_start=warmup_epochs/max_epochs
            ),
            'interval': 'epoch',
            'name': 'learning_rate'
        }
        
        return [optim], [scheduler]
        #cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        #return [optim], [cosine_scheduler]

class MocoModel(BaseSSLModel):
    def __init__(self, dataloader_kNN, num_classes, memory_bank_size=4096, distributed=False):
        super().__init__(dataloader_kNN, num_classes)

        # Create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(feature_dim, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # Create our loss with the optional memory bank
        self.criterion = NTXentLoss(
            temperature=0.1, memory_bank_size=(memory_bank_size, 128)
        )
        self.distributed = distributed

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # Update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = batch_shuffle(x1_, distributed=self.distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = batch_unshuffle(x1_, shuffle, distributed=self.distributed)
            return x0_, x1_

        # Symmetric loss
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log("train_loss_ssl", loss)
        return loss

class SimCLRModel(BaseSSLModel):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        
        # Create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

class BarlowTwinsModel(BaseSSLModel):
    def __init__(self, dataloader_kNN, num_classes, gather_distributed=False):
        super().__init__(dataloader_kNN, num_classes)
        
        # Create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        feature_dim = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Use a 2-layer projection head as described in the paper
        self.projection_head = heads.BarlowTwinsProjectionHead(feature_dim, 2048, 2048)
        self.criterion = BarlowTwinsLoss(gather_distributed=gather_distributed)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

# List of models for easy importing in other scripts
MODELS = [
    BarlowTwinsModel,
    MocoModel,
    SimCLRModel
]
import os
from typing import Optional
import torch
from torchvision import transforms

from src.utils import get_transforms


class Config:
    def __init__(self,
                 num_epochs: int = 11,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 seed: int = 42,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 train_dir: str = r"..\train",
                 val_dir: str = r"..\val",
                 test_dir: str = r"..\test",
                 train_transforms: transforms = get_transforms("train"),
                 predict_transforms: transforms = get_transforms("predict"),
                 checkpoint_folder: str = "checkpoints",
                 pretrained_model_version: Optional[int] = None):
        """
        Configuration file for pneumonia detection
        :param num_epochs: number of epochs
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param seed: torch random seed
        :param device: device
        :param train_dir: train directory with images
        :param val_dir: validation directory with images
        :param test_dir: test directory with images
        :param train_transforms: train transforms
        :param predict_transforms: predict transforms
        :param checkpoint_folder: checkpoints folder with model versions
        :param pretrained_model_version: pretrained model version to use in prediction or in further training
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.pretrained_model_version = pretrained_model_version
        self.device = device
        self.num_workers = os.cpu_count()
        self.seed = seed
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.train_transforms = train_transforms
        self.predict_transforms = predict_transforms
        self.checkpoint_folder = checkpoint_folder

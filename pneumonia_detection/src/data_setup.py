"""
Creating dataloaders
"""
from typing import Dict, Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(
        train_dir: str,
        val_dir: str,
        train_transform: transforms.Compose,
        val_transform: transforms.Compose,
        batch_size: int,
        num_workers: int
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Creates dataloaders for training and validation
    :param train_dir: root to train folder with images
    :param val_dir: root to validation folder with images
    :param train_transform: train transforms
    :param val_transform: validation transforms
    :param batch_size: size of batch
    :param num_workers: number of workers
    :return:
    """
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)

    return train_dataloader, val_dataloader, class_names

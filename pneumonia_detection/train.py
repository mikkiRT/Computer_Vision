"""
Trains Pytorch model with instructions
"""
import torch
from typing import Optional

from src.data_setup import create_dataloaders
from src.engine import train
from src.model_builder import ResNet101
from src.utils import clean_checkpoints_folder, make_dirs
from config import Config

configs = Config()

train_dir = configs.train_dir
val_dir = configs.val_dir

device = configs.device
torch.manual_seed(configs.seed)

train_transform = configs.train_transforms
val_transform = configs.predict_transforms


def main(num_epochs: int = configs.num_epochs,
         batch_size: int = configs.batch_size,
         learning_rate: float = configs.learning_rate,
         pretrained_model_version: Optional[int] = configs.pretrained_model_version):
    make_dirs()
    model = ResNet101().to(device)
    if pretrained_model_version:
        model.load_state_dict(torch.load(f"{configs.checkpoint_folder}/checkpoint_{pretrained_model_version}.pt"))
    else:
        clean_checkpoints_folder()

    train_dataloader, val_dataloader, class_names = create_dataloaders(train_dir=train_dir,
                                                                       val_dir=val_dir,
                                                                       train_transform=train_transform,
                                                                       val_transform=val_transform,
                                                                       batch_size=batch_size,
                                                                       num_workers=configs.num_workers)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    train(model=model,
          train_dataloader=train_dataloader,
          val_dataloader=val_dataloader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          epochs=num_epochs,
          device=device)


if __name__ == "__main__":
    main()

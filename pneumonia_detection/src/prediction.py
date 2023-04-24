from typing import Optional

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model_builder import ResNet101
from pneumonia_detection.config import Config

os.chdir("../")

configs = Config()

device = configs.device

test_dir = configs.test_dir
test_transforms = configs.predict_transforms


def predict(pretrained_model_version: Optional[int] = None) -> None:
    """
    Makes prediction on test split
    :param pretrained_model_version: Version of model state. Saved in "checkpoints" folder
    :return: None
    """
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=configs.batch_size,
                                 shuffle=False,
                                 num_workers=configs.num_workers,
                                 pin_memory=True)

    model = ResNet101().to(device)
    if pretrained_model_version:
        model.load_state_dict(torch.load(f=f"{configs.checkpoint_folder}/checkpoint_{pretrained_model_version}.pt"))
    model.eval()

    loss_fn = torch.nn.BCEWithLogitsLoss()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.type(torch.float32).to(device)
            test_logits = model(X)
            test_preds = torch.round(torch.sigmoid(test_logits))

            loss = loss_fn(test_preds.squeeze(), y)
            test_loss += loss.item()
            test_acc += (test_preds.squeeze() == y).sum().item() / len(y)

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}")


if __name__ == "__main__":
    predict()

"""
Training and testing Pytorch model
"""
import torch

from tqdm import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str) -> Tuple[float, float]:
    """
    Executes one training step
    :param model: model to apply
    :param dataloader: train dataloader
    :param loss_fn: loss function for binary problem
    :param optimizer: optimizer
    :param device: device, "cuda" | "cpu"
    :return: training loss and training accuracy
    """
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.type(torch.float32).to(device)

        train_logits = model(X)
        train_preds = torch.round(torch.sigmoid(train_logits))

        loss = loss_fn(train_logits.squeeze(), y)
        train_loss += loss.item()
        train_acc += (train_preds.squeeze() == y).sum().item() / len(y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: str) -> Tuple[float, float]:
    """
    Executes one validation step
    :param model: model to apply
    :param dataloader: validation dataloader
    :param loss_fn: loss function for binary problem
    :param device: device, "cuda" | "cpu"
    :return: validation loss and validation accuracy
    """
    model.eval()

    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.type(torch.float32).to(device)

            val_logits = model(X)
            val_preds = torch.round(torch.sigmoid(val_logits))

            loss = loss_fn(val_logits.squeeze(), y)
            val_loss += loss.item()
            val_acc += (y == val_preds.squeeze()).sum().item() / len(y)

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: str) -> Dict[str, List]:
    """
    Executes training.
    :param model: torch model to use
    :param train_dataloader: train dataloader
    :param val_dataloader: validation dataloader
    :param loss_fn: loss function for binary problem
    :param optimizer:  optimizer
    :param epochs: number of epochs to learn patterns in data
    :param device: device, "cuda" | "cpu"
    :return: result dictionary with train|validation info(loss|accuracy).
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)

        print(f"Epoch: {epoch} | "
              f"Train Loss: {train_loss:.5f} | "
              f"Train Accuracy: {train_acc:.2f} | "
              f"Validation Loss: {val_loss:.5f} | "
              f"Validation Accuracy: {val_acc:.2f}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if epoch % 2 == 0:
            torch.save(obj=model.state_dict(), f=f"checkpoints/checkpoint_{epoch}.pt")

    return results

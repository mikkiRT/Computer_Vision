import os
from torchvision import transforms


def get_transforms(mode: str = "predict") -> transforms.Compose:
    """
    Gets transforms for training and prediction modes
    :param mode: train or predict. Determines which transforms to apply
    :return: transforms.Compose
    """
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif mode == "predict":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def update_checkpoints_folder(checkpoint_dir: str = "checkpoints") -> None:
    """
    Deletes all checkpoints from "checkpoints" folder. Result: clean "checkpoints" folder
    :param checkpoint_dir: str
    :return: None
    """
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    else:
        files = os.listdir(checkpoint_dir)
        for file in files:
            os.remove(f"{checkpoint_dir}/{file}")

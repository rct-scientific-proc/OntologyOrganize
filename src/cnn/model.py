"""ResNet18 model wrapper for transfer learning classification."""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, device: torch.device = None) -> nn.Module:
    """
    Create a ResNet18 model with a custom classification head.

    Uses ImageNet-pretrained weights with the final fully-connected layer
    replaced to match the number of target classes.

    Args:
        num_classes: Number of output classes
        device: Device to place model on (default: auto-detect)

    Returns:
        ResNet18 model ready for fine-tuning
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model.to(device)


def expand_fc_layer(model: nn.Module, old_class_names: list[str],
                    new_class_names: list[str]) -> nn.Module:
    """
    Expand the final FC layer to accommodate new classes.

    Copies weights for existing classes and initializes new class
    outputs with Xavier initialization.

    Args:
        model: Existing trained model
        old_class_names: Class names from previous training
        new_class_names: Full list of class names including new ones

    Returns:
        Model with expanded FC layer
    """
    old_num = len(old_class_names)
    new_num = len(new_class_names)

    if new_num == old_num and old_class_names == new_class_names:
        return model

    in_features = model.fc.in_features
    old_fc = model.fc

    new_fc = nn.Linear(in_features, new_num)
    nn.init.xavier_uniform_(new_fc.weight)
    nn.init.zeros_(new_fc.bias)

    # Map old class indices to new class indices and copy weights
    old_name_to_idx = {name: idx for idx, name in enumerate(old_class_names)}
    with torch.no_grad():
        for new_idx, name in enumerate(new_class_names):
            if name in old_name_to_idx:
                old_idx = old_name_to_idx[name]
                new_fc.weight[new_idx] = old_fc.weight[old_idx]
                new_fc.bias[new_idx] = old_fc.bias[old_idx]

    model.fc = new_fc
    return model

"""Training loop for ResNet18 classification model."""

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.cnn.dataset import LabeledImageDataset
from src.cnn.model import create_model, expand_fc_layer


class TrainingResult:
    """Holds a trained model and its metadata."""

    def __init__(self, model: nn.Module, class_names: list[str],
                 device: torch.device, train_accuracy: float, epochs: int):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.train_accuracy = train_accuracy
        self.epochs = epochs


def train_model(
    labeled_images: dict[str, str],
    class_names: list[str],
    image_cache: dict = None,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    progress_callback: Callable[[int, int, float, float], bool] | None = None,
    resume_from: 'TrainingResult | None' = None,
) -> TrainingResult | None:
    """
    Train a ResNet18 model on labeled images using transfer learning.

    Args:
        labeled_images: Dict mapping image path strings to label strings
        class_names: Ordered list of class names (index = class id)
        image_cache: Optional dict of cached PIL images {path_str: PIL.Image}
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        progress_callback: Called with (epoch, total_epochs, loss, accuracy).
                          Return False to cancel training.
        resume_from: Optional previous TrainingResult to continue training from.
                    Handles new classes by expanding the FC layer.

    Returns:
        TrainingResult on success, None if cancelled or failed
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(class_names)

    # Build label-to-index mapping
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Collect training data
    image_paths = []
    labels = []
    for path_str, label_str in labeled_images.items():
        if label_str in label_to_idx:
            image_paths.append(Path(path_str))
            labels.append(label_to_idx[label_str])

    if len(image_paths) < 2:
        print("CNN: Not enough labeled images to train")
        return None

    # Create dataset and dataloader
    dataset = LabeledImageDataset(image_paths, labels, image_cache=image_cache)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    # Create or resume model
    if resume_from is not None:
        model = resume_from.model
        # Expand FC layer if classes changed
        if class_names != resume_from.class_names:
            model = expand_fc_layer(model, resume_from.class_names, class_names)
            model = model.to(device)
            print(f"CNN: Expanded model from {len(resume_from.class_names)} "
                  f"to {num_classes} classes")
        else:
            print("CNN: Continuing training from previous model")
    else:
        model = create_model(num_classes, device)

    # Freeze all layers except the final fc layer
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    final_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_images, batch_labels in dataloader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        final_accuracy = epoch_accuracy

        print(f"CNN: Epoch {epoch + 1}/{epochs} - "
              f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2%}")

        if progress_callback is not None:
            should_continue = progress_callback(epoch + 1, epochs,
                                                epoch_loss, epoch_accuracy)
            if should_continue is False:
                print("CNN: Training cancelled by user")
                return None

    model.eval()
    return TrainingResult(model, class_names, device, final_accuracy, epochs)

"""Inference using a trained ResNet18 model."""

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.cnn.dataset import UnlabeledImageDataset
from src.cnn.trainer import TrainingResult


class PredictionResult:
    """Inference results for all images."""

    def __init__(self, predictions: dict[str, dict]):
        self._predictions = predictions

    @property
    def predictions(self) -> dict[str, dict]:
        """Dict mapping image path strings to prediction info.

        Each value is a dict with:
            - 'predicted_label': str — most likely class name
            - 'confidence': float — probability for the predicted class
            - 'probabilities': dict[str, float] — class name → probability
        """
        return self._predictions

    def sort_by_class_confidence(self, class_name: str) -> list[tuple[Path, float]]:
        """Return image paths sorted by descending confidence for a class.

        Args:
            class_name: Target class name to sort by

        Returns:
            List of (Path, confidence) tuples sorted by descending confidence
        """
        items = []
        for path_str, pred in self._predictions.items():
            conf = pred['probabilities'].get(class_name, 0.0)
            items.append((Path(path_str), conf))
        items.sort(key=lambda x: x[1], reverse=True)
        return items

    def get_suggested_labels(self) -> dict[str, str]:
        """Return a dict mapping image path strings to their suggested labels.

        Returns:
            Dict of {image_path_str: predicted_label_str}
        """
        return {path: pred['predicted_label']
                for path, pred in self._predictions.items()}


def run_inference(
    training_result: TrainingResult,
    image_paths: list[Path],
    image_cache: dict = None,
    batch_size: int = 32,
    progress_callback: Callable[[int, int], bool] | None = None,
) -> PredictionResult | None:
    """
    Run inference on a list of images using a trained model.

    Args:
        training_result: Output from train_model()
        image_paths: List of image paths to classify
        image_cache: Optional dict of cached PIL images
        batch_size: Inference batch size
        progress_callback: Called with (processed, total).
                          Return False to cancel.

    Returns:
        PredictionResult on success, None if cancelled
    """
    model = training_result.model
    class_names = training_result.class_names
    device = training_result.device

    dataset = UnlabeledImageDataset(image_paths, image_cache=image_cache)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    predictions = {}
    processed = 0
    total = len(image_paths)

    model.eval()
    with torch.no_grad():
        for batch_tensors, batch_paths in dataloader:
            batch_tensors = batch_tensors.to(device)
            outputs = model(batch_tensors)
            probabilities = F.softmax(outputs, dim=1)

            for i, path_str in enumerate(batch_paths):
                probs = probabilities[i].cpu().numpy()
                predicted_idx = probs.argmax()

                predictions[path_str] = {
                    'predicted_label': class_names[predicted_idx],
                    'confidence': float(probs[predicted_idx]),
                    'probabilities': {
                        name: float(probs[j])
                        for j, name in enumerate(class_names)
                    },
                }

            processed += len(batch_paths)

            if progress_callback is not None:
                should_continue = progress_callback(processed, total)
                if should_continue is False:
                    print("CNN: Inference cancelled by user")
                    return None

    return PredictionResult(predictions)

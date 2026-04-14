"""Background QThread workers for CNN training and inference."""

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """Runs CNN training in a background thread."""

    # (epoch, total_epochs, loss, accuracy)
    progress = pyqtSignal(int, int, float, float)
    finished = pyqtSignal(object)  # TrainingResult or None
    error = pyqtSignal(str)

    def __init__(self, labeled_images, class_names, image_cache,
                 epochs, batch_size, learning_rate=1e-3, resume_from=None):
        super().__init__()
        self.labeled_images = labeled_images
        self.class_names = class_names
        self.image_cache = image_cache
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.resume_from = resume_from
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from src.cnn.trainer import train_model

            def on_progress(epoch, total_epochs, loss, accuracy):
                if self._cancelled:
                    return False
                self.progress.emit(epoch, total_epochs, loss, accuracy)
                return True

            result = train_model(
                labeled_images=self.labeled_images,
                class_names=self.class_names,
                image_cache=self.image_cache,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                progress_callback=on_progress,
                resume_from=self.resume_from,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class InferenceWorker(QThread):
    """Runs CNN inference in a background thread."""

    # (processed, total)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)  # PredictionResult or None
    error = pyqtSignal(str)

    def __init__(self, training_result, image_paths, image_cache,
                 batch_size=32):
        super().__init__()
        self.training_result = training_result
        self.image_paths = image_paths
        self.image_cache = image_cache
        self.batch_size = batch_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from src.cnn.inference import run_inference

            def on_progress(processed, total):
                if self._cancelled:
                    return False
                self.progress.emit(processed, total)
                return True

            result = run_inference(
                training_result=self.training_result,
                image_paths=self.image_paths,
                image_cache=self.image_cache,
                batch_size=self.batch_size,
                progress_callback=on_progress,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

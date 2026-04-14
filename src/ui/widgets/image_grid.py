"""Widget for displaying images in a grid layout."""

from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel, QScrollArea
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QMouseEvent
from pathlib import Path
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.image_utils import create_thumbnail


class ClickableImageLabel(QLabel):
    """A clickable label for displaying images."""
    clicked = pyqtSignal(bool)  # shift_held
    right_clicked = pyqtSignal()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press event."""
        if event.button() == Qt.LeftButton and self.pixmap() and not self.pixmap().isNull():
            self.clicked.emit(bool(event.modifiers() & Qt.ShiftModifier))
        elif event.button() == Qt.RightButton and self.pixmap() and not self.pixmap().isNull():
            self.right_clicked.emit()
        super().mousePressEvent(event)


class ImageGridWidget(QWidget):
    """Widget that displays images in a grid."""
    
    def __init__(self, cols=3, rows=4, parent=None):
        super().__init__(parent)
        self.main_window = None  # Will be set by parent
        self.cols = cols
        self.rows = rows
        self.image_labels = []
        self.current_page = 0
        self.images_per_page = cols * rows
        self.all_images = []
        self.image_size = 150  # Default image size
        self.colormap = "gray"  # Default colormap
        self.transform = "none"  # Current transformation
        self.selected_indices = set()  # Track selected image indices on current page
        self.image_selections = {}  # Maps image path to (label, color) tuple for pending labels
        self.image_label_map = {}  # Maps image path to label
        self.label_colors = {}  # Maps label to color
        self.correlation_mode = False  # Whether we're in correlation base image selection mode
        self.thread_count = 8  # Number of threads for parallel processing
        self.image_cache = {}  # Cache for preloaded images {path: PIL.Image}
        self.force_grayscale = False  # Whether to force grayscale conversion
        self._last_clicked_idx = None  # Track last clicked index for shift-select
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        layout = QGridLayout()
        layout.setSpacing(5)
        self.setLayout(layout)
        
        # Create grid of clickable image labels
        for row in range(self.rows):
            for col in range(self.cols):
                label = ClickableImageLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
                label.setFixedSize(self.image_size, self.image_size)
                label.setScaledContents(False)
                label.setCursor(Qt.PointingHandCursor)
                
                # Connect click signal
                idx = row * self.cols + col
                label.clicked.connect(lambda shift, i=idx: self.on_image_clicked(i, shift))
                label.right_clicked.connect(lambda i=idx: self.on_image_right_clicked(i))
                
                layout.addWidget(label, row, col)
                self.image_labels.append(label)
        
        # Update widget size
        self.update_widget_size()
    
    def update_widget_size(self):
        """Update the widget size based on grid dimensions and image size."""
        spacing = self.layout().spacing()
        width = self.cols * self.image_size + (self.cols - 1) * spacing + 20
        height = self.rows * self.image_size + (self.rows - 1) * spacing + 20
        self.setFixedSize(width, height)
    
    def set_zoom(self, size: int):
        """Set the image size (zoom level)."""
        self.image_size = size
        
        # Update all label sizes
        for label in self.image_labels:
            label.setFixedSize(self.image_size, self.image_size)
        
        # Update widget size
        self.update_widget_size()
        
        # Reload current page with new size
        self.load_page(self.current_page)
    
    def set_colormap(self, colormap: str):
        """Set the colormap to use for displaying images."""
        self.colormap = colormap
    
    def set_transform(self, transform: str):
        """Set the transformation to apply to images."""
        self.transform = transform
    
    def set_correlation_mode(self, enabled: bool):
        """Set whether correlation mode is enabled."""
        self.correlation_mode = enabled
    
    def set_thread_count(self, count: int):
        """Set the number of threads for parallel processing."""
        self.thread_count = max(1, min(32, count))  # Clamp between 1 and 32
    
    def set_force_grayscale(self, enabled: bool):
        """Set whether to force grayscale conversion."""
        self.force_grayscale = enabled
    
    def set_grid_size(self, cols: int, rows: int):
        """Update the grid size."""
        self.cols = cols
        self.rows = rows
        self.images_per_page = cols * rows
        
        # Clear existing widgets
        layout = self.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.image_labels.clear()
        
        # Recreate grid with new size
        for row in range(self.rows):
            for col in range(self.cols):
                label = ClickableImageLabel()
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
                label.setFixedSize(self.image_size, self.image_size)
                label.setScaledContents(False)
                label.setCursor(Qt.PointingHandCursor)
                
                # Connect click signal
                idx = row * self.cols + col
                label.clicked.connect(lambda shift, i=idx: self.on_image_clicked(i, shift))
                label.right_clicked.connect(lambda i=idx: self.on_image_right_clicked(i))
                
                layout.addWidget(label, row, col)
                self.image_labels.append(label)
        
        # Update widget size
        self.update_widget_size()
        
        # Reload current page
        self.load_page(self.current_page)
    
    def set_images(self, image_paths: List[Path]):
        """Set the list of images to display."""
        self.all_images = image_paths
        self.current_page = 0
        self.load_page(0)
    
    def load_page(self, page: int, progress_dialog=None):
        """Load images for the specified page using multithreading."""
        self.current_page = page
        self.selected_indices.clear()  # Clear selections when changing pages
        self._last_clicked_idx = None  # Reset shift-select anchor on page change
        start_idx = page * self.images_per_page
        end_idx = start_idx + self.images_per_page
        
        page_images = self.all_images[start_idx:end_idx]
        
        # Clear all labels first
        for label in self.image_labels:
            label.clear()
            label.setText("")
            label.setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
        
        # Use ThreadPoolExecutor for parallel thumbnail creation
        def create_thumbnail_for_image(idx_img_tuple):
            """Helper function to create thumbnail for a single image."""
            idx, img_path = idx_img_tuple
            if idx < len(self.image_labels):
                # Get cached image if available
                cached_img = self.image_cache.get(str(img_path), None)
                
                pixmap = create_thumbnail(
                    img_path, 
                    size=(self.image_size, self.image_size), 
                    colormap=self.colormap, 
                    transform=self.transform,
                    cached_image=cached_img,
                    force_grayscale=self.force_grayscale
                )
                return idx, img_path, pixmap
            return idx, img_path, None
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            # Submit all tasks
            futures = {
                executor.submit(create_thumbnail_for_image, (idx, img_path)): idx
                for idx, img_path in enumerate(page_images)
            }
            
            # Process completed tasks as they finish
            completed_count = 0
            for future in as_completed(futures):
                if progress_dialog and progress_dialog.wasCanceled():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return
                
                idx, img_path, pixmap = future.result()
                
                if pixmap and idx < len(self.image_labels):
                    # Scale pixmap to fit label while maintaining aspect ratio
                    scaled_pixmap = pixmap.scaled(
                        self.image_size, self.image_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.image_labels[idx].setPixmap(scaled_pixmap)
                    
                    # Apply colored border if image is labeled
                    img_path_str = str(img_path)
                    if img_path_str in self.image_label_map:
                        label_name = self.image_label_map[img_path_str]
                        if label_name in self.label_colors:
                            color = self.label_colors[label_name]
                            self.image_labels[idx].setStyleSheet(f"border: 3px solid {color}; background-color: #f5f5f5;")
                
                # Update progress dialog
                completed_count += 1
                if progress_dialog:
                    progress_dialog.setValue(completed_count)
                    progress_dialog.setLabelText(f"Applying transform...\nProcessed {completed_count}/{len(page_images)} images")
    
    def next_page(self):
        """Load the next page of images."""
        max_page = (len(self.all_images) - 1) // self.images_per_page
        if self.current_page < max_page:
            self.load_page(self.current_page + 1)
    
    def prev_page(self):
        """Load the previous page of images."""
        if self.current_page > 0:
            self.load_page(self.current_page - 1)
    
    def get_current_page(self):
        """Get current page number."""
        return self.current_page
    
    def get_total_pages(self):
        """Get total number of pages."""
        if len(self.all_images) == 0:
            return 0
        return (len(self.all_images) - 1) // self.images_per_page + 1
    
    def on_image_clicked(self, idx: int, shift_held: bool = False):
        """Handle image click and get active label from parent."""
        # Check if we're in correlation mode first
        if self.correlation_mode:
            start_idx = self.current_page * self.images_per_page
            if idx < len(self.image_labels) and start_idx + idx < len(self.all_images):
                img_path = self.all_images[start_idx + idx]
                print(f"Base image selected for correlation: {img_path}")
                # Call main window method to handle correlation
                if self.main_window and hasattr(self.main_window, 'on_base_image_selected'):
                    self.main_window.on_base_image_selected(img_path)
            return
        
        # Normal labeling mode
        # Get active label from main window
        active_label = None
        if self.main_window and hasattr(self.main_window, 'active_label'):
            active_label = self.main_window.active_label
        print(f"DEBUG: on_image_clicked - active_label from main_window: {active_label}")
        
        if shift_held and self._last_clicked_idx is not None:
            # Shift-click: select range between last clicked and current
            lo = min(self._last_clicked_idx, idx)
            hi = max(self._last_clicked_idx, idx)
            start_idx = self.current_page * self.images_per_page
            for i in range(lo, hi + 1):
                if i not in self.selected_indices and i < len(self.image_labels) and start_idx + i < len(self.all_images):
                    self.toggle_selection(i, active_label)
        else:
            self.toggle_selection(idx, active_label)
        
        self._last_clicked_idx = idx
    
    def on_image_right_clicked(self, idx: int):
        """Handle right-click to remove label from image."""
        print(f"DEBUG: on_image_right_clicked - idx={idx}")
        start_idx = self.current_page * self.images_per_page
        if idx < len(self.image_labels) and start_idx + idx < len(self.all_images):
            img_path = self.all_images[start_idx + idx]
            img_path_str = str(img_path)
            print(f"DEBUG: Right-clicked image: {img_path_str}")
            
            # Check if this image has a pending selection (not yet confirmed)
            if img_path_str in self.image_selections:
                label, color = self.image_selections[img_path_str]
                print(f"DEBUG: Image has pending label '{label}', removing selection")
                # Remove from pending selections
                del self.image_selections[img_path_str]
                if idx in self.selected_indices:
                    self.selected_indices.remove(idx)
                # Reset border to default
                self.image_labels[idx].setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
                return
            
            # Check if this image has a confirmed label
            if img_path_str in self.image_label_map:
                old_label = self.image_label_map[img_path_str]
                print(f"DEBUG: Image has confirmed label '{old_label}', requesting removal")
                # Call main window to remove the label
                if self.main_window and hasattr(self.main_window, 'remove_image_label'):
                    self.main_window.remove_image_label(img_path_str)
            else:
                print(f"DEBUG: Image has no label, nothing to remove")
    
    def toggle_selection(self, idx: int, current_label: str = None):
        """Toggle selection of an image."""
        print(f"DEBUG: toggle_selection called with idx={idx}, current_label={current_label}")
        start_idx = self.current_page * self.images_per_page
        if idx < len(self.image_labels) and start_idx + idx < len(self.all_images):
            img_path = self.all_images[start_idx + idx]
            img_path_str = str(img_path)
            
            if idx in self.selected_indices:
                print(f"DEBUG: Deselecting image at index {idx}")
                self.selected_indices.remove(idx)
                # Remove from pending selections
                if img_path_str in self.image_selections:
                    del self.image_selections[img_path_str]
                # Reset to default or labeled color
                if img_path_str in self.image_label_map:
                    label_name = self.image_label_map[img_path_str]
                    color = self.label_colors.get(label_name, "#ccc")
                    self.image_labels[idx].setStyleSheet(f"border: 3px solid {color}; background-color: #f5f5f5;")
                else:
                    self.image_labels[idx].setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
            else:
                print(f"DEBUG: Selecting image at index {idx} with label {current_label}")
                self.selected_indices.add(idx)
                # Store the label this image should get
                if current_label:
                    # Get or generate color for this label
                    if current_label not in self.label_colors:
                        self.label_colors[current_label] = self.generate_temp_color(len(self.label_colors))
                    color = self.label_colors[current_label]
                    self.image_selections[img_path_str] = (current_label, color)
                    # Apply the color for this label
                    self.image_labels[idx].setStyleSheet(f"border: 3px solid {color}; background-color: #e6f2ff;")
                else:
                    # No label specified, use default selection color
                    self.image_labels[idx].setStyleSheet("border: 3px solid #0078d4; background-color: #e6f2ff;")
        else:
            print(f"DEBUG: Invalid index or no image at position {idx}")
    
    def get_selected_images(self) -> List[Path]:
        """Get list of currently selected image paths."""
        start_idx = self.current_page * self.images_per_page
        selected_images = []
        for idx in self.selected_indices:
            if start_idx + idx < len(self.all_images):
                selected_images.append(self.all_images[start_idx + idx])
        return selected_images
    
    def get_pending_label_assignments(self) -> Dict[str, str]:
        """Get dictionary of image paths to their pending label assignments."""
        return {path: label for path, (label, color) in self.image_selections.items()}
    
    def clear_selections(self):
        """Clear all selections."""
        for idx in list(self.selected_indices):
            if idx < len(self.image_labels):
                start_idx = self.current_page * self.images_per_page
                img_path_str = str(self.all_images[start_idx + idx])
                if img_path_str in self.image_label_map:
                    label_name = self.image_label_map[img_path_str]
                    color = self.label_colors.get(label_name, "#ccc")
                    self.image_labels[idx].setStyleSheet(f"border: 3px solid {color}; background-color: #f5f5f5;")
                else:
                    self.image_labels[idx].setStyleSheet("border: 3px solid #ccc; background-color: #f5f5f5;")
        self.selected_indices.clear()
        self.image_selections.clear()
    
    def set_image_labels(self, image_label_map: Dict[str, str]):
        """Set the mapping of image paths to labels."""
        self.image_label_map = image_label_map
    
    def set_label_colors(self, label_colors: Dict[str, str]):
        """Set the mapping of labels to colors."""
        self.label_colors = label_colors
    
    def generate_temp_color(self, index: int) -> str:
        """Generate a temporary color for a label."""
        colors = [
            "#e74c3c",  # Red
            "#3498db",  # Blue
            "#2ecc71",  # Green
            "#f39c12",  # Orange
            "#9b59b6",  # Purple
            "#1abc9c",  # Turquoise
            "#e67e22",  # Carrot
            "#34495e",  # Dark gray
            "#e91e63",  # Pink
            "#00bcd4",  # Cyan
            "#ff9800",  # Amber
            "#795548",  # Brown
            "#607d8b",  # Blue gray
            "#ff5722",  # Deep orange
            "#8bc34a",  # Light green
        ]
        return colors[index % len(colors)]
    
    def set_active_label(self, label: str):
        """Set the active label that will be used for new selections."""
        self.active_label = label
    
    def preload_all_images(self, image_paths: list, progress_dialog=None):
        """Preload all images into memory cache."""
        from PIL import Image
        
        print(f"Preloading {len(image_paths)} images into memory...")
        self.image_cache.clear()
        
        for idx, img_path in enumerate(image_paths):
            if progress_dialog and progress_dialog.wasCanceled():
                print("Preload cancelled.")
                self.image_cache.clear()
                return False
            
            try:
                # Load image and normalize if needed (e.g. float32 TIFFs)
                from src.utils.image_utils import normalize_image
                with Image.open(img_path) as img:
                    img = normalize_image(img)
                    
                    # Force load image data into memory and close file handle
                    img.load()
                    
                    # Store in cache
                    self.image_cache[str(img_path)] = img
                
                if progress_dialog:
                    progress_dialog.setValue(idx + 1)
                    if (idx + 1) % 100 == 0:
                        progress_dialog.setLabelText(f"Preloading images...\nLoaded {idx + 1}/{len(image_paths)} images")
            
            except Exception as e:
                print(f"Error preloading {img_path}: {e}")
        
        print(f"Preloaded {len(self.image_cache)} images.")
        return True
    
    def clear_preloaded_images(self):
        """Clear the preloaded image cache."""
        self.image_cache.clear()
        print("Image cache cleared.")

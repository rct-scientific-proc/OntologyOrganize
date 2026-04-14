"""Main application window."""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QFileDialog, QListWidget,
    QListWidgetItem, QSplitter, QPushButton, QScrollArea, QProgressDialog,
    QWidgetAction, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QWheelEvent
from pathlib import Path

from src.controllers.image_controller import ImageController
from src.ui.widgets.image_grid import ImageGridWidget
from src.ui.widgets.classification_panel import ClassificationPanel
from src.ui.dialogs.settings_dialog import SettingsDialog
from src.ui.dialogs.colormap_dialog import ColormapDialog
from src.ui.view_manager import ViewManager, ViewMode
from src.utils.export_manager import ExportManager
from src.utils.progress_manager import ProgressManager


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.controller = ImageController()
        self.grid_cols = 3
        self.grid_rows = 4
        self.zoom_level = 150  # Default zoom level
        self.colormap = "gray"  # Default colormap
        self.labeled_images = {}  # Maps image path to label
        self.label_colors = {}  # Maps label to color
        self.active_label = None  # Currently active label for selection
        self.view_manager = ViewManager()  # Manages view mode
        self.custom_sort_order = None  # Stores custom sorted order of unlabeled images
        self.export_manager = ExportManager(self)  # Manages exports
        self.progress_manager = ProgressManager(self)  # Manages save/load progress
        self.current_transform = 'none'  # Current image transformation
        self.correlation_mode = False  # Whether we're in correlation selection mode
        self.correlation_method = 'ncc'  # Method to use for correlation
        self.base_image_for_correlation = None  # Base image path for correlation
        self.thread_count = 8  # Number of threads for parallel processing
        self.preload_images = False  # Whether to preload all images into memory
        self.binary_mode = False  # Whether to use binary classification mode
        self.binary_positive_class = "Positive Class"  # Label for positive class in binary mode
        self.binary_negative_class = "Negative Class"  # Label for negative class in binary mode
        self.binary_mode_warning_shown = False  # Track if binary mode warning has been shown
        self.force_grayscale = False  # Whether to force grayscale conversion
        self.cnn_result = None  # Trained CNN model result
        self.cnn_model_path = None  # Path to last saved CNN model
        self.cnn_prediction = None  # Cached inference results
        self.confidence_cutoff = 0.0  # Minimum confidence for sort-by-confidence display
        self.train_batch_size = 16  # Training batch size
        self.inference_batch_size = 32  # Inference batch size
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Classifier Organizer")
        self.setGeometry(100, 100, 1200, 700)
        
        # Create menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        # Add "Open Folder" action to File menu
        open_folder_action = file_menu.addAction("Open Folder...")
        open_folder_action.triggered.connect(self.open_folder_dialog)
        
        # Add Settings menu
        settings_menu = menubar.addMenu("Settings")
        
        # Add grid settings action
        grid_settings_action = settings_menu.addAction("Grid Size...")
        grid_settings_action.triggered.connect(self.open_grid_settings_dialog)
        
        # Add colormap settings action
        colormap_settings_action = settings_menu.addAction("Colormap...")
        colormap_settings_action.triggered.connect(self.open_colormap_dialog)
        
        # Add binary mode settings action
        binary_mode_settings_action = settings_menu.addAction("Binary Mode...")
        binary_mode_settings_action.triggered.connect(self.open_binary_mode_dialog)
        
        # Add thread settings action
        thread_settings_action = settings_menu.addAction("Thread Count...")
        thread_settings_action.triggered.connect(self.open_thread_settings_dialog)
        
        # Add preload images setting
        self.preload_images_action = settings_menu.addAction("Preload All Images")
        self.preload_images_action.setCheckable(True)
        self.preload_images_action.setChecked(self.preload_images)
        self.preload_images_action.triggered.connect(self.toggle_preload_images)
        
        # Add force grayscale setting
        self.force_grayscale_action = settings_menu.addAction("Force Grayscale")
        self.force_grayscale_action.setCheckable(True)
        self.force_grayscale_action.setChecked(self.force_grayscale)
        self.force_grayscale_action.triggered.connect(self.toggle_force_grayscale)
        
        # Add View menu
        view_menu = menubar.addMenu("View")
        
        # Add view mode actions
        self.view_unlabeled_action = view_menu.addAction("Show Unlabeled Images")
        self.view_unlabeled_action.setCheckable(True)
        self.view_unlabeled_action.setChecked(True)  # Default mode
        self.view_unlabeled_action.triggered.connect(self.set_view_unlabeled)
        
        self.view_labeled_action = view_menu.addAction("Show Labeled Images")
        self.view_labeled_action.setCheckable(True)
        self.view_labeled_action.triggered.connect(self.set_view_labeled)
        
        self.view_all_action = view_menu.addAction("Show All Images")
        self.view_all_action.setCheckable(True)
        self.view_all_action.triggered.connect(self.set_view_all)
        
        # Add Export menu
        export_menu = menubar.addMenu("Export")
        
        # Add export actions
        export_json_action = export_menu.addAction("Export to JSON...")
        export_json_action.triggered.connect(self.export_to_json)
        
        export_csv_action = export_menu.addAction("Export to CSV...")
        export_csv_action.triggered.connect(self.export_to_csv)
        
        export_menu.addSeparator()
        
        export_organized_action = export_menu.addAction("Export Organized Copy...")
        export_organized_action.triggered.connect(self.export_organized_copy)
        
        # Add Progress menu
        progress_menu = menubar.addMenu("Progress")
        save_progress_action = progress_menu.addAction("Save Progress...")
        save_progress_action.triggered.connect(self.save_progress)
        save_progress_action.setShortcut("Ctrl+S")
        
        quick_save_action = progress_menu.addAction("Quick Save")
        quick_save_action.triggered.connect(self.quick_save_progress)
        quick_save_action.setShortcut("Ctrl+Shift+S")
        
        load_progress_action = progress_menu.addAction("Load Progress...")
        load_progress_action.triggered.connect(self.load_progress)
        load_progress_action.setShortcut("Ctrl+O")
        
        # Add Transform menu
        transform_menu = menubar.addMenu("Transform")
        
        # Add transform actions
        transform_none_action = transform_menu.addAction("None (Original)")
        transform_none_action.triggered.connect(lambda: self.set_transform('none'))
        
        transform_menu.addSeparator()
        
        transform_sobel_action = transform_menu.addAction("Edge Detector (Sobel)")
        transform_sobel_action.triggered.connect(lambda: self.set_transform('sobel'))
        
        transform_prewitt_action = transform_menu.addAction("Edge Detector (Prewitt)")
        transform_prewitt_action.triggered.connect(lambda: self.set_transform('prewitt'))
        
        transform_canny_action = transform_menu.addAction("Edge Detector (Canny)")
        transform_canny_action.triggered.connect(lambda: self.set_transform('canny'))
        
        transform_gradient_action = transform_menu.addAction("Gradient Magnitude")
        transform_gradient_action.triggered.connect(lambda: self.set_transform('gradient'))
        
        transform_laplacian_action = transform_menu.addAction("Laplacian (Edge Enhancement)")
        transform_laplacian_action.triggered.connect(lambda: self.set_transform('laplacian'))
        
        transform_menu.addSeparator()
        
        transform_blur_action = transform_menu.addAction("Gaussian Blur")
        transform_blur_action.triggered.connect(lambda: self.set_transform('gaussian_blur'))
        
        transform_median_action = transform_menu.addAction("Median Filter")
        transform_median_action.triggered.connect(lambda: self.set_transform('median'))
        
        transform_bilateral_action = transform_menu.addAction("Bilateral Filter")
        transform_bilateral_action.triggered.connect(lambda: self.set_transform('bilateral'))
        
        transform_sharpen_action = transform_menu.addAction("Sharpen")
        transform_sharpen_action.triggered.connect(lambda: self.set_transform('sharpen'))
        
        transform_highpass_action = transform_menu.addAction("High-Pass Filter")
        transform_highpass_action.triggered.connect(lambda: self.set_transform('highpass'))
        
        transform_menu.addSeparator()
        
        # Morphological operations submenu
        morph_menu = transform_menu.addMenu("Morphological")
        
        morph_erosion_action = morph_menu.addAction("Erosion")
        morph_erosion_action.triggered.connect(lambda: self.set_transform('erosion'))
        
        morph_dilation_action = morph_menu.addAction("Dilation")
        morph_dilation_action.triggered.connect(lambda: self.set_transform('dilation'))
        
        morph_opening_action = morph_menu.addAction("Opening")
        morph_opening_action.triggered.connect(lambda: self.set_transform('opening'))
        
        morph_closing_action = morph_menu.addAction("Closing")
        morph_closing_action.triggered.connect(lambda: self.set_transform('closing'))
        
        # Thresholding submenu
        threshold_menu = transform_menu.addMenu("Thresholding")
        
        threshold_otsu_action = threshold_menu.addAction("Otsu's Threshold")
        threshold_otsu_action.triggered.connect(lambda: self.set_transform('otsu'))
        
        threshold_adaptive_action = threshold_menu.addAction("Adaptive Threshold")
        threshold_adaptive_action.triggered.connect(lambda: self.set_transform('adaptive'))
        
        # Feature detection submenu
        feature_menu = transform_menu.addMenu("Feature Detection")
        
        feature_dog_action = feature_menu.addAction("Difference of Gaussians")
        feature_dog_action.triggered.connect(lambda: self.set_transform('dog'))
        
        feature_gabor_action = feature_menu.addAction("Gabor Filter")
        feature_gabor_action.triggered.connect(lambda: self.set_transform('gabor'))
        
        transform_menu.addSeparator()
        
        transform_emboss_action = transform_menu.addAction("Emboss")
        transform_emboss_action.triggered.connect(lambda: self.set_transform('emboss'))
        
        transform_invert_action = transform_menu.addAction("Invert")
        transform_invert_action.triggered.connect(lambda: self.set_transform('invert'))
        
        # Add Sort menu
        sort_menu = menubar.addMenu("Sort")
        
        # Pixel-Based submenu
        pixel_menu = sort_menu.addMenu("Pixel-Based Similarity")
        
        pixel_ncc_action = pixel_menu.addAction("Normalized Cross-Correlation (NCC)")
        pixel_ncc_action.triggered.connect(lambda: self.start_correlation_mode('ncc'))
        
        pixel_ssim_action = pixel_menu.addAction("SSIM (Structural Similarity)")
        pixel_ssim_action.triggered.connect(lambda: self.start_correlation_mode('ssim'))
        
        pixel_mae_action = pixel_menu.addAction("Mean Absolute Error (MAE)")
        pixel_mae_action.triggered.connect(lambda: self.start_correlation_mode('mae'))
        
        pixel_mse_action = pixel_menu.addAction("Mean Squared Error (MSE)")
        pixel_mse_action.triggered.connect(lambda: self.start_correlation_mode('mse'))
        
        pixel_cosine_action = pixel_menu.addAction("Cosine Similarity")
        pixel_cosine_action.triggered.connect(lambda: self.start_correlation_mode('cosine'))
        
        # Histogram-Based submenu
        histogram_menu = sort_menu.addMenu("Histogram-Based Similarity")
        
        hist_corr_action = histogram_menu.addAction("Histogram Correlation")
        hist_corr_action.triggered.connect(lambda: self.start_correlation_mode('hist_corr'))
        
        hist_chi_action = histogram_menu.addAction("Chi-Square Distance")
        hist_chi_action.triggered.connect(lambda: self.start_correlation_mode('chi_square'))
        
        hist_bhatt_action = histogram_menu.addAction("Bhattacharyya Distance")
        hist_bhatt_action.triggered.connect(lambda: self.start_correlation_mode('bhattacharyya'))
        
        hist_emd_action = histogram_menu.addAction("Earth Mover's Distance")
        hist_emd_action.triggered.connect(lambda: self.start_correlation_mode('emd'))
        
        # Feature-Based submenu
        feature_sort_menu = sort_menu.addMenu("Feature-Based Similarity")
        
        sift_action = feature_sort_menu.addAction("SIFT Feature Matching")
        sift_action.triggered.connect(lambda: self.start_correlation_mode('sift'))
        
        hog_action = feature_sort_menu.addAction("HOG Similarity")
        hog_action.triggered.connect(lambda: self.start_correlation_mode('hog'))
        
        mi_action = feature_sort_menu.addAction("Mutual Information")
        mi_action.triggered.connect(lambda: self.start_correlation_mode('mutual_info'))
        
        # Perceptual submenu
        perceptual_menu = sort_menu.addMenu("Perceptual Hashing")
        
        phash_action = perceptual_menu.addAction("Perceptual Hash (pHash)")
        phash_action.triggered.connect(lambda: self.start_correlation_mode('phash'))
        
        dhash_action = perceptual_menu.addAction("Difference Hash (dHash)")
        dhash_action.triggered.connect(lambda: self.start_correlation_mode('dhash'))
        
        sort_menu.addSeparator()
        
        sort_cancel_action = sort_menu.addAction("Cancel Selection")
        sort_cancel_action.triggered.connect(self.cancel_correlation_mode)
        
        # Add CNN menu
        cnn_menu = menubar.addMenu("CNN")
        
        cnn_train_action = cnn_menu.addAction("Train on Labeled Images...")
        cnn_train_action.triggered.connect(self.cnn_train)
        
        cnn_menu.addSeparator()
        
        # Train Batch Size dropdown
        train_batch_widget = QWidget()
        train_batch_layout = QHBoxLayout(train_batch_widget)
        train_batch_layout.setContentsMargins(8, 2, 8, 2)
        train_batch_layout.addWidget(QLabel("Train Batch:"))
        self._train_batch_combo = QComboBox()
        self._train_batch_combo.setEditable(True)
        batch_options = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"]
        self._train_batch_combo.addItems(batch_options)
        self._train_batch_combo.setCurrentText(str(self.train_batch_size))
        self._train_batch_combo.currentTextChanged.connect(self._on_train_batch_changed)
        train_batch_layout.addWidget(self._train_batch_combo)
        train_batch_action = QWidgetAction(self)
        train_batch_action.setDefaultWidget(train_batch_widget)
        cnn_menu.addAction(train_batch_action)
        
        # Inference Batch Size dropdown
        infer_batch_widget = QWidget()
        infer_batch_layout = QHBoxLayout(infer_batch_widget)
        infer_batch_layout.setContentsMargins(8, 2, 8, 2)
        infer_batch_layout.addWidget(QLabel("Inference Batch:"))
        self._infer_batch_combo = QComboBox()
        self._infer_batch_combo.setEditable(True)
        self._infer_batch_combo.addItems(batch_options)
        self._infer_batch_combo.setCurrentText(str(self.inference_batch_size))
        self._infer_batch_combo.currentTextChanged.connect(self._on_infer_batch_changed)
        infer_batch_layout.addWidget(self._infer_batch_combo)
        infer_batch_action = QWidgetAction(self)
        infer_batch_action.setDefaultWidget(infer_batch_widget)
        cnn_menu.addAction(infer_batch_action)
        
        cnn_menu.addSeparator()
        
        cnn_run_inference_action = cnn_menu.addAction("Run Inference")
        cnn_run_inference_action.triggered.connect(self.cnn_run_inference_action)
        
        cnn_sort_menu = cnn_menu.addMenu("Sort by Confidence")
        # Populated dynamically when model is trained
        self._cnn_sort_menu = cnn_sort_menu
        
        cnn_filter_menu = cnn_menu.addMenu("Filter by Prediction")
        self._cnn_filter_menu = cnn_filter_menu
        
        # Confidence cutoff spinbox
        cutoff_widget = QWidget()
        cutoff_layout = QHBoxLayout(cutoff_widget)
        cutoff_layout.setContentsMargins(8, 2, 8, 2)
        cutoff_layout.addWidget(QLabel("Confidence Cutoff:"))
        self._confidence_cutoff_spin = QDoubleSpinBox()
        self._confidence_cutoff_spin.setRange(0.0, 1.0)
        self._confidence_cutoff_spin.setSingleStep(0.05)
        self._confidence_cutoff_spin.setDecimals(2)
        self._confidence_cutoff_spin.setValue(self.confidence_cutoff)
        self._confidence_cutoff_spin.setToolTip(
            "Hide images below this confidence when sorting by confidence"
        )
        self._confidence_cutoff_spin.valueChanged.connect(self._on_confidence_cutoff_changed)
        cutoff_layout.addWidget(self._confidence_cutoff_spin)
        cutoff_action = QWidgetAction(self)
        cutoff_action.setDefaultWidget(cutoff_widget)
        cnn_menu.addAction(cutoff_action)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Create splitter for left/center/right panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Image lists (unlabeled and labeled)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Unlabeled images section
        unlabeled_header = QLabel("Unlabeled Images")
        unlabeled_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        left_layout.addWidget(unlabeled_header)
        
        self.unlabeled_list = QListWidget()
        left_layout.addWidget(self.unlabeled_list)
        
        # Labeled images section
        labeled_header = QLabel("Labeled Images")
        labeled_header.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 10px;")
        left_layout.addWidget(labeled_header)
        
        self.labeled_list = QListWidget()
        left_layout.addWidget(self.labeled_list)
        
        splitter.addWidget(left_panel)
        
        # Center panel - Image grid with navigation
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_panel.setLayout(center_layout)
        
        # Create scroll area for image grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(False)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Image grid
        self.image_grid = ImageGridWidget(self.grid_cols, self.grid_rows)
        self.image_grid.main_window = self  # Set reference to main window
        scroll_area.setWidget(self.image_grid)
        
        # Install event filter for custom scrolling
        scroll_area.viewport().installEventFilter(self)
        self.scroll_area = scroll_area
        
        center_layout.addWidget(scroll_area)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        center_layout.addLayout(nav_layout)
        
        self.prev_btn = QPushButton("← Previous")
        self.prev_btn.clicked.connect(self.prev_page)
        nav_layout.addWidget(self.prev_btn)
        
        self.page_label = QLabel("Page 0 / 0")
        self.page_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.page_label)
        
        self.next_btn = QPushButton("Next →")
        self.next_btn.clicked.connect(self.next_page)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        nav_layout.addWidget(zoom_label)
        
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        nav_layout.addWidget(self.zoom_out_btn)
        
        self.zoom_level_label = QLabel("100%")
        self.zoom_level_label.setMinimumWidth(50)
        self.zoom_level_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.zoom_level_label)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        nav_layout.addWidget(self.zoom_in_btn)
        
        splitter.addWidget(center_panel)
        
        # Right panel - Classification Area
        self.classification_panel = ClassificationPanel(self)
        splitter.addWidget(self.classification_panel)
        
        # Set splitter proportions (20% left, 50% center, 30% right)
        splitter.setSizes([240, 600, 360])
    
    def open_folder_dialog(self):
        """Open folder dialog and load images."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder with Images",
            "",
            QFileDialog.ShowDirsOnly
        )
        
        if folder_path:
            # Create progress dialog
            progress = QProgressDialog("Scanning folder for images...", "Cancel", 0, 100, self)
            progress.setWindowTitle("Loading Images")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            # Progress callback
            def update_progress(current, total, filename):
                if progress.wasCanceled():
                    return
                progress.setMaximum(total)
                progress.setValue(current)
                progress.setLabelText(f"Scanning folder for images...\n{filename}")
            
            # Load folder with progress
            count, folder = self.controller.load_folder(folder_path, update_progress)
            progress.close()
            
            self.custom_sort_order = None  # Clear custom sort when loading new folder
            self.display_images()
    
    def open_grid_settings_dialog(self):
        """Open the grid size settings dialog."""
        dialog = SettingsDialog(self, self.grid_cols, self.grid_rows)
        if dialog.exec_():
            cols, rows = dialog.get_grid_size()
            self.grid_cols = cols
            self.grid_rows = rows
            self.image_grid.set_grid_size(cols, rows)
            self.update_page_label()
    
    def open_binary_mode_dialog(self):
        """Open the binary mode settings dialog."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QLineEdit, QLabel, QPushButton, QGroupBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Binary Mode Settings")
        dialog.setModal(True)
        dialog.resize(400, 250)
        
        layout = QVBoxLayout()
        dialog.setLayout(layout)
        
        # Binary mode checkbox
        mode_group = QGroupBox("Classification Mode")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)
        
        binary_checkbox = QCheckBox("Enable Binary Mode")
        binary_checkbox.setChecked(self.binary_mode)
        binary_checkbox.setToolTip("Enable binary classification with only 2 classes")
        mode_layout.addWidget(binary_checkbox)
        
        help_label = QLabel("When enabled, only 2 classes will be available for classification.")
        help_label.setStyleSheet("font-size: 10px; color: gray;")
        help_label.setWordWrap(True)
        mode_layout.addWidget(help_label)
        
        layout.addWidget(mode_group)
        
        # Class name inputs
        labels_group = QGroupBox("Class Labels")
        labels_layout = QVBoxLayout()
        labels_group.setLayout(labels_layout)
        
        pos_label = QLabel("Positive Class Name:")
        labels_layout.addWidget(pos_label)
        positive_input = QLineEdit()
        positive_input.setText(self.binary_positive_class)
        positive_input.setPlaceholderText("Enter positive class name...")
        labels_layout.addWidget(positive_input)
        
        neg_label = QLabel("Negative Class Name:")
        neg_label.setStyleSheet("margin-top: 10px;")
        labels_layout.addWidget(neg_label)
        negative_input = QLineEdit()
        negative_input.setText(self.binary_negative_class)
        negative_input.setPlaceholderText("Enter negative class name...")
        labels_layout.addWidget(negative_input)
        
        layout.addWidget(labels_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        # Execute dialog
        if dialog.exec_():
            new_binary_mode = binary_checkbox.isChecked()
            new_positive = positive_input.text().strip() or "Positive Class"
            new_negative = negative_input.text().strip() or "Negative Class"
            
            # Update settings
            mode_changed = new_binary_mode != self.binary_mode
            self.binary_mode = new_binary_mode
            self.binary_positive_class = new_positive
            self.binary_negative_class = new_negative
            
            # Update classification panel
            self.update_classification_panel_mode()
            
            # Show confirmation message if mode changed
            if mode_changed:
                from PyQt5.QtWidgets import QMessageBox
                if new_binary_mode:
                    # Show warning about auto-labeling behavior (only once)
                    if not self.binary_mode_warning_shown:
                        QMessageBox.warning(
                            self,
                            "Binary Mode Behavior",
                            f"Binary Mode Enabled\n\n"
                            f"Classification panel now shows only 2 labels:\n"
                            f"'{new_positive}' and '{new_negative}'\n\n"
                            f"IMPORTANT: When labeling images in binary mode, all unlabeled images "
                            f"on the current page will be automatically labeled as '{new_negative}' "
                            f"and moved to labeled images.\n\n"
                            f"This ensures efficient binary classification by assuming unlabeled "
                            f"images are negative examples."
                        )
                        self.binary_mode_warning_shown = True
                    else:
                        QMessageBox.information(
                            self,
                            "Binary Mode Enabled",
                            f"Classification panel now shows only 2 labels:\n'{new_positive}' and '{new_negative}'"
                        )
                else:
                    QMessageBox.information(
                        self,
                        "Binary Mode Disabled",
                        "Classification panel restored to multi-label mode."
                    )
    
    def update_classification_panel_mode(self):
        """Update the classification panel based on binary mode setting."""
        if hasattr(self, 'classification_panel'):
            self.classification_panel.set_binary_mode(
                self.binary_mode,
                self.binary_positive_class,
                self.binary_negative_class
            )
    
    def open_thread_settings_dialog(self):
        """Open the thread count settings dialog."""
        from PyQt5.QtWidgets import QInputDialog
        thread_count, ok = QInputDialog.getInt(
            self,
            "Thread Count Settings",
            "Number of threads for parallel image processing:\n(Recommended: 4-16)",
            self.thread_count,
            1,  # minimum
            32,  # maximum
            1   # step
        )
        if ok:
            self.thread_count = thread_count
            self.image_grid.set_thread_count(thread_count)
            print(f"Thread count set to: {thread_count}")
    
    def toggle_preload_images(self):
        """Toggle preloading all images into memory."""
        from PyQt5.QtWidgets import QMessageBox
        
        self.preload_images = self.preload_images_action.isChecked()
        
        if self.preload_images:
            # Warn about memory usage
            all_images = self.controller.get_images()
            num_images = len(all_images)
            estimated_mb = (num_images * 280 * 280) / (1024 * 1024)  # Rough estimate
            
            reply = QMessageBox.question(
                self,
                "Preload Images",
                f"This will load all {num_images} images into memory.\n"
                f"Estimated memory usage: ~{estimated_mb:.0f} MB\n\n"
                f"This may take a few minutes but will significantly speed up navigation.\n\n"
                f"Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # Start preloading
                progress = QProgressDialog("Preloading images into memory...", "Cancel", 0, num_images, self)
                progress.setWindowTitle("Loading Images")
                progress.setWindowModality(Qt.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                
                success = self.image_grid.preload_all_images(all_images, progress)
                progress.close()
                
                if success:
                    QMessageBox.information(
                        self,
                        "Preload Complete",
                        f"Successfully preloaded {num_images} images into memory."
                    )
                else:
                    self.preload_images = False
                    self.preload_images_action.setChecked(False)
            else:
                self.preload_images = False
                self.preload_images_action.setChecked(False)
        else:
            # Clear preloaded images
            self.image_grid.clear_preloaded_images()
            print("Preloaded images cleared from memory.")
    
    def toggle_force_grayscale(self):
        """Toggle forcing grayscale conversion for all images."""
        self.force_grayscale = self.force_grayscale_action.isChecked()
        self.image_grid.set_force_grayscale(self.force_grayscale)
        self.image_grid.load_page(self.image_grid.current_page)
    
    def open_colormap_dialog(self):
        """Open the colormap settings dialog."""
        dialog = ColormapDialog(self, self.colormap)
        if dialog.exec_():
            colormap = dialog.get_colormap()
            self.colormap = colormap
            self.image_grid.set_colormap(colormap)
            self.image_grid.load_page(self.image_grid.current_page)
            self.update_page_label()
    
    def display_images(self):
        """Display loaded images in the list widgets and grid."""
        self.unlabeled_list.clear()
        self.labeled_list.clear()
        
        all_images = self.controller.get_images()
        
        # Filter images based on view mode
        filtered_images = self.view_manager.filter_images(all_images, self.labeled_images)
        
        # Populate the lists
        for img_path in all_images:
            self.add_image_to_list(img_path)
        
        # Load filtered images into grid
        self.image_grid.set_images(filtered_images)
        self.image_grid.set_image_labels(self.labeled_images)
        self.image_grid.set_label_colors(self.label_colors)
        self.update_page_label()
    
    def add_image_to_list(self, image_path: Path):
        """Add an image to the appropriate list widget."""
        # Check if image is labeled
        if str(image_path) in self.labeled_images:
            # Add to labeled list with label
            label = self.labeled_images[str(image_path)]
            item = QListWidgetItem(f"{image_path.name} [{label}]")
            item.setData(Qt.UserRole, str(image_path))
            self.labeled_list.addItem(item)
        else:
            # Add to unlabeled list
            item = QListWidgetItem(image_path.name)
            item.setData(Qt.UserRole, str(image_path))
            self.unlabeled_list.addItem(item)
    
    def prev_page(self):
        """Go to previous page."""
        # Check if transform is active and create progress dialog
        if self.current_transform != 'none':
            target_page = self.image_grid.current_page - 1
            if target_page >= 0:
                start_idx = target_page * self.image_grid.images_per_page
                end_idx = start_idx + self.image_grid.images_per_page
                page_images = self.image_grid.all_images[start_idx:end_idx]
                num_images = len(page_images)
                
                if num_images > 0:
                    progress = QProgressDialog(f"Applying {self.current_transform} transform...", "Cancel", 0, num_images, self)
                    progress.setWindowTitle("Applying Transform")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)
                    
                    self.image_grid.load_page(target_page, progress_dialog=progress)
                    progress.close()
                    self.update_page_label()
                    return
        
        self.image_grid.prev_page()
        self.update_page_label()
    
    def next_page(self):
        """Go to next page."""
        # Check if transform is active and create progress dialog
        if self.current_transform != 'none':
            target_page = self.image_grid.current_page + 1
            max_page = (len(self.image_grid.all_images) - 1) // self.image_grid.images_per_page
            if target_page <= max_page:
                start_idx = target_page * self.image_grid.images_per_page
                end_idx = start_idx + self.image_grid.images_per_page
                page_images = self.image_grid.all_images[start_idx:end_idx]
                num_images = len(page_images)
                
                if num_images > 0:
                    progress = QProgressDialog(f"Applying {self.current_transform} transform...", "Cancel", 0, num_images, self)
                    progress.setWindowTitle("Applying Transform")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)
                    
                    self.image_grid.load_page(target_page, progress_dialog=progress)
                    progress.close()
                    self.update_page_label()
                    return
        
        self.image_grid.next_page()
        self.update_page_label()
    
    def update_page_label(self):
        """Update the page label."""
        current = self.image_grid.get_current_page() + 1
        total = self.image_grid.get_total_pages()
        self.page_label.setText(f"Page {current} / {total}")
        
        # Enable/disable navigation buttons
        self.prev_btn.setEnabled(current > 1)
        self.next_btn.setEnabled(current < total)
    
    def zoom_in(self):
        """Increase image size."""
        self.zoom_level = min(500, self.zoom_level + 25)
        self.image_grid.set_zoom(self.zoom_level)
        self.update_zoom_label()
    
    def zoom_out(self):
        """Decrease image size."""
        self.zoom_level = max(50, self.zoom_level - 25)
        self.image_grid.set_zoom(self.zoom_level)
        self.update_zoom_label()
    
    def update_zoom_label(self):
        """Update the zoom level label."""
        percentage = int((self.zoom_level / 150) * 100)
        self.zoom_level_label.setText(f"{percentage}%")
    
    def set_transform(self, transform_name: str):
        """Set the current image transformation and refresh display."""
        self.current_transform = transform_name
        self.image_grid.set_transform(transform_name)
        
        # Get number of images on current page
        start_idx = self.image_grid.current_page * self.image_grid.images_per_page
        end_idx = start_idx + self.image_grid.images_per_page
        page_images = self.image_grid.all_images[start_idx:end_idx]
        num_images = len(page_images)
        
        if num_images > 0 and transform_name != 'none':
            # Create progress dialog for transforms
            progress = QProgressDialog(f"Applying {transform_name} transform...", "Cancel", 0, num_images, self)
            progress.setWindowTitle("Applying Transform")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            
            self.image_grid.load_page(self.image_grid.current_page, progress_dialog=progress)
            progress.close()
        else:
            self.image_grid.load_page(self.image_grid.current_page)
        
        print(f"Transform set to: {transform_name}")
    
    def start_correlation_mode(self, method: str = 'ncc'):
        """Start correlation mode - user will select a base image."""
        self.correlation_mode = True
        self.correlation_method = method
        self.base_image_for_correlation = None
        self.image_grid.set_correlation_mode(True)
        self.setWindowTitle(f"Classifier Organizer - SELECT BASE IMAGE ({method.upper()})")
        print(f"Correlation mode started with method: {method}. Click on an image in the grid to use as base.")
    
    def cancel_correlation_mode(self):
        """Cancel correlation mode."""
        self.correlation_mode = False
        self.base_image_for_correlation = None
        self.image_grid.set_correlation_mode(False)
        self.setWindowTitle("Classifier Organizer")
        print("Correlation mode cancelled.")
    
    def on_base_image_selected(self, image_path: Path):
        """Called when user selects a base image for correlation."""
        from src.utils.image_utils import (
            compute_image_correlation, compute_ssim, compute_histogram_correlation,
            compute_chi_square_distance, compute_bhattacharyya_distance, compute_emd,
            compute_mae, compute_cosine_similarity, compute_mutual_information,
            compute_hog_similarity, compute_perceptual_hash, compute_difference_hash,
            compute_sift_similarity
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        self.base_image_for_correlation = image_path
        self.correlation_mode = False
        self.image_grid.set_correlation_mode(False)
        self.setWindowTitle("Classifier Organizer - Computing correlations...")
        
        print(f"Base image selected: {image_path}")
        print(f"Computing {self.correlation_method} with all unlabeled images (transform: {self.current_transform})...")
        
        # Get image cache from image_grid
        image_cache = self.image_grid.image_cache if self.image_grid.image_cache else None
        
        # Pre-compute base image statistics for optimization
        from src.utils.image_utils import _load_image
        from src.utils.image_transforms import apply_transform
        import numpy as np
        
        base_sift_data = None
        base_histogram = None
        base_hog = None
        
        print(f"Pre-computing base image statistics for {self.correlation_method}...")
        
        try:
            # Load base image once
            base_img = _load_image(image_path, image_cache)
            base_arr = np.array(base_img, dtype=np.uint8)
            
            if self.current_transform != 'none':
                base_arr = apply_transform(base_arr, self.current_transform)
            
            # Pre-compute based on method
            if self.correlation_method == 'sift':
                from skimage.feature import SIFT
                base_arr_sift = base_arr.astype(np.float32) / 255.0
                base_sift = SIFT(n_octaves=3, n_scales=3, sigma_min=1.6)
                base_sift.detect_and_extract(base_arr_sift)
                base_sift_data = (base_sift.keypoints, base_sift.descriptors)
                print(f"Extracted {len(base_sift.keypoints)} SIFT keypoints")
                
            elif self.correlation_method in ['hist_corr', 'bhattacharyya']:
                hist, _ = np.histogram(base_arr.flatten(), bins=256, range=(0, 256))
                base_histogram = hist.astype(np.float32) / hist.sum()
                print("Computed normalized histogram")
                
            elif self.correlation_method == 'chi_square':
                hist, _ = np.histogram(base_arr.flatten(), bins=256, range=(0, 256))
                base_histogram = hist.astype(np.float32) + 1e-10
                print("Computed histogram with epsilon")
                
            elif self.correlation_method == 'hog':
                gx = np.gradient(base_arr.astype(float), axis=1)
                gy = np.gradient(base_arr.astype(float), axis=0)
                mag = np.sqrt(gx**2 + gy**2)
                ori = np.arctan2(gy, gx)
                hist, _ = np.histogram(ori.flatten(), bins=9, range=(-np.pi, np.pi), weights=mag.flatten())
                base_hog = hist / (hist.sum() + 1e-10)
                print("Computed HOG histogram")
                
        except Exception as e:
            print(f"Warning: Failed to pre-compute base statistics: {e}")
            # Continue without pre-computation
        
        # Map method names to functions with pre-computed data
        method_map = {
            'ncc': lambda p1, p2: compute_image_correlation(p1, p2, method='ncc', transform=self.current_transform, image_cache=image_cache),
            'mse': lambda p1, p2: compute_image_correlation(p1, p2, method='mse', transform=self.current_transform, image_cache=image_cache),
            'ssim': lambda p1, p2: compute_ssim(p1, p2),
            'mae': lambda p1, p2: compute_mae(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'cosine': lambda p1, p2: compute_cosine_similarity(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'hist_corr': lambda p1, p2: compute_histogram_correlation(p1, p2, transform=self.current_transform, image_cache=image_cache, base_histogram=base_histogram),
            'chi_square': lambda p1, p2: compute_chi_square_distance(p1, p2, transform=self.current_transform, image_cache=image_cache, base_histogram=base_histogram),
            'bhattacharyya': lambda p1, p2: compute_bhattacharyya_distance(p1, p2, transform=self.current_transform, image_cache=image_cache, base_histogram=base_histogram),
            'emd': lambda p1, p2: compute_emd(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'mutual_info': lambda p1, p2: compute_mutual_information(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'hog': lambda p1, p2: compute_hog_similarity(p1, p2, transform=self.current_transform, image_cache=image_cache, base_hog=base_hog),
            'phash': lambda p1, p2: compute_perceptual_hash(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'dhash': lambda p1, p2: compute_difference_hash(p1, p2, transform=self.current_transform, image_cache=image_cache),
            'sift': lambda p1, p2: compute_sift_similarity(p1, p2, transform=self.current_transform, image_cache=image_cache, base_sift_data=base_sift_data),
        }
        
        compute_func = method_map.get(self.correlation_method, method_map['ncc'])
        
        # Get all unlabeled images
        all_images = self.controller.get_images()
        unlabeled_images = [img for img in all_images if str(img) not in self.labeled_images]
        
        # Create progress dialog
        progress = QProgressDialog(f"Computing {self.correlation_method.upper()} similarity...", "Cancel", 0, len(unlabeled_images) - 1, self)
        progress.setWindowTitle("Sorting Images")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Helper function for parallel correlation computation
        def compute_correlation_for_image(img_path):
            """Compute correlation score for a single image."""
            if img_path != image_path:  # Don't compare with itself
                score = compute_func(image_path, img_path)
                return img_path, score
            return img_path, None
        
        # Compute correlation scores in parallel
        correlation_scores = {}
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            # Submit all tasks
            futures = {
                executor.submit(compute_correlation_for_image, img_path): img_path
                for img_path in unlabeled_images
            }
            
            # Process completed tasks as they finish
            for future in as_completed(futures):
                if progress.wasCanceled():
                    executor.shutdown(wait=False, cancel_futures=True)
                    self.setWindowTitle("Classifier Organizer")
                    return
                
                img_path, score = future.result()
                
                if score is not None:
                    correlation_scores[str(img_path)] = score
                
                processed += 1
                progress.setValue(processed)
                progress.setLabelText(f"Computing {self.correlation_method.upper()} similarity...\nProcessed {processed}/{len(unlabeled_images)} images")
        
        progress.close()
        
        # Sort unlabeled images by correlation score (highest first)
        sorted_unlabeled = sorted(unlabeled_images, 
                                 key=lambda x: correlation_scores.get(str(x), -999), 
                                 reverse=True)
        
        # Store the custom sort order
        self.custom_sort_order = sorted_unlabeled
        
        # Update the display with sorted images
        self.display_images_with_custom_order(sorted_unlabeled)
        
        self.setWindowTitle("Classifier Organizer - Sorted by correlation")
    
    def _on_confidence_cutoff_changed(self, value):
        """Update confidence cutoff and re-sort if active."""
        self.confidence_cutoff = value
        if hasattr(self, '_cnn_sort_class') and self.cnn_prediction is not None:
            self.cnn_sort_by_confidence(self._cnn_sort_class)

    def _on_train_batch_changed(self, text):
        """Update train batch size from combo box."""
        try:
            value = int(text)
            if value > 0:
                self.train_batch_size = value
        except ValueError:
            pass
    
    def _on_infer_batch_changed(self, text):
        """Update inference batch size from combo box."""
        try:
            value = int(text)
            if value > 0:
                self.inference_batch_size = value
        except ValueError:
            pass
    
    def cnn_train(self):
        """Train a ResNet18 CNN on labeled images."""
        from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QInputDialog
        
        try:
            from src.cnn.workers import TrainingWorker
        except ImportError:
            QMessageBox.warning(
                self, "PyTorch Not Installed",
                "PyTorch is required for CNN features.\n\n"
                "Install it with:\n  pip install torch torchvision"
            )
            return
        
        # Get class names from ontology
        class_names = self.classification_panel.get_ontology_labels()
        if len(class_names) < 2:
            QMessageBox.warning(
                self, "Not Enough Classes",
                "At least 2 ontology labels are required to train a CNN."
            )
            return
        
        # Check we have enough labeled images
        labels_with_images = set(self.labeled_images.values())
        classes_with_data = [c for c in class_names if c in labels_with_images]
        if len(classes_with_data) < 2:
            QMessageBox.warning(
                self, "Not Enough Labeled Data",
                "At least 2 classes need labeled images to train.\n\n"
                f"Classes with labels: {', '.join(classes_with_data) if classes_with_data else 'none'}"
            )
            return
        
        num_labeled = len(self.labeled_images)
        
        # Determine if we're continuing training
        resuming = self.cnn_result is not None
        resume_text = " (continuing from previous)" if resuming else ""
        
        # Ask for number of epochs
        epochs, ok = QInputDialog.getInt(
            self, "Training Epochs",
            f"Training{resume_text} on {num_labeled} labeled images across "
            f"{len(classes_with_data)} classes.\n\nNumber of epochs:",
            10, 1, 100, 1
        )
        if not ok:
            return
        
        # Create progress dialog
        progress = QProgressDialog("Training CNN...", "Cancel", 0, epochs, self)
        progress.setWindowTitle("CNN Training")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        # Get image cache if available
        image_cache = self.image_grid.image_cache if self.image_grid.image_cache else None
        
        # Create and start background worker
        worker = TrainingWorker(
            labeled_images=self.labeled_images,
            class_names=class_names,
            image_cache=image_cache,
            epochs=epochs,
            batch_size=self.train_batch_size,
            resume_from=self.cnn_result,
        )
        # Keep a reference so the thread isn't garbage-collected
        self._training_worker = worker
        
        def on_progress(epoch, total_epochs, loss, accuracy):
            progress.setValue(epoch)
            progress.setLabelText(
                f"Training CNN...\n"
                f"Epoch {epoch}/{total_epochs}\n"
                f"Loss: {loss:.4f}  Accuracy: {accuracy:.1%}"
            )
        
        def on_cancel():
            worker.cancel()
        
        def on_finished(result):
            progress.close()
            self._training_worker = None
            
            if result is None:
                return
            
            self.cnn_result = result
            self.cnn_prediction = None  # Clear cached inference (model changed)
            
            # Auto-save model next to the image folder
            try:
                current_folder = self.controller.get_current_folder()
                if current_folder:
                    model_path = Path(current_folder).parent / "cnn_model.pth"
                else:
                    model_path = Path("cnn_model.pth")
                self.cnn_model_path = str(result.save(model_path))
            except Exception as e:
                print(f"WARNING: Failed to auto-save CNN model: {e}")
            
            # Update the sort-by-confidence submenu with class names
            self._cnn_sort_menu.clear()
            self._cnn_filter_menu.clear()
            for class_name in class_names:
                action = self._cnn_sort_menu.addAction(class_name)
                action.triggered.connect(
                    lambda checked, cn=class_name: self.cnn_sort_by_confidence(cn)
                )
                filter_action = self._cnn_filter_menu.addAction(class_name)
                filter_action.triggered.connect(
                    lambda checked, cn=class_name: self.cnn_filter_by_prediction(cn)
                )
            
            QMessageBox.information(
                self, "Training Complete",
                f"CNN trained successfully!\n\n"
                f"Final accuracy: {result.train_accuracy:.1%}\n"
                f"Epochs: {result.epochs}\n"
                f"Classes: {', '.join(class_names)}\n\n"
                f"You can now use 'Sort by Confidence' or 'Suggest Labels' from the CNN menu."
            )
        
        def on_error(error_msg):
            progress.close()
            self._training_worker = None
            QMessageBox.critical(self, "Training Error", f"CNN training failed:\n{error_msg}")
        
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(on_cancel)
        
        worker.start()
    
    def _cnn_run_inference(self, on_complete=None):
        """Run CNN inference on unlabeled images in a background thread.

        Args:
            on_complete: Optional callback(prediction) called when done.
                         If None, only caches the result.
        """
        from PyQt5.QtWidgets import QMessageBox, QProgressDialog
        from src.cnn.workers import InferenceWorker
        
        if self.cnn_result is None:
            QMessageBox.warning(
                self, "No Trained Model",
                "Train a CNN first using CNN \u2192 Train on Labeled Images."
            )
            return
        
        # Get unlabeled images
        all_images = self.controller.get_images()
        unlabeled_images = [img for img in all_images
                           if str(img) not in self.labeled_images]
        
        if not unlabeled_images:
            QMessageBox.information(self, "No Unlabeled Images",
                                   "All images are already labeled.")
            return
        
        # Create progress dialog
        progress = QProgressDialog("Running CNN inference...", "Cancel",
                                   0, len(unlabeled_images), self)
        progress.setWindowTitle("CNN Inference")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        image_cache = self.image_grid.image_cache if self.image_grid.image_cache else None
        
        worker = InferenceWorker(
            training_result=self.cnn_result,
            image_paths=unlabeled_images,
            image_cache=image_cache,
            batch_size=self.inference_batch_size,
        )
        self._inference_worker = worker
        
        def on_progress(processed, total):
            progress.setValue(processed)
            progress.setLabelText(
                f"Running CNN inference...\n"
                f"Processed {processed}/{total} images"
            )
        
        def on_cancel():
            worker.cancel()
        
        def on_finished(prediction):
            progress.close()
            self._inference_worker = None
            if prediction is not None:
                self.cnn_prediction = prediction
            if on_complete is not None:
                on_complete(prediction)
        
        def on_error(error_msg):
            progress.close()
            self._inference_worker = None
            QMessageBox.critical(self, "Inference Error", f"CNN inference failed:\n{error_msg}")
        
        worker.progress.connect(on_progress)
        worker.finished.connect(on_finished)
        worker.error.connect(on_error)
        progress.canceled.connect(on_cancel)
        
        worker.start()
    
    def cnn_run_inference_action(self):
        """Menu action to explicitly run inference and cache results."""
        from PyQt5.QtWidgets import QMessageBox
        
        def on_complete(prediction):
            if prediction is not None:
                count = len(prediction.predictions)
                QMessageBox.information(
                    self, "Inference Complete",
                    f"Inference finished on {count} unlabeled images.\n\n"
                    f"You can now use 'Sort by Confidence' or 'Suggest Labels' "
                    f"without re-running inference."
                )
        
        self._cnn_run_inference(on_complete=on_complete)
    
    def cnn_filter_by_prediction(self, class_name: str):
        """Show only images whose predicted class matches class_name."""
        def _apply_filter(prediction=None):
            predictions = self.cnn_prediction.predictions
            filtered = []
            for path_str, pred in predictions.items():
                if pred['predicted_label'] == class_name and pred['confidence'] >= self.confidence_cutoff:
                    filtered.append((Path(path_str), pred['confidence']))
            
            # Sort by confidence descending
            filtered.sort(key=lambda x: x[1], reverse=True)
            sorted_paths = [path for path, _ in filtered]
            
            if not sorted_paths:
                from PyQt5.QtWidgets import QMessageBox
                cutoff_text = (
                    f" above {self.confidence_cutoff:.0%} confidence"
                    if self.confidence_cutoff > 0 else ""
                )
                QMessageBox.information(
                    self, "No Matches",
                    f"No unlabeled images are predicted as '{class_name}'{cutoff_text}."
                )
                return
            
            self.custom_sort_order = sorted_paths
            self.display_images_with_custom_order(sorted_paths)
            cutoff_text = (
                f", cutoff={self.confidence_cutoff:.0%}"
                if self.confidence_cutoff > 0 else ""
            )
            self.setWindowTitle(
                f"Classifier Organizer - Predicted as '{class_name}'{cutoff_text} "
                f"({len(sorted_paths)} images)"
            )
        
        if self.cnn_prediction is None:
            self._cnn_run_inference(on_complete=lambda pred: _apply_filter() if pred else None)
        else:
            _apply_filter()

    def cnn_sort_by_confidence(self, class_name: str):
        """Sort unlabeled images by CNN confidence for a given class."""
        if self.cnn_prediction is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "No Inference Results",
                "Run inference first using CNN \u2192 Run Inference."
            )
            return
        
        self._cnn_sort_class = class_name
        sorted_items = self.cnn_prediction.sort_by_class_confidence(class_name)
        sorted_paths = [path for path, conf in sorted_items
                        if conf >= self.confidence_cutoff]
        
        self.custom_sort_order = sorted_paths
        self.display_images_with_custom_order(sorted_paths)
        cutoff_text = (
            f", cutoff={self.confidence_cutoff:.0%}"
            if self.confidence_cutoff > 0 else ""
        )
        self.setWindowTitle(
            f"Classifier Organizer - Sorted by CNN confidence ({class_name}{cutoff_text})"
        )
    
    def display_images_with_custom_order(self, unlabeled_order: list):
        """Display images with a custom order for unlabeled images."""
        self.unlabeled_list.clear()
        self.labeled_list.clear()
        
        # Add unlabeled images in the specified order
        for img_path in unlabeled_order:
            item = QListWidgetItem(img_path.name)
            item.setData(Qt.UserRole, str(img_path))
            self.unlabeled_list.addItem(item)
        
        # Add labeled images
        all_images = self.controller.get_images()
        for img_path in all_images:
            if str(img_path) in self.labeled_images:
                label = self.labeled_images[str(img_path)]
                item = QListWidgetItem(f"{img_path.name} [{label}]")
                item.setData(Qt.UserRole, str(img_path))
                self.labeled_list.addItem(item)
        
        # Update grid with filtered images based on view mode
        filtered_images = self.view_manager.filter_images(all_images, self.labeled_images)
        
        # If showing unlabeled or all, use the custom order
        if self.view_manager.get_current_view_mode() in [ViewMode.UNLABELED, ViewMode.ALL]:
            # For unlabeled view, just show the sorted unlabeled images
            if self.view_manager.get_current_view_mode() == ViewMode.UNLABELED:
                filtered_images = unlabeled_order
            else:
                # For all view, combine sorted unlabeled with labeled
                labeled_images = [img for img in all_images if str(img) in self.labeled_images]
                filtered_images = unlabeled_order + labeled_images
        
        self.image_grid.set_images(filtered_images)
        self.image_grid.set_image_labels(self.labeled_images)
        self.image_grid.set_label_colors(self.label_colors)
        self.update_page_label()
    
    def eventFilter(self, obj, event):
        """Handle custom scroll events for Ctrl+Wheel horizontal scrolling."""
        if obj == self.scroll_area.viewport() and event.type() == event.Wheel:
            wheel_event = event
            if wheel_event.modifiers() == Qt.ControlModifier:
                # Ctrl+Wheel: scroll horizontally
                h_bar = self.scroll_area.horizontalScrollBar()
                delta = -wheel_event.angleDelta().y()
                h_bar.setValue(h_bar.value() + delta)
                return True  # Event handled
        
        return super().eventFilter(obj, event)
    
    def label_selected_images_with_label(self, label: str):
        """Label the currently selected images in the grid."""
        print(f"DEBUG: label_selected_images_with_label called with label={label}")
        
        # Get pending label assignments from the grid
        pending_assignments = self.image_grid.get_pending_label_assignments()
        print(f"DEBUG: Pending assignments: {pending_assignments}")
        
        if not pending_assignments:
            from PyQt5.QtWidgets import QMessageBox
            print("DEBUG: No images selected, showing warning")
            QMessageBox.warning(
                self,
                "No Images Selected",
                "Please select one or more images in the grid first."
            )
            return
        
        # In binary mode, automatically label remaining unlabeled images on current page as negative
        auto_labeled_count = 0
        if self.binary_mode:
            # Get all images on the current page
            start_idx = self.image_grid.current_page * self.image_grid.images_per_page
            end_idx = start_idx + self.image_grid.images_per_page
            page_images = self.image_grid.all_images[start_idx:end_idx]
            
            # Find unlabeled images that aren't in pending assignments
            for img_path in page_images:
                img_path_str = str(img_path)
                if img_path_str not in self.labeled_images and img_path_str not in pending_assignments:
                    # Auto-label with negative class
                    if self.binary_negative_class not in self.label_colors:
                        self.label_colors[self.binary_negative_class] = self.generate_color_for_label(len(self.label_colors))
                    
                    self.labeled_images[img_path_str] = self.binary_negative_class
                    auto_labeled_count += 1
                    print(f"DEBUG: Auto-labeled {img_path_str} as {self.binary_negative_class}")
        
        # Apply all pending label assignments
        labels_used = set()
        for img_path_str, assigned_label in pending_assignments.items():
            # Assign color to label if not already assigned
            if assigned_label not in self.label_colors:
                self.label_colors[assigned_label] = self.generate_color_for_label(len(self.label_colors))
            
            self.labeled_images[img_path_str] = assigned_label
            labels_used.add(assigned_label)
            print(f"DEBUG: Labeled {img_path_str} as {assigned_label}")
        
        # Show success message
        from PyQt5.QtWidgets import QMessageBox
        if len(labels_used) == 1:
            label_text = f"'{list(labels_used)[0]}'"
        else:
            label_text = f"{len(labels_used)} different labels"
        
        message = f"Successfully labeled {len(pending_assignments)} image(s) with {label_text}."
        if auto_labeled_count > 0:
            message += f"\n\nAuto-labeled {auto_labeled_count} remaining unlabeled image(s) on this page as '{self.binary_negative_class}'."
        
        QMessageBox.information(
            self,
            "Images Labeled",
            message
        )
        
        # Clear selections
        self.image_grid.clear_selections()
        
        # Refresh display
        print("DEBUG: Refreshing display")
        # Save current page to restore it after refresh
        current_page = self.image_grid.get_current_page()
        
        # Update custom sort order by removing labeled images
        if self.custom_sort_order is not None:
            # Remove newly labeled images from custom sort order
            self.custom_sort_order = [img for img in self.custom_sort_order if str(img) not in self.labeled_images]
            # Use custom order display
            self.display_images_with_custom_order(self.custom_sort_order)
        else:
            # Use normal display
            self.display_images()
        
        # Restore page position (or closest valid page if images were removed)
        total_pages = self.image_grid.get_total_pages()
        if total_pages > 0:
            target_page = min(current_page, total_pages - 1)
            self.image_grid.load_page(target_page)
            self.update_page_label()
    
    def generate_color_for_label(self, index: int) -> str:
        """Generate a unique color for a label based on its index."""
        # Predefined color palette for better visual distinction
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
        """Set the currently active label for image selection."""
        self.active_label = label
        print(f"DEBUG: Active label set to: {label}")
    
    def remove_image_label(self, img_path_str: str):
        """Remove the label from an image."""
        print(f"DEBUG: remove_image_label called for {img_path_str}")
        if img_path_str in self.labeled_images:
            old_label = self.labeled_images[img_path_str]
            print(f"DEBUG: Removing label '{old_label}' from image")
            del self.labeled_images[img_path_str]
            
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Label Removed",
                f"Removed label '{old_label}' from image."
            )
            
            # Refresh display to move image back to unlabeled
            print("DEBUG: Refreshing display after label removal")
            self.display_images()
        else:
            print(f"DEBUG: Image not found in labeled_images dictionary")
    
    def set_view_unlabeled(self):
        """Switch to showing only unlabeled images."""
        print("DEBUG: Switching to unlabeled images view")
        self.view_manager.set_view_mode(ViewMode.UNLABELED)
        self.view_unlabeled_action.setChecked(True)
        self.view_labeled_action.setChecked(False)
        self.view_all_action.setChecked(False)
        self.display_images()
    
    def set_view_labeled(self):
        """Switch to showing only labeled images."""
        print("DEBUG: Switching to labeled images view")
        self.view_manager.set_view_mode(ViewMode.LABELED)
        self.view_unlabeled_action.setChecked(False)
        self.view_labeled_action.setChecked(True)
        self.view_all_action.setChecked(False)
        self.display_images()
    
    def set_view_all(self):
        """Switch to showing all images."""
        print("DEBUG: Switching to all images view")
        self.view_manager.set_view_mode(ViewMode.ALL)
        self.view_unlabeled_action.setChecked(False)
        self.view_labeled_action.setChecked(False)
        self.view_all_action.setChecked(True)
        self.display_images()
    
    def export_to_json(self):
        """Export labeled images to JSON file."""
        print("DEBUG: Export to JSON triggered")
        self.export_manager.export_to_json(self.labeled_images, self.controller.get_current_folder())
    
    def export_to_csv(self):
        """Export labeled images to CSV file."""
        print("DEBUG: Export to CSV triggered")
        self.export_manager.export_to_csv(self.labeled_images, self.controller.get_current_folder())
    
    def export_organized_copy(self):
        """Export organized copy of labeled images."""
        print("DEBUG: Export organized copy triggered")
        self.export_manager.export_organized_copy(self.labeled_images)
    
    def save_progress(self):
        """Save current application state to JSON file."""
        print("DEBUG: Save progress triggered")
        
        try:
            # Collect all state data
            state_data = {
                "labeled_images": {k.replace('\\', '/'): v for k, v in self.labeled_images.items()},
                "label_colors": self.label_colors,
                "current_folder": str(self.controller.get_current_folder()).replace('\\', '/') if self.controller.get_current_folder() else None,
                "ontology_labels": self.classification_panel.get_ontology_labels(),
                "grid_cols": self.grid_cols,
                "grid_rows": self.grid_rows,
                "zoom_level": self.zoom_level,
                "colormap": self.colormap,
                "view_mode": self.view_manager.get_current_view_mode().value,
                "active_label": self.active_label,
                "current_transform": self.current_transform,
                "current_page": self.image_grid.current_page,
                "thread_count": self.thread_count,
                "preload_images": self.preload_images,
                "force_grayscale": self.force_grayscale,
                "custom_sort_order": [str(img).replace('\\', '/') for img in self.custom_sort_order] if self.custom_sort_order else None,
                "base_image_for_correlation": str(self.base_image_for_correlation).replace('\\', '/') if self.base_image_for_correlation else None,
                "correlation_method": self.correlation_method,
                "binary_mode": self.binary_mode,
                "binary_positive_class": self.binary_positive_class,
                "binary_negative_class": self.binary_negative_class,
                "cnn_model_path": self.cnn_model_path.replace('\\', '/') if self.cnn_model_path else None
            }
            
            # Save using progress manager
            success = self.progress_manager.save_progress(state_data)
            if success:
                print("DEBUG: Progress saved successfully")
        except Exception as e:
            print(f"ERROR: Failed to collect state data: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to collect application state:\n{str(e)}"
            )
    
    def quick_save_progress(self):
        """Quick save to the last used file without prompting."""
        print("DEBUG: Quick save progress triggered")
        
        try:
            # Collect all state data
            state_data = {
                "labeled_images": {k.replace('\\', '/'): v for k, v in self.labeled_images.items()},
                "label_colors": self.label_colors,
                "current_folder": str(self.controller.get_current_folder()).replace('\\', '/') if self.controller.get_current_folder() else None,
                "ontology_labels": self.classification_panel.get_ontology_labels(),
                "grid_cols": self.grid_cols,
                "grid_rows": self.grid_rows,
                "zoom_level": self.zoom_level,
                "colormap": self.colormap,
                "view_mode": self.view_manager.get_current_view_mode().value,
                "active_label": self.active_label,
                "current_transform": self.current_transform,
                "current_page": self.image_grid.current_page,
                "thread_count": self.thread_count,
                "preload_images": self.preload_images,
                "force_grayscale": self.force_grayscale,
                "custom_sort_order": [str(img).replace('\\', '/') for img in self.custom_sort_order] if self.custom_sort_order else None,
                "base_image_for_correlation": str(self.base_image_for_correlation).replace('\\', '/') if self.base_image_for_correlation else None,
                "correlation_method": self.correlation_method,
                "binary_mode": self.binary_mode,
                "binary_positive_class": self.binary_positive_class,
                "binary_negative_class": self.binary_negative_class,
                "cnn_model_path": self.cnn_model_path.replace('\\', '/') if self.cnn_model_path else None
            }
            
            # Quick save using progress manager
            success = self.progress_manager.quick_save(state_data)
            if success:
                print("DEBUG: Quick save completed")
        except Exception as e:
            print(f"ERROR: Failed to quick save: {e}")
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Quick Save Failed",
                f"Failed to quick save:\n{str(e)}"
            )
    
    def load_progress(self):
        """Load application state from JSON file."""
        print("DEBUG: Load progress triggered")
        
        # Load state data
        state_data = self.progress_manager.load_progress()
        
        if state_data is None:
            return  # User cancelled or load failed
        
        try:
            # Restore labeled images and label colors with validation
            loaded_labeled_images = state_data.get("labeled_images", {})
            if isinstance(loaded_labeled_images, dict):
                # Normalize paths to OS-native format (saved with forward slashes)
                self.labeled_images = {str(Path(k)): v for k, v in loaded_labeled_images.items()}
            else:
                print("WARNING: Invalid labeled_images format, using empty dict")
                self.labeled_images = {}
            
            loaded_label_colors = state_data.get("label_colors", {})
            if isinstance(loaded_label_colors, dict):
                self.label_colors = loaded_label_colors
            else:
                print("WARNING: Invalid label_colors format, using empty dict")
                self.label_colors = {}
            
            # Restore grid settings with validation
            self.grid_cols = max(1, state_data.get("grid_cols", 3))
            self.grid_rows = max(1, state_data.get("grid_rows", 4))
            self.zoom_level = max(50, min(300, state_data.get("zoom_level", 150)))
            self.colormap = state_data.get("colormap", "gray")
            
            # Apply grid size to the image grid widget
            self.image_grid.set_grid_size(self.grid_cols, self.grid_rows)
            
            # Apply zoom level to the image grid widget
            self.image_grid.set_zoom(self.zoom_level)
            
            # Apply colormap to the image grid widget
            self.image_grid.set_colormap(self.colormap)
            
            # Restore additional settings
            self.current_transform = state_data.get("current_transform", "none")
            self.thread_count = max(1, state_data.get("thread_count", 8))
            self.preload_images = state_data.get("preload_images", False)
            if hasattr(self, 'preload_images_action'):
                self.preload_images_action.setChecked(self.preload_images)
            
            # Restore force grayscale setting
            self.force_grayscale = state_data.get("force_grayscale", False)
            if hasattr(self, 'force_grayscale_action'):
                self.force_grayscale_action.setChecked(self.force_grayscale)
            self.image_grid.set_force_grayscale(self.force_grayscale)
            
            # Restore binary mode settings
            self.binary_mode = state_data.get("binary_mode", False)
            self.binary_positive_class = state_data.get("binary_positive_class", "Positive Class")
            self.binary_negative_class = state_data.get("binary_negative_class", "Negative Class")
            
            # Restore sorting state
            custom_sort_order_str = state_data.get("custom_sort_order", None)
            if custom_sort_order_str:
                self.custom_sort_order = [Path(img_str) for img_str in custom_sort_order_str]
            else:
                self.custom_sort_order = None
            
            base_image_str = state_data.get("base_image_for_correlation", None)
            if base_image_str:
                self.base_image_for_correlation = Path(base_image_str)
            else:
                self.base_image_for_correlation = None
            
            self.correlation_method = state_data.get("correlation_method", "ncc")
            
            # Restore active label
            self.active_label = state_data.get("active_label", None)
            
            # Restore ontology labels in classification panel
            ontology_labels = state_data.get("ontology_labels", [])
            if isinstance(ontology_labels, list):
                self.classification_panel.set_ontology_labels(ontology_labels, self.label_colors)
            else:
                print("WARNING: Invalid ontology_labels format")
            
            # Apply binary mode to classification panel
            self.update_classification_panel_mode()
            
            # Restore CNN model if available
            saved_cnn_path = state_data.get("cnn_model_path", None)
            if saved_cnn_path:
                cnn_path = Path(saved_cnn_path)
                if cnn_path.exists():
                    try:
                        from src.cnn.trainer import TrainingResult
                        self.cnn_result = TrainingResult.load(cnn_path)
                        self.cnn_model_path = str(cnn_path)
                        # Populate sort-by-confidence and filter menus
                        self._cnn_sort_menu.clear()
                        self._cnn_filter_menu.clear()
                        for class_name in self.cnn_result.class_names:
                            action = self._cnn_sort_menu.addAction(class_name)
                            action.triggered.connect(
                                lambda checked, cn=class_name: self.cnn_sort_by_confidence(cn)
                            )
                            filter_action = self._cnn_filter_menu.addAction(class_name)
                            filter_action.triggered.connect(
                                lambda checked, cn=class_name: self.cnn_filter_by_prediction(cn)
                            )
                        print(f"DEBUG: Restored CNN model from {cnn_path}")
                    except Exception as e:
                        print(f"WARNING: Failed to load CNN model: {e}")
                else:
                    print(f"WARNING: Saved CNN model path does not exist: {saved_cnn_path}")
            
            # Restore current folder and load images if available
            current_folder = state_data.get("current_folder", None)
            if current_folder:
                folder_path = Path(current_folder)
                if folder_path.exists():
                    # Create progress dialog
                    progress = QProgressDialog("Scanning folder for images...", "Cancel", 0, 100, self)
                    progress.setWindowTitle("Loading Images")
                    progress.setWindowModality(Qt.WindowModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)
                    
                    # Progress callback
                    def update_progress(current, total, filename):
                        if progress.wasCanceled():
                            return
                        progress.setMaximum(total)
                        progress.setValue(current)
                        progress.setLabelText(f"Scanning folder for images...\n{filename}")
                    
                    count, folder = self.controller.load_folder(str(folder_path), update_progress)
                    progress.close()
                    
                    print(f"DEBUG: Loaded {count} images from folder")
                else:
                    print(f"WARNING: Saved folder path does not exist: {current_folder}")
                    from PyQt5.QtWidgets import QMessageBox
                    QMessageBox.warning(
                        self,
                        "Folder Not Found",
                        f"The saved folder path does not exist:\n{current_folder}\n\nYou may need to open a folder manually."
                    )
            
            # Restore view mode
            view_mode_value = state_data.get("view_mode", ViewMode.UNLABELED.value)
            if view_mode_value == ViewMode.UNLABELED.value:
                self.set_view_unlabeled()
            elif view_mode_value == ViewMode.LABELED.value:
                self.set_view_labeled()
            else:
                self.set_view_all()
            
            # Update UI - display_images() handles clearing and repopulating the lists and grid
            # Use custom sort order if it exists
            if self.custom_sort_order:
                self.display_images_with_custom_order(self.custom_sort_order)
            else:
                self.display_images()
            
            # Restore the current page index
            saved_page = state_data.get("current_page", 0)
            if saved_page > 0 and saved_page < self.image_grid.get_total_pages():
                self.image_grid.load_page(saved_page)
                self.update_page_label()
            
            # Apply transform after displaying images
            # We call set_transform which will reload the current page with the transform
            if self.current_transform != "none":
                self.set_transform(self.current_transform)
            
            print("DEBUG: Progress loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to restore state: {e}")
            import traceback
            traceback.print_exc()
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Restore Failed",
                f"Failed to fully restore application state:\n{str(e)}"
            )


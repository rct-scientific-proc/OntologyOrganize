"""Application entry point."""

import sys

# Import torch before PyQt5 to avoid DLL conflicts on Windows
try:
    import torch
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication
from src.ui.main_window import MainWindow


def main():
    """Initialize and run the application."""
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Classifier Organizer")
    app.setOrganizationName("YourOrganization")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

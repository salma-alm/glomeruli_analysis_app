from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
)

from ui.widgets.image_viewer import ImageViewer


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Glomerulus Analysis")

        self.resize(1200, 800)

        # Widget central
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        # Layout principal
        layout = QVBoxLayout()

        central_widget.setLayout(layout)

        # Bouton
        self.select_button = QPushButton(
            "Sélectionner une image"
        )

        layout.addWidget(self.select_button)

        # Notre nouveau visualiseur
        self.image_viewer = ImageViewer()

        layout.addWidget(
            self.image_viewer,
            stretch=1
        )

        # Connexion du bouton
        self.select_button.clicked.connect(
            self.select_image
        )

    def select_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionner une image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )

        if file_path:

            self.image_viewer.load_image(
                file_path
            )
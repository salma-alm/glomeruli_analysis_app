import cv2
import numpy as np
import tifffile as tiff

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QWidget


class ImageViewer(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.image = None

        # Dimensions de l'image originale
        self.image_width = 0
        self.image_height = 0

        # Zone actuellement visible dans l'image originale
        self.x_start = 0
        self.y_start = 0
        self.x_end = 0
        self.y_end = 0

        # Gestion du déplacement
        self.is_dragging = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Image actuellement préparée pour l'affichage
        self.display_image = None

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ---------------------------------------------------------
    # CHARGEMENT
    # ---------------------------------------------------------

    def load_image(self, path):
        print("Chargement de l'image...")

        img = tiff.imread(path)

        print("Shape originale :", img.shape)

        # Cas (C, H, W)
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.transpose(img, (1, 2, 0))

        # Si image RGBA, on garde les 3 premiers canaux
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        # Conversion éventuelle vers uint8
        if img.dtype != np.uint8:
            img = cv2.normalize(
                img,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

        self.image = img

        self.image_height, self.image_width = img.shape[:2]

        print(
            "Image chargée :",
            self.image_width,
            "x",
            self.image_height
        )

        # Afficher toute l'image au départ
        self.x_start = 0
        self.y_start = 0
        self.x_end = self.image_width
        self.y_end = self.image_height

        self.update_display()

    # ---------------------------------------------------------
    # AFFICHAGE
    # ---------------------------------------------------------

    def update_display(self):

        if self.image is None:
            return

        # Sécurité
        self.x_start = max(0, int(self.x_start))
        self.y_start = max(0, int(self.y_start))

        self.x_end = min(
            self.image_width,
            int(self.x_end)
        )

        self.y_end = min(
            self.image_height,
            int(self.y_end)
        )

        # Éviter une zone vide
        if self.x_end <= self.x_start:
            return

        if self.y_end <= self.y_start:
            return

        # -----------------------------------------------------
        # IMPORTANT :
        # On ne transforme que la zone visible.
        # -----------------------------------------------------

        cropped = self.image[
            self.y_start:self.y_end,
            self.x_start:self.x_end
        ]

        # Taille disponible dans notre widget
        widget_width = max(self.width(), 100)
        widget_height = max(self.height(), 100)

        # Redimensionnement uniquement de la zone visible
        display = cv2.resize(
            cropped,
            (widget_width, widget_height),
            interpolation=cv2.INTER_LINEAR
        )

        # OpenCV travaille généralement en BGR.
        # Notre TIFF est normalement RGB.
        if display.shape[2] == 3:
            display = cv2.cvtColor(
                display,
                cv2.COLOR_RGB2BGR
            )

        # Conversion OpenCV → QImage
        height, width, channels = display.shape

        bytes_per_line = channels * width

        self.display_image = QImage(
            display.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_BGR888
        ).copy()

        self.update()

    # ---------------------------------------------------------
    # PAINT
    # ---------------------------------------------------------

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.fillRect(
            self.rect(),
            Qt.black
        )

        if self.display_image is not None:

            painter.drawImage(
                QRectF(self.rect()),
                self.display_image
            )

        painter.end()

    # ---------------------------------------------------------
    # MOLETTE = ZOOM
    # ---------------------------------------------------------

    def wheelEvent(self, event):

        if self.image is None:
            return

        current_width = self.x_end - self.x_start
        current_height = self.y_end - self.y_start

        # Molette vers le haut = zoom
        if event.angleDelta().y() > 0:
            zoom_factor = 0.8
        else:
            zoom_factor = 1.25

        new_width = current_width * zoom_factor
        new_height = current_height * zoom_factor

        # Limites du zoom
        if new_width < 200:
            return

        if new_width > self.image_width:
            new_width = self.image_width

        if new_height > self.image_height:
            new_height = self.image_height

        # Position de la souris dans le widget
        mouse_x = event.position().x()
        mouse_y = event.position().y()

        widget_width = max(self.width(), 1)
        widget_height = max(self.height(), 1)

        # Position relative de la souris
        mouse_pct_x = mouse_x / widget_width
        mouse_pct_y = mouse_y / widget_height

        # Point correspondant dans l'image originale
        image_mouse_x = (
            self.x_start
            + mouse_pct_x * current_width
        )

        image_mouse_y = (
            self.y_start
            + mouse_pct_y * current_height
        )

        # Nouvelle zone visible
        self.x_start = (
            image_mouse_x
            - mouse_pct_x * new_width
        )

        self.y_start = (
            image_mouse_y
            - mouse_pct_y * new_height
        )

        self.x_end = self.x_start + new_width
        self.y_end = self.y_start + new_height

        self.update_display()

    # ---------------------------------------------------------
    # CLIC SOURIS
    # ---------------------------------------------------------

    def mousePressEvent(self, event):

        if event.button() == Qt.LeftButton:

            self.is_dragging = True

            self.last_mouse_x = event.position().x()
            self.last_mouse_y = event.position().y()

            self.setCursor(Qt.ClosedHandCursor)

    # ---------------------------------------------------------
    # DÉPLACEMENT
    # ---------------------------------------------------------

    def mouseMoveEvent(self, event):

        if not self.is_dragging:
            return

        current_width = self.x_end - self.x_start
        current_height = self.y_end - self.y_start

        widget_width = max(self.width(), 1)
        widget_height = max(self.height(), 1)

        current_x = event.position().x()
        current_y = event.position().y()

        dx = (
            self.last_mouse_x - current_x
        ) * current_width / widget_width

        dy = (
            self.last_mouse_y - current_y
        ) * current_height / widget_height

        self.x_start += dx
        self.x_end += dx

        self.y_start += dy
        self.y_end += dy

        # Limites horizontales
        if self.x_start < 0:
            difference = -self.x_start
            self.x_start += difference
            self.x_end += difference

        if self.x_end > self.image_width:
            difference = self.x_end - self.image_width
            self.x_start -= difference
            self.x_end -= difference

        # Limites verticales
        if self.y_start < 0:
            difference = -self.y_start
            self.y_start += difference
            self.y_end += difference

        if self.y_end > self.image_height:
            difference = self.y_end - self.image_height
            self.y_start -= difference
            self.y_end -= difference

        self.last_mouse_x = current_x
        self.last_mouse_y = current_y

        self.update_display()

    # ---------------------------------------------------------
    # FIN DU DRAG
    # ---------------------------------------------------------

    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:

            self.is_dragging = False

            self.setCursor(Qt.ArrowCursor)

    # ---------------------------------------------------------
    # REDIMENSIONNEMENT DE LA FENÊTRE
    # ---------------------------------------------------------

    def resizeEvent(self, event):

        super().resizeEvent(event)

        if self.image is not None:
            self.update_display()
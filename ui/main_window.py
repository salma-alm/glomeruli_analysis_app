from pathlib import Path
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QProgressBar,
    QMessageBox
)

from inference.pipeline import GlomerulusPipeline


# ============================================================
# WORKER
# ============================================================

class AnalysisWorker(QThread):

    finished_success = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        pipeline,
        image_path,
        patches_dir
    ):
        super().__init__()

        self.pipeline = pipeline
        self.image_path = image_path
        self.patches_dir = patches_dir

    def run(self):

        try:

            results = self.pipeline.process(
                self.image_path,
                self.patches_dir
            )

            self.finished_success.emit(
                results
            )

        except Exception:

            self.error.emit(
                traceback.format_exc()
            )


# ============================================================
# MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Glomerulus Analysis"
        )

        self.resize(
            900,
            650
        )

        self.image_path = None
        self.pipeline = None
        self.worker = None

        self.build_ui()

    # ========================================================
    # INTERFACE
    # ========================================================

    def build_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        # ----------------------------------------------------
        # TITRE
        # ----------------------------------------------------

        title = QLabel(
            "Glomerulus Detection & Segmentation"
        )

        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            """
        )

        main_layout.addWidget(
            title
        )

        # ----------------------------------------------------
        # FICHIER TIFF
        # ----------------------------------------------------

        file_layout = QHBoxLayout()

        self.file_label = QLabel(
            "Aucune image sélectionnée"
        )

        self.select_button = QPushButton(
            "Sélectionner une image TIFF"
        )

        self.select_button.clicked.connect(
            self.select_image
        )

        file_layout.addWidget(
            self.file_label
        )

        file_layout.addWidget(
            self.select_button
        )

        main_layout.addLayout(
            file_layout
        )

        # ----------------------------------------------------
        # BOUTON ANALYSE
        # ----------------------------------------------------

        self.analyze_button = QPushButton(
            "Lancer l'analyse"
        )

        self.analyze_button.setEnabled(
            False
        )

        self.analyze_button.setMinimumHeight(
            45
        )

        self.analyze_button.clicked.connect(
            self.start_analysis
        )

        main_layout.addWidget(
            self.analyze_button
        )

        # ----------------------------------------------------
        # PROGRESS BAR
        # ----------------------------------------------------

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            1
        )

        self.progress.setValue(
            0
        )

        main_layout.addWidget(
            self.progress
        )

        # ----------------------------------------------------
        # RESULTATS
        # ----------------------------------------------------

        self.result_label = QLabel(
            "Résultats : aucune analyse"
        )

        self.result_label.setStyleSheet(
            """
            font-size: 18px;
            font-weight: bold;
            padding-top: 10px;
            """
        )

        main_layout.addWidget(
            self.result_label
        )

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        log_title = QLabel(
            "Journal"
        )

        main_layout.addWidget(
            log_title
        )

        self.log = QTextEdit()

        self.log.setReadOnly(
            True
        )

        main_layout.addWidget(
            self.log
        )

    # ========================================================
    # SELECTION IMAGE
    # ========================================================

    def select_image(self):

        file_path, _ = QFileDialog.getOpenFileName(

            self,

            "Sélectionner une image histologique",

            "",

            (
                "Images TIFF (*.tif *.tiff);;"
                "Images PNG (*.png);;"
                "Toutes les images (*)"
            )
        )

        if not file_path:
            return

        self.image_path = Path(
            file_path
        )

        self.file_label.setText(
            self.image_path.name
        )

        self.log.append(
            f"Image sélectionnée : "
            f"{self.image_path}"
        )

        self.analyze_button.setEnabled(
            True
        )

    # ========================================================
    # ANALYSE
    # ========================================================

    def start_analysis(self):

        if self.image_path is None:

            QMessageBox.warning(
                self,
                "Erreur",
                "Veuillez sélectionner une image."
            )

            return

        # ----------------------------------------------------
        # Désactiver boutons
        # ----------------------------------------------------

        self.select_button.setEnabled(
            False
        )

        self.analyze_button.setEnabled(
            False
        )

        # Barre indéterminée
        self.progress.setRange(
            0,
            0
        )

        self.result_label.setText(
            "Analyse en cours..."
        )

        self.log.append(
            ""
        )

        self.log.append(
            "=== DÉBUT ANALYSE ==="
        )

        # ----------------------------------------------------
        # Charger pipeline une seule fois
        # ----------------------------------------------------

        try:

            if self.pipeline is None:

                self.log.append(
                    "Chargement des modèles..."
                )

                self.pipeline = (
                    GlomerulusPipeline()
                )

                self.log.append(
                    "Modèles chargés."
                )

        except Exception as e:

            self.analysis_error(
                str(e)
            )

            return

        # ----------------------------------------------------
        # Dossier patches
        # ----------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parent
            .parent
        )

        patches_dir = (
            project_root
            / "results"
            / "patches"
            / self.image_path.stem
        )

        patches_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.log.append(
            f"Dossier patches : "
            f"{patches_dir}"
        )

        # ----------------------------------------------------
        # Worker
        # ----------------------------------------------------

        self.worker = AnalysisWorker(
            pipeline=self.pipeline,
            image_path=self.image_path,
            patches_dir=patches_dir
        )

        self.worker.finished_success.connect(
            self.analysis_finished
        )

        self.worker.error.connect(
            self.analysis_error
        )

        self.worker.start()

    # ========================================================
    # FIN ANALYSE
    # ========================================================

    def analysis_finished(
        self,
        results
    ):

        self.progress.setRange(
            0,
            1
        )

        self.progress.setValue(
            1
        )

        count = results.get(
            "count",
            0
        )

        width = results.get(
            "width",
            0
        )

        height = results.get(
            "height",
            0
        )

        self.result_label.setText(
            f"{count} glomérule(s) analysé(s)"
        )

        self.log.append(
            ""
        )

        self.log.append(
            f"Image : {width} × {height}"
        )

        self.log.append(
            f"Glomérules : {count}"
        )

        self.log.append(
            "=== ANALYSE TERMINÉE ==="
        )

        self.select_button.setEnabled(
            True
        )

        self.analyze_button.setEnabled(
            True
        )

    # ========================================================
    # ERREUR
    # ========================================================

    def analysis_error(
        self,
        error_message
    ):

        self.progress.setRange(
            0,
            1
        )

        self.progress.setValue(
            0
        )

        self.result_label.setText(
            "Erreur pendant l'analyse"
        )

        self.log.append(
            ""
        )

        self.log.append(
            "=== ERREUR ==="
        )

        self.log.append(
            error_message
        )

        self.select_button.setEnabled(
            True
        )

        self.analyze_button.setEnabled(
            True
        )

        QMessageBox.critical(
            self,
            "Erreur",
            "Une erreur est survenue pendant l'analyse."
        )
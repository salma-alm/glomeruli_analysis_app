from pathlib import Path

from ultralytics import YOLO


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chemin du modèle
MODEL_PATH = PROJECT_ROOT / "models" / "detection" / "best.pt"


class Detector:

    def __init__(
        self,
        model_path=MODEL_PATH,
        confidence=0.25,
        image_size=512
    ):
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.image_size = image_size

        # Vérifier que le modèle existe
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modèle de détection introuvable : {self.model_path}"
            )

        # Charger YOLO une seule fois
        self.model = YOLO(str(self.model_path))

    def detect(self, image_path):
        """
        Lance la détection YOLO sur une image ou un patch.

        Retourne une liste contenant :
        - bounding box
        - confiance
        - identifiant de classe
        - nom de classe
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image introuvable : {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False
        )

        result = results[0]

        detections = []

        if result.boxes is None:
            return detections

        for box in result.boxes:

            class_id = int(box.cls[0])

            detections.append({
                "box": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": self.model.names[class_id]
            })

        return detections
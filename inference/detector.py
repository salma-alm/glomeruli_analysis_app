from ultralytics import YOLO


MODEL_PATH = "models/detection/best.pt"


class Detector:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, image_path):
        results = self.model.predict(
            source=image_path,
            conf=0.25,
            verbose=False
        )

        result = results[0]

        detections = []

        for box in result.boxes:

            detections.append({
                "box": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            })

        return detections
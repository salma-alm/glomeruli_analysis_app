from inference.detector import Detector


detector = Detector()

print("Modèle YOLO chargé avec succès.")
print("Classes :", detector.model.names)
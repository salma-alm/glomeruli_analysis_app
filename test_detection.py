from ultralytics import YOLO

MODEL_PATH = "models/detection/best.pt"
IMAGE_PATH = r"C:\Users\salma\PycharmProjects\projetMedical\patchs\095bf7a1f"

print("Chargement du modèle...")

model = YOLO(MODEL_PATH)

print("Modèle chargé avec succès !")

print("Lancement de la détection...")

results = model(IMAGE_PATH)

print("Détection terminée.")

for result in results:

    print("Nombre de détections :", len(result.boxes))

    for box in result.boxes:

        coordinates = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        print(
            "Box :",
            coordinates,
            "| confiance :",
            confidence
        )
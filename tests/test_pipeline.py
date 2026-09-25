from inference.pipeline import Pipeline


IMAGE_PATH = r"C:\Users\salma\projets\DATA_PFA\train\0486052bb.tiff"


print("Chargement du pipeline...")

pipeline = Pipeline()

print("Traitement de l'image...")

detections = pipeline.process(IMAGE_PATH)

print("Traitement terminé.")
print("Nombre total de détections :", len(detections))

for detection in detections[:20]:

    print(
        "Patch :", detection["patch_path"],
        "| Box :", detection["box"],
        "| Confiance :", round(detection["confidence"], 3)
    )
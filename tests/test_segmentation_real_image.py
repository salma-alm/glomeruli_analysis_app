from pathlib import Path
import sys

import cv2
import numpy as np

from inference.segmenter import Segmenter


def colorize_mask(mask):
    """
    Palette proche de celle utilisée dans tes anciennes visualisations.

    0 = background -> noir
    1 -> rouge
    2 -> jaune
    3 -> vert
    """

    color_mask = np.zeros(
        (mask.shape[0], mask.shape[1], 3),
        dtype=np.uint8
    )

    color_mask[mask == 0] = [0, 0, 0]
    color_mask[mask == 1] = [255, 0, 0]
    color_mask[mask == 2] = [255, 255, 0]
    color_mask[mask == 3] = [0, 255, 0]

    return color_mask


def main():

    if len(sys.argv) < 2:
        print(
            "Utilisation : "
            "python -m tests.test_segmentation_real_image chemin_image.png"
        )
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print("Image introuvable :", image_path)
        return

    print("\n1. Chargement Attention U-Net...")

    segmenter = Segmenter()

    print("Modèle chargé.")
    print("Device :", segmenter.device)

    # ========================================================
    # Image originale
    # ========================================================

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        print("Impossible de lire l'image.")
        return

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    print(
        "Taille image originale :",
        image_rgb.shape
    )

    # ========================================================
    # Segmentation
    # ========================================================

    print("\n2. Segmentation...")

    mask = segmenter.segment(
        image_rgb
    )

    print(
        "Taille masque :",
        mask.shape
    )

    print(
        "Classes présentes :",
        np.unique(mask)
    )

    # ========================================================
    # Coloration
    # ========================================================

    color_mask = colorize_mask(mask)

    # ========================================================
    # Overlay
    # ========================================================

    overlay = cv2.addWeighted(
        image_rgb,
        0.55,
        color_mask,
        0.45,
        0
    )

    # ========================================================
    # Sauvegarde
    # ========================================================

    output_dir = Path(
        "results/segmentation_real_image"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    original_path = (
        output_dir / "original.png"
    )

    prediction_path = (
        output_dir / "prediction.png"
    )

    overlay_path = (
        output_dir / "overlay.png"
    )

    cv2.imwrite(
        str(original_path),
        cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        str(prediction_path),
        cv2.cvtColor(
            color_mask,
            cv2.COLOR_RGB2BGR
        )
    )

    cv2.imwrite(
        str(overlay_path),
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    print("\n3. Résultats enregistrés dans :")
    print(output_dir)

    print("\nFichiers :")
    print(" - original.png")
    print(" - prediction.png")
    print(" - overlay.png")


if __name__ == "__main__":
    main()
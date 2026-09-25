from pathlib import Path
import sys

import cv2
import numpy as np

from inference.detector import Detector
from inference.segmenter import Segmenter


# ============================================================
# CONFIGURATION
# ============================================================

ALPHA = 0.45
ROI_SCALE = 1.4


# ============================================================
# ROI CARREE AUTOUR DU GLOMERULE
# ============================================================

def get_glomerulus_roi(
    box,
    image_width,
    image_height,
    scale=ROI_SCALE
):

    x1, y1, x2, y2 = box

    box_width = x2 - x1
    box_height = y2 - y1

    # Centre de la détection
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Taille de la ROI carrée
    side = max(
        box_width,
        box_height
    ) * scale

    half = side / 2

    roi_x1 = int(round(center_x - half))
    roi_y1 = int(round(center_y - half))
    roi_x2 = int(round(center_x + half))
    roi_y2 = int(round(center_y + half))

    # Limites de l'image
    roi_x1 = max(0, roi_x1)
    roi_y1 = max(0, roi_y1)

    roi_x2 = min(image_width, roi_x2)
    roi_y2 = min(image_height, roi_y2)

    return (
        roi_x1,
        roi_y1,
        roi_x2,
        roi_y2
    )


# ============================================================
# COLORATION DU MASQUE
# ============================================================

def colorize_mask(mask):

    color_mask = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3
        ),
        dtype=np.uint8
    )

    # RGB

    # 0 = background
    color_mask[mask == 0] = [
        0, 0, 0
    ]

    # 1 = rouge
    color_mask[mask == 1] = [
        255, 0, 0
    ]

    # 2 = vert
    color_mask[mask == 2] = [
        0, 255, 0
    ]

    # 3 = jaune
    color_mask[mask == 3] = [
        255, 255, 0
    ]

    return color_mask


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image_rgb,
    color_mask,
    alpha=ALPHA
):

    overlay = cv2.addWeighted(
        image_rgb,
        1 - alpha,
        color_mask,
        alpha,
        0
    )

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Utilisation : "
            "python -m tests.test_models_inference image.png"
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            "Image introuvable :",
            image_path
        )

        return

    if not image_path.is_file():

        print(
            "Le chemin donné n'est pas une image :",
            image_path
        )

        return

    # ========================================================
    # 1. Charger les modèles
    # ========================================================

    print(
        "\n1. Chargement des modèles..."
    )

    detector = Detector()
    segmenter = Segmenter()

    print("YOLO chargé.")
    print("Attention U-Net chargé.")

    # ========================================================
    # 2. Lire le patch
    # ========================================================

    image_bgr = cv2.imread(
        str(image_path)
    )

    if image_bgr is None:

        print(
            "Impossible de lire l'image :",
            image_path
        )

        return

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    height, width = (
        image_rgb.shape[:2]
    )

    print(
        f"Taille du patch : "
        f"{width} x {height}"
    )

    # ========================================================
    # 3. YOLO
    # ========================================================

    print(
        "\n2. Détection YOLO..."
    )

    detections = detector.detect(
        image_path
    )

    print(
        f"Nombre de glomérules détectés : "
        f"{len(detections)}"
    )

    # ========================================================
    # Dossier résultats
    # ========================================================

    output_dir = Path(
        "results/test_individual_glomeruli"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 4. Segmenter chaque glomérule
    # ========================================================

    print(
        "\n3. Segmentation individuelle "
        "des glomérules..."
    )

    for index, detection in enumerate(
        detections,
        start=1
    ):

        # ----------------------------------------------------
        # ROI
        # ----------------------------------------------------

        (
            roi_x1,
            roi_y1,
            roi_x2,
            roi_y2
        ) = get_glomerulus_roi(

            detection["box"],
            width,
            height,
            scale=ROI_SCALE
        )

        if (
            roi_x2 <= roi_x1
            or roi_y2 <= roi_y1
        ):
            continue

        # ----------------------------------------------------
        # Crop
        # ----------------------------------------------------

        crop_rgb = image_rgb[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        if crop_rgb.size == 0:
            continue

        # ----------------------------------------------------
        # Attention U-Net
        # ----------------------------------------------------

        mask = segmenter.segment(
            crop_rgb
        )

        # ----------------------------------------------------
        # Visualisation
        # ----------------------------------------------------

        color_mask = colorize_mask(
            mask
        )

        overlay = create_overlay(
            crop_rgb,
            color_mask,
            alpha=ALPHA
        )

        # ----------------------------------------------------
        # Informations terminal
        # ----------------------------------------------------

        crop_height, crop_width = (
            crop_rgb.shape[:2]
        )

        print(
            f"Glomérule {index:02d} | "
            f"ROI={crop_width}x{crop_height} | "
            f"Mask={mask.shape} | "
            f"Classes={np.unique(mask)} | "
            f"Conf={detection['confidence']:.3f}"
        )

        # ----------------------------------------------------
        # Sauvegarde dans la TAILLE NATIVE
        # ----------------------------------------------------

        crop_path = (
            output_dir
            / f"glomerulus_{index:02d}_image.png"
        )

        mask_path = (
            output_dir
            / f"glomerulus_{index:02d}_prediction.png"
        )

        overlay_path = (
            output_dir
            / f"glomerulus_{index:02d}_overlay.png"
        )

        # RGB -> BGR pour OpenCV

        crop_bgr = cv2.cvtColor(
            crop_rgb,
            cv2.COLOR_RGB2BGR
        )

        mask_bgr = cv2.cvtColor(
            color_mask,
            cv2.COLOR_RGB2BGR
        )

        overlay_bgr = cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            str(crop_path),
            crop_bgr
        )

        cv2.imwrite(
            str(mask_path),
            mask_bgr
        )

        cv2.imwrite(
            str(overlay_path),
            overlay_bgr
        )

    print(
        "\nTest terminé."
    )

    print(
        "Résultats enregistrés dans :",
        output_dir
    )


if __name__ == "__main__":
    main()
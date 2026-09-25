from pathlib import Path

import cv2
import numpy as np
import tifffile

from inference.patcher import create_patches
from inference.detector import Detector
from inference.segmenter import Segmenter


class GlomerulusPipeline:

    def __init__(self):

        print("Chargement YOLO...")
        self.detector = Detector()

        print("Chargement Attention U-Net...")
        self.segmenter = Segmenter()

        print("Modèles chargés avec succès.")

    # =========================================================
    # ROI carrée autour du glomérule détecté
    # =========================================================

    def get_glomerulus_roi(
        self,
        box,
        image_width,
        image_height,
        scale=1.4
    ):

        x1, y1, x2, y2 = box

        box_width = x2 - x1
        box_height = y2 - y1

        # Centre de la bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # ROI carrée légèrement plus grande que la bbox YOLO
        side = max(box_width, box_height) * scale

        half_side = side / 2

        roi_x1 = int(round(center_x - half_side))
        roi_y1 = int(round(center_y - half_side))
        roi_x2 = int(round(center_x + half_side))
        roi_y2 = int(round(center_y + half_side))

        # Limiter aux dimensions du patch
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)

        roi_x2 = min(image_width, roi_x2)
        roi_y2 = min(image_height, roi_y2)

        return roi_x1, roi_y1, roi_x2, roi_y2

    # =========================================================
    # PIPELINE COMPLET
    # =========================================================

    def process(
        self,
        image_path,
        patches_dir
    ):

        image_path = Path(image_path)
        patches_dir = Path(patches_dir)

        # =====================================================
        # 1. Lire la TIFF originale
        # =====================================================

        original_image = tifffile.imread(
            image_path
        )

        # Cas (C, H, W)
        if (
            original_image.ndim == 3
            and original_image.shape[0] in (3, 4)
        ):
            original_image = np.transpose(
                original_image,
                (1, 2, 0)
            )

        image_height, image_width = (
            original_image.shape[:2]
        )

        print(
            f"Image : {image_width} x {image_height}"
        )

        # =====================================================
        # 2. Découpage en patches
        # =====================================================

        patches = create_patches(
            image_path,
            patches_dir
        )

        print(
            f"{len(patches)} patches créés."
        )

        # Masque global de la TIFF
        global_mask = np.zeros(
            (
                image_height,
                image_width
            ),
            dtype=np.uint8
        )

        glomeruli = []

        glomerulus_id = 1

        # =====================================================
        # 3. Parcourir les patches
        # =====================================================

        for patch_index, patch_info in enumerate(
            patches,
            start=1
        ):

            patch_path = Path(
                patch_info["path"]
            )

            patch_x = patch_info["x"]
            patch_y = patch_info["y"]

            real_width = patch_info.get(
                "width",
                min(
                    512,
                    image_width - patch_x
                )
            )

            real_height = patch_info.get(
                "height",
                min(
                    512,
                    image_height - patch_y
                )
            )

            print(
                f"\nPatch {patch_index}/{len(patches)}"
            )

            # =================================================
            # 4. Détection YOLO
            # =================================================

            detections = self.detector.detect(
                patch_path
            )

            print(
                f"  {len(detections)} "
                f"glomérule(s) détecté(s)"
            )

            if len(detections) == 0:
                continue

            # =================================================
            # 5. Lire le patch
            # =================================================

            patch_bgr = cv2.imread(
                str(patch_path)
            )

            if patch_bgr is None:

                print(
                    f"Impossible de lire : "
                    f"{patch_path}"
                )

                continue

            patch_rgb = cv2.cvtColor(
                patch_bgr,
                cv2.COLOR_BGR2RGB
            )

            # =================================================
            # 6. Traiter chaque glomérule
            # =================================================

            for detection in detections:

                original_box = detection["box"]

                # ---------------------------------------------
                # ROI carrée autour du glomérule
                # ---------------------------------------------

                x1, y1, x2, y2 = (
                    self.get_glomerulus_roi(
                        original_box,
                        real_width,
                        real_height,
                        scale=1.4
                    )
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                # ---------------------------------------------
                # Crop individuel du glomérule
                # ---------------------------------------------

                glomerulus_crop = patch_rgb[
                    y1:y2,
                    x1:x2
                ]

                if glomerulus_crop.size == 0:
                    continue

                # =================================================
                # 7. Segmentation Attention U-Net
                # =================================================

                glomerulus_mask = (
                    self.segmenter.segment(
                        glomerulus_crop
                    )
                )

                # 1 détection YOLO
                # =
                # 1 crop
                # =
                # 1 masque individuel

                # =================================================
                # 8. Coordonnées globales
                # =================================================

                global_x1 = patch_x + x1
                global_y1 = patch_y + y1

                global_x2 = patch_x + x2
                global_y2 = patch_y + y2

                # =================================================
                # 9. Ajouter au masque global
                # =================================================

                region = global_mask[
                    global_y1:global_y2,
                    global_x1:global_x2
                ]

                np.copyto(
                    region,
                    glomerulus_mask,
                    where=(glomerulus_mask != 0)
                )

                # =================================================
                # 10. Bounding box YOLO originale
                # =================================================

                tight_box = [
                    patch_x + int(original_box[0]),
                    patch_y + int(original_box[1]),
                    patch_x + int(original_box[2]),
                    patch_y + int(original_box[3])
                ]

                # =================================================
                # 11. Sauvegarder les informations
                # =================================================

                glomeruli.append({

                    "id": glomerulus_id,

                    "detection_box":
                        tight_box,

                    "segmentation_box": [
                        global_x1,
                        global_y1,
                        global_x2,
                        global_y2
                    ],

                    "confidence":
                        detection["confidence"],

                    "class_id":
                        detection["class_id"],

                    "mask":
                        glomerulus_mask
                })

                print(
                    f"    Glomérule "
                    f"{glomerulus_id} : "
                    f"segmenté ✅"
                )

                glomerulus_id += 1

        # =====================================================
        # 12. Résultat final
        # =====================================================

        return {

            "mask":
                global_mask,

            "glomeruli":
                glomeruli,

            "count":
                len(glomeruli),

            "width":
                image_width,

            "height":
                image_height
        }
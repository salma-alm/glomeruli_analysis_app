from pathlib import Path

import cv2
import numpy as np
import tifffile


PATCH_SIZE = 512


def prepare_image(image):
    """
    Prépare l'image TIFF pour le traitement.
    Retourne une image RGB uint8.
    """

    # Cas : image sous forme (C, H, W)
    if image.ndim == 3 and image.shape[0] in (3, 4):
        image = np.transpose(image, (1, 2, 0))

    # Cas : image en niveaux de gris
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Cas : image RGBA
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    # Conversion en uint8 si nécessaire
    if image.dtype != np.uint8:

        image = image.astype(np.float32)

        image_min = image.min()
        image_max = image.max()

        if image_max > image_min:
            image = (
                (image - image_min)
                / (image_max - image_min)
                * 255
            )

        image = image.astype(np.uint8)

    return image


def create_patches(image_path, output_dir):
    """
    Découpe une image TIFF en patches 512x512.

    Les patches situés aux bords sont complétés
    avec un fond blanc afin de conserver une taille
    constante de 512x512.

    Retourne la liste des patches et leurs coordonnées.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lecture selon le type de fichier
    suffix = image_path.suffix.lower()

    if suffix in [".tif", ".tiff"]:

      image = tifffile.imread(image_path)

    else:

     image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Impossible de lire l'image : {image_path}"
        )

    # OpenCV lit en BGR -> conversion RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Préparation image
    image = prepare_image(image)

    height, width = image.shape[:2]

    patches = []

    for y in range(0, height, PATCH_SIZE):

        for x in range(0, width, PATCH_SIZE):

            # Dimensions réellement disponibles
            real_width = min(PATCH_SIZE, width - x)
            real_height = min(PATCH_SIZE, height - y)

            patch = image[
                y:y + real_height,
                x:x + real_width
            ]

            # Création d'un patch 512x512 blanc
            padded_patch = np.full(
                (PATCH_SIZE, PATCH_SIZE, 3),
                255,
                dtype=np.uint8
            )

            # Copier la région réelle dans le patch
            padded_patch[
                :real_height,
                :real_width
            ] = patch

            # Nom du patch
            patch_name = f"patch_{y}_{x}.png"

            patch_path = output_dir / patch_name

            # OpenCV utilise BGR
            patch_bgr = cv2.cvtColor(
                padded_patch,
                cv2.COLOR_RGB2BGR
            )

            cv2.imwrite(
                str(patch_path),
                patch_bgr
            )

            # Informations nécessaires pour reconstruire l'image
            patches.append(
                {
                    "path": patch_path,
                    "x": x,
                    "y": y,
                    "width": real_width,
                    "height": real_height
                }
            )

    return patches
from pathlib import Path
import tifffile
import numpy as np
import cv2


PATCH_SIZE = 512


def create_patches(image_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = tifffile.imread(image_path)

    # Adapter les dimensions si l'image est sous la forme (C, H, W)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    height, width = image.shape[:2]

    patches = []

    for y in range(0, height, PATCH_SIZE):
        for x in range(0, width, PATCH_SIZE):

            patch = image[
                y:min(y + PATCH_SIZE, height),
                x:min(x + PATCH_SIZE, width)
            ]

            patch_name = f"patch_{y}_{x}.png"
            patch_path = output_dir / patch_name

            cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

            patches.append({
                "path": patch_path,
                "x": x,
                "y": y
            })

    return patches
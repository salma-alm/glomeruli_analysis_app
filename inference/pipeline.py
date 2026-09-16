from pathlib import Path

from inference.patcher import create_patches
from inference.detector import Detector


class Pipeline:

    def __init__(self):
        self.detector = Detector()

    def process(self, image_path):

        image_path = Path(image_path)

        patches_dir = Path("results") / "patches"

        patches = create_patches(
            image_path,
            patches_dir
        )

        all_detections = []

        for patch in patches:

            detections = self.detector.detect(
                patch["path"]
            )

            for detection in detections:

                detection["patch_path"] = patch["path"]
                detection["patch_x"] = patch["x"]
                detection["patch_y"] = patch["y"]

                all_detections.append(detection)

        return all_detections
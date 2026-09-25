from inference.segmenter import Segmenter


def test_model_loading():

    print("Chargement du modèle Attention U-Net...")

    segmenter = Segmenter()

    print("Modèle Attention U-Net chargé avec succès.")
    print("Device :", segmenter.device)
    print("Modèle :", type(segmenter.model).__name__)
    print("Chemin :", segmenter.model_path)


if __name__ == "__main__":
    test_model_loading()
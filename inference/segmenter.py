from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "segmentation"
    / "best_attention_unet.pth"
)

IMAGE_SIZE = 512
NUM_CLASSES = 4


# ============================================================
# DOUBLE CONVOLUTION
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ============================================================
# ATTENTION BLOCK
# ============================================================

class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        # Signal venant du décodeur
        self.W_g = nn.Sequential(

            nn.Conv2d(
                F_g,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.BatchNorm2d(F_int)
        )

        # Skip connection venant de l'encodeur
        self.W_x = nn.Sequential(

            nn.Conv2d(
                F_l,
                F_int,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.BatchNorm2d(F_int)
        )

        # Coefficient d'attention
        self.psi = nn.Sequential(

            nn.Conv2d(
                F_int,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),

            nn.BatchNorm2d(1),

            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(
            g1 + x1
        )

        psi = self.psi(psi)

        return x * psi


# ============================================================
# ATTENTION U-NET
# ============================================================

class AttentionUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        num_classes=4
    ):
        super().__init__()

        # ----------------------------------------------------
        # ENCODEUR
        # ----------------------------------------------------

        self.enc1 = DoubleConv(
            in_channels,
            32
        )

        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(
            32,
            64
        )

        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(
            64,
            128
        )

        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(
            128,
            256
        )

        self.pool4 = nn.MaxPool2d(2)

        # ----------------------------------------------------
        # BOTTLENECK
        # ----------------------------------------------------

        self.bottleneck = DoubleConv(
            256,
            512
        )

        # ----------------------------------------------------
        # DECODER 4
        # ----------------------------------------------------

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.att4 = AttentionBlock(
            F_g=256,
            F_l=256,
            F_int=128
        )

        self.dec4 = DoubleConv(
            512,
            256
        )

        # ----------------------------------------------------
        # DECODER 3
        # ----------------------------------------------------

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.att3 = AttentionBlock(
            F_g=128,
            F_l=128,
            F_int=64
        )

        self.dec3 = DoubleConv(
            256,
            128
        )

        # ----------------------------------------------------
        # DECODER 2
        # ----------------------------------------------------

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.att2 = AttentionBlock(
            F_g=64,
            F_l=64,
            F_int=32
        )

        self.dec2 = DoubleConv(
            128,
            64
        )

        # ----------------------------------------------------
        # DECODER 1
        # ----------------------------------------------------

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.att1 = AttentionBlock(
            F_g=32,
            F_l=32,
            F_int=16
        )

        self.dec1 = DoubleConv(
            64,
            32
        )

        # ----------------------------------------------------
        # SORTIE
        # ----------------------------------------------------

        self.final = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        # Encodeur
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        e3 = self.enc3(
            self.pool2(e2)
        )

        e4 = self.enc4(
            self.pool3(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool4(e4)
        )

        # ----------------------------------------------------
        # Decoder 4
        # ----------------------------------------------------

        d4 = self.up4(b)

        e4_att = self.att4(
            g=d4,
            x=e4
        )

        d4 = torch.cat(
            (e4_att, d4),
            dim=1
        )

        d4 = self.dec4(d4)

        # ----------------------------------------------------
        # Decoder 3
        # ----------------------------------------------------

        d3 = self.up3(d4)

        e3_att = self.att3(
            g=d3,
            x=e3
        )

        d3 = torch.cat(
            (e3_att, d3),
            dim=1
        )

        d3 = self.dec3(d3)

        # ----------------------------------------------------
        # Decoder 2
        # ----------------------------------------------------

        d2 = self.up2(d3)

        e2_att = self.att2(
            g=d2,
            x=e2
        )

        d2 = torch.cat(
            (e2_att, d2),
            dim=1
        )

        d2 = self.dec2(d2)

        # ----------------------------------------------------
        # Decoder 1
        # ----------------------------------------------------

        d1 = self.up1(d2)

        e1_att = self.att1(
            g=d1,
            x=e1
        )

        d1 = torch.cat(
            (e1_att, d1),
            dim=1
        )

        d1 = self.dec1(d1)

        return self.final(d1)


# ============================================================
# SEGMENTER
# ============================================================

class Segmenter:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Modèle de segmentation introuvable : "
                f"{self.model_path}"
            )

        # GPU si disponible, sinon CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Créer Attention U-Net
        self.model = AttentionUNet(
            in_channels=3,
            num_classes=NUM_CLASSES
        )

        # Charger les poids
        state_dict = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            state_dict
        )

        # Envoyer modèle vers CPU/GPU
        self.model.to(
            self.device
        )

        # Mode inférence
        self.model.eval()

    def segment(self, image):

        # ----------------------------------------------------
        # Lecture image si chemin
        # ----------------------------------------------------

        if isinstance(image, (str, Path)):

            image = cv2.imread(
                str(image)
            )

            if image is None:

                raise ValueError(
                    f"Impossible de lire l'image : {image}"
                )

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        # Vérification
        if image is None:

            raise ValueError(
                "L'image reçue est vide."
            )

        # Taille originale
        original_height, original_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        image_resized = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        image_resized = (
            image_resized.astype(np.float32)
            / 255.0
        )

        # HWC -> CHW
        image_tensor = np.transpose(
            image_resized,
            (2, 0, 1)
        )

        # numpy -> Tensor
        image_tensor = torch.from_numpy(
            image_tensor
        )

        # Ajouter dimension batch
        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            output = self.model(
                image_tensor
            )

            mask = torch.argmax(
                output,
                dim=1
            )

        # ----------------------------------------------------
        # CONVERSION MASQUE
        # ----------------------------------------------------

        mask = (
            mask
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # Revenir à la taille originale
        mask = cv2.resize(
            mask,
            (
                original_width,
                original_height
            ),
            interpolation=cv2.INTER_NEAREST
        )

        return mask
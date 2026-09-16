from inference.patcher import create_patches


IMAGE_PATH = r"C:\Users\salma\projets\DATA_PFA\train\0486052bb.tiff"
OUTPUT_DIR = "results/test_patches"


print("Découpage de l'image...")

patches = create_patches(
    IMAGE_PATH,
    OUTPUT_DIR
)

print("Découpage terminé.")
print("Nombre de patches :", len(patches))

for patch in patches[:5]:
    print(patch)
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import random

# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# ==========================================================
# LOAD IMAGE PATHS
# ==========================================================

image_files = [
    file for file in IMAGE_DIR.iterdir()
    if file.suffix.lower() in SUPPORTED_EXTENSIONS
]

# Randomly choose 9 images
sample_images = random.sample(image_files, 9)

# ==========================================================
# DISPLAY IMAGES
# ==========================================================

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, image_path in zip(axes.flatten(), sample_images):

    img = Image.open(image_path)

    ax.imshow(img, cmap="gray")

    ax.set_title(
        f"{image_path.suffix.upper()}\n{img.mode} | {img.width}×{img.height}",
        fontsize=9
    )

    ax.axis("off")

plt.tight_layout()

save_path = OUTPUT_DIR / "sample_images.png"

plt.savefig(save_path, dpi=300)

plt.show()

print(f"\nSample visualization saved to:\n{save_path}")
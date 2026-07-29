from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

RAW_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("outputs")

SUPPORTED = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

images = [p for p in RAW_DIR.iterdir() if p.suffix.lower() in SUPPORTED]

# Select one random image
img_path = random.choice(images)

# Open image in grayscale
img = Image.open(img_path).convert("L")
img_np = np.array(img)

# Threshold to separate black background
mask = img_np > 10

rows = np.any(mask, axis=1)
cols = np.any(mask, axis=0)

top = np.argmax(rows)
bottom = len(rows) - np.argmax(rows[::-1])
left = np.argmax(cols)
right = len(cols) - np.argmax(cols[::-1])

cropped = img.crop((left, top, right, bottom))

# Plot comparison
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")
ax[0].axis("off")

ax[1].imshow(cropped, cmap="gray")
ax[1].set_title("ROI Cropped")
ax[1].axis("off")

plt.tight_layout()

OUTPUT_DIR.mkdir(exist_ok=True)

save_path = OUTPUT_DIR / "roi_cropping_demo.png"
plt.savefig(save_path, dpi=300)

plt.show()

print(f"Saved comparison to: {save_path}")

print("\nCropping Statistics")
print("-------------------")
print(f"Original Size : {img.width} x {img.height}")
print(f"Cropped Size  : {cropped.width} x {cropped.height}")
print(f"Pixels Removed: {img.width*img.height - cropped.width*cropped.height}")
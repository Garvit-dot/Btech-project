from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURATION
# =====================================================

IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SUPPORTED = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

image_files = [
    f for f in IMAGE_DIR.iterdir()
    if f.suffix.lower() in SUPPORTED
]

print("="*60)
print("PIXEL INTENSITY ANALYSIS")
print("="*60)

# -----------------------------------------------------
# Histogram accumulator
# -----------------------------------------------------

histogram = np.zeros(256, dtype=np.int64)

pixel_sum = 0
pixel_sq_sum = 0
total_pixels = 0

image_means = {}

for img_path in image_files:

    img = Image.open(img_path).convert("L")

    arr = np.array(img, dtype=np.uint8)

    # histogram
    hist, _ = np.histogram(arr, bins=256, range=(0,256))
    histogram += hist

    # running statistics
    pixel_sum += arr.sum(dtype=np.int64)
    pixel_sq_sum += np.square(arr, dtype=np.int64).sum()

    total_pixels += arr.size

    image_means[img_path.name] = arr.mean()

# -----------------------------------------------------
# Dataset Mean & Std
# -----------------------------------------------------

dataset_mean = pixel_sum / total_pixels

variance = (pixel_sq_sum / total_pixels) - (dataset_mean ** 2)

dataset_std = np.sqrt(variance)

print(f"\nDataset Mean : {dataset_mean:.2f}")
print(f"Dataset Std  : {dataset_std:.2f}")

# -----------------------------------------------------
# Darkest & Brightest
# -----------------------------------------------------

darkest = min(image_means, key=image_means.get)
brightest = max(image_means, key=image_means.get)

print("\nDarkest Image")
print(darkest)

print("\nBrightest Image")
print(brightest)

# -----------------------------------------------------
# Histogram Plot
# -----------------------------------------------------

plt.figure(figsize=(10,5))

plt.bar(np.arange(256), histogram, width=1)

plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "pixel_histogram.png", dpi=300)

plt.show()

print("\nHistogram saved successfully.")
from pathlib import Path
from PIL import Image
from collections import Counter
import csv

# ==========================================================
# CONFIGURATION
# ==========================================================

IMAGE_DIR = Path("data/raw/images")
OUTPUT_DIR = Path("data/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "dataset_summary.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}

# ==========================================================
# FIND ALL IMAGES
# ==========================================================

image_files = [
    file for file in IMAGE_DIR.iterdir()
    if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
]

if not image_files:
    print("❌ No images found!")
    print(f"Checked directory: {IMAGE_DIR.resolve()}")
    exit()

print("=" * 60)
print("DENTAL DATASET INSPECTION REPORT")
print("=" * 60)

print(f"\nTotal Images Found : {len(image_files)}")

# ==========================================================
# FILE FORMAT DISTRIBUTION
# ==========================================================

extensions = Counter(file.suffix.lower() for file in image_files)

print("\nFile Format Distribution")
print("-" * 30)

for ext, count in sorted(extensions.items()):
    print(f"{ext:<8}: {count}")

# ==========================================================
# IMAGE ANALYSIS
# ==========================================================

widths = []
heights = []
modes = Counter()
corrupted = []

dataset_rows = []

for image_path in image_files:

    try:
        with Image.open(image_path) as img:

            width, height = img.size

            widths.append(width)
            heights.append(height)

            modes[img.mode] += 1

            dataset_rows.append({
                "filename": image_path.name,
                "extension": image_path.suffix.lower(),
                "width": width,
                "height": height,
                "mode": img.mode
            })

    except Exception as e:

        corrupted.append(image_path.name)

        dataset_rows.append({
            "filename": image_path.name,
            "extension": image_path.suffix.lower(),
            "width": "",
            "height": "",
            "mode": "CORRUPTED"
        })

# ==========================================================
# IMAGE SIZE STATISTICS
# ==========================================================

print("\nImage Size Statistics")
print("-" * 30)

print(f"Minimum Width   : {min(widths)}")
print(f"Maximum Width   : {max(widths)}")
print(f"Average Width   : {sum(widths)/len(widths):.2f}")

print()

print(f"Minimum Height  : {min(heights)}")
print(f"Maximum Height  : {max(heights)}")
print(f"Average Height  : {sum(heights)/len(heights):.2f}")

# ==========================================================
# IMAGE MODES
# ==========================================================

print("\nImage Modes")
print("-" * 30)

for mode, count in modes.items():
    print(f"{mode:<10}: {count}")

# ==========================================================
# CORRUPTED IMAGES
# ==========================================================

print("\nCorrupted Images")
print("-" * 30)

print(f"Total Corrupted : {len(corrupted)}")

if corrupted:
    print("\nCorrupted Files:")
    for file in corrupted:
        print(file)

# ==========================================================
# WRITE CSV
# ==========================================================

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=["filename", "extension", "width", "height", "mode"]
    )

    writer.writeheader()

    writer.writerows(dataset_rows)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Images Analysed  : {len(image_files)}")
print(f"Corrupted Images : {len(corrupted)}")
print(f"Metadata Saved   : {OUTPUT_CSV}")

print("\nDataset inspection completed successfully! ✅")
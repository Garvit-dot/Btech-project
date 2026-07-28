from pathlib import Path
from PIL import Image

# =====================================================
# CONFIGURATION
# =====================================================

RAW_DIR = Path("data/raw/images")
PROCESSED_DIR = Path("data/processed/images")
OUTPUT_DIR = Path("outputs")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff"
}

# =====================================================
# INITIALIZE COUNTERS
# =====================================================

total_images = 0
rgb_converted = 0
already_grayscale = 0
failed_images = 0

# =====================================================
# PROCESS IMAGES
# =====================================================

for image_path in RAW_DIR.iterdir():

    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        continue

    total_images += 1

    try:

        img = Image.open(image_path)

        # Convert RGB images to grayscale
        if img.mode != "L":
            img = img.convert("L")
            rgb_converted += 1
        else:
            already_grayscale += 1

        # Save image
        save_path = PROCESSED_DIR / image_path.name
        img.save(save_path)

    except Exception as e:

        failed_images += 1
        print(f"Failed : {image_path.name}")
        print(e)

# =====================================================
# REPORT
# =====================================================

report_path = OUTPUT_DIR / "preprocessing_report.txt"

with open(report_path, "w") as f:

    f.write("=" * 50 + "\n")
    f.write("PREPROCESSING REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Total Images            : {total_images}\n")
    f.write(f"RGB Converted           : {rgb_converted}\n")
    f.write(f"Already Grayscale       : {already_grayscale}\n")
    f.write(f"Failed Images           : {failed_images}\n")
    f.write(f"Output Directory        : {PROCESSED_DIR}\n")

# =====================================================
# TERMINAL OUTPUT
# =====================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print(f"Total Images        : {total_images}")
print(f"RGB Converted       : {rgb_converted}")
print(f"Already Grayscale   : {already_grayscale}")
print(f"Failed Images       : {failed_images}")

print(f"\nProcessed Images Saved To :")
print(PROCESSED_DIR)

print(f"\nReport Saved To :")
print(report_path)
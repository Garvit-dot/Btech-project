import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_DATASET = PROJECT_ROOT / "data" / "yolo_dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "faster_rcnn_dataset"

LABEL_DIR = YOLO_DATASET / "labels"
IMAGE_DIR = YOLO_DATASET / "images"

# Create output folders
for split in ["train", "val"]:
    (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "annotations" / split).mkdir(parents=True, exist_ok=True)

# Read class names from dataset.yaml manually
CLASS_NAMES = [
    "GP",
    "boneloss",
    "caries",
    "crown",
    "dental_stone",
    "erupting_tooth",
    "filled",
    "healing_socket",
    "missing",
    "opturation",
    "periapical_lesion",
    "pulp_stone",
    "rootcanal",
    "working_length_rct",
]

def convert_split(split):

    label_path = LABEL_DIR / split
    image_path = IMAGE_DIR / split

    output_images = OUTPUT_DIR / "images" / split
    output_annotations = OUTPUT_DIR / "annotations" / split

    label_files = sorted(label_path.glob("*.txt"))

    print(f"\nProcessing {split}: {len(label_files)} images")

    successful = 0

    for label_file in label_files:

        image_file = None

        for extension in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_path / (label_file.stem + extension)

            if candidate.exists():
                image_file = candidate
                break

        if image_file is None:
            print(f"[WARNING] Image not found for {label_file.name}")
            continue

        # ----------------------------------------------------
        # Read image dimensions
        # ----------------------------------------------------

        from PIL import Image

        with Image.open(image_file) as img:
            width, height = img.size

        annotations = []

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line in lines:

            values = line.split()

            if len(values) != 9:
                print(
                    f"[WARNING] Invalid OBB annotation: "
                    f"{label_file.name}"
                )
                continue

            class_id = int(values[0])

            coords = list(map(float, values[1:]))

            xs = coords[0::2]
            ys = coords[1::2]

            # YOLO coordinates are normalized
            xmin = min(xs) * width
            xmax = max(xs) * width

            ymin = min(ys) * height
            ymax = max(ys) * height

            # Clamp to image boundaries
            xmin = max(0, min(xmin, width))
            xmax = max(0, min(xmax, width))

            ymin = max(0, min(ymin, height))
            ymax = max(0, min(ymax, height))

            if xmax <= xmin or ymax <= ymin:
                continue

            annotations.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "bbox": [
                    round(xmin, 2),
                    round(ymin, 2),
                    round(xmax, 2),
                    round(ymax, 2)
                ]
            })

        # ----------------------------------------------------
        # Save annotation JSON
        # ----------------------------------------------------

        annotation_data = {
            "image": image_file.name,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        output_json = (
            output_annotations /
            f"{label_file.stem}.json"
        )

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=4)

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        shutil.copy2(
            image_file,
            output_images / image_file.name
        )

        successful += 1

    print(f"Successfully converted: {successful}")


# Run conversion
convert_split("train")
convert_split("val")

print("\n======================================")
print("Faster R-CNN dataset created")
print("======================================")
print(f"Output: {OUTPUT_DIR}")
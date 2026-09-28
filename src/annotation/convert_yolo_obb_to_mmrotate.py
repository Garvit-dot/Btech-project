from pathlib import Path
import shutil
import json
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_ROOT = PROJECT_ROOT / "data" / "yolo_dataset"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "mmrotate_dataset"

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


# ============================================================
# CONVERSION
# ============================================================

def convert_split(split):
    """
    Convert one YOLO OBB split into MMRotate-style annotations.

    YOLO format:
        class x1 y1 x2 y2 x3 y3 x4 y4

    Coordinates are normalized [0, 1].

    Output annotation:
        {
            "image": "...",
            "width": ...,
            "height": ...,
            "annotations": [
                {
                    "class_id": ...,
                    "class_name": "...",
                    "polygon": [x1,y1,x2,y2,x3,y3,x4,y4],
                    "bbox": [xmin,ymin,xmax,ymax]
                }
            ]
        }
    """

    image_dir = YOLO_ROOT / "images" / split
    label_dir = YOLO_ROOT / "labels" / split

    output_image_dir = OUTPUT_ROOT / "images" / split
    output_annotation_dir = OUTPUT_ROOT / "annotations" / split

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_annotation_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))

    print("\n" + "=" * 70)
    print(f"Processing split: {split}")
    print(f"Label files found: {len(label_files)}")
    print("=" * 70)

    converted = 0
    skipped = 0
    missing_images = 0
    malformed_labels = 0
    total_objects = 0

    for label_file in label_files:

        image_stem = label_file.stem

        # ----------------------------------------------------
        # Find corresponding image
        # ----------------------------------------------------

        image_path = None

        for extension in [".bmp", ".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".BMP", ".PNG"]:
            candidate = image_dir / f"{image_stem}{extension}"

            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            print(f"[WARNING] Image not found for: {label_file.name}")
            missing_images += 1
            skipped += 1
            continue

        # ----------------------------------------------------
        # Read image dimensions
        # ----------------------------------------------------

        try:
            with Image.open(image_path) as img:
                width, height = img.size

        except Exception as e:
            print(f"[WARNING] Could not read image: {image_path.name}")
            print(f"          Error: {e}")
            skipped += 1
            continue

        # ----------------------------------------------------
        # Read YOLO OBB annotations
        # ----------------------------------------------------

        annotations = []

        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        for line_number, line in enumerate(lines, start=1):

            parts = line.split()

            # class + 8 coordinates = 9 values
            if len(parts) != 9:
                print(
                    f"[WARNING] Malformed annotation: "
                    f"{label_file.name}, line {line_number}"
                )
                malformed_labels += 1
                continue

            try:
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

            except ValueError:
                print(
                    f"[WARNING] Invalid numeric values: "
                    f"{label_file.name}, line {line_number}"
                )
                malformed_labels += 1
                continue

            # ------------------------------------------------
            # Validate class
            # ------------------------------------------------

            if not (0 <= class_id < len(CLASS_NAMES)):
                print(
                    f"[WARNING] Invalid class ID {class_id} "
                    f"in {label_file.name}"
                )
                malformed_labels += 1
                continue

            # ------------------------------------------------
            # Convert normalized coordinates to pixels
            # ------------------------------------------------

            polygon = []

            for i in range(0, 8, 2):

                x_norm = coords[i]
                y_norm = coords[i + 1]

                x = x_norm * width
                y = y_norm * height

                # Clamp to image boundaries
                x = max(0.0, min(float(width), x))
                y = max(0.0, min(float(height), y))

                polygon.extend([x, y])

            # ------------------------------------------------
            # Calculate axis-aligned bounding box
            #
            # This is ONLY stored as auxiliary information.
            # The polygon remains the source OBB.
            # ------------------------------------------------

            xs = polygon[0::2]
            ys = polygon[1::2]

            xmin = min(xs)
            ymin = min(ys)
            xmax = max(xs)
            ymax = max(ys)

            annotation = {
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "polygon": polygon,
                "bbox": [xmin, ymin, xmax, ymax]
            }

            annotations.append(annotation)
            total_objects += 1

        # ----------------------------------------------------
        # Create annotation JSON
        # ----------------------------------------------------

        annotation_data = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        output_json = output_annotation_dir / f"{image_stem}.json"

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(annotation_data, f, indent=2)

        # ----------------------------------------------------
        # Copy image
        # ----------------------------------------------------

        destination_image = output_image_dir / image_path.name

        if not destination_image.exists():
            shutil.copy2(image_path, destination_image)

        converted += 1

    # --------------------------------------------------------
    # Split summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"Split: {split}")
    print(f"Converted images : {converted}")
    print(f"Skipped images   : {skipped}")
    print(f"Missing images   : {missing_images}")
    print(f"Malformed labels : {malformed_labels}")
    print(f"Total objects    : {total_objects}")
    print("-" * 70)

    return {
        "split": split,
        "label_files": len(label_files),
        "converted": converted,
        "skipped": skipped,
        "missing_images": missing_images,
        "malformed_labels": malformed_labels,
        "total_objects": total_objects,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("YOLO OBB → MMROTATE DATASET CONVERTER")
    print("=" * 70)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Input        : {YOLO_ROOT}")
    print(f"Output       : {OUTPUT_ROOT}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_summary = convert_split("train")
    val_summary = convert_split("val")

    # --------------------------------------------------------
    # Save dataset metadata
    # --------------------------------------------------------

    metadata = {
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "source_format": "YOLO OBB",
        "target_format": "MMRotate-compatible polygon annotations",
        "splits": {
            "train": train_summary,
            "val": val_summary
        }
    }

    metadata_path = OUTPUT_ROOT / "dataset_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"Train images converted : {train_summary['converted']}")
    print(f"Val images converted   : {val_summary['converted']}")
    print(f"Train objects          : {train_summary['total_objects']}")
    print(f"Val objects            : {val_summary['total_objects']}")

    print("\nOutput directory:")
    print(OUTPUT_ROOT)

    print("\nOriginal YOLO dataset was NOT modified.")


if __name__ == "__main__":
    main()
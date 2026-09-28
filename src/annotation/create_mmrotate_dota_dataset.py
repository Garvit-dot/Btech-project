from pathlib import Path
import json
import shutil
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = PROJECT_ROOT / "data" / "mmrotate_dataset"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "mmrotate_dota"


# ============================================================
# CLASSES
# ============================================================

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
# CONVERT ONE SPLIT
# ============================================================

def convert_split(split):

    source_images = SOURCE_ROOT / "images" / split
    source_annotations = SOURCE_ROOT / "annotations" / split

    output_images = OUTPUT_ROOT / split / "images"
    output_annotations = OUTPUT_ROOT / split / "annfiles"

    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    json_files = sorted(source_annotations.glob("*.json"))

    print("\n" + "=" * 70)
    print(f"Processing {split} split")
    print(f"JSON files: {len(json_files)}")
    print("=" * 70)

    converted_images = 0
    converted_objects = 0
    skipped = 0

    for json_file in json_files:

        # ----------------------------------------------------
        # Read JSON
        # ----------------------------------------------------

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_name = data["image"]
        image_stem = Path(image_name).stem

        source_image = source_images / image_name

        if not source_image.exists():
            print(f"[WARNING] Image not found: {source_image}")
            skipped += 1
            continue

        # ----------------------------------------------------
        # Convert image to PNG
        # ----------------------------------------------------

        output_image = output_images / f"{image_stem}.png"

        try:

            with Image.open(source_image) as img:

                # Preserve grayscale/RGB appropriately
                if img.mode not in ["RGB", "L"]:
                    img = img.convert("RGB")

                img.save(
                    output_image,
                    format="PNG"
                )

        except Exception as e:

            print(
                f"[WARNING] Could not convert image "
                f"{source_image.name}: {e}"
            )

            skipped += 1
            continue

        # ----------------------------------------------------
        # Create DOTA annotation file
        # ----------------------------------------------------

        output_annotation = (
            output_annotations /
            f"{image_stem}.txt"
        )

        annotation_lines = []

        for ann in data["annotations"]:

            class_id = ann["class_id"]
            class_name = ann["class_name"]
            polygon = ann["polygon"]

            # ------------------------------------------------
            # Safety checks
            # ------------------------------------------------

            if len(polygon) != 8:
                print(
                    f"[WARNING] Invalid polygon in "
                    f"{json_file.name}"
                )
                continue

            if not (0 <= class_id < len(CLASS_NAMES)):
                print(
                    f"[WARNING] Invalid class ID "
                    f"{class_id} in {json_file.name}"
                )
                continue

            if class_name != CLASS_NAMES[class_id]:
                print(
                    f"[WARNING] Class mismatch in "
                    f"{json_file.name}: "
                    f"{class_id} -> {class_name}"
                )

            # ------------------------------------------------
            # DOTA format:
            #
            # x1 y1 x2 y2 x3 y3 x4 y4 class difficulty
            # ------------------------------------------------

            coords = [
                f"{float(value):.2f}"
                for value in polygon
            ]

            line = (
                " ".join(coords)
                + f" {class_name} 0"
            )

            annotation_lines.append(line)

            converted_objects += 1

        # ----------------------------------------------------
        # Write annotation file
        # ----------------------------------------------------

        with open(
            output_annotation,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(annotation_lines)
            )

            if annotation_lines:
                f.write("\n")

        converted_images += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"{split.upper()} SUMMARY")
    print("-" * 70)

    print(f"Images converted : {converted_images}")
    print(f"Objects converted: {converted_objects}")
    print(f"Skipped           : {skipped}")

    return {
        "images": converted_images,
        "objects": converted_objects,
        "skipped": skipped,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CREATING MMROTATE DOTA DATASET")
    print("=" * 70)

    print(f"Source : {SOURCE_ROOT}")
    print(f"Output : {OUTPUT_ROOT}")

    train_summary = convert_split("train")
    val_summary = convert_split("val")

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "IOPA Dental Disease Detection",
        "format": "DOTA / MMRotate",
        "num_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES,
        "train": train_summary,
        "val": val_summary,
        "difficulty": 0,
        "source": "YOLO OBB -> validated polygon JSON -> DOTA",
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_file = (
        OUTPUT_ROOT /
        "dataset_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOTA DATASET CREATION COMPLETE")
    print("=" * 70)

    print(
        f"Train: "
        f"{train_summary['images']} images / "
        f"{train_summary['objects']} objects"
    )

    print(
        f"Val:   "
        f"{val_summary['images']} images / "
        f"{val_summary['objects']} objects"
    )

    print("\nOutput:")
    print(OUTPUT_ROOT)

    print("\nOriginal YOLO dataset remains untouched.")


if __name__ == "__main__":
    main()
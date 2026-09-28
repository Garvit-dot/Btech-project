from pathlib import Path
import json
import random

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "mmrotate_dataset" / "images" / "val"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "mmrotate_dataset" / "annotations" / "val"

OUTPUT_DIR = PROJECT_ROOT / "results" / "mmrotate_annotation_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
# SETTINGS
# ============================================================

NUM_IMAGES = 9
RANDOM_SEED = 42


# ============================================================
# DRAWING
# ============================================================

def draw_annotations(image, annotation_data):

    output = image.copy()

    for ann in annotation_data["annotations"]:

        class_id = ann["class_id"]
        class_name = ann["class_name"]

        polygon = np.array(
            ann["polygon"],
            dtype=np.float32
        ).reshape(4, 2)

        polygon_int = polygon.astype(np.int32)

        # Draw OBB polygon
        cv2.polylines(
            output,
            [polygon_int],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

        # Label position
        x = int(polygon[:, 0].min())
        y = int(polygon[:, 1].min())

        label = f"{class_id}: {class_name}"

        # Background rectangle for text
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )

        cv2.rectangle(
            output,
            (x, max(0, y - th - baseline - 4)),
            (x + tw + 4, y),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (x + 2, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    json_files = sorted(
        ANNOTATION_DIR.glob("*.json")
    )

    if not json_files:
        print("ERROR: No annotation JSON files found.")
        return

    selected = random.sample(
        json_files,
        min(NUM_IMAGES, len(json_files))
    )

    print("=" * 70)
    print("MMROTATE OBB ANNOTATION VERIFICATION")
    print("=" * 70)
    print(f"Validation annotations : {len(json_files)}")
    print(f"Images selected        : {len(selected)}")
    print(f"Output directory       : {OUTPUT_DIR}")
    print()

    for index, json_file in enumerate(selected, start=1):

        with open(json_file, "r", encoding="utf-8") as f:
            annotation_data = json.load(f)

        image_name = annotation_data["image"]
        image_path = IMAGE_DIR / image_name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARNING] Could not read: {image_name}")
            continue

        annotated = draw_annotations(
            image,
            annotation_data
        )

        output_path = (
            OUTPUT_DIR /
            f"{index:02d}_{Path(image_name).stem}_check.jpg"
        )

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(
            f"[{index}/{len(selected)}] "
            f"{image_name} → {output_path.name}"
        )

    print()
    print("=" * 70)
    print("VERIFICATION IMAGES CREATED")
    print("=" * 70)


if __name__ == "__main__":
    main()